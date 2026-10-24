"""Validate private drafts and export only explicitly permitted dictionary records.

JSONL results use {jobId, generatorModel, evidenceHash, draftSha256, draft}.
The draft is generate.py's output_json. SQLite requests rows are read in read-only
mode; their model/output_json/fingerprint/evidence_hash fields form the equivalent
envelope. The generation-time evidence hash is mandatory and is never inferred
from the current job; legacy records without it cannot be approved.
Hashes are SHA-256 of sorted, compact, ensure_ascii=False JSON encoded as UTF-8.
Decisions are private input, never copied into the public export.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import unicodedata
from urllib.parse import urlparse, parse_qs
from .common import normalized_reading, valid_reading, SEMANTIC_REVIEW_VERSION, is_derived_crow_source


FORMAT = "crow-chinese-dictionary/1"
IDENTITY = ("jobId", "canonicalWord", "reading", "spokenLanguage", "targetLanguage")
MAX_DEFINITION = 4000
MAX_SOURCE_IMPORT_SENSES = 8
WEBSITE_PACK_MAX_BYTES = 24 * 1024 * 1024
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PRIVATE_TEXT = re.compile(
    r"(?i)(?:file://|(?:^|[\s\"'`(])(?:/(?:home|root|tmp|etc|var|proc|sys|run|mnt|media|Users|private)(?:/|\b)|[A-Z]:[\\/]|~/|\.\./))"
    r"|(?:^|\n)\s*(?:system|assistant|developer|user)\s*:"
    r"|<\|(?:system|im_start|im_end)|ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions"
    r"|BEGIN [A-Z ]*PRIVATE KEY|Authorization\s*:|Bearer\s+[A-Za-z0-9._~+/-]{12,}"
    r"|\bsk-[A-Za-z0-9_-]{16,}|https?://"
)
LANGUAGE = re.compile(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})?\Z")
SHA = re.compile(r"[a-f0-9]{64}\Z")


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_sha(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _text(value, maximum=MAX_DEFINITION):
    return (isinstance(value, str) and bool(value.strip()) and len(value) <= maximum
            and not CONTROL.search(value) and not PRIVATE_TEXT.search(value))


def _safe_model(value):
    return (isinstance(value, str) and 0 < len(value) <= 200
            and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]*", value))
            and "://" not in value and ".." not in value
            and not re.search(r"(?i)(?:^|/)sk-[A-Za-z0-9_-]{16,}", value))


def model_identity(value):
    """Ignore provider prefixes and speed/routing aliases of the same model."""
    value = str(value).lower().split("/")[-1].split(":")[0]
    value = re.sub(r"(?:[-_](?:highspeed|nitro|free|floor|exacto))+$", "", value)
    return re.sub(r"[^a-z0-9]", "", value)


def evidence_hash(job):
    return canonical_sha(job.get("evidence", []))


def _url(value):
    if not isinstance(value, str) or len(value) > 2048 or CONTROL.search(value):
        return False
    u = urlparse(value)
    return (u.scheme in ("http", "https") and bool(u.netloc) and not u.username
            and not u.password and not any(re.search(r"(?i)(?:api.?key|token|secret|password)", k)
                                          for k in parse_qs(u.query)))


def licence_kind(value):
    if not isinstance(value, str):
        return None
    text = re.sub(r"[\s_-]+", " ", value.strip()).upper()
    if "ND" in text or "SOURCE LICENSE" in text or "+" in text:
        return None
    return {"CC BY SA 4.0": "CC BY-SA 4.0", "CC BY SA 3.0": "CC BY-SA 3.0",
            "CC BY 4.0": "CC BY 4.0", "MIT": "MIT"}.get(text)


def _evidence_issues(item, publication=False):
    issues = []
    if not isinstance(item, dict):
        return ["malformed_evidence"]
    if is_derived_crow_source(item):
        issues.append("derived_corpus_not_primary_evidence")
    if not _text(item.get("id"), 200) or not _text(item.get("sourceId"), 200):
        issues.append("invalid_source_identity")
    licence = licence_kind(item.get("license"))
    if not licence:
        issues.append("blocked_or_unreviewed_license")
    if item.get("quality") != "full_definition":
        issues.append("insufficient_source_definition")
    origin = str(item.get("definitionOrigin", "")).lower()
    if (item.get("machineTranslated") or item.get("translationOnly")
            or any(word in origin for word in ("machine", "ai_", "ai-", "generated"))):
        issues.append("machine_or_translation_only_evidence")
    if publication:
        if not _text(item.get("attribution"), 4000):
            issues.append("missing_source_attribution")
        if not _url(item.get("sourceUrl")):
            issues.append("invalid_source_url")
        if not _url(item.get("licenseUrl")):
            issues.append("missing_license_url")
        for key in ("snapshot", "senseId", "sourceFamily", "language"):
            if item.get(key) is not None and not _text(item[key], 4000):
                issues.append("unsafe_source_metadata")
        pos = item.get("sourcePos", "")
        if not isinstance(pos, str) or len(pos) > 64 or CONTROL.search(pos) or PRIVATE_TEXT.search(pos):
            issues.append("unsafe_source_metadata")
        if item.get("sourceDatasetUrl") is not None and not _url(item["sourceDatasetUrl"]):
            issues.append("invalid_source_dataset_url")
        for field in ("originalRowIndex", "rawOriginalRowIndex"):
            row_index = item.get(field)
            if row_index is not None and (isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0):
                issues.append("invalid_source_row_index")
        if item.get("sourceSha256") is not None and (not isinstance(item["sourceSha256"], str) or not SHA.fullmatch(item["sourceSha256"])):
            issues.append("invalid_source_snapshot_hash")
        if licence == "MIT":
            notice = item.get("licenseNotice", "")
            if (not isinstance(notice, str) or len(notice) > 20000
                    or CONTROL.search(notice) or "copyright" not in notice.lower()
                    or "permission is hereby granted" not in notice.lower()
                    or re.search(r"(?i)(?:file://|/(?:home|root|tmp|etc)/|BEGIN [A-Z ]*PRIVATE KEY|\bsk-[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._~+/-]{12,})", notice)):
                issues.append("missing_mit_copyright_permission_notice")
    return issues


def _source_sense_options(item):
    """Return exact atomic source glosses and permitted whole-row alternatives.

    Tuples are (definition, source POS, canonical register, original tag tuple).
    Raw groups with different tags may never be flattened into an unlabelled row.
    """
    pos = item.get("sourcePos", "") or ""
    if not isinstance(pos, str) or len(pos) > 64 or CONTROL.search(pos) or PRIVATE_TEXT.search(pos):
        return [], [], ["malformed_source_pos"]
    raw = item.get("rawSourceSenses")
    if raw is None or raw == []:
        tags = item.get("usageTags", [])
        if (not isinstance(tags, list) or len(tags) > 32
                or any(not _text(tag, 160) for tag in tags)):
            return [], [], ["malformed_source_usage_tags"]
        tags = tuple(sorted(set(tags)))
        if len(", ".join(tags)) > 160:
            return [], [], ["source_usage_label_too_long"]
        if not _text(item.get("definition")):
            return [], [], ["malformed_source_definition"]
        return [(item.get("definition"), pos, ", ".join(tags), tags)], [], []
    if not isinstance(raw, list) or len(raw) > 256:
        return [], [], ["malformed_raw_source_senses"]
    options, groups = [], []
    for group in raw:
        if not isinstance(group, dict):
            return [], [], ["malformed_raw_source_senses"]
        glosses, tags, raw_tags = group.get("glosses"), group.get("tags", []), group.get("raw_tags", [])
        if (not isinstance(glosses, list) or not glosses or len(glosses) > 32
                or any(not _text(gloss) for gloss in glosses)
                or not isinstance(tags, list) or len(tags) > 32
                or any(not _text(tag, 160) for tag in tags)
                or not isinstance(raw_tags, list) or len(raw_tags) > 32
                or any(not _text(tag, 160) for tag in raw_tags)):
            return [], [], ["malformed_raw_source_senses"]
        tags = tuple(sorted(set(tags + raw_tags)))
        if len(", ".join(tags)) > 160:
            return [], [], ["source_usage_label_too_long"]
        groups.append(tags)
        options.extend((gloss, pos, ", ".join(tags), tags) for gloss in glosses)
    alternatives = []
    if len(set(groups)) == 1 and _text(item.get("definition")):
        tags = groups[0]
        alternatives.append((item.get("definition"), pos, ", ".join(tags), tags))
    return options, alternatives, []


def _source_import_issues(sense, item, language):
    if item.get("language") != language:
        return ["source_import_foreign_language_reference"]
    options, alternatives, issues = _source_sense_options(item)
    if issues:
        return issues
    exact = [option for option in options + alternatives if option[0] == sense.get("definition")]
    if not exact:
        return ["source_import_not_exact_same_language"]
    matching_pos = [option for option in exact if option[1] == sense.get("pos")]
    if not matching_pos:
        return ["source_import_pos_mismatch"]
    if not any(option[2] == sense.get("register") for option in matching_pos):
        return ["source_import_usage_mismatch"]
    return []


def _selected_source_tags(draft, item):
    options, alternatives, _ = _source_sense_options(item)
    tags = set()
    for sense in draft["senses"]:
        if item["id"] in sense["sourceRefs"]:
            for definition, pos, register, source_tags in options + alternatives:
                if (definition, pos, register) == (sense["definition"], sense["pos"], sense["register"]):
                    tags.update(source_tags)
    return sorted(tags)



def _script_form_issues(job):
    """Optional typed spellings must be safe, existing aliases of this job."""
    canonical = job.get('canonicalWord')
    aliases = job.get('aliases', job.get('scriptAliases', []))
    if not isinstance(aliases, list):
        return ['invalid_script_aliases']
    known = {value for value in aliases if isinstance(value, str)}
    if isinstance(canonical, str):
        known.add(canonical)
    issues = []
    for key in ('traditionalWord', 'simplifiedWord'):
        if key not in job:
            continue
        value = job[key]
        if (not _text(value, 128) or re.search(r'\s|[/\\]', value)
                or value not in known):
            issues.append('invalid_' + key)
        elif key == 'simplifiedWord' and value != canonical:
            issues.append('simplified_form_not_canonical')
    return issues

def validate_draft(job, result):
    """Return issues, referenced evidence, and canonical draft/evidence hashes."""
    job = dict(job)
    if "jobId" not in job and "id" in job:
        job["jobId"] = job["id"]
    draft = result.get("draft")
    issues = []
    if job.get("id", job.get("jobId")) != job.get("jobId"):
        issues.append("conflicting_job_identity")
    issues.extend(_script_form_issues(job))
    eh = evidence_hash(job)
    generation_hash = result.get("evidenceHash")
    if not isinstance(generation_hash, str) or not SHA.fullmatch(generation_hash):
        issues.append("missing_generation_evidence_provenance")
    elif generation_hash != eh:
        issues.append("stale_evidence_hash")
    if job.get("evidenceHash", eh) != eh:
        issues.append("stale_evidence_hash")
    if not isinstance(draft, dict):
        return ["malformed_draft"], [], "", eh
    dh = canonical_sha(draft)
    if result.get("draftSha256", dh) != dh:
        issues.append("stale_draft_hash")
    if not _safe_model(result.get("generatorModel")):
        issues.append("invalid_generator_model")
    allowed = set(IDENTITY) | {"senses", "abstainReason"}
    if set(draft) != allowed:
        issues.append("unexpected_or_missing_draft_fields")
    for key in IDENTITY:
        if draft.get(key) != job.get(key):
            issues.append("changed_" + key)
    if result.get("jobId", draft.get("jobId")) != job.get("jobId"):
        issues.append("changed_result_jobId")
    if not _text(job.get("canonicalWord"), 128):
        issues.append("invalid_headword")
    if not _text(job.get("reading"), 512) or not valid_reading(job.get("reading", ""), "yue" if job.get("spokenLanguage") in ("yue", "cantonese") else "cmn"):
        issues.append("missing_or_invalid_reading")
    if job.get("spokenLanguage") not in ("mandarin", "cantonese", "cmn", "yue"):
        issues.append("unsupported_spoken_language")
    if not isinstance(job.get("targetLanguage"), str) or not LANGUAGE.fullmatch(job["targetLanguage"]):
        issues.append("invalid_target_language")
    if draft.get("abstainReason") not in (None, ""):
        issues.append("model_abstained")
    is_import = result.get("generatorModel") == "source-import"
    if is_import:
        if result.get("validationStatus") != "source_imported" or not job.get("reuseEligible"):
            issues.append("ineligible_source_import")
        if set(job.get("holdReasons", [])) - {"same_language_source_reuse_available"}:
            issues.append("source_import_has_unresolved_holds")
    elif not job.get("generationEligible") or job.get("holdReasons"):
        issues.append("generation_ineligible")
    evidence = job.get("evidence", [])
    if not isinstance(evidence, list) or not evidence:
        issues.append("missing_evidence")
        evidence = []
    emap = {item.get("id"): item for item in evidence if isinstance(item, dict)}
    if len(emap) != len(evidence):
        issues.append("duplicate_or_malformed_evidence_identity")
    senses = draft.get("senses")
    referenced = {}
    if not isinstance(senses, list) or not 1 <= len(senses) <= 12:
        issues.append("invalid_senses")
        senses = []
    if is_import:
        atomic_options = set()
        for item in evidence:
            if (isinstance(item, dict) and item.get("language") == job.get("targetLanguage")
                    and not _evidence_issues(item)):
                options, _, source_issues = _source_sense_options(item)
                issues.extend(source_issues)
                atomic_options.update(option[:3] for option in options)
        if len(senses) > MAX_SOURCE_IMPORT_SENSES or len(atomic_options) > MAX_SOURCE_IMPORT_SENSES:
            issues.append("source_sense_overflow")
    total = 0
    for sense in senses:
        if not isinstance(sense, dict) or set(sense) != {"definition", "pos", "register", "sourceRefs"}:
            issues.append("malformed_sense")
            continue
        definition = sense.get("definition")
        if not _text(definition):
            issues.append("empty_huge_or_unsafe_definition")
        else:
            total += len(definition)
            if job.get("targetLanguage") == "en" and not re.search(r"[A-Za-z]{2}", definition):
                issues.append("missing_english_definition_text")
        for key, maximum in (("pos", 64), ("register", 160)):
            value = sense.get(key)
            if not isinstance(value, str) or len(value) > maximum or CONTROL.search(value) or PRIVATE_TEXT.search(value):
                issues.append("unsafe_" + key)
        refs = sense.get("sourceRefs")
        if (not isinstance(refs, list) or not refs or len(refs) > 20
                or any(not isinstance(ref, str) for ref in refs)):
            issues.append("missing_or_invalid_source_refs")
            continue
        if len(set(refs)) != len(refs):
            issues.append("duplicate_source_ref")
        actual = []
        for ref in refs:
            item = emap.get(ref)
            if item is None:
                issues.append("invented_source_ref")
            else:
                issues.extend(_evidence_issues(item))
                if item.get("sourceFamily") == "wordnet-cow" or str(item.get("sourceId", "")).startswith("wordnet-"):
                    issues.append("unreviewed_component_license")
                source_word = item.get("canonicalWord")
                if source_word and source_word not in set(job.get("aliases", job.get("scriptAliases", [])) + [job["canonicalWord"]]):
                    issues.append("source_headword_mismatch")
                if item.get("reading") and normalized_reading(item["reading"]) != normalized_reading(job.get("reading", "")):
                    issues.append("source_reading_mismatch")
                referenced[ref] = item
                actual.append(item)
        if is_import:
            for item in actual:
                issues.extend(_source_import_issues(sense, item, job.get("targetLanguage")))
    if total > 12000:
        issues.append("definition_total_too_large")
    return sorted(set(issues)), list(referenced.values()), dh, eh


def review_status(job, result, decisions, dh, eh):
    """Reviews bind exact draft/evidence/model; uncertainty and rejection hold it."""
    matching = [d for d in decisions if d.get("jobId") == job.get("jobId", job.get("id"))]
    issues, warnings, valid, seen = [], [], [], {}
    for d in matching:
        if d.get("draftSha256") != dh or d.get("evidenceHash") != eh:
            warnings.append("stale_review_hash")
            continue
        if d.get("generatorModel") != result.get("generatorModel"):
            issues.append("review_generator_model_mismatch")
            continue
        if (d.get("decision") not in ("approve", "reject", "uncertain")
                or d.get("reviewType") not in ("human", "independent_model")
                or not _text(d.get("reviewer"), 200)
                or not isinstance(d.get("notes"), str)
                or len(d["notes"]) > 8000 or CONTROL.search(d["notes"])):
            issues.append("invalid_review")
            continue
        if d["reviewType"] == "independent_model":
            try:
                protocol = json.loads(d["notes"]).get("protocol", "")
            except (ValueError, AttributeError):
                protocol = ""
            if (isinstance(protocol, str) and protocol.startswith("crow-semantic-review-")
                    and protocol != SEMANTIC_REVIEW_VERSION and d["decision"] == "approve"):
                warnings.append("superseded_automatic_review_protocol")
                continue
            if (not _safe_model(d["reviewer"]) or d["reviewer"] == "source-import"
                    or model_identity(d["reviewer"]) == model_identity(result["generatorModel"])):
                issues.append("review_model_not_independent")
                continue
        key = (d["reviewType"], d["reviewer"])
        if key in seen:
            if seen[key] != d:
                issues.append("conflicting_duplicate_review")
            continue
        seen[key] = d
        valid.append(d)
    if issues:
        return "machine_draft", sorted(set(issues + warnings))
    if any(d["decision"] in ("reject", "uncertain") for d in valid):
        return "held", sorted(set(["review_rejected_or_uncertain"] + warnings))
    if any(d["decision"] == "approve" and d["reviewType"] == "human" for d in valid):
        return "human_reviewed", sorted(set(warnings))
    if any(d["decision"] == "approve" and d["reviewType"] == "independent_model" for d in valid):
        return "machine_checked", sorted(set(warnings))
    return "machine_draft", sorted(set(warnings))


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except (ValueError, TypeError) as error:
                    raise ValueError(f"Invalid JSONL record at line {number}") from error
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL record at line {number} must be an object")
                yield value


def read_results(path, jobs):
    with open(path, "rb") as stream:
        sqlite = stream.read(16) == b"SQLite format 3\x00"
    if not sqlite:
        for row in read_jsonl(path):
            row = dict(row)
            row["generatorModel"] = row.get("generatorModel", row.get("model"))
            if "draft" not in row and "output_json" in row:
                row["draft"] = json.loads(row["output_json"]) if isinstance(row["output_json"], str) else row["output_json"]
            yield row
        return
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        columns = {row[1] for row in connection.execute("PRAGMA table_info(requests)")}
        required = {"job_id", "model", "status", "output_json", "fingerprint"}
        if not required <= columns:
            raise ValueError("Unsupported results database schema")
        if "evidence_hash" not in columns:
            raise ValueError("Unsupported legacy results provenance: generation-time evidence_hash is missing; regenerate drafts rather than infer it from current jobs")
        for row in connection.execute("SELECT job_id,model,status,output_json,fingerprint,evidence_hash FROM requests WHERE status IN ('machine_draft','source_imported') ORDER BY job_id,fingerprint"):
            draft = json.loads(row["output_json"])
            yield {"jobId": row["job_id"], "generatorModel": row["model"],
                   "validationStatus": row["status"], "draft": draft,
                   "generationFingerprint": row["fingerprint"],
                   "draftSha256": canonical_sha(draft), "evidenceHash": row["evidence_hash"]}


def _public_source(item, selected_tags=None):
    keys = ("id", "sourceId", "language", "license", "licenseUrl", "attribution",
            "sourceUrl", "sourceDatasetUrl", "snapshot", "sourceSha256", "originalRowIndex", "rawOriginalRowIndex", "senseId", "sourceFamily", "sourcePos")
    source = {key: item[key] for key in keys if key in item and item[key] is not None}
    if "originalRowIndex" in source:
        source["rowIndexBasis"] = "normalized_source_entry_zero_based"
    if "rawOriginalRowIndex" in source:
        source["rawRowIndexBasis"] = "original_jsonl_line_zero_based"
    if selected_tags is not None:
        source["usageTags"] = selected_tags
    return source


def _public_record(job, result, sources, dh, eh, status):
    draft = result["draft"]
    aliases = sorted(set(job.get("aliases", job.get("scriptAliases", [])) + [job["canonicalWord"]]))
    aliases = [value for value in aliases if _text(value, 128)]
    return {"id": job["jobId"], "canonicalWord": job["canonicalWord"], "scriptAliases": aliases,
            "scriptForms": {"traditional": job.get("traditionalWord", job["canonicalWord"]),
                            "simplified": job.get("simplifiedWord", job["canonicalWord"])},
            "lexicalReading": job["reading"], "spokenLanguage": job["spokenLanguage"],
            "targetLanguage": job["targetLanguage"], "senses": draft["senses"],
            "sources": sorted((_public_source(item, _selected_source_tags(draft, item) if status == "source_imported" else None)
                               for item in sources), key=lambda x: x["id"]),
            "dataOrigin": "source_imported" if status == "source_imported" else "model_generated",
            "machineTranslated": status != "source_imported", "syntheticDefinition": status != "source_imported",
            "validation": {"status": status, "automaticChecks": "passed", "draftSha256": dh,
                           "evidenceHash": eh, "generatorModel": result["generatorModel"]},
            "pronunciationRef": {"canonicalWord": job["canonicalWord"], "reading": job["reading"],
                                 "spokenLanguage": job["spokenLanguage"], "audioStatus": "not_checked"},
            "license": "CC BY-SA 4.0"}


def _gzip(data):
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as stream:
        stream.write(data)
    return output.getvalue()


def _jsonl(rows):
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"


def _website_entry(record):
    cantonese = record["spokenLanguage"] in ("cantonese", "yue")
    senses = record["senses"]
    definition = "/".join((("(" + sense["register"] + ") ") if sense["register"] else "")
                          + sense["definition"].replace("/", "∕") for sense in senses)
    metadata = {"source": "crow-" + record["targetLanguage"], "canonicalWord": record["canonicalWord"],
                "lookupAliases": record["scriptAliases"], "chineseLanguage": "cantonese" if cantonese else "mandarin",
                "pos": "; ".join(sorted({sense["pos"] for sense in senses if sense["pos"]})),
                # Runtime packs retain the origin/review label. Per-entry
                # random audit hashes stay in the full archival entries;
                # the runtime pack itself is authenticated by its manifest SHA.
                "definitionProvenance": {key: record["validation"][key]
                                         for key in ("status", "automaticChecks", "generatorModel")},
                "dataOrigin": record["dataOrigin"],
                "syntheticDefinition": record["syntheticDefinition"], "sourceSenses": senses}
    if cantonese:
        metadata["jyutping"] = record["lexicalReading"]
    forms = record.get("scriptForms", {})
    return [forms.get("traditional", record["canonicalWord"]), forms.get("simplified", record["canonicalWord"]),
            "" if cantonese else record["lexicalReading"], definition, metadata]


def _notice_source_link(item):
    if item.get("sourceDatasetUrl"):
        return item["sourceDatasetUrl"]
    # Article-specific URLs remain in each entry and must not create one notice
    # block per word. Only clear dataset/home links are suitable fallback keys.
    source = item["sourceUrl"]
    parsed = urlparse(source)
    if parsed.path.rstrip("/") in ("", "/page/download", "/dictionary/Chinese"):
        return source
    if parsed.query == "page=cedict":
        return source
    return ""


def export(jobs_path, results_path, decisions_path, out, *, include_machine_checked=False,
           include_source_imports=False):
    jobs = {}
    for job in read_jsonl(jobs_path):
        key = job.get("jobId", job.get("id"))
        if not _text(key, 200) or key in jobs:
            raise ValueError("Jobs must have unique valid jobId values")
        if "id" in job and "jobId" in job and job["id"] != job["jobId"]:
            raise ValueError("Conflicting id and jobId fields")
        job = dict(job)
        job["jobId"] = key
        jobs[key] = job
    decisions = list(read_jsonl(decisions_path)) if decisions_path else []
    if any(not _text(d.get("jobId"), 200) for d in decisions):
        raise ValueError("Every review decision must identify a job")
    queues, selected, seen, used_evidence = [], [], set(), {}
    jobs_with_results = set()
    counts = {"jobs": len(jobs), "validDrafts": 0, "invalidDrafts": 0, "pending": 0,
              "notGenerated": 0, "humanReviewed": 0, "machineChecked": 0, "sourceImported": 0, "accepted": 0}
    for result in read_results(results_path, jobs):
        job = jobs.get(result.get("jobId"))
        if job is None:
            counts["invalidDrafts"] += 1
            queues.append({"jobId": "unknown-" + canonical_sha(str(result.get("jobId", ""))), "status": "quarantined", "issues": ["unknown_job"]})
            continue
        jobs_with_results.add(job["jobId"])
        issues, sources, dh, eh = validate_draft(job, result)
        duplicate = (job["jobId"], dh, result.get("generatorModel"))
        if duplicate in seen:
            continue
        seen.add(duplicate)
        is_import = result.get("generatorModel") == "source-import"
        if issues:
            status = "quarantined"
            counts["invalidDrafts"] += 1
        else:
            counts["validDrafts"] += 1
            status, issues = ("source_imported", []) if is_import else review_status(job, result, decisions, dh, eh)
        eligible = status == "human_reviewed" or (status == "machine_checked" and include_machine_checked) or (status == "source_imported" and include_source_imports)
        if eligible:
            publication_issues = sorted({issue for item in sources for issue in _evidence_issues(item, publication=True)})
            if publication_issues:
                issues.extend(publication_issues)
                eligible = False
        if eligible:
            selected.append(_public_record(job, result, sources, dh, eh, status))
            for item in sources:
                used_evidence[item["id"]] = item
        else:
            counts["pending"] += 1
        queues.append({"jobId": job["jobId"], "generatorModel": result.get("generatorModel") if _safe_model(result.get("generatorModel")) else None,
                       "draftSha256": dh, "evidenceHash": eh, "status": status,
                       "generationEvidenceHash": result.get("evidenceHash") if isinstance(result.get("evidenceHash"), str) and SHA.fullmatch(result["evidenceHash"]) else None,
                       "exported": eligible, "issues": sorted(set(issues))})
    for key in sorted(set(jobs) - jobs_with_results):
        queues.append({"jobId": key, "evidenceHash": evidence_hash(jobs[key]),
                       "generationEvidenceHash": None, "generatorModel": None,
                       "draftSha256": None, "status": "not_generated", "exported": False,
                       "issues": []})
        counts["notGenerated"] += 1
    occurrences = Counter(row["id"] for row in selected)
    multiple = {key for key, count in occurrences.items() if count > 1}
    if multiple:
        selected = [row for row in selected if row["id"] not in multiple]
        for queue in queues:
            if queue["jobId"] in multiple:
                queue["exported"] = False
                queue["issues"] = sorted(set(queue["issues"] + ["multiple_selected_drafts"]))
    selected.sort(key=lambda row: row["id"])
    used_ids = {source["id"] for row in selected for source in row["sources"]}
    used_evidence = {key: value for key, value in used_evidence.items() if key in used_ids}
    for row in selected:
        key = {"human_reviewed": "humanReviewed", "machine_checked": "machineChecked", "source_imported": "sourceImported"}[row["validation"]["status"]]
        counts[key] += 1
    counts["accepted"] = len(selected)
    # Coverage counts jobs, not duplicate drafts or unknown result records.
    counts["pending"] = len(jobs) - len({row["id"] for row in selected})
    # Each entry retains exact article/row/sense attribution. The manifest indexes
    # datasets once rather than duplicating tens of thousands of source records.
    datasets = {}
    for value in used_evidence.values():
        dataset = {key: value[key] for key in ("sourceId", "language", "license", "licenseUrl", "attribution", "snapshot", "sourceSha256", "sourceFamily") if key in value}
        dataset["sourceUrl"] = _notice_source_link(value)
        datasets[canonical_sha(dataset)] = dataset
    sources = sorted(datasets.values(), key=lambda value: (value["sourceId"], canonical_sha(value)))
    notice = ["Crow Chinese Dictionary", "Derived dictionary records: CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/", "",
              "Original source rights and notices remain applicable. Model-generated text is labelled; automatic checks are not human review.",
              "Individual article URLs and source record identifiers are retained with each entry in entries.jsonl.gz.", ""]
    notice_groups = {}
    for item in used_evidence.values():
        link = _notice_source_link(item)
        key = (item["sourceId"], item["attribution"], item["license"], item["licenseUrl"],
               link, str(item.get("snapshot", "")), item.get("licenseNotice", ""))
        notice_groups.setdefault(key, item)
    for key in sorted(notice_groups):
        item = notice_groups[key]
        notice.extend([item["sourceId"], item["attribution"], _notice_source_link(item) or "Source article URLs: see each entry.",
                       item["license"] + " — " + item["licenseUrl"], "Snapshot: " + str(item.get("snapshot", ""))])
        if item.get("licenseNotice"):
            notice.append(item["licenseNotice"])
        notice.append("")
    assets = {"entries.jsonl.gz": _gzip(_jsonl(selected)), "NOTICE.txt": ("\n".join(notice) + "\n").encode("utf-8"),
              "review-queue.jsonl": _jsonl(sorted(queues, key=lambda q: (q["jobId"], q.get("draftSha256", "")))),
              "validation-report.json": _json({"format": FORMAT, "counts": counts,
                  "includeMachineChecked": include_machine_checked, "includeSourceImports": include_source_imports,
                  "automaticChecksAreHumanReview": False})}
    website = []
    for language in sorted({row["targetLanguage"] for row in selected}):
        rows = [row for row in selected if row["targetLanguage"] == language]
        path = "website/" + language + ".json.gz"
        pack = _gzip(canonical_bytes([_website_entry(row) for row in rows]))
        if len(pack) > WEBSITE_PACK_MAX_BYTES:
            raise ValueError(f"Website pack for {language} exceeds the {WEBSITE_PACK_MAX_BYTES}-byte static-host limit; bounded sharding and loader support are required. No records were truncated or export assets written")
        assets[path] = pack
        website.append({"id": "crow-" + language, "targetLanguage": language, "file": path,
                        "count": len(rows), "license": "CC BY-SA 4.0", "disabledByDefault": True,
                        "source": "crow-chinese-dictionary", "audioCoverage": "not_checked",
                        "definitionOrigins": dict(sorted(Counter(row["dataOrigin"] for row in rows).items())),
                        "containsSyntheticDefinitions": any(row["syntheticDefinition"] for row in rows)})
    assets["website/import-manifest.json"] = _json({"format": FORMAT, "packs": website,
        "definitionOrigins": dict(sorted(Counter(row["dataOrigin"] for row in selected).items())),
        "containsSyntheticDefinitions": any(row["syntheticDefinition"] for row in selected),
        "installation": "Explicit opt-in. Import by source id; removal must preserve personal cards/books/notes. No website mutation is performed by this exporter."})
    manifest = {"format": FORMAT, "name": "crow-chinese-dictionary", "license": "CC BY-SA 4.0",
                "counts": counts, "sources": sources, "sourceRecordCount": len(used_evidence),
                "definitionOrigins": dict(sorted(Counter(row["dataOrigin"] for row in selected).items())),
                "containsSyntheticDefinitions": any(row["syntheticDefinition"] for row in selected),
                "sourceRecordLocation": "entries.jsonl.gz", "automaticChecksAreHumanReview": False,
                "assets": [{"file": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                           for path, data in sorted(assets.items())]}
    assets["manifest.json"] = _json(manifest)
    out = Path(out)
    if out.is_symlink():
        raise ValueError("Public output must not be a symlink")
    if out.exists():
        for path in out.rglob("*"):
            if path.is_symlink():
                raise ValueError("Public output must not contain symlinks")
            if path.is_file():
                relative = path.relative_to(out).as_posix()
                if relative not in assets and not re.fullmatch(r"website/[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})?\.json\.gz", relative):
                    raise ValueError("Public output contains an unrecognised file; use a clean output directory")
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".crow-export-", dir=out.parent) as temporary:
        stage = Path(temporary)
        for name, data in assets.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        for name in sorted(assets, key=lambda name: name == "manifest.json"):
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / name, target)
        for path in (out / "website").glob("*.json.gz"):
            if path.relative_to(out).as_posix() not in assets:
                path.unlink()
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path, help="Runner SQLite DB or result-envelope JSONL[.gz]")
    parser.add_argument("--decisions", type=Path, help="Real private human/model review decisions JSONL")
    parser.add_argument("--out", required=True, type=Path, help="Clean public export directory")
    parser.add_argument("--include-machine-checked", action="store_true")
    parser.add_argument("--include-source-imports", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = export(args.jobs, args.results, args.decisions, args.out,
                          include_machine_checked=args.include_machine_checked,
                          include_source_imports=args.include_source_imports)
    except (ValueError, OSError, sqlite3.Error, TypeError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({"accepted": manifest["counts"]["accepted"], "counts": manifest["counts"],
                      "manifestSha256": canonical_sha(manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
