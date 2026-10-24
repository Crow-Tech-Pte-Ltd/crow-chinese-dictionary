"""Private, local-only recovery of exact source-bound model output.

Only complete unique JSON and explicitly allowed abstention-control formatting
normalizations are considered. Lexical text, identities, readings, languages,
source references, POS and usage labels are never edited. Existing validators
remain authoritative. Nothing here changes a ledger or invents an approval. Original independent
review decisions may be recovered only after their full unchanged text passes
all current semantic validators under the defensive notes-length bound.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import closing
import copy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys

from .generate import PROMPT_VERSION, validate_output
from .common import SEMANTIC_REVIEW_VERSION
from .snapshot import normalize_job, _atomic_bytes, _encode_jsonl
from .review import validate_review, decision_record
from .validate import canonical_bytes, canonical_sha, evidence_hash, read_jsonl, validate_draft, model_identity

FORMAT = "crow-local-json-recovery/1"
REQUIRED = {"jobId", "canonicalWord", "reading", "spokenLanguage", "targetLanguage", "senses", "abstainReason"}
SAFE_FIELDS = REQUIRED - {"abstainReason"}


class RecoveryError(ValueError):
    pass


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RecoveryError("duplicate_json_key")
            result[key] = value
        return result
    def constant(_):
        raise RecoveryError("nonstandard_json_constant")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def complete_json(content):
    reasons = []
    if isinstance(content, dict):
        # Round-trip through the strict parser; no model fields are manufactured.
        return strict_json(canonical_bytes(content).decode()), ["structured_content_object"]
    if isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict) and set(content[0]) == {"type", "text"} and content[0]["type"] == "text":
        content = content[0]["text"]
        reasons.append("single_text_block_unwrapped")
    if not isinstance(content, str):
        raise RecoveryError("missing_string_or_object_content")
    if content.startswith("\ufeff"):
        content = content[1:]
        reasons.append("leading_utf8_bom_removed")
    trimmed = content.strip()
    if trimmed != content:
        reasons.append("outer_whitespace_removed")
    content = trimmed
    fence = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n```", content, re.I)
    if fence:
        content = fence[1].strip()
        reasons.append("single_json_fence_removed")
    result = strict_json(content)
    if not isinstance(result, dict):
        raise RecoveryError("top_level_not_object")
    return result, reasons


def normalize_abstention(value, allow_empty_abstrain_alias=False):
    """Never change definitions/identity/source refs or remove unknown extras."""
    output = copy.deepcopy(value)
    reasons = []
    senses = output.get("senses")
    nonempty_senses = isinstance(senses, list) and bool(senses)
    aliases = ["abstain_reason"] + (["abstrainReason"] if allow_empty_abstrain_alias else [])
    present = [key for key in aliases if key in output]
    if "abstainReason" not in output and len(present) == 1 and set(output) == SAFE_FIELDS | {present[0]}:
        alias = present[0]
        alias_value = output[alias]
        # Explicit typo permission is narrower than the snake_case rename: it
        # covers only empty control values with existing model definitions.
        if alias == "abstrainReason" and (not nonempty_senses or alias_value not in (None, "")):
            raise RecoveryError("conflicting_or_nonempty_typo_abstention")
        output["abstainReason"] = output.pop(alias)
        reasons.append("unique_control_alias_renamed:" + alias)
    if set(output) == SAFE_FIELDS and nonempty_senses:
        output["abstainReason"] = ""
        reasons.append("missing_abstention_empty_for_nonempty_senses")
    elif set(output) == REQUIRED and output["abstainReason"] is None and nonempty_senses:
        output["abstainReason"] = ""
        reasons.append("null_abstention_empty_for_nonempty_senses")
    return output, reasons


def response_content(response_json, expected_model):
    if not isinstance(response_json, str):
        raise RecoveryError("missing_stored_response")
    response = strict_json(response_json)
    if not isinstance(response, dict) or not isinstance(response.get("model"), str) or model_identity(response["model"]) != model_identity(expected_model):
        raise RecoveryError("returned_model_mismatch_or_missing")
    base = response.get("base_resp", {})
    if isinstance(base, dict) and base.get("status_code", 0) != 0:
        raise RecoveryError("provider_error_response")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RecoveryError("missing_or_multiple_choices")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") not in ("stop", "end_turn", "length"):
        raise RecoveryError("unsupported_finish_reason")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("tool_calls") or message.get("refusal"):
        raise RecoveryError("tool_or_refusal_output")
    value, reasons = complete_json(message.get("content"))
    if choice["finish_reason"] == "length":
        reasons.append("complete_unique_json_despite_length_finish")
    content = message.get("content")
    return value, reasons, {"storedResponseSha256": hashlib.sha256(response_json.encode()).hexdigest(),
        "contentSha256": hashlib.sha256(content.encode() if isinstance(content, str) else canonical_bytes(content)).hexdigest(),
        "originalFinishReason": choice["finish_reason"]}


def metadata(row):
    return {"originalRequestFingerprint": row["fingerprint"], "originalAttemptId": row["attempt_id"],
        "originalAttemptNumber": row["attempt_no"], "originalRequestStatus": row["status"],
        "originalPromptVersion": row["prompt_version"], "originalModel": row["model"]}


def recover(ledger, jobs_path, out, *, allow_empty_abstrain_alias=False):
    jobs = {}
    for item in read_jsonl(jobs_path):
        item = normalize_job(item)
        if item["id"] in jobs:
            raise RecoveryError("duplicate_current_job")
        jobs[item["id"]] = item
    eligible = {key: item for key, item in jobs.items() if item.get("generationEligible") is True and not item.get("holdReasons")}
    valid = defaultdict(dict)
    invalid_gen, invalid_review, raw_statuses = [], [], defaultdict(list)
    sql = """SELECT r.fingerprint,r.job_id,r.model,r.prompt_version,r.status,r.updated,r.evidence_hash,r.output_json,r.issues_json,
        a.id AS attempt_id,a.attempt_no,a.response_json FROM requests r LEFT JOIN attempts a ON a.id=(SELECT max(id) FROM attempts WHERE fingerprint=r.fingerprint)
        WHERE r.prompt_version IN (?,?) ORDER BY r.job_id,r.updated,r.fingerprint"""
    with closing(sqlite3.connect(Path(ledger).resolve().as_uri() + "?mode=ro", uri=True, timeout=30)) as db:
        db.row_factory = sqlite3.Row
        for source in db.execute(sql, (PROMPT_VERSION, SEMANTIC_REVIEW_VERSION)):
            row = dict(source)
            job = eligible.get(row["job_id"])
            if not job or row["evidence_hash"] != evidence_hash(job):
                continue
            if row["prompt_version"] == PROMPT_VERSION:
                raw_statuses[row["job_id"]].append(row)
                if row["status"] == "schema_invalid":
                    invalid_gen.append(row)
                if row["status"] == "machine_draft":
                    try:
                        draft = strict_json(row["output_json"])
                        env = {"jobId": job["id"], "generatorModel": row["model"], "draft": draft,
                            "evidenceHash": row["evidence_hash"], "draftSha256": canonical_sha(draft),
                            "generationFingerprint": row["fingerprint"], "promptVersion": row["prompt_version"]}
                        issues = validate_draft(job, env)[0] + validate_output(draft, job, [e["id"] for e in job["evidence"]])[1]
                        if not issues:
                            valid[job["id"]][env["draftSha256"]] = env
                    except (ValueError, TypeError, KeyError):
                        pass
            elif row["status"] in ("review_invalid", "review_completed"):
                invalid_review.append(row)
    failures, recovered_by_job = [], defaultdict(list)
    gen_shapes, gen_issues = Counter(), Counter()
    for row in invalid_gen:
        job = eligible[row["job_id"]]
        try:
            draft, reasons, hashes = response_content(row["response_json"], row["model"])
            gen_shapes[tuple(sorted(draft))] += 1
            normalized, control = normalize_abstention(draft, allow_empty_abstrain_alias)
            reasons.extend(control)
            env = {"jobId": job["id"], "generatorModel": row["model"], "draft": normalized,
                "evidenceHash": row["evidence_hash"], "draftSha256": canonical_sha(normalized),
                "generationFingerprint": row["fingerprint"], "promptVersion": row["prompt_version"],
                "validationStatus": "machine_draft", "requiresFreshIndependentReview": True,
                "localRecovery": {**metadata(row), **hashes, "normalizationKinds": reasons or ["strict_json_revalidated"],
                    "originalParsedOutputSha256": canonical_sha(draft), "normalizedOutputSha256": canonical_sha(normalized),
                    "validatorsUnchanged": True, "createsApproval": False}}
            issues = validate_output(normalized, job, [e["id"] for e in job["evidence"]])[1] + validate_draft(job, env)[0]
            if issues:
                gen_issues.update(set(issues))
                failures.append({"jobId": job["id"], **metadata(row), **hashes, "kind": "generation_not_recovered", "issues": sorted(set(issues))})
            elif job["id"] not in valid:
                recovered_by_job[job["id"]].append((row["updated"], row["fingerprint"], env))
        except (ValueError, TypeError, KeyError) as error:
            reason = str(error) if isinstance(error, RecoveryError) else "incomplete_or_nonunique_or_invalid_json"
            gen_issues[reason] += 1
            failures.append({"jobId": job["id"], **metadata(row), "kind": "generation_not_recovered", "issues": [reason]})
    recovered = []
    for key, candidates in sorted(recovered_by_job.items()):
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        env = candidates[0][2]
        recovered.append(env)
        valid[key][env["draftSha256"]] = env
    reviewed = set()
    observations, notes_breakdown, review_issues = [], Counter(), Counter()
    recovered_decisions, ref_breakdown = [], Counter()
    for row in invalid_review:
        job = eligible[row["job_id"]]
        try:
            # Existing completed checks are inspected only to avoid paying again
            # for a draft already reviewed. Invalid checks never become approvals.
            if row["status"] == "review_completed":
                value = strict_json(row["output_json"])
                hashes, reasons = {}, []
            else:
                value, reasons, hashes = response_content(row["response_json"], row["model"])
            if row["status"] == "review_invalid":
                note = value.get("notes")
                notes_breakdown["type_" + type(note).__name__] += 1
                if isinstance(note, str):
                    notes_breakdown["over500" if len(note) > 500 else "within500"] += 1
                    notes_breakdown["control" if any(ord(c) < 32 and c not in "\n\t" for c in note) else "no_control"] += 1
            env = valid.get(job["id"], {}).get(value.get("draftSha256"))
            if not env:
                raise RecoveryError("no_matching_current_valid_generation")
            plan = {"job": job, "result": env, "draftSha256": env["draftSha256"], "evidenceHash": env["evidenceHash"], "fingerprint": row["fingerprint"]}
            issues = validate_review(value, plan)
            if row["status"] == "review_invalid":
                for check in value.get("senseReviews", []) if isinstance(value.get("senseReviews"), list) else []:
                    if not isinstance(check, dict):
                        continue
                    refs = check.get("sourceRefs")
                    index = check.get("senseIndex")
                    if not isinstance(refs, list) or not refs:
                        ref_breakdown["missing_or_nonlist"] += 1
                    elif any(not isinstance(ref, str) for ref in refs):
                        ref_breakdown["invalid_reference_type"] += 1
                    else:
                        if len(set(refs)) != len(refs):
                            ref_breakdown["duplicate_reference"] += 1
                        if type(index) is int and 0 <= index < len(env["draft"]["senses"]):
                            allowed = env["draft"]["senses"][index]["sourceRefs"]
                            if any(ref not in allowed for ref in refs):
                                ref_breakdown["out_of_scope_for_draft_sense"] += 1
                original_issues = strict_json(row["issues_json"] or "[]")
                note = value.get("notes")
                if (not issues and set(original_issues) == {"invalid_review_notes"}
                        and isinstance(note, str) and 500 < len(note) <= 1024
                        and model_identity(row["model"]) != model_identity(env["generatorModel"])):
                    recovered_decision = decision_record(plan, value, row["model"])
                    recovered_decision["localRecovery"] = {**metadata(row), **hashes,
                        "recoveryKind": "notes_bound_500_to_1024_original_text_retained",
                        "originalValidationIssues": original_issues,
                        "originalReviewSha256": canonical_sha(value), "revalidatedReviewSha256": canonical_sha(value),
                        "semanticProtocolUnchanged": True, "originalVerdictRetained": True,
                        "fullOriginalNotesRetained": True}
                    recovered_decisions.append(recovered_decision)
                    reviewed.add((job["id"], env["draftSha256"]))
            if row["status"] == "review_completed" and not issues and model_identity(row["model"]) != model_identity(env["generatorModel"]):
                reviewed.add((job["id"], env["draftSha256"]))
            elif row["status"] == "review_invalid":
                review_issues.update(issues)
                observations.append({"jobId": job["id"], **metadata(row), **hashes,
                    "kind": "private_review_observation", "usableAsDecision": False,
                    "requiresFreshIndependentReview": True, "normalizationKinds": reasons,
                    "issues": issues, "strictlyValidObservation": not issues,
                    "observedVerdict": value.get("verdict")})
        except (ValueError, TypeError, KeyError) as error:
            if row["status"] == "review_invalid":
                reason = str(error) if isinstance(error, RecoveryError) else "incomplete_or_nonunique_or_invalid_review_json"
                review_issues[reason] += 1
                observations.append({"jobId": job["id"], **metadata(row), "kind": "private_review_observation",
                    "usableAsDecision": False, "strictlyValidObservation": False, "requiresFreshIndependentReview": True,
                    "issues": [reason]})
    need_review = [env for key in sorted(valid) for dh, env in sorted(valid[key].items()) if (key, dh) not in reviewed]
    remaining, categories = [], Counter()
    for key, job in sorted(eligible.items()):
        if key in valid:
            continue
        rows = raw_statuses.get(key, [])
        statuses = {row["status"] for row in rows}
        if statuses & {"quarantined", "requesting"}:
            category = "unknown_outcome_requires_operator_review"
        elif "abstained" in statuses:
            category = "abstained_requires_source_review"
        elif "schema_invalid" in statuses:
            category = "requires_bounded_schema_correction_request"
            remaining.append(job)
        elif "failed" in statuses:
            category = "definitive_failure_requires_operator_requeue"
        else:
            category = "no_current_generation_request"
            remaining.append(job)
        categories[category] += 1
    out = Path(out)
    assets = {"recovered-generation.jsonl.gz": _encode_jsonl(recovered, True),
        "recovered-review-decisions.jsonl.gz": _encode_jsonl(recovered_decisions, True),
        "needs-fresh-review.jsonl.gz": _encode_jsonl(need_review, True),
        "needs-bounded-generation.jsonl.gz": _encode_jsonl(remaining, True),
        "generation-not-recovered.jsonl.gz": _encode_jsonl(failures, True),
        "review-observations-not-decisions.jsonl.gz": _encode_jsonl(observations, True)}
    summary = {"format": FORMAT, "promptVersion": PROMPT_VERSION, "reviewVersion": SEMANTIC_REVIEW_VERSION,
        "counts": {"currentJobs": len(jobs), "generationEligibleJobs": len(eligible), "invalidGenerationRows": len(invalid_gen),
            "recoveredGenerationJobs": len(recovered), "currentValidGeneratedJobsIncludingRecovery": len(valid),
            "needsFreshReviewEnvelopes": len(need_review), "invalidReviewRows": sum(row["status"] == "review_invalid" for row in invalid_review),
            "strictlyValidRecoveredReviewObservations": sum(row["strictlyValidObservation"] for row in observations),
            "newModelApprovalsInvented": 0,
            "originalReviewDecisionsRecovered": len(recovered_decisions),
            "originalReviewVerdictsRecovered": dict(Counter(row["decision"] for row in recovered_decisions))},
        "generationKeyShapes": [{"keys": list(shape), "count": count} for shape, count in gen_shapes.most_common()],
        "remainingGenerationCategories": dict(categories), "unrecoverableGenerationIssues": dict(gen_issues),
        "reviewIssues": dict(review_issues), "reviewNotesBreakdown": dict(notes_breakdown),
        "sourceRefFailureBreakdownBySense": dict(ref_breakdown),
        "allowEmptyAbstrainAlias": allow_empty_abstrain_alias,
        "ledgerModified": False, "networkRequests": 0, "credentialsRead": False,
        "assets": [{"file": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in sorted(assets.items())],
        "recoveredProvenance": [{"jobId": env["jobId"], **env["localRecovery"]} for env in recovered],
        "recoveredReviewProvenance": [{"jobId": row["jobId"], "decision": row["decision"], **row["localRecovery"]} for row in recovered_decisions]}
    for name, data in assets.items():
        _atomic_bytes(out / name, data)
    _atomic_bytes(out / "manifest.json", json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-empty-abstrain-alias", action="store_true", help="Explicitly permit only the observed empty/null abstrainReason control-field typo")
    args = parser.parse_args(argv)
    try:
        report = recover(args.ledger, args.jobs, args.out, allow_empty_abstrain_alias=args.allow_empty_abstrain_alias)
        print(json.dumps({"format": FORMAT, "counts": report["counts"], "remainingGenerationCategories": report["remainingGenerationCategories"],
            "unrecoverableGenerationIssues": report["unrecoverableGenerationIssues"], "reviewIssues": report["reviewIssues"],
            "networkRequests": 0, "ledgerModified": False, "newModelApprovalsInvented": 0,
            "originalReviewDecisionsRecovered": report["counts"]["originalReviewDecisionsRecovered"],
            "originalReviewVerdictsRecovered": report["counts"]["originalReviewVerdictsRecovered"]}, sort_keys=True))
        return 0
    except (ValueError, TypeError, KeyError, OSError, sqlite3.Error):
        print(json.dumps({"error": "local_recovery_input_or_ledger_error", "networkRequests": 0, "ledgerModified": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
