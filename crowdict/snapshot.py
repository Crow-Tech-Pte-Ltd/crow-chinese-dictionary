"""Materialize one active, source-bound machine draft per requested job.

Selection never falls back to an older prompt version. Within the requested
version/model scope, only structurally valid drafts with exact generation-time
source evidence are eligible; newest updated time wins, then greatest fingerprint
as a deterministic tie-break. Chosen and superseded provenance is reported.
The ledger is opened read-only. No credentials or network requests are involved.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import closing
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile

from .generate import PROMPT_VERSION, validate_output
from .validate import canonical_bytes, canonical_sha, evidence_hash, read_jsonl, validate_draft

FORMAT = "crow-active-drafts/1"
SHA = re.compile(r"[a-f0-9]{64}\Z")


class SnapshotError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def normalize_job(value):
    job = dict(value)
    if "jobId" not in job and "id" in job:
        job["jobId"] = job["id"]
    if "id" not in job and "jobId" in job:
        job["id"] = job["jobId"]
    if (not isinstance(job.get("jobId"), str) or not job["jobId"] or len(job["jobId"]) > 200
            or job.get("jobId") != job.get("id")):
        raise SnapshotError("invalid_or_conflicting_job_identity")
    return job


def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix="." + path.name + ".", suffix=".pending", delete=False) as stream:
            name = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        name = None
    finally:
        if name is not None:
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass


def _encode_jsonl(rows, compressed):
    raw = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    # gzip.compress's OS header can differ across Python/zlib versions. Explicit
    # GzipFile with an empty filename yields stable bytes for this environment.
    if compressed:
        import io
        output = io.BytesIO()
        with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as stream:
            stream.write(raw)
        return output.getvalue()
    return raw


def _metadata(row):
    return {"jobId": row["job_id"], "generationFingerprint": row["fingerprint"],
            "generatorModel": row["model"], "promptVersion": row["prompt_version"]}


def materialize(ledger_path, jobs_path, out, *, prompt_version=PROMPT_VERSION, model=None, report_path=None):
    ledger_path, jobs_path, out = Path(ledger_path), Path(jobs_path), Path(out)
    if not isinstance(prompt_version, str) or not prompt_version.strip() or len(prompt_version) > 200:
        raise SnapshotError("explicit_valid_prompt_version_required")
    if model is not None and (not isinstance(model, str) or not model or len(model) > 200):
        raise SnapshotError("invalid_model_filter")
    if not ledger_path.is_file():
        raise SnapshotError("ledger_not_found")
    report_path = Path(report_path) if report_path else out.with_name(out.name + ".report.json")
    if out.resolve() in (ledger_path.resolve(), jobs_path.resolve()) or report_path.resolve() in (ledger_path.resolve(), jobs_path.resolve(), out.resolve()):
        raise SnapshotError("output_must_not_replace_inputs_or_other_output")
    jobs = {}
    for value in read_jsonl(jobs_path):
        job = normalize_job(value)
        if job["jobId"] in jobs:
            raise SnapshotError("duplicate_job_identity")
        jobs[job["jobId"]] = job
    choices, held, exclusions = {}, [], Counter()
    request_statuses = {}
    current_rows = 0
    with closing(sqlite3.connect(ledger_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)) as db:
        db.row_factory = sqlite3.Row
        columns = {row[1] for row in db.execute("PRAGMA table_info(requests)")}
        required = {"fingerprint", "job_id", "model", "prompt_version", "status", "updated", "output_json", "evidence_hash"}
        if not required <= columns:
            raise SnapshotError("unsupported_or_legacy_ledger_provenance")
        request_statuses = dict(db.execute("SELECT status,count(*) FROM requests GROUP BY status"))
        for row in db.execute("SELECT fingerprint,job_id,model,prompt_version,status,updated,output_json,evidence_hash FROM requests WHERE status='machine_draft' ORDER BY job_id,fingerprint"):
            issues = []
            if row["prompt_version"] != prompt_version:
                issues.append("prompt_version_not_selected")
            if model is not None and row["model"] != model:
                issues.append("model_not_selected")
            if issues:
                held.append({**_metadata(row), "issues": issues})
                exclusions.update(issues)
                continue
            current_rows += 1
            job = jobs.get(row["job_id"])
            if job is None:
                issues.append("unknown_job")
            if not isinstance(row["fingerprint"], str) or not SHA.fullmatch(row["fingerprint"]):
                issues.append("invalid_generation_fingerprint")
            timestamp = row["updated"]
            if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
                issues.append("invalid_generation_timestamp")
            stored_hash = row["evidence_hash"]
            if not isinstance(stored_hash, str) or not SHA.fullmatch(stored_hash):
                issues.append("missing_generation_evidence_provenance")
            elif job is not None and stored_hash != evidence_hash(job):
                issues.append("stale_generation_evidence")
            try:
                draft = json.loads(row["output_json"])
                dh = canonical_sha(draft)
            except (ValueError, TypeError):
                draft, dh = None, None
                issues.append("invalid_draft_json")
            result = {"jobId": row["job_id"], "generatorModel": row["model"], "draft": draft,
                      "evidenceHash": stored_hash, "draftSha256": dh, "generationFingerprint": row["fingerprint"],
                      "promptVersion": row["prompt_version"], "validationStatus": "machine_draft"}
            if job is not None and not issues:
                try:
                    validation_issues, _, _, _ = validate_draft(job, result)
                    issues.extend(validation_issues)
                    _, output_issues = validate_output(draft, job, [item["id"] for item in job["evidence"]])
                    issues.extend(output_issues)
                except (ValueError, TypeError, KeyError):
                    issues.append("invalid_draft_structure")
            if issues:
                issues = sorted(set(issues))
                held.append({**_metadata(row), "issues": issues})
                exclusions.update(issues)
                continue
            item = {"envelope": result, "updated": timestamp, "fingerprint": row["fingerprint"]}
            choices.setdefault(row["job_id"], []).append(item)
    selected, provenance = [], []
    for job_id, items in sorted(choices.items()):
        items.sort(key=lambda item: (item["updated"], item["fingerprint"]), reverse=True)
        chosen = items[0]
        selected.append(chosen["envelope"])
        provenance.append({"jobId": job_id, "generationFingerprint": chosen["fingerprint"],
            "generatorModel": chosen["envelope"]["generatorModel"], "promptVersion": prompt_version,
            "updated": chosen["updated"], "evidenceHash": chosen["envelope"]["evidenceHash"],
            "draftSha256": chosen["envelope"]["draftSha256"], "matchingValidCandidates": len(items),
            "supersededFingerprints": [item["fingerprint"] for item in items[1:]]})
    data = _encode_jsonl(selected, str(out).endswith(".gz"))
    report = {"format": FORMAT, "promptVersion": prompt_version, "modelFilter": model,
        "selection": "latest valid exact-source draft by updated descending, then fingerprint descending; no old-version fallback",
        "counts": {"jobs": len(jobs), "ledgerRequests": sum(request_statuses.values()),
            "machineDraftRows": request_statuses.get("machine_draft", 0), "currentScopeRows": current_rows,
            "selected": len(selected), "heldRows": len(held),
            "supersededCurrentDraftRows": sum(len(items) - 1 for items in choices.values())},
        "requestStatuses": request_statuses, "excludedByReason": dict(sorted(exclusions.items())),
        "chosen": provenance, "held": sorted(held, key=lambda item: (str(item["jobId"]), str(item["generationFingerprint"]))),
        "asset": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()},
        "networkRequests": 0, "credentialsRead": False, "humanReviews": 0,
        "validationStatus": "machine_draft"}
    _atomic_bytes(out, data)
    _atomic_bytes(report_path, json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--prompt-version", default=PROMPT_VERSION,
        help="Exact generator prompt version; defaults to the code's current PROMPT_VERSION")
    parser.add_argument("--model", help="Optional exact generator model ID; otherwise latest valid model per job wins")
    parser.add_argument("--out", required=True, type=Path, help="Private result-envelope JSONL or JSONL.gz")
    parser.add_argument("--report", type=Path, help="Private selection report; defaults to OUT.report.json")
    args = parser.parse_args(argv)
    try:
        report = materialize(args.ledger, args.jobs, args.out, prompt_version=args.prompt_version,
                             model=args.model, report_path=args.report)
        print(json.dumps({"format": report["format"], "promptVersion": report["promptVersion"],
            "modelFilter": report["modelFilter"], "counts": report["counts"],
            "excludedByReason": report["excludedByReason"], "asset": report["asset"],
            "networkRequests": 0, "credentialsRead": False, "humanReviews": 0}, sort_keys=True))
        return 0
    except SnapshotError as error:
        print(json.dumps({"error": error.code, "networkRequests": 0, "credentialsRead": False}), file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error):
        print(json.dumps({"error": "snapshot_input_or_ledger_error", "networkRequests": 0, "credentialsRead": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
