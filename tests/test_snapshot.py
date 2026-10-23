"""Active snapshot fixtures do not inspect the real ledger or credentials."""
import contextlib
from decimal import Decimal
import gzip
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from crowdict.generate import Settings, ENDPOINTS, Ledger
from crowdict.snapshot import materialize, SnapshotError, main
from crowdict.validate import canonical_sha, canonical_bytes, evidence_hash, read_results


def fixture():
    source = {"id": "fixture-source", "sourceId": "fixture-de", "language": "de", "definition": "Katze",
        "reading": "mao1", "license": "CC BY-SA 4.0", "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/",
        "sourceUrl": "https://example.org/cat", "attribution": "Synthetic fixture author", "quality": "full_definition",
        "snapshot": "fixture", "originalRowIndex": 0, "definitionOrigin": "source_attested"}
    job = {"id": "cat-job", "canonicalWord": "猫", "aliases": ["猫", "貓"], "reading": "mao1",
        "spokenLanguage": "cmn", "targetLanguage": "en", "generationEligible": True,
        "holdReasons": [], "evidence": [source]}
    job["evidenceHash"] = evidence_hash(job)
    draft = {"jobId": job["id"], "canonicalWord": "猫", "reading": "mao1", "spokenLanguage": "cmn",
        "targetLanguage": "en", "senses": [{"definition": "a cat", "pos": "", "register": "",
                                           "sourceRefs": ["fixture-source"]}], "abstainReason": ""}
    return job, draft


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.path = self.directory / "ledger.sqlite"
        cfg = Settings("openrouter", "fixture/model", ENDPOINTS["openrouter"], Decimal("0.1"), Decimal("0.6"),
                       Decimal("1"), 10, 1000000, max_output_tokens=256)
        self.ledger = Ledger(self.path, cfg)
        self.job, self.draft = fixture()
        self.jobs = self.directory / "jobs.jsonl"
        self.jobs.write_bytes(canonical_bytes(self.job) + b"\n")
        self.out = self.directory / "active.jsonl.gz"

    def tearDown(self):
        self.temp.cleanup()

    def insert(self, *, version="v2", model="fixture/model", updated=10, fingerprint=None,
               stored_hash="current", draft=None, status="machine_draft", job_id=None):
        if fingerprint is None:
            fingerprint = canonical_sha([version, model, updated, status, len(draft or self.draft)])
        if stored_hash == "current":
            stored_hash = self.job["evidenceHash"]
        with self.ledger.connect() as db:
            db.execute("INSERT INTO requests(fingerprint,job_id,provider,model,prompt_version,status,updated,output_json,issues_json,evidence_hash) VALUES(?,?,'openrouter',?,?,?,?,?,'[]',?)",
                (fingerprint, job_id or self.job["id"], model, version, status, updated,
                 json.dumps(draft or self.draft, ensure_ascii=False), stored_hash))
            db.commit()
        return fingerprint

    def read(self):
        with gzip.open(self.out, "rt", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    def snapshot(self, **kwargs):
        return materialize(self.path, self.jobs, self.out, prompt_version="v2", **kwargs)

    def test_current_version_only_and_no_old_version_fallback(self):
        self.insert(version="v1", updated=10000)
        report = self.snapshot()
        self.assertEqual(self.read(), [])
        self.assertEqual(report["excludedByReason"], {"prompt_version_not_selected": 1})
        fp = self.insert(version="v2", updated=1)
        report = self.snapshot()
        self.assertEqual(self.read()[0]["generationFingerprint"], fp)
        self.assertEqual(self.read()[0]["promptVersion"], "v2")
        self.assertEqual(report["counts"]["selected"], 1)

    def test_latest_valid_current_model_wins_and_is_reported(self):
        older = self.insert(model="fixture/older", updated=1)
        newest = self.insert(model="fixture/newer", updated=2)
        report = self.snapshot()
        row = self.read()[0]
        self.assertEqual(row["generationFingerprint"], newest)
        self.assertEqual(row["generatorModel"], "fixture/newer")
        chosen = report["chosen"][0]
        self.assertEqual(chosen["matchingValidCandidates"], 2)
        self.assertEqual(chosen["supersededFingerprints"], [older])
        self.assertEqual(report["counts"]["supersededCurrentDraftRows"], 1)
        self.snapshot(model="fixture/older")
        self.assertEqual(self.read()[0]["generationFingerprint"], older)

    def test_identical_timestamps_have_deterministic_fingerprint_tiebreak(self):
        low, high = "1" * 64, "f" * 64
        self.insert(updated=5, fingerprint=high)
        self.insert(updated=5, fingerprint=low)
        self.snapshot()
        self.assertEqual(self.read()[0]["generationFingerprint"], high)
        first = self.out.read_bytes()
        self.snapshot()
        self.assertEqual(self.out.read_bytes(), first)
        self.assertEqual(self.out.read_bytes()[4:8], b"\0\0\0\0")

    def test_missing_and_stale_stored_evidence_hash_are_held(self):
        self.insert(fingerprint="1" * 64, stored_hash=None)
        self.insert(fingerprint="2" * 64, stored_hash="0" * 64)
        report = self.snapshot()
        self.assertEqual(self.read(), [])
        self.assertEqual(report["counts"]["heldRows"], 2)
        self.assertIn("missing_generation_evidence_provenance", report["excludedByReason"])
        self.assertIn("stale_generation_evidence", report["excludedByReason"])

    def test_changed_same_id_source_cannot_inherit_old_generation(self):
        old_hash = self.job["evidenceHash"]
        self.insert(stored_hash=old_hash)
        self.job["evidence"][0]["definition"] = "a changed meaning at the same evidence ID"
        self.job["evidenceHash"] = evidence_hash(self.job)
        self.jobs.write_bytes(canonical_bytes(self.job) + b"\n")
        report = self.snapshot()
        self.assertEqual(self.read(), [])
        self.assertEqual(report["excludedByReason"], {"stale_generation_evidence": 1})

    def test_wrong_reading_unknown_job_and_invalid_model_output_are_held(self):
        altered = dict(self.draft, reading="mao2")
        self.insert(fingerprint="1" * 64, draft=altered)
        self.insert(fingerprint="2" * 64, job_id="unknown-job")
        altered = dict(self.draft, confidence=1)
        self.insert(fingerprint="3" * 64, draft=altered)
        report = self.snapshot()
        self.assertEqual(self.read(), [])
        self.assertEqual(report["counts"]["heldRows"], 3)
        self.assertIn("changed_reading", report["excludedByReason"])
        self.assertIn("unknown_job", report["excludedByReason"])
        self.assertIn("unexpected_or_missing_draft_fields", report["excludedByReason"])

    def test_only_machine_draft_status_is_selected(self):
        for index, status in enumerate(("abstained", "schema_invalid", "quarantined", "failed", "review_completed")):
            self.insert(fingerprint=str(index + 1) * 64, status=status)
        report = self.snapshot()
        self.assertEqual(self.read(), [])
        self.assertEqual(report["counts"]["ledgerRequests"], 5)
        self.assertEqual(report["counts"]["machineDraftRows"], 0)

    def test_snapshot_is_readonly_and_does_not_copy_raw_responses(self):
        self.insert()
        before = self.path.read_bytes()
        report = self.snapshot()
        self.assertEqual(before, self.path.read_bytes())
        row = self.read()[0]
        self.assertEqual(set(row), {"jobId", "generatorModel", "draft", "evidenceHash", "draftSha256", "generationFingerprint", "promptVersion", "validationStatus"})
        self.assertEqual(row["evidenceHash"], self.job["evidenceHash"])
        self.assertEqual(row["draftSha256"], canonical_sha(row["draft"]))
        self.assertEqual(report["asset"]["sha256"], hashlib.sha256(self.out.read_bytes()).hexdigest())
        envelopes = list(read_results(self.out, {self.job["id"]: self.job}))
        self.assertEqual(envelopes[0]["draft"], self.draft)

    def test_plain_jsonl_and_cli_report_have_no_network_or_key_access(self):
        self.insert()
        out = self.directory / "active.jsonl"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = main(["--ledger", str(self.path), "--jobs", str(self.jobs), "--prompt-version", "v2", "--out", str(out)])
        self.assertEqual(result, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["networkRequests"], 0)
        self.assertFalse(output["credentialsRead"])
        self.assertEqual(output["humanReviews"], 0)
        self.assertEqual(json.loads(out.read_text())["promptVersion"], "v2")

    def test_legacy_schema_rejected_and_inputs_never_overwritten(self):
        with self.assertRaises(SnapshotError):
            materialize(self.path, self.jobs, self.path, prompt_version="v2")
        self.jobs.write_text(canonical_bytes(self.job).decode() + "\n" + canonical_bytes(self.job).decode() + "\n")
        with self.assertRaises(SnapshotError):
            self.snapshot()
        legacy = self.directory / "legacy.sqlite"
        with contextlib.closing(sqlite3.connect(legacy)) as db:
            db.execute("CREATE TABLE requests(job_id TEXT)")
            db.commit()
        self.jobs.write_bytes(canonical_bytes(self.job) + b"\n")
        with self.assertRaises(SnapshotError) as caught:
            materialize(legacy, self.jobs, self.out, prompt_version="v2")
        self.assertEqual(caught.exception.code, "unsupported_or_legacy_ledger_provenance")


if __name__ == "__main__":
    unittest.main()
