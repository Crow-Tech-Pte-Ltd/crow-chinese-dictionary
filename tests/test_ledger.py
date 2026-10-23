"""Operator recovery fixtures never touch the real pilot ledger or credentials."""
import contextlib
from dataclasses import replace
from decimal import Decimal
import io
import json
from pathlib import Path
import tempfile
import unittest

from crowdict.generate import Settings, ENDPOINTS, Ledger, SafeHTTPError, process_job, build_request
from crowdict.ledger import summary, requeue_rejected, LedgerOperationError, main


def config():
    return Settings("openrouter", "fixture/model", ENDPOINTS["openrouter"], Decimal("0.1"), Decimal("0.6"),
                    Decimal("1"), 10, 1000000, max_output_tokens=256)


def job():
    return {"id": "operator-fixture", "canonicalWord": "猫", "aliases": ["猫", "貓"], "reading": "mao1",
        "spokenLanguage": "cmn", "targetLanguage": "en", "gapKind": "missing_target_headword",
        "generationEligible": True, "holdReasons": [], "evidence": [{"id": "source-fixture", "sourceId": "fixture",
        "language": "de", "definition": "Katze", "reading": "mao1", "license": "CC BY-SA 4.0", "quality": "full_definition"}]}


class LedgerOperatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.path = self.directory / "budget.db"
        self.settings = config()
        self.ledger = Ledger(self.path, self.settings)

    def tearDown(self):
        self.temp.cleanup()

    def reject(self, code=402):
        def transport(*args):
            raise SafeHTTPError(code)
        return process_job(job(), self.settings, self.ledger, "synthetic-token", transport, backoff=lambda _: None)

    def rows(self):
        with self.ledger.connect() as db:
            return (list(db.execute("select * from budget")), list(db.execute("select * from attempts")))

    def test_requeue_auth_rejection_preserves_every_usage_and_attempt(self):
        self.assertEqual(self.reject()["status"], "failed")
        before = self.rows()
        result = requeue_rejected(self.path, job()["id"])
        self.assertTrue(result["usageAndReservationsRetained"])
        self.assertEqual(result["httpCode"], 402)
        self.assertEqual(self.rows(), before)
        with self.ledger.connect() as db:
            row = db.execute("select fingerprint,status,evidence_hash from requests").fetchone()
            self.assertEqual(row[0], build_request(job(), self.settings)[1])
            self.assertEqual(row[1], "retryable")
            self.assertIsNotNone(row[2])
            self.assertEqual(db.execute("select count(*) from ledger_operator_events").fetchone()[0], 1)
        with self.assertRaises(LedgerOperationError):
            requeue_rejected(self.path, job()["id"])
        # Resume consumes another allowed attempt; no history is erased.
        self.reject(401)
        self.assertEqual(self.ledger.summary()["calls"], 2)
        requeue_rejected(self.path, job()["id"])
        self.reject(401)
        self.assertEqual(self.ledger.summary()["calls"], 2)  # Existing max_attempts still applies.

    def test_all_allowed_rejection_codes_require_explicit_one_job(self):
        for code in (401, 402, 403, 404, 422):
            with self.subTest(code=code):
                path = self.directory / (str(code) + ".db")
                ledger = Ledger(path, self.settings)
                def transport(*args):
                    raise SafeHTTPError(code)
                process_job(job(), self.settings, ledger, "synthetic", transport)
                self.assertEqual(requeue_rejected(path, job()["id"])["httpCode"], code)

    def test_unknown_charge_cannot_be_requeued(self):
        def transport(*args):
            raise SafeHTTPError(0, uncertain=True)
        process_job(job(), self.settings, self.ledger, "synthetic", transport)
        before = self.rows()
        with self.assertRaises(LedgerOperationError):
            requeue_rejected(self.path, job()["id"])
        self.assertEqual(self.rows(), before)

    def test_success_schema_invalid_and_provider_anomaly_are_not_requeued(self):
        self.reject()
        for status in ("machine_draft", "abstained", "schema_invalid", "review_completed", "quarantined"):
            with self.subTest(status=status):
                with self.ledger.connect() as db:
                    db.execute("update requests set status=?", (status,))
                    db.commit()
                with self.assertRaises(LedgerOperationError):
                    requeue_rejected(self.path, job()["id"])
        with self.ledger.connect() as db:
            db.execute("update requests set status='failed'")
            db.execute("update budget set halted=1")
            db.commit()
        with self.assertRaises(LedgerOperationError) as caught:
            requeue_rejected(self.path, job()["id"])
        self.assertEqual(caught.exception.code, "halted_ledger_requires_separate_investigation")

    def test_transient_or_ambiguous_rejection_is_not_operator_requeued(self):
        self.reject(500)
        with self.assertRaises(LedgerOperationError):
            requeue_rejected(self.path, job()["id"])

    def test_prior_unknown_history_cannot_be_hidden_by_final_auth_error(self):
        self.reject()
        with self.ledger.connect() as db:
            db.execute("update attempts set status='quarantined'")
            db.commit()
        with self.assertRaises(LedgerOperationError):
            requeue_rejected(self.path, job()["id"])

    def test_summary_has_no_prompt_responses_or_credentials(self):
        self.reject()
        before = self.rows()
        value = summary(self.path)
        self.assertEqual(value["calls"], 1)
        self.assertEqual(value["networkRequests"], 0)
        self.assertFalse(value["credentialsRead"])
        self.assertEqual(value["definitivelyRejectedRequests"][0]["jobId"], job()["id"])
        text = json.dumps(value)
        self.assertNotIn("Katze", text)
        self.assertNotIn("synthetic-token", text)
        self.assertEqual(self.rows(), before)

    def test_cli_never_implicitly_requeues_all_requests(self):
        self.reject()
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            self.assertEqual(main(["--ledger", str(self.path), "--requeue-rejected"]), 2)
        self.assertEqual(json.loads(errors.getvalue())["error"], "job_id_required_for_requeue")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["--ledger", str(self.path), "--summary"]), 0)
        self.assertEqual(json.loads(output.getvalue())["calls"], 1)


if __name__ == "__main__":
    unittest.main()
