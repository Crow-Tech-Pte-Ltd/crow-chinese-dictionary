import contextlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from crowdict.generate import (Settings, ENDPOINTS, Ledger, BudgetStop, SafeHTTPError,
    NoRedirect, canonical, eligibility, build_request, validate_output, process_job,
    http_transport, main, BillingKeyError, check_billing_key, PROMPT_VERSION)


def config(**changes):
    value = Settings("openrouter", "qwen/test", ENDPOINTS["openrouter"], Decimal("0.1"),
        Decimal("0.6"), Decimal("1"), 10, 1_000_000, max_output_tokens=256)
    return replace(value, **changes)


def job(id="crow-example"):
    return {"id": id, "canonicalWord": "测试词", "aliases": ["測試詞", "测试词"],
        "reading": "ce4 shi4 ci2", "normalizedReading": "cèshìcí", "spokenLanguage": "cmn",
        "targetLanguage": "en", "generationEligible": True, "holdReasons": [], "gapKind": "missing_definition",
        "evidence": [{"id": "evidence-1", "sourceId": "dictionary-de", "language": "de",
           "definition": "Testwort", "reading": "ce4 shi4 ci2", "license": "CC BY-SA 4.0",
           "sourceUrl": "https://example.invalid/dictionary", "snapshot": "fixture",
           "originalRowIndex": 1, "quality": "editorial"}], "sourceLicenses": ["CC BY-SA 4.0"]}


def output(j):
    return {"jobId": j["id"], "canonicalWord": j["canonicalWord"], "reading": j["reading"],
        "spokenLanguage": j["spokenLanguage"], "targetLanguage": j["targetLanguage"],
        "senses": [{"definition": "a word used for testing", "pos": "", "register": "",
                    "sourceRefs": ["evidence-1"]}], "abstainReason": ""}


def response(j):
    return {"model": "qwen/test", "choices": [{"finish_reason": "stop", "message": {"content": canonical(output(j))}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 60, "cost": 0.000046}}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.cfg = config()
        self.ledger = Ledger(self.directory / "ledger.db", self.cfg)

    def tearDown(self):
        self.tmp.cleanup()

    def test_identity_and_reference_validation(self):
        j = job()
        value = output(j)
        self.assertTrue(validate_output(value, j, ["evidence-1"])[0])
        value["reading"] = "wrong"
        self.assertIn("identity_mismatch:reading", validate_output(value, j, ["evidence-1"])[1])
        value = output(j)
        value["senses"][0]["sourceRefs"] = ["invented-reference"]
        self.assertIn("invalid_source_reference", validate_output(value, j, ["evidence-1"])[1])
        value = output(j)
        value["confidence"] = 1
        self.assertFalse(validate_output(value, j, ["evidence-1"])[0])

    def test_restricted_and_machine_only_jobs_are_not_eligible(self):
        j = job()
        j["generationEligible"] = False
        self.assertIsNotNone(eligibility(j))
        j = job()
        j["evidence"][0]["license"] = "CC BY-ND 3.0 TW"
        self.assertEqual(eligibility(j), "restricted_or_unknown_evidence_license")
        j = job()
        j["evidence"][0]["quality"] = "machine_translated"
        self.assertEqual(eligibility(j), "machine_only_or_empty_evidence")
        j = job()
        j["reading"] = ""
        self.assertIsNotNone(eligibility(j))

    def test_known_crow_evidence_cannot_become_primary_by_omitting_machine_flag(self):
        for marker in [{"sourceId": "crow-en"}, {"sourceId": "crow-chinese-dictionary"},
                       {"sourceFamily": "derived-crow-family"}]:
            candidate = job()
            candidate["evidence"][0].update(marker)
            self.assertEqual(eligibility(candidate), "derived_corpus_not_primary_evidence")

    def test_unknown_and_component_licences_never_reach_prompt(self):
        for licence in ['All rights reserved', 'Source licenses + CC BY-SA 4.0']:
            candidate = job(); candidate['evidence'][0]['license'] = licence
            self.assertEqual(eligibility(candidate), 'restricted_or_unknown_evidence_license')
        candidate = job(); candidate['evidence'][0]['sourceFamily'] = 'wordnet-cow'
        self.assertEqual(eligibility(candidate), 'unreviewed_component_evidence_license')
        candidate = job(); candidate['evidence'][0]['quality'] = 'translation_only'
        self.assertEqual(eligibility(candidate), 'insufficient_source_definition')

    def test_prompt_preserves_attested_source_pos_and_usage_restrictions(self):
        j = job()
        j["evidence"][0]["sourcePos"] = "noun"
        j["evidence"][0]["rawSourceSenses"] = [{"id": "inner-source-sense", "glosses": ["Testwort"],
            "tags": ["archaic"], "raw_tags": ["veraltet"]}]
        payload = build_request(j, self.cfg)[0]
        shown = json.loads(payload["messages"][1]["content"])["evidence"][0]
        self.assertEqual(shown["sourcePos"], "noun")
        self.assertEqual(shown["rawSourceSenses"], j["evidence"][0]["rawSourceSenses"])
        self.assertEqual(PROMPT_VERSION, "crow-definition-v3")
        system = payload["messages"][0]["content"]
        self.assertIn("raw_tags", system)
        self.assertIn("inside the definition", system)
        self.assertIn("pos and", system)
        self.assertIn("BOTH be empty strings", system)
        self.assertIn("OUTER evidence IDs", system)

    def test_provider_parameters_and_url_allowlist(self):
        payload, _, bound, reserve, refs = build_request(job(), self.cfg)
        self.assertEqual(payload["provider"]["data_collection"], "deny")
        self.assertFalse(payload["provider"]["allow_fallbacks"])
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(payload["max_completion_tokens"], 256)
        self.assertGreater(bound, 1000)
        self.assertGreater(reserve, 0)
        with self.assertRaises(ValueError):
            replace(self.cfg, endpoint="https://example.invalid/steal")
        with self.assertRaises(SafeHTTPError):
            NoRedirect().redirect_request(None, None, 307, "", {}, "https://example.invalid")
        mini = replace(self.cfg, provider="minimax", endpoint=ENDPOINTS["minimax"])
        payload = build_request(job(), mini)[0]
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertTrue(payload["reasoning_split"])

    def test_draft_resume_does_not_send_twice(self):
        calls = []
        def transport(*args):
            calls.append(1)
            return response(job())
        first = process_job(job(), self.cfg, self.ledger, "TEST-SECRET", transport)
        second = process_job(job(), self.cfg, self.ledger, "TEST-SECRET", transport)
        self.assertEqual(first["status"], "machine_draft")
        self.assertEqual(second["status"], "skipped_existing_or_quarantined")
        reordered = job()
        reordered["priority"] = 9999
        process_job(reordered, self.cfg, self.ledger, "TEST-SECRET", transport)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.ledger.summary()["calls"], 1)
        text = (self.directory / "ledger.db").read_bytes()
        self.assertNotIn(b"TEST-SECRET", text)

    def test_budget_and_call_cap_include_attempts(self):
        cfg = replace(self.cfg, max_calls=1)
        ledger = Ledger(self.directory / "one.db", cfg)
        def limited(*args):
            raise SafeHTTPError(429)
        with self.assertRaises(BudgetStop):
            process_job(job(), cfg, ledger, "SECRET", limited, backoff=lambda _: None)
        self.assertEqual(ledger.summary()["calls"], 1)
        self.assertGreater(Decimal(ledger.summary()["reservedOrKnownUsd"]), 0)
        with self.assertRaises(ValueError):
            Ledger(self.directory / "one.db", replace(cfg, max_calls=2))

    def test_usd_cap_is_reserved_before_transport(self):
        cfg = replace(self.cfg, max_usd=Decimal("0.000001"))
        ledger = Ledger(self.directory / "tiny.db", cfg)
        calls = []
        with self.assertRaises(BudgetStop):
            process_job(job(), cfg, ledger, "SECRET", lambda *args: calls.append(1))
        self.assertEqual(calls, [])
        self.assertEqual(ledger.summary()["calls"], 0)

    def test_crash_reservation_survives_and_is_not_resent(self):
        j = job()
        _, fp, bound, amount, _ = build_request(j, self.cfg)
        self.ledger.reserve(j, fp, bound, amount)
        restarted = Ledger(self.directory / "ledger.db", self.cfg)
        calls = []
        result = process_job(j, self.cfg, restarted, "SECRET", lambda *args: calls.append(1))
        self.assertEqual(result["status"], "skipped_existing_or_quarantined")
        self.assertEqual(calls, [])
        self.assertEqual(restarted.summary()["calls"], 1)
        self.assertEqual(restarted.summary()["reservedOrKnownUsd"], str(Decimal(amount) / 1_000_000))

    def test_uncertain_failure_is_not_retried(self):
        calls = []
        def fail(*args):
            calls.append(1)
            raise SafeHTTPError(0, uncertain=True)
        result = process_job(job(), self.cfg, self.ledger, "SECRET", fail)
        self.assertEqual(result["status"], "quarantined")
        self.assertEqual(calls, [1])
        self.assertEqual(self.ledger.summary()["statuses"], {"quarantined": 1})

    def test_unknown_generation_is_not_resent_after_input_fingerprint_changes(self):
        def fail(*args):
            raise SafeHTTPError(0, uncertain=True)
        self.assertEqual(process_job(job(), self.cfg, self.ledger, "SECRET", fail)["status"], "quarantined")
        changed = job()
        changed["aliases"] = ["测试词"]
        changed["evidence"][0]["definition"] = "Another source wording"
        with patch("crowdict.generate.http_transport") as network:
            result = process_job(changed, self.cfg, self.ledger, "SECRET", network)
        network.assert_not_called()
        self.assertEqual(self.ledger.summary()["calls"], 1)

    def test_response_identity_error_is_schema_invalid(self):
        bad = response(job())
        value = output(job())
        value["targetLanguage"] = "fr"
        bad["choices"][0]["message"]["content"] = canonical(value)
        result = process_job(job(), self.cfg, self.ledger, "SECRET", lambda *args: bad)
        self.assertEqual(result["status"], "schema_invalid")
        self.assertIn("identity_mismatch:targetLanguage", result["issues"])

    def test_missing_cost_is_reserved_not_free(self):
        value = response(job())
        del value["usage"]["cost"]
        result = process_job(job(), self.cfg, self.ledger, "SECRET", lambda *args: value)
        self.assertEqual(result["status"], "machine_draft")
        summary = self.ledger.summary()
        self.assertTrue(summary["actualUsdMayBeUnknown"])
        self.assertGreater(Decimal(summary["reservedOrKnownUsd"]), 0)

    def test_provider_price_or_token_anomaly_halts_future_calls(self):
        value = response(job())
        value["usage"]["cost"] = 0.5
        process_job(job(), self.cfg, self.ledger, "SECRET", lambda *args: value)
        self.assertTrue(self.ledger.summary()["halted"])
        with self.assertRaises(BudgetStop):
            process_job(job("different"), self.cfg, self.ledger, "SECRET", lambda *args: response(job("different")))

    def test_independent_sqlite_connections_cannot_over_reserve(self):
        cfg = replace(self.cfg, max_calls=1)
        path = self.directory / "concurrent.db"
        Ledger(path, cfg)
        def worker(number):
            ledger = Ledger(path, cfg)
            try:
                return process_job(job(str(number)), cfg, ledger, "SECRET", lambda *args: response(job(str(number))))["status"]
            except BudgetStop:
                return "budget_stopped"
        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses = list(pool.map(worker, range(2)))
        self.assertEqual(statuses.count("machine_draft"), 1)
        self.assertEqual(statuses.count("budget_stopped"), 1)
        self.assertEqual(Ledger(path, cfg).summary()["calls"], 1)

    def test_http_errors_are_redacted_and_no_redirect_followed(self):
        bad = urllib.error.HTTPError(ENDPOINTS["openrouter"], 401, "SENSITIVE-HEADER-KEY", {}, io.BytesIO(b"SENSITIVE-KEY"))
        with patch("urllib.request.OpenerDirector.open", side_effect=bad):
            with self.assertRaises(SafeHTTPError) as caught:
                http_transport(ENDPOINTS["openrouter"], {}, "SENSITIVE-KEY", 1)
        self.assertEqual(caught.exception.code, 401)
        self.assertNotIn("SENSITIVE", str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)

    def test_utf8_prompt_bound_and_budget_increase_preserve_spend(self):
        j = job()
        j["evidence"][0]["definition"] = "汉字词语" * 100
        payload, _, bound, amount, _ = build_request(j, self.cfg)
        self.assertGreaterEqual(bound, len(canonical(payload).encode("utf-8")))
        ledger = Ledger(self.directory / "grow.db", replace(self.cfg, max_calls=1))
        process_job(job(), replace(self.cfg, max_calls=1), ledger, "SECRET", lambda *args: response(job()))
        before = ledger.summary()
        grown = Ledger(self.directory / "grow.db", self.cfg, allow_budget_increase=True)
        self.assertEqual(grown.summary()["calls"], before["calls"])
        self.assertEqual(grown.summary()["reservedOrKnownUsd"], before["reservedOrKnownUsd"])

    def test_unexpected_provider_secret_echo_is_redacted_before_storage(self):
        j = job()
        value = response(j)
        out = output(j)
        out["senses"][0]["definition"] = "SENSITIVE-KEY"
        value["choices"][0]["message"]["content"] = canonical(out)
        result = process_job(j, self.cfg, self.ledger, "SENSITIVE-KEY", lambda *args: value)
        self.assertEqual(result["status"], "schema_invalid")
        with contextlib.closing(sqlite3.connect(self.directory / "ledger.db")) as db:
            text = " ".join(str(r) for r in db.execute("select response_json from attempts"))
            self.assertNotIn("SENSITIVE-KEY", text)
            self.assertIn("[REDACTED]", text)

    def test_missing_or_wrong_returned_model_quarantines_and_halts(self):
        for actual_model in [None, "MiniMax-M3.1-Flash"]:
            with self.subTest(model=actual_model):
                value = response(job())
                if actual_model is None:
                    del value["model"]
                else:
                    value["model"] = actual_model
                ledger = Ledger(self.directory / ((actual_model or "missing") + ".db"), self.cfg)
                result = process_job(job(), self.cfg, ledger, "SECRET", lambda *args: value)
                self.assertEqual(result["status"], "quarantined")
                self.assertIn("returned_generator_model_mismatch_halted", result["issues"])
                self.assertTrue(ledger.summary()["halted"])
                with self.assertRaises(BudgetStop):
                    process_job(job("another"), self.cfg, ledger, "SECRET", lambda *args: response(job("another")))

    def test_equivalent_returned_model_alias_is_accepted(self):
        value = response(job())
        value["model"] = "Qwen/Test:free"
        result = process_job(job(), self.cfg, self.ledger, "SECRET", lambda *args: value)
        self.assertEqual(result["status"], "machine_draft")

    def test_token_plan_rejects_known_standard_key_before_dispatch(self):
        standard = "sk-api-" + "synthetic-fixture-not-a-real-key"
        with self.assertRaises(BillingKeyError):
            check_billing_key(standard, "token-plan")
        check_billing_key(standard, "payg")
        check_billing_key("future-subscription-format-fixture", "token-plan")
        cfg = replace(self.cfg, provider="minimax", endpoint=ENDPOINTS["minimax"],
                      billing="token-plan", allow_credit_pack_debit=True)
        ledger = Ledger(self.directory / "key-guard.db", cfg)
        with self.assertRaises(BillingKeyError):
            process_job(job(), cfg, ledger, standard, lambda *args: self.fail("Network dispatch"))
        self.assertEqual(ledger.summary()["calls"], 0)

    def test_evidence_hash_is_saved_before_dispatch(self):
        from crowdict.validate import evidence_hash
        expected = evidence_hash(job())
        def transport(*args):
            with self.ledger.connect() as db:
                value = db.execute("select evidence_hash from requests").fetchone()[0]
            self.assertEqual(value, expected)
            return response(job())
        process_job(job(), self.cfg, self.ledger, "SECRET", transport)
        altered = job()
        altered["evidence"][0]["definition"] = "changed source meaning"
        self.assertNotEqual(expected, evidence_hash(altered))
        with self.ledger.connect() as db:
            self.assertEqual(db.execute("select evidence_hash from requests").fetchone()[0], expected)

    def test_legacy_ledger_migration_leaves_old_hash_unknown(self):
        path = self.directory / "legacy.db"
        with contextlib.closing(sqlite3.connect(path)) as db:
            db.executescript("""CREATE TABLE requests(fingerprint TEXT PRIMARY KEY,job_id TEXT NOT NULL,
                provider TEXT NOT NULL,model TEXT NOT NULL,prompt_version TEXT NOT NULL,status TEXT NOT NULL,
                updated REAL NOT NULL,output_json TEXT,issues_json TEXT);
                CREATE TABLE budget(id INTEGER PRIMARY KEY,max_micro INTEGER,max_calls INTEGER,max_tokens INTEGER,
                spent_micro INTEGER,calls INTEGER,tokens_reserved INTEGER,halted INTEGER);
                INSERT INTO budget VALUES(1,1000000,10,1000000,123,1,234,0);
                INSERT INTO requests VALUES('old','old-job','openrouter','qwen/test','legacy','failed',1,NULL,'[]');""")
            db.commit()
        ledger = Ledger(path, self.cfg)
        with ledger.connect() as db:
            self.assertIsNone(db.execute("select evidence_hash from requests where fingerprint='old'").fetchone()[0])
            self.assertEqual(db.execute("select spent_micro,calls,tokens_reserved from budget").fetchone(), (123, 1, 234))

    def test_default_dry_run_reads_no_key_and_makes_no_network_request(self):
        path = self.directory / "jobs.jsonl"
        path.write_text(canonical(job()) + "\n")
        arguments = ["--jobs", str(path), "--provider", "openrouter", "--model", "qwen/test",
            "--input-usd-per-million", "0.1", "--output-usd-per-million", "0.6", "--max-usd", "1",
            "--max-calls", "3", "--max-total-tokens", "1000000", "--key-file", "/does/not/exist"]
        out = io.StringIO()
        with patch("crowdict.generate.read_key", side_effect=AssertionError("read key")), patch("crowdict.generate.http_transport", side_effect=AssertionError("network")), contextlib.redirect_stdout(out):
            self.assertEqual(main(arguments), 0)
        value = json.loads(out.getvalue())
        self.assertEqual(value["networkRequests"], 0)
        self.assertFalse(value["secretsRead"])
        self.assertTrue(value["dryRun"])
        err = io.StringIO()
        arguments[arguments.index("openrouter")] = "minimax"
        with contextlib.redirect_stderr(err):
            self.assertEqual(main(arguments + ["--execute", "--billing", "token-plan"]), 2)
        self.assertNotIn("SENSITIVE", err.getvalue())


if __name__ == "__main__":
    unittest.main()
