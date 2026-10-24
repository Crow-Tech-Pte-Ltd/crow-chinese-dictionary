"""Control-only retry variants; no provider requests or credentials."""
import contextlib
import copy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from crowdict.generate import (build_request, canonical, Ledger, SYNTAX_RETRY_VERSION,
    PROMPT_VERSION, OUTPUT_SCHEMA, main as generate_main, validate_output)
from crowdict.review import (prepare, reserve_review, validate_review, REVIEW_VERSION,
    SCHEMA, main as review_main)
from crowdict.validate import canonical_sha, evidence_hash
from test_generate import config, job, output
from test_review import fixture, settings, verdict


class SyntaxRetryTests(unittest.TestCase):
    def test_default_payloads_remain_exactly_identical_and_schemas_unchanged(self):
        generator_schema, review_schema = copy.deepcopy(OUTPUT_SCHEMA), copy.deepcopy(SCHEMA)
        original = build_request(job(), config())[0]
        current_job, result = fixture()
        review = prepare(current_job, result, settings())["payload"]
        self.assertEqual(hashlib.sha256(canonical(original).encode()).hexdigest(),
            "2a6e9f0d86d8bda4092ae3351aeb81dc5595326f8263489e691d5948382b180f")
        self.assertEqual(hashlib.sha256(canonical(review).encode()).hexdigest(),
            "4c1e75110183a240ee3049da1e04dcc99c08baa153b0598ddfc2460ecd7c6aca")
        build_request(job(), config(syntax_retry=True))
        prepare(current_job, result, replace(settings(), syntax_retry=True))
        self.assertEqual(OUTPUT_SCHEMA, generator_schema)
        self.assertEqual(SCHEMA, review_schema)
        self.assertEqual(build_request(job(), config())[0], original)
        self.assertEqual(prepare(current_job, result, settings())["payload"], review)

    def test_generator_variant_changes_only_control_format_and_fingerprint(self):
        current_job = job()
        original, old_fingerprint, *_ = build_request(current_job, config())
        retried, fingerprint, *_ = build_request(current_job, config(syntax_retry=True))
        self.assertNotEqual(fingerprint, old_fingerprint)
        self.assertEqual(PROMPT_VERSION, "crow-definition-v3")
        self.assertEqual(original["messages"][1], retried["messages"][1])
        self.assertTrue(retried["messages"][0]["content"].startswith(original["messages"][0]["content"]))
        template = json.loads(retried["messages"][0]["content"].split("Template: ")[-1])
        for key in ("canonicalWord", "reading", "spokenLanguage", "targetLanguage"):
            self.assertEqual(template[key], current_job[key])
        self.assertEqual(template["jobId"], current_job["id"])
        self.assertEqual(set(template), set(output(current_job)))
        self.assertEqual(template["abstainReason"], "")
        schema = retried["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["reading"]["enum"], [current_job["reading"]])
        self.assertEqual(schema["properties"]["senses"]["items"]["properties"]["sourceRefs"]["items"]["enum"], ["evidence-1"])
        invalid = output(current_job); invalid["reading"] = "wrong"
        self.assertFalse(validate_output(invalid, current_job, ["evidence-1"])[0])

    def test_review_schema_binds_refs_to_each_sense_without_relaxing_validator(self):
        current_job, result = fixture()
        extra = copy.deepcopy(current_job["evidence"][0]); extra.update(id="source-other", definition="ein anderes Tier")
        current_job["evidence"].append(extra)
        current_job["evidenceHash"] = result["evidenceHash"] = evidence_hash(current_job)
        result["draft"]["senses"].append({"definition": "another animal", "pos": "", "register": "", "sourceRefs": ["source-other"]})
        result["draftSha256"] = canonical_sha(result["draft"])
        original = prepare(current_job, result, settings())
        retried = prepare(current_job, result, replace(settings(), syntax_retry=True))
        self.assertEqual(REVIEW_VERSION, "crow-semantic-review-v5")
        self.assertNotEqual(original["fingerprint"], retried["fingerprint"])
        self.assertEqual(original["payload"]["messages"][1], retried["payload"]["messages"][1])
        schema = retried["payload"]["response_format"]["json_schema"]["schema"]
        reviews = schema["properties"]["senseReviews"]
        self.assertEqual((reviews["minItems"], reviews["maxItems"]), (2, 2))
        alternatives = reviews["items"]["anyOf"]
        for index, ref in enumerate(("source-cat", "source-other")):
            self.assertEqual(alternatives[index]["properties"]["senseIndex"]["enum"], [index])
            self.assertEqual(alternatives[index]["properties"]["sourceRefs"]["items"]["enum"], [ref])
        value = verdict(retried)
        value["senseReviews"].append({"senseIndex": 1, "verdict": "approve", "sourceRefs": ["source-other"], "explanation": "Synthetic source-supported check."})
        self.assertEqual(validate_review(value, retried), [])
        value["senseReviews"][0]["sourceRefs"] = ["source-other"]
        self.assertIn("invalid_review_source_refs", validate_review(value, retried))

    def test_ledger_variant_migration_preserves_history_and_records_both_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            ledger = Ledger(path, config())
            payload, fingerprint, bound, amount, refs = build_request(job(), config())
            ledger.reserve(job(), fingerprint, bound, amount)
            ledger.settle(1, fingerprint, "schema_invalid", issues=["fixture"])
            before = ledger.summary()
            with ledger.connect() as db:
                db.execute("ALTER TABLE requests DROP COLUMN request_variant")
                db.commit()
            cfg = config(syntax_retry=True)
            ledger = Ledger(path, cfg)
            self.assertEqual(ledger.summary(), before)
            with ledger.connect() as db:
                self.assertIsNone(db.execute("SELECT request_variant FROM requests WHERE fingerprint=?", (fingerprint,)).fetchone()[0])
            _, retry_fingerprint, bound, amount, refs = build_request(job(), cfg)
            self.assertIsNotNone(ledger.reserve(job(), retry_fingerprint, bound, amount))
            self.assertIsNone(ledger.reserve(job(), retry_fingerprint, bound, amount))
            current_job, result = fixture()
            reviewer = replace(settings(), syntax_retry=True, max_total_tokens=1_000_000)
            plan = prepare(current_job, result, reviewer)
            self.assertIsNotNone(reserve_review(plan, reviewer, ledger))
            with ledger.connect() as db:
                variants = dict(db.execute("SELECT fingerprint,request_variant FROM requests"))
                self.assertEqual(variants[retry_fingerprint], SYNTAX_RETRY_VERSION)
                self.assertEqual(variants[plan["fingerprint"]], SYNTAX_RETRY_VERSION)
                self.assertEqual(db.execute("SELECT count(*) FROM attempts").fetchone()[0], 3)

    def test_both_cli_flags_dry_run_without_secrets_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            current_job, result = fixture()
            jobs = directory / "jobs.jsonl"; jobs.write_text(canonical(current_job) + "\n")
            results = directory / "results.jsonl"; results.write_text(canonical(result) + "\n")
            common = ["--jobs", str(jobs), "--ledger", str(directory / "ledger.sqlite"), "--provider", "openrouter", "--model", "qwen/test", "--input-usd-per-million", "0.1", "--output-usd-per-million", "0.6", "--max-usd", "1", "--max-calls", "10", "--max-total-tokens", "1000000", "--syntax-retry", "--key-file", "/does/not/exist"]
            with patch("crowdict.generate.read_key", side_effect=AssertionError("secret read")), patch("crowdict.generate.http_transport", side_effect=AssertionError("network")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(generate_main(common), 0)
            with patch("crowdict.review.read_key", side_effect=AssertionError("secret read")), patch("crowdict.review.http_transport", side_effect=AssertionError("network")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(review_main(common + ["--results", str(results)]), 0)


if __name__ == "__main__":
    unittest.main()
