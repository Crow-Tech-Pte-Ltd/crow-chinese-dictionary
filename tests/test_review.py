"""Synthetic semantic-review protocol fixtures, never real corpus approvals."""
import contextlib
import copy
from dataclasses import replace
from decimal import Decimal
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from crowdict.generate import Settings, ENDPOINTS, Ledger, BudgetStop, SafeHTTPError, canonical, BillingKeyError
from crowdict.review import (CHECKS, prepare, validate_review, decision_record,
    review_one, write_outputs, main, normalize_job, run_reviews, REVIEW_VERSION)
from crowdict.validate import canonical_sha, evidence_hash, export


def fixture():
    evidence = {"id": "source-cat", "sourceId": "fixture-de", "language": "de", "definition": "Katze",
        "reading": "mao1", "license": "CC BY-SA 4.0", "quality": "full_definition",
        "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/", "sourceUrl": "https://example.org/cat",
        "attribution": "Synthetic fixture author", "snapshot": "test", "originalRowIndex": 0,
        "sourceFamily": "fixture", "definitionOrigin": "source_attested"}
    job = {"id": "cat-job", "canonicalWord": "猫", "aliases": ["貓", "猫"], "reading": "mao1",
        "spokenLanguage": "cmn", "targetLanguage": "en", "generationEligible": True,
        "holdReasons": [], "evidence": [evidence]}
    job["evidenceHash"] = evidence_hash(job)
    draft = {"jobId": job["id"], "canonicalWord": "猫", "reading": "mao1", "spokenLanguage": "cmn",
        "targetLanguage": "en", "senses": [{"definition": "a cat", "pos": "", "register": "",
        "sourceRefs": ["source-cat"]}], "abstainReason": ""}
    result = {"jobId": job["id"], "generatorModel": "MiniMax-M3.1-Flash", "draft": draft,
        "draftSha256": canonical_sha(draft), "evidenceHash": job["evidenceHash"], "generationFingerprint": "a" * 64}
    return job, result


def settings():
    return Settings("openrouter", "qwen/qwen-review-fixture", ENDPOINTS["openrouter"], Decimal("0.1"),
        Decimal("0.6"), Decimal("1"), 10, 2_000_000, max_output_tokens=2048)


def verdict(plan, decision="approve"):
    return {"jobId": plan["job"]["jobId"], "draftSha256": plan["draftSha256"],
        "evidenceHash": plan["evidenceHash"], "verdict": decision,
        "checks": {key: "pass" for key in CHECKS},
        "senseReviews": [{"senseIndex": 0, "verdict": decision, "sourceRefs": ["source-cat"],
                          "explanation": "The supplied German source denotes a cat; the proposed English sense preserves it."}],
        "notes": "Synthetic review fixture."}


def response(value, model="qwen/qwen-review-fixture"):
    return {"model": model, "choices": [{"finish_reason": "stop", "message": {"content": canonical(value)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 100, "cost": 0.00007}}


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.job, self.result = fixture()
        self.settings = settings()
        self.plan = prepare(self.job, self.result, self.settings)
        self.ledger = Ledger(self.directory / "ledger.db", self.settings)

    def tearDown(self):
        self.temp.cleanup()

    def test_request_has_draft_evidence_and_safe_provider_controls(self):
        payload = self.plan["payload"]
        self.assertIn('"definition":"Katze"', payload["messages"][1]["content"])
        self.assertIn(self.result["draftSha256"], payload["messages"][1]["content"])
        self.assertEqual(payload["provider"]["data_collection"], "deny")
        self.assertFalse(payload["provider"]["allow_fallbacks"])
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        mini = replace(self.settings, provider="minimax", endpoint=ENDPOINTS["minimax"])
        payload = prepare(self.job, self.result, mini)["payload"]
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertTrue(payload["reasoning_split"])
        self.assertNotIn("response_format", payload)

    def test_invalid_or_nd_draft_is_not_sent_for_review(self):
        job, result = fixture()
        job["evidence"][0]["license"] = "CC BY-ND 3.0 TW"
        job["evidenceHash"] = result["evidenceHash"] = evidence_hash(job)
        with self.assertRaises(ValueError):
            prepare(job, result, self.settings)
        job, result = fixture()
        result["draft"]["reading"] = "mao2"
        result["draftSha256"] = canonical_sha(result["draft"])
        with self.assertRaises(ValueError):
            prepare(job, result, self.settings)

    def test_hash_binding_and_per_sense_refs_are_required(self):
        value = verdict(self.plan)
        self.assertEqual(validate_review(value, self.plan), [])
        value["draftSha256"] = "0" * 64
        self.assertIn("review_identity_or_hash_mismatch", validate_review(value, self.plan))
        value = verdict(self.plan)
        value["senseReviews"][0]["sourceRefs"] = ["made-up-source"]
        self.assertIn("invalid_review_source_refs", validate_review(value, self.plan))
        value = verdict(self.plan)
        value["senseReviews"] = []
        self.assertIn("missing_per_sense_review", validate_review(value, self.plan))

    def test_uncertainty_cannot_hide_behind_overall_approval(self):
        for key in CHECKS:
            value = verdict(self.plan)
            value["checks"][key] = "uncertain"
            self.assertIn("contradictory_review_approval", validate_review(value, self.plan))
        value = verdict(self.plan)
        value["senseReviews"][0]["verdict"] = "reject"
        self.assertIn("contradictory_review_approval", validate_review(value, self.plan))

    def test_source_compatibility_is_mandatory_even_with_valid_citations(self):
        self.assertIn("sourceCompatibility", CHECKS)
        bound_schema = self.plan["payload"]["response_format"]["json_schema"]["schema"]
        self.assertEqual(bound_schema["properties"]["checks"]["properties"]["sourceCompatibility"]["enum"],
                         ["pass", "fail", "uncertain"])
        self.assertIn("sourceCompatibility", bound_schema["properties"]["checks"]["required"])
        value = verdict(self.plan)
        del value["checks"]["sourceCompatibility"]
        self.assertIn("invalid_review_checks", validate_review(value, self.plan))
        for check in ("fail", "uncertain"):
            value = verdict(self.plan)
            value["checks"]["sourceCompatibility"] = check
            # Cited refs and all other checks pass; source incompatibility alone
            # must still block an overall approval.
            self.assertIn("contradictory_review_approval", validate_review(value, self.plan))
        rejected = verdict(self.plan, "reject")
        rejected["checks"]["sourceCompatibility"] = "fail"
        self.assertEqual(validate_review(rejected, self.plan), [])

    def test_protocol_calls_out_numeric_and_lexical_source_errors_without_repairs(self):
        system = self.plan["payload"]["messages"][0]["content"]
        for evaluation_case in ("二千", "电子产品", "électroménager"):
            self.assertNotIn(evaluation_case, system)
        self.assertIn("Contradictory numeric values", system)
        self.assertIn("Preserve semantic scope", system)
        self.assertIn("same or related sourceFamily is not independent", system)
        self.assertIn("not proof that the source is correct", system)
        self.assertIn("ONLY to flag or hold", system)
        self.assertIn("never use", system)
        self.assertIn("DIFFERENT language", system)
        self.assertIn("raw_tags", system)

    def test_same_model_and_routing_alias_can_never_approve(self):
        for reviewer in ["MiniMax-M3.1-Flash", "minimax/MiniMax-M3.1-Flash:free", "MiniMax-M3.1-Flash-highspeed"]:
            record = decision_record(self.plan, verdict(self.plan), reviewer)
            self.assertEqual(record["reviewType"], "same_model_check")
            self.assertEqual(record["decision"], "uncertain")
        record = decision_record(self.plan, verdict(self.plan), "qwen/qwen-review-fixture")
        self.assertEqual(record["reviewType"], "independent_model")
        self.assertEqual(record["decision"], "approve")

    def test_ledger_resume_and_shared_budget(self):
        calls = []
        def transport(*args):
            calls.append(1)
            return response(verdict(self.plan))
        self.assertEqual(review_one(self.plan, self.settings, self.ledger, "SECRET", transport), "review_completed")
        self.assertEqual(review_one(self.plan, self.settings, self.ledger, "SECRET", transport), "skipped_existing_or_unknown")
        self.assertEqual(calls, [1])
        decisions, checks = self.directory / "decisions.jsonl", self.directory / "checks.jsonl"
        counts = write_outputs([self.plan], self.settings, self.ledger, decisions, checks)
        self.assertEqual(counts["approvals"], 1)
        self.assertEqual(counts["humanReviews"], 0)
        self.assertEqual(json.loads(decisions.read_text())["reviewType"], "independent_model")
        # A generation-stage reservation already exhausted the shared call cap.
        limited = replace(self.settings, max_calls=1)
        ledger = Ledger(self.directory / "limited.db", limited)
        with ledger.connect() as db:
            db.execute("UPDATE budget SET calls=1 WHERE id=1")
            db.commit()
        with self.assertRaises(BudgetStop):
            review_one(self.plan, limited, ledger, "SECRET", transport)
        self.assertEqual(calls, [1])

    def test_uncertain_charge_holds_review_without_retry(self):
        calls = []
        def transport(*args):
            calls.append(1)
            raise SafeHTTPError(0, uncertain=True)
        self.assertEqual(review_one(self.plan, self.settings, self.ledger, "SECRET", transport), "quarantined")
        self.assertEqual(calls, [1])
        decisions, checks = self.directory / "decisions.jsonl", self.directory / "checks.jsonl"
        self.assertEqual(write_outputs([self.plan], self.settings, self.ledger, decisions, checks)["approvals"], 0)
        self.assertEqual(decisions.read_text(), "")

    def test_actual_reviewer_model_identity_must_match_request(self):
        value = response(verdict(self.plan), "MiniMax-M3.1-Flash")
        status = review_one(self.plan, self.settings, self.ledger, "SECRET", lambda *args: value)
        self.assertEqual(status, "quarantined")
        self.assertTrue(self.ledger.summary()["halted"])
        decisions, checks = self.directory / "decisions.jsonl", self.directory / "checks.jsonl"
        self.assertEqual(write_outputs([self.plan], self.settings, self.ledger, decisions, checks)["approvals"], 0)

    def test_same_model_checks_remain_separate_from_decisions(self):
        same = replace(self.settings, model="MiniMax-M3.1-Flash")
        plan = prepare(self.job, self.result, same)
        self.assertFalse(plan["independent"])
        ledger = Ledger(self.directory / "same.db", same)
        review_one(plan, same, ledger, "SECRET", lambda *args: response(verdict(plan), "MiniMax-M3.1-Flash"))
        decisions, checks = self.directory / "decisions.jsonl", self.directory / "checks.jsonl"
        counts = write_outputs([plan], same, ledger, decisions, checks)
        self.assertEqual(counts["independentDecisions"], 0)
        self.assertEqual(counts["sameModelChecks"], 1)
        self.assertEqual(decisions.read_text(), "")
        self.assertEqual(json.loads(checks.read_text())["decision"], "uncertain")

    def test_output_can_only_export_as_machine_checked_opt_in(self):
        value = verdict(self.plan)
        decision = decision_record(self.plan, value, self.settings.model)
        paths = {name: self.directory / (name + ".jsonl") for name in ("jobs", "results", "decisions")}
        for name, row in (("jobs", self.job), ("results", self.result), ("decisions", decision)):
            paths[name].write_text(canonical(row) + "\n")
        manifest = export(paths["jobs"], paths["results"], paths["decisions"], self.directory / "default-export")
        self.assertEqual(manifest["counts"]["accepted"], 0)
        manifest = export(paths["jobs"], paths["results"], paths["decisions"], self.directory / "opt-in-export", include_machine_checked=True)
        self.assertEqual(manifest["counts"]["machineChecked"], 1)
        self.assertEqual(manifest["counts"]["humanReviewed"], 0)

    def test_dry_run_reads_no_credentials_or_network(self):
        jobs, results = self.directory / "jobs.jsonl", self.directory / "results.jsonl"
        jobs.write_text(canonical(self.job) + "\n")
        results.write_text(canonical(self.result) + "\n")
        argv = ["--jobs", str(jobs), "--results", str(results), "--provider", "openrouter",
            "--model", "qwen/qwen-review-fixture", "--input-usd-per-million", "0.1",
            "--output-usd-per-million", "0.6", "--max-usd", "1", "--max-calls", "10",
            "--max-total-tokens", "2000000", "--key-file", "/does/not/exist"]
        stdout = io.StringIO()
        with patch("crowdict.review.read_key", side_effect=AssertionError("read credential")), patch("crowdict.review.http_transport", side_effect=AssertionError("network")), contextlib.redirect_stdout(stdout):
            self.assertEqual(main(argv), 0)
        value = json.loads(stdout.getvalue())
        self.assertEqual(value["selectedDrafts"], 1)
        self.assertEqual(value["networkRequests"], 0)
        self.assertFalse(value["secretsRead"])

    def test_token_plan_standard_key_is_blocked_before_review_dispatch(self):
        cfg = replace(self.settings, provider="minimax", endpoint=ENDPOINTS["minimax"],
                      billing="token-plan", allow_credit_pack_debit=True)
        plan = prepare(self.job, self.result, cfg)
        ledger = Ledger(self.directory / "guard.db", cfg)
        with self.assertRaises(BillingKeyError):
            review_one(plan, cfg, ledger, "sk-api-" + "synthetic-fixture", lambda *args: self.fail("Network dispatch"))
        self.assertEqual(ledger.summary()["calls"], 0)

    def test_foreign_source_restrictions_are_visible_and_must_be_checked(self):
        job, result = fixture()
        job["evidence"][0]["sourcePos"] = "noun"
        job["evidence"][0]["rawSourceSenses"] = [{"glosses": ["Katze"], "tags": ["archaic"],
                                                 "raw_tags": ["veraltet"]}]
        job["evidenceHash"] = result["evidenceHash"] = evidence_hash(job)
        result["draft"]["senses"][0]["definition"] = "(archaic) a cat"
        result["draftSha256"] = canonical_sha(result["draft"])
        plan = prepare(job, result, self.settings)
        user = json.loads(plan["payload"]["messages"][1]["content"])
        shown = user["evidence"][0]
        self.assertEqual(shown["sourcePos"], "noun")
        self.assertEqual(shown["sourceFamily"], "fixture")
        self.assertEqual(shown["rawSourceSenses"], job["evidence"][0]["rawSourceSenses"])
        self.assertEqual(REVIEW_VERSION, "crow-semantic-review-v5")
        self.assertIn("usageRestrictions", CHECKS)
        self.assertIn("DIFFERENT language", plan["payload"]["messages"][0]["content"])
        self.assertIn("raw_tags", plan["payload"]["messages"][0]["content"])
        approved = verdict(plan)
        self.assertEqual(validate_review(approved, plan), [])
        approved["checks"]["usageRestrictions"] = "fail"
        self.assertIn("contradictory_review_approval", validate_review(approved, plan))

    def plans(self, count):
        plans = []
        for number in range(count):
            job, result = fixture()
            job["id"] = result["jobId"] = result["draft"]["jobId"] = "cat-job-" + str(number)
            result["draftSha256"] = canonical_sha(result["draft"])
            result["generationFingerprint"] = canonical_sha([number])
            plans.append(prepare(job, result, self.settings))
        return plans

    def test_bounded_concurrency_and_shared_reservations(self):
        cfg = replace(self.settings, workers=2)
        plans = self.plans(6)
        by_id = {plan["job"]["jobId"]: plan for plan in plans}
        lock = threading.Lock()
        active, peak = 0, 0
        def transport(endpoint, payload, key, timeout):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(.01)
                job_id = json.loads(payload["messages"][1]["content"])["jobId"]
                return response(verdict(by_id[job_id]))
            finally:
                with lock:
                    active -= 1
        run = run_reviews(plans, cfg, self.ledger, "SECRET", transport)
        self.assertFalse(run["circuitStopped"])
        self.assertLessEqual(peak, 2)
        self.assertEqual(run["statuses"], {"review_completed": 6})
        self.assertEqual(self.ledger.summary()["calls"], 6)

    def test_auth_or_quota_failure_stops_unstarted_review_jobs(self):
        cfg = replace(self.settings, workers=2)
        plans = self.plans(6)
        by_id = {plan["job"]["jobId"]: plan for plan in plans}
        barrier = threading.Barrier(2)
        calls = []
        lock = threading.Lock()
        def transport(endpoint, payload, key, timeout):
            job_id = json.loads(payload["messages"][1]["content"])["jobId"]
            with lock:
                calls.append(job_id)
            barrier.wait(timeout=2)
            if job_id == "cat-job-0":
                raise SafeHTTPError(401)
            time.sleep(.02)
            return response(verdict(by_id[job_id]))
        run = run_reviews(plans, cfg, self.ledger, "SECRET", transport)
        self.assertTrue(run["circuitStopped"])
        self.assertEqual(set(calls), {"cat-job-0", "cat-job-1"})
        self.assertEqual(self.ledger.summary()["calls"], 2)
        self.assertEqual(run["statuses"].get("review_failed"), 1)
        self.assertEqual(run["statuses"].get("provider_limit_stopped"), 1)

    def test_conflicting_job_aliases_rejected(self):
        job, _ = fixture()
        job["jobId"] = "different"
        with self.assertRaises(ValueError):
            normalize_job(job)


if __name__ == "__main__":
    unittest.main()
