"""Provider cap fixtures use only disposable databases and mocked/local work."""
import contextlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crowdict.generate import (Settings, ENDPOINTS, Ledger, BudgetStop, build_request,
                              process_job, canonical, main as generate_main)
from crowdict.review import reserve_review, main as review_main
from crowdict.ledger import summary as ledger_summary
from crowdict.validate import canonical_sha, evidence_hash


def settings(provider="openrouter", cap=None):
    return Settings(provider, "fixture/model", ENDPOINTS[provider], Decimal("0.1"), Decimal("0.6"),
                    Decimal("50"), 100, 10000000, max_output_tokens=256, provider_max_usd=cap)


def job(number=0):
    return {"id": "job-" + str(number), "canonicalWord": "猫", "aliases": ["猫", "貓"], "reading": "mao1",
        "spokenLanguage": "cmn", "targetLanguage": "en", "generationEligible": True, "holdReasons": [],
        "gapKind": "missing_target_headword", "evidence": [{"id": "fixture-cat", "sourceId": "fixture",
            "language": "de", "definition": "Katze", "reading": "mao1", "license": "CC BY-SA 4.0",
            "quality": "full_definition", "attribution": "Synthetic fixture", "sourceFamily": "fixture",
            "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/",
            "sourceUrl": "https://example.org/cat"}]}


class ProviderBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "shared.sqlite"

    def tearDown(self):
        self.temp.cleanup()

    def reserve(self, ledger, number, amount):
        return ledger.reserve(job(number), canonical_sha([ledger.config.provider, number]), 1, amount)

    def test_provider_cap_survives_omission_and_other_provider_clients(self):
        cfg = settings(cap=Decimal("0.0001"))
        ledger = Ledger(self.path, cfg)
        token = self.reserve(ledger, 0, 60)
        ledger.settle(token, canonical_sha([cfg.provider, 0]), "failed", error_code=401)
        no_argument = Ledger(self.path, settings())
        with self.assertRaises(BudgetStop):
            self.reserve(no_argument, 1, 41)
        mini = Ledger(self.path, settings("minimax"))
        self.reserve(mini, 2, 200)
        summary = Ledger(self.path, settings()).summary()
        self.assertEqual(summary["providerBudgets"]["openrouter"], {"maxUsd": "0.0001", "reservedOrKnownUsd": "0.00006"})
        self.assertEqual(summary["providerBudgets"]["minimax"]["maxUsd"], None)
        self.assertEqual(summary["calls"], 2)
        self.assertEqual(ledger_summary(self.path)["providerBudgets"], summary["providerBudgets"])

    def test_first_cap_counts_historical_attempts_and_unknown_reservations(self):
        ledger = Ledger(self.path, settings())
        self.reserve(ledger, 0, 60)
        capped = Ledger(self.path, settings(cap=Decimal("0.0001")))
        self.reserve(capped, 1, 40)
        with self.assertRaises(BudgetStop):
            self.reserve(capped, 2, 1)
        self.assertEqual(capped.summary()["providerBudgets"]["openrouter"]["reservedOrKnownUsd"], "0.0001")
        self.assertEqual(capped.summary()["calls"], 2)

    def test_provider_cap_increase_requires_explicit_flag_and_keeps_spend(self):
        ledger = Ledger(self.path, settings(cap=Decimal("0.0001")))
        self.reserve(ledger, 0, 60)
        before = ledger.summary()
        with self.assertRaises(ValueError):
            Ledger(self.path, settings(cap=Decimal("0.0002")))
        self.assertEqual(ledger.summary(), before)
        raised = Ledger(self.path, settings(cap=Decimal("0.0002")), allow_budget_increase=True)
        self.assertEqual(raised.summary()["calls"], 1)
        self.assertEqual(raised.summary()["providerBudgets"]["openrouter"]["reservedOrKnownUsd"], "0.00006")
        self.reserve(raised, 1, 140)
        with self.assertRaises(BudgetStop):
            self.reserve(raised, 2, 1)
        self.assertEqual(raised.summary()["providerBudgets"]["openrouter"]["maxUsd"], "0.0002")
        self.assertEqual(raised.summary()["calls"], 2)

    def test_actual_settlement_refund_updates_provider_spend_without_reset(self):
        ledger = Ledger(self.path, settings(cap=Decimal("0.0001")))
        token = self.reserve(ledger, 0, 60)
        ledger.settle(token, canonical_sha(["openrouter", 0]), "failed", response={"usage": {"cost": 0.00002}})
        self.assertEqual(ledger.summary()["providerBudgets"]["openrouter"]["reservedOrKnownUsd"], "0.00002")
        self.reserve(ledger, 1, 80)
        with self.assertRaises(BudgetStop):
            self.reserve(ledger, 2, 1)
        self.assertEqual(ledger.summary()["calls"], 2)

    def test_generator_and_review_reservations_share_atomic_provider_limit(self):
        cfg = settings(cap=Decimal("0.0001"))
        Ledger(self.path, cfg)
        def worker(kind):
            ledger = Ledger(self.path, cfg)
            try:
                if kind == "generator":
                    self.reserve(ledger, 0, 60)
                else:
                    plan = {"job": {"jobId": "review-job"}, "fingerprint": "f" * 64,
                            "inputBound": 1, "reservedMicro": 60, "evidenceHash": "a" * 64}
                    reserve_review(plan, cfg, ledger)
                return "reserved"
            except BudgetStop:
                return "blocked"
        with ThreadPoolExecutor(max_workers=2) as pool:
            values = list(pool.map(worker, ["generator", "review"]))
        self.assertCountEqual(values, ["reserved", "blocked"])
        summary = Ledger(self.path, cfg).summary()
        self.assertEqual(summary["calls"], 1)
        self.assertEqual(summary["providerBudgets"]["openrouter"]["reservedOrKnownUsd"], "0.00006")

    def test_global_budget_still_applies_even_with_larger_provider_cap(self):
        ledger = Ledger(self.path, replace(settings(cap=Decimal("50")), max_usd=Decimal("0.0001")))
        with self.assertRaises(BudgetStop):
            self.reserve(ledger, 0, 101)
        self.assertEqual(ledger.summary()["calls"], 0)

    def test_invalid_provider_caps_are_rejected(self):
        for value in ("0", "-1", "NaN", "Infinity"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                settings(cap=Decimal(value))

    def test_explicit_cli_cap_reaches_settings_and_dry_run_stays_local(self):
        jobs = Path(self.temp.name) / "jobs.jsonl"
        candidate = job()
        jobs.write_text(canonical(candidate) + "\n")
        common = ["--jobs", str(jobs), "--provider", "openrouter", "--model", "fixture/model",
            "--input-usd-per-million", ".1", "--output-usd-per-million", ".6", "--max-usd", "50",
            "--provider-max-usd", ".000001", "--max-calls", "100", "--max-total-tokens", "10000000"]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), patch("crowdict.generate.read_key", side_effect=AssertionError("credential read")):
            self.assertEqual(generate_main(common), 0)
        self.assertEqual(json.loads(stdout.getvalue())["eligibleSelected"], 0)
        self.assertFalse(self.path.exists())
        draft = {"jobId": candidate["id"], "canonicalWord": "猫", "reading": "mao1", "spokenLanguage": "cmn",
                 "targetLanguage": "en", "senses": [{"definition": "a cat", "pos": "", "register": "", "sourceRefs": ["fixture-cat"]}], "abstainReason": ""}
        result = {"jobId": candidate["id"], "generatorModel": "minimax/fixture-other", "draft": draft,
                  "draftSha256": canonical_sha(draft), "evidenceHash": evidence_hash(candidate)}
        results = Path(self.temp.name) / "results.jsonl"
        results.write_text(canonical(result) + "\n")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), patch("crowdict.review.read_key", side_effect=AssertionError("credential read")):
            self.assertEqual(review_main(common + ["--results", str(results)]), 0)
        self.assertEqual(json.loads(stdout.getvalue())["selectedDrafts"], 0)


if __name__ == "__main__":
    unittest.main()
