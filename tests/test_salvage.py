"""Synthetic local recovery fixtures; no corpus approvals or provider requests."""
import copy
import contextlib
import io
from unittest.mock import patch
from dataclasses import replace
from decimal import Decimal
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from crowdict.generate import Settings, ENDPOINTS, Ledger, canonical, validate_output
from crowdict.review import CHECKS, prepare, reserve_review, validate_review
from crowdict.salvage import complete_json, normalize_abstention, RecoveryError, recover, main
from crowdict.validate import canonical_sha, evidence_hash


def fixture(number=0):
    source={"id":"fixture-evidence","sourceId":"fixture-de","language":"de","definition":"Katze",
        "reading":"mao1","license":"CC BY-SA 4.0","licenseUrl":"https://creativecommons.org/licenses/by-sa/4.0/",
        "sourceUrl":"https://example.org/cat","attribution":"Synthetic test contributor","quality":"full_definition","sourceFamily":"fixture"}
    job={"id":"job-"+str(number),"canonicalWord":"猫","aliases":["猫","貓"],"reading":"mao1","spokenLanguage":"cmn","targetLanguage":"en",
         "generationEligible":True,"holdReasons":[],"evidence":[source]}
    job["evidenceHash"]=evidence_hash(job)
    draft={"jobId":job["id"],"canonicalWord":"猫","reading":"mao1","spokenLanguage":"cmn","targetLanguage":"en",
        "senses":[{"definition":"a cat","pos":"","register":"","sourceRefs":["fixture-evidence"]}],"abstainReason":""}
    return job,draft


def read(path):
    with gzip.open(path,"rt") as f:return [json.loads(line) for line in f if line.strip()]


class RecoveryTests(unittest.TestCase):
    def test_only_complete_unique_json_and_benign_transport_wrappers(self):
        j,d=fixture()
        value,reasons=complete_json(" \ufeff"+canonical(d)) if False else complete_json("\ufeff  ```json\n"+canonical(d)+"\n``` \n")
        self.assertEqual(value,d)
        self.assertIn("single_json_fence_removed",reasons)
        for invalid in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1} {"a":2}', 'prefix {"a":1}', '{"a":']:
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):complete_json(invalid)

    def test_missing_null_and_known_snake_control_normalizations_are_narrow(self):
        j,d=fixture()
        for variant in ("missing","null","snake"):
            value=copy.deepcopy(d)
            if variant=="missing":del value["abstainReason"]
            elif variant=="null":value["abstainReason"]=None
            else:value["abstain_reason"]=value.pop("abstainReason")
            original=copy.deepcopy(value)
            result,reasons=normalize_abstention(value)
            self.assertEqual(result,d)
            self.assertEqual(value,original)
            self.assertTrue(reasons)
            self.assertTrue(validate_output(result,j,["fixture-evidence"])[0])

    def test_typo_requires_explicit_permission_and_empty_unique_control(self):
        j,d=fixture()
        value=copy.deepcopy(d);value["abstrainReason"]=value.pop("abstainReason")
        self.assertFalse(validate_output(normalize_abstention(value)[0],j,["fixture-evidence"])[0])
        self.assertEqual(normalize_abstention(value,True)[0],d)
        for bad in ("insufficient evidence",[],False):
            value["abstrainReason"]=bad
            with self.assertRaises(RecoveryError):normalize_abstention(value,True)
        value=copy.deepcopy(d);value["abstrainReason"]=""
        result,_=normalize_abstention(value,True)
        self.assertFalse(validate_output(result,j,["fixture-evidence"])[0])

    def test_no_lexical_identity_label_or_reference_repairs(self):
        j,d=fixture()
        value=copy.deepcopy(d);value.pop("abstainReason")
        value["reading"]="mao2";value["senses"][0]["sourceRefs"]=["invented"]
        value["senses"][0]["definition"]="(archaic) changed definition"
        value["senses"][0]["pos"]="noun";value["senses"][0]["register"]="archaic"
        result,_=normalize_abstention(value)
        for key in value:self.assertEqual(result[key],value[key])
        self.assertFalse(validate_output(result,j,["fixture-evidence"])[0])

    def test_full_long_notes_survive_but_urls_private_text_and_controls_do_not(self):
        j,d=fixture();j["jobId"]=j["id"]
        plan={"job":j,"result":{"draft":d},"draftSha256":canonical_sha(d),"evidenceHash":evidence_hash(j)}
        value={"jobId":j["id"],"draftSha256":plan["draftSha256"],"evidenceHash":plan["evidenceHash"],"verdict":"approve",
            "checks":{key:"pass" for key in CHECKS},"senseReviews":[{"senseIndex":0,"verdict":"approve","sourceRefs":["fixture-evidence"],"explanation":"Source and definition match."}],"notes":"n"*700}
        self.assertEqual(validate_review(value,plan),[])
        for unsafe in ("n"*1025,"https://example.org/"+"n"*650,"/home/private/key"+"n"*650,"n"*650+"\x00"):
            bad=copy.deepcopy(value);bad["notes"]=unsafe
            self.assertIn("invalid_review_notes",validate_review(bad,plan))

    def test_cli_reports_recovered_counts_without_reading_keys_or_network(self):
        report={"counts":{"originalReviewDecisionsRecovered":1,"originalReviewVerdictsRecovered":{"approve":1}},
            "remainingGenerationCategories":{},"unrecoverableGenerationIssues":{},"reviewIssues":{}}
        output=io.StringIO()
        with patch("crowdict.salvage.recover",return_value=report),contextlib.redirect_stdout(output):
            self.assertEqual(main(["--ledger","/unused","--jobs","/unused-jobs","--out","/unused-out"]),0)
        self.assertEqual(json.loads(output.getvalue())["originalReviewDecisionsRecovered"],1)
        self.assertEqual(json.loads(output.getvalue())["networkRequests"],0)

    def test_end_to_end_recovery_never_mutates_ledger_or_invents_verdict(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/"ledger.sqlite"
            cfg=Settings("openrouter","fixture/generator",ENDPOINTS["openrouter"],Decimal(".1"),Decimal(".6"),Decimal("1"),10,1000000,max_output_tokens=2048)
            ledger=Ledger(path,cfg)
            j1,d1=fixture(1);j2,d2=fixture(2)
            bad=copy.deepcopy(d1);bad["abstrainReason"]=bad.pop("abstainReason")
            fp1=canonical_sha(["gen",1]);t1=ledger.reserve(j1,fp1,100,100)
            response={"model":cfg.model,"choices":[{"finish_reason":"stop","message":{"content":canonical(bad)}}]}
            ledger.settle(t1,fp1,"schema_invalid",response=response,output=bad,issues=["unexpected_or_missing_fields","invalid_abstain_reason"])
            fp2=canonical_sha(["gen",2]);t2=ledger.reserve(j2,fp2,100,100)
            ledger.settle(t2,fp2,"machine_draft",output=d2)
            env={"jobId":j2["id"],"generatorModel":cfg.model,"draft":d2,"draftSha256":canonical_sha(d2),"evidenceHash":evidence_hash(j2),"generationFingerprint":fp2}
            reviewcfg=replace(cfg,model="fixture/reviewer")
            reviewledger=Ledger(path,reviewcfg)
            plan=prepare(j2,env,reviewcfg)
            value={"jobId":j2["id"],"draftSha256":plan["draftSha256"],"evidenceHash":plan["evidenceHash"],"verdict":"approve",
                "checks":{key:"pass" for key in CHECKS},"senseReviews":[{"senseIndex":0,"verdict":"approve","sourceRefs":["fixture-evidence"],"explanation":"Source and definition match."}],"notes":"n"*700}
            token=reserve_review(plan,reviewcfg,reviewledger)
            reviewledger.settle(token,plan["fingerprint"],"review_invalid",response={"model":reviewcfg.model,"choices":[{"finish_reason":"stop","message":{"content":canonical(value)}}]},output=value,issues=["invalid_review_notes"])
            jobs=root/"jobs.jsonl";jobs.write_text(canonical(j1)+"\n"+canonical(j2)+"\n")
            before=path.read_bytes()
            report=recover(path,jobs,root/"recovery",allow_empty_abstrain_alias=True)
            self.assertEqual(before,path.read_bytes())
            self.assertEqual(report["counts"]["recoveredGenerationJobs"],1)
            self.assertEqual(report["counts"]["originalReviewDecisionsRecovered"],1)
            self.assertEqual(report["counts"]["newModelApprovalsInvented"],0)
            recovered=read(root/"recovery/recovered-generation.jsonl.gz")[0]
            self.assertEqual(recovered["draft"],d1)
            self.assertTrue(recovered["requiresFreshIndependentReview"])
            self.assertNotEqual(recovered["localRecovery"]["originalParsedOutputSha256"],recovered["draftSha256"])
            decision=read(root/"recovery/recovered-review-decisions.jsonl.gz")[0]
            self.assertEqual(decision["decision"],"approve")
            self.assertEqual(json.loads(decision["notes"])["notes"],value["notes"])
            self.assertEqual(decision["localRecovery"]["originalReviewSha256"],decision["localRecovery"]["revalidatedReviewSha256"])
            self.assertEqual(len(read(root/"recovery/needs-fresh-review.jsonl.gz")),1)


if __name__=="__main__":unittest.main()
