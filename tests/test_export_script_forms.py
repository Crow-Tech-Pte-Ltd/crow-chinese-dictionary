"""Typed display forms are source metadata, never model-generated evidence."""
import copy
import unittest
from crowdict.inventory import job_script_forms
from crowdict.validate import (validate_draft, _public_record, _website_entry,
                               evidence_hash, canonical_sha)
from crowdict.generate import build_request
from test_generate import config, job as generation_job
from test_validate import fixture
from test_inventory import InventoryTests as _InventoryFixture


class TypedScriptFormsTests(unittest.TestCase):
    def electronic_product(self):
        job, result = fixture()
        job.update(canonicalWord='电子产品', aliases=['電子產品', '电子产品'], reading='dian4 zi3 chan3 pin3', jobId=job['id'])
        job['evidence'][0]['reading'] = job['reading']
        job['evidenceHash'] = evidence_hash(job)
        result['draft'].update(canonicalWord=job['canonicalWord'], reading=job['reading'])
        result['evidenceHash'] = job['evidenceHash']
        result['draftSha256'] = canonical_sha(result['draft'])
        return job, result

    def test_source_order_and_typed_equivalence_required(self):
        refs = [{'traditional': '電子産品', 'simplified': '电子产品'},
                {'traditional': '電子產品', 'simplified': '电子产品'}]
        self.assertEqual(job_script_forms(refs, '电子产品', ['電子產品', '电子产品']),
                         {'traditionalWord': '電子產品', 'simplifiedWord': '电子产品'})
        self.assertEqual(job_script_forms(refs, '电子产品', ['電子産品', '電子產品', '电子产品'])['traditionalWord'], '電子産品')
        self.assertEqual(job_script_forms([{'traditional': '電子產品', 'simplified': 'other'}], '电子产品', ['電子產品'])['traditionalWord'], '电子产品')

    def test_correct_both_website_slots_and_source_provenance(self):
        job, result = self.electronic_product()
        before = copy.deepcopy(job['evidence'])
        job.update(job_script_forms([{'traditional': '電子產品', 'simplified': '电子产品'}],job['canonicalWord'],job['aliases']))
        self.assertEqual(validate_draft(job, result)[0], [])
        self.assertEqual(job['evidence'], before)
        self.assertEqual(evidence_hash(job), result['evidenceHash'])
        record = _public_record(job,result,job['evidence'],result['draftSha256'],job['evidenceHash'],'human_reviewed')
        self.assertEqual(record['scriptForms'], {'traditional': '電子產品', 'simplified': '电子产品'})
        self.assertEqual(_website_entry(record)[:2], ['電子產品','电子产品'])
        self.assertEqual(_website_entry(record)[4]['canonicalWord'], '电子产品')

    def test_untyped_alias_never_invents_traditional_form(self):
        job, result = self.electronic_product()
        job.update(job_script_forms([],job['canonicalWord'],job['aliases']))
        self.assertEqual(job['traditionalWord'], '电子产品')
        record = _public_record(job,result,job['evidence'],result['draftSha256'],job['evidenceHash'],'human_reviewed')
        self.assertEqual(_website_entry(record)[:2], ['电子产品','电子产品'])
        for key in ['traditionalWord','simplifiedWord']:job.pop(key)
        record = _public_record(job,result,job['evidence'],result['draftSha256'],job['evidenceHash'],'human_reviewed')
        self.assertEqual(record['scriptForms'], {'traditional':'电子产品','simplified':'电子产品'})

    def test_unsafe_unlisted_or_noncanonical_forms_are_rejected(self):
        for key, value, expected in [('traditionalWord','捏造词','invalid_traditionalWord'),
                                     ('traditionalWord',None,'invalid_traditionalWord'),
                                     ('traditionalWord','電子 產品','invalid_traditionalWord'),
                                     ('simplifiedWord','電子產品','simplified_form_not_canonical')]:
            job,result=self.electronic_product();job[key]=value
            self.assertIn(expected, validate_draft(job,result)[0])

    def test_typed_form_metadata_does_not_change_prompt_or_cache_identity(self):
        job=generation_job();before=build_request(job,config())
        job.update(job_script_forms([{'traditional':'測試詞','simplified':'测试词'}],job['canonicalWord'],job['aliases']))
        self.assertEqual(build_request(job,config()),before)


class TypedFormsJobIntegrationTests(unittest.TestCase):
    setUp = _InventoryFixture.setUp
    add_pack = _InventoryFixture.add_pack
    build_inventory = _InventoryFixture.build_inventory
    make_jobs = _InventoryFixture.make_jobs

    def test_jobs_preserve_known_source_typed_form(self):
        self.add_pack('cfdict','French',[['電子產品','电子产品','dian4 zi3 chan3 pin3','produit électronique']])
        self.build_inventory()
        summary,generated,reused,held=self.make_jobs()
        self.assertEqual(len(generated),1)
        self.assertEqual((generated[0]['traditionalWord'],generated[0]['simplifiedWord']),('電子產品','电子产品'))
        self.assertEqual(generated[0]['evidenceHash'],evidence_hash(generated[0]))


del _InventoryFixture
