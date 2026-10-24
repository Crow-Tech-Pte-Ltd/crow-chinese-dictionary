"""Runtime provenance is compact; full archival audit identity is retained."""
import copy
import unittest
from crowdict.validate import _public_record, _website_entry, canonical_sha, evidence_hash
from test_validate import fixture


class RuntimeProvenanceTests(unittest.TestCase):
    def test_runtime_omits_hashes_without_mutating_archival_hashes_or_source_fields(self):
        job, result = fixture()
        job['jobId'] = job['id']
        draft_hash = canonical_sha(result['draft'])
        source_hash = evidence_hash(job)
        record = _public_record(job, result, job['evidence'], draft_hash, source_hash, 'human_reviewed')
        before = copy.deepcopy(record)
        runtime = _website_entry(record)
        self.assertEqual(runtime[4]['definitionProvenance'], {
            'status': 'human_reviewed', 'automaticChecks': 'passed', 'generatorModel': result['generatorModel']})
        self.assertNotIn('draftSha256', runtime[4]['definitionProvenance'])
        self.assertNotIn('evidenceHash', runtime[4]['definitionProvenance'])
        self.assertEqual(record['validation']['draftSha256'], draft_hash)
        self.assertEqual(record['validation']['evidenceHash'], source_hash)
        self.assertEqual(record, before)
        self.assertEqual(runtime[4]['sourceSenses'], before['senses'])
        self.assertEqual(runtime[4]['lookupAliases'], before['scriptAliases'])
        self.assertEqual(runtime[4]['canonicalWord'], before['canonicalWord'])
        self.assertEqual(runtime[4]['dataOrigin'], before['dataOrigin'])
        self.assertEqual(runtime[0:3], [job['canonicalWord'], job['canonicalWord'], job['reading']])
        self.assertEqual(runtime[3], 'a cat')
