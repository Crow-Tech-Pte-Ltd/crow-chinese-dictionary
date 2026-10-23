import unittest
from crowdict.reuse import source_import
from crowdict.validate import validate_draft, evidence_hash
from test_validate import fixture

class ReuseTests(unittest.TestCase):
    def job(self):
        job, _ = fixture()
        job.update(reuseEligible=True, generationEligible=False, holdReasons=['same_language_source_reuse_available'])
        job['evidence'][0].update(language='en', definition='a cat; a fellow',
            rawSourceSenses=[{'glosses':['a cat'],'tags':[]},{'glosses':['a fellow'],'tags':['slang','dated']}])
        job['evidenceHash'] = evidence_hash(job)
        return job

    def test_exact_atomic_senses_preserve_pos_and_usage(self):
        job = self.job()
        result, error = source_import(job)
        self.assertIsNone(error)
        self.assertEqual(result['draft']['senses'][1]['register'], 'dated, slang')
        self.assertEqual(result['draft']['senses'][0]['pos'], 'noun')
        self.assertEqual(validate_draft(job,result)[0], [])

    def test_overflow_is_held_without_silent_truncation(self):
        job=self.job()
        job['evidence'][0]['rawSourceSenses']=[{'glosses':[f'sense {i}' for i in range(9)],'tags':[]}]
        result, error=source_import(job)
        self.assertIsNone(result)
        self.assertEqual(error, 'source_sense_overflow')

    def test_conflicting_reading_hold_stops_import(self):
        job=self.job();job['holdReasons'].append('unresolved_reading')
        self.assertEqual(source_import(job)[1], 'unresolved_holds')
