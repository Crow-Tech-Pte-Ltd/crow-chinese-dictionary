import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from crowdict.batch import main, stopped
from crowdict.common import write_jsonl

class BatchTests(unittest.TestCase):
    def test_dry_run_never_starts_provider_process_or_reads_keys(self):
        with tempfile.TemporaryDirectory() as d:
            jobs=Path(d)/'jobs.jsonl.gz'
            write_jsonl(jobs,[{'id':'a','generationEligible':True},{'id':'held','generationEligible':False}])
            out=io.StringIO()
            with patch('crowdict.batch.run_step') as process, contextlib.redirect_stdout(out):
                main(['--jobs',str(jobs),'--ledger',str(Path(d)/'absent.sqlite'),
                      '--max-usd','2','--max-calls','10','--max-total-tokens','100000',
                      '--minimax-key-file','/nonexistent/private-key'])
            process.assert_not_called()
            self.assertFalse((Path(d)/'absent.sqlite').exists())
            self.assertEqual(json.loads(out.getvalue())['jobs'],1)
            self.assertFalse(json.loads(out.getvalue())['secretsRead'])

    def test_provider_and_budget_circuit_stop(self):
        self.assertTrue(stopped({'runResults':{'provider_limit_stopped':1}}))
        self.assertTrue(stopped({'statuses':{'budget_stopped':1}}))
        self.assertTrue(stopped({'ledger':{'halted':True}}))
        self.assertFalse(stopped({'runResults':{'skipped_completed':4}}))
