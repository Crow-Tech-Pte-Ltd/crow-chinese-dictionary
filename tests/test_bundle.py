import tempfile
import unittest
from pathlib import Path
from crowdict.bundle import combine
from crowdict.common import write_jsonl

class BundleTests(unittest.TestCase):
    def test_mismatched_duplicate_scope_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.jsonl.gz',Path(d)/'b.jsonl.gz'
            write_jsonl(a,[{'id':'same','evidenceHash':'old'}]);write_jsonl(b,[{'id':'same','evidenceHash':'new'}])
            with self.assertRaisesRegex(ValueError,'Conflicting job scope'):combine([a,b])
            self.assertEqual(len(combine([a,a])),1)
