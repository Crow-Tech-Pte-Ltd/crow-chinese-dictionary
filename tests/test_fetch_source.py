import gzip
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crowdict.fetch_source import MAX_DOWNLOAD_BYTES, fetch_source


class FakeResponse(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.headers = {'Content-Length': str(len(data))}


class FetchSourceTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        data = gzip.compress(b'{"word":"fixture"}\n', mtime=0)
        record = dict(languageBucket='Chinese', url='https://kaikki.org/dictionary/Chinese/kaikki.org-dictionary-Chinese.jsonl.gz', compressedBytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        provenance = root / 'provenance.json'
        provenance.write_text(json.dumps({'downloads': [record]}))
        return root, provenance, data, record

    def test_existing_source_is_verified_without_network(self):
        root, provenance, data, _ = self.fixture()
        destination = root / 'public/data/kaikki.org-dictionary-Chinese.jsonl.gz'
        destination.parent.mkdir(parents=True)
        destination.write_bytes(data)
        with patch('urllib.request.build_opener') as network:
            result = fetch_source(provenance, root, download=True)
            network.assert_not_called()
        self.assertTrue(result['verified'])
        self.assertEqual(result['downloadedBytesThisRun'], 0)

    def test_default_missing_source_never_downloads(self):
        root, provenance, _, _ = self.fixture()
        with patch('urllib.request.build_opener') as network:
            with self.assertRaises(FileNotFoundError):
                fetch_source(provenance, root)
            network.assert_not_called()

    def test_explicit_download_verifies_before_atomic_promotion(self):
        root, provenance, data, _ = self.fixture()
        with patch('urllib.request.build_opener') as network:
            network.return_value.open.return_value = FakeResponse(data)
            result = fetch_source(provenance, root, download=True)
        self.assertTrue(result['verified'])
        self.assertEqual(result['downloadedBytesThisRun'], len(data))
        self.assertEqual((root / 'public/data/kaikki.org-dictionary-Chinese.jsonl.gz').read_bytes(), data)

    def test_bad_existing_source_and_unapproved_or_oversized_download_are_rejected(self):
        root, provenance, data, record = self.fixture()
        destination = root / 'public/data/kaikki.org-dictionary-Chinese.jsonl.gz'
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            fetch_source(provenance, root, download=True)
        self.assertEqual(destination.read_bytes(), b'changed')
        for mutation in [{'url': 'https://example.invalid/source.jsonl.gz'}, {'compressedBytes': MAX_DOWNLOAD_BYTES + 1}]:
            provenance.write_text(json.dumps({'downloads': [record | mutation]}))
            with patch('urllib.request.build_opener') as network:
                with self.assertRaises(ValueError):
                    fetch_source(provenance, root, download=True)
                network.assert_not_called()


if __name__ == '__main__':
    unittest.main()
