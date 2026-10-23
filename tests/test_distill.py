import contextlib
import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path

from crowdict.distill import distill


class DistillTests(unittest.TestCase):
    def test_streamed_english_source_requires_mandarin_and_omits_other_variety_senses(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'download.jsonl.gz'
            records = [
                {'word': '普通词', 'lang_code': 'zh', 'pos': 'noun',
                 'sounds': [{'zh_pron': 'pǔtōngcí', 'tags': ['Mandarin', 'Standard-Chinese', 'Pinyin']}],
                 'senses': [{'id': 'en-one', 'glosses': ['ordinary word'], 'examples': [{'text': 'NEVER_COPY_EXAMPLE'}]},
                            {'id': 'en-two', 'glosses': ['NEVER_COPY_CANTONESE'], 'tags': ['Cantonese']},
                            {'id': 'en-three', 'glosses': ['NEVER_COPY_HAKKA'], 'tags': ['Hakka']},
                            {'id': 'en-composite', 'glosses': ['NEVER_COPY_COMPOSITE_HOKKIEN'], 'tags': ['Taiwanese-Hokkien']},
                            {'id': 'en-regional', 'glosses': ['NEVER_COPY_REGIONAL_MANDARIN'], 'tags': ['Southwestern-Mandarin']},
                            {'id': 'en-uncertain', 'glosses': ['NEVER_COPY_UNCERTAIN_VARIETY'], 'tags': ['dialectal']},
                            {'id': 'en-error', 'glosses': ['NEVER_COPY_EXTRACTOR_ERROR'], 'tags': ['error-lua-timeout']}]},
                {'word': '方言词', 'lang_code': 'zh', 'sounds': [{'zh_pron': 'fong1jin4ci4', 'tags': ['Cantonese', 'Pinyin']}],
                 'senses': [{'glosses': ['NEVER_COPY_CANTONESE_RECORD']}]},
                {'word': '泛词', 'lang_code': 'zh', 'sounds': [{'zh_pron': 'fàncí', 'tags': ['Pinyin']}],
                 'senses': [{'glosses': ['NEVER_ASSUME_GENERIC_CHINESE']}]},
                {'word': '行', 'lang_code': 'zh', 'sounds': [{'zh_pron': 'xíng, háng', 'tags': ['Mandarin', 'Pinyin']}],
                 'senses': [{'id': 'en-four', 'glosses': ['go or row']}]},
                {'word': 'huǒ', 'lang_code': 'cmn', 'pos': 'romanization', 'senses': [{'glosses': ['NEVER_COPY_ROMANIZATION']}]},
                {'word': '汉', 'lang_code': 'Hani', 'lang': 'Translingual',
                 'sounds': [{'zh_pron': 'hàn', 'tags': ['Mandarin', 'Pinyin']}],
                 'senses': [{'glosses': ['Han character']}]},
            ]
            source.write_bytes(gzip.compress(('\n'.join(json.dumps(r, ensure_ascii=False) for r in records) + '\n').encode(), mtime=0))
            with contextlib.redirect_stdout(io.StringIO()):
                catalog = distill(source, root, snapshot_date='2026-10-21', dump_date='2026-10-01')
            descriptor = catalog['sources'][0]
            compact = root / 'public' / descriptor['file'].lstrip('/')
            with gzip.open(compact, 'rt') as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual([row[1] for row in rows], ['普通词', '行', '汉'])
            self.assertEqual(rows[0][3], 'ordinary word')
            self.assertEqual(rows[1][2], '')
            self.assertEqual(rows[1][4]['readingCandidates'], ['xíng', 'háng'])
            self.assertEqual(rows[0][4]['rawOriginalRowIndex'], 0)
            self.assertEqual(rows[0][4]['sourceArticleUrl'], 'https://en.wiktionary.org/wiki/%E6%99%AE%E9%80%9A%E8%AF%8D')
            self.assertEqual(descriptor['dataFormat'], 'entry-jsonl')
            self.assertFalse(descriptor['baseline'])
            self.assertEqual(descriptor['count'], 3)
            serialized = gzip.decompress(compact.read_bytes()).decode()
            self.assertNotIn('NEVER_', serialized)
            metrics = json.loads((root / 'distill-metrics.json').read_text())
            self.assertEqual(metrics['counts']['sensesExcludedOtherChineseVariety'], 4)
            self.assertEqual(metrics['counts']['ambiguousReadingRows'], 1)
            self.assertEqual(metrics['generationApiCalls'], 0)
            # Deterministic source bytes and catalog/checksums on resume.
            before = compact.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                repeated = distill(source, root, snapshot_date='2026-10-21', dump_date='2026-10-01')
            self.assertEqual(before, compact.read_bytes())
            self.assertEqual(catalog, repeated)


if __name__ == '__main__':
    unittest.main()
