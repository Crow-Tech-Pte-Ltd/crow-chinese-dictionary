import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from crowdict.raw_readings import RawReadingSourceError, iter_raw_readings


class RawReadingsTests(unittest.TestCase):
    def fixture(self, records, **overrides):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        path = root / 'public/data/source.jsonl.gz'
        path.parent.mkdir(parents=True)
        path.write_bytes(gzip.compress(('\n'.join(json.dumps(r, ensure_ascii=False) for r in records) + '\n').encode(), mtime=0))
        dictionary = {'id': 'wiktionary-test', 'license': 'CC BY-SA 4.0', 'original': '/data/source.jsonl.gz', **overrides}
        return root, dictionary, path

    def test_french_pinyin_and_cantonese_are_distinguished(self):
        root, dictionary, path = self.fixture([{'word': '火', 'lang_code': 'zh', 'sounds': [
            {'zh_pron': 'huǒ', 'tags': ['Mandarin', 'Pinyin']},
            {'zh_pron': 'fo2', 'tags': ['Cantonese', 'Pinyin']},
            {'ipa': 'xuɔ', 'tags': ['Mandarin', 'IPA']},
        ]}])
        result = list(iter_raw_readings(root, [dictionary]))
        self.assertEqual([r['reading'] for r in result], ['huǒ'])
        self.assertEqual(result[0]['sourceRowIndex'], 0)
        self.assertEqual(result[0]['field'], 'sounds[0].zh_pron')
        self.assertEqual(result[0]['sourceSha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(result[0]['spokenLanguage'], 'cmn')

    def test_japanese_mandarin_pinyin_excludes_tongyong_and_other_varieties(self):
        root, dictionary, _ = self.fixture([{'word': '保護', 'lang_code': 'zh', 'sounds': [
            {'zh_pron': 'bǎohù', 'raw_tags': ['官話', '拼音']},
            {'zh_pron': 'bǎohù', 'raw_tags': ['官話', '通用拼音']},
            {'zh_pron': 'pao3 hu4', 'tags': ['Wade-Giles'], 'raw_tags': ['官話']},
            {'zh_pron': 'bou2 wu6', 'tags': ['Cantonese', 'Jyutping']},
            {'zh_pron': 'bao3 hu4', 'raw_tags': ['客家話', '拼音']},
        ]}])
        self.assertEqual([r['reading'] for r in iter_raw_readings(root, [dictionary])], ['bǎohù'])

    def test_german_pinyin_forms_and_portuguese_raw_tag(self):
        root, dictionary, _ = self.fixture([
            {'word': '本', 'lang_code': 'zh', 'forms': [
                {'form': 'běn (ben3)', 'tags': ['transcription', 'Pinyin']},
                {'form': 'pen3', 'tags': ['transcription', 'Wade-Giles']},
            ]},
            {'word': '二', 'lang_code': 'zh', 'sounds': [{'zh_pron': 'èr (er4)', 'raw_tags': ['Pinyin:'], 'tags': ['romanization']}]},
        ])
        rows = list(iter_raw_readings(root, [dictionary]))
        self.assertEqual([r['reading'] for r in rows], ['běn', 'ben3', 'èr', 'er4'])
        self.assertEqual(rows[-1]['sourceRowIndex'], 1)
        self.assertEqual(rows[0]['field'], 'forms[0].form')

    def test_reading_lists_keep_all_attested_candidates(self):
        root, dictionary, _ = self.fixture([{'word': '蛙', 'lang_code': 'zh', 'sounds': [{'zh_pron': 'wā (wa1), jué (jue2)', 'tags': ['Pinyin']}]}])
        self.assertEqual([r['reading'] for r in iter_raw_readings(root, [dictionary])], ['wā', 'wa1', 'jué', 'jue2'])

    def test_no_derivatives_ai_and_unlicensed_originals_are_never_read(self):
        for overrides in [{'license': 'CC BY-ND 3.0 TW'}, {'machineTranslated': True}, {'license': 'All rights reserved'}]:
            root, dictionary, path = self.fixture([], **overrides)
            path.unlink()
            self.assertEqual(list(iter_raw_readings(root, [dictionary])), [])

    def test_wrong_language_ipa_and_non_mandarin_han_are_excluded(self):
        root, dictionary, _ = self.fixture([
            {'word': '水', 'lang_code': 'ja', 'sounds': [{'zh_pron': 'shuǐ', 'tags': ['Pinyin']}]},
            {'word': '水', 'lang_code': 'zh', 'sounds': [{'zh_pron': '/ʂweɪ/', 'tags': ['Mandarin', 'IPA']}]},
            {'word': '水', 'lang_code': 'Hani', 'lang': 'Translingual', 'sounds': [{'zh_pron': 'shuǐ', 'tags': ['Pinyin']}]},
        ])
        self.assertEqual(list(iter_raw_readings(root, [dictionary])), [])

    def test_malformed_original_is_reported(self):
        root, dictionary, path = self.fixture([])
        path.write_bytes(gzip.compress(b'{bad}\n', mtime=0))
        with self.assertRaisesRegex(RawReadingSourceError, 'Malformed JSON.*record 0'):
            list(iter_raw_readings(root, [dictionary]))

    def test_path_traversal_is_rejected(self):
        root, dictionary, _ = self.fixture([], original='/data/../../.minimax-key.jsonl.gz')
        with self.assertRaises(RawReadingSourceError):
            list(iter_raw_readings(root, [dictionary]))


if __name__ == '__main__':
    unittest.main()
