import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from crowdict.raw_readings import RawReadingSourceError
from crowdict.raw_sources import load_raw_source_rows, raw_source_catalog


class RawSourcesTests(unittest.TestCase):
    def fixture(self, records, **overrides):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        path = root / 'public/data/raw.jsonl.gz'
        path.parent.mkdir(parents=True)
        path.write_bytes(gzip.compress(('\n'.join(json.dumps(r, ensure_ascii=False) for r in records) + '\n').encode(), mtime=0))
        source = dict(id='wiktionary-fr', name='Wiktionary French', language='Chinese → French',
                      license='CC BY-SA 4.0', attribution='Source contributors', snapshot='old snapshot',
                      original='/data/raw.jsonl.gz', file='/data/compact.json.gz', count=99, **overrides)
        return root, source, path

    def test_catalog_has_actual_original_checksum_and_is_not_app_baseline(self):
        root, original, path = self.fixture([])
        source = raw_source_catalog(root, [original])[0]
        self.assertEqual(source['id'], 'raw-wiktionary-fr')
        self.assertEqual(source['file'], original['original'])
        self.assertEqual(source['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertFalse(source['baseline'])
        self.assertEqual(source['lang'], 'fr')
        self.assertEqual(source['spoken'], 'cmn')
        self.assertEqual(source['dataFormat'], 'wiktionary-jsonl')
        self.assertNotIn('count', source)
        self.assertEqual(original['file'], '/data/compact.json.gz')

    def test_single_reading_preserves_raw_position_gloss_and_sense_id(self):
        root, original, _ = self.fixture([
            {'word': 'not Chinese', 'lang_code': 'en'},
            {'word': '长时间', 'lang_code': 'zh', 'pos': 'noun',
             'sounds': [{'zh_pron': 'chángshíjiān', 'tags': ['Pinyin']}],
             'senses': [{'id': 'sense-one', 'glosses': ['Un long moment.']}]},
        ])
        row = load_raw_source_rows(root, raw_source_catalog(root, [original])[0])[0]
        self.assertEqual(row[:4], ['长时间', '长时间', 'chángshíjiān', 'Un long moment.'])
        self.assertEqual(row[4]['rawOriginalRowIndex'], 1)
        self.assertEqual(row[4]['sourceSense'], 'sense-one')
        self.assertEqual(row[4]['rawReadingFields'][0]['field'], 'sounds[0].zh_pron')
        self.assertNotIn('lookupAliases', row[4])

    def test_ambiguous_readings_keep_glosses_unassigned(self):
        root, original, _ = self.fixture([{'word': '行', 'lang_code': 'zh', 'pos': 'verb',
            'sounds': [{'zh_pron': 'xíng (xing2), háng (hang2)', 'tags': ['Pinyin']}],
            'senses': [{'id': 'one', 'glosses': ['Aller.']}, {'id': 'two', 'glosses': ['Rangée.']}]}])
        row = load_raw_source_rows(root, raw_source_catalog(root, [original])[0])[0]
        self.assertEqual(row[2], '')
        self.assertEqual(row[4]['readingCandidates'], ['xíng', 'háng'])
        self.assertEqual(row[4]['senseIds'], ['one', 'two'])
        self.assertEqual(row[3], 'Aller./Rangée.')

    def test_raw_only_scope_restrictions_are_preserved_without_changing_identity(self):
        root, original, _ = self.fixture([{'word': '零', 'lang_code': 'zh', 'pos': 'verb',
            'sounds': [{'zh_pron': 'líng', 'tags': ['Pinyin']}],
            'senses': [{'glosses': ['to fall'], 'raw_tags': ['of precipitation or tears']},
                       {'glosses': ['to wither'], 'tags': ['literary'], 'raw_tags': ['of plants or flowers']}]}])
        row = load_raw_source_rows(root, raw_source_catalog(root, [original])[0])[0]
        self.assertEqual(row[:4], ['零', '零', 'líng', 'to fall/to wither'])
        self.assertEqual(row[4]['rawSourceSenses'][0],
                         {'glosses': ['to fall'], 'raw_tags': ['of precipitation or tears']})
        self.assertEqual(row[4]['rawSourceSenses'][1],
                         {'glosses': ['to wither'], 'tags': ['literary'], 'raw_tags': ['of plants or flowers']})

    def test_numbered_and_marked_equivalents_are_one_candidate(self):
        root, original, _ = self.fixture([{'word': '本', 'lang_code': 'zh', 'pos': 'noun',
            'forms': [{'form': 'běn (ben3)', 'tags': ['transcription', 'Pinyin']}],
            'senses': [{'glosses': ['Buch.']}]}])
        row = load_raw_source_rows(root, raw_source_catalog(root, [original])[0])[0]
        self.assertEqual(row[2], 'běn')

    def test_japanese_tongyong_and_cantonese_are_not_chosen(self):
        root, original, _ = self.fixture([{'word': '保護', 'lang_code': 'zh', 'pos': 'verb',
            'sounds': [{'zh_pron': 'bǎohù', 'raw_tags': ['官話', '拼音']},
                       {'zh_pron': 'baohuh', 'raw_tags': ['官話', '通用拼音']},
                       {'zh_pron': 'bou2 wu6', 'tags': ['Cantonese', 'Pinyin']}],
            'senses': [{'glosses': ['守る。']}]}])
        row = load_raw_source_rows(root, raw_source_catalog(root, [original])[0])[0]
        self.assertEqual(row[2], 'bǎohù')
        self.assertEqual(len(row[4]['rawReadingFields']), 1)

    def test_mixed_reading_is_not_assigned(self):
        root, original, _ = self.fixture([{'word': '有劲儿', 'lang_code': 'zh',
            'sounds': [{'zh_pron': 'yǒujìnr5', 'tags': ['Pinyin']}],
            'senses': [{'glosses': ['Énergique.']}]}])
        row = load_raw_source_rows(root, raw_source_catalog(root, [original])[0])[0]
        self.assertEqual(row[2], '')
        self.assertEqual(row[4]['readingCandidates'], [])
        self.assertEqual(row[4]['rawReadingFields'][0]['value'], 'yǒujìnr5')

    def test_no_gloss_requests_romanizations_and_other_languages_are_skipped(self):
        root, original, _ = self.fixture([
            {'word': '词', 'lang_code': 'zh', 'senses': [{'tags': ['no-gloss'], 'glosses': ['placeholder']}]},
            {'word': '词', 'lang_code': 'zh', 'senses': [{'glosses': ['{{rfdef}}', 'This entry needs a definition.']}]},
            {'word': '词', 'lang_code': 'zh', 'pos': 'romanization', 'senses': [{'glosses': ['mot']}]},
            {'word': '本', 'lang_code': 'ja', 'senses': [{'glosses': ['book']}]},
            {'word': '本', 'lang_code': 'Hani', 'lang': 'Translingual', 'senses': [{'glosses': ['tree']}]},
            {'word': '词', 'lang_code': 'zh', 'senses': [{'glosses': ['mot/sens', 'mot/sens']}]},
        ])
        rows = load_raw_source_rows(root, raw_source_catalog(root, [original])[0])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][3], 'mot∕sens')
        self.assertEqual(rows[0][2], '')

    def test_nd_and_ai_originals_are_never_opened(self):
        for attributes in [{'license': 'CC BY-ND 3.0 TW'}, {'machineTranslated': True}]:
            root, source, path = self.fixture([])
            source.update(attributes)
            path.unlink()
            self.assertEqual(raw_source_catalog(root, [source]), [])

    def test_malformed_source_is_an_error(self):
        root, original, path = self.fixture([])
        source = raw_source_catalog(root, [original])[0]
        path.write_bytes(gzip.compress(b'{broken}\n', mtime=0))
        with self.assertRaisesRegex(RawReadingSourceError, 'Malformed JSON.*record 0'):
            load_raw_source_rows(root, source)


if __name__ == '__main__':
    unittest.main()
