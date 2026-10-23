"""Small, source-backed gap-inventory regressions; no network or real keys."""
import contextlib
import gzip
import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crowdict.common import normalized_reading, source_family, source_policy, valid_reading
from crowdict.inventory import build, jobs


class InventoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / 'public/data'
        self.data.mkdir(parents=True)
        self.catalog = []
        self.database = self.root / 'work/inventory.sqlite'
        self.reports = self.root / 'reports'

    def add_pack(self, source_id, language, rows, **attributes):
        path = self.data / (source_id + '.json')
        raw = json.dumps(rows, ensure_ascii=False).encode()
        path.write_bytes(raw)
        source = dict(id=source_id, name=source_id, language='Chinese → ' + language,
                      license='CC BY-SA 4.0', licenseUrl='https://creativecommons.org/licenses/by-sa/4.0/',
                      attribution='Synthetic fixture contributors', url='https://example.invalid/source',
                      snapshot='synthetic-fixture', file='/data/' + path.name,
                      original='/data/' + source_id + '-source.jsonl.gz',
                      count=len(rows), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        source.update(attributes)
        if source_id.startswith('wiktionary-'):
            # Inventory also ingests retained raw originals; this fixture has
            # no additional source records beyond its patched attestations.
            (self.data / (source_id + '-source.jsonl.gz')).write_bytes(gzip.compress(b'', mtime=0))
        self.catalog.append(source)
        return source

    def build_inventory(self, raw=None):
        (self.data / 'catalog.json').write_text(json.dumps(self.catalog), encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            if raw is None:
                result = build(self.root, self.database, self.reports, raw_readings=False)
            else:
                with patch('crowdict.raw_readings.iter_raw_readings', return_value=iter(raw)):
                    result = build(self.root, self.database, self.reports, raw_readings=True)
        return result

    def make_jobs(self, language='en'):
        out = self.root / ('jobs-' + language)
        with contextlib.redirect_stdout(io.StringIO()):
            summary = jobs(self.database, self.root, language, out, limit=100, min_frequency=0)
        def read(kind):
            with gzip.open(out / (kind + '-' + language + '.jsonl.gz'), 'rt') as stream:
                return [json.loads(line) for line in stream]
        return summary, read('jobs'), read('reuse'), read('held')

    def resolutions(self, source):
        with contextlib.closing(sqlite3.connect(self.database)) as database:
            database.row_factory = sqlite3.Row
            return [dict(row) for row in database.execute(
                'SELECT z.*,r.source_id,r.row_index FROM resolutions z JOIN records r ON r.id=z.record_id WHERE r.source_id=? ORDER BY r.row_index', (source,))]

    def raw(self, reading, word='闪词'):
        return dict(sourceId='wiktionary-fr', word=word, reading=reading,
                    sourceFile='/data/raw-source.jsonl.gz', sourceRowIndex=0,
                    sourceSha256='f' * 64, field='sounds[0].zh_pron', spokenLanguage='cmn')

    def test_derived_crow_pack_is_coverage_but_never_primary_evidence(self):
        source = self.add_pack('crow-en', 'English', [['貓', '猫', 'mao1', 'a cat']])
        self.assertEqual(source_policy(source), 'derived_corpus_not_evidence')
        self.assertEqual(source_family(source), 'derived-crow-family')
        self.add_pack('cfdict', 'French', [['貓', '猫', 'mao1', 'chat']])
        self.build_inventory()
        self.assertEqual(self.make_jobs()[0]['selectedLexemes'], 0)
        with sqlite3.connect(self.database) as database:
            self.assertEqual(database.execute("SELECT policy,family,baseline FROM sources WHERE id='crow-en'").fetchone(),
                             ('derived_corpus_not_evidence', 'derived-crow-family', 1))

    def test_derived_crow_only_definitions_do_not_create_paid_jobs(self):
        self.add_pack('crow-fr', 'French', [['貓', '猫', 'mao1', 'chat']])
        self.build_inventory()
        self.assertEqual(self.make_jobs()[1], [])

    def test_reading_specific_aliases_do_not_merge_ni_and_nai(self):
        self.add_pack('cedict', 'English', [
            ['妳', '你', 'ni3', 'you'], ['妳', '奶', 'nai3', 'milk'],
        ])
        self.add_pack('cfdict', 'French', [
            ['妳', '妳', 'ni3', 'tu'], ['妳', '妳', 'nai3', 'lait'],
            ['妳', '妳', '', 'sens indéterminé'],
        ])
        self.build_inventory()
        rows = self.resolutions('cfdict')
        self.assertEqual([(r['canonical'], r['nr']) for r in rows[:2]], [('你', normalized_reading('ni3')), ('奶', normalized_reading('nai3'))])
        self.assertNotEqual(rows[0]['lexeme_id'], rows[1]['lexeme_id'])
        self.assertEqual(rows[2]['reading_status'], 'ambiguous')
        self.assertEqual(rows[2]['alias_status'], 'conflict')
        self.assertIsNone(rows[2]['lexeme_id'])
        self.assertEqual(self.make_jobs()[0]['generationEligible'], 0)

    def test_productive_quantity_expressions_are_not_automatic_core_entries(self):
        self.add_pack('cfdict', 'French', [['兩年','两年','liang3 nian2','deux ans']])
        self.build_inventory()
        summary, generated, reused, held = self.make_jobs()
        self.assertEqual(generated + reused, [])
        self.assertIn('productive_quantity_expression_review', held[0]['holdReasons'])
        self.assertEqual(held[0]['candidateKind'], 'possible_productive_quantity_expression')

    def test_existing_traditional_spelling_with_wrong_source_reading_is_held(self):
        self.add_pack('cedict', 'English', [['動作','动作','dong4 zuo4','movement']])
        self.add_pack('handedict', 'German', [['搭載','動作','guan1 ji1','ausschalten']])
        self.build_inventory()
        summary, generated, reused, held = self.make_jobs()
        self.assertEqual(generated + reused, [])
        self.assertIn('existing_target_alias_or_reading_review', held[0]['holdReasons'])

    def test_traditional_alias_is_existing_english_coverage(self):
        self.add_pack('cedict', 'English', [['學習', '学习', 'xue2 xi2', 'to study']])
        self.add_pack('cfdict', 'French', [['學習', '學習', 'xue2 xi2', 'étudier']])
        report = self.build_inventory()
        self.assertEqual(self.resolutions('cfdict')[0]['canonical'], '学习')
        self.assertEqual(report['referenceHeadwordsBySpokenLanguage']['cmn'], 1)
        self.assertEqual(self.make_jobs()[0]['selectedLexemes'], 0)

    def test_known_english_headword_with_other_reading_is_not_paid_gap(self):
        self.add_pack('cedict', 'English', [['行', '行', 'xing2', 'to walk']])
        self.add_pack('handedict', 'German', [['行', '行', 'hang2', 'Reihe']])
        self.build_inventory()
        self.assertEqual(self.resolutions('handedict')[0]['nr'], normalized_reading('hang2'))
        self.assertEqual(self.make_jobs()[0]['selectedLexemes'], 0)
        with contextlib.closing(sqlite3.connect(self.database)) as database:
            self.assertEqual(database.execute('SELECT count(*) FROM lexemes WHERE canonical="行"').fetchone()[0], 2)

    def test_nd_and_ai_sources_cannot_authorize_generation_or_reading_recovery(self):
        self.add_pack('moe-revised', 'Chinese', [['秘詞', '秘词', 'mi4 ci2', 'ND source definition']], license='CC BY-ND 3.0 TW')
        self.add_pack('ai-fr', 'French', [['夢詞', '梦词', 'meng4 ci2', 'machine only']], machineTranslated=True)
        self.add_pack('cfdict', 'French', [['秘詞', '秘词', '', 'sans lecture']])
        self.build_inventory()
        self.assertEqual(self.resolutions('cfdict')[0]['reading_status'], 'missing')
        summary, generated, reused, held = self.make_jobs()
        self.assertEqual(summary['selectedLexemes'], 0)
        self.assertEqual(generated + reused + held, [])

    def test_one_explicit_raw_reading_can_resolve_a_blank(self):
        self.add_pack('wiktionary-fr', 'French', [['闪词', '闪词', '', 'mot attesté']])
        self.build_inventory(raw=[self.raw('shan3 ci2')])
        row = self.resolutions('wiktionary-fr')[0]
        self.assertEqual(row['reading_status'], 'resolved_unique')
        self.assertEqual(row['nr'], normalized_reading('shan3 ci2'))
        summary, generated, reused, held = self.make_jobs()
        self.assertEqual(summary['generationEligible'], 1)
        self.assertEqual(generated[0]['readingProvenance'][0]['field'], 'sounds[0].zh_pron')
        self.assertEqual(reused + held, [])

    def test_conflicting_explicit_raw_readings_hold_a_blank(self):
        self.add_pack('wiktionary-fr', 'French', [['闪词', '闪词', '', 'mot attesté']])
        self.build_inventory(raw=[self.raw('shan3 ci2'), self.raw('shan4 ci2')])
        row = self.resolutions('wiktionary-fr')[0]
        self.assertEqual(row['reading_status'], 'ambiguous')
        self.assertIsNone(row['lexeme_id'])
        self.assertEqual(self.make_jobs()[0]['selectedLexemes'], 0)

    def test_one_non_ai_pack_reading_can_resolve_a_blank(self):
        self.add_pack('cfdict', 'French', [['闪词', '闪词', '', 'mot attesté']])
        self.add_pack('handedict', 'German', [['闪词', '闪词', 'shan3 ci2', 'belegtes Wort']])
        self.build_inventory()
        row = self.resolutions('cfdict')[0]
        self.assertEqual(row['reading_status'], 'resolved_unique')
        self.assertEqual(row['nr'], normalized_reading('shan3 ci2'))

    def test_three_wordnet_languages_are_one_family_and_need_component_review(self):
        for lang, name in [('fr', 'French'), ('de', 'German'), ('es', 'Spanish')]:
            self.add_pack('wordnet-' + lang, name, [['词', '词', 'ci2', 'translation']], translationOnly=True)
        self.build_inventory()
        self.assertEqual({source_family(s) for s in self.catalog}, {'wordnet-cow'})
        with contextlib.closing(sqlite3.connect(self.database)) as database:
            self.assertEqual(database.execute('SELECT count(DISTINCT family) FROM sources').fetchone()[0], 1)
            self.assertEqual(database.execute('SELECT count(*) FROM sources WHERE policy="component_review"').fetchone()[0], 3)
        self.assertEqual(self.make_jobs()[0]['generationEligible'], 0)

    def test_existing_target_ai_definition_prevents_duplicate_paid_work(self):
        self.add_pack('handedict-en-ai', 'English', [['闪词', '闪词', 'shan3 ci2', 'existing machine definition']], machineTranslated=True)
        self.add_pack('cfdict', 'French', [['闪词', '闪词', 'shan3 ci2', 'mot attesté']])
        self.build_inventory()
        self.assertEqual(self.make_jobs()[0]['selectedLexemes'], 0)

    def test_direct_target_hsk_definition_is_reuse_only(self):
        source = [{'simplified': '补词', 'forms': [{'traditional': '補詞', 'transcriptions': {'numeric': 'bu3 ci2'}, 'meanings': ['source English meaning']}]}]
        (self.data / 'hsk-source.json.gz').write_bytes(gzip.compress(json.dumps(source).encode(), mtime=0))
        (self.data / 'hsk-LICENSE.txt').write_text('Synthetic fixture permission under MIT')
        self.build_inventory()
        summary, generated, reused, held = self.make_jobs()
        self.assertEqual(summary['sourceReuseEligible'], 1)
        self.assertEqual(generated + held, [])
        self.assertEqual(reused[0]['evidence'][0]['definition'], 'source English meaning')
        self.assertTrue(reused[0]['reuseEligible'])
        self.assertFalse(reused[0]['generationEligible'])
        self.assertEqual(reused[0]['evidence'][0]['license'], 'MIT')

    def test_mixed_tone_marks_and_numbers_are_not_accepted(self):
        for reading in ['yǒujìnr5', 'bu4 yīhuǐr5', 'mǎ3', 'mǎ1']:
            with self.subTest(reading=reading):
                self.assertFalse(valid_reading(reading, 'cmn'))
        self.add_pack('cfdict', 'French', [['混词', '混词', 'mǎ3', 'mot']])
        self.build_inventory()
        self.assertEqual(self.resolutions('cfdict')[0]['reading_status'], 'invalid')
        self.assertEqual(self.make_jobs()[0]['selectedLexemes'], 0)

    def test_placeholder_or_non_pinyin_source_readings_are_not_attestations(self):
        for reading in ['unknown', 'not available', 'foo', 'zh4']:
            with self.subTest(reading=reading):
                self.assertFalse(valid_reading(reading, 'cmn'))

    def test_source_pos_name_stays_in_review_instead_of_paid_jobs(self):
        self.add_pack('cfdict', 'French', [['小青', '小青', 'xiao3 qing1', 'nom propre', {'pos': 'name'}]])
        self.build_inventory()
        summary, generated, reused, held = self.make_jobs()
        self.assertEqual(summary['generationEligible'], 0)
        self.assertEqual(generated + reused, [])
        self.assertEqual(len(held), 1)
        self.assertIn('proper_name_variant_or_abbreviation_needs_review', held[0]['holdReasons'])

    def test_external_english_snapshot_is_reuse_not_existing_app_coverage(self):
        self.add_pack('cfdict', 'French', [['新词','新词','xin1 ci2','mot nouveau']])
        self.build_inventory()
        external = self.root / 'upstream'
        directory = external / 'public/data'; directory.mkdir(parents=True)
        row = ['新词','新词','xin1 ci2','a new word',{'source':'external-en','pos':'noun'}]
        blob = gzip.compress((json.dumps(row,ensure_ascii=False)+'\n').encode(),mtime=0)
        (directory/'entries.jsonl.gz').write_bytes(blob)
        descriptor = {'root':str(external),'sources':[dict(id='external-en',name='External fixture',lang='en',spoken='cmn',baseline=False,language='Chinese → English',license='CC BY-SA 4.0',licenseUrl='https://creativecommons.org/licenses/by-sa/4.0/',attribution='Fixture authors',url='https://example.invalid/source',snapshot='fixture',dataFormat='entry-jsonl',file='/data/entries.jsonl.gz',original='/data/entries.jsonl.gz',sha256=hashlib.sha256(blob).hexdigest(),count=1)]}
        catalog = external/'catalog.json';catalog.write_text(json.dumps(descriptor))
        with contextlib.redirect_stdout(io.StringIO()):
            build(self.root,self.database,self.reports,raw_readings=False,extra_catalog=catalog)
        summary,generated,reused,held = self.make_jobs()
        self.assertEqual(summary['sourceReuseEligible'],1)
        self.assertEqual(generated+held,[])
        self.assertEqual(reused[0]['evidence'][0]['definition'],'a new word')
        self.assertEqual(reused[0]['evidence'][0]['sourcePos'],'noun')
        with contextlib.closing(sqlite3.connect(self.database)) as database:
            self.assertEqual(database.execute('SELECT baseline FROM sources WHERE id="external-en"').fetchone()[0],0)

    def test_secret_files_are_not_read_or_reflected_in_output(self):
        marker = 'SYNTHETIC_SECRET_MUST_NEVER_BE_READ'
        for name in ['.minimax-key', '.env']:
            (self.root / name).write_text(marker)
        self.add_pack('cfdict', 'French', [['词', '词', 'ci2', 'mot']])
        original = Path.open
        def guarded_open(path, *args, **kwargs):
            if path.name in {'.minimax-key', '.env'}:
                raise AssertionError('Inventory tried to read a credential file')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded_open):
            self.build_inventory()
            self.make_jobs()
        for folder in [self.reports, self.root / 'jobs-en']:
            for path in folder.rglob('*'):
                if path.is_file():
                    content = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
                    self.assertNotIn(marker.encode(), content)


if __name__ == '__main__':
    unittest.main()
