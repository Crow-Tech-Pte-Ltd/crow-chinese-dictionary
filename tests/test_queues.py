import contextlib
import gzip
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from crowdict.common import canonical_json, normalized_reading, sha
from crowdict.inventory import SCHEMA
from crowdict.queues import build_queues


class QueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / 'inventory.sqlite'
        self.database = sqlite3.connect(self.path)
        self.addCleanup(self.database.close)
        self.database.executescript(SCHEMA)
        self.database.execute('INSERT INTO metadata VALUES(?,?)', ('sourceFingerprint', 'fixture-fingerprint'))

    def source(self, sid, lang, quality='full_definition', policy='compatible', baseline=True, spoken='cmn', family=None):
        catalog = dict(id=sid, license='CC BY-SA 4.0', sha256='a' * 64, file='/data/' + sid + '.json', secret='DO_NOT_EXPORT_SOURCE_SECRET')
        self.database.execute('INSERT INTO sources VALUES(?,?,?,?,?,?,?,?)',
                              (sid, lang, spoken, policy, family or sid, quality, int(baseline), canonical_json(catalog)))

    def record(self, sid, word, reading='ci2', status=None, sense='', spoken=None, row_index=0):
        language, source_spoken = self.database.execute('SELECT language,spoken FROM sources WHERE id=?', (sid,)).fetchone()
        spoken = spoken or source_spoken
        nr = normalized_reading(reading) if spoken == 'cmn' else reading.replace(' ', '').lower()
        status = status or ('source_attested' if reading else 'missing')
        cursor = self.database.execute('''INSERT INTO records(source_id,row_index,canonical,traditional,simplified,aliases,reading,nr,spoken,language,has_definition,sense_id,pos,proper)
                                          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                                       (sid, row_index, word, word, word, canonical_json([word]), reading, nr, spoken, language, 1, sense, '', 0))
        lid = sha(canonical_json([spoken, word, nr])) if nr else None
        self.database.execute('INSERT INTO resolutions VALUES(?,?,?,?,?,?,?,?)',
                              (cursor.lastrowid, word, reading, nr, spoken, status, 'unchanged', lid))
        if lid:
            self.database.execute('INSERT OR IGNORE INTO lexemes VALUES(?,?,?,?,?,?,?,?,?)', (lid, spoken, word, nr, reading, 0, 0, 99, len(word)))

    def run_queues(self, limit=1000, label='queues'):
        self.database.executescript('''DROP TABLE IF EXISTS head_coverage;
          CREATE TABLE head_coverage AS SELECT DISTINCT z.spoken,z.canonical,r.language,s.baseline,s.policy,s.quality
          FROM resolutions z JOIN records r ON r.id=z.record_id JOIN sources s ON s.id=r.source_id WHERE r.has_definition=1;
          CREATE INDEX IF NOT EXISTS resolved_identity ON resolutions(spoken,canonical,nr);
          CREATE INDEX coverage_lookup ON head_coverage(spoken,canonical,language,baseline);''')
        self.database.commit()
        out = self.root / label
        before = self.path.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            summary = build_queues(self.path, out, limit)
        self.assertEqual(before, self.path.read_bytes(), 'Review command must not change the inventory')
        def samples(kind):
            with gzip.open(out / (kind + '.jsonl.gz'), 'rt') as stream:
                return [json.loads(line) for line in stream]
        return summary, samples, out

    def test_unresolved_candidates_exclude_ai_nd_and_existing_english(self):
        self.source('fr', 'fr')
        self.source('nd', 'zh', policy='discovery_only_nd')
        self.source('ai', 'de', quality='machine_translated', policy='machine_evidence_only')
        self.source('cedict', 'en')
        self.record('fr', '缺词', '')
        self.record('nd', '禁词', '')
        self.record('ai', '梦词', '')
        self.record('fr', '已有', '')
        self.record('cedict', '已有', 'yi3 you3')
        self.record('fr', '缺词', '', spoken='yue', row_index=1)
        summary, sample, _ = self.run_queues()
        self.assertEqual(summary['queues']['unresolved_english_reading']['countsBySpokenLanguage'], {'cmn': 1, 'yue': 1})
        self.assertTrue(all(not row['generationEligible'] for row in sample('unresolved_english_reading')))
        self.assertEqual({ref['sourceId'] for row in sample('unresolved_english_reading') for ref in row['sourceRefs']}, {'fr'})

    def test_additional_readings_are_unverified_leads_not_missing_heads(self):
        self.source('cedict', 'en')
        self.source('fr', 'fr')
        self.record('cedict', '行', 'xing2')
        self.record('fr', '行', 'hang2')
        self.record('fr', '行', 'hang2', row_index=1)
        self.record('fr', '行', 'xing2', row_index=2)
        summary, sample, _ = self.run_queues()
        self.assertEqual(summary['queues']['additional_reading_lead']['totalHeads'], 1)
        record = sample('additional_reading_lead')[0]
        self.assertTrue(record['notMissingEnglishHeadword'])
        self.assertFalse(record['trueMissingMeaningConfirmed'])
        self.assertIn('UNVERIFIED', record['warning'])
        self.assertEqual({r['normalizedReading'] for r in record['sourceRefs']}, {normalized_reading('hang2')})

    def test_machine_only_and_available_optional_are_quality_not_missing(self):
        self.source('cedict', 'en')
        self.source('handedict-en-ai', 'en', quality='machine_translated', policy='machine_evidence_only')
        self.source('wiktionary', 'en')
        self.record('handedict-en-ai', '机器词')
        self.record('handedict-en-ai', '人工也有')
        self.record('wiktionary', '人工也有')
        self.record('wiktionary', '可启用')
        self.record('wiktionary', '已默认')
        self.record('cedict', '已默认')
        summary, sample, _ = self.run_queues()
        self.assertEqual(summary['queues']['machine_only_english']['totalHeads'], 1)
        self.assertEqual(summary['queues']['available_optional_english']['totalHeads'], 3)
        self.assertEqual(sample('machine_only_english')[0]['canonicalWord'], '机器词')

    def test_supplements_are_new_only_when_all_baseline_languages_lack_head(self):
        self.source('raw-wiktionary-fr', 'fr', baseline=False)
        self.source('hsk-open', 'en', baseline=False)
        self.source('de', 'de')
        self.record('raw-wiktionary-fr', '新增')
        self.record('hsk-open', '新增')
        self.record('raw-wiktionary-fr', '已有外文')
        self.record('de', '已有外文')
        summary, sample, _ = self.run_queues()
        self.assertEqual(summary['queues']['supplemental_new_head']['totalHeads'], 1)
        self.assertEqual(sample('supplemental_new_head')[0]['canonicalWord'], '新增')
        self.assertEqual(sample('supplemental_new_head')[0]['sourceCount'], 2)

    def test_external_upstream_snapshot_is_in_supplemental_discovery_queue(self):
        self.source('wiktionary-en-upstream', 'en', baseline=False)
        self.record('wiktionary-en-upstream', '新补词')
        summary, sample, _ = self.run_queues()
        self.assertEqual(summary['queues']['supplemental_new_head']['totalHeads'], 1)
        self.assertEqual(sample('supplemental_new_head')[0]['sourceRefs'][0]['sourceId'], 'wiktionary-en-upstream')

    def test_sense_leads_require_explicit_scoped_anchor_and_shared_family_is_one(self):
        self.source('cedict', 'en')
        self.source('zh-wiktionary-en', 'en', family='wiktionary-family')
        self.source('zh-wiktionary-fr', 'fr', family='wiktionary-family')
        self.source('fr', 'fr')
        self.record('cedict', '意义词')
        self.record('cedict', '无锚点')
        self.record('fr', '无锚点')
        self.record('zh-wiktionary-en', '已映射', sense='sense-a')
        self.record('zh-wiktionary-fr', '已映射', sense='sense-a')
        for language in ['fr', 'de', 'es']:
            self.source('wordnet-' + language, language, quality='translation_only', policy='component_review', family='wordnet-cow')
            self.record('wordnet-' + language, '意义词', sense='01234567-n')
        summary, sample, _ = self.run_queues()
        self.assertEqual(summary['queues']['additional_sense_lead']['totalHeads'], 1)
        record = sample('additional_sense_lead')[0]
        self.assertEqual(record['sourceCount'], 3)
        self.assertEqual(record['sourceFamilyCount'], 1)
        self.assertIn('DOES NOT', record['warning'])
        self.assertEqual({r['senseAnchorScope'] for r in record['sourceRefs']}, {'pwn-3.0'})

    def test_counts_are_exact_when_sample_is_limited_and_exports_are_deterministic(self):
        self.source('fr', 'fr')
        for word in ['甲词', '乙词', '丙词']:
            self.record('fr', word, '')
        first, sample, first_out = self.run_queues(limit=1)
        self.assertEqual(first['queues']['unresolved_english_reading']['totalHeads'], 3)
        self.assertEqual(len(sample('unresolved_english_reading')), 1)
        second, _, second_out = self.run_queues(limit=1, label='second')
        self.assertEqual(first, second)
        for path in first_out.iterdir():
            self.assertEqual(path.read_bytes(), (second_out / path.name).read_bytes())

    def test_zero_sample_limit_produces_counts_only_and_no_secrets(self):
        (self.root / '.minimax-key').write_text('DO_NOT_READ_REAL_OR_SYNTHETIC_KEYS')
        self.source('fr', 'fr')
        self.record('fr', '缺词', '')
        summary, sample, out = self.run_queues(limit=0)
        self.assertEqual(summary['queues']['unresolved_english_reading']['totalHeads'], 1)
        self.assertEqual(sample('unresolved_english_reading'), [])
        self.assertNotIn('DO_NOT_EXPORT_SOURCE_SECRET', (out / 'summary.json').read_text())
        self.assertNotIn('DO_NOT_READ_REAL_OR_SYNTHETIC_KEYS', (out / 'summary.json').read_text())
        _, records, populated = self.run_queues(limit=1, label='populated')
        self.assertEqual(len(records('unresolved_english_reading')), 1)
        for path in populated.iterdir():
            content = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
            self.assertNotIn(b'DO_NOT_EXPORT_SOURCE_SECRET', content)
            self.assertNotIn(b'DO_NOT_READ_REAL_OR_SYNTHETIC_KEYS', content)
        with self.assertRaises(ValueError):
            build_queues(self.path, out, -1)


if __name__ == '__main__':
    unittest.main()
