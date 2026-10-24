"""Source-only quality upgrades never convert existing AI into source evidence."""
import contextlib
import gzip
import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from crowdict.inventory import build
from crowdict.source_upgrades import build_source_upgrades,KIND
from crowdict.validate import validate_draft
from test_inventory import InventoryTests as _InventoryFixture


class SourceUpgradeTests(unittest.TestCase):
    add_pack = _InventoryFixture.add_pack

    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.data=self.root/'public/data';self.data.mkdir(parents=True)
        self.catalog=[];self.database=self.root/'inventory.sqlite';self.reports=self.root/'reports'
        self.out=self.root/'source-upgrades';self.upstream=self.root/'upstream';(self.upstream/'public/data').mkdir(parents=True)
        self.source=self.upstream/'public/data/entries.jsonl.gz'

    def raw_row(self,word='电子产品',traditional='電子產品',reading='dian4 zi3 chan3 pin3',gloss='electronic product',pos='noun',tags=None):
        return [traditional,word,reading,gloss,{'source':'wiktionary-en-upstream','pos':pos,
            'rawOriginalRowIndex':7,'rawReadingFields':[{'field':'sounds[0].zh_pron','value':reading,'tags':['Mandarin','Pinyin']}],
            'rawSourceSenses':[{'glosses':[gloss],**({'tags':tags} if tags else {})}],
            'sourceArticleUrl':'https://example.org/dictionary/electronics'}]

    def prepare(self,rows=None,ai_reading='dian4 zi3 chan3 pin3',human_rows=None,extra_attributes=None):
        self.add_pack('handedict-en-ai','English',[
            ['電子產品','电子产品',ai_reading,'NEVER_SOURCE_AI_DEFINITION']],machineTranslated=True)
        if human_rows:self.add_pack('cedict','English',human_rows)
        rows=rows if rows is not None else [self.raw_row()]
        self.source.write_bytes(gzip.compress(('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n').encode(),mtime=0))
        source={'id':'wiktionary-en-upstream','name':'Fixture English source','language':'Chinese → English','lang':'en','spoken':'cmn',
            'license':'CC BY-SA 4.0','licenseUrl':'https://creativecommons.org/licenses/by-sa/4.0/',
            'attribution':'Fixture source contributors','url':'https://example.org/source','snapshot':'fixture',
            'dataFormat':'entry-jsonl','file':'/data/entries.jsonl.gz','original':'/data/entries.jsonl.gz',
            'sha256':hashlib.sha256(self.source.read_bytes()).hexdigest(),'count':len(rows)}
        source.update(extra_attributes or {})
        catalog=self.upstream/'extra-catalog.json';catalog.write_text(json.dumps({'root':'.','sources':[source]}))
        (self.data/'catalog.json').write_text(json.dumps(self.catalog))
        with contextlib.redirect_stdout(io.StringIO()):
            build(self.root,self.database,self.reports,raw_readings=False,extra_catalog=catalog)

    def run_upgrade(self):
        with contextlib.redirect_stdout(io.StringIO()):
            summary=build_source_upgrades(self.database,self.out,web_root=self.root,report=self.reports/'source-upgrades.json')
        def read(name):
            with gzip.open(self.out/(name+'-en.jsonl.gz'),'rt') as stream:return [json.loads(line) for line in stream]
        return summary,read('jobs'),read('results'),read('held'),read('recovery')

    def test_copies_all_source_labels_and_scripts_never_ai_evidence_or_paid_eligibility(self):
        self.prepare(rows=[self.raw_row(gloss='electronic product',tags=['technical']),
                           self.raw_row(gloss='electronic goods',pos='plural-noun')])
        before=self.database.read_bytes();summary,jobs,results,held,_=self.run_upgrade()
        self.assertEqual(summary['sourceOnlyQualityUpgradeJobs'],1);self.assertEqual(summary['newHeadwords'],0)
        self.assertEqual(summary['generationApiCalls'],0);self.assertEqual(held,[])
        job=jobs[0];self.assertEqual(job['gapKind'],KIND);self.assertFalse(job['generationEligible']);self.assertTrue(job['reuseEligible'])
        self.assertEqual((job['traditionalWord'],job['simplifiedWord']),('電子產品','电子产品'))
        self.assertEqual(len(job['evidence']),2)
        self.assertTrue(all(e['sourceId']=='wiktionary-en-upstream' for e in job['evidence']))
        self.assertNotIn('NEVER_SOURCE_AI_DEFINITION',json.dumps(job))
        senses=results[0]['draft']['senses'];self.assertEqual({s['definition'] for s in senses},{'electronic product','electronic goods'})
        self.assertEqual(next(s for s in senses if s['definition']=='electronic product')['register'],'technical')
        self.assertEqual(validate_draft(job,results[0])[0],[])
        self.assertEqual(job['readingProvenance'][0]['sourceSha256'],job['evidence'][0]['sourceSha256'])
        self.assertEqual(self.database.read_bytes(),before)
        before_outputs={p.name:p.read_bytes() for p in self.out.glob('*.gz')};self.run_upgrade()
        self.assertEqual({p.name:p.read_bytes() for p in self.out.glob('*.gz')},before_outputs)

    def test_combined_multi_pos_sense_overflow_holds_entire_identity(self):
        rows=[]
        for index in range(2):
            row=self.raw_row(gloss='whole source row',pos='noun' if index==0 else 'verb')
            row[4]['rawSourceSenses']=[{'glosses':['sense '+str(n)]} for n in range(index*5,index*5+5)]
            rows.append(row)
        self.prepare(rows=rows);summary,jobs,results,held,_=self.run_upgrade()
        self.assertEqual(jobs,[]);self.assertEqual(results,[]);self.assertEqual(summary['heldJobs'],1)
        self.assertIn('source_sense_overflow',held[0]['holdReasons'])

    def test_present_human_baseline_is_not_a_machine_only_quality_gap(self):
        self.prepare(human_rows=[['電子產品','电子产品','dian4 zi3 chan3 pin3','edited human English']])
        summary,jobs,_,_,_=self.run_upgrade();self.assertEqual(jobs,[]);self.assertEqual(summary['baselineMachineOnlyEnglishHeads'],0)

    def test_different_ai_reading_is_not_used_for_paid_or_source_import(self):
        self.prepare(ai_reading='dian3 zi3 chan3 pin3');summary,jobs,results,_,recovery=self.run_upgrade()
        self.assertEqual(jobs,[]);self.assertEqual(results,[])
        self.assertTrue(any(r['reason']=='same_reading_baseline_ai_not_proven' for r in recovery))

    def test_ai_reading_cannot_supply_missing_source_pronunciation(self):
        self.prepare(rows=[self.raw_row(reading='')]);summary,jobs,results,_,recovery=self.run_upgrade()
        self.assertEqual(jobs,[]);self.assertEqual(results,[])
        self.assertTrue(any(r['reason']=='missing_invalid_or_conflicting_source_reading' for r in recovery))

    def test_resolved_only_non_ai_reading_is_held_until_archival_proof_support(self):
        self.add_pack('handedict','German',[['電子產品','电子产品','dian4 zi3 chan3 pin3','elektronisches Produkt']])
        self.prepare(rows=[self.raw_row(reading='')]);summary,jobs,results,held,_=self.run_upgrade()
        self.assertEqual(jobs,[]);self.assertEqual(results,[])
        self.assertIn('explicit_english_source_reading_required',held[0]['holdReasons'])

    def test_same_lexeme_proper_name_sense_holds_whole_group_without_cherry_picking(self):
        self.prepare(rows=[self.raw_row(),self.raw_row(gloss='a proper product name',pos='proper name')])
        _,jobs,results,held,_=self.run_upgrade();self.assertEqual(jobs,[]);self.assertEqual(results,[])
        self.assertIn('proper_name_variant_or_abbreviation_needs_review',held[0]['holdReasons'])

    def test_source_snapshot_tampering_aborts_without_output(self):
        self.prepare();self.source.write_bytes(self.source.read_bytes()+b'tampered')
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):
            self.run_upgrade()
        self.assertFalse(self.out.exists())

    def test_nd_or_machine_supplements_never_supply_evidence(self):
        for attributes in [{'license':'CC BY-ND 3.0 TW'},{'machineTranslated':True}]:
            with self.subTest(attributes=attributes):
                # Separate fixture roots because build intentionally rejects an existing DB.
                other=SourceUpgradeTests();other.setUp();self.addCleanup(other.doCleanups)
                other.prepare(extra_attributes=attributes)
                summary,jobs,results,_,_=other.run_upgrade();self.assertEqual(jobs,[]);self.assertEqual(results,[])


del _InventoryFixture
