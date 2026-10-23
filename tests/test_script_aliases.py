"""Only explicit plain source script equivalents may affect lexical identity."""
import contextlib
import gzip
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote

from crowdict.distill import distill
from crowdict.raw_sources import _script_forms, load_raw_source_rows
from test_inventory import InventoryTests as _InventoryFixture


class ScriptFormsTests(unittest.TestCase):
    def test_unique_simplified_form_has_cedict_direction(self):
        record = {'forms': [{'form': '电子产品', 'tags': ['Simplified-Chinese']}]}
        display, canonical, meta = _script_forms(record, '電子產品')
        self.assertEqual((display, canonical), ('電子產品', '电子产品'))
        self.assertEqual(meta['canonicalWord'], '电子产品')
        self.assertEqual(meta['lookupAliases'], ['電子產品'])
        self.assertEqual(meta['rawScriptFormFields'][0]['field'], 'forms[0].form')

    def test_unique_traditional_form_keeps_source_simplified_canonical(self):
        display, canonical, meta = _script_forms({'forms': [
            {'form': '電子產品', 'tags': ['Traditional-Chinese']}]}, '电子产品')
        self.assertEqual((display, canonical), ('電子產品', '电子产品'))
        self.assertEqual(meta['lookupAliases'], ['電子產品'])

    def test_alternative_qualified_second_round_and_non_han_forms_are_excluded(self):
        forms = [{'form': '码', 'tags': ['alternative', 'Simplified-Chinese']},
                 {'form': '满', 'tags': ['Simplified-Chinese'], 'raw_tags': ['alternative']},
                 {'form': '卤', 'tags': ['Second-Round-Simplified-Chinese']},
                 {'form': '马 a', 'tags': ['Simplified-Chinese']},
                 {'form': '马 ', 'tags': ['Simplified-Chinese']},
                 {'form': '马', 'tags': []}, {'form': '', 'tags': ['Simplified-Chinese']}]
        self.assertEqual(_script_forms({'forms': forms}, '馬'), ('馬', '馬', {}))

    def test_multiple_distinct_simplified_forms_never_choose_a_canonical(self):
        display, canonical, meta = _script_forms({'forms': [
            {'form': '干', 'tags': ['Simplified-Chinese']},
            {'form': '乾', 'tags': ['Simplified-Chinese']}]}, '乾')
        self.assertEqual((display, canonical), ('乾', '乾'))
        self.assertEqual(meta['rawScriptFormStatus'], 'ambiguous_simplified_forms')
        self.assertNotIn('canonicalWord', meta)
        self.assertNotIn('lookupAliases', meta)

    def test_original_article_and_ambiguous_reading_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / 'source.jsonl.gz'
            record = {'word': '電子產品', 'lang_code': 'zh', 'pos': 'noun',
                      'forms': [{'form': '电子产品', 'tags': ['Simplified-Chinese']}],
                      'sounds': [{'zh_pron': 'diànzǐchǎnpǐn, diǎnzǐchǎnpǐn', 'tags': ['Mandarin', 'Pinyin']}],
                      'senses': [{'id': 'en-fixture', 'glosses': ['electronic product']}]}
            original.write_bytes(gzip.compress((json.dumps(record, ensure_ascii=False)+'\n').encode(), mtime=0))
            with contextlib.redirect_stdout(io.StringIO()):
                catalog = distill(original, root, snapshot_date='fixture', dump_date='fixture')
            with gzip.open(root / 'public/data/english-upstream-entries.jsonl.gz', 'rt') as stream:
                row = json.loads(next(stream))
            self.assertEqual(row[:3], ['電子產品', '电子产品', ''])
            self.assertEqual(row[4]['sourceWord'], '電子產品')
            self.assertEqual(row[4]['sourceArticleUrl'], 'https://en.wiktionary.org/wiki/'+quote('電子產品', safe=''))
            self.assertEqual(len(row[4]['readingCandidates']), 2)
            self.assertEqual(row[4]['rawOriginalRowIndex'], 0)
            self.assertEqual(catalog['root'], '.')


class ScriptAliasInventoryTests(unittest.TestCase):
    # Reuse fixture methods without inheriting its unrelated regression tests.
    setUp = _InventoryFixture.setUp
    add_pack = _InventoryFixture.add_pack
    build_inventory = _InventoryFixture.build_inventory
    resolutions = _InventoryFixture.resolutions
    make_jobs = _InventoryFixture.make_jobs
    def test_cedict_and_source_attested_alias_do_not_cycle_or_create_false_gap(self):
        self.add_pack('cedict', 'English', [['電子產品', '电子产品', 'dian4 zi3 chan3 pin3', 'electronic product']])
        self.add_pack('cfdict', 'French', [['電子産品', '电子产品', 'dian4 zi3 chan3 pin3', 'produit électronique']])
        meta = {'sourceWord': '電子產品', 'canonicalWord': '电子产品', 'lookupAliases': ['電子產品']}
        self.add_pack('wiktionary-fixture', 'English', [['電子產品', '电子产品', 'diànzǐchǎnpǐn', 'electronic product', meta]], baseline=False)
        self.build_inventory()
        resolution = self.resolutions('wiktionary-fixture')[0]
        self.assertEqual(resolution['canonical'], '电子产品')
        self.assertNotEqual(resolution['alias_status'], 'conflict')
        self.assertIsNotNone(resolution['lexeme_id'])
        self.assertEqual(self.make_jobs()[0]['generationEligible'], 0)


del _InventoryFixture
