"""Stream a complete Kaikki Chinese/English snapshot into reusable English rows.

This is a source-reuse importer, not a definition generator. General Chinese/Han
records need explicit standard Mandarin Pinyin attribution. Other Chinese
varieties' senses, quotations, examples and media are never copied to this pack.
No provider, key, paid API, inferred pronunciation or inferred script alias is
used. Original and compact checksums are retained in an external source catalog.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

from .common import canonical_json, write_jsonl
from .raw_readings import _is_pinyin_field, _tokens
from .raw_sources import _hash, iter_raw_source_rows

_SOURCE_ID = 'wiktionary-en-upstream'
_ORIGINAL_NAME = 'kaikki.org-dictionary-Chinese.jsonl.gz'
_COMPACT_NAME = 'english-upstream-entries.jsonl.gz'
_DEFAULT_URL = 'https://kaikki.org/dictionary/Chinese/kaikki.org-dictionary-Chinese.jsonl.gz'
_OTHER_VARIETIES = {
    'cantonese', 'yue', 'hakka', 'hak', 'wu', 'wuu', 'gan', 'jin', 'xiang', 'hsn',
    'min', 'min-nan', 'nan', 'hokkien', 'teochew', 'chaozhou', 'min-dong', 'cdo',
    'min-bei', 'mnp', 'min-zhong', 'czo', 'pu-xian', 'cpx', 'southern-min',
    'eastern-min', 'western-min', 'northern-min', 'hainanese', 'shanghainese',
    'taishanese', 'sichuanese', 'sichuan', 'wuhan', 'wuhu', 'chengdu', 'nanjing',
    'dungan', 'pinghua', 'huizhou', 'waxiang', 'sixian', 'hailu', 'meixian',
    'old-chinese', 'middle-chinese', 'medieval-chinese',
}
_NONSTANDARD_PRONUNCIATION = {'nonstandard', 'obsolete', 'historical', 'dialectal'}


def explicit_standard_pinyin(record: dict, value: dict) -> bool:
    """Require language attribution AND Pinyin; generic Chinese is insufficient."""
    tags = _tokens(value.get('tags', []))
    raw_tags = _tokens(value.get('raw_tags', []))
    if tags & _NONSTANDARD_PRONUNCIATION or raw_tags & _NONSTANDARD_PRONUNCIATION:
        return False
    pinyin = 'pinyin' in tags or 'pinyin' in raw_tags
    mandarin = 'mandarin' in tags or record.get('lang_code') == 'cmn'
    return pinyin and mandarin and _is_pinyin_field(record, value)


def distill(input_path: Path, root: Path, *, snapshot_date: str, dump_date: str,
            source_url: str = _DEFAULT_URL) -> dict:
    """Create compact entry JSONL, an external catalog and auditable metrics."""
    input_path = Path(input_path).resolve()
    root = Path(root).resolve()
    data = root / 'public/data'
    data.mkdir(parents=True, exist_ok=True)
    original = data / _ORIGINAL_NAME
    if input_path != original.resolve():
        if original.exists():
            if _hash(original) != _hash(input_path):
                raise ValueError('Different original already exists at external source path')
        else:
            # The official download and its public/data reference share storage.
            os.link(input_path, original)
    original_hash = _hash(original)
    source = {
        'id': _SOURCE_ID, 'originalSourceId': _SOURCE_ID,
        'name': 'English Wiktionary · full source Mandarin subset',
        'language': 'Chinese → English', 'lang': 'en', 'spoken': 'cmn',
        'license': 'CC BY-SA 4.0', 'dataFormat': 'wiktionary-jsonl',
        'file': '/data/' + _ORIGINAL_NAME, 'original': '/data/' + _ORIGINAL_NAME,
    }
    stats = collections.Counter()
    variety_tags = collections.Counter()
    words = set()
    source_words = set()
    readable_words = set()
    ambiguous_words = set()
    sense_id_prefixes = collections.Counter()

    def record_filter(record):
        for group in ('sounds', 'forms'):
            values = record.get(group, [])
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, dict) and explicit_standard_pinyin(record, value):
                    stats['recordsWithExplicitStandardMandarinPinyin'] += 1
                    return True
        stats['recordsWithoutExplicitStandardMandarinPinyin'] += 1
        return False

    def sense_filter(sense):
        tags = _tokens(sense.get('tags', [])) | _tokens(sense.get('raw_tags', []))
        if any(tag.startswith('error-') for tag in tags):
            stats['sensesExcludedExtractorError'] += 1
            return False
        if tags & {'dialectal', 'regional', 'nonstandard'}:
            stats['sensesExcludedUncertainStandardVariety'] += 1
            variety_tags.update(tags & {'dialectal', 'regional', 'nonstandard'})
            return False
        other = tags & _OTHER_VARIETIES
        # Tags frequently combine locality and variety: Taiwanese-Hokkien,
        # Zhangzhou-Hokkien, Guangzhou-Cantonese, etc. Check components too.
        families = {'cantonese', 'hakka', 'hokkien', 'min', 'wu', 'gan', 'jin', 'xiang', 'teochew', 'chaozhou', 'hainanese', 'taishanese'}
        other.update(tag for tag in tags if set(re.split(r'[-_ ]+', tag)) & families)
        # This pack promises a strict standard subset. Regional Mandarin senses
        # remain available in the original for a separately labelled future pack.
        other.update(tag for tag in tags if tag.endswith('-mandarin') and tag != 'standard-mandarin')
        if other:
            stats['sensesExcludedOtherChineseVariety'] += 1
            variety_tags.update(other)
            return False
        identity = str(sense.get('id', ''))
        if identity:
            sense_id_prefixes[identity.split('-', 1)[0]] += 1
        stats['sensesAllowedBeforeGlossValidation'] += 1
        return True

    def entries():
        for row in iter_raw_source_rows(root, source, record_filter=record_filter,
                                        sense_filter=sense_filter,
                                        pronunciation_filter=explicit_standard_pinyin,
                                        stats=stats):
            words.add(row[1])
            source_words.add(row[4]['sourceWord'])
            if row[2]:
                readable_words.add(row[1])
            if row[4].get('readingCandidates'):
                ambiguous_words.add(row[1])
            row[4]['chineseLanguage'] = 'mandarin'
            row[4]['sourceArticleUrl'] = 'https://en.wiktionary.org/wiki/' + quote(row[4]['sourceWord'], safe='')
            row[4]['sourceReuseStatus'] = 'source_imported_not_human_reviewed'
            yield row

    compact = data / _COMPACT_NAME
    write_jsonl(compact, entries())
    compact_hash = _hash(compact)
    descriptor = {
        'id': _SOURCE_ID,
        'name': 'English Wiktionary · full source Mandarin subset',
        'lang': 'en', 'spoken': 'cmn', 'language': 'Chinese → English',
        'baseline': False, 'dataFormat': 'entry-jsonl',
        'file': '/data/' + _COMPACT_NAME,
        'original': '/data/' + _ORIGINAL_NAME,
        'sha256': compact_hash, 'originalSha256': original_hash,
        'bytes': compact.stat().st_size, 'originalBytes': original.stat().st_size,
        'count': stats['emittedRows'], 'headwordCount': len(words),
        'license': 'CC BY-SA 4.0',
        'licenseUrl': 'https://creativecommons.org/licenses/by-sa/4.0/',
        'attribution': 'English Wiktionary contributors; extracted and postprocessed by Kaikki / Wiktextract (Tatu Ylonen and contributors).',
        'sourceUrl': source_url, 'url': 'https://kaikki.org/dictionary/Chinese/',
        'snapshot': snapshot_date + ' · enwiktionary dump ' + dump_date,
        'originalSnapshotDate': snapshot_date, 'wiktionaryDumpDate': dump_date,
        'description': 'Source-imported English definitions from the full Chinese snapshot, limited to explicitly attested standard Mandarin Pinyin and senses not explicitly assigned to another Chinese variety. Ambiguous readings remain unresolved. Only plain source-attested simplified/traditional forms are retained as reading-conditional aliases; no inferred script aliases, machine definitions, examples, quotations or media.',
        'licenseNotice': 'Wiktionary original text is available under CC BY-SA 4.0 / GFDL. This adaptation uses CC BY-SA 4.0. Attribution and source entry links are retained; see https://kaikki.org/dictionary/ and https://en.wiktionary.org/wiki/Wiktionary:Copyrights . Only source glosses, lexical tags, identifiers, pronunciations and provenance are retained; quotations/examples/media are omitted. Kaikki labels these postprocessed language downloads deprecated in favour of its full raw download format.',
        'machineTranslated': False, 'translationOnly': False,
    }
    catalog = {'root': '.', 'sources': [descriptor]}
    metrics = {
        'sourceId': _SOURCE_ID, 'sourceUrl': source_url,
        'snapshotDate': snapshot_date, 'wiktionaryDumpDate': dump_date,
        'originalSha256': original_hash, 'originalCompressedBytes': original.stat().st_size,
        'compactSha256': compact_hash, 'compactCompressedBytes': compact.stat().st_size,
        'counts': dict(sorted(stats.items())), 'distinctEmittedHeadwords': len(words),
        'distinctSourceHeadwordSpellings': len(source_words),
        'distinctHeadsWithAtLeastOneUniqueReadingRow': len(readable_words),
        'distinctHeadsWithAmbiguousReadingRows': len(ambiguous_words),
        'excludedSenseVarietyTags': dict(sorted(variety_tags.items())),
        'retainedSenseIdPrefixesBeforeGlossValidation': dict(sorted(sense_id_prefixes.items())),
        'generationApiCalls': 0, 'humanReviewed': False,
        'limitations': [
            'General Chinese/Han labeling alone is not treated as Mandarin; explicit Mandarin Pinyin attribution is required.',
            'Multiple source readings remain unresolved, and source errors or sense/pronunciation alignment errors can still exist.',
            'Source lexical tags and source sense IDs are retained; downstream display must preserve important usage restrictions.',
            'This is a source-imported subset, not proof of complete English or Chinese coverage.',
        ],
    }
    for filename, value in [('extra-catalog.json', catalog), ('distill-metrics.json', metrics)]:
        pending = root / (filename + '.pending')
        pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        pending.replace(root / filename)
    print(canonical_json(metrics), flush=True)
    return catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('work/upstream'))
    parser.add_argument('--snapshot-date', required=True)
    parser.add_argument('--dump-date', required=True)
    parser.add_argument('--source-url', default=_DEFAULT_URL)
    args = parser.parse_args(argv)
    distill(args.input, args.root, snapshot_date=args.snapshot_date,
            dump_date=args.dump_date, source_url=args.source_url)


if __name__ == '__main__':
    main()
