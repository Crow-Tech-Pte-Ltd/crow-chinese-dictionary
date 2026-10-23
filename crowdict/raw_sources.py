"""Broader source-backed discovery from already-retained localized originals.

No source data is downloaded, translated or guessed here. Returned packs are
non-baseline source evidence; they do not pretend Hanzi Desk already covers a
word. Original JSONL positions, pronunciation fields and sense IDs remain
available for audit and downstream validation.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Iterator

from .common import LANGUAGES, data_path, normalized_reading, valid_reading
from .raw_readings import (
    RawReadingSourceError, _CHINESE_NAMES, _attested_parts, _eligible,
    _is_pinyin_field,
)

_HEAD = re.compile(r"^[A-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af·・-]+$")
_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]")
_REQUEST = re.compile(
    r"(?:needs? (?:a |an )?definition|definition (?:needed|requested)|please add (?:a )?definition|"
    r"memerlukan definisi|cần một định nghĩa|traduction à compléter|définition manquante)", re.I
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def raw_source_catalog(web_root: Path, catalog: list[dict]) -> list[dict]:
    """Return optional, non-baseline descriptors for eligible original JSONL."""
    result = []
    hashes: dict[Path, str] = {}
    reverse_languages = {name: code for code, name in LANGUAGES.items()}
    for original in catalog:
        if not _eligible(original):
            continue
        source_file = original.get('original')
        if not isinstance(source_file, str) or not source_file.endswith('.jsonl.gz'):
            raise RawReadingSourceError(f"Unsupported original for {original.get('id')}")
        path = data_path(web_root, source_file)
        if path not in hashes:
            hashes[path] = _hash(path)
        language = original.get('language', '')
        target = language.split('→')[-1].strip()
        lang = original.get('lang') or reverse_languages.get(target)
        if not lang:
            raise RawReadingSourceError(f"Unknown target language for {original.get('id')}")
        source = dict(original)
        source.update(
            id='raw-' + original['id'],
            name=original.get('name', original['id']) + ' · retained raw source',
            baseline=False,
            dataFormat='wiktionary-jsonl',
            file=source_file,
            original=source_file,
            sha256=hashes[path],
            bytes=path.stat().st_size,
            snapshot='original-sha256:' + hashes[path],
            originalSourceId=original['id'],
            originalSnapshot=original.get('snapshot', ''),
            language=language,
            lang=lang,
            spoken=original.get('spoken') or ('yue' if original.get('sourceLanguage') == 'cantonese' else 'cmn'),
            description='Broader retained localized Wiktionary records; exact source spellings, source glosses and explicit Pinyin. Readings and source senses still need validation.',
        )
        # The original compact pack count is not the number of retained source
        # records. Source verification uses the compressed original SHA-256.
        source.pop('count', None)
        result.append(source)
    return result


def _chinese_record(record: dict) -> bool:
    language = record.get('lang_code')
    if language in {'zh', 'cmn'}:
        return True
    if language == 'Hani':
        if str(record.get('lang', '')).strip().casefold() in _CHINESE_NAMES:
            return True
        # Translingual Han characters require explicit Mandarin attribution.
        return any(isinstance(value, dict) and 'Mandarin' in value.get('tags', []) and 'Pinyin' in value.get('tags', []) for value in record.get('sounds', []))
    return False


def _glosses(record: dict, sense_filter=None) -> tuple[list[str], list[dict]]:
    glosses: list[str] = []
    source_senses: list[dict] = []
    for sense in record.get('senses', []):
        if not isinstance(sense, dict):
            continue
        if sense_filter is not None and not sense_filter(sense):
            continue
        tags = sense.get('tags', [])
        if 'no-gloss' in tags or 'form-of' in tags:
            continue
        kept = []
        for gloss in sense.get('glosses', []):
            if not isinstance(gloss, str):
                continue
            value = gloss.strip()
            if not value or value in {'?', '-', '—', '…', '...'} or '{{' in value or '}}' in value or _REQUEST.search(value):
                continue
            # Slash is the website's sense separator, so escape literal slashes.
            value = value.replace('/', '∕')
            if value not in glosses:
                glosses.append(value)
            if value not in kept:
                kept.append(value)
        if kept:
            item = {'glosses': kept}
            identity = sense.get('id') or sense.get('senseid') or sense.get('sense_id')
            if identity is not None:
                item['id'] = str(identity)
            if tags:
                item['tags'] = list(tags)
            # Wiktextract raw_tags can hold lexical scope restrictions that
            # have no normalized tag (e.g. "of plants or flowers"). Preserve
            # them verbatim so source reuse cannot silently widen the meaning.
            if sense.get('raw_tags'):
                item['raw_tags'] = sense['raw_tags']
            source_senses.append(item)
    return glosses, source_senses



def _script_forms(record: dict, word: str) -> tuple[str, str, dict]:
    """Retain only plain, explicitly labelled source script equivalents.

    Qualifiers such as alternative, historic and second-round simplified are
    lexical/orthographic leads, not a safe simplified/traditional identity edge.
    Never infer a script conversion or choose between distinct simplified forms.
    """
    pure_han = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]+$")
    if not pure_han.fullmatch(word):
        return word, word, {}
    fields = []
    forms = record.get('forms', [])
    if not isinstance(forms, list):
        return word, word, {}
    by_script = {'Simplified-Chinese': [], 'Traditional-Chinese': []}
    for index, value in enumerate(forms):
        if not isinstance(value, dict) or value.get('raw_tags'):
            continue
        tags = value.get('tags', [])
        if not isinstance(tags, list) or len(tags) != 1 or tags[0] not in by_script:
            continue
        form = value.get('form')
        if not isinstance(form, str) or not pure_han.fullmatch(form):
            continue
        fields.append({'field': f'forms[{index}].form', 'value': form, 'tags': tags[:]})
        if form not in by_script[tags[0]]:
            by_script[tags[0]].append(form)
    if not fields:
        return word, word, {}
    meta = {'rawScriptFormFields': fields}
    simplified = by_script['Simplified-Chinese']
    traditional = by_script['Traditional-Chinese']
    if len(simplified) > 1:
        meta['rawScriptFormStatus'] = 'ambiguous_simplified_forms'
        return word, word, meta
    canonical = simplified[0] if simplified else word
    display = word if word != canonical else traditional[0] if len(traditional) == 1 else word
    aliases = list(dict.fromkeys([word, *simplified, *traditional]))
    aliases = [alias for alias in aliases if alias != canonical]
    meta['rawScriptFormStatus'] = 'source_attested_plain_script_forms'
    meta['canonicalWord'] = canonical
    if aliases:
        meta['lookupAliases'] = aliases
    return display, canonical, meta

def iter_raw_source_rows(web_root: Path, source: dict, *, record_filter=None, sense_filter=None, pronunciation_filter=None, stats=None) -> Iterator[list]:
    """Stream website-shaped rows, keeping ambiguous source readings blank.

    Optional filters let a separately audited importer constrain spoken varieties
    without changing the localized-source default. Stats contain only counters.
    """
    if source.get('dataFormat') != 'wiktionary-jsonl':
        raise RawReadingSourceError('Not a retained Wiktionary JSONL descriptor')
    original_id = source.get('originalSourceId') or source.get('id', '').removeprefix('raw-')
    original = dict(source, id=original_id)
    if not _eligible(original):
        raise RawReadingSourceError('Raw source is not eligible for adapted definitions')
    path = data_path(web_root, source['file'])
    if stats is None:
        stats = {}
    def count(key):
        stats[key] = stats.get(key, 0) + 1
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for row_index, line in enumerate(stream):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RawReadingSourceError(f"Malformed JSON in {source['file']} at record {row_index}") from error
            if not isinstance(record, dict):
                raise RawReadingSourceError(f"Non-object record in {source['file']} at record {row_index}")
            count('inputRecords')
            word = record.get('word')
            if (not isinstance(word, str) or not word or len(word) > 128 or
                    not _HAN.search(word) or not _HEAD.fullmatch(word) or
                    not _chinese_record(record) or record.get('pos') in {'romanization', 'SOFT REDIRECT'}):
                count('nonLexicalOrNonChineseRecords')
                continue
            count('chineseLexicalRecords')
            if record_filter is not None and not record_filter(record):
                count('recordFilterExcluded')
                continue
            glosses, senses = _glosses(record, sense_filter)
            if not glosses:
                count('noUsableGlossRecords')
                continue
            candidates: dict[str, str] = {}
            fields = []
            for group, field in (('sounds', 'zh_pron'), ('forms', 'form')):
                values = record.get(group, [])
                if not isinstance(values, list):
                    continue
                for value_index, value in enumerate(values):
                    if not isinstance(value, dict) or not _is_pinyin_field(record, value):
                        continue
                    if pronunciation_filter is not None and not pronunciation_filter(record, value):
                        continue
                    raw = value.get(field)
                    if not isinstance(raw, str):
                        continue
                    attested = []
                    for reading in _attested_parts(raw):
                        if valid_reading(reading, 'cmn'):
                            attested.append(reading)
                            candidates.setdefault(normalized_reading(reading), reading)
                    fields.append({'field': f'{group}[{value_index}].{field}', 'value': raw,
                                   'candidates': attested, 'tags': list(value.get('tags', [])),
                                   'raw_tags': list(value.get('raw_tags', []))})
            reading = next(iter(candidates.values())) if len(candidates) == 1 else ''
            meta = {'source': source['id'], 'sourceWord': word, 'pos': record.get('pos', ''),
                    'rawOriginalRowIndex': row_index, 'rawReadingFields': fields,
                    'rawSourceSenses': senses}
            if len(candidates) != 1:
                meta['readingCandidates'] = list(candidates.values())
            sense_ids = [sense['id'] for sense in senses if 'id' in sense]
            if sense_ids:
                meta['senseIds'] = sense_ids
                if len(sense_ids) == 1:
                    meta['sourceSense'] = sense_ids[0]
            display, canonical, script_meta = _script_forms(record, word)
            meta.update(script_meta)
            if script_meta:
                count('rowsWithPlainScriptForms')
            if canonical != word:
                count('rowsCanonicalizedToAttestedSimplified')
            if meta.get('lookupAliases'):
                count('rowsWithSourceAttestedScriptAliases')
            if meta.get('rawScriptFormStatus') == 'ambiguous_simplified_forms':
                count('rowsWithAmbiguousSimplifiedForms')
            count('emittedRows')
            count('uniqueReadingRows' if len(candidates) == 1 else 'ambiguousReadingRows' if candidates else 'missingReadingRows')
            yield [display, canonical, reading, '/'.join(glosses), meta]


def load_raw_source_rows(web_root: Path, source: dict) -> list[list]:
    """Compatibility list wrapper; streaming callers should use the iterator."""
    return list(iter_raw_source_rows(web_root, source))
