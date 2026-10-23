"""Recover attested Mandarin Pinyin from already-downloaded Wiktionary originals.

This module performs no transliteration, syllable prediction, alias generation or
sense assignment. Every yielded string is an exact substring of an explicit
source pronunciation field. Downstream validation must normalize and validate
all candidates, retain conflicts and verify their lexical/sense applicability.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Iterator

# Romanizations and Chinese varieties must never silently become Mandarin Pinyin.
_EXCLUDED = {
    "cantonese", "jyutping", "yale", "tongyong-pinyin", "wade-giles",
    "gwoyeu-romatsyh", "palladius", "bopomofo", "ipa", "sinological-ipa",
    "efeo", "hakka", "wu", "gan", "jin", "min-nan", "min-bei", "min-dong",
    "min-zhong", "pu-xian", "hokkien", "teochew", "chaozhou", "old-chinese",
    "middle-chinese", "medieval-chinese", "sichuanese", "chengdu", "nanjing",
    "beijing", "taiwanese", "hakka-pinyin", "hakka-romanization-system",
    "cantonese-pinyin", "shanghainese", "phak-fa-su", "peh-oe-ji", "tai-lo",
    "xiang", "hui", "huizhou", "min", "hainanese", "puxian", "northern-min",
    "southern-min", "eastern-min", "western-min", "tongyong pinyin",
}
_EXCLUDED_RAW = {
    "通用拼音", "国語ローマ字", "國語羅馬字", "ウェード式", "粤語", "粵語", "広東語",
    "广东话", "廣東話", "閩南語", "闽南语", "閩東語", "客家語", "客家话", "客家話",
    "呉語", "吳語", "吴语", "潮州拼音", "台湾語", "台灣話",
}
# This is only a safe field-shape filter. Actual syllable/tone validation belongs
# downstream; a matching string is never automatically an accepted reading.
_PINYIN_SHAPE = re.compile(r"^[A-Za-züÜvV:āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜńňǹḿêÊ1-5 '\u2019\u00b7-]+$")
_TONE = re.compile(r"[1-5āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜńňǹḿ]", re.I)
_CHINESE_NAMES = {
    "chinese", "mandarin", "standard chinese", "chinois", "chinesisch", "cinese",
    "čínština", "język chiński standardowy", "chinês", "chino", "中国語", "中國語",
    "汉语", "漢語", "中文", "官話", "官话", "國語", "国语", "普通话", "普通話",
}


class RawReadingSourceError(ValueError):
    """An original source is malformed; its caller should report it explicitly."""


def _tokens(values: object) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(value).strip(" :：").casefold() for value in values}


def _is_pinyin_field(record: dict, value: dict) -> bool:
    tags = _tokens(value.get("tags", []))
    raw_tags = _tokens(value.get("raw_tags", []))
    if tags & _EXCLUDED or raw_tags & _EXCLUDED or raw_tags & {s.casefold() for s in _EXCLUDED_RAW}:
        return False
    declared = record.get("lang_code") in {"zh", "cmn"}
    if record.get("lang_code") == "Hani":
        declared = str(record.get("lang", "")).strip().casefold() in _CHINESE_NAMES or "mandarin" in tags
    if not declared:
        return False
    explicit_pinyin = "pinyin" in tags or "pinyin" in raw_tags
    japanese_pinyin = "官話".casefold() in raw_tags and "拼音".casefold() in raw_tags
    return explicit_pinyin or japanese_pinyin or "mandarin" in tags


def _attested_parts(raw: str) -> Iterator[str]:
    """Split explicit pronunciation alternatives without inferring any reading.

    An annotation such as ``wā (wa1), jué (jue2)`` yields both source spellings
    for each pronunciation. The caller must normalize equivalent spellings and
    preserve distinct readings. Non-pronunciation annotations are omitted only
    when they contain no tone evidence and thus cannot be a numbered/tone-marked
    Pinyin pronunciation. Unseparable complex patterns stay out of this parser.
    """
    for part in re.split(r"[,，;；/、]|\bor\b", raw):
        part = part.strip()
        if not part:
            continue
        parentheses = re.findall(r"\(([^()]*)\)", part)
        body = re.sub(r"\([^()]*\)", "", part).strip()
        values = [body, *parentheses]
        for index, value in enumerate(values):
            value = value.strip()
            if not value or not _PINYIN_SHAPE.fullmatch(value):
                continue
            # A parenthetical English usage label is not a pronunciation. The
            # primary field may be unmarked neutral Pinyin; downstream checks
            # decide whether it provides enough tone evidence to use.
            if index and not _TONE.search(value):
                continue
            yield value


def _eligible(dictionary: dict) -> bool:
    license_name = str(dictionary.get("license", "")).casefold()
    return (
        str(dictionary.get("id", "")).startswith("wiktionary-")
        and not dictionary.get("machineTranslated")
        and "by-nd" not in license_name
        and "no-derivative" not in license_name
        and ("cc by-sa" in license_name or "cc-by-sa" in license_name)
    )


def iter_raw_readings(web_root: Path, catalog: list[dict]) -> Iterator[dict]:
    """Yield source-attested Mandarin Pinyin with immutable source provenance.

    ``web_root`` is the website repository root, or its ``public`` directory.
    ``sourceRowIndex`` is the zero-based JSONL line index. Missing files, invalid
    gzip or malformed JSON raise rather than being silently counted as coverage.
    Files outside /data and path traversal are rejected before any file is read.
    """
    web_root = Path(web_root)
    public = web_root if (web_root / "data").is_dir() else web_root / "public"
    hashes: dict[Path, str] = {}
    for dictionary in catalog:
        if not _eligible(dictionary):
            continue
        source_file = dictionary.get("original")
        if not isinstance(source_file, str):
            raise RawReadingSourceError(f"Missing original path for {dictionary.get('id')}")
        parts = PurePosixPath(source_file).parts
        if not source_file.startswith("/data/") or ".." in parts or not source_file.endswith(".jsonl.gz"):
            raise RawReadingSourceError(f"Unsafe or unsupported original path for {dictionary.get('id')}")
        path = public / source_file.lstrip("/")
        if path not in hashes:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[path] = digest.hexdigest()
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for row_index, line in enumerate(stream):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RawReadingSourceError(
                        f"Malformed JSON in {source_file} at record {row_index}"
                    ) from error
                if not isinstance(record, dict):
                    raise RawReadingSourceError(f"Non-object record in {source_file} at record {row_index}")
                if record.get("lang_code") not in {"zh", "cmn", "Hani"}:
                    continue
                if record.get("pos") in {"romanization", "SOFT REDIRECT"}:
                    continue
                word = record.get("word")
                if not isinstance(word, str) or not word:
                    continue
                seen: set[tuple[str, str]] = set()
                for group, field in (("sounds", "zh_pron"), ("forms", "form")):
                    values = record.get(group, [])
                    if not isinstance(values, list):
                        continue
                    for value_index, value in enumerate(values):
                        if not isinstance(value, dict) or not _is_pinyin_field(record, value):
                            continue
                        raw = value.get(field)
                        if not isinstance(raw, str):
                            continue
                        for reading in _attested_parts(raw):
                            identity = (word, unicodedata.normalize("NFC", reading))
                            if identity in seen:
                                continue
                            seen.add(identity)
                            yield {
                                "sourceId": dictionary["id"],
                                "word": word,
                                "reading": reading,
                                "sourceFile": source_file,
                                "sourceRowIndex": row_index,
                                "sourceSha256": hashes[path],
                                "field": f"{group}[{value_index}].{field}",
                                "spokenLanguage": "cmn",
                                "rawReading": raw,
                                "sourceTags": list(value.get("tags", [])),
                                "sourceRawTags": list(value.get("raw_tags", [])),
                            }
