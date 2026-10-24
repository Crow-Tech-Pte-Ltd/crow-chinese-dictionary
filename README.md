# Crow Chinese Dictionary

A free, open Chinese dictionary dataset by [Crow Tech](https://github.com/Crow-Tech-Pte-Ltd),
built to fill vocabulary gaps in [Hanzi Desk](https://hanzidesk.com).
Reuse it in your own apps, research or commercial products under the licences below.

Repository: https://github.com/Crow-Tech-Pte-Ltd/crow-chinese-dictionary

## Current release

- **52,568 entries**, covering **46,287 distinct canonical headwords**.
- **32 definition languages**; these are unevenly sized, not 32 complete dictionaries.
- **45,487 attributed source imports** and **7,081 model-generated definitions**
  checked by a different model. English includes 45,487 source imports and 294
  generated definitions.
- **Zero entries claimed as human-reviewed.** Source imports retain their upstream
  text and labels; automatic checks do not guarantee correctness.

Each entry identifies its Chinese spelling, script forms, lexical reading, spoken
variety, definition language, source attribution and validation status. Different
readings remain separate; definition languages can share one Chinese pronunciation.
The [release manifest](data/release/manifest.json) is the authoritative count and
checksum record. Its current scope has **1,396 pending jobs**, excluded from release.

This is not complete coverage of Chinese, a comparison with Pleco's proprietary
collection, or a copy of Oxford, ABC or New Century. Audio coverage has not been
audited by this dataset; pronunciation references do not promise a playable clip.

## Download and use

- [Full records: gzip JSONL](data/release/entries.jsonl.gz), including per-entry provenance.
- [Compact browser packs and import manifest](data/release/website/import-manifest.json).
- [Source credits and notices](data/release/NOTICE.txt).

For example, read the full archive with Python's standard library:

```python
import gzip
import json

with gzip.open("data/release/entries.jsonl.gz", "rt", encoding="utf-8") as records:
    for line in records:
        entry = json.loads(line)
        print(entry["canonicalWord"], entry["lexicalReading"], entry["targetLanguage"])
```

The browser packs are adapters; retain access to the full attribution and notices
when redistributing them. Each record labels source imports versus synthetic
text, and each pack reports its mixture. Keep those labels in downstream products.

## Free and open

**Pipeline code:** [MIT](LICENSE). **Compatible derived dictionary data:**
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Both permit commercial use. Data reuse requires attribution, retained notices and
licence links; shared adaptations must follow ShareAlike. Original source licences
remain applicable, including CC BY-SA 3.0 and the full MIT notice where used.
See [data licensing](DATA_LICENSE.md) and [source notices](data/release/NOTICE.txt).
This is free to use under those terms, rather than public-domain or unrestricted data.

## Build or contribute

Python 3.10+, standard library only. Tests and offline inspection need no API key:

```sh
python3 -m unittest discover -s tests
```

The [pipeline guide](docs/pipeline.md) explains source downloads, gap discovery,
exact source reuse, bounded generation, independent checking, resumable ledgers and
release export. Missing or conflicting readings, restricted source material,
unreviewed drafts and failed checks stay out of accepted releases. ND definitions
and unreviewed mixed-licence sources are excluded from derived glosses.

Corrections with an attested reading and permissible source are welcome.
