"""Read-only discovery/review queues, deliberately separate from paid jobs.

Counts cover distinct (spoken language, resolved canonical headword) keys in the
supplied inventory. They are not a comparison against Pleco, not proof that every
source headword is correct, and not proof of missing English meanings. Queue
records never authorize model generation; inventory.jobs applies that policy.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sqlite3
from pathlib import Path

from .common import canonical_json, sha, write_jsonl

DEFAULT_SOURCE_IDS = ('cedict', 'phrasebook-en')
_SOURCE_SCOPE = "CASE WHEN r.source_id LIKE 'zh-wiktionary-%' THEN 'zh-wiktionary-sense' WHEN r.source_id LIKE 'wordnet-%' THEN 'pwn-3.0' ELSE r.source_id END"

SPECS = {
    'unresolved_english_reading': {
        'label': 'English canonical head absent, with unresolved permitted definition evidence',
        'description': 'At least one permitted non-AI full-definition source row has a missing, ambiguous or invalid reading, and the canonical head has no baseline English definition. Other source rows for the head can already have readings.',
        'query': '''SELECT DISTINCT a.spoken,a.canonical FROM cq_allowed a
                    WHERE a.reading_status IN ('missing','ambiguous','invalid')
                    AND NOT EXISTS(SELECT 1 FROM cq_english e WHERE e.spoken=a.spoken AND e.canonical=a.canonical)''',
        'refs': "s.policy='compatible' AND s.quality='full_definition' AND r.has_definition=1 AND z.reading_status IN ('missing','ambiguous','invalid')",
        'notMissing': False,
        'warning': 'Resolve source readings and aliases before treating these as publishable missing words. Invalid readings are held alongside missing/ambiguous readings.',
    },
    'additional_reading_lead': {
        'label': 'Existing English head with an unmatched foreign-source reading lead',
        'description': 'The English head already exists. A permitted non-AI foreign-language definition has an indexed reading not represented by the English head at the exact same normalized reading.',
        'query': '''SELECT DISTINCT a.spoken,a.canonical FROM cq_allowed a
                    WHERE a.language!='en' AND a.nr!='' AND a.alias_status!='conflict'
                    AND a.lexeme_id IS NOT NULL
                    AND EXISTS(SELECT 1 FROM cq_english e WHERE e.spoken=a.spoken AND e.canonical=a.canonical)
                    AND NOT EXISTS(SELECT 1 FROM cq_english_readings e WHERE e.spoken=a.spoken AND e.canonical=a.canonical AND e.nr=a.nr)''',
        'refs': "s.policy='compatible' AND s.quality='full_definition' AND r.has_definition=1 AND r.language!='en' AND z.nr!='' AND z.alias_status!='conflict' AND z.lexeme_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM cq_english_readings e WHERE e.spoken=z.spoken AND e.canonical=z.canonical AND e.nr=z.nr)",
        'notMissing': True,
        'warning': 'UNVERIFIED reading lead, not a proven missing pronunciation or meaning. Sandhi, regional conventions, source errors and notation differences can explain the difference.',
    },
    'machine_only_english': {
        'label': 'Existing English coverage supplied only by machine-translated packs',
        'description': 'The English canonical head exists in baseline packs, and none of those English definitions is from a non-machine-translated pack.',
        'query': 'SELECT spoken,canonical FROM cq_english WHERE has_nonmachine=0',
        'refs': "s.baseline=1 AND r.language='en' AND r.has_definition=1 AND s.quality='machine_translated'",
        'notMissing': True,
        'warning': 'Quality-upgrade queue, not a missing-word queue. Existing AI definitions prevent duplicate automatic paid generation.',
    },
    'available_optional_english': {
        'label': 'English coverage already available outside the default two packs',
        'description': 'The English canonical head exists in baseline packs but not in CC-CEDICT or the phrase supplement. Availability is independent of a particular user enabling/downloading the pack.',
        'query': '''SELECT e.spoken,e.canonical FROM cq_english e
                    WHERE NOT EXISTS(SELECT 1 FROM cq_default d WHERE d.spoken=e.spoken AND d.canonical=e.canonical)''',
        'refs': "s.baseline=1 AND r.language='en' AND r.has_definition=1 AND r.source_id NOT IN ('cedict','phrasebook-en')",
        'notMissing': True,
        'warning': 'No new paid definition is needed merely to expose an existing optional pack. Some optional coverage is machine-translated; inspect source quality.',
    },
    'supplemental_new_head': {
        'label': 'Supplemental canonical heads absent from all baseline languages',
        'description': 'A permitted supplemental source supplies a canonical head that does not appear in any baseline dictionary, across all definition languages.',
        'query': '''SELECT DISTINCT a.spoken,a.canonical FROM cq_allowed a JOIN sources s ON s.id=a.source_id
                    WHERE s.baseline=0
                    AND NOT EXISTS(SELECT 1 FROM cq_baseline b WHERE b.spoken=a.spoken AND b.canonical=a.canonical)''',
        'refs': "s.baseline=0 AND s.policy='compatible' AND s.quality='full_definition' AND r.has_definition=1",
        'notMissing': False,
        'warning': 'Source-backed discovery lead, not automatically an accepted new word. Source errors, script aliases, missing readings and nonlexical expressions still need review.',
    },
    'additional_sense_lead': {
        'label': 'Existing English heads with unmatched explicit foreign sense anchors',
        'description': 'Only explicit source sense IDs/synset anchors are considered. An unmatched foreign anchor identifies something to compare, not a demonstrated missing English sense.',
        'query': '''SELECT DISTINCT a.spoken,a.canonical FROM cq_anchors a
                    WHERE a.language!='en'
                    AND EXISTS(SELECT 1 FROM cq_english e WHERE e.spoken=a.spoken AND e.canonical=a.canonical)
                    AND NOT EXISTS(SELECT 1 FROM cq_english_anchors e WHERE e.spoken=a.spoken AND e.canonical=a.canonical AND e.scope=a.scope AND e.sense_id=a.sense_id)''',
        'refs': "s.policy IN ('compatible','component_review') AND s.quality!='machine_translated' AND r.has_definition=1 AND r.language!='en' AND r.sense_id!='' AND NOT EXISTS(SELECT 1 FROM cq_english_anchors e WHERE e.spoken=z.spoken AND e.canonical=z.canonical AND e.scope=(" + _SOURCE_SCOPE + ") AND e.sense_id=r.sense_id)",
        'notMissing': True,
        'warning': 'Absence of an English sense-ID mapping DOES NOT prove a missing English meaning. Most sources do not align their sense IDs. Component-review sources are discovery metadata only.',
    },
}


def _prepare(database: sqlite3.Connection) -> None:
    """Materialize small indexed review tables in connection-local TEMP storage."""
    database.executescript('''
        CREATE TEMP TABLE cq_english AS
          SELECT spoken,canonical,max(quality!='machine_translated') AS has_nonmachine
          FROM head_coverage WHERE language='en' AND baseline=1 GROUP BY spoken,canonical;
        CREATE UNIQUE INDEX cq_english_key ON cq_english(spoken,canonical);
        CREATE TEMP TABLE cq_default AS
          SELECT DISTINCT z.spoken,z.canonical FROM resolutions z JOIN records r ON r.id=z.record_id
          JOIN sources s ON s.id=r.source_id
          WHERE s.baseline=1 AND r.language='en' AND r.has_definition=1 AND r.source_id IN ('cedict','phrasebook-en');
        CREATE UNIQUE INDEX cq_default_key ON cq_default(spoken,canonical);
        CREATE TEMP TABLE cq_baseline AS
          SELECT DISTINCT z.spoken,z.canonical FROM resolutions z JOIN records r ON r.id=z.record_id
          JOIN sources s ON s.id=r.source_id WHERE s.baseline=1;
        CREATE UNIQUE INDEX cq_baseline_key ON cq_baseline(spoken,canonical);
        CREATE TEMP TABLE cq_english_readings AS
          SELECT DISTINCT z.spoken,z.canonical,z.nr FROM resolutions z JOIN records r ON r.id=z.record_id
          JOIN sources s ON s.id=r.source_id
          WHERE s.baseline=1 AND r.language='en' AND r.has_definition=1 AND z.nr!='' AND z.alias_status!='conflict';
        CREATE UNIQUE INDEX cq_english_readings_key ON cq_english_readings(spoken,canonical,nr);
        CREATE TEMP TABLE cq_allowed AS
          SELECT z.record_id,z.spoken,z.canonical,z.nr,z.reading_status,z.alias_status,z.lexeme_id,
                 r.source_id,r.language,r.sense_id
          FROM resolutions z JOIN records r ON r.id=z.record_id JOIN sources s ON s.id=r.source_id
          WHERE s.policy='compatible' AND s.quality='full_definition' AND r.has_definition=1;
        CREATE INDEX cq_allowed_key ON cq_allowed(spoken,canonical,nr);
    ''')
    database.execute('''CREATE TEMP TABLE cq_anchors AS
        SELECT z.record_id,z.spoken,z.canonical,r.language,r.sense_id,''' + _SOURCE_SCOPE + ''' AS scope
        FROM resolutions z JOIN records r ON r.id=z.record_id JOIN sources s ON s.id=r.source_id
        WHERE s.policy IN ('compatible','component_review') AND s.quality!='machine_translated'
        AND r.has_definition=1 AND r.sense_id!='' ''')
    database.executescript('''
        CREATE INDEX cq_anchor_key ON cq_anchors(spoken,canonical,scope,sense_id);
        CREATE TEMP TABLE cq_english_anchors AS
          SELECT DISTINCT a.spoken,a.canonical,a.scope,a.sense_id FROM cq_anchors a
          JOIN records r ON r.id=a.record_id JOIN sources s ON s.id=r.source_id
          WHERE a.language='en' AND s.baseline=1;
        CREATE UNIQUE INDEX cq_english_anchor_key ON cq_english_anchors(spoken,canonical,scope,sense_id);
    ''')


def _scope(source_id: str) -> str:
    if source_id.startswith('zh-wiktionary-'):
        return 'zh-wiktionary-sense'
    if source_id.startswith('wordnet-'):
        return 'pwn-3.0'
    return source_id


def _sample(database: sqlite3.Connection, kind: str, spoken: str, word: str, sources: dict, fingerprint: str) -> dict:
    specification = SPECS[kind]
    base = '''FROM resolutions z JOIN records r ON r.id=z.record_id JOIN sources s ON s.id=r.source_id
              WHERE z.spoken=? AND z.canonical=? AND (''' + specification['refs'] + ')'
    parameters = (spoken, word)
    totals = database.execute('SELECT count(*),count(DISTINCT s.id),count(DISTINCT s.family) ' + base, parameters).fetchone()
    rows = database.execute('''SELECT r.source_id,r.row_index,r.language,r.reading,z.reading AS resolved_reading,
                              z.nr,z.reading_status,z.alias_status,r.sense_id,s.family,s.quality,s.policy,s.baseline ''' + base +
                            ' ORDER BY r.source_id,r.row_index LIMIT 24', parameters)
    references = []
    for row in rows:
        source = sources[row['source_id']]
        reference = dict(sourceId=row['source_id'], rowIndex=row['row_index'], targetLanguage=row['language'],
                         reading=row['reading'], resolvedReading=row['resolved_reading'], normalizedReading=row['nr'],
                         readingStatus=row['reading_status'], aliasStatus=row['alias_status'],
                         sourceFamily=row['family'], quality=row['quality'], licencePolicy=row['policy'],
                         baseline=bool(row['baseline']), sourceSha256=source.get('sha256', ''),
                         sourceFile=source.get('file', ''), licence=source.get('license', ''))
        if row['sense_id']:
            reference['senseAnchor'] = row['sense_id']
            reference['senseAnchorScope'] = _scope(row['source_id'])
        references.append(reference)
    english_readings = [row[0] for row in database.execute('SELECT nr FROM cq_english_readings WHERE spoken=? AND canonical=? ORDER BY nr', parameters)]
    machine = database.execute('SELECT has_nonmachine FROM cq_english WHERE spoken=? AND canonical=?', parameters).fetchone()
    return {
        'id': sha(canonical_json(['review-queue', kind, spoken, word])),
        'queueKind': kind, 'canonicalWord': word, 'spokenLanguage': spoken,
        'targetLanguage': 'en', 'generationEligible': False, 'reviewRequired': True,
        'notMissingEnglishHeadword': specification['notMissing'],
        'trueMissingMeaningConfirmed': False,
        'warning': specification['warning'], 'sourceFingerprint': fingerprint,
        'existingEnglishNormalizedReadings': english_readings,
        'existingEnglishMachineOnly': bool(machine is not None and not machine[0]),
        'sourceReferenceCount': totals[0], 'sourceCount': totals[1], 'sourceFamilyCount': totals[2],
        'sourceRefs': references, 'sourceRefsTruncated': totals[0] > len(references),
    }


def build_queues(db_path: Path, out: Path = Path('work/queues'), sample_limit: int = 1000) -> dict:
    """Export exact counts and deterministic metadata-only samples per variety.

    ``sample_limit`` caps each queue *per spoken language*, so Mandarin samples
    cannot hide the Cantonese sample. Counts always scan the full query results.
    The inventory itself is opened mode=ro and cannot be changed by this command.
    """
    if sample_limit < 0:
        raise ValueError('sample_limit must be zero or positive')
    db_path = Path(db_path).resolve()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True)) as database:
        database.row_factory = sqlite3.Row
        database.execute('PRAGMA temp_store=FILE')
        database.execute('PRAGMA cache_size=-32768')
        row = database.execute("SELECT value FROM metadata WHERE key='sourceFingerprint'").fetchone()
        if row is None:
            raise ValueError('Inventory source fingerprint is missing')
        fingerprint = row[0]
        sources = {row['id']: json.loads(row['catalog_json']) for row in database.execute('SELECT id,catalog_json FROM sources')}
        _prepare(database)
        summary = {
            'schemaVersion': 1, 'sourceFingerprint': fingerprint, 'targetLanguage': 'en',
            'countUnit': 'distinct (spokenLanguage, resolved canonicalWord) heads',
            'sampleLimitPerQueuePerSpokenLanguage': sample_limit,
            'generationEligible': False,
            'defaultEnglishSourceIds': list(DEFAULT_SOURCE_IDS),
            'warnings': [
                'Categories can overlap; adding their counts does not give a unique missing-word total.',
                'Counts describe this supplied reference universe, not an exact Pleco comparison or proof of complete vocabulary coverage.',
                'Canonical-key absence can still be a script-alias or reading-resolution issue. Inventory.jobs has additional alias, source-quality and frequency gates.',
                'Reading and sense leads are unverified; they do not prove a missing pronunciation or English meaning.',
                'No source definitions, ND snippets, secrets or model prompts are exported. No paid generation is authorised by these queues.',
            ],
            'queues': {},
        }
        for kind, specification in SPECS.items():
            query = specification['query']
            counts = {'cmn': 0, 'yue': 0}
            counts.update({row['spoken']: row['n'] for row in database.execute(
                'SELECT spoken,count(*) AS n FROM (' + query + ') GROUP BY spoken')})
            samples = []
            sampled = {spoken: 0 for spoken in counts}
            if sample_limit:
                for spoken in sorted(counts):
                    for row in database.execute('SELECT spoken,canonical FROM (' + query + ') WHERE spoken=? ORDER BY canonical LIMIT ?', (spoken, sample_limit)):
                        samples.append(_sample(database, kind, row['spoken'], row['canonical'], sources, fingerprint))
                        sampled[spoken] += 1
            write_jsonl(out / (kind + '.jsonl.gz'), samples)
            summary['queues'][kind] = {
                'label': specification['label'], 'description': specification['description'],
                'warning': specification['warning'], 'countsBySpokenLanguage': counts,
                'totalHeads': sum(counts.values()), 'exportedSamplesBySpokenLanguage': sampled,
                'sampleFile': kind + '.jsonl.gz', 'generationEligible': False,
            }
            print(kind + ': ' + canonical_json(counts), flush=True)
    pending = out / 'summary.json.pending'
    pending.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pending.replace(out / 'summary.json')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=Path('work/inventory.sqlite'))
    parser.add_argument('--out', type=Path, default=Path('work/queues'))
    parser.add_argument('--sample-limit', type=int, default=1000,
                        help='maximum metadata samples per queue per spoken language; exact counts are never truncated')
    args = parser.parse_args(argv)
    build_queues(args.db, args.out, args.sample_limit)


if __name__ == '__main__':
    main()
