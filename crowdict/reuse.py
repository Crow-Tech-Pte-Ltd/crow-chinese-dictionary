"""Copy exact same-language source senses with their lexical labels; no model calls."""
import argparse
import gzip
import json
from pathlib import Path
from .common import canonical_json, sha, write_jsonl
from .validate import MAX_SOURCE_IMPORT_SENSES, _source_sense_options


def source_import(job):
    if not job.get('reuseEligible'):
        return None, 'not_reuse_eligible'
    if set(job.get('holdReasons', [])) - {'same_language_source_reuse_available'}:
        return None, 'unresolved_holds'
    senses = {}
    for evidence in job['evidence']:
        if evidence['language'] != job['targetLanguage'] or evidence['quality'] != 'full_definition':
            continue
        options, _, issues = _source_sense_options(evidence)
        if issues:
            return None, ','.join(issues)
        for definition, pos, register, _ in options:
            key = (definition, pos, register)
            sense = senses.setdefault(key, {'definition': definition, 'pos': pos,
                                           'register': register, 'sourceRefs': []})
            if evidence['id'] not in sense['sourceRefs']:
                sense['sourceRefs'].append(evidence['id'])
    if not senses:
        return None, 'no_same_language_senses'
    if len(senses) > MAX_SOURCE_IMPORT_SENSES:
        return None, 'source_sense_overflow'
    payload = {'jobId': job['id'], 'canonicalWord': job['canonicalWord'],
               'reading': job['reading'], 'spokenLanguage': job['spokenLanguage'],
               'targetLanguage': job['targetLanguage'], 'senses': list(senses.values()),
               'abstainReason': ''}
    return {'jobId': job['id'], 'generatorModel': 'source-import',
            'validationStatus': 'source_imported', 'draft': payload,
            'evidenceHash': job['evidenceHash'], 'draftSha256': sha(canonical_json(payload))}, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    counts = {'imported': 0, 'held': 0, 'skipped': 0, 'holdReasons': {}}
    held = []
    def records():
        opener = gzip.open if args.jobs.suffix == '.gz' else open
        with opener(args.jobs, 'rt', encoding='utf-8') as stream:
            for line in stream:
                job = json.loads(line)
                if not job.get('reuseEligible'):
                    counts['skipped'] += 1
                    continue
                result, reason = source_import(job)
                if reason:
                    counts['held'] += 1
                    counts['holdReasons'][reason] = counts['holdReasons'].get(reason, 0) + 1
                    held.append({'jobId': job['id'], 'reason': reason})
                else:
                    counts['imported'] += 1
                    yield result
    write_jsonl(args.out, records())
    report = args.report or args.out.with_name(args.out.name + '.report.json')
    report.write_text(json.dumps({'counts': counts, 'held': held}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(counts))

if __name__ == '__main__':
    main()
