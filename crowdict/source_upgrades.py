"""Copy licensed English sources over existing machine-only coverage, without APIs.

This is a separate quality-upgrade adjunct, not a missing-headword inventory.
The frozen database is opened read-only. Existing AI rows establish coverage and
same-reading comparison only; no AI definition or alias is used as evidence.
All eligible English source POS/sense records for an identity are kept together.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import json
from pathlib import Path
import re
import sqlite3

from .common import canonical_json, is_derived_crow_source, normalized_reading, sha, source_policy, valid_reading, write_jsonl
from .inventory import COMPOSITIONAL, job_script_forms, source_file, source_rows
from .raw_sources import _hash
from .reuse import source_import
from .validate import _evidence_issues, validate_draft

KIND = 'existing_machine_only_english_quality_upgrade'


def _source_evidence(source, ref, row, canonical, reading):
    index = ref['row_index']
    meta = row[4] if len(row) > 4 and isinstance(row[4], dict) else {}
    evidence = {
        'id': sha(canonical_json([source['id'], source['sha256'], index]))[:24],
        'sourceId': source['id'], 'language': 'en', 'definition': row[3],
        'reading': reading, 'license': source['license'],
        'licenseUrl': source.get('licenseUrl', ''), 'attribution': source.get('attribution', ''),
        'sourceUrl': source.get('url', ''), 'snapshot': source.get('snapshot', ''),
        'sourceSha256': source['sha256'], 'originalRowIndex': index,
        'quality': 'full_definition', 'sourceFamily': source['family'],
        'canonicalWord': canonical, 'sourcePos': ref['pos'],
    }
    if source.get('licenseNotice'):
        evidence['licenseNotice'] = source['licenseNotice']
    if ref['sense_id']:
        evidence['senseId'] = ref['sense_id']
    if meta.get('sourceArticleUrl'):
        evidence['sourceUrl'] = meta['sourceArticleUrl']
        evidence['sourceDatasetUrl'] = source.get('url', '')
    if 'rawOriginalRowIndex' in meta:
        evidence.update(rawOriginalRowIndex=meta['rawOriginalRowIndex'],
                        rawSourceFile=source['file'],
                        rawReadingFields=meta.get('rawReadingFields', []),
                        rawSourceSenses=meta.get('rawSourceSenses', []))
    return evidence


def build_source_upgrades(db_path, out, *, web_root=None, report=None):
    """Write an independent, source-only jobs/results/held queue atomically.

    Does not rebuild inventory, modify its database or read an existing release.
    A source snapshot/hash mismatch aborts before any output is published.
    """
    db_path, out = Path(db_path).resolve(), Path(out)
    jobs, results, held, recovery = [], [], [], []
    held_counts, exclusions = Counter(), Counter()
    with contextlib.closing(sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        metadata = dict(db.execute('SELECT key,value FROM metadata'))
        web_root = Path(web_root or metadata['webRoot'])
        sources = {row['id']: json.loads(row['catalog_json']) | {
            'id': row['id'], 'family': row['family'], 'policy': row['policy'],
            'quality': row['quality'], 'baseline': bool(row['baseline']),
        } for row in db.execute('SELECT * FROM sources')}
        machine_heads = {tuple(row) for row in db.execute(
            "SELECT spoken,canonical FROM head_coverage WHERE language='en' AND baseline=1 "
            "GROUP BY spoken,canonical HAVING max(quality!='machine_translated')=0")}
        baseline_human_aliases = defaultdict(set)
        for row in db.execute(
            "SELECT r.spoken,r.aliases FROM records r JOIN sources s ON s.id=r.source_id "
            "WHERE r.language='en' AND r.has_definition=1 AND s.baseline=1 AND s.quality!='machine_translated'"):
            baseline_human_aliases[row['spoken']].update(json.loads(row['aliases']))
        ai_readings = {tuple(row) for row in db.execute(
            "SELECT z.spoken,z.canonical,z.nr FROM resolutions z JOIN records r ON r.id=z.record_id "
            "JOIN sources s ON s.id=r.source_id WHERE r.language='en' AND r.has_definition=1 "
            "AND s.baseline=1 AND s.quality='machine_translated' AND z.alias_status!='conflict' AND z.nr!=''")}
        groups, wanted = defaultdict(list), defaultdict(set)
        supplement_heads = set()
        query = """SELECT r.*,z.canonical AS resolved_canonical,z.reading AS resolved_reading,
                          z.nr AS resolved_nr,z.spoken AS resolved_spoken,z.lexeme_id,z.reading_status,z.alias_status
                   FROM records r JOIN sources s ON s.id=r.source_id JOIN resolutions z ON z.record_id=r.id
                   WHERE r.language='en' AND s.baseline=0 AND s.policy='compatible'
                   AND s.quality='full_definition' AND r.has_definition=1
                   ORDER BY r.source_id,r.row_index"""
        for original in db.execute(query):
            ref = dict(original)
            source = sources[ref['source_id']]
            if is_derived_crow_source(source) or source_policy(source) != 'compatible':
                continue
            head = (ref['resolved_spoken'], ref['resolved_canonical'])
            if head not in machine_heads:
                continue
            supplement_heads.add(head)
            identity = (*head, ref['resolved_nr'])
            reason = None
            if not ref['lexeme_id'] or ref['alias_status'] == 'conflict' or not valid_reading(ref['resolved_reading'], head[0]):
                reason = 'missing_invalid_or_conflicting_source_reading'
            elif identity not in ai_readings:
                reason = 'same_reading_baseline_ai_not_proven'
            if reason:
                exclusions[reason] += 1
                recovery.append({'kind': KIND, 'generationEligible': False, 'canonicalWord': head[1],
                                 'spokenLanguage': head[0], 'normalizedReading': ref['resolved_nr'],
                                 'sourceId': source['id'], 'sourceRowIndex': ref['row_index'],
                                 'sourceSha256': source.get('sha256', ''), 'reason': reason})
                continue
            groups[identity].append(ref)
            wanted[source['id']].add(ref['row_index'])
        # Verify compressed source bytes once, then retain only the required rows.
        values, verified_sources = {}, {}
        for sid, indexes in sorted(wanted.items()):
            source = sources[sid]
            path = source_file(web_root, source)
            digest = _hash(path)
            if not source.get('sha256') or digest != source['sha256']:
                raise ValueError('Source snapshot checksum mismatch: ' + sid)
            verified_sources[sid] = {'sourceId': sid, 'sha256': digest,
                                     'license': source['license'], 'rowsReferenced': len(indexes)}
            for index, row in enumerate(source_rows(web_root, source)):
                if index in indexes:
                    values[(sid, index)] = row
            if _hash(path) != digest:
                raise ValueError('Source snapshot changed during loading: ' + sid)
            if any((sid, index) not in values for index in indexes):
                raise ValueError('Source row missing from verified snapshot: ' + sid)
        for identity, refs in sorted(groups.items()):
            spoken, canonical, nr = identity
            job_id = sha(canonical_json([spoken, canonical, nr, 'en']))
            aliases = {canonical}
            for ref in refs:
                aliases.update(json.loads(ref['aliases']))
            reasons = []
            if aliases & baseline_human_aliases[spoken]:
                reasons.append('existing_nonmachine_english_alias_overlap')
            if any(ref['proper'] for ref in refs):
                reasons.append('proper_name_variant_or_abbreviation_needs_review')
            if COMPOSITIONAL.fullmatch(canonical):
                reasons.append('productive_quantity_expression_review')
            if len(canonical) > 12:
                reasons.append('long_expression_needs_lexical_review')
            if spoken != 'cmn':
                reasons.append('spoken_variety_needs_separate_review')
            attested, reading_provenance = [], []
            for ref in refs:
                row = values[(ref['source_id'], ref['row_index'])]
                explicit_reading_matches = valid_reading(row[2], spoken) and normalized_reading(row[2]) == nr
                if ref['reading_status'] == 'source_attested' and not explicit_reading_matches:
                    reasons.append('source_reading_snapshot_mismatch')
                if ref['reading_status'] == 'source_attested' and explicit_reading_matches:
                    attested.append(row[2].strip())
                    reading_provenance.append({'sourceId': ref['source_id'],
                        'sourceSha256': sources[ref['source_id']]['sha256'], 'sourceFile': sources[ref['source_id']]['file'],
                        'rowIndex': ref['row_index'], 'field': 'pack.lexicalReading', 'reading': row[2].strip()})
                if re.search(r'Cantonese|cantonais|cantonés|Kantonesisch', row[3], re.I):
                    reasons.append('source_spoken_variety_needs_review')
            # A baseline AI reading never fills an unknown English source reading.
            # Resolved-only groups remain held until their independent reading
            # proof can be retained in the public archival format as well.
            if not attested:
                reasons.append('explicit_english_source_reading_required')
            reading = attested[0] if attested else refs[0]['resolved_reading']
            evidence = [_source_evidence(sources[ref['source_id']], ref,
                         values[(ref['source_id'], ref['row_index'])], canonical, reading) for ref in refs]
            job = {'id': job_id, 'jobId': job_id, 'canonicalWord': canonical,
                   'aliases': sorted(aliases), 'reading': reading, 'normalizedReading': nr,
                   'spokenLanguage': spoken, 'targetLanguage': 'en', 'gapKind': KIND, 'candidateKind': KIND,
                   'qualityUpgrade': True, 'generationEligible': False, 'reuseEligible': not reasons,
                   'holdReasons': sorted(set(reasons)), 'evidence': evidence,
                   'evidenceHash': sha(canonical_json(evidence)), 'readingProvenance': reading_provenance,
                   'sourceLicenses': sorted({item['license'] for item in evidence}),
                   'sourceFamilies': sorted({item['sourceFamily'] for item in evidence})}
            job.update(job_script_forms(refs, canonical, aliases))
            result = None
            if not reasons:
                result, reason = source_import(job)
                if reason:
                    reasons.append(reason)
                else:
                    issues, _, _, _ = validate_draft(job, result)
                    issues += [issue for item in evidence for issue in _evidence_issues(item, publication=True)]
                    reasons.extend(issues)
            if reasons:
                reasons = sorted(set(reasons));held_counts.update(reasons)
                held.append({'jobId': job_id, 'kind': KIND, 'canonicalWord': canonical, 'reading': reading,
                             'normalizedReading': nr, 'spokenLanguage': spoken, 'targetLanguage': 'en',
                             'generationEligible': False, 'reuseEligible': False, 'holdReasons': reasons,
                             'sourceRefs': [{key: item[key] for key in ('sourceId','originalRowIndex','sourceSha256')} for item in evidence]})
            else:
                jobs.append(job);results.append(result)
        summary = {
            'kind': KIND, 'sourceFingerprint': metadata['sourceFingerprint'],
            'baselineMachineOnlyEnglishHeads': len(machine_heads),
            'supplementHeadsAtExistingMachineOnlyEnglishCoverage': len(supplement_heads),
            'candidateHeadsWithExactBaselineAiReading': len({identity[:2] for identity in groups}),
            'candidateLexicalReadings': len(groups),
            'sourceOnlyQualityUpgradeJobs': len(jobs),
            'sourceOnlyQualityUpgradeHeads': len({(job['spokenLanguage'],job['canonicalWord']) for job in jobs}),
            'newHeadwords': 0, 'modelGeneratedDefinitions': 0, 'generationApiCalls': 0,
            'heldJobs': len(held), 'holdReasons': dict(sorted(held_counts.items())),
            'excludedSourceRows': dict(sorted(exclusions.items())),
            'recoveryRows': len(recovery), 'verifiedEnglishSources': list(verified_sources.values()),
            'automaticChecksAreHumanReview': False,
            'scope': 'A separate source-only quality adjunct for existing AI-only English heads. Existing missing-word jobs, inventory and release are unchanged.',
            'selectionPolicy': 'Frozen identity/reading mapping; exact baseline-AI reading comparison; all eligible compatible non-AI English source POS/sense rows grouped without caps or cherry-picking. Combined sense overflow holds the entire job. Baseline AI is never source evidence.',
        }
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [('jobs-en.jsonl.gz',jobs),('results-en.jsonl.gz',results),
                       ('held-en.jsonl.gz',held),('recovery-en.jsonl.gz',recovery)]:
        write_jsonl(out/name, rows)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    if report:
        report=Path(report);report.parent.mkdir(parents=True,exist_ok=True)
        report.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(canonical_json(summary),flush=True)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,default=Path('work/inventory.sqlite'))
    parser.add_argument('--out',type=Path,default=Path('work/source-upgrades'))
    parser.add_argument('--web-root',type=Path)
    parser.add_argument('--report',type=Path,default=Path('reports/source-upgrades.json'))
    args=parser.parse_args(argv)
    build_source_upgrades(args.db,args.out,web_root=args.web_root,report=args.report)


if __name__=='__main__':
    main()
