"""Combine current source imports and checked drafts into one offline release."""
import argparse
import json
from pathlib import Path
from .common import write_jsonl
from .validate import export, read_jsonl, read_results


def combine(paths):
    rows = {}
    for path in paths:
        for job in read_jsonl(path):
            key = job.get('jobId', job.get('id'))
            if key in rows and rows[key] != job:
                raise ValueError('Conflicting job scope; use current matching inputs only')
            rows[key] = job
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=Path,action='append',required=True)
    parser.add_argument('--results',type=Path,action='append',required=True)
    parser.add_argument('--decisions',type=Path,action='append',default=[])
    parser.add_argument('--work',type=Path,default=Path('work/bundle'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--include-source-imports',action='store_true')
    parser.add_argument('--include-machine-checked',action='store_true')
    args=parser.parse_args(argv)
    jobs=combine(args.jobs)
    args.work.mkdir(parents=True,exist_ok=True)
    job_file=args.work/'jobs.jsonl.gz'
    result_file=args.work/'results.jsonl.gz'
    decision_file=args.work/'decisions.jsonl.gz'
    write_jsonl(job_file,jobs.values())
    write_jsonl(result_file,(row for path in args.results for row in read_results(path,jobs)))
    write_jsonl(decision_file,(row for path in args.decisions for row in read_jsonl(path)))
    manifest=export(job_file,result_file,decision_file,args.out,
                    include_source_imports=args.include_source_imports,
                    include_machine_checked=args.include_machine_checked)
    print(json.dumps({'counts':manifest['counts'],'networkRequests':0,'humanReviewIsNotInferred':True}))

if __name__=='__main__':main()
