"""Prepare priority queues for all requested definition languages, with no API calls."""
import argparse,gzip,json,itertools
from pathlib import Path
from .common import LANGUAGES,write_jsonl
from .inventory import jobs

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--web-root',type=Path,required=True);p.add_argument('--db',type=Path,default=Path('work/inventory.sqlite'));p.add_argument('--out',type=Path,default=Path('work/jobs'));p.add_argument('--languages',default='en,es,fr,de,pt,it,ja,ko,id,vi,ru');p.add_argument('--per-language-limit',type=int,default=2000);p.add_argument('--english-limit',type=int,default=10000);p.add_argument('--min-frequency',type=int,default=250);a=p.parse_args(argv)
    selected=list(LANGUAGES) if a.languages=='all' else a.languages.split(',')
    if len(set(selected))!=len(selected) or any(v not in LANGUAGES for v in selected):p.error('Unknown or duplicate definition language')
    summaries=[]
    for language in selected:
        summaries.append(jobs(a.db,a.web_root,language,a.out,a.english_limit if language=='en' else a.per_language_limit,a.min_frequency))
    def combined():
        for language in selected:
            with gzip.open(a.out/f'jobs-{language}.jsonl.gz','rt',encoding='utf-8') as stream:
                for line in stream:yield json.loads(line)
    write_jsonl(a.out/'jobs-priority.jsonl.gz',combined())
    def balanced():
        queues = {}
        for language in selected:
            with gzip.open(a.out/f'jobs-{language}.jsonl.gz','rt',encoding='utf-8') as stream:
                queues[language] = [json.loads(line) for line in stream]
        yield from queues.pop('en', [])
        for group in itertools.zip_longest(*queues.values()):
            for row in group:
                if row is not None:yield row
    write_jsonl(a.out/'jobs-balanced.jsonl.gz',balanced())
    report={'balancedOrdering':'English first, then round-robin across other languages, frequency/HSK priority within each language','languages':selected,'summaries':summaries,'generationEligible':sum(s['generationEligible'] for s in summaries),'sourceReuseEligible':sum(s['sourceReuseEligible'] for s in summaries),'held':sum(s['held'] for s in summaries),'networkRequests':0,'ordering':'English first, then explicitly requested language order; source-backed frequency/HSK priority within language.'}
    (a.out/'plan-summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='summaries'}))
if __name__=='__main__':main()
