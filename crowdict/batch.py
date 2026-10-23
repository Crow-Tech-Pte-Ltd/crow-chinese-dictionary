"""Resume bounded draft/check/export batches using one persistent credit ledger."""
import argparse
import atexit
import gzip
import json
from pathlib import Path
import subprocess
import sys
from .common import write_jsonl
from .generate import read_jobs
from .validate import read_jsonl


def run_step(module, arguments, log):
    result = subprocess.run([sys.executable, '-m', 'crowdict.' + module, *arguments],
                            capture_output=True, text=True)
    log.write_text(result.stdout + result.stderr, encoding='utf-8')
    summary = None
    for line in reversed(result.stdout.splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                summary = value
                break
        except ValueError:
            pass
    if summary is None or (result.returncode and not (result.returncode == 3 and summary.get('ledger', {}).get('halted'))):
        raise RuntimeError('stage_failed:' + module + ':see_private_log')
    return summary


def stopped(summary):
    statuses = summary.get('runResults', summary.get('runStatuses', summary.get('statuses', {})))
    return bool(summary.get('ledger', {}).get('halted') or
                any(statuses.get(k, 0) for k in ('budget_stopped', 'provider_limit_stopped')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('work/batch'))
    parser.add_argument('--export', type=Path, default=Path('data/generated'))
    parser.add_argument('--max-usd', type=float, required=True)
    parser.add_argument('--openrouter-max-usd', type=float, default=20, help='Separate durable cap for billed OpenRouter calls')
    parser.add_argument('--max-calls', type=int, required=True)
    parser.add_argument('--max-total-tokens', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-jobs', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--minimax-key-file')
    parser.add_argument('--openrouter-key-file')
    parser.add_argument('--additional-decisions', action='append', type=Path, default=[], help='Independent audit decisions; rejections/uncertainty remain holds')
    parser.add_argument('--raise-budget-caps', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error('max-jobs must be positive')
    if not 1 <= args.batch_size <= 200 or not 1 <= args.workers <= 4:
        parser.error('batch-size must be 1..200 and workers 1..4')
    jobs = [job for job in read_jobs(args.jobs) if job.get('generationEligible')]
    if args.max_jobs:
        jobs = jobs[:args.max_jobs]
    if len({job['id'] for job in jobs}) != len(jobs):
        parser.error('duplicate job identities')
    if not args.execute:
        print(json.dumps({'dryRun': True, 'networkRequests': 0, 'secretsRead': False,
                          'jobs': len(jobs), 'batches': (len(jobs) + args.batch_size - 1) // args.batch_size,
                          'maxUsd': args.max_usd, 'note': 'Shared cap includes prior ledger usage; no publication.'}))
        return
    if not args.minimax_key_file or not args.openrouter_key_file:
        parser.error('execution requires both explicit key-file paths')
    args.out.mkdir(parents=True, exist_ok=True)
    scope = args.out / 'jobs.jsonl.gz'
    # Fixed job scope protects resumed batch-to-path mapping.
    if scope.exists() and list(read_jsonl(scope)) != jobs:
        parser.error('batch job scope changed; choose a new work directory and retain the ledger')
    write_jsonl(scope, jobs)
    common = ['--ledger', str(args.ledger), '--max-usd', str(args.max_usd),
              '--max-calls', str(args.max_calls), '--max-total-tokens', str(args.max_total_tokens),
              '--max-output-tokens', '2048', '--execute']
    if args.raise_budget_caps:
        common += ['--raise-budget-caps']
    extra_decisions = [row for path in args.additional_decisions for row in read_jsonl(path)]
    all_results, all_decisions, status = [], [], {'status': 'running', 'completedBatches': 0, 'jobs': len(jobs)}
    def interrupted_checkpoint():
        if status['status'] == 'running':
            status['status'] = 'interrupted_or_failed'
            pending = args.out / 'status.json.pending'
            pending.write_text(json.dumps(status, indent=2) + '\n')
            pending.replace(args.out / 'status.json')
    atexit.register(interrupted_checkpoint)
    for offset in range(0, len(jobs), args.batch_size):
        if (args.out / 'STOP').exists():
            status['status'] = 'stop_requested'
            (args.out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
            print(json.dumps(status), flush=True)
            break
        folder = args.out / f'{offset // args.batch_size:05d}'
        folder.mkdir(exist_ok=True)
        shard = folder / 'jobs.jsonl.gz'
        write_jsonl(shard, jobs[offset:offset + args.batch_size])
        existing = folder / 'existing-drafts.jsonl.gz'
        run_step('snapshot', ['--jobs', str(shard), '--ledger', str(args.ledger), '--out', str(existing),
                              '--report', str(folder / 'existing-snapshot.json')], folder / 'existing-snapshot.log')
        completed = {row['jobId'] for row in read_jsonl(existing)}
        pending_jobs = folder / 'pending-jobs.jsonl.gz'
        write_jsonl(pending_jobs, (job for job in jobs[offset:offset + args.batch_size] if job['id'] not in completed))
        generation = run_step('generate', ['--jobs', str(pending_jobs), '--provider', 'minimax',
            '--model', 'MiniMax-M3.1-Flash', '--input-usd-per-million', '.15',
            '--output-usd-per-million', '.60', '--billing', 'token-plan', '--allow-credit-pack-debit',
            '--workers', str(args.workers), '--key-file', args.minimax_key_file, *common], folder / 'generation.log')
        snapshot = folder / 'drafts.jsonl.gz'
        run_step('snapshot', ['--jobs', str(shard), '--ledger', str(args.ledger), '--out', str(snapshot),
                              '--report', str(folder / 'snapshot.json')], folder / 'snapshot.log')
        decisions = folder / 'decisions.jsonl'
        review = run_step('review', ['--jobs', str(shard), '--results', str(snapshot),
            '--provider', 'openrouter', '--model', 'z-ai/glm-5.3',
            '--input-usd-per-million', '1.10', '--output-usd-per-million', '2.20',
            '--workers', str(args.workers), '--max-jobs', str(args.batch_size),
            '--provider-max-usd', str(args.openrouter_max_usd),
            '--key-file', args.openrouter_key_file, '--out-decisions', str(decisions),
            '--out-checks', str(folder / 'checks.jsonl'), *common], folder / 'review.log')
        all_results.extend(read_jsonl(snapshot))
        all_decisions.extend(read_jsonl(decisions))
        results_path, decisions_path = args.out / 'drafts.jsonl.gz', args.out / 'decisions.jsonl.gz'
        write_jsonl(results_path, all_results)
        write_jsonl(decisions_path, [*all_decisions, *extra_decisions])
        exported = run_step('validate', ['--jobs', str(scope), '--results', str(results_path),
            '--decisions', str(decisions_path), '--include-machine-checked', '--out', str(args.export)], folder / 'export.log')
        status.update(completedBatches=offset // args.batch_size + 1, processedJobs=min(offset + args.batch_size, len(jobs)),
                      accepted=exported['accepted'], counts=exported['counts'], ledger=review.get('ledger', generation.get('ledger')))
        should_stop = stopped(generation) or stopped(review)
        status['status'] = 'provider_or_budget_stopped' if should_stop else 'queue_finished' if offset + args.batch_size >= len(jobs) else 'running'
        pending = args.out / 'status.json.pending'
        pending.write_text(json.dumps(status, indent=2) + '\n')
        pending.replace(args.out / 'status.json')
        print(json.dumps(status), flush=True)
        if should_stop:
            break

if __name__ == '__main__':
    main()
