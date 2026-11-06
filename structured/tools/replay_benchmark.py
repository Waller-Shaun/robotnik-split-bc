"""Replay the published paired benchmark in a fresh folder; never retrain."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_experiment import verify
from dynamic_pd.common import write
from dynamic_pd.evaluate import initialize, episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, help='First N goals, pipeline check only; omit for all 500')
    parser.add_argument('--variant', choices=('paired', 'fixed', 'dynamic'), default='paired')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('workers must be 1..8')
    if args.limit is not None and not 1 <= args.limit <= 500:
        parser.error('limit must be 1..500')
    selection = verify(runtime=True)
    if not (ROOT/'FINAL_RESULTS.json').exists():
        raise ValueError('Original independent final test has not been finalized yet')
    protocol = json.loads((ROOT/'protocol.json').read_text())
    entries = protocol['final'][:args.limit] if args.limit else protocol['final']
    variants = ('fixed', 'dynamic') if args.variant == 'paired' else (args.variant,)
    output = ROOT/'outputs/replays'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    (output/'logs').mkdir(parents=True)
    (output/'rollouts').mkdir()
    jobs = []
    for entry in entries:
        for repeat, seed in enumerate(entry['noise_seeds'][:3]):
            for variant in variants:
                stem = f'{variant}_{entry["target_index"]:04d}_{repeat}'
                jobs.append(dict(entry, stage='published_benchmark_replay', variant=variant,
                    repeat_index=repeat, seed=seed,
                    rollout_file=str(output/'rollouts'/(stem+'.npz')),
                    log_file=str(output/'logs'/(stem+'.log'))))
    configuration = dict(purpose='Replay, not new unseen test data', target_count=len(entries),
        planned=len(jobs), frozen_files=selection['frozen_files'], variants=variants,
        subset_only=len(entries)!=500, original_result_not_overwritten=True)
    write(output/'configuration.json', configuration)
    rows = []
    print(f'Replay: {len(jobs)} trials -> {output}', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize) as pool:
        for future in as_completed([pool.submit(episode, job) for job in jobs]):
            row = future.result()
            rows.append(row)
            if row['status'] == 'process_error':
                print(row['error'], flush=True)
            if len(rows)%10 == 0 or len(rows) == len(jobs):
                write(output/'results.json', rows)
                print(f'Completed {len(rows)}/{len(jobs)}', flush=True)
    result = dict(state='completed', subset_only=configuration['subset_only'],
        published_benchmark_replay=True, process_errors=sum(r['status']=='process_error' for r in rows),
        variants={v:dict(successes=sum(r['success'] for r in rows if r['variant']==v),
            episodes=3*len(entries), success_rate_percent=100*sum(r['success'] for r in rows if r['variant']==v)/(3*len(entries)))
            for v in variants})
    verify(runtime=False)
    write(output/'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)
    if result['subset_only']:
        print('Pipeline subset only: NOT the 500-target success rate.')


if __name__ == '__main__':
    main()
