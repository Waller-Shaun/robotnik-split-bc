"""Re-run the published primary target/seed list without training or filtering."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import run_model

ROOT = Path(__file__).resolve().parent


def weight_hash(controller):
    h = hashlib.sha256()
    for label, module in (('arm', controller.model), ('nav', controller.nav)):
        for name, value in module.state_dict().items():
            h.update((label + name).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def initialize(model):
    import torch
    global CONTROLLER, WEIGHT_HASH
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    CONTROLLER = run_model.load_controller(model)
    WEIGHT_HASH = weight_hash(CONTROLLER)


def episode(job):
    import numpy as np
    row = {k: job[k] for k in ('target_index', 'repeat_index', 'seed', 'target')}
    row.update(success=False, status='process_error', error='', rollout_file=job['result_file'])
    before = time.perf_counter()
    try:
        if weight_hash(CONTROLLER) != WEIGHT_HASH:
            raise ValueError('Weights changed between episodes')
        CONTROLLER.reset()
        if len(CONTROLLER.history) or getattr(CONTROLLER, 'q_ref', None) is not None:
            raise ValueError('Episode memory was not reset')
        args = SimpleNamespace(**job, duration=20., headless=True, display_fps=30, unpaced=True)
        with Path(job['log_file']).open('w') as stream:
            out_fd, err_fd = os.dup(1), os.dup(2)
            try:
                os.dup2(stream.fileno(), 1)
                os.dup2(stream.fileno(), 2)
                with redirect_stdout(stream), redirect_stderr(stream):
                    result = run_model.run_episode(args, CONTROLLER)
                stream.flush()
            finally:
                os.dup2(out_fd, 1)
                os.dup2(err_fd, 2)
                os.close(out_fd)
                os.close(err_fd)
        with np.load(job['result_file'], allow_pickle=False) as data:
            errors = np.linalg.norm(data['ee_positions'] - data['target'], axis=1)
            np.testing.assert_allclose(errors, data['ee_errors'], atol=1e-14, rtol=0)
            if bool((errors < .01).any()) != result['success']:
                raise ValueError('Raw true-state trajectory does not support success flag')
        if weight_hash(CONTROLLER) != WEIGHT_HASH:
            raise ValueError('Inference modified model weights')
        if any(result['sensor_noise_enabled']) != (job['noise_profile'] == 'friend'):
            raise ValueError('Unexpected noise condition')
        row.update(success=result['success'], status='success' if result['success'] else 'episode_failure',
                   min_error_m=result['minimum_error_m'], final_error_m=result['final_error_m'],
                   end_time_s=result['end_time_s'])
    except BaseException:
        row['error'] = traceback.format_exc()
    row['wall_s'] = time.perf_counter() - before
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=('tcn', 'mlp', 'baseline'), default='tcn')
    p.add_argument('--noise-profile', choices=('friend', 'off'), default='friend')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--limit', type=int, help='First N shared goals; pipeline/subset only')
    args = p.parse_args()
    if not 1 <= args.workers <= 8 or args.limit is not None and not 1 <= args.limit <= 500:
        p.error('workers must be 1..8; limit must be 1..500')
    shared = json.loads((ROOT / 'results/shared_test_targets.json').read_text())
    targets = shared['targets'][:args.limit]
    repeat_count = 3 if args.noise_profile == 'friend' else 1
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output = ROOT / 'outputs' / f'batch_{args.model}_{args.noise_profile}_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'rollouts').mkdir()
    (output / 'logs').mkdir()
    jobs = []
    for target in targets:
        for i in range(repeat_count):
            stem = f"target_{target['target_index']:04d}_seed_{i}"
            jobs.append(dict(target_index=target['target_index'], target=target['target'], repeat_index=i,
                             seed=target['noise_seeds'][i], model=args.model, noise_profile=args.noise_profile,
                             result_file=str(output / 'rollouts' / (stem + '.npz')),
                             log_file=str(output / 'logs' / (stem + '.log'))))
    config = dict(model=args.model, noise_profile=args.noise_profile, distinct_targets=len(targets),
                  planned=len(jobs), full_primary_benchmark=len(targets) == 500,
                  subset_not_population_result=len(targets) != 500,
                  shared_protocol_sha256=run_model.sha(ROOT / 'results/shared_test_targets.json'),
                  package_manifest_sha256=run_model.sha(ROOT / 'release_manifest.json'))
    (output / 'configuration.json').write_text(json.dumps(config, indent=2))
    rows = []

    def save(state):
        summary = dict(config, state=state, processed=len(rows),
                       successes=sum(bool(r['success']) for r in rows),
                       process_errors=sum(r['status'] == 'process_error' for r in rows),
                       unprocessed=len(jobs) - len(rows), percentage_is_final=len(rows) == len(jobs))
        summary['success_rate_percent'] = 100 * summary['successes'] / len(jobs)
        (output / 'results.json').write_text(json.dumps(rows, indent=2))
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        return summary

    save('running')
    print(f'Frozen batch: {len(jobs)} episodes; {output}', flush=True)
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                                 initializer=initialize, initargs=(args.model,)) as pool:
            for future in as_completed([pool.submit(episode, job) for job in jobs]):
                rows.append(future.result())
                if len(rows) % 10 == 0 or len(rows) == len(jobs):
                    summary = save('running')
                    print(f"[{len(rows)}/{len(jobs)}] successes={summary['successes']}; errors={summary['process_errors']}", flush=True)
        print(json.dumps(save('completed')), flush=True)
    except BaseException:
        save('interrupted')
        raise


if __name__ == '__main__':
    main()
