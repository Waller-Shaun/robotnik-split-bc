"""Paired fixed/dynamic evaluation, never online training or target filtering."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout,redirect_stderr
from datetime import datetime,timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import numpy as np
from .common import ROOT,BASE,sha,write,check_baseline,load_fixed


def files():
    paths = [ROOT/'runs/full/best_model.pt',ROOT/'runs/full/config.json',ROOT/'protocol.json']
    paths += list((ROOT/'dynamic_pd').glob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def model_hash(controller):
    import hashlib
    h = hashlib.sha256()
    for label in ('model','nav','gain_model'):
        if not hasattr(controller,label):
            continue
        for key,value in getattr(controller,label).state_dict().items():
            h.update((label+key).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def initialize():
    import torch
    from .controller import DynamicPDController
    global CONTROLLERS,HASHES
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    CONTROLLERS = dict(fixed=load_fixed(),dynamic=DynamicPDController())
    HASHES = {name:model_hash(c) for name,c in CONTROLLERS.items()}


def episode(job):
    import torch
    from rollout_noisy import rollout,NOISY_CONFIG
    controller = CONTROLLERS[job['variant']]
    row = {k:job[k] for k in ('variant','target_index','repeat_index','seed','height_bin','target')}
    row.update(success=False,status='process_error',error='',rollout_file=job['rollout_file'])
    start = time.perf_counter()
    try:
        if model_hash(controller)!=HASHES[job['variant']]:
            raise ValueError('Weights changed between episodes')
        controller.reset()
        if len(controller.history) or controller.q_ref is not None or controller.q_hold is not None:
            raise ValueError('Reference/history not reset')
        if job['variant']=='dynamic' and (controller.gain_tick or len(controller.gain_trace)):
            raise ValueError('Gain filter/log not reset')
        np.random.seed(job['seed'])
        torch.manual_seed(job['seed'])
        args = SimpleNamespace(target=job['target'],seed=job['seed'],duration=20.,headless=True,
            sensor_noise='config',sim_config=NOISY_CONFIG,roboenv_dir=BASE/'external/RoboEnv',
            result_file=Path(job['rollout_file']))
        controller.provenance.update(evaluation_stage=job['stage'],target_index=job['target_index'],
            noise_repeat_index=job['repeat_index'],experiment_protocol_sha256=sha(ROOT/'protocol.json'),
            episode_memory_reset=True,model_weights_unchanged=True)
        with Path(job['log_file']).open('w') as stream:
            import sys
            sys.stdout.flush();sys.stderr.flush()
            out,err = os.dup(1),os.dup(2)
            try:
                os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
                with redirect_stdout(stream),redirect_stderr(stream):
                    rollout(args,controller)
                stream.flush()
            finally:
                os.dup2(out,1);os.dup2(err,2);os.close(out);os.close(err)
        result = json.loads(args.result_file.with_suffix('.json').read_text())
        with np.load(args.result_file,allow_pickle=False) as raw:
            errors = np.linalg.norm(raw['ee_positions']-raw['target'],axis=1)
            np.testing.assert_allclose(errors,raw['ee_errors'],rtol=0,atol=1e-14)
            if bool((errors<.01).any())!=result['success']:
                raise ValueError('True-state trajectory does not support success')
            # Lossless scoring/action arrays, lower-frequency diagnostic state
            # only. This changes storage, never the live control or scoring.
            compact = {key:raw[key].copy() for key in raw.files
                       if key not in ('observations','ground_truth_states')}
            diagnostic_indices = np.arange(0,len(raw['observations']),10)
            compact['diagnostic_sample_indices'] = diagnostic_indices
            compact['observations_10ms'] = raw['observations'][diagnostic_indices].copy()
            compact['ground_truth_states_10ms'] = raw['ground_truth_states'][diagnostic_indices].copy()
        temporary = args.result_file.with_suffix('.compact.npz')
        np.savez_compressed(temporary,**compact)
        with np.load(temporary,allow_pickle=False) as stored:
            np.testing.assert_array_equal(stored['ee_errors'],compact['ee_errors'])
            np.testing.assert_array_equal(stored['actions'],compact['actions'])
        temporary.replace(args.result_file)
        if not all(result['sensor_noise_enabled']) or result['ground_truth_exposed_to_policy']:
            raise ValueError('Noise or ground truth protocol differs')
        if result['reach_tolerance_m']!=.01 or result['duration_limit_s']!=20 or result['required_hold_s']!=0:
            raise ValueError('Success protocol changed')
        if model_hash(controller)!=HASHES[job['variant']]:
            raise ValueError('Inference modified weights')
        row.update(success=result['success'],status='success' if result['success'] else 'episode_failure',
            min_error_m=result['minimum_error_m'],final_error_m=result['final_error_m'],
            end_time_s=result['end_time_s'],clipped_fraction=result['clipped_fraction'],
            phase_switch_time_s=result['phase_switch_time_s'])
        if job['variant']=='dynamic':
            row['gain_summary'] = controller.gain_summary()
            # Keep a compact explicit gain trace without rewriting base NPZs.
            gain_file = args.result_file.with_suffix('.gains.npz')
            np.savez_compressed(gain_file,trace=np.asarray(controller.gain_trace,dtype=np.float64),
                                columns=np.array(['time_s']+[f'p_{j+1}' for j in range(7)]+[f'd_{j+1}' for j in range(7)]+[f'raw_p_{j+1}' for j in range(7)]+[f'raw_d_{j+1}' for j in range(7)]))
            write(args.result_file.with_suffix('.gains.json'),row['gain_summary'])
    except BaseException:
        row.update(success=False,status='process_error',error=traceback.format_exc())
    row['wall_s'] = time.perf_counter()-start
    return row


def run(stage,workers,resume):
    check_baseline()
    protocol = json.loads((ROOT/'protocol.json').read_text())
    trained = json.loads((ROOT/'runs/full/training_finished.json').read_text())
    if trained['state']!='completed' or trained['subset_only']:
        raise ValueError('Full training incomplete')
    frozen = files()
    selection_path = ROOT/'selection.json'
    if stage=='final':
        selection = json.loads(selection_path.read_text())
        if selection['frozen_files']!=frozen:
            raise ValueError('Candidate changed after selection')
        if not selection['candidate_trained_and_frozen_before_final']:
            raise ValueError('No pre-final selection')
    output = ROOT/'outputs'/stage
    output.mkdir(parents=True,exist_ok=resume)
    (output/'logs').mkdir(exist_ok=resume)
    (output/'rollouts').mkdir(exist_ok=resume)
    repeats = 3 if stage=='final' else 1
    jobs = []
    for entry in protocol[stage]:
        for repeat in range(repeats):
            for variant in ('fixed','dynamic'):
                stem = f'{variant}_{entry["target_index"]:04d}_{repeat}'
                jobs.append(dict(entry,stage=stage,variant=variant,repeat_index=repeat,
                    seed=entry['noise_seeds'][repeat],
                    rollout_file=str(output/'rollouts'/(stem+'.npz')),
                    log_file=str(output/'logs'/(stem+'.log'))))
    configuration = dict(stage=stage,planned=len(jobs),repeats=repeats,target_count=len(protocol[stage]),
                         frozen_files=frozen,protocol_sha256=sha(ROOT/'protocol.json'),noise='friend_0p001',
                         storage='true scoring positions/errors and actions at 1ms; diagnostic observations/states at 10ms')
    if resume:
        if json.loads((output/'configuration.json').read_text())!=configuration:
            raise ValueError('Resume configuration differs')
    else:
        write(output/'configuration.json',configuration)
    rows = json.loads((output/'results.json').read_text()) if resume and (output/'results.json').exists() else []
    completed = {(r['variant'],r['target_index'],r['repeat_index']) for r in rows}
    pending = [j for j in jobs if (j['variant'],j['target_index'],j['repeat_index']) not in completed]
    started = time.perf_counter()

    def update(state):
        summary = dict(stage=stage,state=state,planned=len(jobs),processed=len(rows),
            process_errors=sum(r['status']=='process_error' for r in rows),wall_s=time.perf_counter()-started,
            variants={v:dict(successes=sum(r['success'] for r in rows if r['variant']==v),
                processed=sum(r['variant']==v for r in rows),planned=len(jobs)//2,
                success_rate_percent=100*sum(r['success'] for r in rows if r['variant']==v)/(len(jobs)//2)) for v in ('fixed','dynamic')})
        write(output/'results.json',rows);write(output/'summary.json',summary)
        return summary

    update('running')
    print(f'PAIRED {stage}: {len(jobs)} episodes, workers={workers}, pending={len(pending)}',flush=True)
    try:
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
            for future in as_completed([pool.submit(episode,j) for j in pending]):
                row = future.result();rows.append(row)
                if row['status']=='process_error':
                    print('PROCESS ERROR: '+row['error'],flush=True)
                if len(rows)%10==0 or len(rows)==len(jobs):
                    if files()!=frozen:
                        raise ValueError('Candidate changed during evaluation')
                    check_baseline()
                    summary = update('running')
                    print(f'[{len(rows)}/{len(jobs)}] '+json.dumps(summary['variants']),flush=True)
        summary = update('completed')
    except BaseException:
        update('interrupted');raise
    if stage=='development':
        pairs = {v:{r['target_index']:r['success'] for r in rows if r['variant']==v} for v in ('fixed','dynamic')}
        write(selection_path,dict(candidate_trained_and_frozen_before_final=True,
            checkpoint_sha256=sha(ROOT/'runs/full/best_model.pt'),frozen_files=frozen,
            created_utc=datetime.now(timezone.utc).isoformat(),final_outcomes_seen=False,
            offline_validation_selected_epoch=trained['best_epoch'],development=summary,
            paired_improvements=sum(pairs['dynamic'][i] and not pairs['fixed'][i] for i in pairs['fixed']),
            paired_regressions=sum(pairs['fixed'][i] and not pairs['dynamic'][i] for i in pairs['fixed']),
            final_comparison_is_experiment_not_automatic_deployment=True))
    print('COMPLETE '+json.dumps(summary),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('development','final'))
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--resume',action='store_true')
    a = p.parse_args()
    if not 1<=a.workers<=8:
        p.error('workers must be 1..8')
    run(a.stage,a.workers,a.resume)


if __name__=='__main__':
    main()
