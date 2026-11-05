"""[Recovery probe] 20个低高度训练目标的成对恢复诊断；不训练，不报正式成功率。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
from batch_test_bc_split_nav_arm_aux_b2_parallel import available_memory_bytes,cpu_busy_fraction
from recovery_probe_logic import select_training_targets

BASELINE='bc_training_arm_ring_r08_aux_b2_control/run_20261104_025706_206181'


def main():
    root=Path(__file__).resolve().parents[3]/'RoboEnv'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm-run-dir',type=Path,default=root/BASELINE)
    p.add_argument('--cases',type=int,default=20)
    p.add_argument('--workers',default='auto')
    p.add_argument('--sensor-noise',choices=['config','off'],default='config')
    p.add_argument('--policy-seconds',type=float,default=20.)
    p.add_argument('--recovery-seconds',type=float,default=10.)
    p.add_argument('--hold-seconds',type=float,default=1.)
    p.add_argument('--min-switch-time',type=float,default=8.)
    p.add_argument('--switch-at',type=float,help='Optional fixed-time diagnostic instead of automatic stall detection')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    if a.cases<1 or any(not np.isfinite(v) or v<=0 for v in [a.policy_seconds,a.recovery_seconds,a.hold_seconds]):p.error('Invalid times/case count')
    run=a.arm_run_dir.resolve()
    manifest=json.loads((run/'data_split.json').read_text(encoding='utf-8'))
    cases=select_training_targets(manifest,a.cases)
    mem=available_memory_bytes();busy=cpu_busy_fraction()
    slots=max(1,int((mem-1536*1024**2)//(900*1024**2))) if mem else 2
    workers=min(4,slots,max(1,int((os.cpu_count() or 1)*(1-busy))//2)) if a.workers=='auto' else int(a.workers)
    if workers<1:p.error('workers must be positive')
    workers=min(workers,len(cases))
    config={'diagnostic_only':True,'target_owner':'train','pure_model_success_rate':None,'cases':cases,
            'arm_run':str(run),'workers':workers,'sensor_noise':a.sensor_noise,'policy_seconds':a.policy_seconds,
            'recovery_seconds':a.recovery_seconds,'hold_seconds':a.hold_seconds,'min_switch_time':a.min_switch_time,
            'switch_at':a.switch_at,'available_memory_gib':mem/1024**3 if mem else None,'cpu_busy_fraction':busy,
            'training_data_generated':False,'expert_takeover_is_not_pure_model_success':True}
    if a.plan_only:
        print(json.dumps(config,indent=2));return
    output=(a.output_dir or run/'recovery_diagnostics'/datetime.now().strftime('train_probe_%Y%m%d_%H%M%S_%f')).resolve()
    output.mkdir(parents=True,exist_ok=False)
    (output/'logs').mkdir()
    script=Path(__file__).with_name('diagnose_arm_recovery.py')
    fixed=[run/n for n in ['best_model.pt','normalization.npz','config.json','data_split.json']]
    fixed+=[root/'configs/robotnik_arm.json',script,Path(__file__).with_name('robotnik_arm_control.py')]
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in fixed}
    config['source_sha256']=hashes
    (output/'diagnostic_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    rows=[];active={};lock=threading.Lock();stop=threading.Event();start=time.perf_counter()
    def stop_children():
        stop.set()
        with lock:children=list(active.values())
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:child.kill();child.wait()
    def one(index,case):
        if stop.is_set():return None
        if any(hashlib.sha256(f.read_bytes()).hexdigest()!=hashes[str(f)] for f in fixed):
            raise RuntimeError('Config/model changed during diagnostic')
        destination=output/f'case_{index:03d}'
        command=[sys.executable,'-B',str(script),'--arm-run-dir',str(run),'--target',*map(str,case['target']),
            '--seed',str(case['seed']),'--headless','--sensor-noise',a.sensor_noise,'--policy-seconds',str(a.policy_seconds),
            '--recovery-seconds',str(a.recovery_seconds),'--hold-seconds',str(a.hold_seconds),
            '--min-switch-time',str(a.min_switch_time),'--output-dir',str(destination)]
        if a.switch_at is not None:command+=['--switch-at',str(a.switch_at)]
        row={'index':index,**case,'status':'process_error','report':str(destination/'summary.json')}
        child=None
        try:
            with (output/'logs'/f'case_{index:03d}.log').open('w',encoding='utf-8') as f:
                with lock:
                    if stop.is_set():return None
                    child=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    active[index]=child
                code=child.wait(timeout=300)
            if code:raise RuntimeError(f'Child exit {code}; see log')
            report=json.loads((destination/'summary.json').read_text(encoding='utf-8'))
            if report['target_owner']!='train' or not report['diagnostic_only']:raise ValueError('Wrong diagnostic ownership')
            row.update(status='completed',category=report['category'],trigger=report['trigger'],branches=report['branches'])
        except Exception as exc:
            if child is not None and child.poll() is None:child.kill();child.wait()
            row['error']=f'{type(exc).__name__}: {exc}'
        finally:
            with lock:active.pop(index,None)
        return row
    def save():
        categories={}
        for r in rows:
            cat=r.get('category','process_error');categories[cat]=categories.get(cat,0)+1
        summary={'diagnostic_only':True,'planned_cases':len(cases),'processed_cases':len(rows),
                 'process_errors':sum(r['status']=='process_error' for r in rows),'categories':categories,
                 'pure_model_success_rate':None,'training_data_generated':False,'workers':workers,
                 'wall_seconds':time.perf_counter()-start,'results':sorted(rows,key=lambda r:r['index'])}
        temp=output/'summary.json.tmp';temp.write_text(json.dumps(summary,indent=2),encoding='utf-8');temp.replace(output/'summary.json')
        return summary
    print(f'DIAGNOSTIC ONLY: {len(cases)} TRAIN targets, workers={workers}; results={output}',flush=True)
    pool=ThreadPoolExecutor(max_workers=workers)
    try:
        futures=[pool.submit(one,i,case) for i,case in enumerate(cases,1)]
        for future in as_completed(futures):
            row=future.result()
            if row is not None:
                rows.append(row);save()
                print(f"[{len(rows)}/{len(cases)}] {row['target']} -> {row.get('category',row.get('error'))}",flush=True)
    except BaseException:
        stop_children()
        for f in futures:f.cancel()
        save();raise
    finally:pool.shutdown(wait=True,cancel_futures=True)
    summary=save()
    print(json.dumps(summary['categories'],indent=2));print('Process errors:',summary['process_errors'])
    print('Hybrid diagnostic only; NOT pure-model/generalization success.\nSummary:',output/'summary.json')


if __name__=='__main__':main()
