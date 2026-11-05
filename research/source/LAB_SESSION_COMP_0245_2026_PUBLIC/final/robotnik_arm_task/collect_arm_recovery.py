"""[Recovery change] 只采集：模型停滞→专家从当前状态接管→导出恢复标签；不训练。

复用已通过配对检查的diagnose_arm_recovery，专家与模型续跑都保存。
默认扫描全部低高度TRAIN目标，不读取val/test失败清单。
"""
import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
from recovery_dataset import BASELINE, export_diagnostics, read_json


def main():
    root = Path(__file__).resolve().parents[3]/'RoboEnv'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm-run-dir', type=Path, default=root/BASELINE)
    p.add_argument('--scope', choices=['low', 'all'], default='low', help='low: world target z<0.5m; all: all TRAIN targets')
    p.add_argument('--cases', default='all', help='all or a positive count (deterministic TRAIN subset)')
    p.add_argument('--workers', default='auto')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--policy-seconds', type=float, default=20.)
    p.add_argument('--recovery-seconds', type=float, default=10.)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--reuse-diagnostics', type=Path, help='Export an existing completed paired diagnostic; no new simulation')
    a = p.parse_args()
    parent = a.arm_run_dir.resolve()
    split = read_json(parent/'data_split.json')
    indices = [i for i, t in enumerate(split['targets']['train']) if a.scope == 'all' or t[2] < .5-1e-6]
    count = len(indices) if a.cases == 'all' else int(a.cases)
    if not 1 <= count <= len(indices):
        p.error('cases must be between 1 and eligible TRAIN target count')
    # [Recovery sampling] 只根据train坐标分散取样，不能根据验证集成败挑训练目标。
    order = np.random.default_rng(42).permutation(indices).tolist()[:count]
    cases = [{'train_index':i, 'target':split['targets']['train'][i], 'seed':30042+i} for i in order]
    if a.plan_only and a.reuse_diagnostics:
        s = read_json(a.reuse_diagnostics/'summary.json')
        print('Export only; categories:', s['categories']); return
    import batch_diagnose_arm_recovery as paired
    # 仅替换本进程的选点函数，原诊断文件和阈值均不改动。
    paired.select_training_targets = lambda manifest, n: cases
    output = (a.output_dir or root/'expert_recovery_data'/datetime.now().strftime('collection_%Y%m%d_%H%M%S_%f')).resolve()
    if a.reuse_diagnostics:
        if output.exists(): raise FileExistsError(output)
        diagnostics = a.reuse_diagnostics.resolve()
    else:
        diagnostics = output/'diagnostics'
        argv = ['--arm-run-dir', str(parent), '--cases', str(count), '--workers', a.workers,
            '--sensor-noise', 'config', '--policy-seconds', str(a.policy_seconds),
            '--recovery-seconds', str(a.recovery_seconds), '--hold-seconds', '1',
            '--output-dir', str(diagnostics)]
        if a.plan_only: argv += ['--plan-only']
        old = sys.argv
        try:
            sys.argv = [str(Path(paired.__file__)), *argv]
            paired.main()
        finally:
            sys.argv = old
        if a.plan_only:
            print('Scope:', a.scope, '; eligible targets:', len(indices), '; new data directory:', output)
            return
    manifest = export_diagnostics(diagnostics, parent, output/'episodes')
    print('Recovery export finished. Expert-only trajectories:', manifest['trajectories'], '; frames:', manifest['frames'])
    print('Manifest:', output/'episodes/recovery_manifest.json')
    print('Collection only. Run finetune_arm_recovery.py separately; no model updated.')


if __name__ == '__main__':
    main()
