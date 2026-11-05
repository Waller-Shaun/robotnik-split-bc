"""[Recovery change] 只测试：全程学习模型，绝不调用专家；复用基线的带噪真实位置评估。"""
from pathlib import Path
import sys

from recovery_dataset import RUN_ROOT, read_json, sha256, FROZEN


def main():
    root=Path(__file__).resolve().parents[3]/'RoboEnv'
    if '--arm-run-dir' not in sys.argv:
        choices=[]
        for d in (root/RUN_ROOT).glob('run_*'):
            if (d/'training_finished.json').is_file():
                c=read_json(d/'config.json')
                if c.get('recovery_finetune') and not c.get('smoke_only',True):choices.append(d)
        if not choices:raise FileNotFoundError('No completed non-smoke recovery finetune; train separately first')
        sys.argv+=['--arm-run-dir',str(sorted(choices)[-1])]
    run=Path(sys.argv[sys.argv.index('--arm-run-dir')+1]).resolve()
    c=read_json(run/'config.json')
    if not c.get('recovery_finetune') or c.get('aux_weight')!=0:raise ValueError('Expected torque-only recovery finetune')
    parent=Path(c['recovery_parent_run_dir'])
    if c['recovery_parent_file_sha256']!={n:sha256(parent/n) for n in FROZEN}:raise ValueError('Recovery parent changed')
    # 原评估器接受显式目录，检查相同归一化/底盘绑定；这里只增加新实验选择入口。
    from batch_test_bc_split_nav_arm_aux_b2_parallel import main as evaluate
    evaluate()


if __name__=='__main__':main()
