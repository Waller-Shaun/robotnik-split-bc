"""[Residual120 change] 区分冻结120版的训练起点和113版生成的恢复数据来源。

原恢复manifest及文件绝不改写；显式验证两者的划分/归一化一致，再复用原标签。
"""
from pathlib import Path

from robotnik_arm_aux_policy import resolve_finetuned_pair as resolve_aux_pair
from recovery_dataset import FROZEN, read_json, sha256, load_recovery as load_source_recovery

BASELINE = 'bc_training_arm_ring_r08_recovery/run_20261104_152933_829681'
GENERATOR_BASELINE = 'bc_training_arm_ring_r08_aux_b2_control/run_20261104_025706_206181'


def verify_parent_context(parent):
    parent=Path(parent).resolve();c=read_json(parent/'config.json')
    if (c.get('recovery_finetune') is not True or c.get('aux_weight')!=0
            or c.get('navigation_weights_updated') is not False or c.get('smoke_only',True)):
        raise ValueError('Residual120 requires the completed non-smoke recovery-finetuned MLP, not the113 baseline or previous residual model')
    root=parent.parents[1]
    if parent != (root/BASELINE).resolve():
        raise ValueError('This independent experiment is bound to the verified120/144 run')
    done=read_json(parent/'training_finished.json')
    if done['best_epoch']!=138 or done['smoke_only'] is not False:
        raise ValueError('Expected completed epoch138 recovery parent')
    generator=Path(c['recovery_parent_run_dir']).resolve()
    if generator != (root/GENERATOR_BASELINE).resolve():
        raise ValueError('Unexpected recovery collection-policy provenance')
    if c['recovery_parent_file_sha256']!={n:sha256(generator/n) for n in FROZEN}:
        raise ValueError('Original113 collection policy changed')
    for n in ('normalization.npz','data_split.json'):
        if sha256(parent/n)!=sha256(generator/n):raise ValueError(f'120 parent changed inherited {n}')
    return c,generator


def resolve_parent_pair(roboenv,arm_run=None,nav_run=None):
    parent=Path(arm_run).resolve() if arm_run is not None else (Path(roboenv)/BASELINE).resolve()
    verify_parent_context(parent)
    # 原B2解析器核验120版继承的祖先模型、相同输入归一化和冻结导航绑定。
    return resolve_aux_pair(roboenv,parent,nav_run)


def load_recovery(manifest_path,parent,split,stats):
    c,generator=verify_parent_context(parent)
    if sha256(manifest_path)!=c['recovery_data_manifest_sha256']:
        raise ValueError('Use the exact same recovery data as the120 parent experiment')
    # [Residual120 provenance] 读取时按113采集起点核验，不把manifest中的父模型伪改为120。
    # 新残差标签随后由120模型在同一观测上的输出重新计算，不能沿用113版差值。
    return load_source_recovery(manifest_path,generator,split,stats)
