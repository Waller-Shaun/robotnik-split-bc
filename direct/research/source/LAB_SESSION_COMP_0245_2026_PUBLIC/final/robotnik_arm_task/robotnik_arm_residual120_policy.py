# [Residual120 change] 新父模型独立实验；修正网络从零输出开始，不载入113残差网络。
"""[Residual change] 冻结第138轮120/144恢复微调策略，另学26→64→64→7的力矩修正。

输入沿用标准化26维观测；修正以N*m限幅，再除以原y_std转回标准化动作空间。
前4维旧动作不改、部署不用；真实轮速仍来自独立冻结导航模型。
"""
from pathlib import Path
import json

import numpy as np
import torch
from robotnik_arm_aux_policy import (
    AuxiliaryArmPolicy, load_policy_from_checkpoint as load_parent_policy,
    FROZEN_FILES, sha256,
)

from recovery_residual120_context import resolve_parent_pair

VARIANT = 'frozen_recovery120_mlp26_bounded_arm_residual_v1'
RUN_ROOT = 'bc_training_arm_ring_r08_residual120'


class ResidualArmPolicy(torch.nn.Module):
    def __init__(self, base_hidden_dims, arm_std, residual_hidden_dims=(64,64), residual_limit_nm=5.):
        super().__init__()
        self.base = AuxiliaryArmPolicy(base_hidden_dims)
        for p in self.base.parameters():p.requires_grad_(False)
        self.base.eval()
        layers=[];dims=[26,*residual_hidden_dims,7]
        for i,(n,m) in enumerate(zip(dims,dims[1:])):
            layers.append(torch.nn.Linear(n,m))
            if i<len(dims)-2:layers.append(torch.nn.ReLU())
        self.residual=torch.nn.Sequential(*layers)
        # [Residual initialization] 最后层为0：任何输入的组合策略都从原baseline精确起步。
        torch.nn.init.zeros_(self.residual[-1].weight)
        torch.nn.init.zeros_(self.residual[-1].bias)
        std=torch.as_tensor(arm_std,dtype=torch.float32)
        if std.shape!=(7,) or not torch.isfinite(std).all() or not torch.all(std>0):
            raise ValueError('Expected positive finite7-dimensional arm y_std')
        if not 0<float(residual_limit_nm)<=30:raise ValueError('Residual limit must be in (0,30] N*m')
        self.register_buffer('arm_std',std.clone())
        self.register_buffer('limit_nm',torch.tensor(float(residual_limit_nm),dtype=torch.float32))

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # 原模型始终eval，禁止训练模式影响基础策略。
        return self

    def correction_nm(self,x):
        return self.limit_nm*torch.tanh(self.residual(x))

    def correction_normalized(self,x):
        return self.correction_nm(x)/self.arm_std

    def forward(self,x):
        with torch.no_grad():base=self.base(x)
        correction=self.correction_normalized(x)
        return torch.cat([base[...,:4],base[...,4:11]+correction],dim=-1)


def build_from_parent(checkpoint, stats, residual_hidden_dims=(64,64), residual_limit_nm=5.):
    # [Residual120 change] 明确冻结已验证的第138轮，拒绝把旧113版或last错当起点。
    if checkpoint['epoch']!=138 or not checkpoint['config'].get('recovery_finetune'):
        raise ValueError('Expected120/144 recovery parent best epoch138')
    parent=load_parent_policy(checkpoint)
    model=ResidualArmPolicy(checkpoint['config']['hidden_dims'],stats['y_std'][4:11],
        residual_hidden_dims,residual_limit_nm)
    model.base.load_state_dict(parent.state_dict(),strict=True)
    return model


def load_policy_from_checkpoint(checkpoint):
    c=checkpoint['config']
    if (c.get('experiment_variant')!=VARIANT or c.get('input_dim')!=26 or c.get('output_dim')!=11
            or c.get('residual_output_dim')!=7 or c.get('base_parameters_frozen') is not True
            or c.get('navigation_weights_updated') is not False or c.get('gru_used') is not False
            or c.get('expert_references_used_as_input') is not False or c.get('aux_weight')!=0):
        raise ValueError('Expected frozen-baseline residual checkpoint')
    model=ResidualArmPolicy(c['hidden_dims'],c['arm_action_std'],c['residual_hidden_dims'],c['residual_limit_nm'])
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite residual parameters')
    if not torch.equal(model.arm_std,torch.tensor(c['arm_action_std'],dtype=torch.float32)):
        raise ValueError('Residual unit conversion buffer/config mismatch')
    if abs(model.limit_nm.item()-c['residual_limit_nm'])>1e-6:raise ValueError('Residual bound/config mismatch')
    parent_cp=torch.load(Path(c['source_model_run_dir'])/'best_model.pt',map_location='cpu',weights_only=True)
    if parent_cp['epoch']!=138 or not parent_cp['config'].get('recovery_finetune'):
        raise ValueError('Frozen120 parent checkpoint changed')
    assert_base_unchanged(model,parent_cp)
    return model


def assert_base_unchanged(model, parent_checkpoint):
    expected=parent_checkpoint['model_state_dict']
    actual=model.base.state_dict()
    if actual.keys()!=expected.keys() or any(not torch.equal(v.detach().cpu(),expected[k].cpu()) for k,v in actual.items()):
        raise ValueError('Frozen base parameters changed')
    if any(p.requires_grad for p in model.base.parameters()):raise ValueError('Base has trainable parameters')


def resolve_finetuned_pair(roboenv, arm_run=None, nav_run=None):
    if arm_run is None:
        choices=[]
        for p in (Path(roboenv)/RUN_ROOT).glob('run_*'):
            if (p/'training_finished.json').is_file() and (p/'best_model.pt').is_file():
                c=json.loads((p/'config.json').read_text(encoding='utf-8'))
                if c.get('experiment_variant')==VARIANT and not c.get('smoke_only',True):choices.append(p)
        if not choices:raise FileNotFoundError('Train a residual run separately first, or specify --arm-run-dir')
        arm_run=sorted(choices)[-1]
    arm_run=Path(arm_run).resolve();c=json.loads((arm_run/'config.json').read_text(encoding='utf-8'))
    if (c.get('experiment_variant')!=VARIANT or c.get('training_mode')!='frozen_base_arm_residual_navigation_frozen'
            or not c.get('base_parameters_frozen') or c.get('navigation_weights_updated') is not False):
        raise ValueError('Wrong residual experiment schema')
    parent,nav,pc,nc=resolve_parent_pair(roboenv,c['source_model_run_dir'],nav_run)
    if pc['aux_weight']!=0 or str(nav)!=c['nav_run_dir']:raise ValueError('Wrong baseline/navigation binding')
    for field,folder in [('source_model_file_sha256',parent),('frozen_nav_file_sha256',nav)]:
        if set(c[field])!=set(FROZEN_FILES):raise ValueError('Incomplete source hashes')
        for n,h in c[field].items():
            if sha256(folder/n)!=h:raise ValueError(f'Frozen source changed: {folder/n}')
    for n in ('normalization.npz','data_split.json'):
        if sha256(arm_run/n)!=sha256(parent/n):raise ValueError(f'Inherited {n} changed')
    with np.load(arm_run/'normalization.npz') as stats:
        if not np.array_equal(np.asarray(c['arm_action_std'],dtype=np.float32),stats['y_std'][4:11]):
            raise ValueError('Residual physical unit conversion differs from inherited action normalization')
    for n,h in c['inference_module_sha256'].items():
        if sha256(Path(__file__).with_name(n))!=h:raise ValueError(f'Residual inference implementation changed: {n}')
    return arm_run,nav,c,nc
