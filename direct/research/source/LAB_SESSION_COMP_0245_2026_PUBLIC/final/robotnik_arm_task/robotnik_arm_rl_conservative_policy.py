"""[RL V2] 保守TD3副本：只在近目标区平滑介入，冻结120版与导航。

复用V1零初始化actor/回放结构；不加载失败的V1训练权重。
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from robotnik_arm_rl_policy import Actor, Replay, TD3 as BaseTD3
from recovery_residual120_context import resolve_parent_pair
from recovery_dataset import FROZEN, sha256
from robotnik_arm_aux_policy import load_policy_from_checkpoint as load_base

VARIANT = 'TD3_conservative_near_target_recovery120_v2'
RUN_ROOT = 'bc_training_arm_ring_r08_rl_td3_conservative'


def near_gate(distance, inner=.02, outer=.05):
    """[V2 gate] 2cm内全幅，5cm外严格零；中间smoothstep，非突变开关。"""
    fraction = float(np.clip((outer-float(distance))/(outer-inner), 0., 1.))
    return fraction*fraction*(3.-2.*fraction)


class ResidualController:
    """顺序单帧推理；初次进入近区才开始100Hz时钟，此后动作保持10步。

每1ms重新计算连续门控；离开近区立即归零，但不重置动作采样时钟。
门控只看观测末端误差，不读取PyBullet真值。
"""
    def __init__(self, actor, stats, limit=.3, repeat=10, inner=.02, outer=.05):
        self.actor = actor.eval()
        self.mean = torch.tensor(stats['x_mean'], dtype=torch.float32)
        self.std = torch.tensor(stats['x_std'], dtype=torch.float32)
        self.limit, self.repeat, self.inner, self.outer = limit, repeat, inner, outer
        self.started = False
        self.ticks = 0
        self.held = None
        self.last_gate = 0.

    def delta(self, x):
        if x.shape != (1,26): raise ValueError('Sequential CPU inference requires one 26D frame')
        raw = x[0]*self.std+self.mean
        active = bool(raw[22]<.5)
        self.last_gate = near_gate(float(torch.linalg.vector_norm(raw[23:26])), self.inner, self.outer) if active else 0.
        if not active:
            self.started=False; self.ticks=0; self.held=None
            return torch.zeros(7)
        if not self.started:
            if self.last_gate<=0.: return torch.zeros(7)
            self.started=True
        if self.held is None or self.ticks%self.repeat==0:
            self.held=self.actor(x).detach()[0]
        self.ticks+=1
        return self.held*self.limit*self.last_gate


class DeploymentPolicy(nn.Module):
    def __init__(self, base, actor, stats, config):
        super().__init__()
        self.base=base.eval(); self.actor=actor.eval()
        for p in self.parameters(): p.requires_grad_(False)
        self.register_buffer('arm_std',torch.tensor(stats['y_std'][4:11],dtype=torch.float32))
        self.controller=ResidualController(actor,stats,config['residual_limit_nm'],config['control_repeat'],
                                           config['near_inner_m'],config['near_outer_m'])
        self.last_correction_nm=torch.zeros(7)

    def forward(self,x):
        base=self.base(x)
        self.last_correction_nm=self.controller.delta(x)
        return torch.cat([base[:,:4],base[:,4:11]+self.last_correction_nm.unsqueeze(0)/self.arm_std],dim=-1)


class TD3(BaseTD3):
    """[V2 learner] critic先学习；actor延后启用，零修正正则增强。"""
    def __init__(self,device='cpu',actor_lr=2e-5,critic_lr=3e-4,penalty=.05):
        super().__init__(device,actor_lr,critic_lr,penalty)
        self.actor_updates=0

    def update(self,batch,allow_actor=True):
        s,a,r,sp,discount,terminal=batch
        with torch.no_grad():
            ap=(self.actor_target(sp)+(torch.randn_like(a)*.1).clamp(-.2,.2)).clamp(-1,1)
            q1,q2=self.critic_target(sp,ap)
            target=r+discount*(1-terminal)*torch.minimum(q1,q2)
        q1,q2=self.critic(s,a)
        critic_loss=(q1-target).square().mean()+(q2-target).square().mean()
        if not torch.isfinite(critic_loss): raise RuntimeError('Nonfinite critic loss')
        self.critic_optimizer.zero_grad(set_to_none=True); critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(),10.); self.critic_optimizer.step()
        self.updates+=1; actor_loss=None
        if self.updates%2==0:
            if allow_actor:
                for p in self.critic.parameters(): p.requires_grad_(False)
                proposal=self.actor(s)
                actor_loss=-self.critic.q1(torch.cat([s,proposal],-1)).mean()+self.penalty*proposal.square().mean()
                if not torch.isfinite(actor_loss): raise RuntimeError('Nonfinite actor loss')
                self.actor_optimizer.zero_grad(set_to_none=True); actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),10.); self.actor_optimizer.step()
                self.actor_updates+=1
                for p in self.critic.parameters(): p.requires_grad_(True)
            # critic预热时也更新target critic，actor及其target继续保持零初始化。
            with torch.no_grad():
                pairs=[(self.critic,self.critic_target)]
                if allow_actor: pairs.append((self.actor,self.actor_target))
                for online,target_net in pairs:
                    for p,tp in zip(online.parameters(),target_net.parameters()):tp.mul_(.995).add_(p,alpha=.005)
        return dict(critic_loss=float(critic_loss.detach().cpu()),
                    actor_loss=float(actor_loss.detach().cpu()) if actor_loss is not None else None)


def load_policy_from_checkpoint(cp):
    c=cp['config']
    if (c.get('experiment_variant')!=VARIANT or c.get('base_parameters_frozen') is not True
            or c.get('true_state_used_as_policy_input') is not False): raise ValueError('Wrong conservative checkpoint')
    parent=Path(c['source_model_run_dir'])
    base_cp=torch.load(parent/'best_model.pt',map_location='cpu',weights_only=True)
    if base_cp['epoch']!=138: raise ValueError('Expected frozen recovery120 epoch138')
    actor=Actor(); actor.load_state_dict(cp['actor_state_dict'],strict=True)
    if not all(torch.isfinite(v).all() for v in actor.state_dict().values()):raise ValueError('Nonfinite actor')
    with np.load(parent/'normalization.npz') as n:stats={k:n[k].copy() for k in n.files}
    return DeploymentPolicy(load_base(base_cp),actor,stats,c)


def resolve_finetuned_pair(roboenv,arm_run=None,nav_run=None):
    if arm_run is None:
        choices=[]
        for p in (Path(roboenv)/RUN_ROOT).glob('run_*'):
            if (p/'training_finished.json').exists() and (p/'best_model.pt').exists():
                c=json.loads((p/'config.json').read_text(encoding='utf-8'))
                if c.get('experiment_variant')==VARIANT and not c.get('smoke_only',True):choices.append(p)
        if not choices:raise FileNotFoundError('Train conservative TD3 separately first')
        arm_run=sorted(choices)[-1]
    arm_run=Path(arm_run).resolve();c=json.loads((arm_run/'config.json').read_text(encoding='utf-8'))
    if c.get('experiment_variant')!=VARIANT:raise ValueError('Not a conservative TD3 run')
    parent,nav,_,nc=resolve_parent_pair(roboenv,c['source_model_run_dir'],nav_run)
    if str(nav)!=c['nav_run_dir']:raise ValueError('Navigation binding changed')
    for field,folder in [('source_model_file_sha256',parent),('frozen_nav_file_sha256',nav)]:
        if c[field]!={n:sha256(folder/n) for n in FROZEN}:raise ValueError('Frozen source changed')
    for n in ('normalization.npz','data_split.json'):
        if sha256(parent/n)!=sha256(arm_run/n):raise ValueError('Inherited split/stats changed')
    for n,h in c['inference_module_sha256'].items():
        if sha256(Path(__file__).with_name(n))!=h:raise ValueError('Inference code changed: '+n)
    return arm_run,nav,c,nc
