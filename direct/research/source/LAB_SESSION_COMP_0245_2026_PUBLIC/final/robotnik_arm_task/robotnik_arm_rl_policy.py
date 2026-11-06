"""[RL change] TD3训练小幅机械臂残差；冻结120版MLP和独立导航。

参考TD3三项机制：双Q最小目标、延迟actor更新、目标动作平滑。
部署仅用actor，不用critic；不输入真实末端或专家动作。
"""
import copy
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn

from recovery_residual120_context import BASELINE, resolve_parent_pair
from recovery_dataset import FROZEN, sha256
from robotnik_arm_aux_policy import load_policy_from_checkpoint as load_base_policy

VARIANT='TD3_frozen_recovery120_arm_residual_v1'
RUN_ROOT='bc_training_arm_ring_r08_rl_td3'


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(26,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,7),nn.Tanh())
        nn.init.zeros_(self.network[-2].weight);nn.init.zeros_(self.network[-2].bias)

    def forward(self,x):return self.network(x)


class TwinCritic(nn.Module):
    def __init__(self):
        super().__init__()
        def make():return nn.Sequential(nn.Linear(33,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
        self.q1,self.q2=make(),make()

    def forward(self,s,a):
        z=torch.cat([s,a],dim=-1)
        return self.q1(z).squeeze(-1),self.q2(z).squeeze(-1)


class Replay:
    def __init__(self,capacity):
        self.s=np.empty((capacity,26),np.float32);self.a=np.empty((capacity,7),np.float32)
        self.next_s=np.empty_like(self.s);self.r=np.empty(capacity,np.float32)
        self.discount=np.empty(capacity,np.float32);self.terminal=np.empty(capacity,np.float32)
        self.capacity=capacity;self.size=0;self.pointer=0

    def add(self,s,a,r,next_s,discount,terminal):
        i=self.pointer
        for name,value in [('s',s),('a',a),('r',r),('next_s',next_s),('discount',discount),('terminal',terminal)]:getattr(self,name)[i]=value
        self.pointer=(i+1)%self.capacity;self.size=min(self.size+1,self.capacity)

    def sample(self,n,rng,device):
        ix=rng.integers(self.size,size=n)
        return [torch.as_tensor(getattr(self,k)[ix],device=device) for k in ('s','a','r','next_s','discount','terminal')]


class TD3:
    def __init__(self,device='cpu',actor_lr=1e-4,critic_lr=3e-4,penalty=.001):
        self.device=torch.device(device);self.actor=Actor().to(self.device);self.critic=TwinCritic().to(self.device)
        self.actor_target=copy.deepcopy(self.actor);self.critic_target=copy.deepcopy(self.critic)
        for m in (self.actor_target,self.critic_target):
            for p in m.parameters():p.requires_grad_(False)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.updates=0;self.penalty=penalty

    def update(self,batch):
        s,a,r,sp,discount,terminal=batch
        with torch.no_grad():
            noise=(torch.randn_like(a)*.1).clamp(-.2,.2)
            ap=(self.actor_target(sp)+noise).clamp(-1,1)
            q1,q2=self.critic_target(sp,ap)
            # [RL terminal] 时间上限为truncated仍bootstrap；真正成功/失控terminal才清除后续价值。
            target=r+discount*(1-terminal)*torch.minimum(q1,q2)
        q1,q2=self.critic(s,a)
        critic_loss=(q1-target).square().mean()+(q2-target).square().mean()
        if not torch.isfinite(critic_loss):raise RuntimeError('Nonfinite critic loss')
        self.critic_optimizer.zero_grad(set_to_none=True);critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(),10.);self.critic_optimizer.step()
        self.updates+=1;actor_loss=None
        if self.updates%2==0:
            for p in self.critic.parameters():p.requires_grad_(False)
            action=self.actor(s)
            actor_loss=-self.critic.q1(torch.cat([s,action],dim=-1)).mean()+self.penalty*action.square().mean()
            if not torch.isfinite(actor_loss):raise RuntimeError('Nonfinite actor loss')
            self.actor_optimizer.zero_grad(set_to_none=True);actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),10.);self.actor_optimizer.step()
            for p in self.critic.parameters():p.requires_grad_(True)
            with torch.no_grad():
                for online,target_net in ((self.actor,self.actor_target),(self.critic,self.critic_target)):
                    for p,tp in zip(online.parameters(),target_net.parameters()):tp.mul_(.995).add_(p,alpha=.005)
        return {'critic_loss':float(critic_loss.detach().cpu()),
            'actor_loss':float(actor_loss.detach().cpu()) if actor_loss is not None else None}


class DeploymentPolicy(nn.Module):
    """[RL deployment] 兼容原1kHz测试：每10步更新一次残差，基础反馈每步重算。"""
    def __init__(self,base,actor,stats,limit,repeat):
        super().__init__();self.base=base.eval();self.actor=actor.eval()
        for p in self.parameters():p.requires_grad_(False)
        self.register_buffer('arm_std',torch.tensor(stats['y_std'][4:11],dtype=torch.float32))
        self.register_buffer('phase_mean',torch.tensor(float(stats['x_mean'][22])))
        self.register_buffer('phase_std',torch.tensor(float(stats['x_std'][22])))
        self.limit=limit;self.repeat=repeat;self.ticks=0;self.held=None
        self.last_correction_nm=torch.zeros(7)

    def forward(self,x):
        if x.ndim!=2 or len(x)!=1:raise ValueError('Stateful100Hz deployment requires one chronological observation per call')
        result=self.base(x)
        active=bool((x[0,22]*self.phase_std+self.phase_mean)<.5)
        if not active:
            self.held=None;self.ticks=0;self.last_correction_nm=torch.zeros(7,device=x.device)
            return result
        if self.held is None or self.ticks%self.repeat==0:self.held=self.actor(x).detach()
        self.ticks+=1;delta=self.held*self.limit
        self.last_correction_nm=delta[0].clone()
        return torch.cat([result[:,:4],result[:,4:11]+delta/self.arm_std],dim=-1)


def load_policy_from_checkpoint(checkpoint):
    c=checkpoint['config']
    if (c.get('experiment_variant')!=VARIANT or c.get('input_dim')!=26
            or c.get('base_parameters_frozen') is not True or c.get('critic_used_at_inference') is not False
            or c.get('true_state_used_as_policy_input') is not False):raise ValueError('Wrong TD3 checkpoint')
    p=Path(c['source_model_run_dir'])
    parent=torch.load(p/'best_model.pt',map_location='cpu',weights_only=True)
    if parent['epoch']!=138:raise ValueError('TD3 parent must be120 best138')
    base=load_base_policy(parent);actor=Actor();actor.load_state_dict(checkpoint['actor_state_dict'],strict=True)
    if not all(torch.isfinite(v).all() for v in actor.state_dict().values()):raise ValueError('Nonfinite actor')
    with np.load(p/'normalization.npz') as n:stats={k:n[k].copy() for k in n.files}
    return DeploymentPolicy(base,actor,stats,c['residual_limit_nm'],c['control_repeat'])


def resolve_finetuned_pair(roboenv,arm_run=None,nav_run=None):
    if arm_run is None:
        choices=[]
        for p in (Path(roboenv)/RUN_ROOT).glob('run_*'):
            if (p/'training_finished.json').exists() and (p/'best_model.pt').exists():
                c=json.loads((p/'config.json').read_text(encoding='utf-8'))
                if c.get('experiment_variant')==VARIANT and not c.get('smoke_only',True):choices.append(p)
        if not choices:raise FileNotFoundError('Run TD3 training separately first')
        arm_run=sorted(choices)[-1]
    arm_run=Path(arm_run).resolve();c=json.loads((arm_run/'config.json').read_text(encoding='utf-8'))
    if c.get('experiment_variant')!=VARIANT:raise ValueError('Not a TD3 residual run')
    parent,nav,pc,nc=resolve_parent_pair(roboenv,c['source_model_run_dir'],nav_run)
    if str(nav)!=c['nav_run_dir']:raise ValueError('Navigation binding changed')
    for field,folder in [('source_model_file_sha256',parent),('frozen_nav_file_sha256',nav)]:
        if c[field]!={n:sha256(folder/n) for n in FROZEN}:raise ValueError('Frozen source changed')
    for n in ('normalization.npz','data_split.json'):
        if sha256(arm_run/n)!=sha256(parent/n):raise ValueError('Inherited stats/split changed')
    for n,h in c['inference_module_sha256'].items():
        if sha256(Path(__file__).with_name(n))!=h:raise ValueError(f'TD3 inference implementation changed: {n}')
    return arm_run,nav,c,nc
