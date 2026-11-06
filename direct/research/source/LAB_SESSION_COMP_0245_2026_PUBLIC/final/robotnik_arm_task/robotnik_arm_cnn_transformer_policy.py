"""[CNN-T change] 顺序历史→因果1D CNN→Transformer→当前7维力矩。

推理模块不导入训练入口，不调用PD/IK/专家动作，不叠加旧MLP。
"""
from collections import deque
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from recovery_dataset import FROZEN, sha256
from recovery_residual120_context import resolve_parent_pair
from robotnik_bodyframe_features import BODYFRAME_FEATURE_VERSION

VARIANT='supervised_causal_cnn_transformer_arm_torque_v1'
HISTORY_VERSION='causal_26d_oldest_to_current_repeat_episode_first_v1'
RUN_ROOT='bc_training_arm_ring_r08_cnn_transformer'
CNN_RUN_ROOT='bc_training_arm_ring_r08_cnn_history'


class CausalConv1d(nn.Module):
    """[CNN-T causal] 仅左侧填充；时间t的卷积特征不使用t之后的帧。"""
    def __init__(self,channels_in,channels_out,kernel_size=3):
        super().__init__()
        self.conv=nn.Conv1d(channels_in,channels_out,kernel_size,padding=0)
        self.left_padding=kernel_size-1

    def forward(self,x):return self.conv(F.pad(x,(self.left_padding,0)))


class HistoryArmPolicy(nn.Module):
    def __init__(self,history_length=16,architecture='cnn_transformer',dropout=.1):
        super().__init__()
        if history_length<1 or architecture not in ('cnn_transformer','cnn'):raise ValueError('Invalid architecture')
        self.history_length=history_length;self.architecture=architecture
        self.cnn=nn.Sequential(CausalConv1d(26,64),nn.GELU(),CausalConv1d(64,128),nn.GELU())
        self.embedding_norm=nn.LayerNorm(128)
        if architecture=='cnn_transformer':
            self.position=nn.Parameter(torch.empty(1,history_length,128))
            layer=nn.TransformerEncoderLayer(d_model=128,nhead=4,dim_feedforward=256,
                dropout=dropout,activation='gelu',batch_first=True,norm_first=True)
            self.encoder=nn.TransformerEncoder(layer,num_layers=2,norm=nn.LayerNorm(128),enable_nested_tensor=False)
            self.head=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,7))
        else:
            self.register_parameter('position',None);self.encoder=None
            # CNN对照也读取全部16帧：均值特征＋当前特征；不隐藏为只看最后5帧。
            self.head=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,7))
        self.register_buffer('causal_mask',torch.triu(torch.ones(history_length,history_length,dtype=torch.bool),diagonal=1),persistent=False)
        # TransformerEncoder复制的层默认初值相同；这里逐个参数独立初始化。
        for name,p in self.named_parameters():
            if name=='position':nn.init.normal_(p,std=.02)
            elif p.ndim>1:nn.init.xavier_uniform_(p)
            elif name.endswith('bias'):nn.init.zeros_(p)
            elif name.endswith('weight'):nn.init.ones_(p)

    def encode(self,history):
        if history.ndim!=3 or history.shape[1:]!=(self.history_length,26):
            raise ValueError(f'Expected (batch,{self.history_length},26) history')
        # Conv1d的channel是26个状态特征，长度轴才是时间；绝不沿关节编号当时间卷积。
        features=self.embedding_norm(self.cnn(history.transpose(1,2)).transpose(1,2))
        if self.encoder is not None:
            features=self.encoder(features+self.position,mask=self.causal_mask)
        return features

    def forward(self,history):
        features=self.encode(history)
        representation=features[:,-1] if self.encoder is not None else torch.cat([features[:,-1],features.mean(1)],-1)
        return self.head(representation)


class CausalObservationHistory:
    def __init__(self,length,stride):
        if type(length) is not int or type(stride) is not int or min(length,stride)<1:raise ValueError('Invalid history dimensions')
        self.length,self.stride=length,stride
        self.buffer=deque(maxlen=(length-1)*stride+1)

    def append(self,observation):
        x=np.asarray(observation,dtype=np.float32)
        if x.shape!=(26,) or not np.isfinite(x).all():raise ValueError('Expected finite normalized26D frame')
        self.buffer.append(x.copy());states=list(self.buffer)
        return np.stack([states[max(0,len(states)-1-offset)] for offset in range((self.length-1)*self.stride,-1,-self.stride)])


def load_policy_from_checkpoint(checkpoint):
    c=checkpoint['config']
    if (c.get('experiment_variant')!=VARIANT or c.get('input_dim')!=26 or c.get('output_dim')!=7
            or c.get('input_feature_version')!=BODYFRAME_FEATURE_VERSION or c.get('history_feature_version')!=HISTORY_VERSION
            or c.get('pretrained_weights_loaded') is not False or c.get('baseline_residual_branch') is not False
            or c.get('navigation_weights_updated') is not False or c.get('expert_references_used_as_input') is not False):
        raise ValueError('Wrong direct CNN/Transformer torque checkpoint')
    model=HistoryArmPolicy(c['history_length'],c['architecture'],c['dropout'])
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite weights')
    return model


def resolve_finetuned_pair(roboenv,arm_run=None,nav_run=None):
    if arm_run is None:
        choices=[]
        for p in (Path(roboenv)/RUN_ROOT).glob('run_*'):
            if (p/'best_model.pt').exists() and (p/'training_finished.json').exists():
                c=json.loads((p/'config.json').read_text(encoding='utf-8'))
                if c.get('experiment_variant')==VARIANT and not c.get('smoke_only',True):choices.append(p)
        if not choices:raise FileNotFoundError('Train CNN-Transformer separately first')
        arm_run=sorted(choices)[-1]
    arm_run=Path(arm_run).resolve();c=json.loads((arm_run/'config.json').read_text(encoding='utf-8'))
    if c.get('experiment_variant')!=VARIANT:raise ValueError('Wrong experiment')
    parent,nav,_,nc=resolve_parent_pair(roboenv,c['source_model_run_dir'],nav_run)
    if str(nav)!=c['nav_run_dir']:raise ValueError('Navigation binding changed')
    for field,folder in [('source_model_file_sha256',parent),('frozen_nav_file_sha256',nav)]:
        if c[field]!={n:sha256(folder/n) for n in FROZEN}:raise ValueError('Frozen context changed')
    for n in ('normalization.npz','data_split.json'):
        if sha256(parent/n)!=sha256(arm_run/n):raise ValueError('Split/stats changed')
    for n,h in c['inference_module_sha256'].items():
        if sha256(Path(__file__).with_name(n))!=h:raise ValueError('Inference code changed: '+n)
    return arm_run,nav,c,nc
