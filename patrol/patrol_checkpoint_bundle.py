"""[Portable release] Load original frozen checkpoints without historical drive paths.

Only file location/dependency resolution changes. Both MLPs retain all original
state-dict keys, normalization and forward computations. No training is performed.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


ARM_HASH='751fe1d4d9310ee79ccb36d9d78db820ae930dd8145146938b22b5295534d86d'
NAV_HASH='110e547e46b8bec3deb2b83b5142294be2f60562aabbbd8aecd843bb4ca07aba'


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PublishedArmPolicy(nn.Module):
    """Same 26→256→256→128→11 path; saved auxiliary head is unused at inference."""
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(26,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),
                                   nn.Linear(256,128),nn.ReLU(),nn.Linear(128,11))
        self.aux_head=nn.Linear(128,7)
    def forward(self,x):return self.network(x)


class PublishedNavigationPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(8,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,4))
    def forward(self,x):return self.network(x)


def _stats(folder,nx,ny):
    with np.load(folder/'normalization.npz',allow_pickle=False) as data:
        result={k:data[k].copy() for k in ('x_mean','x_std','y_mean','y_std')}
    for k,n in [('x_mean',nx),('x_std',nx),('y_mean',ny),('y_std',ny)]:
        if result[k].shape!=(n,) or not np.isfinite(result[k]).all():raise ValueError('Invalid normalization '+k)
        if k.endswith('std') and np.any(result[k]<=0):raise ValueError('Nonpositive normalization '+k)
    return result


def load_patrol_bundle(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    expected={f'{folder}/{name}' for folder in ('arm120','navigation')
              for name in ('best_model.pt','normalization.npz','config.json')}|{'configs/robotnik_arm.json'}
    if manifest.get('schema')!='patrol120_portable_original_bytes_v1' or set(manifest['files'])!=expected:
        raise ValueError('Wrong or incomplete patrol120 bundle')
    for rel,digest in manifest['files'].items():
        if sha256(root/rel)!=digest:raise ValueError('Bundle checksum mismatch: '+rel)
    if sha256(root/'arm120/best_model.pt')!=ARM_HASH or sha256(root/'navigation/best_model.pt')!=NAV_HASH:
        raise ValueError('Expected original recovery120 and frozen navigation weights')
    arm_checkpoint=torch.load(root/'arm120/best_model.pt',map_location='cpu',weights_only=True)
    nav_checkpoint=torch.load(root/'navigation/best_model.pt',map_location='cpu',weights_only=True)
    ac=arm_checkpoint['config'];nc=nav_checkpoint['config']
    if (ac!=json.loads((root/'arm120/config.json').read_text(encoding='utf-8'))
            or nc!=json.loads((root/'navigation/config.json').read_text(encoding='utf-8'))):
        raise ValueError('Checkpoint/config mismatch')
    if (ac.get('input_dim')!=26 or ac.get('hidden_dims')!=[256,256,128] or ac.get('output_dim')!=11
            or ac.get('aux_weight')!=0 or not ac.get('recovery_finetune')
            or ac.get('input_feature_version')!='obs23_target_and_ee_error_base_yaw_frame_v1'
            or nc.get('input_dim')!=8 or nc.get('hidden_dims')!=[64,64] or nc.get('output_dim')!=4
            or nc.get('input_feature_version')!='parking_body_xy_signed_heading_errors_phase4_v1'):
        raise ValueError('Unexpected frozen policy architecture/feature version')
    for name in ('robotnik_arm_control.py','robotnik_bodyframe_features.py','robotnik_navigation_policy.py'):
        if sha256(Path(__file__).with_name(name))!=ac['inference_module_sha256'][name]:
            raise ValueError('Frozen inference helper changed: '+name)
    arm=PublishedArmPolicy();nav=PublishedNavigationPolicy()
    arm.load_state_dict(arm_checkpoint['model_state_dict'],strict=True)
    nav.load_state_dict(nav_checkpoint['model_state_dict'],strict=True)
    for model in (arm,nav):
        if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite weights')
        model.eval().requires_grad_(False)
    return arm,nav,_stats(root/'arm120',26,11),_stats(root/'navigation',8,4),root/'arm120',root/'navigation'
