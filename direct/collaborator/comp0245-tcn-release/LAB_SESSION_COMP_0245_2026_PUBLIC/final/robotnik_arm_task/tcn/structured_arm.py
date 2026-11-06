"""Learned movement field and gravity, with explicit joint-reference memory.

This is a HYBRID structured neural controller, not end-to-end torque TCN.
No expert IK, Jacobian, inverse dynamics or gravity oracle is used online.
The final torques use learned velocity/gravity plus the disclosed fixed
100/30 joint tracking law and 1 ms reference integrator from the data.
"""
import numpy as np
import torch
from torch import nn
from tcn_model import CausalConv1d

FEATURE_VERSION='sin_cos_q_velocity_local_goal_phase_error_base_gravity_v1'


def features(observations, quaternion):
    x=np.asarray(observations,dtype=np.float32)
    quat=np.asarray(quaternion,dtype=np.float32)
    quat=quat/np.linalg.norm(quat,axis=-1,keepdims=True)
    a,b,c,d=np.moveaxis(quat,-1,0)
    gravity=np.stack((2*(d*b-a*c),-2*(b*c+d*a),-(1-2*(a*a+b*b))),axis=-1)
    result=np.concatenate((np.sin(x[...,5:12]),np.cos(x[...,5:12]),x[...,12:19],
                           x[...,19:22],x[...,22:23],x[...,23:26],gravity),axis=-1)
    if result.shape[-1]!=31 or not np.isfinite(result).all():
        raise ValueError('Invalid structured feature vector')
    return np.ascontiguousarray(result,dtype=np.float32)


class StructuredArm(nn.Module):
    def __init__(self,kind='mlp'):
        super().__init__()
        self.kind=kind
        if kind=='mlp':
            self.encoder=nn.Sequential(nn.Linear(31,256),nn.ReLU(),nn.Linear(256,256),
                                       nn.ReLU(),nn.Linear(256,128),nn.ReLU())
        elif kind=='tcn':
            self.encoder=nn.Sequential(CausalConv1d(31,128,3,1),nn.ReLU(),
                                       CausalConv1d(128,128,3,2),nn.ReLU(),
                                       CausalConv1d(128,128,3,4),nn.ReLU())
        else:
            raise ValueError(kind)
        self.head=nn.Linear(128,28)

    def forward(self,history):
        if self.kind=='mlp':
            hidden=self.encoder(history[:,-1])
        else:
            hidden=self.encoder(history.transpose(1,2))[:,:,-1]
        return self.head(hidden)


def decode(output,current,g_mean,g_std):
    matrix=15*torch.tanh(output[:,:21].reshape(-1,7,3)/15)
    error=current[:,25:28]
    error=error*torch.clamp(.2/error.norm(dim=1,keepdim=True).clamp_min(1e-8),max=1.)
    velocity=(matrix@error.unsqueeze(-1)).squeeze(-1)
    velocity=velocity/velocity.abs().amax(dim=1,keepdim=True).clamp_min(1.)
    velocity=velocity*(1-current[:,24:25])
    gravity=output[:,21:]*g_std+g_mean
    return velocity,gravity
