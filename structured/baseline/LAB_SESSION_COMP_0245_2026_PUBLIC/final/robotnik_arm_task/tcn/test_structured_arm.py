"""Evaluate hybrid neural reference-memory policy, without expert arm control."""
from collections import deque
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from test_tcn import ROOT,rollout
from check_reference_mlp import NAV_RUN,load_stats,sha256
from robotnik_navigation_policy import load_navigation_policy,build_navigation_observation
from structured_arm import StructuredArm,features,decode,FEATURE_VERSION


class StructuredController:
    def __init__(self,run_dir,checkpoint='best_model.pt'):
        self.run_dir=Path(run_dir).resolve()
        if Path(checkpoint).name!=checkpoint or not checkpoint.endswith('.pt'):
            raise ValueError('Checkpoint must be a local .pt filename')
        self.checkpoint_path=self.run_dir/checkpoint
        ckpt=torch.load(self.checkpoint_path,map_location='cpu',weights_only=True)
        self.config=ckpt['config']
        if ckpt['schema_version']!=3 or self.config['feature_version']!=FEATURE_VERSION:
            raise ValueError('Wrong structured policy schema')
        self.model_source_path=Path(__file__).with_name('structured_arm.py')
        if sha256(self.model_source_path)!=self.config['source_sha256'][self.model_source_path.name]:
            raise ValueError('Model source changed')
        self.model=StructuredArm(ckpt['model_type']).eval()
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.stats={k:v.numpy().copy() for k,v in ckpt['normalization'].items()}
        with np.load(self.run_dir/'normalization.npz') as data:
            for key,value in self.stats.items():
                np.testing.assert_array_equal(value,data[key])
                if not np.isfinite(value).all() or (key.endswith('std') and np.any(value<=0)):
                    raise ValueError('Invalid normalization')
        self.g_mean=torch.from_numpy(self.stats['g_mean'])
        self.g_std=torch.from_numpy(self.stats['g_std'])
        self.nav_stats=load_stats(NAV_RUN,8,4)
        for name,expected in self.config['frozen_nav_file_sha256'].items():
            if sha256(NAV_RUN/name)!=expected:raise ValueError('Frozen nav changed')
        nav=torch.load(NAV_RUN/'best_model.pt',map_location='cpu',weights_only=True)
        self.nav=load_navigation_policy(nav).eval()
        self.manifest=json.loads((self.run_dir/'data_split.json').read_text())
        if sha256(self.run_dir/'data_split.json')!=self.config['source_manifest_sha256']:
            raise ValueError('Split changed')
        urdf=ROOT/'external/RoboEnv/models/summit_xl_arm_description/summit_xl_complete.urdf'
        tree=ET.parse(urdf)
        self.lower=np.full(7,-np.inf);self.upper=np.full(7,np.inf)
        for index in range(7):
            joint=tree.find(f"./joint[@name='robot_j2s7s300_joint_{index+1}']")
            if joint.get('type')!='continuous':
                self.lower[index]=float(joint.find('limit').get('lower'))+.05
                self.upper[index]=float(joint.find('limit').get('upper'))-.05
        self.epoch=ckpt['epoch']
        self.val_mse=ckpt['validation']['teacher_forced_torque_mse']
        self.window_size,self.stride=20,10
        self.policy_type=f'structured_neural_{ckpt["model_type"]}_reference_memory_PD'
        self.provenance=dict(experiment='structured_hybrid_neural_arm',is_hybrid_controller=True,
             evaluator_sha256=sha256(Path(__file__)),expert_ik_during_inference=False,
             expert_gravity_during_inference=False,recorded_expert_references_used_online=False,
             internal_reference='integral of predicted velocities, initialized from measured joint position',
             fixed_tracking_gains={'kp':100.,'kd':30.},joint_limits_sha256=sha256(urdf),
             success_threshold_changed=False,navigation_unchanged=True)
        self.reset()

    def reset(self):
        self.history=deque(maxlen=191)
        self.q_ref=None;self.q_hold=None

    def predict(self,observation,**context):
        raw=features(observation,context['base_orientation'])
        self.history.append(raw)
        frames=np.asarray(self.history)
        history=frames[np.maximum(len(frames)-1-np.arange(19,-1,-1)*10,0)]
        x=np.ascontiguousarray((history-self.stats['x_mean'])/self.stats['x_std'])
        nav_x=build_navigation_observation(context['base_position'],context['bearing'],
                     context['goal_position'],context['goal_bearing'],context['phase'])
        with torch.inference_mode():
            velocity,gravity=decode(self.model(torch.from_numpy(x)[None]),torch.from_numpy(raw)[None],self.g_mean,self.g_std)
            velocity,gravity=velocity[0].numpy(),gravity[0].numpy()
            wheels=self.nav(torch.from_numpy((nav_x-self.nav_stats['x_mean'])/self.nav_stats['x_std'])[None])[0].numpy()
        q=np.asarray(observation[5:12],dtype=np.float64)
        qd=np.asarray(observation[12:19],dtype=np.float64)
        if self.q_ref is None:
            self.q_hold=q.copy();self.q_ref=q.copy()
        if observation[22]>.5:
            self.q_ref=self.q_hold.copy();velocity=np.zeros(7)
        else:
            blocked=((q<=self.lower)&(velocity<0))|((q>=self.upper)&(velocity>0))
            velocity[blocked]=0
            self.q_ref=self.q_ref+.001*velocity
            self.q_ref=np.clip(self.q_ref,q-.15,q+.15)
            self.q_ref=np.clip(self.q_ref,self.lower,self.upper)
        # Explicit, disclosed hybrid tracking law. No IK/Jacobian/gravity oracle.
        torque=100*(self.q_ref-q)+30*(velocity-qd)+gravity
        action=np.r_[wheels*self.nav_stats['y_std']+self.nav_stats['y_mean'],torque]
        if not np.isfinite(action).all():raise FloatingPointError('Invalid neural policy action')
        return action


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--checkpoint',default='best_model.pt')
    p.add_argument('--val-index',type=int,default=0)
    p.add_argument('--target',type=float,nargs=3)
    p.add_argument('--seed',type=int)
    p.add_argument('--duration',type=float,default=20.)
    p.add_argument('--sensor-noise',choices=('off','config'),default='off')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--result-file',type=Path)
    args=p.parse_args()
    if not 0<=args.val_index<144 or not np.isfinite(args.duration) or args.duration<.001:
        p.error('Invalid index/duration')
    torch.set_num_threads(1)
    controller=StructuredController(args.run_dir,args.checkpoint)
    if args.target is None:args.target=controller.manifest['targets']['val'][args.val_index]
    if args.seed is None:args.seed=int(controller.manifest['evaluation_seeds']['val'][args.val_index])
    args.roboenv_dir=ROOT/'external/RoboEnv'
    np.random.seed(args.seed);torch.manual_seed(args.seed)
    print('HYBRID NEURAL CONTROL: learned gain field/gravity + reference memory + fixed PD; no expert IK online',flush=True)
    rollout(args,controller)


if __name__=='__main__':main()
