"""[RL environment] 无Gym依赖；基础反馈1kHz，小幅修正动作100Hz保持。

只使用训练目标交互；真实位置用于奖励与评分，不进入26维策略输入。
"""
from pathlib import Path
import numpy as np
import torch
import pinocchio as pin
from simulation_and_control import pb,MotorCommands,PinWrapper
from recovery_residual120_context import resolve_parent_pair,BASELINE
from robotnik_arm_aux_policy import load_policy_from_checkpoint
from robotnik_navigation_policy import parking_goal,build_navigation_observation,load_navigation_policy,NAV_PARAMS
from robotnik_bodyframe_features import build_bodyframe_observation
from robotnik_arm_control import BaseNavState,BaseControllerParams,base_controller,configure_gripper_fingers,quaternion2bearing


class FrozenContext:
    def __init__(self,root,parent=None):
        self.root=Path(root)
        self.parent,self.nav,self.pc,self.nc=resolve_parent_pair(root,parent or self.root/BASELINE)
        self.parent_cp=torch.load(self.parent/'best_model.pt',map_location='cpu',weights_only=True)
        nav_cp=torch.load(self.nav/'best_model.pt',map_location='cpu',weights_only=True)
        if self.parent_cp['config']!=self.pc or self.parent_cp['epoch']!=138 or nav_cp['config']!=self.nc:
            raise ValueError('Wrong frozen parent/navigation checkpoints')
        self.base=load_policy_from_checkpoint(self.parent_cp).eval();self.navigation=load_navigation_policy(nav_cp).eval()
        for m in (self.base,self.navigation):
            for p in m.parameters():p.requires_grad_(False)
        with np.load(self.parent/'normalization.npz') as n:self.stats={k:n[k].copy() for k in n.files}
        with np.load(self.nav/'normalization.npz') as n:self.ns={k:n[k].copy() for k in n.files}

    def actions(self,frame):
        with torch.inference_mode():
            base=self.base(torch.from_numpy(frame['normalized']).unsqueeze(0))[0].numpy()*self.stats['y_std']+self.stats['y_mean']
            n=self.ns
            wheels=self.navigation(torch.from_numpy((frame['navigation']-n['x_mean'])/n['x_std']).unsqueeze(0))[0].numpy()*n['y_std']+n['y_mean']
        return wheels,base[4:11]


class FineReachEnv:
    def __init__(self,context,target,seed,control_repeat=10,limit_nm=2.,fine_seconds=6.,
                 warmup_seconds=12.,hold_seconds=.2,gamma=.99,sensor_noise='config'):
        self.context=context;self.target=np.asarray(target,dtype=float);self.repeat=control_repeat
        self.limit=limit_nm;self.fine_seconds=fine_seconds;self.warmup_seconds=warmup_seconds
        self.hold_seconds=hold_seconds;self.gamma=gamma;self.seed=int(seed)
        self.sim=None;np.random.seed(self.seed)
        try:
            self.sim=pb.SimInterface('robotnik_arm.json',conf_file_path_ext=str(context.root),use_gui=False)
            self.client=self.sim.pybullet_client;self.rid=self.sim.bot[0].bot_pybullet
            if sensor_noise=='off':
                for b in self.sim.bot:b.noise_flag=False
            elif not all(b.noise_flag for b in self.sim.bot):raise ValueError('Config sensor noise must be enabled')
            self.sim.SetFloorFriction(1.)
            for j in range(self.client.getNumJoints(self.rid)):
                if 'wheel' in self.client.getJointInfo(self.rid,j)[1].decode():
                    self.client.changeDynamics(self.rid,j,contactStiffness=30000,contactDamping=1000,anisotropicFriction=[1.,.01,1.])
            self.sim.bot[0].servo_motor_model.set_motor_gains(400,30)
            configure_gripper_fingers(self.sim,self.client,self.rid,np.full(6,.1))
            names=np.expand_dims(np.array(self.sim.getNameActiveJoints()),0)
            self.dyn=PinWrapper('robotnik_arm.json','pybullet',names,['pybullet'],False,0,str(context.root))
            self.ee_id=self.dyn.pin_model.getFrameId('robot_j2s7s300_end_effector')
            links=[j for j in range(self.client.getNumJoints(self.rid)) if self.client.getJointInfo(self.rid,j)[12].decode()=='robot_j2s7s300_end_effector']
            if len(links)!=1:raise ValueError('EE link missing')
            self.ee_link=links[0];self.dt=self.sim.GetTimeStep()
            if abs(self.dt-.001)>1e-9:raise ValueError('This prototype is configured for1ms physics')
            self.state=BaseNavState();self.params=BaseControllerParams(**NAV_PARAMS)
            self.goal,self.bearing_goal=parking_goal(self.target)
            self.steps=0;self.fine_steps=0;self.hold=0.;self.cmd=MotorCommands()
            self.frame=self.observe();self.initial_error=self.frame['error'];self.min_error=self.initial_error
        except BaseException:
            self.close();raise

    def close(self):
        if self.sim is not None:
            self.client.disconnect();self.sim=None

    def observe(self):
        base=np.array(self.sim.GetBasePosition());ori=np.array(self.sim.GetBaseOrientation())
        yaw=quaternion2bearing(ori[3],ori[0],ori[1],ori[2])
        q=np.array(self.sim.GetMotorAngles(0));qd=np.array(self.sim.GetMotorVelocities(0))
        qp=self.dyn.ReoderJoints2PinVec(np.concatenate([base,ori,q]),'pos')
        pin.forwardKinematics(self.dyn.pin_model,self.dyn.pin_data,qp);pin.updateFramePlacements(self.dyn.pin_model,self.dyn.pin_data)
        ee=self.dyn.pin_data.oMf[self.ee_id].translation.copy()
        truth=np.array(self.client.getLinkState(self.rid,self.ee_link,computeForwardKinematics=True)[4])
        error=float(np.linalg.norm(self.target-truth))
        # 原专家仅提供阶段规则；其轮速返回值一律丢弃，动作不调用专家机械臂。
        base_controller(self.state,base,yaw,self.goal,self.bearing_goal,self.steps*self.dt,self.params)
        src=np.concatenate([base,[np.sin(yaw),np.cos(yaw)],q[4:11],qd[4:11],self.target,[0. if self.state.done else 1.]]).astype(np.float32)
        obs=build_bodyframe_observation(src,ee)
        normalized=((obs-self.context.stats['x_mean'])/self.context.stats['x_std']).astype(np.float32)
        if not np.isfinite(normalized).all() or not np.isfinite(error):raise RuntimeError('Nonfinite simulation state')
        nav=build_navigation_observation(base,yaw,self.goal,self.bearing_goal,self.state)
        return {'normalized':normalized,'observation':obs,'navigation':nav,'truth':truth,'error':error,
            'qd':qd[4:11].copy(),'done':self.state.done}

    def physics_step(self,delta_nm):
        wheels,arm=self.context.actions(self.frame)
        action=np.clip(np.concatenate([wheels,arm+delta_nm]),-np.array([3.]*4+[30.]*7),np.array([3.]*4+[30.]*7))
        if not np.isfinite(action).all():raise RuntimeError('Nonfinite simulation action')
        self.cmd.SetControlCmd(np.concatenate([action,np.zeros(6)]),['velocity']*4+['torque']*13)
        self.sim.Step(self.cmd,'torque');self.steps+=1;self.frame=self.observe()
        self.min_error=min(self.min_error,self.frame['error'])
        return action

    def warmup(self):
        # 只用120版和原学习导航达到fine阶段，不直接设置关节角或把底盘强制停下。
        while not self.state.done and self.steps*self.dt<self.warmup_seconds:
            self.physics_step(np.zeros(7))
        return self.frame['normalized'].copy(),self.state.done

    def step(self,normalized_action):
        if not self.state.done:raise RuntimeError('Fine controller cannot run before navigation done')
        a=np.clip(np.asarray(normalized_action,dtype=np.float32),-1,1)
        if a.shape!=(7,) or not np.isfinite(a).all():raise ValueError('TD3 action must be finite7-vector')
        delta_nm=self.limit*a;before=self.frame['error'];running=0.;terminal=False;truncated=False;reason='running'
        executed=0
        for _ in range(self.repeat):
            self.physics_step(delta_nm);self.fine_steps+=1;executed+=1
            error=self.frame['error']
            self.hold=self.hold+self.dt if error<.01 else 0.
            # [RL reward] 真值仅作训练奖励，按物理时间积分，避免改频率就改奖励尺度。
            running+=self.dt*(-10*min(error,1.)+(1. if error<.01 else 0.)
                -.01*float(np.mean(delta_nm**2))-.001*float(np.mean(self.frame['qd']**2)))
            if self.hold+1e-9>=self.hold_seconds:
                terminal=True;reason='true_target_held';running+=5.;break
            if error>1.0:
                terminal=True;reason='diverged';running-=5.;break
            if self.fine_steps*self.dt>=self.fine_seconds-1e-9:
                truncated=True;reason='fine_time_limit';break
        reward=5*(before-self.frame['error'])+running
        # 半个控制块终止时使用实际物理时长折算discount。
        discount=self.gamma**(executed/self.repeat)
        info={'reason':reason,'true_error_m':self.frame['error'],'min_error_m':self.min_error,
            'fine_seconds':self.fine_steps*self.dt,'simulation_time_s':self.steps*self.dt,
            'executed_physics_steps':executed,'discount':discount,'held_seconds':self.hold}
        return self.frame['normalized'].copy(),float(reward),terminal,truncated,info
