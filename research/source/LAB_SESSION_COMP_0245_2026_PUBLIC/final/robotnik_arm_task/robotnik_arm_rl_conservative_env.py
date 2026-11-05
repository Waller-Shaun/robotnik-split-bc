"""[RL V2 environment] 只在带噪观测的近区探索；未到近区继续由冻结基线实际运动。"""
import math
import numpy as np
from robotnik_arm_rl_env import FrozenContext, FineReachEnv as BaseEnv
from robotnik_arm_rl_conservative_policy import near_gate


class FineReachEnv(BaseEnv):
    def __init__(self,context,target,seed,control_repeat=10,limit_nm=.3,fine_seconds=6.,
                 warmup_seconds=20.,hold_seconds=.2,discount_tau_s=10.,sensor_noise='config',
                 near_inner_m=.02,near_outer_m=.05):
        self.inner,self.outer=near_inner_m,near_outer_m
        self.discount_tau_s=discount_tau_s
        super().__init__(context,target,seed,control_repeat,limit_nm,fine_seconds,warmup_seconds,
                         hold_seconds,math.exp(-.001*control_repeat/discount_tau_s),sensor_noise)

    def gate(self):
        # 真值不参与门控；23:26是未标准化的底盘坐标末端位置误差。
        return near_gate(np.linalg.norm(self.frame['observation'][23:26]),self.inner,self.outer) if self.state.done else 0.

    def warmup(self):
        while self.steps*self.dt<self.warmup_seconds:
            if self.state.done and self.gate()>0.:
                return self.frame['normalized'].copy(),True
            self.physics_step(np.zeros(7))
        return self.frame['normalized'].copy(),False

    def step(self,proposal):
        if not self.state.done:raise RuntimeError('Navigation must complete first')
        a=np.clip(np.asarray(proposal,dtype=np.float32),-1,1)
        if a.shape!=(7,) or not np.isfinite(a).all():raise ValueError('Invalid residual proposal')
        before=self.frame['error'];integrated=0.;terminal=False;truncated=False;reason='running'
        square_sum=0.;active_steps=0;executed=0
        for _ in range(self.repeat):
            gate=self.gate();delta_nm=self.limit*a*gate
            self.physics_step(delta_nm);self.fine_steps+=1;executed+=1
            active_steps+=int(gate>0.);square_sum+=float(np.mean(delta_nm**2))
            error=self.frame['error'];self.hold=self.hold+self.dt if error<.01 else 0.
            # [V2 reward] 距离/保持的连续奖励；修正幅度惩罚按比例并按真实dt积分。
            integrated+=self.dt*(-10*min(error,1.)+(1. if error<.01 else 0.)
                                 -.2*float(np.mean((a*gate)**2))-.001*float(np.mean(self.frame['qd']**2)))
            if self.hold+1e-9>=self.hold_seconds:
                terminal=True;reason='true_target_held';break
            if error>.25:
                terminal=True;reason='fine_excursion_limit';break
            if self.fine_steps*self.dt>=self.fine_seconds-1e-9:
                truncated=True;reason='fine_time_limit';break
        discount=math.exp(-executed*self.dt/self.discount_tau_s)
        # [V2 shaping] 使用同一discount的势能差；Phi=-5*distance，避免奖励与折扣时标不一致。
        potential_error_after=0. if terminal else self.frame['error']
        reward=5*(before-discount*potential_error_after)+integrated
        if terminal:reward+=5. if reason=='true_target_held' else -5.
        info=dict(reason=reason,true_error_m=self.frame['error'],min_error_m=self.min_error,
                  simulation_time_s=self.steps*self.dt,fine_seconds=self.fine_steps*self.dt,
                  executed_physics_steps=executed,discount=discount,held_seconds=self.hold,
                  effective_delta_rms_nm=float(np.sqrt(square_sum/max(1,executed))),
                  gate_active_fraction=active_steps/max(1,executed))
        return self.frame['normalized'].copy(),float(reward),terminal,truncated,info
