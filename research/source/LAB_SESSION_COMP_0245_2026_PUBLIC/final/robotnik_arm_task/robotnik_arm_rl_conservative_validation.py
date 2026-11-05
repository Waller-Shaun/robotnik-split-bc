"""[RL V2 validation] 固定验证点、无动作探索、真实1cm评分；从不写入训练回放。

这里只服务训练时选模型和早停；完整测试仍由独立batch入口执行。
"""
import contextlib
from dataclasses import dataclass
import json
import numpy as np
import torch
from recovery_dataset import sha256
from robotnik_arm_rl_conservative_env import FineReachEnv
from robotnik_arm_rl_conservative_policy import ResidualController


def choose_cases(manifest,count,seed=991):
    targets=manifest['targets']['val']
    low=[i for i,t in enumerate(targets) if t[2]<.5]
    other=[i for i,t in enumerate(targets) if t[2]>=.5]
    rng=np.random.default_rng(seed)
    nl=count//2;nh=count-nl
    if not low or not other or nl>len(low) or nh>len(other):raise ValueError('Validation count exceeds height groups')
    selected=sorted([*rng.choice(low,nl,replace=False).tolist(),*rng.choice(other,nh,replace=False).tolist()])
    return [dict(index=i,target=targets[i],seed=manifest['evaluation_seeds']['val'][i]) for i in selected]


def evaluate(context,actor,config,cases,log):
    """冻结actor在固定目标上实际运行；真实位置仅评分，输入/门控仍为观测。"""
    rows=[]
    for number,case in enumerate(cases,1):
        if sha256(context.root/'configs'/'robotnik_arm.json')!=config['robot_config_sha256']:
            raise RuntimeError('Robot config changed during validation')
        env=None
        try:
            with contextlib.redirect_stdout(log):
                env=FineReachEnv(context,case['target'],case['seed'],
                    control_repeat=config['control_repeat'],limit_nm=config['residual_limit_nm'],
                    sensor_noise=config['sensor_noise_mode'],near_inner_m=config['near_inner_m'],near_outer_m=config['near_outer_m'])
            control=ResidualController(actor,context.stats,config['residual_limit_nm'],config['control_repeat'],
                                       config['near_inner_m'],config['near_outer_m'])
            max_steps=int(round(config['validation_duration_s']/env.dt))
            success=False;minimum=env.frame['error'];delta_square=0.;actions=0
            for step in range(max_steps+1):
                error=env.frame['error'];minimum=min(minimum,error)
                if error<.01:success=True;break
                if step==max_steps:break
                with torch.inference_mode():
                    delta=control.delta(torch.from_numpy(env.frame['normalized']).unsqueeze(0)).numpy()
                delta_square+=float(np.mean(delta**2));actions+=1
                env.physics_step(delta)
            row=dict(index=case['index'],target=case['target'],seed=case['seed'],success=success,
                     min_error_m=minimum,final_error_m=env.frame['error'],end_time_s=env.steps*env.dt,
                     residual_rms_nm=float(np.sqrt(delta_square/max(1,actions))))
            rows.append(row)
            print(f'  validation {number}/{len(cases)}: target#{case["index"]} success={success} error={row["final_error_m"]:.5f}m',flush=True)
        finally:
            if env is not None:env.close()
    return dict(cases=rows,successes=sum(r['success'] for r in rows),count=len(rows),
                mean_final_error_m=float(np.mean([r['final_error_m'] for r in rows])))


@dataclass
class Selection:
    """必须保留固定小验证组中所有baseline成功点；再比成功数、平均最终误差。"""
    baseline: dict
    best: dict
    min_error_improvement_m: float=.001
    bad_checks: int=0

    def consider(self,result):
        baseline_success={r['index'] for r in self.baseline['cases'] if r['success']}
        current_success={r['index'] for r in result['cases'] if r['success']}
        lost=sorted(baseline_success-current_success)
        improved=(not lost and (result['successes']>self.best['successes'] or
            (result['successes']==self.best['successes'] and
             result['mean_final_error_m']<self.best['mean_final_error_m']-self.min_error_improvement_m)))
        if improved:self.best=result;self.bad_checks=0
        else:self.bad_checks+=1
        return improved,lost
