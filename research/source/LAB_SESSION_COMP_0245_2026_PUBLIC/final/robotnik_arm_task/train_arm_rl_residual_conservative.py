"""[RL V2] 新建保守训练副本：近目标平滑门控、小力矩、物理时间折扣、定期验证选优。

冻结120版及导航，修正actor从零初始化。验证点只用于选优/早停，不进入经验回放。
正式测试单独运行batch_test_arm_rl_residual_conservative_parallel.py。
"""
import argparse
import contextlib
import csv
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from robotnik_arm_rl_conservative_policy import Actor, Replay, TD3, VARIANT, RUN_ROOT
from robotnik_arm_rl_conservative_env import FrozenContext, FineReachEnv
from robotnik_arm_rl_conservative_validation import choose_cases, evaluate, Selection
from recovery_dataset import FROZEN, sha256


def save_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roboenv',type=Path,default=Path(__file__).resolve().parents[3]/'RoboEnv')
    p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--max-episodes',type=int,default=500)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--replay-size',type=int,default=100000)
    p.add_argument('--learning-starts',type=int,default=1000)
    p.add_argument('--actor-start',type=int,default=3000,help='Critic learns first; actor stays exactly zero before this step')
    p.add_argument('--update-every',type=int,default=10)
    p.add_argument('--gradient-steps',type=int,default=10)
    p.add_argument('--actor-lr',type=float,default=2e-5)
    p.add_argument('--critic-lr',type=float,default=3e-4)
    p.add_argument('--actor-penalty',type=float,default=.05)
    p.add_argument('--residual-limit-nm',type=float,default=.3)
    p.add_argument('--exploration-std',type=float,default=.05,help='Normalized: .05 * .3 Nm = .015 Nm before gate')
    p.add_argument('--near-inner-m',type=float,default=.02)
    p.add_argument('--near-outer-m',type=float,default=.05)
    p.add_argument('--discount-tau-s',type=float,default=10.,help='discount = exp(-elapsed_physical_seconds / tau)')
    p.add_argument('--fine-seconds',type=float,default=6.)
    p.add_argument('--warmup-seconds',type=float,default=20.)
    p.add_argument('--hold-seconds',type=float,default=.2)
    p.add_argument('--low-target-fraction',type=float,default=.5)
    p.add_argument('--sensor-noise',choices=['config','off'],default='config')
    p.add_argument('--val-every',type=int,default=5000,help='Check at episode boundary after this many RL steps')
    p.add_argument('--val-count',type=int,default=8,help='Fixed balanced validation subset; NOT full validation success rate')
    p.add_argument('--patience',type=int,default=3,help='Consecutive validation checks without a new admissible best')
    p.add_argument('--min-steps',type=int,default=10000)
    p.add_argument('--val-duration',type=float,default=20.)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    for name in ('steps','max_episodes','batch_size','replay_size','update_every','gradient_steps','val_every','patience'):
        if getattr(a,name)<=0:p.error(name+' must be positive')
    for name in ('actor_lr','critic_lr','actor_penalty','residual_limit_nm','discount_tau_s','fine_seconds','warmup_seconds','hold_seconds','val_duration'):
        if not math.isfinite(getattr(a,name)) or getattr(a,name)<=0:p.error(name+' must be finite and positive')
    if not 0<a.near_inner_m<a.near_outer_m<=.1:p.error('Require 0 < near-inner < near-outer <= .1 m')
    if not math.isfinite(a.low_target_fraction) or not 0<a.low_target_fraction<1:p.error('low-target-fraction must be in (0,1)')
    if not math.isfinite(a.exploration_std) or not 0<=a.exploration_std<=.2:p.error('exploration-std must be in [0,.2]')
    if a.residual_limit_nm>.5:p.error('Conservative experiment caps residual limit at .5 Nm')
    if a.val_count<2 or a.val_count%2:p.error('val-count must be positive even, >=2')
    if (a.learning_starts<0 or a.actor_start<a.learning_starts or a.min_steps<0 or a.seed<0
            or a.replay_size<a.batch_size):p.error('Invalid learning/actor start, seed, minimum steps or replay size')
    if not a.smoke and a.steps<=a.actor_start:p.error('Formal steps must exceed actor-start')
    return a


def main():
    a=arguments();torch.set_num_threads(1);torch.manual_seed(a.seed)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    context=FrozenContext(a.roboenv)
    manifest=json.loads((context.parent/'data_split.json').read_text(encoding='utf-8'))
    train=manifest['targets']['train'];train_keys={tuple(t) for t in train}
    if any(train_keys & {tuple(t) for t in manifest['targets'][k]} for k in ('val','test')):raise ValueError('Split leakage')
    low=[i for i,t in enumerate(train) if t[2]<.5];other=[i for i,t in enumerate(train) if t[2]>=.5]
    cases=choose_cases(manifest,a.val_count)
    robot_path=a.roboenv/'configs'/'robotnik_arm.json'
    robot_hash=sha256(robot_path);robot=json.loads(robot_path.read_text(encoding='utf-8'))
    if a.sensor_noise=='config' and not all(robot['robot_pybullet']['noise_flag']):raise ValueError('Config noise is disabled')
    print(f'Conservative TD3: learner={device}; base/nav frozen CPU inference; fresh zero actor.',flush=True)
    print(f'Residual +/-{a.residual_limit_nm} Nm; full within {a.near_inner_m}m; ZERO beyond {a.near_outer_m}m.',flush=True)
    print(f'Discount tau={a.discount_tau_s}s; gamma10ms={math.exp(-.01/a.discount_tau_s):.7f}; actor start={a.actor_start}.',flush=True)
    print(f'Budget={a.steps}; fixed val subset={len(cases)}; interval={a.val_every}; patience={a.patience} checks; min steps={a.min_steps}.',flush=True)
    print('No TEST targets used. Validation rollouts are never put in replay. Initial zero actor remains a selectable baseline.',flush=True)
    if a.plan_only:return
    run=a.roboenv/RUN_ROOT/f'run_{datetime.now():%Y%m%d_%H%M%S_%f}'
    run.mkdir(parents=True,exist_ok=False);(run/'validation').mkdir()
    for n in ('normalization.npz','data_split.json'):shutil.copy2(context.parent/n,run/n)
    keys=('input_dim','source_input_dim','output_dim','hidden_dims','input_feature_version','dataset_variant',
          'base_active_column','target_position_frame','ee_error_frame')
    c={k:context.pc[k] for k in keys}
    modules=('robotnik_arm_rl_conservative_policy.py','robotnik_arm_rl_policy.py','recovery_residual120_context.py',
             'robotnik_arm_aux_policy.py','robotnik_bodyframe_features.py','robotnik_navigation_policy.py','robotnik_arm_control.py')
    c.update(dict(experiment_variant=VARIANT,source_model_run_dir=str(context.parent),source_model_epoch=138,
        nav_run_dir=str(context.nav),source_model_file_sha256={n:sha256(context.parent/n) for n in FROZEN},
        frozen_nav_file_sha256={n:sha256(context.nav/n) for n in FROZEN},base_parameters_frozen=True,
        navigation_weights_updated=False,aux_weight=0.,gru_used=False,expert_references_used_as_input=False,
        true_state_used_as_policy_input=False,critic_used_at_inference=False,exploration_used_at_test=False,
        residual_limit_nm=a.residual_limit_nm,control_repeat=10,near_inner_m=a.near_inner_m,near_outer_m=a.near_outer_m,
        gate_observation_columns=[23,24,25],gate_uses_true_state=False,gate_type='smoothstep_observed_distance',
        residual_activation='navigation_done_AND_observed_near_target_smooth_gate',
        sensor_noise_mode=a.sensor_noise,robot_config_path=str(robot_path),robot_config_sha256=robot_hash,robot_config_snapshot=robot,
        discount_time_constant_s=a.discount_tau_s,gamma_10ms=math.exp(-.01/a.discount_tau_s),
        actor_lr=a.actor_lr,critic_lr=a.critic_lr,actor_action_penalty=a.actor_penalty,actor_start=a.actor_start,
        learning_starts=a.learning_starts,exploration_std=a.exploration_std,batch_size=a.batch_size,
        replay_capacity=a.replay_size,update_every=a.update_every,gradient_steps=a.gradient_steps,
        total_transition_budget=a.steps,max_episodes=a.max_episodes,low_target_fraction=a.low_target_fraction,seed=a.seed,
        fine_time_limit_s=a.fine_seconds,warmup_time_limit_s=a.warmup_seconds,training_hold_s=a.hold_seconds,
        reward='time_integrated_distance_and_hold_with_normalized_correction_penalty; discounted_distance_potential; terminal held +5/excursion -5',
        validation_cases=cases,validation_count=len(cases),validation_duration_s=a.val_duration,
        validation_interval_steps=a.val_every,patience_checks=a.patience,minimum_training_steps=a.min_steps,
        validation_metric='no_lost_initial_success_then_success_count_then_mean_final_error_1mm_improvement',
        validation_is_partial_subset=True,validation_exploration=False,validation_replay_included=False,
        initial_zero_actor_is_baseline=True,failed_v1_weights_loaded=False,smoke_only=a.smoke,
        inference_module_sha256={n:sha256(Path(__file__).with_name(n)) for n in modules},
        training_source_sha256=sha256(Path(__file__)),
        environment_source_sha256={n:sha256(Path(__file__).with_name(n)) for n in ('robotnik_arm_rl_env.py',
             'robotnik_arm_rl_conservative_env.py','robotnik_arm_rl_conservative_validation.py')},
        learner_device=device,policy_architecture='frozen_recovery120_plus_near_gated_TD3_26_64_64_7'))
    save_json(run/'config.json',c)
    learner=TD3(device,a.actor_lr,a.critic_lr,a.actor_penalty);actor=Actor().eval()
    actor.load_state_dict(cpu_state(learner.actor));replay=Replay(a.replay_size)
    target_rng,explore_rng,replay_rng=[np.random.default_rng(s) for s in np.random.SeedSequence(a.seed).spawn(3)]
    frozen_base=cpu_state(context.base);frozen_nav=cpu_state(context.navigation)
    steps=episodes=0;best_step=0;selector=None;started=time.perf_counter();next_check=a.val_every
    last_metrics=dict(critic_loss=None,actor_loss=None);stop_reason='transition_budget'

    def checkpoint(name):
        v=dict(config=c,epoch=steps,transition_steps=steps,episodes=episodes,actor_state_dict=cpu_state(learner.actor),
            critic_state_dict=cpu_state(learner.critic),actor_target_state_dict=cpu_state(learner.actor_target),
            critic_target_state_dict=cpu_state(learner.critic_target),actor_optimizer_state_dict=learner.actor_optimizer.state_dict(),
            critic_optimizer_state_dict=learner.critic_optimizer.state_dict(),td3_updates=learner.updates,
            actor_updates=learner.actor_updates,selection_step=steps,
            validation=selector.best if selector is not None else None)
        temp=run/(name+'.tmp');torch.save(v,temp);temp.replace(run/name)

    def check_validation():
        nonlocal selector,best_step
        if sha256(robot_path)!=robot_hash:raise RuntimeError('Robot config changed')
        print(f'Validation at RL step {steps}: deterministic actor, config sensor noise, true anytime1cm.',flush=True)
        result=evaluate(context,actor,c,cases,simlog)
        result['transition_steps']=steps
        if selector is None:
            selector=Selection(result,result);improved=True;lost=[]
        else:improved,lost=selector.consider(result)
        save_json(run/'validation'/f'step_{steps:06d}.json',dict(result=result,new_best=improved,
                  lost_initial_success_indices=lost,bad_checks=selector.bad_checks,partial_validation_subset=True))
        if improved:best_step=steps;checkpoint('best_model.pt')
        print(f'  VAL SUBSET {result["successes"]}/{result["count"]}; mean final={result["mean_final_error_m"]:.5f}m; '
              f'lost baseline points={lost}; new best={improved}; no improvement={selector.bad_checks}/{a.patience}',flush=True)

    columns=['episode','target_index','target_x','target_y','target_z','seed','near_ready','warmup_s',
             'prefix_reached_1cm','initial_fine_error_m','transition_steps','episode_steps','reason','held_success',
             'min_error_m','final_error_m','train_return','effective_residual_rms_nm','gate_active_fraction',
             'critic_loss','actor_loss','wall_seconds']
    print(f'Results: {run}',flush=True)
    try:
        with (run/'simulation.log').open('w',encoding='utf-8') as simlog,(run/'episodes.csv').open('w',newline='',encoding='utf-8') as log:
            writer=csv.DictWriter(log,fieldnames=columns);writer.writeheader();log.flush()
            # [V2 selection] 先评价并保存零修正；即使后面全退化，best仍可退回这份。
            checkpoint('initial_model.pt');check_validation()
            while steps<a.steps and episodes<a.max_episodes:
                if sha256(robot_path)!=robot_hash:raise RuntimeError('Robot config changed during training')
                episodes+=1;pool=low if target_rng.random()<a.low_target_fraction else other
                index=int(target_rng.choice(pool));seed=int(target_rng.integers(0,2**31-1));target=train[index]
                print(f'[{episodes}] train[{index}] target={target}; seek observed near zone...',flush=True)
                env=None;total=0.;ep_steps=0;rms_sum=0.;gate_sum=0.;held=False;reason='no_eligible_near_zone'
                try:
                    with contextlib.redirect_stdout(simlog):
                        env=FineReachEnv(context,target,seed,10,a.residual_limit_nm,a.fine_seconds,a.warmup_seconds,
                            a.hold_seconds,a.discount_tau_s,a.sensor_noise,a.near_inner_m,a.near_outer_m)
                        state,ready=env.warmup()
                    prefix_min=env.min_error;initial=env.frame['error'];warmup=env.steps*env.dt
                    if ready:
                        while steps<a.steps:
                            with torch.inference_mode():proposal=actor(torch.from_numpy(state).unsqueeze(0))[0].numpy()
                            proposal=np.clip(proposal+explore_rng.normal(0,a.exploration_std,7),-1,1).astype(np.float32)
                            sp,reward,terminal,truncated,info=env.step(proposal)
                            replay.add(state,proposal,reward,sp,info['discount'],terminal)
                            state=sp;total+=reward;steps+=1;ep_steps+=1
                            rms_sum+=info['effective_delta_rms_nm']**2;gate_sum+=info['gate_active_fraction']
                            if steps>=a.learning_starts and replay.size>=a.batch_size and steps%a.update_every==0:
                                for _ in range(a.gradient_steps):
                                    metrics=learner.update(replay.sample(a.batch_size,replay_rng,learner.device),allow_actor=steps>=a.actor_start)
                                    for k,v in metrics.items():
                                        if v is not None:last_metrics[k]=v
                                actor.load_state_dict(cpu_state(learner.actor))
                            if steps%1000==0:print(f'  RL {steps}/{a.steps}; error={env.frame["error"]:.4f}m; actor updates={learner.actor_updates}',flush=True)
                            reason=info['reason'];held=terminal and reason=='true_target_held'
                            if terminal or truncated:break
                        if steps==a.steps and reason=='running':reason='training_budget_reached'
                    writer.writerow(dict(episode=episodes,target_index=index,target_x=target[0],target_y=target[1],target_z=target[2],
                        seed=seed,near_ready=ready,warmup_s=warmup,prefix_reached_1cm=prefix_min<.01,initial_fine_error_m=initial,
                        transition_steps=steps,episode_steps=ep_steps,reason=reason,held_success=held,min_error_m=env.min_error,
                        final_error_m=env.frame['error'],train_return=total,effective_residual_rms_nm=float(np.sqrt(rms_sum/max(1,ep_steps))),
                        gate_active_fraction=gate_sum/max(1,ep_steps),**last_metrics,wall_seconds=time.perf_counter()-started));log.flush()
                    print(f'  {reason}; error={env.frame["error"]:.5f}m; steps={steps}; effective delta RMS={np.sqrt(rms_sum/max(1,ep_steps)):.4f}Nm',flush=True)
                finally:
                    if env is not None:env.close()
                checkpoint('last_model.pt')
                if steps>=next_check or steps==a.steps:
                    check_validation()
                    while next_check<=steps:next_check+=a.val_every
                    if steps>=a.min_steps and selector.bad_checks>=a.patience:
                        stop_reason='validation_early_stopping';break
            if steps<a.steps and episodes==a.max_episodes:stop_reason='episode_cap'
            # 回合上限也评价最后策略；已评价相同步数则不重复。
            if selector.best is not None and not (run/'validation'/f'step_{steps:06d}.json').exists():check_validation()
    except BaseException:
        checkpoint('interrupted_model.pt');save_json(run/'interrupted.json',dict(steps=steps,episodes=episodes));raise
    for original,model in ((frozen_base,context.base),(frozen_nav,context.navigation)):
        if not all(torch.equal(v,model.state_dict()[k]) for k,v in original.items()):raise RuntimeError('Frozen model changed')
    for key,folder in [('source_model_file_sha256',context.parent),('frozen_nav_file_sha256',context.nav)]:
        if c[key]!={n:sha256(folder/n) for n in FROZEN}:raise RuntimeError('Frozen source files changed')
    checkpoint('last_model.pt')
    save_json(run/'training_finished.json',dict(stop_reason=stop_reason,transition_steps=steps,episodes=episodes,
        best_step=best_step,best_is_zero_initial_baseline=best_step==0,best_validation_subset=selector.best,
        validation_subset_only=True,smoke_only=a.smoke,td3_updates=learner.updates,actor_updates=learner.actor_updates,
        wall_seconds=time.perf_counter()-started))
    print(f'Finished: {stop_reason}; best RL step={best_step}; subset success={selector.best["successes"]}/{selector.best["count"]}.',flush=True)
    if best_step==0:print('No admissible improvement: best_model.pt is the initial ZERO-residual baseline, NOT an RL improvement.',flush=True)
    print(f'Results: {run}\nTraining only. Run batch_test_arm_rl_residual_conservative_parallel.py separately.',flush=True)


if __name__=='__main__':main()
