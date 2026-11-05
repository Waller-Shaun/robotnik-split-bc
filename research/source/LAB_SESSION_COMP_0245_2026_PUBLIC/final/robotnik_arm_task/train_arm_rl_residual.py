"""[RL change] 独立TD3训练：冻结120版MLP和导航，学习7维小幅力矩残差。

不重采专家示范、不使用验证/测试目标交互、不在本文件运行正式测试。
训练预算按100Hz环境transition计数，不是BC epoch；物理仿真仍为1kHz。
"""
import argparse
import contextlib
import csv
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from robotnik_arm_rl_policy import Actor, Replay, TD3, VARIANT, RUN_ROOT
from robotnik_arm_rl_env import FrozenContext, FineReachEnv
from recovery_dataset import FROZEN, sha256


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roboenv', type=Path, default=Path(__file__).resolve().parents[3]/'RoboEnv')
    parser.add_argument('--steps', type=int, default=20000, help='Total RL transitions, NOT physical milliseconds or epochs')
    parser.add_argument('--max-episodes', type=int, default=200)
    parser.add_argument('--scope', choices=['low', 'all'], default='all', help='all by default; low: TRAIN targets with world z < 0.5 m only')
    parser.add_argument('--low-target-fraction', type=float, default=.5, help='Probability of sampling z < 0.5 when scope=all')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--replay-size', type=int, default=100000)
    parser.add_argument('--learning-starts', type=int, default=1000)
    parser.add_argument('--warmup-steps', type=int, default=2000, help='Initial zero actor + small Gaussian exploration')
    parser.add_argument('--update-every', type=int, default=10)
    parser.add_argument('--gradient-steps', type=int, default=10)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--exploration-std', type=float, default=.05, help='Normalized residual; .05 * 2 Nm = .1 Nm')
    parser.add_argument('--residual-limit-nm', type=float, default=2.)
    parser.add_argument('--control-repeat', type=int, default=10)
    parser.add_argument('--fine-seconds', type=float, default=6.)
    parser.add_argument('--warmup-seconds', type=float, default=12.)
    parser.add_argument('--hold-seconds', type=float, default=.2)
    parser.add_argument('--sensor-noise', choices=['config', 'off'], default='config')
    parser.add_argument('--plan-only', action='store_true', help='Verify context and show training plan, no simulation or writes')
    parser.add_argument('--smoke', action='store_true', help='Program check only; excluded from automatic formal evaluation')
    args = parser.parse_args()
    for name in ('steps', 'max_episodes', 'batch_size', 'replay_size', 'update_every', 'gradient_steps', 'control_repeat'):
        if getattr(args, name) <= 0: parser.error(name+' must be positive')
    for name in ('actor_lr', 'critic_lr', 'residual_limit_nm', 'fine_seconds', 'warmup_seconds', 'hold_seconds'):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0: parser.error(name+' must be finite and positive')
    if (args.seed < 0 or args.learning_starts < 0 or args.warmup_steps < 0
            or not np.isfinite(args.exploration_std) or args.exploration_std < 0): parser.error('Invalid seed/exploration/learning start')
    if args.replay_size < args.batch_size: parser.error('replay-size must be >= batch-size')
    if not np.isfinite(args.low_target_fraction) or not 0<args.low_target_fraction<1:
        parser.error('low-target-fraction must be between 0 and 1')
    if args.residual_limit_nm > 5: parser.error('This initial experiment limits correction to at most 5 Nm')

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable in this interpreter')
    # [RL speed] 单帧物理反馈在CPU；批量actor/critic梯度可在GPU，避免每1ms做GPU拷贝。
    context = FrozenContext(args.roboenv)
    manifest = json.loads((context.parent/'data_split.json').read_text(encoding='utf-8'))
    train = manifest['targets']['train']
    train_keys = {tuple(t) for t in train}
    for split in ('val', 'test'):
        if train_keys & {tuple(t) for t in manifest['targets'][split]}: raise ValueError('Target split leakage')
    indices = [i for i,t in enumerate(train) if args.scope=='all' or t[2] < .5]
    if not indices: raise ValueError('No eligible train targets')
    low_indices=[i for i in indices if train[i][2]<.5]
    other_indices=[i for i in indices if train[i][2]>=.5]
    if args.scope=='all' and (not low_indices or not other_indices):raise ValueError('Missing height group')
    robot_config = args.roboenv/'configs'/'robotnik_arm.json'
    robot_snapshot = json.loads(robot_config.read_text(encoding='utf-8'))
    if args.sensor_noise=='config' and not all(robot_snapshot['robot_pybullet']['noise_flag']):
        raise ValueError('Enable config noise first, or explicitly choose --sensor-noise off')
    print(f'TD3 learner: {device}; frozen inference: CPU; physics: 1 kHz; residual: {1000/args.control_repeat:g} Hz', flush=True)
    print(f'Frozen parent: {context.parent}\nFrozen navigation: {context.nav}', flush=True)
    print(f'TRAIN targets: {len(indices)} / {len(train)}; scope={args.scope}; no val/test interaction', flush=True)
    print(f'Budget: {args.steps:,} transitions; at most {args.max_episodes} episodes; correction +/-{args.residual_limit_nm:g} Nm', flush=True)
    print('No validation early stopping. Fixed-budget final actor is saved; train return is NOT generalization success.', flush=True)
    if args.plan_only: return

    run = args.roboenv/RUN_ROOT/f'run_{datetime.now():%Y%m%d_%H%M%S_%f}'
    run.mkdir(parents=True, exist_ok=False)
    for name in ('normalization.npz', 'data_split.json'): shutil.copy2(context.parent/name, run/name)
    # [RL provenance] 不继承BC的epoch、采样比例等训练设置；只保留输入/输出和推理定义。
    feature_keys = ('input_dim','source_input_dim','output_dim','hidden_dims','input_feature_version',
                    'dataset_variant','base_active_column','target_position_frame','ee_error_frame')
    config = {k:context.pc[k] for k in feature_keys}
    config.update(dict(
        experiment_variant=VARIANT, training_mode='online_TD3_residual_on_frozen_recovery120',
        source_model_run_dir=str(context.parent), source_model_epoch=138, nav_run_dir=str(context.nav),
        source_model_file_sha256={n:sha256(context.parent/n) for n in FROZEN},
        frozen_nav_file_sha256={n:sha256(context.nav/n) for n in FROZEN},
        base_parameters_frozen=True, navigation_weights_updated=False, aux_weight=0.,
        gru_used=False, history_input=False, expert_references_used_as_input=False,
        critic_used_at_inference=False, true_state_used_as_policy_input=False,
        policy_architecture='frozen_MLP26_256_256_128_plus_TD3_actor26_64_64_7',
        residual_limit_nm=args.residual_limit_nm, control_repeat=args.control_repeat,
        physics_dt_s=.001, residual_activation='navigation_done_only',
        gamma=.99, policy_delay=2, target_noise_std=.1, target_noise_clip=.2, polyak=.995,
        actor_action_penalty=.001, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        exploration_std=args.exploration_std, exploration_used_at_test=False,
        replay_capacity=args.replay_size, batch_size=args.batch_size,
        total_transition_budget=args.steps, max_episodes=args.max_episodes,
        learning_starts=args.learning_starts, zero_policy_warmup_steps=args.warmup_steps,
        update_every=args.update_every, gradient_steps=args.gradient_steps,
        scope=args.scope, eligible_train_target_indices=indices, seed=args.seed,
        low_target_fraction=args.low_target_fraction if args.scope=='all' else 1.,
        fine_time_limit_s=args.fine_seconds, warmup_time_limit_s=args.warmup_seconds,
        training_required_hold_s=args.hold_seconds, training_tolerance_m=.01,
        training_success_bonus=5., training_divergence_error_m=1.,
        reward='5*(distance_before-distance_after) + integral[-10*min(distance,1)+I(distance<.01)-.01*mean(deltaNm^2)-.001*mean(qd^2)]dt; +5 held; -5 diverged',
        reward_position_source='pybullet_link_world_frame_not_policy_input',
        truncation_bootstrapped=True, sensor_noise_mode=args.sensor_noise,
        robot_config_path=str(robot_config), robot_config_sha256=sha256(robot_config), robot_config_snapshot=robot_snapshot,
        selection_metric='none_fixed_transition_budget',
        best_model_filename_meaning='compatibility_alias_of_last_actor_NOT_validation_selected_best',
        inference_module_sha256={n:sha256(Path(__file__).with_name(n)) for n in (
            'robotnik_arm_rl_policy.py','recovery_residual120_context.py','robotnik_arm_aux_policy.py',
            'robotnik_bodyframe_features.py','robotnik_navigation_policy.py','robotnik_arm_control.py')},
        training_source_sha256=sha256(Path(__file__)), environment_source_sha256=sha256(Path(__file__).with_name('robotnik_arm_rl_env.py')),
        learner_device=device, inference_device='cpu', smoke_only=args.smoke,
    ))
    save_json(run/'config.json', config)
    learner = TD3(device, args.actor_lr, args.critic_lr)
    inference_actor = Actor().eval()
    inference_actor.load_state_dict(cpu_state(learner.actor))
    replay = Replay(args.replay_size)
    # 分开目标、探索与经验回放的随机流；仿真自身沿用numpy全局种子。
    target_rng, exploration_rng, replay_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(args.seed).spawn(3)]
    frozen_base = cpu_state(context.base)
    frozen_nav = cpu_state(context.navigation)
    steps = 0; episodes = 0; skipped = 0; started = time.perf_counter(); reached_budget = False
    last_metrics = {'critic_loss':None, 'actor_loss':None}

    def checkpoint(name):
        # [RL save] 保存训练状态便于后续扩展续训；当前脚本不提供resume，不能当作BC续训加载。
        value = dict(config=config, epoch=steps, transition_steps=steps, episodes=episodes,
                     actor_state_dict=cpu_state(learner.actor), critic_state_dict=cpu_state(learner.critic),
                     actor_target_state_dict=cpu_state(learner.actor_target), critic_target_state_dict=cpu_state(learner.critic_target),
                     actor_optimizer_state_dict=learner.actor_optimizer.state_dict(),
                     critic_optimizer_state_dict=learner.critic_optimizer.state_dict(), td3_updates=learner.updates,
                     frozen_base_equal=all(torch.equal(v,context.base.state_dict()[k]) for k,v in frozen_base.items()))
        temp = run/(name+'.tmp'); torch.save(value, temp); temp.replace(run/name)

    checkpoint('initial_model.pt')
    columns = ['episode','train_target_index','target_x','target_y','target_z','seed','navigation_ready',
               'warmup_simulation_s','prefix_reached_1cm','prefix_min_error_m','initial_fine_error_m','transition_steps','fine_transition_steps',
               'reason','held_success','min_error_m','final_error_m','train_return','critic_loss','actor_loss','wall_seconds']
    print(f'Results: {run}', flush=True)
    try:
        with (run/'episodes.csv').open('w',newline='',encoding='utf-8') as log, (run/'simulation.log').open('w',encoding='utf-8') as simlog:
            writer=csv.DictWriter(log,fieldnames=columns); writer.writeheader(); log.flush()
            while steps < args.steps and episodes < args.max_episodes:
                if sha256(robot_config)!=config['robot_config_sha256']: raise RuntimeError('Robot config changed during training')
                episodes += 1
                pool=indices if args.scope=='low' else (low_indices if target_rng.random()<args.low_target_fraction else other_indices)
                index=int(target_rng.choice(pool)); target=train[index]
                episode_seed=int(target_rng.integers(0,2**31-1)); env=None; total_reward=0.; episode_steps=0
                # [RL warmup] 物理导航仍由冻结模型实际跑出；仅在导航完成后探索，不瞬移/专家接管。
                print(f'[{episodes}] train[{index}] target={target}; seed={episode_seed}; navigation warmup...',flush=True)
                try:
                    with contextlib.redirect_stdout(simlog):
                        env=FineReachEnv(context,target,episode_seed,args.control_repeat,args.residual_limit_nm,
                                         args.fine_seconds,args.warmup_seconds,args.hold_seconds,.99,args.sensor_noise)
                        state,ready=env.warmup()
                    prefix_min=env.min_error; fine_initial=env.frame['error']; warmup_time=env.steps*env.dt
                    reason='navigation_not_ready'; held=False
                    if not ready: skipped+=1
                    else:
                        while steps < args.steps:
                            with torch.inference_mode():
                                action=(np.zeros(7,np.float32) if steps < args.warmup_steps else
                                        inference_actor(torch.from_numpy(state).unsqueeze(0))[0].numpy())
                            action=np.clip(action+exploration_rng.normal(0,args.exploration_std,7),-1,1).astype(np.float32)
                            next_state,reward,terminal,truncated,info=env.step(action)
                            replay.add(state,action,reward,next_state,info['discount'],terminal)
                            state=next_state; total_reward+=reward; steps+=1; episode_steps+=1
                            if (steps>=args.learning_starts and replay.size>=args.batch_size and steps%args.update_every==0):
                                for _ in range(args.gradient_steps):
                                    metrics=learner.update(replay.sample(args.batch_size,replay_rng,learner.device))
                                    for k,v in metrics.items():
                                        if v is not None:last_metrics[k]=v
                                inference_actor.load_state_dict(cpu_state(learner.actor))
                            if steps%1000==0:
                                print(f'  RL steps {steps}/{args.steps}; error={info["true_error_m"]:.4f}m; updates={learner.updates}; elapsed={time.perf_counter()-started:.1f}s',flush=True)
                            reason=info['reason']; held=terminal and reason=='true_target_held'
                            if terminal or truncated:break
                        if steps==args.steps and reason=='running':reason='training_budget_reached'
                    writer.writerow(dict(episode=episodes,train_target_index=index,target_x=target[0],target_y=target[1],target_z=target[2],
                        seed=episode_seed,navigation_ready=ready,warmup_simulation_s=warmup_time,
                        prefix_reached_1cm=prefix_min<.01,prefix_min_error_m=prefix_min,initial_fine_error_m=fine_initial,
                        transition_steps=steps,fine_transition_steps=episode_steps,reason=reason,held_success=held,
                        min_error_m=env.min_error,final_error_m=env.frame['error'],train_return=total_reward,
                        **last_metrics,wall_seconds=time.perf_counter()-started));log.flush()
                    print(f'  {reason}; error={env.frame["error"]:.5f}m; RL steps={steps}; train return={total_reward:.3f}',flush=True)
                finally:
                    if env is not None:env.close()
                checkpoint('last_model.pt')
            reached_budget = steps==args.steps
    except BaseException:
        checkpoint('interrupted_model.pt')
        save_json(run/'interrupted.json',dict(steps=steps,episodes=episodes,wall_seconds=time.perf_counter()-started))
        raise

    if not all(torch.equal(v,context.base.state_dict()[k]) for k,v in frozen_base.items()):raise RuntimeError('Base parameters changed')
    if not all(torch.equal(v,context.navigation.state_dict()[k]) for k,v in frozen_nav.items()):raise RuntimeError('Navigation parameters changed')
    for field,folder in [('source_model_file_sha256',context.parent),('frozen_nav_file_sha256',context.nav)]:
        if config[field]!={n:sha256(folder/n) for n in FROZEN}:raise RuntimeError('Frozen source files changed')
    checkpoint('last_model.pt')
    if reached_budget and learner.updates>0:
        # 兼容既有独立测试读取best_model.pt的约定；这是固定预算最后模型，不冒充验证最优。
        shutil.copy2(run/'last_model.pt',run/'best_model.pt')
        save_json(run/'training_finished.json',dict(transition_steps=steps,episodes=episodes,td3_updates=learner.updates,
            navigation_skipped=skipped,budget_reached=True,smoke_only=args.smoke,base_and_nav_frozen=True,
            selection='last_fixed_budget_no_validation_selection',wall_seconds=time.perf_counter()-started))
        print('Fixed-budget TD3 training finished. Evaluate separately; training success/return is NOT validation success.',flush=True)
    else:
        save_json(run/'budget_not_completed.json',dict(transition_steps=steps,episodes=episodes,td3_updates=learner.updates,navigation_skipped=skipped))
        print('Budget NOT completed: no formal best_model published. Inspect episodes.csv before extending the experiment.',flush=True)
    print(f'Results: {run}',flush=True)
    print('Training only. Run batch_test_arm_rl_residual_parallel.py separately.',flush=True)


if __name__=='__main__':main()
