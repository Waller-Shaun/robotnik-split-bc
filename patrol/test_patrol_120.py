"""[Patrol120 change] Frozen recovery120, continuous targets, no robot resets.

Stage bookkeeping, clocks, route ordering and relative parking geometry are new.
All 4 wheel velocities and 7 arm torques still come from the original models.
No new training, expert arm control, forced parking, IK or arm PD is added.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import secrets
import time

import numpy as np
import pinocchio as pin
import torch
from simulation_and_control import pb, MotorCommands, PinWrapper

from patrol_task_state import PatrolMonitor
from patrol_parking_geometry import PARKING_GEOMETRY_VERSION, parking_goal_from_current_base
from patrol_route_planner import ROUTE_PLANNER_VERSION, plan_patrol_route
from patrol_route_display import PatrolRouteDisplay, save_route_preview
from patrol_target_sampling import (TARGET_SAMPLING_VERSION, RADIUS_RANGE_M, HEIGHT_RANGE_M,
                                    MIN_PAIR_XY_DISTANCE_M, MIN_TRIANGLE_XY_AREA_M2, sample_patrol_targets)
from patrol_checkpoint_bundle import sha256, load_patrol_bundle
from robotnik_arm_control import (BaseNavState, BaseControllerParams, base_controller,
                                  configure_gripper_fingers, quaternion2bearing)
from robotnik_bodyframe_features import build_bodyframe_observation
from robotnik_navigation_policy import (NAV_PARAMS,
                                        build_navigation_observation)

ARM_SHA256 = '751fe1d4d9310ee79ccb36d9d78db820ae930dd8145146938b22b5295534d86d'
NAV_SHA256 = '110e547e46b8bec3deb2b83b5142294be2f60562aabbbd8aecd843bb4ca07aba'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=float, nargs=3, action='append', metavar=('X','Y','Z'),
                   help='Repeat to set explicit targets, overriding random generation. Shortest route is the default.')
    # [随机目标参数改动] 默认仍为3点；显式--target优先于这些随机采样设置。
    p.add_argument('--num-targets',type=int,default=3,help='Number of random targets; default 3. Ignored with --target.')
    p.add_argument('--radius-range',type=float,nargs=2,default=RADIUS_RANGE_M,metavar=('MIN','MAX'),
                   help='Initial horizontal radius in metres; default 0.8 2. Wider ranges are allowed for exploration.')
    p.add_argument('--height-range',type=float,nargs=2,default=HEIGHT_RANGE_M,metavar=('MIN','MAX'),
                   help='Random target heights in metres; must stay within 0.5 to 1.3. Equal bounds give fixed height.')
    p.add_argument('--min-target-spacing',type=float,default=MIN_PAIR_XY_DISTANCE_M,
                   help='Minimum horizontal distance between random targets, metres; default 0.4.')
    # [随机目标改动] 不传时每次生成新点；传入可复现同一组目标，独立于传感器seed。
    p.add_argument('--target-seed',type=int,default=None,
                   help='Reproduce random target coordinates; omit for a fresh set each run. Independent of --seed.')
    # [路线排序改动] 默认预计最短停车路程；保留旧输入顺序供对照。
    p.add_argument('--route-order',choices=['shortest','given'],default='shortest')
    # [路线显示改动] 默认在GUI显示青色预定路线，允许--no-show-planned-route隐藏。
    p.add_argument('--show-planned-route',action=argparse.BooleanOptionalAction,default=True,
                   help='Show estimated route/visit labels in GUI; a route_plan.png is saved in either mode.')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--sensor-noise', choices=['config','off'], default='config')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--duration-per-target', type=float, default=20.0)
    p.add_argument('--hold-seconds', type=float, default=1.0)
    # [Patrol application change] 5cm inspection tolerance; old coursework tests remain 1cm.
    p.add_argument('--reach-tolerance', type=float, default=0.05,
                   help='Application arrival/holding tolerance in metres; default 0.05, historical check 0.01.')
    p.add_argument('--continue-after-timeout', action='store_true',
                   help='Keep trying remaining targets after an observed task timeout; no resets.')
    # [可移植发布适配] 资源根目录可复用现有RoboEnv；配置快照独立读取，不覆盖环境。
    p.add_argument('--roboenv-root', type=Path, default=Path(__file__).resolve().parents[1]/'RoboEnv')
    p.add_argument('--robot-config', type=Path, default=Path(__file__).resolve().parents[1]/'artifacts/patrol120/configs/robotnik_arm.json')
    p.add_argument('--output-dir', type=Path, default=None)
    args = p.parse_args()
    if (args.seed < 0 or not np.isfinite(args.duration_per_target) or args.duration_per_target <= 0
            or not np.isfinite(args.hold_seconds) or args.hold_seconds < 0
            or not np.isfinite(args.reach_tolerance) or args.reach_tolerance <= 0):
        p.error('Seed/time/hold parameters invalid')
    if args.target_seed is not None and args.target_seed < 0:
        p.error('target-seed must be nonnegative')
    if args.num_targets<1:p.error('num-targets must be positive')
    if args.target:
        args.targets = np.asarray(args.target, dtype=np.float64)
        args.target_selection = {'mode':'explicit','random_targets_generated':False,
                                 'target_seed':None,'target_seed_argument_used':False}
    else:
        target_seed=args.target_seed if args.target_seed is not None else secrets.randbits(32)
        try:
            args.targets=sample_patrol_targets(target_seed,args.num_targets,args.radius_range,
                                               args.height_range,args.min_target_spacing)
        except ValueError as exc:p.error(str(exc))
        args.target_selection = {
            'mode':'random','random_targets_generated':True,'count':args.num_targets,
            'sampling_version':TARGET_SAMPLING_VERSION,'target_seed':target_seed,
            'target_seed_source':'argument' if args.target_seed is not None else 'system_entropy',
            'independent_of_sensor_noise_seed':True,
            'initial_horizontal_radius_range_m':list(args.radius_range),
            'height_range_m':list(args.height_range),'height_distribution':'continuous_uniform_or_fixed',
            'horizontal_distribution':('area_uniform_annulus_conditioned_on_separation_and_triangle_area'
                                       if args.num_targets>=3 else 'area_uniform_annulus_conditioned_on_separation'),
            'minimum_pair_xy_distance_m':args.min_target_spacing,
            'minimum_triangle_xy_area_m2':MIN_TRIANGLE_XY_AREA_M2 if args.num_targets>=3 else None,
            'model_success_used_for_target_selection':False}
    if not np.isfinite(args.targets).all():
        p.error('Targets must be finite')
    if np.any((args.targets[:,2] < .5) | (args.targets[:,2] > 1.3)):
        p.error('This high-height verification accepts only 0.5 <= z <= 1.3 m')
    radii = np.linalg.norm(args.targets[:,:2], axis=1)
    # [参数改动] 半径可扩展到训练范围外；如实记录，不默默裁剪坐标。
    args.target_selection['initial_target_ids_outside_training_radius']=(
        np.flatnonzero((radii<.8-1e-6)|(radii>2+1e-6))+1).tolist()
    return args


def load_models(root):
    # [可移植发布适配] 校验原始模型字节，不要求本机存在历史训练父目录。
    return load_patrol_bundle(Path(__file__).resolve().parents[1] / 'artifacts/patrol120')


def physical_snapshot(client, rid):
    """Noise-free logging/check only; these values are never sent to the policies."""
    pos, ori = client.getBasePositionAndOrientation(rid)
    vel, omega = client.getBaseVelocity(rid)
    joints = client.getJointStates(rid, list(range(client.getNumJoints(rid))))
    return np.r_[pos, ori, vel, omega, [x[0] for x in joints], [x[1] for x in joints]]


def save_plot(out, stages, route_plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for stage in stages:
        with np.load(out/stage['trace_file']) as d:
            t=d['state_times']; label=f"Visit {stage['target_index']} / input #{stage['input_target_index']}"
            axes[0].plot(t,d['true_ee_errors'],label=label)
            axes[1].plot(d['true_base_positions'][:,0],d['true_base_positions'][:,1],label=label)
            axes[1].scatter(*stage['parking_goal_xy'],marker='x',s=70)
    # [路线排序改动] 虚线是理想停车路程估计，实线仍是实际底盘轨迹。
    estimated=np.asarray([route_plan['observed_start_base_xyz'][:2]]+
                         [leg['estimated_parking_xy'] for leg in route_plan['estimated_legs']])
    axes[1].plot(estimated[:,0],estimated[:,1],'k--',alpha=.5,label='Estimated parking route')
    for visit,leg in enumerate(route_plan['estimated_legs'],1):
        axes[1].annotate(f"{visit}:#{leg['input_target_index']}",leg['estimated_parking_xy'])
    tolerance=stages[0]['reach_tolerance_m']
    axes[0].axhline(tolerance,color='red',linestyle='--',label=f'{100*tolerance:g} cm')
    axes[0].set(xlabel='Continuous simulation time (s)',ylabel='True EE error (m)')
    axes[1].set(xlabel='Base world X (m)',ylabel='Base world Y (m)')
    axes[1].axis('equal')
    for ax in axes:ax.grid(alpha=.3);ax.legend()
    fig.suptitle('Frozen recovery120: continuous multi-target verification')
    fig.tight_layout();fig.savefig(out/'patrol.png',dpi=160);plt.close(fig)


def main():
    args=parse_args()
    root=args.roboenv_root.resolve()
    np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(1)
    arm,nav,stats,ns,run,nav_run=load_models(root)
    out=args.output_dir or root/'patrol_validation_120'/f'patrol_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    config_path=args.robot_config.resolve()
    source_hashes={str(p):sha256(p) for p in [run/'best_model.pt',nav_run/'best_model.pt',config_path]}
    config={
        'schema':'frozen_recovery120_continuous_patrol_v6','arm_run':str(run),'arm_epoch':138,
        'navigation_run':str(nav_run),'input_targets':args.targets.tolist(),
        'targets':args.targets.tolist(),'visit_order':args.route_order,
        'route_planner_version':ROUTE_PLANNER_VERSION,
        'route_planner_sha256':sha256(Path(__file__).with_name('patrol_route_planner.py')),
        'target_selection':args.target_selection,
        'roboenv_root':str(root),'robot_config_path':str(config_path),
        'target_sampling_sha256':sha256(Path(__file__).with_name('patrol_target_sampling.py')),
        'route_display_sha256':sha256(Path(__file__).with_name('patrol_route_display.py')),
        'planned_route_gui_enabled':bool(args.show_planned_route and not args.headless),
        'planned_route_preview_file':'route_plan.png',
        'sensor_noise_mode':args.sensor_noise,'seed':args.seed,'per_target_limit_s':args.duration_per_target,
        'continue_after_timeout':args.continue_after_timeout,
        'required_hold_s':args.hold_seconds,'reach_tolerance_m':args.reach_tolerance,
        'task_switch_source':'observed_FK_continuous_hold_only','success_scoring':'true_PyBullet_EE_continuous_hold',
        'true_state_used_as_model_input':False,'true_state_used_for_task_switch':False,
        'robot_reset_between_targets':False,'forced_parking':False,'expert_arm_actions':False,
        'route_optimality_claimed':False,'training_performed':False,'source_hashes':source_hashes,
        'script_sha256':sha256(Path(__file__)),
        'task_state_sha256':sha256(Path(__file__).with_name('patrol_task_state.py')),
        # [相对停车改动] 每段首帧带噪底盘观测确定几何，不使用真实位置规划。
        'parking_geometry_version':PARKING_GEOMETRY_VERSION,
        'parking_geometry_source':'first_observed_base_pose_of_each_stage',
        'parking_goal_fixed_within_stage':True,
        'parking_geometry_sha256':sha256(Path(__file__).with_name('patrol_parking_geometry.py')),
    }
    write_json(out/'config.json',config)
    stages=[];boundaries=[];global_step=0;total_path=0.;previous_base=None
    mission_started=time.perf_counter();mission_reason='completed';previous_end=None
    sim=None
    try:
        sim=pb.SimInterface(str(config_path),conf_file_path_ext=str(root),use_gui=not args.headless)
        client=sim.pybullet_client;rid=sim.bot[0].bot_pybullet
        if args.sensor_noise=='off':
            for robot in sim.bot:robot.noise_flag=False
        enabled=[bool(b.noise_flag) for b in sim.bot]
        if args.sensor_noise=='config' and not all(enabled):
            raise ValueError('Config noise is off. Use --sensor-noise off explicitly or enable it yourself.')
        config['sensor_noise_enabled']=enabled;write_json(out/'config.json',config)
        print(f'Frozen recovery120 epoch 138; targets={len(args.targets)}; noise={enabled}; tolerance={100*args.reach_tolerance:g}cm',flush=True)
        print(f'Results: {out}',flush=True)
        print(f"Target selection: {args.target_selection['mode']}; target seed={args.target_selection['target_seed']}",flush=True)
        for target_id,coordinates in enumerate(args.targets,1):
            print(f'  input target #{target_id}: {coordinates.tolist()}',flush=True)
        outside=args.target_selection['initial_target_ids_outside_training_radius']
        if outside:print(f'Outside trained initial radius 0.8-2m: target IDs {outside}',flush=True)
        sim.SetFloorFriction(1.0)
        for j in range(client.getNumJoints(rid)):
            if 'wheel' in client.getJointInfo(rid,j)[1].decode():
                client.changeDynamics(rid,j,contactStiffness=30000,contactDamping=1000,anisotropicFriction=[1.,.01,1.])
        sim.bot[0].servo_motor_model.set_motor_gains(400,30)
        # Same initial finger setup as the original baseline, called ONCE only.
        configure_gripper_fingers(sim,client,rid,np.full(6,.1))
        markers=[]
        for target in args.targets:
            visual=client.createVisualShape(client.GEOM_SPHERE,radius=.03,rgbaColor=[.3,.4,1.,.85])
            markers.append(client.createMultiBody(baseMass=0,baseVisualShapeIndex=visual,basePosition=target.tolist()))
        joints=np.expand_dims(np.array(sim.getNameActiveJoints()),axis=0)
        dyn=PinWrapper(str(config_path),'pybullet',joints,['pybullet'],False,0,str(root))
        frame=dyn.pin_model.getFrameId('robot_j2s7s300_end_effector')
        links=[j for j in range(client.getNumJoints(rid)) if client.getJointInfo(rid,j)[12].decode()=='robot_j2s7s300_end_effector']
        if len(links)!=1:raise ValueError('Expected exactly one EE link')
        dt=float(sim.GetTimeStep());budget=int(np.ceil(args.duration_per_target/dt))
        cmd=MotorCommands();limits=np.r_[[3.]*4,[30.]*7]
        # [路线排序改动] 复用原第1帧位置/朝向读数，排序不额外消耗噪声样本。
        initial_pos=np.asarray(sim.GetBasePosition()).copy()
        initial_ori=np.asarray(sim.GetBaseOrientation()).copy()
        initial_yaw=quaternion2bearing(initial_ori[3],initial_ori[0],initial_ori[1],initial_ori[2])
        route_plan=plan_patrol_route(args.targets,initial_pos,initial_yaw,args.route_order)
        order=route_plan['visit_order_indices_zero_based']
        execution_targets=args.targets[order].copy()
        markers=[markers[i] for i in order]
        config['targets']=execution_targets.tolist()
        config['visit_order_input_ids']=route_plan['visit_order_input_ids']
        config['route_plan']=route_plan
        write_json(out/'route_plan.json',route_plan);write_json(out/'config.json',config)
        # [路线显示改动] 图形只展示预计停车路程，不改变物理世界或模型输入。
        save_route_preview(out,route_plan)
        route_display=PatrolRouteDisplay(client,enabled=args.show_planned_route and not args.headless)
        route_display.draw_plan(route_plan)
        print('Route preview: '+str(out/'route_plan.png'),flush=True)
        if args.show_planned_route and not args.headless:
            print('GUI: cyan=planned parking route; orange=current parking leg; labels=visit order / original target ID',flush=True)
        print('Visit input target IDs: '+' -> '.join(map(str,route_plan['visit_order_input_ids'])),flush=True)
        print(f"Estimated parking distance: input {route_plan['input_order_estimated_distance_m']:.3f}m; selected {route_plan['selected_order_estimated_distance_m']:.3f}m; saving {route_plan['estimated_saving_percent']:.1f}%",flush=True)
        print(f"Route method: {route_plan['algorithm']}; actual shortest path is NOT guaranteed",flush=True)
        for index,target in enumerate(execution_targets):
            # [Patrol120 change] Only reset task clock/state; NEVER reset physics, q or velocity.
            start_snapshot=physical_snapshot(client,rid)
            if previous_end is not None:
                jump=float(np.max(np.abs(start_snapshot-previous_end)))
                boundaries.append({'from_target':index,'to_target':index+1,'time_s':global_step*dt,
                                   'max_physical_state_difference':jump,'exact_state_continuity':jump<=1e-12})
                if jump>1e-12:raise AssertionError('Physical state changed at task boundary without a simulation step')
            start_step=global_step;goal=None;goal_bearing=None;parking_start_base=None
            phase=BaseNavState();params=BaseControllerParams(**NAV_PARAMS)
            monitor=PatrolMonitor(args.hold_seconds,args.reach_tolerance);switch_time=None;clipped=0
            trace={k:[] for k in ['state_times','stage_elapsed_times','observed_ee_positions','true_ee_positions',
                   'observed_ee_errors','true_ee_errors','true_base_positions','observed_base_positions',
                   'action_times','observations','navigation_observations','raw_actions','actions','nav_subphases']}
            reason='time_limit';observed_complete=False
            client.changeVisualShape(markers[index],-1,rgbaColor=[1.,0.,0.,.9])
            input_id=order[index]+1
            print(f'[{index+1}/{len(args.targets)}] input target #{input_id}={target.tolist()}; global t={global_step*dt:.3f}s',flush=True)
            for step in range(budget+1):
                elapsed=step*dt;t=global_step*dt
                if index==0 and step==0:
                    pos=initial_pos.copy();ori=initial_ori.copy()
                else:
                    pos=np.asarray(sim.GetBasePosition());ori=np.asarray(sim.GetBaseOrientation())
                yaw=quaternion2bearing(ori[3],ori[0],ori[1],ori[2])
                if step==0:
                    # [相对停车改动] 复用首帧观测，不额外读传感器，避免改变噪声序列。
                    # 本段内不重算，否则停车点会跟着底盘移动。
                    goal,goal_bearing=parking_goal_from_current_base(target,pos,yaw)
                    parking_start_base=pos.copy()
                    route_display.show_current_leg(pos,goal,index+1)
                    print(f'  relative parking goal={goal.tolist()}; heading={goal_bearing:.4f}rad',flush=True)
                q=np.asarray(sim.GetMotorAngles(0));qd=np.asarray(sim.GetMotorVelocities(0))
                qp=dyn.ReoderJoints2PinVec(np.r_[pos,ori,q],'pos')
                pin.forwardKinematics(dyn.pin_model,dyn.pin_data,qp);pin.updateFramePlacements(dyn.pin_model,dyn.pin_data)
                ee=dyn.pin_data.oMf[frame].translation.copy()
                true_ee=np.asarray(client.getLinkState(rid,links[0],computeForwardKinematics=True)[4],dtype=np.float64)
                true_base=np.asarray(client.getBasePositionAndOrientation(rid)[0],dtype=np.float64)
                observed_error=float(np.linalg.norm(target-ee));true_error=float(np.linalg.norm(target-true_ee))
                for k,v in [('state_times',t),('stage_elapsed_times',elapsed),('observed_ee_positions',ee.copy()),
                            ('true_ee_positions',true_ee.copy()),('observed_ee_errors',observed_error),
                            ('true_ee_errors',true_error),('true_base_positions',true_base.copy()),('observed_base_positions',pos.copy())]:
                    trace[k].append(v)
                if previous_base is not None:total_path+=float(np.linalg.norm(true_base[:2]-previous_base[:2]))
                previous_base=true_base.copy()
                if not np.isfinite(observed_error) or not np.isfinite(true_error):reason='nonfinite_state';break
                observed_complete=monitor.update(elapsed,observed_error,true_error)
                if observed_complete:reason='observed_hold_completed';break
                if step==budget:break
                was_done=phase.done
                # Same phase rules as baseline, but with a per-target clock. Expert wheel return discarded.
                base_controller(phase,pos,yaw,goal,goal_bearing,elapsed,params)
                if phase.done and not was_done:switch_time=elapsed
                source=np.r_[pos,np.sin(yaw),np.cos(yaw),q[4:11],qd[4:11],target,0. if phase.done else 1.].astype(np.float32)
                x=build_bodyframe_observation(source,ee)
                nx=build_navigation_observation(pos,yaw,goal,goal_bearing,phase)
                with torch.inference_mode():
                    all_arm=arm(torch.from_numpy((x-stats['x_mean'])/stats['x_std'])[None])[0].numpy()*stats['y_std']+stats['y_mean']
                    wheels=nav(torch.from_numpy((nx-ns['x_mean'])/ns['x_std'])[None])[0].numpy()*ns['y_std']+ns['y_mean']
                raw=np.r_[wheels,all_arm[4:11]]
                if not np.isfinite(raw).all():reason='nonfinite_action';break
                action=np.clip(raw,-limits,limits);clipped+=int(np.any(action!=raw))
                for k,v in [('action_times',t),('observations',x.copy()),('navigation_observations',nx.copy()),
                            ('raw_actions',raw.copy()),('actions',action.copy()),('nav_subphases','done' if phase.done else phase.phase)]:
                    trace[k].append(v)
                cmd.SetControlCmd(np.r_[action,np.zeros(6)],['velocity']*4+['torque']*13)
                sim.Step(cmd,'torque');global_step+=1
                if step%5000==0:print(f'  elapsed={elapsed:.1f}s; true={true_error:.4f}m; phase={phase.phase}; done={phase.done}',flush=True)
                if not args.headless:time.sleep(dt)
            previous_end=physical_snapshot(client,rid)
            stage={
                'target_index':index+1,'input_target_index':int(input_id),
                'target':target.tolist(),'parking_goal_xy':goal.tolist(),'goal_bearing':float(goal_bearing),
                'parking_geometry_version':PARKING_GEOMETRY_VERSION,
                'parking_start_observed_base_xyz':parking_start_base.tolist(),
                'reach_tolerance_m':args.reach_tolerance,
                'start_global_time_s':start_step*dt,'end_global_time_s':global_step*dt,
                'elapsed_s':(global_step-start_step)*dt,'observed_task_completed':bool(observed_complete),
                'termination_reason':reason,**monitor.report(),'navigation_completed':bool(phase.done),
                'navigation_end_phase':phase.phase,'phase_switch_elapsed_s':switch_time,
                'final_true_error_m':true_error,'min_true_error_m':float(min(trace['true_ee_errors'])),
                'final_observed_error_m':observed_error,'clipped_steps':clipped,'action_steps':len(trace['actions']),
                'trace_file':f'target_{index+1:02d}.npz','physical_start_state':start_snapshot.tolist(),
                'physical_end_state':previous_end.tolist(),
            }
            np.savez_compressed(out/stage['trace_file'],**{k:np.asarray(v) for k,v in trace.items()},target=target,
                                true_state_used_for_control=False,all_actions_model_generated=True)
            stages.append(stage);write_json(out/f'target_{index+1:02d}.json',stage)
            print(f"  {reason}; first true {100*args.reach_tolerance:g}cm={stage['true_first_reach_elapsed_s']}; longest true hold={stage['true_longest_continuous_hold_s']:.3f}s; true pass={stage['true_hold_passed']}",flush=True)
            client.changeVisualShape(markers[index],-1,rgbaColor=[0.,.8,0.,.9] if observed_complete else [1.,.5,0.,.9])
            if reason.startswith('nonfinite') or (not observed_complete and not args.continue_after_timeout):
                mission_reason=reason;break
    except Exception as exc:
        write_json(out/'ERROR.json',{'type':type(exc).__name__,'message':str(exc),'completed_stage_records':len(stages)})
        raise
    finally:
        if sim is not None:sim.pybullet_client.disconnect()
    for p,expected in source_hashes.items():
        if sha256(p)!=expected:raise AssertionError(f'Frozen model/config changed: {p}')
    summary={
        **config,'planned_targets':len(args.targets),'attempted_targets':len(stages),
        'unattempted_targets':len(args.targets)-len(stages),'first_reach_count':sum(s['true_first_reach_elapsed_s'] is not None for s in stages),
        'observed_completed_count':sum(s['observed_task_completed'] for s in stages),
        'true_hold_passed_count':sum(s['true_hold_passed'] for s in stages),
        'mission_success':bool(len(stages)==len(args.targets) and all(s['observed_task_completed'] and s['true_hold_passed'] for s in stages)),
        'termination_reason':mission_reason,'total_simulation_time_s':global_step*dt,
        'elapsed_wall_s':time.perf_counter()-mission_started,'actual_base_path_length_m':total_path,
        'physical_state_continuity_checks':boundaries,'frozen_model_config_hashes_unchanged':True,'stages':stages,
        'interpretation':'Continuous patrol with estimated-route ordering; not the former single-target validation success rate. Actual route optimality is not claimed.'
    }
    write_json(out/'summary.json',summary)
    columns=['target_index','input_target_index','target','observed_task_completed','true_first_reach_elapsed_s','true_longest_continuous_hold_s',
             'true_hold_passed','navigation_completed','elapsed_s','final_true_error_m','min_true_error_m','termination_reason']
    with (out/'inspection_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(stages)
    save_plot(out,stages,route_plan)
    print(f"Mission success: {summary['mission_success']}; true held={summary['true_hold_passed_count']}/{len(args.targets)}; first reached={summary['first_reach_count']}/{len(args.targets)}",flush=True)
    print(f'Summary: {out / "summary.json"}',flush=True)


if __name__=='__main__':main()
