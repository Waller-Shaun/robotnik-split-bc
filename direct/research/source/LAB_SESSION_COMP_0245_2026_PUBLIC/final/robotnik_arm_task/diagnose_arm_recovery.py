"""[Recovery probe] 只诊断训练目标：相同模型状态，比较继续模型与专家机械臂恢复。

不训练、不生成训练清单；不使用验证/测试目标；不强制停车。真实位置只作评分。
"""
import argparse
from collections import deque
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import pinocchio as pin
from simulation_and_control import pb, MotorCommands, PinWrapper
from robotnik_arm_aux_policy import resolve_finetuned_pair, load_policy_from_checkpoint
from robotnik_navigation_policy import parking_goal, build_navigation_observation, load_navigation_policy, NAV_PARAMS
from robotnik_bodyframe_features import build_bodyframe_observation
from robotnik_arm_control import (BaseNavState, BaseControllerParams, base_controller,
    ArmControllerParams, arm_controller, configure_gripper_fingers, quaternion2bearing)

from recovery_probe_logic import select_training_targets, stalled_observations

BASELINE="bc_training_arm_ring_r08_aux_b2_control/run_20261104_025706_206181"





def load_context(root, run):
    arm,nav,ac,nc=resolve_finetuned_pair(root,run)
    if ac['aux_weight']!=0:
        raise ValueError('Recovery probe is bound to the torque-only baseline; auxiliary weight must be zero')
    model_cp=torch.load(arm/'best_model.pt',map_location='cpu',weights_only=True)
    nav_cp=torch.load(nav/'best_model.pt',map_location='cpu',weights_only=True)
    if model_cp['config']!=ac or nav_cp['config']!=nc:raise ValueError('Checkpoint config mismatch')
    model=load_policy_from_checkpoint(model_cp).eval();nav_model=load_navigation_policy(nav_cp).eval()
    for name,field in [('robotnik_navigation_policy.py','feature_module_sha256'),('robotnik_arm_control.py','controller_sha256')]:
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=nc[field]:
            raise ValueError('Frozen controller/navigation implementation changed')
    with np.load(arm/'normalization.npz') as n:
        stats={k:n[k].copy() for k in ['x_mean','x_std','y_mean','y_std']}
    with np.load(nav/'normalization.npz') as n:
        ns={k:n[k].copy() for k in ['x_mean','x_std','y_mean','y_std']}
    return arm,nav,model,nav_model,stats,ns


def run_probe(args):
    root=Path(__file__).resolve().parents[3]/'RoboEnv'
    run=args.arm_run_dir or root/BASELINE
    arm,nav,model,nav_model,stats,ns=load_context(root,run)
    manifest=json.loads((arm/'data_split.json').read_text(encoding='utf-8'))
    target=np.array(args.target,dtype=float)
    owner=next((i for i,t in enumerate(manifest['targets']['train']) if np.allclose(t,target,rtol=0,atol=1e-6)),None)
    if owner is None:raise ValueError('Diagnostic accepts TRAIN targets only; validation/test labels must not be collected')
    torch.set_num_threads(1);np.random.seed(args.seed)
    sim=pb.SimInterface('robotnik_arm.json',conf_file_path_ext=str(root),use_gui=not args.headless)
    client=sim.pybullet_client
    try:
        if args.sensor_noise=='off':
            for bot in sim.bot:bot.noise_flag=False
        enabled=[bool(bot.noise_flag) for bot in sim.bot]
        if args.sensor_noise=='config' and not all(enabled):raise ValueError('Configuration noise is not enabled')
        rid=sim.bot[0].bot_pybullet
        sim.SetFloorFriction(1.)
        for j in range(client.getNumJoints(rid)):
            if 'wheel' in client.getJointInfo(rid,j)[1].decode():
                client.changeDynamics(rid,j,contactStiffness=30000,contactDamping=1000,anisotropicFriction=[1.,.01,1.])
        sim.bot[0].servo_motor_model.set_motor_gains(400,30)
        configure_gripper_fingers(sim,client,rid,np.full(6,.1))
        visual=client.createVisualShape(client.GEOM_SPHERE,radius=.03,rgbaColor=[1,0,0,.85])
        client.createMultiBody(baseMass=0,baseVisualShapeIndex=visual,basePosition=target.tolist())
        names=np.expand_dims(np.array(sim.getNameActiveJoints()),axis=0)
        dyn=PinWrapper('robotnik_arm.json','pybullet',names,['pybullet'],False,0,str(root))
        ee_id=dyn.pin_model.getFrameId('robot_j2s7s300_end_effector')
        shoulder=dyn.pin_model.getFrameId('robot_j2s7s300_link_2')
        ee_links=[j for j in range(client.getNumJoints(rid)) if client.getJointInfo(rid,j)[12].decode()=='robot_j2s7s300_end_effector']
        if len(ee_links)!=1:raise ValueError('EE link frame missing')
        lower,upper=sim.GetBotJointsLimit();lower=np.asarray(lower[4:11]);upper=np.asarray(upper[4:11])
        expert_params=ArmControllerParams(reach_limit=.85)
        phase_params=BaseControllerParams(**NAV_PARAMS);goal,bearing_goal=parking_goal(target)
        limits=np.array([3.]*4+[30.]*7)
        dt=sim.GetTimeStep();cmd=MotorCommands()

        def observe(state):
            base=np.array(sim.GetBasePosition());ori=np.array(sim.GetBaseOrientation())
            bearing=quaternion2bearing(ori[3],ori[0],ori[1],ori[2])
            q=np.array(sim.GetMotorAngles(0));qd=np.array(sim.GetMotorVelocities(0))
            qp=dyn.ReoderJoints2PinVec(np.concatenate([base,ori,q]),'pos')
            pin.forwardKinematics(dyn.pin_model,dyn.pin_data,qp);pin.updateFramePlacements(dyn.pin_model,dyn.pin_data)
            ee=dyn.pin_data.oMf[ee_id].translation.copy()
            truth=np.array(client.getLinkState(rid,ee_links[0],computeForwardKinematics=True)[4])
            return {'base':base,'ori':ori,'bearing':bearing,'q':q,'qd':qd,'qp':qp,'ee':ee,'truth':truth,
                    'observed_error':float(np.linalg.norm(target-ee)),'true_error':float(np.linalg.norm(target-truth))}

        def predict(frame,state,t):
            # 相位仍仅由带噪观测更新，专家轮速一律丢弃。
            base_controller(state,frame['base'],frame['bearing'],goal,bearing_goal,t,phase_params)
            source=np.concatenate([frame['base'],[np.sin(frame['bearing']),np.cos(frame['bearing'])],
                frame['q'][4:11],frame['qd'][4:11],target,[0. if state.done else 1.]]).astype(np.float32)
            x=build_bodyframe_observation(source,frame['ee'])
            nx=build_navigation_observation(frame['base'],frame['bearing'],goal,bearing_goal,state)
            with torch.inference_mode():
                raw=model(torch.from_numpy((x-stats['x_mean'])/stats['x_std']).unsqueeze(0)).squeeze(0).numpy()*stats['y_std']+stats['y_mean']
                wheels=nav_model(torch.from_numpy((nx-ns['x_mean'])/ns['x_std']).unsqueeze(0)).squeeze(0).numpy()*ns['y_std']+ns['y_mean']
            return x,wheels,raw[4:11].copy()

        def apply(wheels,arm_action):
            raw=np.concatenate([wheels,arm_action]);action=np.clip(raw,-limits,limits)
            if not np.isfinite(action).all():raise ValueError('Nonfinite action')
            cmd.SetControlCmd(np.concatenate([action,np.zeros(6)]),['velocity']*4+['torque']*13)
            sim.Step(cmd,'torque')
            if not args.headless:time.sleep(dt)
            return action

        def empty_trace():
            return {k:[] for k in ['timestamps','ee_positions','ee_errors','observed_ee_positions','observed_ee_errors',
                'observations','actions','q_des_arm','qd_des_arm','action_source']}

        def record_state(trace,f,t):
            trace['timestamps'].append(t);trace['ee_positions'].append(f['truth']);trace['ee_errors'].append(f['true_error'])
            trace['observed_ee_positions'].append(f['ee']);trace['observed_ee_errors'].append(f['observed_error'])

        trace=empty_trace();state=BaseNavState();history=deque();trigger=None;switch_step=None
        arrival_before=False
        for step in range(int(np.ceil(args.policy_seconds/dt))+1):
            t=step*dt;rng_before=np.random.get_state();state_before=copy.deepcopy(state)
            frame=observe(state);record_state(trace,frame,t)
            if frame['true_error']<.01:
                arrival_before=True;break
            if step==int(np.ceil(args.policy_seconds/dt)):break
            x,wheels,arm_action=predict(frame,state,t)
            if state.done:
                history.append((t,frame['observed_error']))
                while history and t-history[0][0]>args.stall_window+dt/2:history.popleft()
            else:history.clear()
            eligible=state.done and t>=args.min_switch_time
            forced=args.switch_at is not None and t+dt/2>=args.switch_at
            stalled=(step % 50 == 0 and stalled_observations(history,t,args.near_radius,args.stall_window,args.stall_improvement))
            if eligible and (forced if args.switch_at is not None else stalled):
                switch_step=step
                # 保存动作前的真实仿真、Python侧缓存、噪声RNG和相位，两个分支从完全相同起点开始。
                saved=client.saveState()
                cache={k:copy.deepcopy(v) for k,v in sim.bot[0].__dict__.items()
                       if k.startswith(('motor_','prev_','base_'))
                       or k in ['state_buffer','applied_motor_commands','servo_motor_model']}
                sim_cache={k:copy.deepcopy(getattr(sim,k)) for k in ['step_counter','last_action','last_action_time','observation_history']
                           if hasattr(sim,k)}
                trigger={'step':step,'time_s':t,'reason':'fixed_time_diagnostic' if forced else 'near_goal_observed_stall',
                    'true_error_m':frame['true_error'],'observed_error_m':frame['observed_error']}
                snapshot=(saved,cache,sim_cache,rng_before,state_before)
                print('Takeover snapshot:',json.dumps(trigger),flush=True)
                break
            trace['observations'].append(x);trace['actions'].append(apply(wheels,arm_action))
            trace['q_des_arm'].append(np.full(7,np.nan));trace['qd_des_arm'].append(np.full(7,np.nan));trace['action_source'].append('model')

        results={};branches={}
        if trigger is not None:
            for mode in ['model','expert']:
                saved,cache,sim_cache,rng,phase=snapshot
                client.restoreState(stateId=saved)
                for k,v in cache.items():setattr(sim.bot[0],k,copy.deepcopy(v))
                for k,v in sim_cache.items():setattr(sim,k,copy.deepcopy(v))
                np.random.set_state(rng);phase=copy.deepcopy(phase)
                branch=empty_trace();qref=None;hold_start=None;held=0.;first_arrival=None
                max_step=int(np.ceil(args.recovery_seconds/dt))
                for local in range(max_step+1):
                    t=(switch_step+local)*dt
                    f=observe(phase);record_state(branch,f,t)
                    if local==0:
                        np.testing.assert_allclose(f['truth'],trace['ee_positions'][-1],atol=1e-7,rtol=0)
                        np.testing.assert_array_equal(f['ee'],trace['observed_ee_positions'][-1])
                    if f['true_error']<.01:
                        if first_arrival is None:first_arrival=t
                        if hold_start is None:hold_start=t
                        held=t-hold_start
                        if held+1e-9>=args.hold_seconds:break
                    else:hold_start=None;held=0.
                    if local==max_step:break
                    x,wheels,model_arm=predict(f,phase,t)
                    if mode=='expert':
                        # 接管时参考重置为当前实测角，不借用旧专家轨迹或把目标角直接跳过去。
                        if qref is None:qref=f['q'][4:11].copy()
                        actual,qref,qdref=arm_controller(dyn,f['qp'],f['q'][4:11],f['qd'][4:11],qref,
                            f['q'][4:11],target,f['ee'],ee_id,shoulder,lower,upper,dt,expert_params,reach_active=True)
                    else:actual=model_arm;qref_saved=np.full(7,np.nan);qdref=np.full(7,np.nan)
                    branch['observations'].append(x);branch['actions'].append(apply(wheels,actual))
                    branch['q_des_arm'].append(qref.copy() if mode=='expert' else qref_saved)
                    branch['qd_des_arm'].append(qdref.copy());branch['action_source'].append(mode)
                branches[mode]=branch
                results[mode]={'reached':first_arrival is not None,'reached_and_held':held+1e-9>=args.hold_seconds,
                    'first_arrival_time_s':first_arrival,'final_error_m':float(branch['ee_errors'][-1]),
                    'min_error_m':float(min(branch['ee_errors'])),'end_time_s':float(branch['timestamps'][-1]),
                    'continuous_hold_at_end_s':held,'arm_source':mode,'wheel_source':'frozen_learned_navigation'}
                print(mode,json.dumps(results[mode]),flush=True)
            client.removeState(saved)
        category=('model_reached_before_trigger' if arrival_before else 'no_eligible_near_stall')
        if trigger:
            eh=results['expert']['reached_and_held'];mh=results['model']['reached_and_held']
            category='expert_only_recovered_and_held' if eh and not mh else ('both_recovered_and_held' if eh and mh else ('model_only_recovered_and_held' if mh else 'neither_held'))
        output=args.output_dir or arm/'recovery_diagnostics'/datetime.now().strftime('single_%Y%m%d_%H%M%S_%f')
        output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
        def save_trace(name,tr):
            arrays={k:np.asarray(v) for k,v in tr.items()}
            for k,width in [('actions',11),('observations',26),('q_des_arm',7),('qd_des_arm',7)]:arrays[k]=arrays[k].reshape(-1,width)
            np.savez_compressed(output/(name+'.npz'),**arrays,target=target,diagnostic_only=True,
                training_data_ready=False,true_state_used_for_control=False,evaluation_position_source='pybullet_link_world_frame')
        save_trace('model_prefix',trace)
        for name,tr in branches.items():save_trace(name+'_continuation',tr)
        report={'diagnostic_only':True,'pure_model_success_rate':None,'training_data_generated':False,
            'target':target.tolist(),'target_owner':'train','train_index':owner,'seed':args.seed,
            'arm_run':str(arm),'nav_run':str(nav),'sensor_noise':args.sensor_noise,'sensor_noise_enabled':enabled,
            'policy_seconds':args.policy_seconds,'recovery_seconds_per_branch':args.recovery_seconds,
            'hold_seconds':args.hold_seconds,'true_reach_tolerance_m':.01,'trigger':trigger,
            'trigger_uses_observed_error':True,'category':category,'branches':results,
            'expert_reference_initialization':'current_measured_joint_angles_at_takeover',
            'state_and_rng_restored_for_each_branch':trigger is not None,
            'robot_config_snapshot':json.loads((root/'configs/robotnik_arm.json').read_text(encoding='utf-8')),
            'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [arm/'best_model.pt',nav/'best_model.pt',
                root/'configs/robotnik_arm.json',Path(__file__),Path(__file__).with_name('robotnik_arm_control.py')]}}
        (output/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        print('DIAGNOSTIC ONLY:',category,'\nResults:',output,flush=True)
        if not args.headless:
            import matplotlib.pyplot as plt
            plt.plot(trace['timestamps'],trace['ee_errors'],label='Model prefix')
            for name,tr in branches.items():plt.plot(tr['timestamps'],tr['ee_errors'],label=name+' continuation')
            plt.axhline(.01,color='red',ls='--',label='1cm true distance')
            if trigger:plt.axvline(trigger['time_s'],color='gray',ls=':',label='paired snapshot')
            plt.xlabel('Simulation time (s)');plt.ylabel('True EE distance (m)')
            plt.title('Recovery diagnostic; NOT pure-model success');plt.grid(alpha=.3);plt.legend();plt.tight_layout();plt.show()
        return report
    finally:client.disconnect()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',nargs=3,type=float,required=True)
    p.add_argument('--arm-run-dir',type=Path)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--sensor-noise',choices=['config','off'],default='config')
    p.add_argument('--policy-seconds',type=float,default=20.)
    p.add_argument('--recovery-seconds',type=float,default=10.)
    p.add_argument('--hold-seconds',type=float,default=1.)
    p.add_argument('--min-switch-time',type=float,default=8.)
    p.add_argument('--stall-window',type=float,default=2.)
    p.add_argument('--stall-improvement',type=float,default=.002)
    p.add_argument('--near-radius',type=float,default=.05)
    p.add_argument('--switch-at',type=float,help='Optional fixed-time diagnostic, still requires navigation done')
    p.add_argument('--output-dir',type=Path)
    a=p.parse_args()
    values=[a.policy_seconds,a.recovery_seconds,a.hold_seconds,a.stall_window,a.near_radius]
    if (not np.isfinite(a.target).all() or not all(np.isfinite(v) and v>0 for v in values)
            or a.seed<0 or not np.isfinite(a.min_switch_time) or a.min_switch_time<0
            or not np.isfinite(a.stall_improvement) or a.stall_improvement<0
            or (a.switch_at is not None and (not np.isfinite(a.switch_at) or a.switch_at<0))):p.error('Invalid diagnostic parameters')
    if a.output_dir is not None and a.output_dir.exists():raise FileExistsError(a.output_dir)
    run_probe(a)


if __name__=='__main__':main()
