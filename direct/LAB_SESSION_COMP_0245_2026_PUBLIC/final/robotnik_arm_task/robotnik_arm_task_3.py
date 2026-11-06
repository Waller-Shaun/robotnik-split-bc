import numpy as np
import time
import os
from pathlib import Path
from datetime import datetime

import argparse
import json  # [Task3 change] Save the disturbance configuration with each episode.

# Headless mode and step cap for CI / direct runs
headless = os.environ.get("ROBOENV_HEADLESS", "").strip().lower() in ("1", "true", "yes")
max_steps = int(os.environ.get("ROBOENV_MAX_STEPS", "40000"))

import matplotlib
if headless:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as p
import pinocchio as pin
from simulation_and_control import pb, MotorCommands, PinWrapper
from robotnik_arm_control import (
    ArmControllerParams, BaseControllerParams, BaseNavState,
    arm_controller, base_controller, configure_gripper_fingers, quaternion2bearing,
)


# ==============================================================================
# TARGET SPECIFICATION
# Fixed 3D Cartesian coordinates for the Kinova end-effector in the world frame:
# [X (forward/meters), Y (lateral/meters), Z (height/meters)]
parser = argparse.ArgumentParser()
parser.add_argument(    
    "--target",
    type=float,
    nargs=3,
    default=[-0.6, 1.1, 0.3],
    metavar=("X", "Y", "Z"),
)
# [Task3 change] Small, bounded torque pulses; all units are explicit.
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--noise-std", type=float, default=0.5, help="Torque noise std (N m); 0 disables noise")
parser.add_argument("--noise-duration", type=float, default=0.10, help="Pulse duration (s)")
parser.add_argument("--noise-period", type=float, default=0.80, help="Pulse start-to-start period (s)")
parser.add_argument("--noise-delay", type=float, default=0.30, help="Delay after navigation ends (s)")
parser.add_argument("--hold-seconds", type=float, default=1.0, help="Continuous in-tolerance hold required (s)")
parser.add_argument("--output-dir", type=Path, default=None)
args = parser.parse_args()
if not np.isfinite([args.noise_std, args.noise_duration, args.noise_period,
                   args.noise_delay, args.hold_seconds, *args.target]).all():
    parser.error("All numeric parameters must be finite")
if args.seed < 0 or args.noise_std < 0 or args.noise_delay < 0 or args.hold_seconds <= 0:
    parser.error("seed/std/delay must be nonnegative; hold-seconds must be positive")
if not 0 < args.noise_duration < args.noise_period:
    parser.error("Require 0 < noise-duration < noise-period to leave recovery gaps")
TARGET_EE_POS = np.array(args.target)
#TARGET_EE_POS = np.array([-0.6, 1.1, 0.3])

# Fixed gripper finger configuration (angles in radians, held fixed during simulation):
# [finger_1, finger_tip_1, finger_2, finger_tip_2, finger_3, finger_tip_3]
# Set to None to use initial configuration from config/URDF, or an array of 6 angles.
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Max time (s) allowed for the base to reach its standoff pose before the arm starts reaching.
# Targets behind the robot need a large turn, so this must be generous.
BASE_NAV_TIMEOUT = 25.0
# Max EE distance from the shoulder the arm is asked to reach (m). Targets further away are projected
# onto this sphere, so the arm settles at the closest pose instead of fully stretching (singularity).
ARM_REACH_LIMIT = 0.85
# Base-to-target horizontal standoff (m) as a function of target height: larger for low targets,
# default for mid heights, smaller for high targets (far above the shoulder, which sits ~0.49 m high).
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]
# Stop once the EE is this close to the target (m); max iterations set via ROBOENV_MAX_STEPS
REACH_TOLERANCE = 0.01
# [Task3 change] Preserve the original minimum evaluation time, then require
# at least one uninterrupted hold interval. Reaching alone no longer ends a run.
MIN_EPISODE_SECONDS = 8.0
MAX_NOISE_SIGMAS = 3.0
# ==============================================================================


# [Task3 change] Piecewise-constant bounded pulses, rather than fresh noise
# every millisecond. Randomness is local and reproducible for a given seed.
class TorquePulseNoise:
    def __init__(self, seed, std, duration_steps, period_steps):
        self.rng = np.random.default_rng(seed)
        self.std = std
        self.duration_steps = duration_steps
        self.period_steps = period_steps
        self.last_pulse = -1
        self.value = np.zeros(7)

    def sample(self, relative_step, enabled):
        if not enabled or self.std == 0 or relative_step < 0:
            return np.zeros(7)
        pulse, offset = divmod(relative_step, self.period_steps)
        if offset >= self.duration_steps:
            return np.zeros(7)
        if pulse != self.last_pulse:
            self.value = np.clip(self.rng.normal(0.0, self.std, size=7),
                                 -MAX_NOISE_SIGMAS * self.std,
                                 MAX_NOISE_SIGMAS * self.std)
            self.last_pulse = pulse
        return self.value.copy()


def main():
    # --------------------------------------------------------------------------
    # Configuration and Simulation Setup
    # --------------------------------------------------------------------------
    conf_file_name = "robotnik_arm.json"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=not headless)

    # Configure realistic ground contact and skid-steer friction dynamics
    sim.SetFloorFriction(1.0)
    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet
    wheel_joint_ids = [
        j for j in range(client.getNumJoints(rid))
        if "wheel" in client.getJointInfo(rid, j)[1].decode()
    ]
    for j in wheel_joint_ids:
        client.changeDynamics(
            rid, j,
            contactStiffness=30000,
            contactDamping=1000,
            anisotropicFriction=[1.0, 0.01, 1.0]
        )
    sim.bot[0].servo_motor_model.set_motor_gains(400, 30)

    # --------------------------------------------------------------------------
    # Gripper Fingers: Fix gripper finger configuration
    # --------------------------------------------------------------------------
    target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)

    # Visualize the target reach reference in the PyBullet 3D environment as a red dot
    target_pos_list = TARGET_EE_POS.tolist()
    visual_shape_id = client.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.03,
        rgbaColor=[1.0, 0.0, 0.0, 0.85]
    )
    client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=target_pos_list
    )
    client.addUserDebugPoints(
        pointPositions=[target_pos_list],
        pointColorsRGB=[[1.0, 0.0, 0.0]],
        pointSize=10
    )

    # Active joints from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names_arr = np.expand_dims(np.array(ext_names), axis=0)
    source_names = ["pybullet"]

    # Pinocchio dynamic & kinematic model
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names_arr, source_names, False, 0, cur_dir)

    controlled_frame_name = "robot_j2s7s300_end_effector"
    ee_frame_id = dyn_model.pin_model.getFrameId(controlled_frame_name)
    shoulder_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2")

    init_joint_angles = np.array(sim.GetMotorAngles(0))
    init_base_pos = np.array(sim.GetBasePosition())
    init_base_ori = np.array(sim.GetBaseOrientation())

    # Forward kinematics for initial end-effector position
    init_q_all = np.concatenate((init_base_pos, init_base_ori, init_joint_angles))
    init_q_pin = dyn_model.ReoderJoints2PinVec(init_q_all, "pos")
    pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, init_q_pin)
    pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
    init_ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()

    print("==================================================================")
    print(f"Target Cartesian EE Position : {TARGET_EE_POS}")
    print(f"Initial Cartesian EE Position: {np.round(init_ee_pos, 4)}")
    print(f"Initial Cartesian Error (m)  : {np.linalg.norm(TARGET_EE_POS - init_ee_pos):.4f}")
    print("==================================================================")

    # Joint limits
    lower_limits, upper_limits = sim.GetBotJointsLimit()

    time_step = sim.GetTimeStep()
    current_time = 0

    # --------------------------------------------------------------------------
    # Coordinated Motion Planning: Base Navigation & Arm Reach
    # --------------------------------------------------------------------------
    # Standoff distance between robot base center and target, depending on target height.
    standoff = float(np.interp(TARGET_EE_POS[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    target_xy = TARGET_EE_POS[:2]
    target_bearing = np.arctan2(target_xy[1], target_xy[0]) if np.linalg.norm(target_xy) > 1e-4 else 0.0

    # Mobile base goal location facing the target
    base_goal_pos = target_xy - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])
    base_goal_bearing = target_bearing

    # Base navigation controller (align -> drive -> orient -> done), see BaseControllerParams
    base_state = BaseNavState()
    base_navigation_done = base_state.done
    base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)

    # Arm controller (differential IK + joint PD), see ArmControllerParams
    arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
    q_init_arm = init_joint_angles[4:11].copy()
    q_des_arm = q_init_arm.copy()
    qd_des_arm = np.zeros(7)
    arm_lower = np.array(lower_limits[4:11])
    arm_upper = np.array(upper_limits[4:11])
    # [Task3 change] A stall can be intentional during a disturbance/recovery.
    # Use a finite step cap instead of the old check_stop/stall detector.
    step_limit = max_steps if max_steps > 0 else 40000
    sim_step = 0
    navigation_done_step = None
    first_reach_time = None
    hold_start = None
    held_seconds = 0.0
    best_error = np.inf
    success = False
    termination_reason = "step_limit"
    noise_disabled = False
    delay_steps = max(0, int(round(args.noise_delay / time_step)))
    duration_steps = max(1, int(round(args.noise_duration / time_step)))
    period_steps = max(1, int(round(args.noise_period / time_step)))
    if duration_steps >= period_steps:
        raise ValueError("Noise duration/period must leave at least one recovery step")
    pulse_noise = TorquePulseNoise(args.seed, args.noise_std, duration_steps, period_steps)
    print(f"Task3: std={args.noise_std} N m; seed={args.seed}; hold={args.hold_seconds}s")

    # Command interface
    cmd = MotorCommands()

    # Data logging for plotting
    time_all = []
    ee_pos_all = []
    ee_error_all = []
    base_pos_all = []
    q_mes_all = []
    qd_mes_all = []
    q_d_all = []
    qd_d_all = []

    #加入额外记录 初始化
    observations = []
    actions = []
    timestamps = []
    # [Task3 change] Labels and executed actions are deliberately separate.
    applied_actions = []
    requested_torque_noise_all = []
    applied_torque_noise_all = []
    disturbance_active_all = []
    hold_active_all = []
    ee_positions_all = []
    ee_dist_errors_all = []
    base_orientations_all = []
    desired_arm_angles_all = []
    desired_arm_velocities_all = []

    # --------------------------------------------------------------------------
    # Main Simulation Loop
    # --------------------------------------------------------------------------
    while True:
        # Measure current states
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))

        # Full kinematic state vector in Pinocchio format
        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")

        # Compute end-effector forward kinematics
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_error = TARGET_EE_POS - ee_pos
        ee_dist_error = np.linalg.norm(ee_error)
        # [Task3 change] Termination is evaluated on the CURRENT observed state,
        # before applying another action; terminal state is saved separately.
        best_error = min(best_error, float(ee_dist_error))
        if not np.isfinite(np.concatenate((base_pos, base_ori, q_mes, qd_mes, [ee_dist_error]))).all():
            termination_reason = "nonfinite_state"
            break

        # ----------------------------------------------------------------------
        # 1. Mobile Base Control (Wheel velocities)
        # ----------------------------------------------------------------------
        wheel_cmds = base_controller(
            base_state, base_pos, base_bearing, base_goal_pos, base_goal_bearing, current_time, base_params
        )
        base_navigation_done = base_state.done
        # [23D change] Record the phase AFTER the base controller updates it.
        # 1 = navigation active; 0 = navigation done (including timeout).
        # This is a controller phase flag, not measured wheel motion.
        base_active = 0.0 if base_navigation_done else 1.0
        # [Task3 change] Start disturbances only after the arm is enabled.
        if base_navigation_done and navigation_done_step is None:
            navigation_done_step = sim_step

        if base_navigation_done and ee_dist_error < REACH_TOLERANCE:
            if first_reach_time is None:
                first_reach_time = current_time
                noise_disabled = True  # No more noise after first arrival.
                print(f"First arrival at {current_time:.3f}s; noise disabled; holding")
            if hold_start is None:
                hold_start = current_time
            held_seconds = current_time - hold_start
            if (held_seconds + 1e-9 >= args.hold_seconds
                    and current_time >= MIN_EPISODE_SECONDS):
                success = True
                termination_reason = "target_reached_and_held"
                break
        else:
            # Leaving tolerance resets the hold timer, but does NOT restart noise.
            hold_start = None
            held_seconds = 0.0
        if sim_step >= step_limit:
            termination_reason = "step_limit"
            break

        # ----------------------------------------------------------------------
        # 2. Arm Control (reaching via differential IK + gravity-compensated joint PD)
        # ----------------------------------------------------------------------
        # While the base is travelling the arm holds its nominal configuration.
        torque_arm, q_des_arm, qd_des_arm = arm_controller(
            dyn_model, q_pin, q_mes[4:11], qd_mes[4:11], q_des_arm, q_init_arm, TARGET_EE_POS, ee_pos,
            ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, time_step, arm_params,
            reach_active=base_navigation_done,
        )

        # ----------------------------------------------------------------------
        # 3. Gripper / Finger Control (Fingers locked at fixed configuration)
        # ----------------------------------------------------------------------
        torque_fingers = np.zeros(6)

        # ----------------------------------------------------------------------
        # 保存执行前状态st，目标g，专家动作at
        observation = np.concatenate([
            base_pos,
            [np.sin(base_bearing), np.cos(base_bearing)],
            q_mes[4:11],
            qd_mes[4:11],
            TARGET_EE_POS,
            # [23D change] Column 22: binary phase flag, always 0.0 or 1.0.
            [base_active],
        ])
        
        action = np.concatenate([wheel_cmds, torque_arm])

        # [Task3 change] Expert actions are clean supervised labels.
        # Noise changes the transition, so the expert labels NEW recovery states
        # on subsequent steps. Keep the expert's internal references intact.
        relative_step = (sim_step - navigation_done_step - delay_steps
                         if navigation_done_step is not None else -1)
        requested_noise = pulse_noise.sample(
            relative_step, enabled=base_navigation_done and not noise_disabled)
        applied_arm = np.clip(torque_arm + requested_noise,
                              -arm_params.max_torque, arm_params.max_torque)
        executed_action = np.concatenate((wheel_cmds, applied_arm))
        effective_noise = applied_arm - torque_arm
        if not np.isfinite(np.concatenate((action, executed_action))).all():
            termination_reason = "nonfinite_action"
            break
        applied_actions.append(executed_action.copy())
        requested_torque_noise_all.append(requested_noise.copy())
        applied_torque_noise_all.append(effective_noise.copy())
        disturbance_active_all.append(bool(np.any(requested_noise != 0.0)))
        hold_active_all.append(hold_start is not None)
        ee_positions_all.append(ee_pos.copy())
        ee_dist_errors_all.append(float(ee_dist_error))
        base_orientations_all.append(base_ori.copy())
        desired_arm_angles_all.append(q_des_arm.copy())
        desired_arm_velocities_all.append(qd_des_arm.copy())

        observations.append(observation.copy())
        actions.append(action.copy())
        timestamps.append(current_time)

        # ----------------------------------------------------------------------
        # Send Multi-Interface Control Commands
        # ----------------------------------------------------------------------
        # 4 wheels (velocity) + 7 arm joints (torque) + 6 fingers (torque)
        # [Task3 change] Only the executed arm torques include disturbance.
        # Wheel control, 23D inputs and clean 11D labels keep their original meaning.
        cmd_all = np.concatenate((wheel_cmds, applied_arm, torque_fingers))
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        sim.Step(cmd, "torque")
        sim_step += 1  # [Task3 change] Count every executed transition, including manual exit.

        # Visualization in Pinocchio GUI if active
        if dyn_model.visualizer:
            for index in range(len(sim.bot)):
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)

        # Exit logic with 'q' key in PyBullet GUI window
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            # [Task3 change] Record interruption separately from task success.
            termination_reason = "user_exit"
            break

        # Data logging for plotting
        time_all.append(current_time)
        ee_pos_all.append(ee_pos)
        ee_error_all.append(ee_error)
        base_pos_all.append(base_pos)
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        current_desired_q = np.concatenate((q_mes[:4], q_des_arm, target_finger_angles))
        current_desired_qd = np.concatenate((wheel_cmds, qd_des_arm, np.zeros(6)))
        q_d_all.append(current_desired_q)
        qd_d_all.append(current_desired_qd)

        if not headless:
            time.sleep(0.001)

        # [Task3 change] Integer steps keep pulse/hold timing reproducible.
        current_time = sim_step * time_step
        if int(current_time / time_step) % 500 == 0:
            print(f"t={current_time:.2f}s | Base pos: {np.round(base_pos[:2], 3)} | EE pos: {np.round(ee_pos, 3)} | EE error: {ee_dist_error:.4f}m | yaw {base_bearing:.2f} phase {base_state.phase} nav_done {base_navigation_done}")

    # [Task3 change] Re-read terminal state (also accurate after manual exit).
    base_pos = np.array(sim.GetBasePosition())
    base_ori = np.array(sim.GetBaseOrientation())
    q_mes = np.array(sim.GetMotorAngles(0))
    qd_mes = np.array(sim.GetMotorVelocities(0))
    q_pin = dyn_model.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q_mes)), "pos")
    pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
    pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
    ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
    ee_dist_error = float(np.linalg.norm(TARGET_EE_POS - ee_pos))
    best_error = min(best_error, ee_dist_error)
    current_time = sim_step * time_step
    print(f"Task3 success={success}; reason={termination_reason}; continuous hold={held_seconds:.3f}s")

    # Final summary report
    print("==================================================================")
    print(f"Final Base Position          : {np.round(base_pos[:2], 4)}")
    print(f"Final End-Effector Position  : {np.round(ee_pos, 4)}")
    print(f"Target End-Effector Position : {TARGET_EE_POS}")
    print(f"Final Cartesian Error (m)    : {np.linalg.norm(TARGET_EE_POS - ee_pos):.4f}")
    print(f"Fixed Finger Angles          : {np.round(target_finger_angles, 4)}")
    print(f"Final Finger Angles          : {np.round(q_mes[11:17], 4)}")
    print(f"Max Finger Deviation (rad)   : {np.max(np.abs(q_mes[11:17] - target_finger_angles)):.6f}")
    print("==================================================================")
    
    # ----------------------------------------------------------------------
    # 保存单条专家轨迹数据
    # [23D change] Keep these trajectories separate from the 22D baseline.
    # [Task3 change] Preserve the original clean dataset for comparison.
    data_dir = (args.output_dir if args.output_dir is not None else
                Path(__file__).resolve().parents[3] / "RoboEnv" / "expert_data_23d_noise")
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / f"expert_episode_{datetime.now():%Y%m%d_%H%M%S_%f}.npz"
    np.savez_compressed(
        data_path,
        observations=np.asarray(observations, dtype=np.float32).reshape(-1, 23),
        actions=np.asarray(actions, dtype=np.float32).reshape(-1, 11),
        timestamps=np.asarray(timestamps),
        target=TARGET_EE_POS,
        final_error=ee_dist_error,
        # [Task3 change] Applied actions are NOT the supervised training labels.
        applied_actions=np.asarray(applied_actions, dtype=np.float32).reshape(-1, 11),
        requested_torque_noise=np.asarray(requested_torque_noise_all, dtype=np.float32).reshape(-1, 7),
        applied_torque_noise=np.asarray(applied_torque_noise_all, dtype=np.float32).reshape(-1, 7),
        disturbance_active=np.asarray(disturbance_active_all, dtype=bool),
        hold_active=np.asarray(hold_active_all, dtype=bool),
        ee_positions=np.asarray(ee_positions_all, dtype=np.float64).reshape(-1, 3),
        ee_errors=np.asarray(ee_dist_errors_all, dtype=np.float64),
        base_orientations=np.asarray(base_orientations_all, dtype=np.float64).reshape(-1, 4),
        q_des_arm=np.asarray(desired_arm_angles_all, dtype=np.float64).reshape(-1, 7),
        qd_des_arm=np.asarray(desired_arm_velocities_all, dtype=np.float64).reshape(-1, 7),
        success=success,
        termination_reason=termination_reason,
        best_error=best_error,
        first_reach_time=np.nan if first_reach_time is None else first_reach_time,
        final_timestamp=current_time,
        final_ee_position=ee_pos,
        hold_seconds_required=args.hold_seconds,
        continuous_hold_seconds=held_seconds,
        navigation_done_time=np.nan if navigation_done_step is None else navigation_done_step * time_step,
        seed=args.seed,
        dataset_version="task3_torque_pulses_hold_v1",
        action_label_definition="clean_expert_action_at_observation; applied_actions_drive_simulation",
        noise_config_json=json.dumps({
            "std_nm": args.noise_std, "clip_sigmas": MAX_NOISE_SIGMAS,
            "duration_seconds": duration_steps * time_step,
            "period_seconds": period_steps * time_step,
            "delay_seconds": delay_steps * time_step,
            "joints": list(range(1, 8)), "disabled_after_first_arrival": True,
            "minimum_episode_seconds": MIN_EPISODE_SECONDS,
            "step_limit": step_limit,
        }),
        # [23D change] Metadata documents the new input schema.
        input_dim=23,
        base_active_column=22,
        base_active_definition="1=navigation_active;0=navigation_done_including_timeout",
    )
    print(f"Expert trajectory saved to: {data_path}")
    # ----------------------------------------------------------------------

    # [Task3 change] Release the simulator after recording the final state.
    client.disconnect()
    if headless:
        print(f"Headless run finished at t={current_time:.2f}s")
        return

    # --------------------------------------------------------------------------
    # Plotting Performance
    # --------------------------------------------------------------------------
    ee_pos_all = np.array(ee_pos_all)
    ee_error_all = np.array(ee_error_all)
    time_all = np.array(time_all)

    # 1. Cartesian end-effector coordinates tracking
    plt.figure(figsize=(10, 8))
    coords = ['X', 'Y', 'Z']
    for idx in range(3):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_all, ee_pos_all[:, idx], label=f'Actual EE {coords[idx]}', color='b')
        plt.axhline(y=TARGET_EE_POS[idx], color='r', linestyle='--', label=f'Target EE {coords[idx]}')
        plt.ylabel(f'{coords[idx]} Position (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('End-Effector Cartesian Position Reaching')
    plt.tight_layout()
    plt.show()

    # 2. Cartesian distance error over time
    plt.figure(figsize=(8, 4))
    ee_dist_norm = np.linalg.norm(ee_error_all, axis=1)
    plt.plot(time_all, ee_dist_norm, label='EE Euclidean Error (m)', color='g')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Cartesian End-Effector Reaching Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Arm joint position tracking
    plt.figure(figsize=(10, 8))
    for i in range(7):
        plt.subplot(4, 2, i + 1)
        plt.plot(time_all, [q[4 + i] for q in q_mes_all], label=f'Measured J{i+1}')
        plt.plot(time_all, [q[4 + i] for q in q_d_all], label=f'Desired J{i+1}', linestyle='--')
        plt.ylabel(f'J{i+1} (rad)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('Arm Joint Position Profiles')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
