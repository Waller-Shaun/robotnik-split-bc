"""Native RoboEnv screenshot noise, with true-state scoring kept out of policy.

The original noiseless evaluator is retained unchanged. This protocol reuses
its physics/control loop, but noisy FK is used ONLY for policy observations;
success is scored using a separate FK pass on cached, noise-free state.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from test_tcn import ROOT, BODYFRAME_FEATURE_VERSION, build_bodyframe_observation

NOISY_CONFIG = ROOT / "artifacts/RoboEnv/configs/robotnik_arm_noise_0p001.json"
NOISE_KEYS = ("joint_cov", "joint_vel_cov", "joint_acc_cov", "base_pos_cov",
              "base_ori_cov", "base_lin_vel_cov", "base_ang_vel_cov", "joint_torque_cov")


def validate_noise_config(path):
    config = json.loads(Path(path).read_text())
    robot = config["robot_pybullet"]
    if robot["noise_flag"] != [1] or len(robot["robot_noise"]) != 1:
        raise ValueError("Expected screenshot profile for one robot, noise_flag=[1]")
    noise = robot["robot_noise"][0]
    if set(noise) != set(NOISE_KEYS) or any(noise[key] != .001 for key in NOISE_KEYS):
        raise ValueError("All eight screenshot noise scales must be exactly 0.001")
    return config


def ground_truth_ee(sim, dyn, frame):
    import pinocchio as pin
    robot = sim.bot[0]
    # No getters here: do not draw noise or feed ground truth to controller.
    state = np.concatenate((robot.base_position, robot.base_orientation,
                            robot.motor_angles)).copy()
    q_pin = dyn.ReoderJoints2PinVec(state, "pos")
    pin.forwardKinematics(dyn.pin_model, dyn.pin_data, q_pin)
    pin.updateFramePlacements(dyn.pin_model, dyn.pin_data)
    return dyn.pin_data.oMf[frame].translation.copy(), state


def rollout(args, controller):
    import pinocchio as pin
    from simulation_and_control import pb, MotorCommands, PinWrapper
    from robotnik_arm_control import (
        configure_gripper_fingers, quaternion2bearing,
        BaseNavState, BaseControllerParams, base_controller,
    )

    roboenv = args.roboenv_dir.resolve()
    config_path = Path(getattr(args, "sim_config", NOISY_CONFIG)).resolve()
    requested_config = validate_noise_config(config_path)
    if not (roboenv / "models/summit_xl_arm_description/summit_xl_complete.urdf").exists():
        raise FileNotFoundError(f"Robot assets missing under {roboenv}")
    target = np.asarray(args.target if args.target is not None
                        else controller.manifest["targets"]["val"][0], dtype=np.float64)
    print(f"Target: {target.tolist()} ({'manual' if args.target is not None else 'first validation target'})", flush=True)
    print(f"Policy: epoch {controller.epoch}; offline val MSE {controller.val_mse:.6f}", flush=True)
    # Both constructors join config name with root/configs; an absolute filename
    # selects our unchanged experimental config, while root/models supplies the
    # pinned meshes/URDFs. No upstream configuration/assets are copied or edited.
    sim = pb.SimInterface(str(config_path), conf_file_path_ext=str(roboenv), use_gui=not args.headless)
    client = sim.pybullet_client
    times, errors, positions, observations, actions, raw_actions, inference_ms = [], [], [], [], [], [], []
    estimated_positions, ground_truth_states = [], []
    success, reason, phase_time = False, "time_limit", None
    phase_reason = "not_switched"
    clipped_steps = 0
    dt = float(sim.GetTimeStep())
    output = args.result_file or controller.run_dir / "rollouts" / (
        f"tcn_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.npz"
    )
    if output.suffix != ".npz":
        client.disconnect()
        raise ValueError("--result-file must end in .npz")
    if output.exists() or output.with_suffix(".json").exists():
        client.disconnect()
        raise FileExistsError(output)
    try:
        if not np.isclose(dt, controller.config["raw_dt_s"], rtol=0, atol=1e-9):
            raise ValueError("Simulation timestep differs from the training history timestep")
        if args.sensor_noise == "off":
            for robot in sim.bot:
                robot.noise_flag = False
        noise_enabled = [bool(robot.noise_flag) for robot in sim.bot]
        if not all(noise_enabled):
            raise ValueError("Noisy protocol requires noise enabled for every robot")
        active_noise = [{key: np.asarray(value).tolist() for key, value in robot.robot_noise.items()}
                        for robot in sim.bot]
        print("NOISE ON: eight native Gaussian getter scales = 0.001; success uses ground truth", flush=True)
        rid = sim.bot[0].bot_pybullet
        sim.SetFloorFriction(1.0)
        for joint_id in range(client.getNumJoints(rid)):
            if "wheel" in client.getJointInfo(rid, joint_id)[1].decode():
                client.changeDynamics(rid, joint_id, contactStiffness=30000,
                                      contactDamping=1000, anisotropicFriction=[1.0, 0.01, 1.0])
        sim.bot[0].servo_motor_model.set_motor_gains(400, 30)
        configure_gripper_fingers(sim, client, rid, np.full(6, 0.1))
        visual = client.createVisualShape(client.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 0.85])
        client.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, basePosition=target.tolist())
        joint_names = np.expand_dims(np.asarray(sim.getNameActiveJoints()), 0)
        dyn = PinWrapper(str(config_path), "pybullet", joint_names, ["pybullet"], False, 0, str(roboenv))
        ee_frame = dyn.pin_model.getFrameId("robot_j2s7s300_end_effector")
        if ee_frame >= dyn.pin_model.nframes:
            raise ValueError("End-effector frame is missing")
        standoff = float(np.interp(target[2], [0.1, 0.5, 1.1, 1.25], [0.80, 0.65, 0.65, 0.35]))
        bearing_goal = float(np.arctan2(target[1], target[0])) if np.linalg.norm(target[:2]) > 1e-4 else 0.0
        goal = target[:2] - standoff * np.array([np.cos(bearing_goal), np.sin(bearing_goal)])
        phase, phase_params = BaseNavState(), BaseControllerParams(nav_timeout=25.0)
        limits = np.array([3.0] * 4 + [30.0] * 7)
        cmd = MotorCommands()
        # floor keeps the evaluated horizon at or below the requested duration.
        max_steps = int(np.floor(args.duration / dt + 1e-9))
        controller.reset()
        for step in range(max_steps + 1):
            t = step * dt
            base_pos = np.asarray(sim.GetBasePosition())
            base_ori = np.asarray(sim.GetBaseOrientation())
            bearing = quaternion2bearing(base_ori[3], *base_ori[:3])
            q, qd = np.asarray(sim.GetMotorAngles(0)), np.asarray(sim.GetMotorVelocities(0))
            q_pin = dyn.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q)), "pos")
            pin.forwardKinematics(dyn.pin_model, dyn.pin_data, q_pin)
            pin.updateFramePlacements(dyn.pin_model, dyn.pin_data)
            ee = dyn.pin_data.oMf[ee_frame].translation.copy()
            true_ee, true_state = ground_truth_ee(sim, dyn, ee_frame)
            error = float(np.linalg.norm(target - true_ee))
            times.append(t)
            errors.append(error)
            positions.append(true_ee)
            estimated_positions.append(ee)
            ground_truth_states.append(true_state)
            if not np.isfinite(error):
                reason = "nonfinite_state"
                break
            if error < 0.01:
                success, reason = True, "target_reached"
                break
            if step == max_steps:
                break
            was_done = phase.done
            # The returned expert wheel speeds are discarded; only phase is used.
            base_controller(phase, base_pos, bearing, goal, bearing_goal, t, phase_params)
            if phase.done and not was_done:
                phase_time = t
                phase_reason = "navigation_timeout" if t > phase_params.nav_timeout else "navigation_complete"
                print(f"Phase 1 -> 0 at {t:.3f}s ({phase_reason})", flush=True)
            source = np.concatenate((base_pos, [np.sin(bearing), np.cos(bearing)],
                                     q[4:11], qd[4:11], target, [0.0 if phase.done else 1.0])).astype(np.float32)
            features = build_bodyframe_observation(source, ee)
            started = time.perf_counter()
            raw = controller.predict(
                features, base_position=base_pos, base_orientation=base_ori, bearing=bearing,
                goal_position=goal, goal_bearing=bearing_goal, phase=phase,
            )
            inference_ms.append((time.perf_counter() - started) * 1000)
            action = np.clip(raw, -limits, limits)
            clipped_steps += int(np.any(action != raw))
            observations.append(features)
            actions.append(action)
            raw_actions.append(raw)
            cmd.SetControlCmd(np.concatenate((action, np.zeros(6))), ["velocity"] * 4 + ["torque"] * 13)
            sim.Step(cmd, "torque")
            if step % 1000 == 0:
                print(f"t={t:.1f}s | EE error={error:.4f} m", flush=True)
            if not args.headless:
                time.sleep(dt)
    finally:
        client.disconnect()

    evaluation_split = getattr(controller, "provenance", {}).get("evaluation_split", "val")
    summary = {
        "evaluation_split": evaluation_split,
        "evaluation_protocol": "native_sensor_noise_0p001_true_state_scoring_v1",
        "success_position_source": "noise_free_cached_state_forward_kinematics",
        "policy_position_source": "noisy_simulator_getters_forward_kinematics",
        "ground_truth_exposed_to_policy": False,
        "robot_noise_effective": active_noise,
        "noise_scale_semantics": "numpy.random.normal scale = standard deviation, despite _cov names",
        "noise_sampling": "native simulator: independent draws per getter call, including internal servo reads",
        "simulation_config_parameters": requested_config,
        "noise_evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "success": success, "termination_reason": reason, "target": target.tolist(),
        "final_error_m": float(errors[-1]) if np.isfinite(errors[-1]) else None,
        "minimum_error_m": float(np.nanmin(errors)) if np.isfinite(errors).any() else None,
        "end_time_s": float(times[-1]), "first_reach_time_s": float(times[-1]) if success else None,
        "clipped_steps": clipped_steps, "action_steps": len(actions),
        "clipped_fraction": clipped_steps / len(actions) if actions else 0.0,
        "reach_tolerance_m": 0.01, "required_hold_s": 0.0, "min_success_time_s": 0.0,
        "duration_limit_s": args.duration, "raw_dt_s": dt, "policy_update_dt_s": dt,
        "window_size": controller.window_size, "history_stride": controller.stride,
        "history_interval_s": dt * controller.stride, "padding": "repeat_first_observation",
        "sensor_noise_mode": args.sensor_noise, "sensor_noise_enabled": noise_enabled,
        "phase_switch_time_s": phase_time, "phase_switch_reason": phase_reason,
        "phase_source": "expert_navigation_rules_commands_discarded", "forced_parking": False,
        "model_run": str(controller.run_dir), "model_epoch": controller.epoch,
        "policy_type": getattr(controller, "policy_type", "TCNPolicy"),
        "effective_context_samples": controller.config.get("effective_context_samples", controller.window_size),
        "additional_policy_provenance": getattr(controller, "provenance", {}),
        "feature_version": BODYFRAME_FEATURE_VERSION, "evaluation_seed": args.seed,
        "simulation_config": str(config_path),
        "simulation_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(getattr(controller, "checkpoint_path", controller.run_dir / "best_model.pt").read_bytes()).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_source_sha256": hashlib.sha256(getattr(controller, "model_source_path", Path(__file__).with_name("tcn_model.py")).read_bytes()).hexdigest(),
        "roboenv_assets": str(roboenv), "torch_version": str(torch.__version__),
        "pinocchio_version": pin.__version__, "numpy_version": np.__version__,
        "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms else None,
        "note": f"Single-target {evaluation_split} rollout; not a full split success rate",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, target=target, timestamps=np.asarray(times), ee_positions=np.asarray(positions),
        estimated_ee_positions=np.asarray(estimated_positions),
        ground_truth_states=np.asarray(ground_truth_states),
        ee_errors=np.asarray(errors), observations=np.asarray(observations, dtype=np.float32).reshape(-1, 26),
        actions=np.asarray(actions, dtype=np.float32).reshape(-1, 11),
        raw_actions=np.asarray(raw_actions, dtype=np.float32).reshape(-1, 11),
        action_timestamps=np.asarray(times[:len(actions)]), inference_ms=np.asarray(inference_ms),
        success=success, termination_reason=reason, summary_json=json.dumps(summary, allow_nan=False),
    )
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Success: {success}; reason: {reason}")
    minimum_text = "nonfinite" if summary["minimum_error_m"] is None else f"{summary['minimum_error_m']:.6f}"
    final_text = "nonfinite" if summary["final_error_m"] is None else f"{summary['final_error_m']:.6f}"
    print(f"Minimum EE error: {minimum_text} m; final: {final_text} m")
    print(f"End time: {times[-1]:.3f} s; clipped steps: {clipped_steps}/{len(actions)}")
    print(f"Saved: {output}")
