"""Single-target TCN/controlled-MLP rollout using shared evaluation conditions.

Reuse the original simulation configuration, wheel/contact gains, navigation
phase rules, fixed fingers, 1 cm anytime success criterion and action limits.
Only the policy and its historical preprocessing change. No expert actions
are executed. Robot assets stay in the pinned external RoboEnv checkout.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from tcn_model import TCNPolicy
from mlp_model import MLPPolicy

TASK_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))
from robotnik_bodyframe_features import (  # noqa: E402
    BODYFRAME_FEATURE_VERSION, build_bodyframe_observations, build_bodyframe_observation,
)


def latest_full_run():
    runs = sorted((ROOT / "RoboEnv/bc_training_tcn_ring").glob("run_*_full"), reverse=True)
    for run in runs:
        finished = run / "training_finished.json"
        if finished.exists() and json.loads(finished.read_text())["status"] == "completed":
            return run
    raise FileNotFoundError("No completed full TCN training run; supply --run-dir")


class TCNController:
    """CPU policy with a physics-rate buffer and strided history selection.

    At every 1 ms tick, take frames t-190, t-180, ..., t for the default
    window/stride. Before t=190, negative indices repeat the episode's first
    frame. Every prediction uses a sliding window, with no action hold.
    """

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir).resolve()
        checkpoint = torch.load(self.run_dir / "best_model.pt", map_location="cpu", weights_only=True)
        classes = {"TCNPolicy": TCNPolicy, "MLPPolicy": MLPPolicy}
        self.policy_type = checkpoint.get("model_type")
        if self.policy_type not in classes or checkpoint.get("schema_version") != 1:
            raise ValueError("Unsupported controlled BC checkpoint")
        self.config = checkpoint["config"]
        if self.config["feature_version"] != BODYFRAME_FEATURE_VERSION:
            raise ValueError("Checkpoint feature definition mismatch")
        if self.config["architecture"]["input_dim"] != 26:
            raise ValueError("Expected 26D features")
        if self.config["start_padding"] != "repeat_first_observation":
            raise ValueError("Unsupported history padding")
        self.window_size, self.stride = self.config["window_size"], self.config["stride"]
        if self.window_size < 1 or self.stride < 1:
            raise ValueError("Invalid history settings")
        self.history = deque(maxlen=1 + (self.window_size - 1) * self.stride)
        self.offsets = np.arange(self.window_size - 1, -1, -1) * self.stride
        self.model = classes[self.policy_type](**self.config["architecture"]).eval()
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError("Non-finite checkpoint parameters")
        self.stats = {k: v.numpy().copy() for k, v in checkpoint["normalization"].items()}
        with np.load(self.run_dir / "normalization.npz", allow_pickle=False) as stored:
            for key, dim in (("x_mean", 26), ("x_std", 26), ("y_mean", 11), ("y_std", 11)):
                value = self.stats[key]
                if value.shape != (dim,) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid normalization: {key}")
                if key.endswith("std") and np.any(value <= 0):
                    raise ValueError(f"Nonpositive standard deviation: {key}")
                np.testing.assert_array_equal(value, stored[key])
        self.epoch = checkpoint["epoch"]
        self.val_mse = checkpoint["validation"]["mse"]
        self.manifest_path = self.run_dir / "data_split.json"
        if hashlib.sha256(self.manifest_path.read_bytes()).hexdigest() != self.config["source_manifest_sha256"]:
            raise ValueError("Saved split manifest fingerprint mismatch")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def reset(self):
        self.history.clear()

    def push(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (26,) or not np.isfinite(observation).all():
            raise ValueError("Expected finite 26D feature vector")
        self.history.append(observation.copy())
        frames = np.asarray(self.history, dtype=np.float32)
        indices = np.maximum(len(frames) - 1 - self.offsets, 0)
        return frames[indices]

    def predict(self, observation, **context):
        window = self.push(observation)
        normalized = np.ascontiguousarray(
            (window - self.stats["x_mean"]) / self.stats["x_std"], dtype=np.float32
        )
        with torch.inference_mode():
            output = self.model(torch.from_numpy(normalized).unsqueeze(0))[0].numpy()
        action = output * self.stats["y_std"] + self.stats["y_mean"]
        if not np.isfinite(action).all():
            raise FloatingPointError("Policy produced non-finite actions")
        return action


def check_history(controller):
    """Compare online physics-rate buffering with offline training windows."""
    data_dir = Path(controller.manifest["data_dir"].replace("@WORKSPACE@", str(ROOT)))
    path = data_dir / controller.manifest["splits"]["train"][0]
    ncheck = max(250, controller.history.maxlen + controller.stride + 1)
    with np.load(path, allow_pickle=False) as source:
        features = build_bodyframe_observations(
            source["observations"][:ncheck], source["ee_positions"][:ncheck]
        )
    controller.reset()
    for frame, observation in enumerate(features):
        actual = controller.push(observation)
        indices = np.maximum(frame - controller.offsets, 0)
        np.testing.assert_array_equal(actual, features[indices])
    controller.reset()
    action = controller.predict(features[0])
    assert action.shape == (11,) and np.isfinite(action).all()
    controller.reset()
    print("PASS: checkpoint load, normalization, online/offline history alignment, 11D action", flush=True)


def rollout(args, controller):
    import pinocchio as pin
    from simulation_and_control import pb, MotorCommands, PinWrapper
    from robotnik_arm_control import (
        configure_gripper_fingers, quaternion2bearing,
        BaseNavState, BaseControllerParams, base_controller,
    )

    roboenv = args.roboenv_dir.resolve()
    config_path = (ROOT / "artifacts/RoboEnv/configs/robotnik_arm.json").resolve()
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
            error = float(np.linalg.norm(target - ee))
            times.append(t)
            errors.append(error)
            positions.append(ee)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--roboenv-dir", type=Path, default=ROOT / "external/RoboEnv")
    parser.add_argument("--target", type=float, nargs=3)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sensor-noise", choices=("off", "config"), default="off")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration < 0.001 or args.seed < 0:
        parser.error("duration must be finite and >= 1 ms; seed must be nonnegative")
    if args.target is not None and not np.isfinite(args.target).all():
        parser.error("target must be finite")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    controller = TCNController(args.run_dir or latest_full_run())
    print(f"Policy run: {controller.run_dir}; type: {controller.policy_type}", flush=True)
    check_history(controller)
    if not args.check_only:
        rollout(args, controller)


if __name__ == "__main__":
    main()
