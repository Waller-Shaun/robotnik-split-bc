import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import pinocchio as pin

from simulation_and_control import pb, MotorCommands, PinWrapper
from robotnik_arm_control import (
    configure_gripper_fingers,
    quaternion2bearing,
    # [23D change] Reuse navigation rules ONLY to generate the phase input.
    BaseNavState, BaseControllerParams, base_controller,
)
# [Architecture test change] 从检查点配置构建网络，兼容旧双128和新256/256/128模型。
# 只改变模型加载；阶段判断、动作限幅和成功标准保持原样。
class CheckpointPolicy(torch.nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        layers = []
        dimensions = [23, *hidden_dims, 11]
        for index, (input_size, output_size) in enumerate(zip(dimensions, dimensions[1:])):
            layers.append(torch.nn.Linear(input_size, output_size))
            if index < len(dimensions) - 2:
                layers.append(torch.nn.ReLU())
        # 与训练脚本保持相同的参数名 network.0.weight 等。
        self.network = torch.nn.Sequential(*layers)

    def forward(self, observation):
        return self.network(observation)


def load_policy_from_checkpoint(checkpoint):
    config = checkpoint.get("config", {})
    hidden_dims = config.get("hidden_dims", [128, 128])
    if (not isinstance(hidden_dims, (list, tuple)) or not hidden_dims
            or any(type(size) is not int or size <= 0 for size in hidden_dims)):
        raise ValueError("Invalid hidden_dims in checkpoint config")
    if config.get("input_dim", 23) != 23 or config.get("output_dim", 11) != 11:
        raise ValueError("Expected a 23-input, 11-output policy")
    if config.get("activation", "ReLU") != "ReLU":
        raise ValueError("This evaluator supports ReLU checkpoints only")
    model = CheckpointPolicy(hidden_dims)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite parameters")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, nargs=3, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=10.0)
    # [23D change] Select a specific trained run explicitly when needed.
    parser.add_argument("--run-dir", type=Path, default=None)
    # [Batch evaluation change] Explicit noise policy, independent of config edits.
    # Existing single-run behavior is preserved by the default "config" mode.
    parser.add_argument("--sensor-noise", choices=["off", "config"], default="config")
    parser.add_argument("--result-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration 必须大于 0")

    # [Batch evaluation change] Seed observation noise if config mode is used.
    if args.seed < 0 or not np.isfinite(args.duration):
        parser.error("seed must be nonnegative and duration must be finite")
    if args.target is not None and not np.isfinite(args.target).all():
        parser.error("target coordinates must be finite")
    np.random.seed(args.seed)
    if args.result_file is not None and args.result_file.exists():
        raise FileExistsError(f"Refusing to overwrite rollout: {args.result_file}")

    roboenv_dir = Path(__file__).resolve().parents[3] / "RoboEnv"
    # [23D change] Default to the new 23D training run.
    run_dir = args.run_dir or (
        roboenv_dir / "bc_training_23d" / "run_20261028_012502_944390"
    )

    # 默认选验证集的第一个目标，先做开发测试。
    # 正式测试集留到方法确定后再评估。
    manifest = json.loads(
        (run_dir / "data_split.json").read_text(encoding="utf-8")
    )
    target = np.array(
        args.target
        if args.target is not None
        else manifest["targets"]["val"][0],
        dtype=np.float64,
    )

    # 小型网络每次只预测一个状态，先使用 CPU 推理
    torch.set_num_threads(1)
    checkpoint = torch.load(
        run_dir / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    # [Architecture test change] 加载对应结构；旧实验仍可用原来的命令测试。
    model = load_policy_from_checkpoint(checkpoint)
    print("Network:", " -> ".join(map(str, [23, *checkpoint.get("config", {}).get("hidden_dims", [128, 128]), 11])))
    model.eval()

    with np.load(run_dir / "normalization.npz") as data:
        x_mean = data["x_mean"].copy()
        x_std = data["x_std"].copy()
        y_mean = data["y_mean"].copy()
        y_std = data["y_std"].copy()

    # [23D change] Fail early if the model and preprocessing do not match.
    if x_mean.shape != (23,) or x_std.shape != (23,):
        raise ValueError("Expected 23D normalization statistics")
    if y_mean.shape != (11,) or y_std.shape != (11,):
        raise ValueError("Expected 11D action statistics")
    if x_mean[22] != 0.0 or x_std[22] != 1.0:
        raise ValueError("Phase column must remain binary: mean=0, std=1")

    print(f"目标：{target}")
    print(f"加载模型轮次：{checkpoint['epoch']}")
    print(f"对应验证 MSE：{checkpoint['val_mse']:.6f}")

    # 与采集专家数据时保持相同的仿真设置
    sim = pb.SimInterface(
        "robotnik_arm.json",
        conf_file_path_ext=str(roboenv_dir),
        use_gui=not args.headless,
    )
    client = sim.pybullet_client
    # [Batch evaluation change] Disable measurement noise at runtime, without
    # changing either robotnik_arm.json file or touching physical control rules.
    if args.sensor_noise == "off":
        for robot in sim.bot:
            robot.noise_flag = False
    sensor_noise_enabled = [bool(robot.noise_flag) for robot in sim.bot]
    print(f"Sensor noise mode: {args.sensor_noise}; enabled: {sensor_noise_enabled}")

    times = []
    errors = []
    positions = []
    actions = []
    observations = []

    # [23D change] Same standoff geometry, tolerances and timeout as the expert.
    # The phase is latched by BaseNavState and never resets during an episode.
    standoff = float(np.interp(
        target[2], [0.1, 0.5, 1.1, 1.25], [0.80, 0.65, 0.65, 0.35]
    ))
    target_xy = target[:2]
    goal_bearing = (
        np.arctan2(target_xy[1], target_xy[0])
        if np.linalg.norm(target_xy) > 1e-4 else 0.0
    )
    goal_pos = target_xy - standoff * np.array([
        np.cos(goal_bearing), np.sin(goal_bearing)
    ])
    phase_state = BaseNavState()
    phase_params = BaseControllerParams(nav_timeout=25.0)
    phase_switch_time = None
    phase_switch_reason = "not_switched"

    success = False
    reason = "time_limit"
    final_error = float("inf")
    clipped_steps = 0

    try:
        rid = sim.bot[0].bot_pybullet
        sim.SetFloorFriction(1.0)

        for joint_id in range(client.getNumJoints(rid)):
            name = client.getJointInfo(rid, joint_id)[1].decode()
            if "wheel" in name:
                client.changeDynamics(
                    rid,
                    joint_id,
                    contactStiffness=30000,
                    contactDamping=1000,
                    anisotropicFriction=[1.0, 0.01, 1.0],
                )

        sim.bot[0].servo_motor_model.set_motor_gains(400, 30)

        configure_gripper_fingers(
            sim,
            client,
            rid,
            np.full(6, 0.1),
        )

        visual_id = client.createVisualShape(
            client.GEOM_SPHERE,
            radius=0.03,
            rgbaColor=[1.0, 0.0, 0.0, 0.85],
        )
        client.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id,
            basePosition=target.tolist(),
        )

        joint_names = np.expand_dims(
            np.array(sim.getNameActiveJoints()),
            axis=0,
        )
        dyn_model = PinWrapper(
            "robotnik_arm.json",
            "pybullet",
            joint_names,
            ["pybullet"],
            False,
            0,
            str(roboenv_dir),
        )
        ee_frame_id = dyn_model.pin_model.getFrameId(
            "robot_j2s7s300_end_effector"
        )

        dt = sim.GetTimeStep()
        max_steps = int(np.ceil(args.duration / dt))
        cmd = MotorCommands()

        # 仅作动作限幅，不使用专家控制器生成动作
        action_limits = np.array([3.0] * 4 + [30.0] * 7)

        for step in range(max_steps + 1):
            current_time = step * dt

            base_pos = np.array(sim.GetBasePosition())
            base_ori = np.array(sim.GetBaseOrientation())
            bearing = quaternion2bearing(
                base_ori[3],
                base_ori[0],
                base_ori[1],
                base_ori[2],
            )
            q = np.array(sim.GetMotorAngles(0))
            qd = np.array(sim.GetMotorVelocities(0))

            q_all = np.concatenate([base_pos, base_ori, q])
            q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")
            pin.forwardKinematics(
                dyn_model.pin_model,
                dyn_model.pin_data,
                q_pin,
            )
            pin.updateFramePlacements(
                dyn_model.pin_model,
                dyn_model.pin_data,
            )
            ee_pos = dyn_model.pin_data.oMf[
                ee_frame_id
            ].translation.copy()

            final_error = float(np.linalg.norm(target - ee_pos))
            times.append(current_time)
            errors.append(final_error)
            positions.append(ee_pos)

            if not np.isfinite(final_error):
                reason = "nonfinite_state"
                break

            # 使用统一的 1 cm 到达标准
            if final_error < 0.01:
                success = True
                reason = "target_reached"
                break

            if step == max_steps:
                break

            # [23D change] Update the phase using the ACTUAL model-controlled base.
            # Discard the expert wheel commands: they NEVER control the robot here.
            # This is a rule-provided phase input, not a learned phase classifier.
            was_done = phase_state.done
            base_controller(
                phase_state, base_pos, bearing, goal_pos, goal_bearing,
                current_time, phase_params,
            )
            if phase_state.done and not was_done:
                phase_switch_time = current_time
                phase_switch_reason = (
                    "navigation_timeout" if current_time > phase_params.nav_timeout
                    else "navigation_complete"
                )
                print(
                    f"Phase 1 -> 0 at {current_time:.3f}s ({phase_switch_reason})",
                    flush=True,
                )
            base_active = 0.0 if phase_state.done else 1.0

            observation = np.concatenate([
                base_pos,
                [np.sin(bearing), np.cos(bearing)],
                q[4:11],
                qd[4:11],
                target,
                # [23D change] Column 22 matches the new expert dataset.
                [base_active],
            ]).astype(np.float32)

            if not np.isfinite(observation).all():
                reason = "nonfinite_state"
                break

            # 输入标准化 → 网络预测 → 输出反标准化
            normalized_x = (observation - x_mean) / x_std

            with torch.inference_mode():
                normalized_action = model(
                    torch.from_numpy(normalized_x).unsqueeze(0)
                ).squeeze(0).numpy()

            raw_action = normalized_action * y_std + y_mean

            if not np.isfinite(raw_action).all():
                reason = "nonfinite_action"
                break

            # [23D change] No forced parking: all 11 actions remain model outputs.
            # Keeping only the baseline limits isolates the effect of phase input.
            action = np.clip(
                raw_action,
                -action_limits,
                action_limits,
            )
            clipped_steps += int(np.any(action != raw_action))

            observations.append(observation.copy())
            actions.append(action.copy())

            # 四个轮速 + 七个关节力矩 + 固定手指
            cmd_all = np.concatenate([action, np.zeros(6)])
            modes = ["velocity"] * 4 + ["torque"] * 13
            cmd.SetControlCmd(cmd_all, modes)
            sim.Step(cmd, "torque")

            if step % 1000 == 0:
                print(
                    f"t={current_time:.1f}s | "
                    f"末端误差={final_error:.4f} m",
                    flush=True,
                )

            if not args.headless:
                time.sleep(dt)

    finally:
        client.disconnect()

    # 模型产生的数据单独保存，避免混入专家训练数据
    # [Batch evaluation change] Each target has a known, independent result path.
    output_path = (args.result_file if args.result_file is not None else
                   run_dir / "rollouts" / f"bc_{datetime.now():%Y%m%d_%H%M%S_%f}.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite rollout: {output_path}")

    np.savez_compressed(
        output_path,
        target=target,
        timestamps=np.asarray(times),
        ee_positions=np.asarray(positions),
        ee_errors=np.asarray(errors),
        observations=np.asarray(observations, dtype=np.float32).reshape(-1, 23),
        actions=np.asarray(actions, dtype=np.float32).reshape(-1, 11),
        success=success,
        termination_reason=reason,
        final_error=final_error,
        clipped_steps=clipped_steps,
        # [23D change] Record phase provenance for reproducible comparison.
        input_dim=23,
        base_active_column=22,
        phase_source="expert_navigation_rules_commands_discarded",
        forced_parking=False,
        phase_switch_time=np.nan if phase_switch_time is None else phase_switch_time,
        phase_switch_reason=phase_switch_reason,
        model_run=str(run_dir),
        # [Batch evaluation change] Document the actual evaluation conditions.
        sensor_noise_mode=args.sensor_noise,
        sensor_noise_enabled=np.asarray(sensor_noise_enabled, dtype=bool),
        evaluation_seed=args.seed,
        duration_limit_seconds=args.duration,
        reach_tolerance_m=0.01,
        min_success_time_seconds=0.0,  # [Anytime success change] No time gate.
        required_hold_seconds=0.0,
    )

    print("\n========== 模型控制结果 ==========")
    print(f"成功到达：{success}")
    print(f"终止原因：{reason}")
    print(f"最终误差：{final_error:.6f} m")
    print(f"最小误差：{np.min(errors):.6f} m")
    print(f"结束时间：{times[-1]:.3f} s")
    print(f"动作被限幅的步数：{clipped_steps}")
    print(f"结果保存：{output_path}")

    # 状态比动作可能多一行：最后一行是执行完最后动作后的状态
    if not args.headless:
        import matplotlib.pyplot as plt

        plt.plot(times, errors, label="BC policy")
        plt.axhline(
            0.01, color="red", linestyle="--", label="1 cm"
        )
        plt.xlabel("Simulation time (s)")
        plt.ylabel("End-effector distance error (m)")
        plt.title("Closed-loop behaviour cloning evaluation")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()