# [Hybrid diagnostic change] 复制自test_bc_26d_ring_bodyframe.py，仅用于分支诊断。
# 模型和专家都读取当前状态，再选择四轮／七关节动作来源；不回放专家轨迹。
# 专家q_des每步连续更新，不能拿另一条轨迹的参考角直接代入。
# 默认专家底盘＋模型手臂；两种混合结果均不是纯模型成功率。
# 保留20s、无噪声、任意时刻1cm、相同物理设置和统一动作限幅。
# [Body frame test change] 保留原阶段、限幅和任意时刻1cm标准。
# 训练／推理复用同一模块，替换相对目标及末端误差的坐标表达；其余状态保留。
# 只接受底盘yaw坐标系特征的26维检查点，防止同维度的旧模型混用。
import argparse
from dataclasses import asdict
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import pinocchio as pin

from simulation_and_control import pb, MotorCommands, PinWrapper
from robotnik_bodyframe_features import (
    BODYFRAME_FEATURE_VERSION,
    build_bodyframe_observation,
)
from robotnik_arm_control import (
    configure_gripper_fingers,
    quaternion2bearing,
    # [23D change] Reuse navigation rules ONLY to generate the phase input.
    BaseNavState, BaseControllerParams, base_controller,
    ArmControllerParams, arm_controller,
)


CONTROL_MODES = ("expert_base_model_arm", "model_base_expert_arm", "model", "expert")


def select_control_action(model_action, expert_action, mode):
    """[Hybrid diagnostic change] 明确选择动作来源；不改变原模型权重。"""
    model_action = np.asarray(model_action)
    expert_action = np.asarray(expert_action)
    if model_action.shape != (11,) or expert_action.shape != (11,):
        raise ValueError("Expected 11D model and expert actions")
    if mode == "expert_base_model_arm":
        return np.concatenate([expert_action[:4], model_action[4:]])
    if mode == "model_base_expert_arm":
        return np.concatenate([model_action[:4], expert_action[4:]])
    if mode == "model":
        return model_action.copy()
    if mode == "expert":
        return expert_action.copy()
    raise ValueError(f"Unknown control mode: {mode}")
# [Architecture test change] 从检查点配置构建网络，兼容旧双128和新256/256/128模型。
# 只改变模型加载；阶段判断、动作限幅和成功标准保持原样。
class CheckpointPolicy(torch.nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        layers = []
        dimensions = [26, *hidden_dims, 11]  # [26D test change]
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
    if config.get("input_dim") != 26 or config.get("output_dim") != 11:
        raise ValueError("Expected a 26-input, 11-output policy")
    # [26D test change] 防止同维度但特征顺序/定义不同的模型被静默加载。
    if (config.get("input_feature_version") != BODYFRAME_FEATURE_VERSION
            or config.get("source_input_dim") != 23
            or config.get("target_position_columns") != [19, 20, 21]
            or config.get("source_base_world_position_columns") != [0, 1, 2]
            or config.get("source_yaw_sin_cos_columns") != [3, 4]
            or config.get("target_position_frame") != "base_yaw_aligned"
            or config.get("ee_error_frame") != "base_yaw_aligned"
            or config.get("yaw_sin_cos_normalized_for_rotation") is not True
            or config.get("base_world_position_retained") is not True
            or config.get("absolute_yaw_sin_cos_retained") is not True
            or config.get("ee_error_columns") != [23, 24, 25]
            or config.get("base_active_column") != 22
            or config.get("ee_frame_name") != "robot_j2s7s300_end_effector"):
        raise ValueError("Checkpoint does not match the 26D base-yaw-frame feature schema")
    if config.get("activation", "ReLU") != "ReLU":
        raise ValueError("This evaluator supports ReLU checkpoints only")
    model = CheckpointPolicy(hidden_dims)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite parameters")
    return model


def build_26d_observation(source_observation, ee_position):
    """[Body frame test change] 调用训练共用函数，不复制一套旋转公式。"""
    return build_bodyframe_observation(source_observation, ee_position)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, nargs=3, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=20.0)  # [26D test change]
    # [23D change] Select a specific trained run explicitly when needed.
    parser.add_argument("--run-dir", type=Path, default=None)
    # [Batch evaluation change] Explicit noise policy, independent of config edits.
    # [Body frame test change] 默认off，与上一环状世界坐标评估一致。
    parser.add_argument("--sensor-noise", choices=["off", "config"], default="off")  # [26D test change]
    parser.add_argument("--result-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--control-mode", choices=CONTROL_MODES, default="expert_base_model_arm")
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
    # [Body frame test change] 默认本次最佳133轮；只接受匹配坐标定义的模型。
    run_dir = args.run_dir or (
        roboenv_dir / "bc_training_26d_noise_ring_bodyframe" / "run_20261102_184221_743102"
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
    print("Network:", " -> ".join(map(str, [26, *checkpoint.get("config", {}).get("hidden_dims", [128, 128]), 11])))
    model.eval()

    with np.load(run_dir / "normalization.npz") as data:
        x_mean = data["x_mean"].copy()
        x_std = data["x_std"].copy()
        y_mean = data["y_mean"].copy()
        y_std = data["y_std"].copy()

    # [23D change] Fail early if the model and preprocessing do not match.
    if x_mean.shape != (26,) or x_std.shape != (26,):
        raise ValueError("Expected 26D normalization statistics")
    if y_mean.shape != (11,) or y_std.shape != (11,):
        raise ValueError("Expected 11D action statistics")
    if (not all(np.isfinite(a).all() for a in (x_mean, x_std, y_mean, y_std))
            or np.any(x_std <= 0) or np.any(y_std <= 0)):
        raise ValueError("Normalization statistics must be finite with positive std")
    if x_mean[22] != 0.0 or x_std[22] != 1.0:
        raise ValueError("Phase column must remain binary: mean=0, std=1")

    print(f"目标：{target}")
    print(f"加载模型轮次：{checkpoint['epoch']}")
    print(f"对应验证 MSE：{checkpoint['val_mse']:.6f}")
    print(f"特征版本：{BODYFRAME_FEATURE_VERSION}")
    base_source = "expert" if args.control_mode in ("expert_base_model_arm", "expert") else "model"
    arm_source = "expert" if args.control_mode in ("model_base_expert_arm", "expert") else "model"
    print(f"诊断模式：{args.control_mode} | 底盘：{base_source} | 机械臂：{arm_source}")

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
    # [Hybrid diagnostic change] 同时保存候选动作与实际动作，便于核验来源。
    model_raw_actions = []
    expert_actions = []
    selected_raw_actions = []
    q_des_arm_all = []
    nav_subphases = []

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
        # [Hybrid diagnostic change] 与task_3的专家参数／初始化保持一致。
        shoulder_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2")
        if ee_frame_id >= dyn_model.pin_model.nframes or shoulder_frame_id >= dyn_model.pin_model.nframes:
            raise ValueError("Required expert control frames are missing")
        lower_limits, upper_limits = sim.GetBotJointsLimit()
        arm_lower = np.asarray(lower_limits[4:11])
        arm_upper = np.asarray(upper_limits[4:11])
        q_init_arm = np.asarray(sim.GetMotorAngles(0))[4:11].copy()
        q_des_arm = q_init_arm.copy()
        arm_params = ArmControllerParams(reach_limit=0.85)

        dt = sim.GetTimeStep()
        max_steps = int(np.ceil(args.duration / dt))
        cmd = MotorCommands()

        # [Hybrid diagnostic change] 四种模式统一沿用原评估的动作限幅。
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

            # [Hybrid diagnostic change] 阶段始终基于当前实际底盘；只更新一次。
            # 返回的专家轮速仅在expert_base_*或expert模式实际使用。
            was_done = phase_state.done
            wheel_expert = base_controller(
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

            source_observation = np.concatenate([
                base_pos,
                [np.sin(bearing), np.cos(bearing)],
                q[4:11],
                qd[4:11],
                target,
                # [23D change] Column 22 matches the new expert dataset.
                [base_active],
            ]).astype(np.float32)
            # [Body frame test change] 仍先构造世界坐标源23列，再调用共用转换。
            # 19:22是底盘相对目标，23:26是底盘坐标末端误差，22仍为阶段。
            observation = build_26d_observation(source_observation, ee_pos)

            if not np.isfinite(observation).all():
                reason = "nonfinite_state"
                break

            # 输入标准化 → 网络预测 → 输出反标准化
            normalized_x = (observation - x_mean) / x_std

            with torch.inference_mode():
                normalized_action = model(
                    torch.from_numpy(normalized_x).unsqueeze(0)
                ).squeeze(0).numpy()

            model_action = normalized_action * y_std + y_mean

            # [Hybrid diagnostic change] 使用当前状态在线计算专家动作。
            # 即便手臂由模型驱动，也连续维护这份影子专家参考供日志比较；
            # 只有选为expert_arm时才把对应力矩发送给仿真。
            torque_expert, q_des_arm, _ = arm_controller(
                dyn_model, q_pin, q[4:11], qd[4:11], q_des_arm, q_init_arm,
                target, ee_pos, ee_frame_id, shoulder_frame_id,
                arm_lower, arm_upper, dt, arm_params,
                reach_active=phase_state.done,
            )
            expert_action = np.concatenate([wheel_expert, torque_expert])
            raw_action = select_control_action(model_action, expert_action, args.control_mode)

            if not np.isfinite(np.concatenate([model_action, expert_action, raw_action])).all():
                reason = "nonfinite_action"
                break

            # [Hybrid diagnostic change] 不添加额外停车覆盖；专家底盘结束后
            # 的零轮速来自其原有控制规则，而非纯模型控制的结果。
            action = np.clip(
                raw_action,
                -action_limits,
                action_limits,
            )
            clipped_steps += int(np.any(action != raw_action))

            observations.append(observation.copy())
            actions.append(action.copy())
            model_raw_actions.append(model_action.copy())
            expert_actions.append(expert_action.copy())
            selected_raw_actions.append(raw_action.copy())
            q_des_arm_all.append(q_des_arm.copy())
            nav_subphases.append("done" if phase_state.done else phase_state.phase)

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
                   run_dir / "hybrid_diagnostics" / "single_rollouts" /
                   f"{args.control_mode}_{datetime.now():%Y%m%d_%H%M%S_%f}.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite rollout: {output_path}")

    np.savez_compressed(
        output_path,
        target=target,
        timestamps=np.asarray(times),
        ee_positions=np.asarray(positions),
        ee_errors=np.asarray(errors),
        observations=np.asarray(observations, dtype=np.float32).reshape(-1, 26),
        actions=np.asarray(actions, dtype=np.float32).reshape(-1, 11),
        # [Hybrid diagnostic change] expert_actions使用本次观测及连续影子参考，
        # 不是原采集轨迹中的动作。actions仍为实际执行、限幅后的11维动作。
        model_raw_actions=np.asarray(model_raw_actions, dtype=np.float32).reshape(-1, 11),
        expert_actions=np.asarray(expert_actions, dtype=np.float32).reshape(-1, 11),
        selected_raw_actions=np.asarray(selected_raw_actions, dtype=np.float32).reshape(-1, 11),
        q_des_arm=np.asarray(q_des_arm_all, dtype=np.float64).reshape(-1, 7),
        nav_subphases=np.asarray(nav_subphases, dtype="U8"),
        diagnostic_only=True,
        control_mode=args.control_mode,
        base_control_source=base_source,
        arm_control_source=arm_source,
        expert_reference_definition="continuously_updated_on_current_observed_state_not_trajectory_replay",
        controller_parameters_json=json.dumps({"base": asdict(phase_params), "arm": asdict(arm_params)}),
        action_limits=action_limits,
        base_goal_position=goal_pos,
        base_goal_bearing=goal_bearing,
        success=success,
        termination_reason=reason,
        final_error=final_error,
        clipped_steps=clipped_steps,
        # [23D change] Record phase provenance for reproducible comparison.
        input_dim=26,  # [26D test change] 保存实际送入网络的26维原始特征。
        source_input_dim=23,
        input_feature_version=BODYFRAME_FEATURE_VERSION,
        target_position_columns=np.array([19, 20, 21]),
        target_position_frame="base_yaw_aligned",
        ee_error_frame="base_yaw_aligned",
        ee_error_columns=np.array([23, 24, 25]),
        ee_frame_name="robot_j2s7s300_end_effector",
        base_active_column=22,
        phase_source="expert_navigation_rules_on_actual_state",
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

    print("\n========== 分支诊断结果（不作为纯模型成功率） ==========")
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

        plt.plot(times, errors, label=args.control_mode)
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
