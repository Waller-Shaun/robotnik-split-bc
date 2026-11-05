# [B4 change] 独立测试期望速度辅助标签实验；实际控制仍只执行力矩。
# [B4 change] 独立测试B4检查点；共享梯度门控不改变部署模型、物理设置或评分。
# [B2 change] 独立测试多任务MLP；仅力矩路径参与控制，辅助参考从不作为输入。
# [Noise truth change] 新副本：带噪观测控制，PyBullet真实末端位置评估；不改旧测试或模型。
# [Low Z test change] 仅测试：冻结底盘＋使用998条合并清单微调的机械臂。
# 不导入训练脚本、不更新参数、不使用专家动作、不强制停车。
# [Noise truth change] 保留20s物理设置；默认配置噪声；任意时刻真实距离<1cm成功。
# 训练／推理复用同一模块，替换相对目标及末端误差的坐标表达；其余状态保留。
# 只接受底盘yaw坐标系特征的26维检查点，防止同维度的旧模型混用。
import argparse
import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
from robotnik_arm_velocity_aux_policy import resolve_finetuned_pair, load_policy_from_checkpoint

import numpy as np
import torch
import pinocchio as pin

from simulation_and_control import pb, MotorCommands, PinWrapper
from robotnik_navigation_policy import (
    NAV_FEATURE_VERSION, NAV_PARAMS, build_navigation_observation,
    parking_goal, load_navigation_policy,
)
from robotnik_bodyframe_features import (
    BODYFRAME_FEATURE_VERSION,
    build_bodyframe_observation,
)
from robotnik_arm_control import (
    configure_gripper_fingers,
    quaternion2bearing,
    # [23D change] Reuse navigation rules ONLY to generate the phase input.
    BaseNavState, BaseControllerParams, base_controller,
)
# [Architecture test change] 从检查点配置构建网络，兼容旧双128和新256/256/128模型。
# 只改变模型加载；阶段判断、动作限幅和成功标准保持原样。
def build_26d_observation(source_observation, ee_position):
    """[Body frame test change] 调用训练共用函数，不复制一套旋转公式。"""
    return build_bodyframe_observation(source_observation, ee_position)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, nargs=3, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=20.0)  # [26D test change]
    # [23D change] Select a specific trained run explicitly when needed.
    parser.add_argument("--nav-run-dir", type=Path, default=None,
                        help="Optional check: must match the navigation run bound to the arm finetune")
    # [Arm finetune change] 默认选最新完成的正式机械臂微调，底盘由其配置绑定。
    parser.add_argument("--arm-run-dir", type=Path, default=None,
                        help="Arm finetune directory; omitted selects latest finished non-smoke run")
    # [Batch evaluation change] Explicit noise policy, independent of config edits.
    # [Body frame test change] 默认off，与上一环状世界坐标评估一致。
    parser.add_argument("--sensor-noise", choices=["off", "config"], default="config")  # [26D test change]
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
    # [Split policy test change] 最新选择只看已完成的正式训练，不自动训练或选程序小样本。
    run_dir, nav_run_dir, arm_config, nav_config = resolve_finetuned_pair(
        roboenv_dir, args.arm_run_dir, args.nav_run_dir)
    nav_checkpoint = torch.load(nav_run_dir/"best_model.pt",map_location="cpu",weights_only=True)
    if nav_checkpoint["config"] != nav_config:
        raise ValueError("Navigation checkpoint/config differ")
    nav_model = load_navigation_policy(nav_checkpoint)
    for name, expected in [("robotnik_navigation_policy.py",nav_config["feature_module_sha256"]),
                           ("robotnik_arm_control.py",nav_config["controller_sha256"])]:
        actual = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Feature/controller code changed since navigation training: {name}")
    # [Arm finetune change] run_dir指向新机械臂；共用核验函数已核对原底盘及继承的归一化。
    with np.load(nav_run_dir/"normalization.npz") as n:
        nav_x_mean,nav_x_std,nav_y_mean,nav_y_std = (n[k].copy() for k in ["x_mean","x_std","y_mean","y_std"])
    if (nav_x_mean.shape!=(8,) or nav_x_std.shape!=(8,) or nav_y_mean.shape!=(4,) or nav_y_std.shape!=(4,)
            or not all(np.isfinite(a).all() for a in [nav_x_mean,nav_x_std,nav_y_mean,nav_y_std])
            or np.any(nav_x_std<=0) or np.any(nav_y_std<=0)
            or np.any(nav_x_mean[4:]!=0) or np.any(nav_x_std[4:]!=1)):
        raise ValueError("Invalid navigation normalization/one-hot statistics")

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
    if checkpoint["config"] != arm_config:
        raise ValueError("Arm checkpoint/config differ")
    # [Architecture test change] 加载对应结构；旧实验仍可用原来的命令测试。
    model = load_policy_from_checkpoint(checkpoint)
    print("Network:", " -> ".join(map(str, [26, *checkpoint.get("config", {}).get("hidden_dims", [128, 128]), 11])))
    model.eval()
    print("B4 auxiliary desired-velocity head is NOT used for control or input.")

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
    print(f"机械臂验证 MSE（7维归一化力矩）：{checkpoint['val_mse']:.6f}")
    print(f"Finetuned arm: {run_dir}")
    print(f"Frozen navigation: {nav_run_dir}")
    print(f"特征版本：{BODYFRAME_FEATURE_VERSION}")
    print(f"Navigation model: epoch {nav_checkpoint['epoch']}, 8 -> 64 -> 64 -> 4")
    print("All 11 applied actions come from two learned policies; no expert actions are applied.")

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
    # [Noise truth change] config模式要求配置确实开启，避免误把无噪声当作带噪声结果。
    if args.sensor_noise == "config" and not all(sensor_noise_enabled):
        client.disconnect()
        raise ValueError("Config noise is disabled: check RoboEnv/configs/robotnik_arm.json")
    print(f"Sensor noise mode: {args.sensor_noise}; enabled: {sensor_noise_enabled}")

    times = []
    errors = []
    positions = []
    observed_positions = []
    observed_errors = []
    actions = []
    observations = []
    nav_observations = []
    nav_raw_actions = []
    arm_raw_actions = []
    nav_subphases = []

    # [23D change] Same standoff geometry, tolerances and timeout as the expert.
    # The phase is latched by BaseNavState and never resets during an episode.
    goal_pos, goal_bearing = parking_goal(target)
    phase_state = BaseNavState()
    phase_params = BaseControllerParams(**NAV_PARAMS)
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

        # [Noise truth change] 使用link frame位置（getLinkState第4项），不是质心第0项。
        # 该读取不加测量噪声、不使用随机数，不参与模型输入或阶段判断。
        ee_links = [j for j in range(client.getNumJoints(rid))
                    if client.getJointInfo(rid, j)[12].decode() == "robot_j2s7s300_end_effector"]
        if len(ee_links) != 1:
            raise ValueError("Expected exactly one matching PyBullet EE link frame")
        ee_link_id = ee_links[0]

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

            # [Noise truth change] ee_pos仍是带噪FK估计，仅供模型使用。
            true_ee_pos = np.asarray(client.getLinkState(
                rid, ee_link_id, computeForwardKinematics=True)[4], dtype=np.float64)
            observed_error = float(np.linalg.norm(target - ee_pos))
            final_error = float(np.linalg.norm(target - true_ee_pos))
            times.append(current_time)
            errors.append(final_error)
            positions.append(true_ee_pos)
            observed_positions.append(ee_pos.copy())
            observed_errors.append(observed_error)

            if not np.isfinite(final_error) or not np.isfinite(observed_error):
                reason = "nonfinite_state"
                break

            # [Noise truth change] 只用真实末端位置判定1cm成功；模型不读取真实位置。
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
            nav_observation = build_navigation_observation(base_pos,bearing,goal_pos,goal_bearing,phase_state)

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
            # [Split policy test change] 旧模型的轮速输出不使用；新底盘网络只看底盘特征。
            with torch.inference_mode():
                nav_normalized_action = nav_model(torch.from_numpy((nav_observation-nav_x_mean)/nav_x_std).unsqueeze(0)).squeeze(0).numpy()
            nav_action = nav_normalized_action*nav_y_std+nav_y_mean
            arm_action = raw_action[4:11].copy()
            raw_action = np.concatenate([nav_action,arm_action])

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
            nav_observations.append(nav_observation.copy())
            nav_raw_actions.append(nav_action.copy())
            arm_raw_actions.append(arm_action.copy())
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
                   run_dir / "rollouts" / f"velocity_aux_b4_noise_truth_{datetime.now():%Y%m%d_%H%M%S_%f}.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite rollout: {output_path}")

    np.savez_compressed(
        output_path,
        target=target,
        timestamps=np.asarray(times),
        ee_positions=np.asarray(positions),
        ee_errors=np.asarray(errors),  # [Noise truth change] 主误差字段改为真实值。
        observed_ee_positions=np.asarray(observed_positions),
        observed_ee_errors=np.asarray(observed_errors),
        evaluation_position_source="pybullet_link_world_frame",
        model_position_source="forward_kinematics_from_sensor_observations",
        true_state_used_for_model_input=False,
        ee_link_id=ee_link_id,
        robot_config_path=str(roboenv_dir / "configs" / "robotnik_arm.json"),
        robot_config_sha256=hashlib.sha256((roboenv_dir / "configs" / "robotnik_arm.json").read_bytes()).hexdigest(),
        observations=np.asarray(observations, dtype=np.float32).reshape(-1, 26),
        actions=np.asarray(actions, dtype=np.float32).reshape(-1, 11),
        navigation_observations=np.asarray(nav_observations,dtype=np.float32).reshape(-1,8),
        navigation_raw_actions=np.asarray(nav_raw_actions,dtype=np.float32).reshape(-1,4),
        arm_raw_actions=np.asarray(arm_raw_actions,dtype=np.float32).reshape(-1,7),
        nav_subphases=np.asarray(nav_subphases,dtype="U8"),
        navigation_input_dim=8,
        navigation_feature_version=NAV_FEATURE_VERSION,
        policy_architecture="frozen_learned_nav8_velocity_aux_b4_arm26",
        expert_references_used_as_input=False,
        auxiliary_output_applied_to_robot=False,
        auxiliary_training_weight=arm_config["aux_weight"],
        gradient_mode=arm_config["gradient_mode"],
        auxiliary_label=arm_config["auxiliary_label"],
        auxiliary_label_units=arm_config["auxiliary_label_units"],
        arm_training_dataset_variant=arm_config["dataset_variant"],
        arm_finetuned=True,
        arm_parent_run=arm_config["source_model_run_dir"],
        all_actions_model_generated=True,
        expert_wheel_commands_applied=False,
        navigation_run=str(nav_run_dir),
        arm_run=str(run_dir),
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

        plt.plot(times, errors, label="True EE error (success criterion)")
        plt.plot(times, observed_errors, alpha=0.45, label="Observed EE error (model input)")
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
