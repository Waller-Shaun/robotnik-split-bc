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
)
from train_bc import BCPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, nargs=3, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration 必须大于 0")

    roboenv_dir = Path(__file__).resolve().parents[3] / "RoboEnv"
    run_dir = (
        roboenv_dir
        / "bc_training"
        / "run_20261023_135447_849619"
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
    model = BCPolicy()
    checkpoint = torch.load(
        run_dir / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with np.load(run_dir / "normalization.npz") as data:
        x_mean = data["x_mean"].copy()
        x_std = data["x_std"].copy()
        y_mean = data["y_mean"].copy()
        y_std = data["y_std"].copy()

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

    times = []
    errors = []
    positions = []
    actions = []
    observations = []

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

            # 使用统一的 1 cm 到达标准，并保留模板的 8 s 最短时间
            if current_time > 8.0 and final_error < 0.01:
                success = True
                reason = "target_reached"
                break

            if step == max_steps:
                break

            observation = np.concatenate([
                base_pos,
                [np.sin(bearing), np.cos(bearing)],
                q[4:11],
                qd[4:11],
                target,
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
    output_dir = run_dir / "rollouts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"bc_{datetime.now():%Y%m%d_%H%M%S_%f}.npz"
    )

    np.savez_compressed(
        output_path,
        target=target,
        timestamps=np.asarray(times),
        ee_positions=np.asarray(positions),
        ee_errors=np.asarray(errors),
        observations=np.asarray(observations, dtype=np.float32).reshape(-1, 22),
        actions=np.asarray(actions, dtype=np.float32).reshape(-1, 11),
        success=success,
        termination_reason=reason,
        final_error=final_error,
        clipped_steps=clipped_steps,
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