# [Demo change] 演示入口：指定XYZ优先，否则随机生成高于等于0.5m的目标。
# 保留两个学习模型的控制方式；当前任务为末端到达，不添加实体抓取逻辑。
# 不导入训练脚本、不更新参数、不使用专家动作、不强制停车。
# 保留原物理设置、20s、无噪声、任意时刻1cm标准。
# 训练／推理复用同一模块，替换相对目标及末端误差的坐标表达；其余状态保留。
# 只接受底盘yaw坐标系特征的26维检查点，防止同维度的旧模型混用。
import argparse
import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
from arm_lowz_supplement_experiment import resolve_finetuned_pair as resolve_supplement_pair
from arm_finetune_experiment import resolve_finetuned_pair as resolve_previous_pair

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


def generate_target(center_xy, seed, distance="inside", radius_range=None, height_range=(.5, 1.3), height_mode="continuous"):
    """[Demo change] 整圈均匀方向、环面面积均匀半径、连续均匀高度，不挑选成功点。"""
    if distance not in ("inside", "outside"):
        raise ValueError("distance must be inside or outside")
    radii = radius_range if radius_range is not None else ((.8, 2.) if distance=="inside" else (2.05, 3.))
    lo, hi = map(float, radii)
    zlo, zhi = map(float, height_range)
    if not np.isfinite([*center_xy,lo,hi,zlo,zhi]).all() or len(center_xy)!=2:
        raise ValueError("Sampling bounds/center must be finite")
    if not 0 < lo <= hi:
        raise ValueError("Require 0 < radius_min <= radius_max")
    if distance=="inside" and (lo<.8 or hi>2.):
        raise ValueError("inside sampling must remain in horizontal radius [0.8, 2] m")
    if distance=="outside" and lo<=2.:
        raise ValueError("outside sampling requires radius_min > 2 m")
    if not .5 <= zlo <= zhi <= 1.3:
        raise ValueError("Random demo heights must satisfy 0.5 <= min <= max <= 1.3 m")
    rng = np.random.default_rng(seed)
    radius = np.sqrt(rng.uniform(lo*lo,hi*hi))
    theta = rng.uniform(-np.pi,np.pi)
    if height_mode=="continuous":
        z = rng.uniform(zlo,zhi)
    elif height_mode=="trained":
        choices=[z for z in [.5,.75,1.,1.15,1.3] if zlo<=z<=zhi]
        if not choices:
            raise ValueError("height-range contains no trained height; choose continuous mode or widen the range")
        z = float(rng.choice(choices))
    else:
        raise ValueError("height_mode must be continuous or trained")
    return np.array([center_xy[0]+radius*np.cos(theta), center_xy[1]+radius*np.sin(theta),z]), [lo,hi]


def resolve_demo_pair(roboenv, model_name, arm_run=None, nav_run=None):
    """[Demo change] 默认用刚完成的补采版；可切回84.48%版本或明确指定目录。"""
    if arm_run is None:
        relative = ("bc_training_arm_ring_r08_lowz_supplement/run_20261103_141232_983306"
                    if model_name=="supplement" else "bc_training_arm_ring_r08_finetune/run_20261103_111655_849512")
        arm_run = Path(roboenv) / relative
    config = json.loads((Path(arm_run)/"config.json").read_text(encoding="utf-8"))
    if config.get("experiment_variant")=="arm_lowz_continuous_supplement_replay_r08_v1":
        return resolve_supplement_pair(roboenv,arm_run,nav_run)
    return resolve_previous_pair(roboenv,arm_run,nav_run)


def main():
    parser = argparse.ArgumentParser(description="移动机械臂模型演示：手动XYZ或随机目标，默认r=0.8～2m、z=0.5～1.3m。")
    parser.add_argument("--target", type=float, nargs=3, default=None, metavar=("X","Y","Z"),
                        help="给出时严格使用此世界坐标；优先于所有随机目标范围")
    parser.add_argument("--distance", choices=["inside","outside"], default="inside",
                        help="随机目标的水平距离：inside=0.8～2m；outside=2.05～3m")
    parser.add_argument("--radius-range", type=float, nargs=2, default=None, metavar=("MIN","MAX"),
                        help="可选自定义随机水平半径范围，须符合inside/outside模式")
    parser.add_argument("--height-range", type=float, nargs=2, default=[.5,1.3], metavar=("MIN","MAX"),
                        help="随机高度区间，限制在0.5～1.3m；相同两值可固定高度")
    parser.add_argument("--height-mode", choices=["continuous","trained"], default="continuous",
                        help="continuous=连续随机高度（默认）；trained=从0.5/0.75/1/1.15/1.3m随机选择")
    parser.add_argument("--model", choices=["supplement","previous"], default="supplement",
                        help="supplement=本次第30轮补采版；previous=先前84.48%%版本")
    parser.add_argument("--dry-run", action="store_true", help="只显示选中的目标，不启动仿真")
    parser.add_argument("--plot", action="store_true", help="仿真结束后显示末端误差曲线")
    parser.add_argument("--no-pause", action="store_true", help="GUI结束后直接关闭窗口；默认保留现场，终端按Enter继续")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=20.0)  # [26D test change]
    # [23D change] Select a specific trained run explicitly when needed.
    parser.add_argument("--nav-run-dir", type=Path, default=None,
                        help="Optional check: must match the navigation run bound to the arm finetune")
    # [Arm finetune change] 默认选最新完成的正式机械臂微调，底盘由其配置绑定。
    parser.add_argument("--arm-run-dir", type=Path, default=None,
                        help="可选模型目录，优先于--model；省略时使用对应已核验的固定版本")
    # [Batch evaluation change] Explicit noise policy, independent of config edits.
    # [Body frame test change] 默认off，与上一环状世界坐标评估一致。
    parser.add_argument("--sensor-noise", choices=["off", "config"], default="off")  # [26D test change]
    parser.add_argument("--result-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="可选复现种子；省略时每次生成新随机种子和目标")
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration 必须大于 0")

    # [Batch evaluation change] Seed observation noise if config mode is used.
    if args.seed is None:
        args.seed = int(np.random.SeedSequence().generate_state(1)[0])
    if not 0 <= args.seed <= 2**32-1 or not np.isfinite(args.duration):
        parser.error("seed must be in [0, 2**32-1] and duration must be finite")
    if args.target is not None and not np.isfinite(args.target).all():
        parser.error("target coordinates must be finite")
    np.random.seed(args.seed)
    if args.result_file is not None and args.result_file.exists():
        raise FileExistsError(f"Refusing to overwrite rollout: {args.result_file}")

    roboenv_dir = Path(__file__).resolve().parents[3] / "RoboEnv"
    # [Demo change] 从实际仿真配置读取初始底盘中心；手动XYZ不受随机范围约束。
    robot_config=json.loads((roboenv_dir/"configs/robotnik_arm.json").read_text(encoding="utf-8"))
    center_xy=robot_config["robot_pybullet"]["init_link_base_position"][0][:2]
    if args.target is None:
        try:
            target,sampling_radii=generate_target(center_xy,args.seed,args.distance,args.radius_range,args.height_range,args.height_mode)
        except ValueError as exc:
            parser.error(str(exc))
        target_source="random"
    else:
        target=np.asarray(args.target,dtype=np.float64)
        sampling_radii=[]
        target_source="manual"
    target_radius=float(np.linalg.norm(target[:2]-np.asarray(center_xy)))
    actual_distance="inside" if .8-1e-9<=target_radius<=2.+1e-9 else ("outside" if target_radius>2. else "near_robot")
    demo_settings={"target_source":target_source,"seed":args.seed,"requested_distance_mode":args.distance,
                   "actual_horizontal_radius_m":target_radius,"actual_distance_regime":actual_distance,
                   "random_radius_range_m":sampling_radii,
                   "random_height_range_m":args.height_range if target_source=="random" else [],
                   "random_height_mode":args.height_mode if target_source=="random" else "manual",
                   "sampling":"uniform_area_annulus_and_selected_height_mode" if target_source=="random" else "explicit_XYZ",
                   "center_xy_m":center_xy,"task":"end_effector_position_reaching_fixed_gripper"}
    print(f"演示目标：{target.tolist()}；来源={target_source}；水平距离={target_radius:.3f}m；高度={target[2]:.3f}m；seed={args.seed}",flush=True)
    if args.dry_run:
        print(json.dumps(demo_settings,ensure_ascii=False,indent=2))
        print("Dry run：没有启动仿真或写文件。")
        return
    # [Split policy test change] 最新选择只看已完成的正式训练，不自动训练或选程序小样本。
    run_dir, nav_run_dir, arm_config, nav_config = resolve_demo_pair(
        roboenv_dir, args.model, args.arm_run_dir, args.nav_run_dir)
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

    # [Demo change] 不从训练／验证／测试目标表选点；使用上面已确定的目标。

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
    print(f"Sensor noise mode: {args.sensor_noise}; enabled: {sensor_noise_enabled}")

    times = []
    errors = []
    positions = []
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
        target_body_id = client.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id,
            basePosition=target.tolist(),
        )
        if not args.headless:
            # [Demo change] 相机同时覆盖初始小车和目标，隐藏侧栏，便于课堂展示。
            client.configureDebugVisualizer(client.COV_ENABLE_GUI,0)
            camera_target=[float((center_xy[0]+target[0])/2),float((center_xy[1]+target[1])/2),.45]
            client.resetDebugVisualizerCamera(max(2.6,1.15*target_radius+1.2),50,-35,camera_target)
            client.addUserDebugText(f"Target | r={target_radius:.2f} m | z={target[2]:.2f} m",
                                    [float(target[0]),float(target[1]),float(target[2]+.12)],
                                    textColorRGB=[.75,.05,.05],textSize=1.1)

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
                # [Demo change] 只把目标标记改成绿色，不改变控制动作或机器人姿态。
                client.changeVisualShape(target_body_id,-1,rgbaColor=[.1,.8,.25,1.])
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

        if not args.headless and not args.no_pause:
            # [Demo change] 结束后暂停推进仿真以查看最终现场，不代表已学会持续保持。
            print(f"演示结束：{'到达目标' if success else reason}；末端误差={final_error*100:.2f}cm。",flush=True)
            try:
                input("仿真窗口保留最终现场；在此终端按 Enter 关闭并保存结果。")
            except EOFError:
                pass

    finally:
        if client.isConnected():
            client.disconnect()

    # 模型产生的数据单独保存，避免混入专家训练数据
    # [Batch evaluation change] Each target has a known, independent result path.
    output_path = (args.result_file if args.result_file is not None else
                   roboenv_dir / "demo_runs" / f"demo_{datetime.now():%Y%m%d_%H%M%S_%f}.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite rollout: {output_path}")

    np.savez_compressed(
        output_path,
        demo_settings_json=json.dumps(demo_settings,ensure_ascii=False),
        target=target,
        timestamps=np.asarray(times),
        ee_positions=np.asarray(positions),
        ee_errors=np.asarray(errors),
        observations=np.asarray(observations, dtype=np.float32).reshape(-1, 26),
        actions=np.asarray(actions, dtype=np.float32).reshape(-1, 11),
        navigation_observations=np.asarray(nav_observations,dtype=np.float32).reshape(-1,8),
        navigation_raw_actions=np.asarray(nav_raw_actions,dtype=np.float32).reshape(-1,4),
        arm_raw_actions=np.asarray(arm_raw_actions,dtype=np.float32).reshape(-1,7),
        nav_subphases=np.asarray(nav_subphases,dtype="U8"),
        navigation_input_dim=8,
        navigation_feature_version=NAV_FEATURE_VERSION,
        policy_architecture="frozen_learned_nav8_arm26_demo",
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
    if not args.headless and args.plot:
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
