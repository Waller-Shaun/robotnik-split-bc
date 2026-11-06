"""[Split policy change] 新底盘策略共用的特征、阶段重建及模型定义。

只有特征计算／网络类，没有训练入口或仿真入口。
输入8维：停车点相对XY、朝向停车点角误差、最终朝向角误差、4维阶段one-hot。
角误差采用[-pi, pi)标量，避免初始180度左右转的相反标签挤在相邻sin/cos上。
"""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from robotnik_arm_control import BaseControllerParams

NAV_FEATURE_VERSION = "parking_body_xy_signed_heading_errors_phase4_v1"
NAV_PARAMS = asdict(BaseControllerParams(nav_timeout=25.0))
PHASE_NAMES = ("align", "drive", "orient", "done")


def wrap_angle(value):
    return (value + np.pi) % (2 * np.pi) - np.pi


def parking_goal(target):
    """与原专家相同的停车几何；所有坐标单位m。"""
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("Expected a finite 3D target")
    bearing = float(np.arctan2(target[1], target[0])) if np.linalg.norm(target[:2]) > 1e-4 else 0.0
    standoff = float(np.interp(target[2], [.1, .5, 1.1, 1.25], [.8, .65, .65, .35]))
    return target[:2] - standoff * np.array([np.cos(bearing), np.sin(bearing)]), bearing


def phase_code(state):
    return 3 if state.done else PHASE_NAMES.index(state.phase)


def build_navigation_features(base_positions, yaws, goal_position, goal_bearing, phases):
    """批量／单步共用；只使用当前底盘与目标几何，不读取机械臂状态。"""
    base = np.asarray(base_positions, dtype=np.float32).astype(np.float64)
    yaw = np.asarray(yaws, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.int64)
    if base.ndim != 2 or base.shape[1] not in (2, 3) or yaw.shape != (len(base),) or phases.shape != (len(base),):
        raise ValueError("Misaligned navigation state shapes")
    if not np.isfinite(base).all() or not np.isfinite(yaw).all() or not np.isin(phases, [0, 1, 2, 3]).all():
        raise ValueError("Invalid navigation states")
    delta = np.asarray(goal_position, dtype=np.float64) - base[:, :2]
    if delta.shape != (len(base), 2) or not np.isfinite(delta).all() or not np.isfinite(goal_bearing):
        raise ValueError("Invalid parking goal")
    c, s = np.cos(yaw), np.sin(yaw)
    features = np.empty((len(base), 8), dtype=np.float32)
    features[:, 0] = c * delta[:, 0] + s * delta[:, 1]
    features[:, 1] = -s * delta[:, 0] + c * delta[:, 1]
    features[:, 2] = wrap_angle(np.arctan2(delta[:, 1], delta[:, 0]) - yaw)
    features[:, 3] = wrap_angle(goal_bearing - yaw)
    features[:, 4:8] = np.eye(4, dtype=np.float32)[phases]
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite navigation features")
    return features


def build_navigation_observation(base_position, yaw, goal_position, goal_bearing, state):
    return build_navigation_features(np.asarray(base_position)[None, :], np.array([yaw]),
                                     goal_position, goal_bearing, np.array([phase_code(state)]))[0]


def recorded_yaws(observations, orientations):
    """使用保存的完整四元数恢复与专家相同的yaw，避免sin/cos的float32舍入。"""
    if orientations is None:
        return np.arctan2(observations[:, 3].astype(np.float64), observations[:, 4].astype(np.float64))
    q = np.asarray(orientations, dtype=np.float64)
    if q.shape != (len(observations), 4) or not np.isfinite(q).all():
        raise ValueError("Invalid stored base quaternions")
    norm = np.linalg.norm(q, axis=1)
    if (norm < 1e-8).any():
        raise ValueError("Invalid quaternion norm")
    x, y, z, w = (q / norm[:, None]).T
    return np.arctan2(2*(w*z + x*y), 1-2*(y*y + z*z))


def reconstruct_navigation(observations, actions, times, target, orientations=None):
    """按原状态机的单向转移重建执行前、更新后的阶段，并核对已有标签。

    不运行仿真，不生成替代训练标签；训练输出仍使用NPZ中的原actions[:,:4]。
    阶段退出是首次满足阈值的时刻，因此向量化结果等价于逐步锁存状态。
    """
    obs = np.asarray(observations)
    actions = np.asarray(actions)
    times = np.asarray(times, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[1] != 23 or actions.shape != (len(obs), 11) or times.shape != (len(obs),) or not len(obs):
        raise ValueError("Invalid source trajectory shapes")
    if not np.isfinite(obs).all() or not np.isfinite(actions).all() or not np.isfinite(times).all():
        raise ValueError("Nonfinite source trajectory")
    if not np.isin(obs[:, 22], [0, 1]).all() or (np.diff(times) <= 0).any():
        raise ValueError("Invalid phase or timestamps")
    yaw = recorded_yaws(obs, orientations)
    goal, bearing = parking_goal(target)
    delta = goal - obs[:, :2].astype(np.float64)
    distance = np.linalg.norm(delta, axis=1)
    heading_error = wrap_angle(np.arctan2(delta[:, 1], delta[:, 0]) - yaw)
    final_error = wrap_angle(bearing - yaw)
    n = len(obs)
    stages = np.zeros(n, dtype=np.int8)

    def first(mask, start=0):
        indices = np.flatnonzero(mask[start:])
        return start + int(indices[0]) if len(indices) else n

    aligned = first((distance < NAV_PARAMS["pos_tol"]) | (np.abs(heading_error) < NAV_PARAMS["ang_tol"]))
    if aligned < n:
        stages[aligned:] = 1
        oriented = (aligned if distance[aligned] < NAV_PARAMS["pos_tol"] else
                    first((distance < NAV_PARAMS["pos_tol"]) | ((distance < .15) & (np.cos(heading_error) < 0)), aligned))
        if oriented < n:
            stages[oriented:] = 2
            done = first(np.abs(final_error) < NAV_PARAMS["ang_tol"], oriented)
            stages[done:] = 3
    timeout = first(times > NAV_PARAMS["nav_timeout"])
    stages[timeout:] = 3
    mismatch = int(np.count_nonzero((stages != 3).astype(np.float32) != obs[:, 22]))
    if mismatch:
        raise ValueError(f"Reconstructed phase disagrees with stored binary phase in {mismatch} steps")

    # 标签交叉核验：防止阶段虽然匹配，但align/drive/orient划分错位。
    v = np.zeros(n); omega = np.zeros(n)
    mask = stages == 0
    omega[mask] = np.clip(NAV_PARAMS["kp_ori"]*heading_error[mask], -NAV_PARAMS["max_ang_vel"], NAV_PARAMS["max_ang_vel"])
    mask = stages == 1
    v[mask] = np.clip(NAV_PARAMS["kp_pos"]*distance[mask], NAV_PARAMS["min_lin_vel"], NAV_PARAMS["max_lin_vel"])*np.maximum(0, np.cos(heading_error[mask]))
    omega[mask] = np.clip(1.5*heading_error[mask], -.3, .3)
    mask = stages == 2
    omega[mask] = np.clip(NAV_PARAMS["kp_ori"]*final_error[mask], -NAV_PARAMS["max_ang_vel"], NAV_PARAMS["max_ang_vel"])
    left = (v-omega*NAV_PARAMS["wheel_base_width"]/2)/NAV_PARAMS["wheel_radius"]
    right = (v+omega*NAV_PARAMS["wheel_base_width"]/2)/NAV_PARAMS["wheel_radius"]
    rebuilt = np.column_stack([right, left, left, right])
    check = np.ones(n, dtype=bool)
    if timeout < n:
        check[timeout] = False  # 原控制器在首次超时步计算动作后才锁存done。
    max_error = float(np.max(np.abs(rebuilt[check] - actions[check, :4]))) if check.any() else 0.0
    if max_error > .002:
        raise ValueError(f"Reconstructed wheel labels disagree: max error {max_error:.6g} rad/s")
    x = build_navigation_features(obs[:, :3], yaw, goal, bearing, stages)
    return x, stages, {"phase_mismatch_steps": mismatch, "max_rebuilt_wheel_error_rad_s": max_error}


class NavigationPolicy(nn.Module):
    def __init__(self, hidden_dims=(64, 64)):
        super().__init__()
        layers = []
        dims = [8, *hidden_dims, 4]
        for index, (n_in, n_out) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(n_in, n_out))
            if index < len(dims)-2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_navigation_policy(checkpoint):
    c = checkpoint["config"]
    if (c.get("input_feature_version") != NAV_FEATURE_VERSION or c.get("input_dim") != 8
            or c.get("output_dim") != 4 or c.get("navigation_parameters") != NAV_PARAMS):
        raise ValueError("Wrong navigation feature/controller version")
    model = NavigationPolicy(c["hidden_dims"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("Nonfinite navigation weights")
    return model.eval()


def latest_finished_run(root):
    import json
    runs = sorted(p for p in Path(root).glob("run_*") if (p/"training_finished.json").is_file()
                  and (p/"best_model.pt").is_file()
                  and not json.loads((p/"training_finished.json").read_text(encoding="utf-8")).get("smoke_only",False))
    if not runs:
        raise FileNotFoundError(f"No completed navigation training run in {root}; train first or pass --nav-run-dir")
    return runs[-1]
