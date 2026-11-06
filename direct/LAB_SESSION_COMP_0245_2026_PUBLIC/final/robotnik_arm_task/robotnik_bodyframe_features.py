"""[Body frame change] 训练和推理共用的26维特征转换，不改写原始NPZ。

只替换目标位置与末端误差两组特征；其余原始状态保留。
这里是按yaw旋转的水平底盘坐标系，局部X朝前、Y朝左、Z向上，
不是包含roll/pitch的完整三维姿态变换。
"""
import numpy as np


BODYFRAME_FEATURE_VERSION = "obs23_target_and_ee_error_base_yaw_frame_v1"


def build_bodyframe_observations(observations, ee_positions):
    """同一执行前时刻：(N,23)源观测和(N,3)世界末端位置转为(N,26)。

    0:3  保留底盘世界xyz；3:5 保留sin(yaw)、cos(yaw)。
    5:19 保留机械臂q和qd；22 保留原二值阶段。
    19:22 替换为 Rz(-yaw) @ (target_world - base_world)。
    23:26 追加 Rz(-yaw) @ (target_world - ee_world)。

    此实验保留绝对底盘位置和朝向，不能称为完全旋转／平移不变。
    高度采样条件必须在调用本函数前用原始世界目标z计算。
    """
    obs = np.asarray(observations, dtype=np.float32)
    ee = np.asarray(ee_positions, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[1] != 23:
        raise ValueError(f"Expected source observations (N,23), got {obs.shape}")
    if ee.shape != (len(obs), 3):
        raise ValueError(f"EE positions must align with observations, got {ee.shape}")
    if not np.isfinite(obs).all() or not np.isfinite(ee).all():
        raise ValueError("Source observations and EE positions must be finite")

    # [Body frame change] 源列顺序是sin在前、cos在后，避免旋转方向用反。
    sine = obs[:, 3].astype(np.float64)
    cosine = obs[:, 4].astype(np.float64)
    norm = np.hypot(sine, cosine)
    if np.any(norm < 1e-6):
        raise ValueError("Invalid yaw sin/cos pair")
    sine = sine / norm
    cosine = cosine / norm

    def rotate_world_vector(vector):
        # Rz(-yaw): x_b=cos(yaw)*dx+sin(yaw)*dy; y_b=-sin(yaw)*dx+cos(yaw)*dy.
        result = np.empty_like(vector, dtype=np.float64)
        result[:, 0] = cosine * vector[:, 0] + sine * vector[:, 1]
        result[:, 1] = -sine * vector[:, 0] + cosine * vector[:, 1]
        result[:, 2] = vector[:, 2]  # 水平旋转保持Z分量。
        return result

    target_world = obs[:, 19:22].astype(np.float64)
    base_world = obs[:, 0:3].astype(np.float64)
    relative_target = rotate_world_vector(target_world - base_world)
    ee_error = rotate_world_vector(target_world - ee)

    # [Body frame change] 创建新数组，绝不就地修改源数据或影响原始采样条件。
    result = np.empty((len(obs), 26), dtype=np.float32)
    result[:, :23] = obs
    result[:, 19:22] = relative_target
    result[:, 23:26] = ee_error
    if not np.isfinite(result).all():
        raise ValueError("Derived body-frame observations must be finite")
    return result


def build_bodyframe_observation(source_observation, ee_position):
    """推理时的单步入口；复用训练的同一函数，避免另写一套公式。"""
    obs = np.asarray(source_observation, dtype=np.float32)
    ee = np.asarray(ee_position, dtype=np.float64)
    if obs.shape != (23,) or ee.shape != (3,):
        raise ValueError("Expected a 23D source observation and a 3D world EE position")
    return build_bodyframe_observations(obs[None, :], ee[None, :])[0]
