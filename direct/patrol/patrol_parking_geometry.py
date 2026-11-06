"""[巡检相对停车改动] 每段由当前观测位置规划，段内固定停车点。

仅生成导航目标，不生成轮速或力矩。不修改历史训练特征模块。
输入/返回可使用世界坐标保存；几何只依赖目标与当前底盘的相对XY，
因此整体水平平移或旋转时，停车点随之平移或旋转。
"""
import numpy as np


PARKING_GEOMETRY_VERSION = 'target_minus_observed_stage_start_base_xy_v1'


def parking_goal_from_current_base(target, base_position, base_yaw):
    """使用一段的首帧带噪观测；调用者必须将结果固定到该段结束。

    保留原专家随目标离地高度变化的standoff曲线；仿真地面为世界z=0。
    最终车头指向目标。目标水平位置与底盘重合时，使用当前朝向。
    """
    target = np.asarray(target, dtype=np.float64)
    base = np.asarray(base_position, dtype=np.float64)
    if (target.shape != (3,) or base.shape != (3,)
            or not np.isfinite(target).all() or not np.isfinite(base).all()
            or not np.isfinite(base_yaw)):
        raise ValueError('Expected finite target XYZ, observed base XYZ and yaw')

    # [改动] 必须减去本段起始底盘位置，不能atan2(target_y, target_x)。
    delta = target[:2] - base[:2]
    if np.linalg.norm(delta) > 1e-8:
        direction = delta / np.linalg.norm(delta)
    else:
        direction = np.array([np.cos(base_yaw), np.sin(base_yaw)])
    bearing = float(np.arctan2(direction[1], direction[0]))
    standoff = float(np.interp(target[2], [.1, .5, 1.1, 1.25], [.8, .65, .65, .35]))
    goal_world_xy = target[:2] - standoff * direction
    return goal_world_xy, bearing
