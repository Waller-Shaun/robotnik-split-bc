"""
检查 速度峰值发生在什么时候、哪个关节，以及持续多久
"""
from collections import Counter
from pathlib import Path

import numpy as np


DATA_DIR = (
    Path(__file__).resolve().parents[3]
    / "RoboEnv"
    / "expert_data"
)

peak_times = []
peak_values = []
peak_joints = []
high_speed_fractions = []
failed_targets = []

for path in sorted(DATA_DIR.glob("*.npz")):
    with np.load(path, allow_pickle=False) as data:
        times = data["timestamps"]
        velocities = data["observations"][:, 12:19]

        # 找到整条轨迹中绝对速度最大的时间步和关节
        abs_velocity = np.abs(velocities)
        step, joint = np.unravel_index(
            np.argmax(abs_velocity),
            abs_velocity.shape,
        )

        peak_times.append(float(times[step]))
        peak_values.append(float(abs_velocity[step, joint]))
        peak_joints.append(int(joint + 1))

        # 某个时间步，只要任意关节超过阈值，就计为高速时间步
        high_speed = np.any(abs_velocity > 3.0, axis=1)
        high_speed_fractions.append(float(high_speed.mean()))

        final_error = float(data["final_error"])
        if final_error >= 0.01:
            failed_targets.append(
                (data["target"].tolist(), final_error)
            )

if not peak_times:
    raise SystemExit(f"没有找到轨迹文件：{DATA_DIR}")

print("========== 速度检查 ==========")
print(f"轨迹数量：{len(peak_times)}")

print("\n速度峰值大小（rad/s）：")
print("最小／中位数／最大：",
      np.round(np.percentile(peak_values, [0, 50, 100]), 4))

print("\n速度峰值发生时间（秒）：")
print("最小／中位数／最大：",
      np.round(np.percentile(peak_times, [0, 50, 100]), 4))

print("\n峰值发生在各关节的轨迹数量：")
print(dict(sorted(Counter(peak_joints).items())))

print("\n高速时间步占整条轨迹的比例（%）：")
print("最小／中位数／最大：",
      np.round(
          np.percentile(high_speed_fractions, [0, 50, 100]) * 100,
          4,
      ))

print("\n最终误差未达标的目标：")
for target, error in failed_targets:
    print(f"目标：{target}，最终误差：{error:.6f} m")