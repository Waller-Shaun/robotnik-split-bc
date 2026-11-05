"""
检查 expert_data 中的轨迹文件，生成汇总报告。
"""
from collections import Counter
from pathlib import Path
import csv

import numpy as np


# 根据当前项目的目录关系定位 RoboEnv
ROBOENV_DIR = Path(__file__).resolve().parents[3] / "RoboEnv"
DATA_DIR = ROBOENV_DIR / "expert_data"
REPORT_PATH = DATA_DIR / "inspection_report.csv"

ERROR_THRESHOLD = 0.01  # 1 厘米；用于初步筛选，不等同于确认终止原因


def inspect_file(path):
    """检查一条轨迹，返回用于汇总的结果。"""
    result = {
        "file": path.name,
        "target_x": "",
        "target_y": "",
        "target_z": "",
        "samples": "",
        "last_timestamp_s": "",
        "final_error_m": "",
        "max_abs_joint_velocity": "",
        "status": "invalid",
        "reason": "",
    }

    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "observations",
                "actions",
                "timestamps",
                "target",
                "final_error",
            }
            missing = required - set(data.files)

            if missing:
                raise ValueError(f"缺少字段：{sorted(missing)}")

            obs = data["observations"]
            actions = data["actions"]
            times = data["timestamps"]
            target = data["target"]
            final_error_array = data["final_error"]

            # 先验证形状，再读取对应列，避免误读数据
            if obs.ndim != 2 or obs.shape[1] != 22:
                raise ValueError(f"observations 形状异常：{obs.shape}")

            n = len(obs)
            if n < 2:
                raise ValueError("轨迹少于两个样本")

            if actions.shape != (n, 11):
                raise ValueError(f"actions 形状异常：{actions.shape}")

            if times.shape != (n,):
                raise ValueError(f"timestamps 形状异常：{times.shape}")

            if target.shape != (3,):
                raise ValueError(f"target 形状异常：{target.shape}")

            if final_error_array.shape != ():
                raise ValueError("final_error 应是标量")

            for name in required:
                if not np.isfinite(data[name]).all():
                    raise ValueError(f"{name} 包含 NaN 或无穷值")

            final_error = float(final_error_array)
            if final_error < 0:
                raise ValueError("最终距离误差不能为负数")

            dt = np.diff(times)
            if not np.all(dt > 0):
                raise ValueError("时间戳没有严格递增")

            if not np.allclose(dt, dt[0], rtol=1e-5, atol=1e-8):
                raise ValueError("采样时间间隔不一致")

            if not np.allclose(obs[:, 19:22], target, atol=1e-6):
                raise ValueError("输入中的目标与 target 字段不一致")

            heading_norm = obs[:, 3] ** 2 + obs[:, 4] ** 2
            if not np.allclose(heading_norm, 1.0, atol=1e-5):
                raise ValueError("朝向 sin、cos 不满足单位圆关系")

            if np.any(np.abs(actions[:, 4:11]) > 30.0 + 1e-4):
                raise ValueError("专家关节力矩超过 ±30 N·m")

            max_velocity = float(np.max(np.abs(obs[:, 12:19])))

            result.update({
                "target_x": float(target[0]),
                "target_y": float(target[1]),
                "target_z": float(target[2]),
                "samples": n,
                "last_timestamp_s": float(times[-1]),
                "final_error_m": final_error,
                "max_abs_joint_velocity": max_velocity,
                "status": (
                    "candidate"
                    if final_error < ERROR_THRESHOLD
                    else "above_threshold"
                ),
                "reason": "",
            })

    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"

    return result


def main():
    files = sorted(DATA_DIR.glob("*.npz"))

    if not files:
        raise SystemExit(f"没有找到 npz 文件：{DATA_DIR}")

    results = []

    for index, path in enumerate(files, start=1):
        result = inspect_file(path)
        results.append(result)

        print(
            f"[{index}/{len(files)}] "
            f"{path.name}: {result['status']}",
            flush=True,
        )

    # 每条轨迹占一行，便于查看和后续筛选
    with REPORT_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    counts = Counter(row["status"] for row in results)
    valid = [row for row in results if row["status"] != "invalid"]

    # 四舍五入后比较坐标，避免浮点表示差异影响重复统计
    target_counts = Counter(
        (
            round(row["target_x"], 6),
            round(row["target_y"], 6),
            round(row["target_z"], 6),
        )
        for row in valid
    )

    duplicate_extras = sum(count - 1 for count in target_counts.values())
    high_velocity = sum(
        row["max_abs_joint_velocity"] > 3.0
        for row in valid
    )

    print("\n========== 检查汇总 ==========")
    print(f"文件总数：{len(results)}")
    print(f"结构检查通过且误差 < 1 cm：{counts['candidate']}")
    print(f"结构检查通过但误差 >= 1 cm：{counts['above_threshold']}")
    print(f"数据无效：{counts['invalid']}")
    print(f"有效文件中的不同目标数：{len(target_counts)}")
    print(f"相同目标的额外轨迹数：{duplicate_extras}")
    print(f"实际关节速度曾超过 3 rad/s 的轨迹数：{high_velocity}")

    if valid:
        targets = np.array(list(target_counts))
        print(f"目标坐标最小值：{targets.min(axis=0)}")
        print(f"目标坐标最大值：{targets.max(axis=0)}")

    print(f"\n检查报告已保存：{REPORT_PATH}")


if __name__ == "__main__":
    main()