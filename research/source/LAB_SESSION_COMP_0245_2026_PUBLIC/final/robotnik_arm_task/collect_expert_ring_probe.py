"""[Ring change] 沿用collect_expert的subprocess循环，只把网格换成12个环状目标。

task_3读取同目录configs/robotnik_arm.json；测量噪声完全由该配置控制。
不修改配置，不覆盖noise_flag，不使用runpy或仿真包装器。
"""
import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

TASK_SCRIPT = Path(__file__).resolve().with_name("robotnik_arm_task_3.py")
ROBOENV_DIR = Path(__file__).resolve().parents[3] / "RoboEnv"
CONFIG_PATH = TASK_SCRIPT.parent / "configs" / "robotnik_arm.json"

# [Ring change] 本次只改变目标方向，机器人初始位置、朝向、噪声均沿用配置。
RADIUS = 1.1
TARGET_Z = 0.5
ANGLE_STEP = 30                 # 0、30、...、330度，共12个目标；不重复360度。
MAX_STEPS = 20000              # 保留本轮方向探索的20秒上限（配置步长0.001秒）。


def make_targets(center_xy):
    targets = []
    for angle in range(0, 360, ANGLE_STEP):
        theta = math.radians(angle)
        xyz = [round(center_xy[0] + RADIUS * math.cos(theta), 12),
               round(center_xy[1] + RADIUS * math.sin(theta), 12), TARGET_Z]
        targets.append((angle, [0.0 if v == 0 else v for v in xyz]))
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只列出目标，不启动仿真")
    parser.add_argument("--limit", type=int, default=None, help="可选：只运行前几个目标")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    # [Ring change] 只读配置。圆心取配置中的初始底盘XY；不会改动机器人起点。
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    targets = make_targets(config["robot_pybullet"]["init_link_base_position"][0][:2])
    selected = targets if args.limit is None else targets[:args.limit]
    print(f"配置：{CONFIG_PATH}\n测量噪声沿用配置，不在运行时覆盖。", flush=True)
    for angle, target in selected:
        print(f"{angle:3d} deg -> {target}", flush=True)
    if args.dry_run:
        return

    output = args.output_dir or ROBOENV_DIR / "expert_ring_probe" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    data_dir, log_dir = output / "episodes", output / "logs"
    data_dir.mkdir()
    log_dir.mkdir()
    (output / "collection_config.json").write_text(json.dumps({
        "config_path": str(CONFIG_PATH), "robot_config_snapshot": config,
        "radius_m": RADIUS, "height_m": TARGET_Z, "selected_targets": selected,
        "max_steps": MAX_STEPS, "sensor_noise": "unchanged_config",
        "torque_noise_std_nm": 0.0, "hold_seconds": 1.0,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 与原collect_expert一样，直接运行task_3，关闭逐条画图。
    env = os.environ.copy()
    env.update(ROBOENV_HEADLESS="1", ROBOENV_MAX_STEPS=str(MAX_STEPS), PYTHONIOENCODING="utf-8")
    fields = ["angle_deg", "target_x", "target_y", "target_z", "status", "held_success",
              "min_error_m", "final_error_m", "navigation_done_time_s", "rollout_file", "error"]
    rows = []
    print(f"结果：{output}", flush=True)
    with (output / "results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, (angle, target) in enumerate(selected, 1):
            print(f"[{index}/{len(selected)}] {angle} deg; target={target}", flush=True)
            log = log_dir / f"episode_{index:03d}.log"
            row = dict.fromkeys(fields, "")
            row.update(angle_deg=angle, target_x=target[0], target_y=target[1], target_z=target[2],
                       status="process_error", held_success=False)
            try:
                with log.open("w", encoding="utf-8") as logfile:
                    subprocess.run([
                        sys.executable, "-B", "-u", str(TASK_SCRIPT), "--target", *map(str, target),
                        # 仅关闭task_3额外的力矩脉冲，配置中的测量噪声保持不变。
                        "--noise-std", "0", "--hold-seconds", "1.0", "--output-dir", str(data_dir),
                    ], env=env, stdout=logfile, stderr=subprocess.STDOUT, check=True, timeout=300)
                prefix = "Expert trajectory saved to: "
                saved = [line.split(prefix, 1)[1].strip()
                         for line in log.read_text(encoding="utf-8").splitlines() if prefix in line]
                if len(saved) != 1:
                    raise ValueError("Expected one saved expert trajectory; see log")
                with np.load(saved[0], allow_pickle=False) as data:
                    if not np.allclose(data["target"], target, atol=1e-8, rtol=0):
                        raise ValueError("Saved target does not match requested target")
                    success = bool(data["success"])
                    nav_time = float(data["navigation_done_time"])
                    row.update(status="held_success" if success else "episode_failure", held_success=success,
                               min_error_m=float(data["best_error"]), final_error_m=float(data["final_error"]),
                               navigation_done_time_s=nav_time if np.isfinite(nav_time) else "",
                               rollout_file=saved[0])
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                row["error"] = str(error)
            rows.append(row)
            writer.writerow(row)
            file.flush()
            summary = {"state": "completed" if len(rows) == len(selected) else "running",
                       "planned_targets": len(selected), "processed_targets": len(rows),
                       "held_successes": sum(r["held_success"] for r in rows),
                       "process_errors": sum(r["status"] == "process_error" for r in rows),
                       "sensor_noise": "unchanged_config", "output_dir": str(output)}
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  {row['status']} | final_error={row['final_error_m']} m", flush=True)
    print(f"结束：{summary['held_successes']}/{len(selected)}完成保持；汇总：{output / 'summary.json'}")


if __name__ == "__main__":
    main()
