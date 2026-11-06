"""[Ring dataset change] 复制自collect_expert.py，采集931个正式环状目标。

沿用task_3、原配置测量噪声和连续保持1秒；原NPZ仍为23维并保存ee_positions。
修改仅涉及目标分布、独立数据/日志目录和到达统计，不修改机器人配置或控制器。
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


task_script = Path(__file__).resolve().with_name("robotnik_arm_task_3.py")
ROBOENV_DIR = Path(__file__).resolve().parents[3] / "RoboEnv"
COLLECT_OUTPUT_DIR = ROBOENV_DIR / "expert_data_23d_noise_ring"  # [Ring dataset change] 与旧1014条分开。

# [Ring dataset change] 高度两端0.15m、中间0.25m；仅采931基础目标，暂不加重复轨迹。
HEIGHTS = [0.20, 0.35, 0.50, 0.75, 1.00, 1.15, 1.30]
# (水平半径m, 均匀角度数)，每个高度133点；圆心只采一次，不重复360度。
RADIUS_ANGLE_COUNTS = [(0.0, 1), (0.2, 4), (0.4, 6), (0.6, 8), (0.8, 10),
                       (1.0, 12), (1.2, 14), (1.4, 16), (1.6, 18), (1.8, 20), (2.0, 24)]
MAX_STEPS = 40000  # 沿用原正式采集上限40s，给2m外圈留出转向、移动和保持时间。


def make_targets(center_xy):
    # [Ring dataset change] 固定配置中的机器人初始化，只改变目标位置。
    # 角度从世界+X轴逆时针计；使用k/count，避免包含360度的重复端点。
    targets = []
    for z in HEIGHTS:
        for radius, count in RADIUS_ANGLE_COUNTS:
            for k in range(count):
                theta = 2 * math.pi * k / count
                xyz = [round(center_xy[0] + radius * math.cos(theta), 12),
                       round(center_xy[1] + radius * math.sin(theta), 12), z]
                targets.append([0.0 if v == 0 else v for v in xyz])
    if len({target_key(t) for t in targets}) != len(targets):
        raise ValueError("Duplicate targets in ring layout")
    return targets


def describe_target(target, center_xy):
    """[Ring dataset change] 在CSV中附加半径和角度，方便按区域检查结果。"""
    dx, dy = target[0] - center_xy[0], target[1] - center_xy[1]
    radius = math.hypot(dx, dy)
    return {"radius_m": round(radius, 6),
            "angle_deg": round(math.degrees(math.atan2(dy, dx)) % 360, 6) if radius > 1e-9 else None}


def target_key(target):
    values = np.asarray(target, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Target must contain three finite coordinates")
    return tuple(round(float(value), 6) for value in values)


def read_existing_targets(data_dir):
    # [Ring dataset change] 只扫描本环状数据目录，实现断点续采；不自动合并旧单侧数据。
    # 保留原实验的全部已有轨迹，包括未满足保持条件的轨迹；不自动覆盖或重采。
    existing = set()
    files = sorted(data_dir.glob("expert_episode_*.npz"))
    for path in files:
        try:
            with np.load(path, allow_pickle=False) as data:
                if int(data["input_dim"].item()) != 23:
                    raise ValueError("Expected 23D expert data")
                existing.add(target_key(data["target"]))
        except (OSError, ValueError, KeyError) as error:
            raise RuntimeError(f"Cannot inspect existing episode; no files modified: {path}: {error}") from error
    return existing, len(files)


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def inspect_new_episode(log_path, data_dir, target):
    # [Height expansion] 专家在日志中给出实际保存路径，读取该文件核对目标和结果。
    prefix = "Expert trajectory saved to: "
    saved = [line.split(prefix, 1)[1].strip()
             for line in log_path.read_text(encoding="utf-8").splitlines() if prefix in line]
    if len(saved) != 1:
        raise RuntimeError(f"Expected one saved episode in log, found {len(saved)}")
    path = Path(saved[0]).resolve()
    if path.parent != data_dir.resolve() or path.suffix != ".npz":
        raise RuntimeError(f"Unexpected saved episode location: {path}")
    with np.load(path, allow_pickle=False) as data:
        if target_key(data["target"]) != target_key(target):
            raise RuntimeError("Saved target differs from requested target")
        success = bool(data["success"].item())
        final_error = float(data["final_error"].item())
        best_error = float(data["best_error"].item())
        # 非有限结果明确标记，不改变或删除已经保存的原始文件。
        return {"rollout_file": str(path), "held_success": success,
                # [Ring dataset change] 到达和连续保持分开记录，保持失败不等于未到达。
                "reached_once": bool(best_error < 0.01),
                "first_reach_time_s": float(data["first_reach_time"]) if np.isfinite(data["first_reach_time"]) else None,
                "navigation_done_time_s": float(data["navigation_done_time"]) if np.isfinite(data["navigation_done_time"]) else None,
                "end_time_s": float(data["final_timestamp"]),
                "continuous_hold_seconds": float(data["continuous_hold_seconds"]),
                "final_error_m": final_error if np.isfinite(final_error) else None,
                "min_error_m": best_error if np.isfinite(best_error) else None,
                "termination_reason": str(data["termination_reason"].item())}


def main():
    parser = argparse.ArgumentParser(description="采集931个环绕小车、半径0～2m、7层非均匀高度的专家目标。")
    parser.add_argument("--data-dir", type=Path, default=COLLECT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="只统计已有和待采集目标，不运行仿真")
    parser.add_argument("--limit", type=int, default=None, help="可选：本次只运行前几个缺失目标")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--wall-timeout", type=float, default=300., help="每个子进程的真实时间上限（秒）")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not np.isfinite(args.wall_timeout) or args.wall_timeout <= 0:
        parser.error("--wall-timeout must be finite and positive")

    # [Ring dataset change] 只读取项目这份配置；不覆盖测量噪声开关或修改磁盘配置。
    config_path = task_script.parent / "configs" / "robotnik_arm.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    center_xy = config["robot_pybullet"]["init_link_base_position"][0][:2]
    if len(center_xy) != 2 or not np.isfinite(center_xy).all():
        raise ValueError("Initial base XY must contain two finite values")
    targets = make_targets(center_xy)
    data_dir = args.data_dir.resolve()
    existing, file_count = read_existing_targets(data_dir)
    pending = [target for target in targets if target_key(target) not in existing]
    skipped = len(targets) - len(pending)
    print(f"配置：{config_path}；测量噪声沿用配置。", flush=True)
    print(f"环状目标：{len(targets)} 个；圆心XY={center_xy}；半径0～2m", flush=True)
    print(f"高度：{HEIGHTS}；每个高度{sum(n for _, n in RADIUS_ANGLE_COUNTS)}个位置", flush=True)
    print(f"已有文件：{file_count} 个；其中本次网格已覆盖：{skipped} 个目标", flush=True)
    print(f"待补采目标：{len(pending)} 个", flush=True)
    print(f"待补采高度：{sorted({target[2] for target in pending})}", flush=True)
    selected = pending if args.limit is None else pending[:args.limit]
    print(f"本次计划：{len(selected)}条；每条上限{MAX_STEPS * config['sim']['time_step']:g}s，连续保持1秒。", flush=True)
    if args.dry_run:
        print(f"Dry run：本次将运行 {len(selected)} 个目标，未写入或修改任何数据。")
        return
    if not selected:
        print("目标已全部有轨迹文件，无需重复采集。")
        return

    # [Ring dataset change] 原task_3直接子进程调用，无仿真包装器或噪声覆盖。
    # 继续关闭力矩脉冲、连续保持1秒、每条最多40000步；不修改初始朝向。
    fixed_paths = [task_script, task_script.with_name("robotnik_arm_control.py"), config_path]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in fixed_paths}
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir or ROBOENV_DIR / "expert_collection_logs" / datetime.now().strftime("ring_%Y%m%d_%H%M%S_%f")
    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "logs").mkdir()
    save_json(log_dir / "collection_config.json", {
        # [Ring dataset change] 明确保存采样方案与原配置，便于后续按目标分组。
        "data_dir": str(data_dir), "dataset_variant": "config_noise_ring_0_to_2m_7_heights_v1",
        "config_path": str(config_path), "center_xy_m": center_xy, "heights_m": HEIGHTS,
        "radius_angle_counts": RADIUS_ANGLE_COUNTS, "extra_low_height_repeats": 0,
        "total_grid_targets": len(targets), "existing_files": file_count, "skipped_existing_targets": skipped,
        "missing_targets_before_run": len(pending), "selected_targets": selected,
        "sensor_noise": "unchanged_config", "torque_noise_std_nm": 0., "hold_seconds": 1.,
        "max_steps": MAX_STEPS, "robot_config_snapshot": config, "source_sha256": hashes,
        "existing_episode_policy": "skip any existing 23D target; preserve files regardless of hold-success status"})
    env = os.environ.copy()
    env.update(ROBOENV_HEADLESS="1", ROBOENV_MAX_STEPS=str(MAX_STEPS), PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    fields = ["index", "radius_m", "angle_deg", "target_x", "target_y", "target_z", "status", "reached_once", "held_success", "min_error_m",
              "first_reach_time_s", "navigation_done_time_s", "end_time_s", "continuous_hold_seconds",
              "final_error_m", "termination_reason", "rollout_file", "log_file", "error"]
    rows = []
    print(f"采集输出：{data_dir}\n本次日志：{log_dir}", flush=True)
    with (log_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields);writer.writeheader();stream.flush()
        for index, target in enumerate(selected, 1):
            for path, expected in hashes.items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"采集期间源代码或配置发生变化，请停止修改后重新运行：{path}")
            print(f"[{index}/{len(selected)}] Target: {target}", flush=True)
            log = log_dir / "logs" / f"episode_{index:04d}.log"
            row = dict.fromkeys(fields)
            row.update(index=index, target_x=target[0], target_y=target[1], target_z=target[2],
                       status="process_error", reached_once=False, held_success=False, log_file=str(log), error="")
            row.update(describe_target(target, center_xy))
            try:
                with log.open("w", encoding="utf-8") as logfile:
                    result = subprocess.run([sys.executable, "-B", "-u", str(task_script),
                        "--target", *map(str, target), "--noise-std", "0", "--hold-seconds", "1.0",
                        "--output-dir", str(data_dir)], env=env, stdout=logfile, stderr=subprocess.STDOUT,
                        timeout=args.wall_timeout, check=False)
                if result.returncode:
                    raise RuntimeError(f"Expert process exited with code {result.returncode}; see log")
                row.update(inspect_new_episode(log, data_dir, target))
                row["status"] = "held_success" if row["held_success"] else "episode_failure"
                existing.add(target_key(target))
            except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
                row["error"] = str(error)
            # [Height expansion] 失败不删除轨迹、不终止后续目标；逐条写入进度，方便断点续采。
            rows.append(row);writer.writerow(row);stream.flush()
            saved_count = sum(bool(r["rollout_file"]) for r in rows)
            save_json(log_dir / "summary.json", {
                "state": "completed" if len(rows) == len(selected) else "running",
                "selected_targets": len(selected), "processed_targets": len(rows),
                "saved_episodes": saved_count, "held_successes": sum(r["held_success"] for r in rows),
                "reached_once_count": sum(r["reached_once"] for r in rows),
                "total_layout_targets": len(targets), "skipped_existing_targets": skipped,
                "episode_failures": sum(r["status"] == "episode_failure" for r in rows),
                "process_errors": sum(r["status"] == "process_error" for r in rows),
                "missing_targets_before_run": len(pending), "missing_after_saved_episodes": len(pending)-saved_count,
                "data_dir": str(data_dir), "log_dir": str(log_dir),
                "partial_run": len(selected) < len(pending)})
            print(f"  {row['status']} | min={row['min_error_m']} m | final={row['final_error_m']} m", flush=True)
    print(f"本次采集结束。轨迹：{data_dir}\n汇总：{log_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
