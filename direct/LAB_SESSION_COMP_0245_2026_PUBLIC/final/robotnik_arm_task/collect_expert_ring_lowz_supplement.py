"""[Low Z supplement] 在原环状数据之外补采200条低高度示范，不重采931条。

50个分布于整圈的XY，每个XY在4个连续高度区间各随机取1点。
沿用task_3、项目config的测量噪声和1秒保持，不改变专家／训练／旧划分。
"""
import argparse
from collections import Counter
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

# [Low Z supplement] 复用原采集器的读取、结果检查和保存函数；导入不会运行采集。
from collect_expert_ring import (
    target_key, read_existing_targets, inspect_new_episode, save_json, describe_target,
)

PROJECT_DIR = Path(__file__).resolve().parent
ROBOENV_DIR = PROJECT_DIR.parents[2] / "RoboEnv"
TASK_SCRIPT = PROJECT_DIR / "robotnik_arm_task_3.py"
CONFIG_PATH = PROJECT_DIR / "configs/robotnik_arm.json"
BASE_SPLIT = ROBOENV_DIR / "bc_training_nav_ring_r08/run_20261103_022339_570160/data_split.json"
OUTPUT_DIR = ROBOENV_DIR / "expert_data_23d_noise_ring_lowz_supplement"
SEED = 42
MAX_STEPS = 40000
# [Low Z supplement] 半径5段×方向10段=50个新XY，避免只补某一侧。
RADIUS_EDGES = np.linspace(.8, 2., 6)
ANGLE_SECTORS = 10
# [Low Z supplement] 不重复固定高度层；每个XY在四段各随机取一个新高度。
HEIGHT_EDGES = [.20, .275, .35, .425, .50]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan(center_xy, seed):
    """固定种子生成目标计划；重启时生成相同200点，便于按target跳过已有数据。"""
    rng = np.random.default_rng(seed)
    split_rng = np.random.default_rng(seed + 1)
    plan = []
    for radius_band in range(5):
        # 每个半径段7个train XY，剩余分给val/test；整列4个高度一起分组。
        order = split_rng.permutation(ANGLE_SECTORS)
        val_count = 1 if radius_band % 2 == 0 else 2
        owners = {int(k): "train" for k in order[:7]}
        owners.update({int(k): "val" for k in order[7:7 + val_count]})
        owners.update({int(k): "test" for k in order[7 + val_count:]})
        for sector in range(ANGLE_SECTORS):
            radius = rng.uniform(RADIUS_EDGES[radius_band], RADIUS_EDGES[radius_band + 1])
            theta = 2 * math.pi * (sector + rng.uniform(.05, .95)) / ANGLE_SECTORS
            x = round(float(center_xy[0] + radius * math.cos(theta)), 6)
            y = round(float(center_xy[1] + radius * math.sin(theta)), 6)
            group = f"xy_{radius_band * ANGLE_SECTORS + sector + 1:03d}"
            for height_band, (lo, hi) in enumerate(zip(HEIGHT_EDGES, HEIGHT_EDGES[1:])):
                # 留开端点，避免舍入后恰好变成旧的0.2／0.35／0.5m固定高度。
                z = round(float(rng.uniform(lo + .0001, hi - .0001)), 6)
                plan.append({"target": [x, y, z], "xy_group": group,
                             "proposed_split": owners[sector], "height_band": height_band,
                             "radius_band": radius_band, "angle_sector": sector})
    if len(plan) != 200 or len({target_key(p["target"]) for p in plan}) != 200:
        raise ValueError("Expected 200 distinct supplemental targets")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED, help="Target-plan seed; not a measurement-noise seed")
    parser.add_argument("--dry-run", action="store_true", help="只查看计划和待采数量，不启动仿真或写文件")
    parser.add_argument("--limit", type=int, default=None, help="只运行前几个未采目标；不改变完整计划")
    parser.add_argument("--wall-timeout", type=float, default=300.)
    args = parser.parse_args()
    if args.seed < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("seed must be nonnegative and limit positive")
    if not math.isfinite(args.wall_timeout) or args.wall_timeout <= 0:
        parser.error("wall-timeout must be finite and positive")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    center = config["robot_pybullet"]["init_link_base_position"][0][:2]
    if len(center) != 2 or not np.isfinite(center).all():
        raise ValueError("Invalid initial base XY")
    base_manifest = json.loads(BASE_SPLIT.read_text(encoding="utf-8"))
    old_data_dir = Path(base_manifest["data_dir"]).resolve()
    data_dir = args.data_dir.resolve()
    if data_dir == old_data_dir:
        raise ValueError("Use the separate supplement folder; preserve the original ring data directory")
    entries = make_plan(center, args.seed)
    # [Low Z supplement] 原931条只读核对；绝不删除、覆盖或重新运行这些目标。
    old_targets, old_count = read_existing_targets(old_data_dir)
    old_xy = {t[:2] for t in old_targets}
    if any(target_key(e["target"]) in old_targets or target_key(e["target"])[:2] in old_xy for e in entries):
        raise ValueError("Supplement overlaps an old target/XY; choose another plan seed")
    existing, existing_count = read_existing_targets(data_dir)
    pending = [e for e in entries if target_key(e["target"]) not in existing]
    selected = pending if args.limit is None else pending[:args.limit]
    fixed_paths = [Path(__file__).resolve(), TASK_SCRIPT, PROJECT_DIR / "robotnik_arm_control.py",
                   PROJECT_DIR / "collect_expert_ring.py", CONFIG_PATH, BASE_SPLIT]
    hashes = {str(p): sha256(p) for p in fixed_paths}
    plan = {"dataset_variant": "ring_lowz_continuous_supplement_v1", "plan_seed": args.seed,
            "center_xy_m": center, "radius_range_m": [.8, 2.], "height_edges_m": HEIGHT_EDGES,
            "horizontal_groups": 50, "total_targets": 200, "entries": entries,
            "proposed_split_counts": dict(Counter(e["proposed_split"] for e in entries)),
            "split_note": "XY groups stay together; provisional supplement-only split. Old splits are not modified.",
            "base_split_path": str(BASE_SPLIT), "base_split_sha256": sha256(BASE_SPLIT),
            "robot_config_sha256": sha256(CONFIG_PATH), "expert_script_sha256": sha256(TASK_SCRIPT),
            "controller_sha256": sha256(PROJECT_DIR / "robotnik_arm_control.py")}
    plan_path = data_dir / "collection_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Existing plan differs. Keep the same seed/config/controller or use a new output folder.")
    print(f"原环状数据：{old_count}条，仅读取；旧划分和训练脚本不改。", flush=True)
    print(f"补采计划：50个XY × 4个连续随机高度 = 200条；seed={args.seed}", flush=True)
    print(f"半径0.8～2m；高度0.2～0.5m；预留新分组={plan['proposed_split_counts']}", flush=True)
    print(f"补采目录已有{existing_count}个文件；本计划已有{len(entries)-len(pending)}点；待补{len(pending)}点；本次运行{len(selected)}点。", flush=True)
    print(f"输出：{data_dir}\n配置：{CONFIG_PATH}，沿用测量噪声；额外力矩脉冲关闭；连续保持1秒。", flush=True)
    if args.dry_run:
        print("Dry run完成：没有启动仿真，也没有写入文件。")
        return
    if not selected:
        print("这200个目标已全部有记录，无需重采。")
        return

    log_dir = (args.log_dir or ROBOENV_DIR / "expert_collection_logs" /
               datetime.now().strftime("ring_lowz_supplement_%Y%m%d_%H%M%S_%f")).resolve()
    if log_dir.exists():
        raise FileExistsError(f"Refusing to overwrite collection logs: {log_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True)
    (log_dir / "logs").mkdir()
    if not plan_path.exists():
        save_json(plan_path, plan)
    save_json(log_dir / "collection_config.json", {
        "plan": plan, "data_dir": str(data_dir), "selected_entries": selected,
        "source_sha256": hashes, "robot_config_snapshot": config,
        "noise_policy": "project_config_measurement_noise_unchanged_no_extra_torque_noise",
        "measurement_noise_seed_fixed": False, "hold_seconds": 1., "max_steps": MAX_STEPS,
        "existing_episode_policy": "skip any recorded target, including episodes not meeting hold criterion",
        "original_data_files": old_count, "original_data_dir": str(old_data_dir)})
    env = os.environ.copy()
    env.update(ROBOENV_HEADLESS="1", ROBOENV_MAX_STEPS=str(MAX_STEPS), PYTHONIOENCODING="utf-8",
               PYTHONDONTWRITEBYTECODE="1")
    fields = ["index", "xy_group", "proposed_split", "height_band", "radius_m", "angle_deg",
              "target_x", "target_y", "target_z", "status", "reached_once", "held_success",
              "min_error_m", "first_reach_time_s", "navigation_done_time_s", "end_time_s",
              "continuous_hold_seconds", "final_error_m", "termination_reason", "rollout_file", "log_file", "error"]
    rows = []

    def summary(state):
        saved = sum(bool(r["rollout_file"]) for r in rows)
        save_json(log_dir / "summary.json", {
            "state": state, "dataset_variant": plan["dataset_variant"], "total_plan_targets": 200,
            "selected_targets": len(selected), "processed_targets": len(rows), "saved_episodes": saved,
            "reached_once_count": sum(r["reached_once"] for r in rows),
            "held_successes": sum(r["held_success"] for r in rows),
            "episode_failures": sum(r["status"] == "episode_failure" for r in rows),
            "process_errors": sum(r["status"] == "process_error" for r in rows),
            "skipped_existing_targets": 200 - len(pending), "missing_after_saved_episodes": len(pending) - saved,
            "partial_run": len(selected) < len(pending), "data_dir": str(data_dir), "log_dir": str(log_dir),
            "old_data_and_splits_modified": False})

    summary("running")
    with (log_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); stream.flush()
        try:
            for index, entry in enumerate(selected, 1):
                for path, expected in hashes.items():
                    if sha256(path) != expected:
                        raise RuntimeError(f"Source/config changed during collection: {path}")
                target = entry["target"]
                log = log_dir / "logs" / f"episode_{index:04d}.log"
                row = dict.fromkeys(fields)
                row.update(index=index, xy_group=entry["xy_group"], proposed_split=entry["proposed_split"],
                           height_band=entry["height_band"], target_x=target[0], target_y=target[1], target_z=target[2],
                           status="process_error", reached_once=False, held_success=False, log_file=str(log), error="")
                row.update(describe_target(target, center))
                print(f"[{index}/{len(selected)}] target={target}, {entry['xy_group']}, reserved={entry['proposed_split']}", flush=True)
                try:
                    with log.open("w", encoding="utf-8") as logfile:
                        result = subprocess.run([sys.executable, "-B", "-u", str(TASK_SCRIPT),
                            "--target", *map(str, target), "--noise-std", "0", "--hold-seconds", "1.0",
                            "--output-dir", str(data_dir)], env=env, stdout=logfile, stderr=subprocess.STDOUT,
                            timeout=args.wall_timeout, check=False)
                    if result.returncode:
                        raise RuntimeError(f"Expert exited with code {result.returncode}; see log")
                    row.update(inspect_new_episode(log, data_dir, target))
                    row["status"] = "held_success" if row["held_success"] else "episode_failure"
                except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
                    row["error"] = str(error)
                # 不删除失败记录、不替换标签，后续合并时再决定是否使用。
                rows.append(row); writer.writerow(row); stream.flush()
                summary("running")
                print(f"  {row['status']} | min={row['min_error_m']}m | final={row['final_error_m']}m", flush=True)
        except BaseException:
            summary("interrupted_or_aborted")
            raise
    summary("completed")
    print(f"补采结束。轨迹：{data_dir}\n汇总：{log_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
