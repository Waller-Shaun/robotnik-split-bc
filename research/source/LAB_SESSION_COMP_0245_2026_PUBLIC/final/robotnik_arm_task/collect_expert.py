import argparse
import csv
from datetime import datetime
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


task_script = Path(__file__).resolve().with_name("robotnik_arm_task_3.py")
ROBOENV_DIR = Path(__file__).resolve().parents[3] / "RoboEnv"
COLLECT_OUTPUT_DIR = ROBOENV_DIR / "expert_data_23d_noise"


def make_targets():
    # [Height expansion] 只扩展z：0.1～1.3m；x/y范围、0.1m步长保持原样。
    # 共13×6×13=1014个目标；已有468个目标时，应补采546个。
    return [[x / 10, y / 10, z / 10]
            for x, y, z in product(range(-6, 7), range(9, 15), range(1, 14))]


def target_key(target):
    values = np.asarray(target, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Target must contain three finite coordinates")
    return tuple(round(float(value), 6) for value in values)


def read_existing_targets(data_dir):
    # [Height expansion] 根据NPZ内部target去重，不依赖时间戳文件名。
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
                "final_error_m": final_error if np.isfinite(final_error) else None,
                "min_error_m": best_error if np.isfinite(best_error) else None,
                "termination_reason": str(data["termination_reason"].item())}


def main():
    parser = argparse.ArgumentParser(description="补采单侧网格中缺失的23维专家轨迹，z扩展至0.1～1.3m。")
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

    targets = make_targets()
    data_dir = args.data_dir.resolve()
    existing, file_count = read_existing_targets(data_dir)
    pending = [target for target in targets if target_key(target) not in existing]
    skipped = len(targets) - len(pending)
    print(f"完整目标网格：{len(targets)} 个", flush=True)
    print(f"已有文件：{file_count} 个；其中本次网格已覆盖：{skipped} 个目标", flush=True)
    print(f"待补采目标：{len(pending)} 个", flush=True)
    print(f"待补采高度：{sorted({target[2] for target in pending})}", flush=True)
    selected = pending if args.limit is None else pending[:args.limit]
    if args.dry_run:
        print(f"Dry run：本次将运行 {len(selected)} 个目标，未写入或修改任何数据。")
        return
    if not selected:
        print("目标已全部有轨迹文件，无需重复采集。")
        return

    # [Height expansion] 沿用原task_3和磁盘配置：测量噪声仍由原config控制。
    # 继续关闭力矩脉冲、连续保持1秒、每条最多40000步；不修改初始朝向。
    config_path = task_script.parent / "configs" / "robotnik_arm.json"
    fixed_paths = [task_script, task_script.with_name("robotnik_arm_control.py"), config_path]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in fixed_paths}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir or ROBOENV_DIR / "expert_collection_logs" / datetime.now().strftime("extend_z_%Y%m%d_%H%M%S_%f")
    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "logs").mkdir()
    save_json(log_dir / "collection_config.json", {
        "data_dir": str(data_dir), "grid": {"x": [-.6, .6, .1], "y": [.9, 1.4, .1], "z": [.1, 1.3, .1]},
        "total_grid_targets": len(targets), "existing_files": file_count, "skipped_existing_targets": skipped,
        "missing_targets_before_run": len(pending), "selected_targets": selected,
        "sensor_noise": "unchanged_config", "torque_noise_std_nm": 0., "hold_seconds": 1.,
        "max_steps": 40000, "robot_config_snapshot": config, "source_sha256": hashes,
        "existing_episode_policy": "skip any existing 23D target; preserve files regardless of hold-success status"})
    env = os.environ.copy()
    env.update(ROBOENV_HEADLESS="1", ROBOENV_MAX_STEPS="40000", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    fields = ["index", "target_x", "target_y", "target_z", "status", "held_success", "min_error_m",
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
                       status="process_error", held_success=False, log_file=str(log), error="")
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
                "episode_failures": sum(r["status"] == "episode_failure" for r in rows),
                "process_errors": sum(r["status"] == "process_error" for r in rows),
                "missing_targets_before_run": len(pending), "missing_after_saved_episodes": len(pending)-saved_count,
                "data_dir": str(data_dir), "log_dir": str(log_dir),
                "partial_run": len(selected) < len(pending)})
            print(f"  {row['status']} | min={row['min_error_m']} m | final={row['final_error_m']} m", flush=True)
    print(f"本次采集结束。轨迹：{data_dir}\n汇总：{log_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
