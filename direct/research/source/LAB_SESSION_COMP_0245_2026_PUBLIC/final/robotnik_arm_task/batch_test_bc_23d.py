"""Evaluate a fixed target split through independent closed-loop simulations.

Defaults: formal test targets, 20 s simulation limit, measurement noise OFF.
Every subprocess/error remains in the planned-target success-rate denominator.
No training, expert trajectory playback, or forced wheel stopping is performed.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    # Atomic replacement leaves a readable summary if the run is interrupted.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def summarize(rows, planned):
    successes = sum(bool(row["success"]) for row in rows)
    process_errors = sum(row["status"] == "process_error" for row in rows)
    finite_errors = [row["final_error_m"] for row in rows if isinstance(row["final_error_m"], (int, float))
                     and math.isfinite(row["final_error_m"])]
    times = [row["end_time_s"] for row in rows if row["success"]]
    return {
        "planned_targets": planned,
        "processed_targets": len(rows),
        "successes": successes,
        "unsuccessful_completed_episodes": sum(row["status"] == "episode_failure" for row in rows),
        "process_errors": process_errors,
        "unprocessed_targets": planned - len(rows),
        "success_rate_percent": 100.0 * successes / planned,
        "mean_final_error_m_finite_results": float(np.mean(finite_errors)) if finite_errors else None,
        "median_final_error_m_finite_results": float(np.median(finite_errors)) if finite_errors else None,
        "finite_error_results": len(finite_errors),
        "mean_success_time_s": float(np.mean(times)) if times else None,
        "all_planned_targets_processed": len(rows) == planned,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    roboenv = Path(__file__).resolve().parents[3] / "RoboEnv"
    parser.add_argument("--run-dir", type=Path, default=(
        roboenv / "bc_training_23d_noise" / "run_20261029_024749_959833"))
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Simulation seconds per target (default: 20)")
    parser.add_argument("--sensor-noise", choices=["off", "config"], default="off",
                        help="off overrides sensor noise at runtime; config uses RoboEnv config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional partial smoke run; omitted = full split")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--wall-timeout", type=float, default=300.0,
                        help="Process timeout in wall-clock seconds; distinct from simulation horizon")
    args = parser.parse_args()
    if (not math.isfinite(args.duration) or args.duration <= 0 or args.seed < 0
            or not math.isfinite(args.wall_timeout) or args.wall_timeout <= 0):
        parser.error("duration/timeout must be finite and positive; seed must be nonnegative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    run_dir = args.run_dir.resolve()
    script = Path(__file__).resolve().with_name("test_bc_23d.py")
    checkpoint = run_dir / "best_model.pt"
    normalization = run_dir / "normalization.npz"
    manifest_path = run_dir / "data_split.json"
    robot_config_path = roboenv / "configs" / "robotnik_arm.json"
    for path in [script, checkpoint, normalization, manifest_path, robot_config_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("input_dim") != 23 or manifest.get("output_dim") != 11:
        raise ValueError("This evaluator requires a 23D-input, 11D-output model")
    available_targets = manifest["targets"][args.split]
    targets = available_targets[:args.limit] if args.limit is not None else available_targets
    values = np.asarray(targets, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid target list")
    if len({tuple(t) for t in targets}) != len(targets):
        raise ValueError("Duplicate target coordinates in the split")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output_dir or run_dir / "batch_tests" / f"{args.split}_{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "rollouts").mkdir()
    (output / "logs").mkdir()
    fixed_files = [script, checkpoint, normalization, manifest_path, robot_config_path]
    fingerprints = {str(path): sha256(path) for path in fixed_files}
    conditions = {
        "run_dir": str(run_dir), "split": args.split,
        "available_split_targets": len(available_targets), "selected_targets": len(targets),
        "partial_subset": len(targets) != len(available_targets),
        "duration_limit_seconds": args.duration,
        "sensor_noise_mode": args.sensor_noise,
        # [Anytime success change] Match single-episode success without the old 8s gate.
        "reach_tolerance_m": 0.01, "min_success_time_seconds": 0.0,
        "required_hold_seconds": 0.0,
        "success_definition": "first observed EE distance < 0.01 m at any time within horizon; no minimum time or hold",
        "phase_source": "expert_navigation_rules_commands_discarded",
        "forced_parking": False, "action_limits": [3.0] * 4 + [30.0] * 7,
        "base_seed": args.seed, "target_seeds": [args.seed + i for i in range(len(targets))],
        "targets": targets, "file_sha256": fingerprints,
        "robot_config_snapshot": json.loads(robot_config_path.read_text(encoding="utf-8")),
        "noise_override_note": "off disables measurement noise in memory; snapshot itself is not modified",
    }
    save_json(output / "evaluation_config.json", conditions)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    rows = []
    fields = ["index", "target_x", "target_y", "target_z", "success", "status", "termination_reason",
              "final_error_m", "min_error_m", "end_time_s", "phase_switch_time_s", "action_steps",
              "clipped_steps", "clipped_step_fraction", "seed", "return_code", "rollout_file", "log_file", "error"]

    def update_summary(state):
        summary = summarize(rows, len(targets))
        summary.update({"state": state, "split": args.split, "partial_subset": conditions["partial_subset"],
                        "available_split_targets": len(available_targets), "output_dir": str(output)})
        save_json(output / "summary.json", summary)
        return summary

    print(f"Split={args.split}; targets={len(targets)}/{len(available_targets)}; "
          f"horizon={args.duration}s; sensor noise={args.sensor_noise}", flush=True)
    print(f"Results: {output}", flush=True)
    update_summary("running")
    with (output / "results.csv").open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        csv_file.flush()
        try:
            for offset, target in enumerate(targets):
                # Stop rather than silently evaluate different checkpoints/configs mid-batch.
                if any(sha256(path) != fingerprints[str(path)] for path in fixed_files):
                    raise RuntimeError("Model, test script or config changed during evaluation; start a new batch")
                index = offset + 1
                seed = args.seed + offset
                rollout = output / "rollouts" / f"episode_{index:03d}.npz"
                log = output / "logs" / f"episode_{index:03d}.log"
                row = dict.fromkeys(fields, "")
                row.update({"index": index, "target_x": target[0], "target_y": target[1], "target_z": target[2],
                            "success": False, "status": "process_error", "seed": seed,
                            "rollout_file": str(rollout), "log_file": str(log)})
                command = [sys.executable, "-B", str(script), "--run-dir", str(run_dir),
                           "--target", *map(str, target), "--headless", "--duration", str(args.duration),
                           "--sensor-noise", args.sensor_noise, "--seed", str(seed), "--result-file", str(rollout)]
                print(f"[{index}/{len(targets)}] target={target}", flush=True)
                try:
                    with log.open("w", encoding="utf-8") as log_file:
                        process = subprocess.run(command, env=env, stdout=log_file, stderr=subprocess.STDOUT,
                                                 timeout=args.wall_timeout, check=False)
                    row["return_code"] = process.returncode
                    if process.returncode:
                        raise RuntimeError(f"Child process exited with code {process.returncode}; see log")
                    with np.load(rollout, allow_pickle=False) as result:
                        if not np.allclose(result["target"], target, rtol=0, atol=1e-6):
                            raise ValueError("Rollout target differs from requested target")
                        if args.sensor_noise == "off" and result["sensor_noise_enabled"].any():
                            raise ValueError("Measurement noise was not disabled")
                        if not np.isclose(float(result["duration_limit_seconds"]), args.duration):
                            raise ValueError("Wrong simulation horizon")
                        action_steps = len(result["actions"])
                        success = bool(result["success"])
                        row.update({"success": success, "status": "success" if success else "episode_failure",
                                    "termination_reason": str(result["termination_reason"]),
                                    "final_error_m": float(result["final_error"]),
                                    "min_error_m": float(np.min(result["ee_errors"])),
                                    "end_time_s": float(result["timestamps"][-1]),
                                    "phase_switch_time_s": float(result["phase_switch_time"]),
                                    "action_steps": action_steps, "clipped_steps": int(result["clipped_steps"]),
                                    "clipped_step_fraction": int(result["clipped_steps"]) / action_steps if action_steps else 0.0})
                except subprocess.TimeoutExpired:
                    row.update({"termination_reason": "process_wall_timeout", "error": f"Wall-clock timeout: {args.wall_timeout}s"})
                except Exception as exc:
                    row.update({"success": False, "status": "process_error", "termination_reason": "process_error",
                                "error": f"{type(exc).__name__}: {exc}"})
                rows.append(row)
                writer.writerow(row)
                csv_file.flush()
                summary = update_summary("running")
                print(f"  {row['status']} | error={row['final_error_m']} m | "
                      f"successes={summary['successes']}/{index}", flush=True)
        except BaseException:
            update_summary("interrupted_or_aborted")
            raise
    summary = update_summary("completed")
    print(f"\nSuccess rate: {summary['successes']}/{len(targets)} = {summary['success_rate_percent']:.2f}%")
    print(f"Process errors: {summary['process_errors']} (included in denominator)")
    if conditions["partial_subset"]:
        print("Partial subset only; this is NOT the full test-set success rate.")
    print(f"Summary: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
