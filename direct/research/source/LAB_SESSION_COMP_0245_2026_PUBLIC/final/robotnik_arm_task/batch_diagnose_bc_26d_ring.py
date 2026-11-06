"""[Hybrid diagnostic change] 固定10个失败目标，分别替换底盘／机械臂分支。

默认两种模式，各10个回合；无噪声、20s、任意时刻1cm。
这里是有意挑选的失败案例诊断，不是140目标的泛化成功率。
不训练、不重采、不修改原模型、配置或原评估脚本。
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
from robotnik_bodyframe_features import BODYFRAME_FEATURE_VERSION


# [Hybrid diagnostic change] 前5个未完成导航切换，后5个已切换但最终失败。
# 来自已经完成的bodyframe验证批次，固定坐标便于逐点比较。
CASES = [
    ([0.2, 0.0, 0.2], "baseline_navigation_not_switched"),
    ([0.0, 0.6, 0.2], "baseline_navigation_not_switched"),
    ([-0.866025, -0.5, 0.2], "baseline_navigation_not_switched"),
    ([0.647214, -0.470228, 0.75], "baseline_navigation_not_switched"),
    ([-0.424264, -0.424264, 1.3], "baseline_navigation_not_switched"),
    ([0.866025, 0.5, 0.35], "baseline_switched_then_failed"),
    ([1.732051, -1.0, 0.5], "baseline_switched_then_failed"),
    ([-0.535757, -1.293431, 0.75], "baseline_switched_then_failed"),
    ([0.556231, -1.711902, 1.15], "baseline_switched_then_failed"),
    ([0.2, 0.34641, 1.3], "baseline_switched_then_failed"),
]
MODES = ["expert_base_model_arm", "model_base_expert_arm", "model", "expert"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def target_key(target):
    return tuple(round(float(v), 6) for v in target)


def main():
    project = Path(__file__).resolve().parent
    roboenv = project.parents[2] / "RoboEnv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=roboenv /
                        "bc_training_26d_noise_ring_bodyframe/run_20261102_184221_743102")
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--sensor-noise", choices=["off", "config"], default="off")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of selected TARGETS; default 10")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES[:2],
                        help="Default: both hybrids. model/expert are optional branch-source controls.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--wall-timeout", type=float, default=300.0)
    args = parser.parse_args()
    if (not math.isfinite(args.duration) or args.duration <= 0
            or not math.isfinite(args.wall_timeout) or args.wall_timeout <= 0):
        parser.error("duration and wall-timeout must be finite and positive")
    if args.limit is not None and not 1 <= args.limit <= len(CASES):
        parser.error("limit must be between 1 and 10")
    if len(set(args.modes)) != len(args.modes):
        parser.error("duplicate modes")
    run_dir = args.run_dir.resolve()
    baseline = (args.baseline_dir or run_dir / "batch_tests/val_20261102_230205_714031").resolve()
    script = project / "test_bc_26d_ring_hybrid.py"
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("input_feature_version") != BODYFRAME_FEATURE_VERSION:
        raise ValueError("This diagnostic requires the body-yaw-frame model")
    manifest = json.loads((run_dir / "data_split.json").read_text(encoding="utf-8"))
    val_targets = {target_key(t) for t in manifest["targets"]["val"]}
    with (baseline / "results.csv").open(encoding="utf-8-sig", newline="") as f:
        baseline_rows = {target_key([r["target_" + c] for c in "xyz"]): r for r in csv.DictReader(f)}
    baseline_config = json.loads((baseline / "evaluation_config.json").read_text(encoding="utf-8"))
    if Path(baseline_config["run_dir"]).resolve() != run_dir:
        raise ValueError("Baseline result is from a different model directory")
    cases = CASES[:args.limit] if args.limit is not None else CASES
    for target, group in cases:
        key = target_key(target)
        if key not in val_targets or key not in baseline_rows:
            raise ValueError(f"Target not found in validation baseline: {target}")
        row = baseline_rows[key]
        if row["status"] != "episode_failure" or row["success"] != "False":
            raise ValueError(f"Expected a completed failure in baseline: {target}")
        switched = np.isfinite(float(row["phase_switch_time_s"]))
        if switched != (group == "baseline_switched_then_failed"):
            raise ValueError(f"Baseline failure category changed: {target}")
    robot_config = roboenv / "configs/robotnik_arm.json"
    robot_snapshot = json.loads(robot_config.read_text(encoding="utf-8"))
    same_conditions = (args.duration == baseline_config["duration_limit_seconds"]
                       and args.sensor_noise == baseline_config["sensor_noise_mode"]
                       and robot_snapshot == baseline_config["robot_config_snapshot"])
    # [Hybrid diagnostic change] 原模型基线只作同目标参照，不与混合结果混为一类。
    fixed = [Path(__file__).resolve(), script, project / "robotnik_arm_control.py",
             project / "robotnik_bodyframe_features.py", run_dir / "best_model.pt",
             run_dir / "normalization.npz", config_path, run_dir / "data_split.json",
             robot_config, baseline / "results.csv", baseline / "evaluation_config.json"]
    for file in fixed:
        if not file.is_file():
            raise FileNotFoundError(file)
    for name in ["best_model.pt", "normalization.npz", "config.json", "data_split.json"]:
        expected = baseline_config["file_sha256"].get(str(run_dir / name))
        if expected is None or sha256(run_dir / name) != expected:
            raise ValueError(f"Model/baseline files differ: {name}")
    fingerprints = {str(f): sha256(f) for f in fixed}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output_dir or run_dir / "hybrid_diagnostics" / f"batch_{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    for mode in args.modes:
        (output / "rollouts" / mode).mkdir(parents=True)
        (output / "logs" / mode).mkdir(parents=True)
    save_json(output / "evaluation_config.json", {
        "diagnostic_only": True, "selection": "10_fixed_validation_failures_not_a_generalization_sample",
        "run_dir": str(run_dir), "baseline_dir": str(baseline), "modes": args.modes,
        "cases": [{"target": t, "baseline_group": g,
                   "seed": int(baseline_rows[target_key(t)]["seed"])} for t, g in cases],
        "duration_limit_seconds": args.duration, "sensor_noise_mode": args.sensor_noise,
        "baseline_conditions_match": same_conditions,
        "reach_tolerance_m": .01, "min_success_time_seconds": 0., "required_hold_seconds": 0.,
        "action_limits": [3.] * 4 + [30.] * 7, "input_feature_version": BODYFRAME_FEATURE_VERSION,
        "expert_reference": "updated_every_step_on_current_observed_state_not_old_trajectory_actions",
        "robot_config_snapshot": robot_snapshot, "file_sha256": fingerprints,
    })
    rows = []
    planned = len(cases) * len(args.modes)

    def update(state):
        by_mode = {}
        for mode in args.modes:
            group = [r for r in rows if r["control_mode"] == mode]
            wins = sum(r["success"] for r in group)
            by_mode[mode] = {"planned_episodes": len(cases), "processed_episodes": len(group),
                             "successes": wins, "process_errors": sum(r["status"] == "process_error" for r in group),
                             "diagnostic_success_percent_of_planned": 100*wins/len(cases)}
        save_json(output / "summary.json", {
            "state": state, "diagnostic_only": True, "selected_targets": len(cases),
            "planned_episodes": planned, "processed_episodes": len(rows),
            "process_errors": sum(r["status"] == "process_error" for r in rows),
            "by_mode": by_mode, "baseline_conditions_match": same_conditions,
            "baseline_model_successes_on_selected_targets": 0, "baseline_dir": str(baseline),
            "interpretation_note": "Selected failure-case interventions only; never report as pure-model/generalization success rate.",
            "output_dir": str(output),
        })
        # [Hybrid diagnostic change] 同目标两种模式并排保存，避免只比总成功数。
        with (output / "paired_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
            fields = ["target_x", "target_y", "target_z", "baseline_group", "baseline_final_error_m"]
            for mode in args.modes:
                fields += [mode + "_success", mode + "_final_error_m", mode + "_status"]
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            for index, (target, group) in enumerate(cases, 1):
                paired = {"target_x": target[0], "target_y": target[1], "target_z": target[2],
                          "baseline_group": group, "baseline_final_error_m": baseline_rows[target_key(target)]["final_error_m"]}
                for r in rows:
                    if r["target_index"] == index:
                        mode = r["control_mode"]
                        paired.update({mode + "_success": r["success"], mode + "_final_error_m": r["final_error_m"],
                                       mode + "_status": r["status"]})
                writer.writerow(paired)
        return by_mode

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    fields = ["target_index", "target_x", "target_y", "target_z", "baseline_group", "control_mode",
              "success", "status", "termination_reason", "final_error_m", "min_error_m", "end_time_s",
              "phase_switch_time_s", "clipped_steps", "seed", "rollout_file", "log_file", "error"]
    print(f"Diagnostic only: {len(cases)} selected failures x {len(args.modes)} modes = {planned} episodes", flush=True)
    print(f"Results: {output}", flush=True)
    if not same_conditions:
        print("Baseline conditions differ (e.g. shortened smoke run); do not interpret as matched baseline comparison.", flush=True)
    update("running")
    with (output / "results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); f.flush()
        try:
            for index, (target, group) in enumerate(cases, 1):
                seed = int(baseline_rows[target_key(target)]["seed"])
                for mode in args.modes:
                    if any(sha256(Path(file)) != expected for file, expected in fingerprints.items()):
                        raise RuntimeError("Diagnostic dependencies changed during run")
                    rollout = output / "rollouts" / mode / f"episode_{index:02d}.npz"
                    log = output / "logs" / mode / f"episode_{index:02d}.log"
                    row = dict.fromkeys(fields, "")
                    row.update({"target_index": index, "target_x": target[0], "target_y": target[1], "target_z": target[2],
                                "baseline_group": group, "control_mode": mode, "success": False, "status": "process_error",
                                "seed": seed, "rollout_file": str(rollout), "log_file": str(log)})
                    command = [sys.executable, "-B", str(script), "--run-dir", str(run_dir), "--target", *map(str, target),
                               "--control-mode", mode, "--headless", "--duration", str(args.duration),
                               "--sensor-noise", args.sensor_noise, "--seed", str(seed), "--result-file", str(rollout)]
                    print(f"[{len(rows)+1}/{planned}] target={target} | {mode}", flush=True)
                    try:
                        with log.open("w", encoding="utf-8") as lf:
                            process = subprocess.run(command, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                                     timeout=args.wall_timeout, check=False)
                        if process.returncode:
                            raise RuntimeError(f"Exit code {process.returncode}; see {log}")
                        with np.load(rollout, allow_pickle=False) as d:
                            if (not bool(d["diagnostic_only"]) or str(d["control_mode"]) != mode
                                    or str(d["input_feature_version"]) != BODYFRAME_FEATURE_VERSION):
                                raise ValueError("Wrong diagnostic mode or input schema")
                            np.testing.assert_allclose(d["target"], target, rtol=0, atol=1e-6)
                            if not np.isclose(float(d["duration_limit_seconds"]), args.duration):
                                raise ValueError("Wrong duration")
                            if args.sensor_noise == "off" and d["sensor_noise_enabled"].any():
                                raise ValueError("Measurement noise was not disabled")
                            row.update({"success": bool(d["success"]), "status": "success" if bool(d["success"]) else "episode_failure",
                                        "termination_reason": str(d["termination_reason"]), "final_error_m": float(d["final_error"]),
                                        "min_error_m": float(d["ee_errors"].min()), "end_time_s": float(d["timestamps"][-1]),
                                        "phase_switch_time_s": float(d["phase_switch_time"]), "clipped_steps": int(d["clipped_steps"])})
                    except subprocess.TimeoutExpired:
                        row.update({"termination_reason": "process_wall_timeout", "error": "Wall-clock timeout"})
                    except Exception as exc:
                        row.update({"termination_reason": "process_error", "error": f"{type(exc).__name__}: {exc}"})
                    rows.append(row); writer.writerow(row); f.flush(); update("running")
                    print(f"  {row['status']} | final error={row['final_error_m']} m", flush=True)
        except BaseException:
            update("interrupted_or_aborted")
            raise
    by_mode = update("completed")
    for mode, s in by_mode.items():
        print(f"{mode}: {s['successes']}/{s['planned_episodes']} selected cases; process errors={s['process_errors']}")
    print("Diagnostic cases only; this is NOT pure-model/generalization success.")
    print(f"Summary: {output / 'summary.json'}")
    print(f"Paired comparison: {output / 'paired_results.csv'}")


if __name__ == "__main__":
    main()
