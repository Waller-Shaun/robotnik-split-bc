"""Probe expert completion versus target Z without changing existing datasets.

[Z probe change] Fixed X/Y, configurable Z grid, isolated runs and per-target logs.
Uses the existing task_3 controller. Measurement noise is disabled in memory by
default; task/config files and the original collector are never modified.
"""
import argparse
import csv
from datetime import datetime
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


# [Z probe change] 独立子进程只覆盖噪声开关，不修改磁盘配置或专家控制规律。
WORKER_CODE = r'''
import sys, runpy
from pathlib import Path
import numpy as np
task = Path(sys.argv[1]).resolve()
mode, seed = sys.argv[2], int(sys.argv[3])
forwarded = sys.argv[4:]
sys.path.insert(0, str(task.parent))
np.random.seed(seed)
from simulation_and_control import pb
original = pb.SimInterface
def make_sim(*args, **kwargs):
    sim = original(*args, **kwargs)
    if mode == "off":
        for robot in sim.bot:
            robot.noise_flag = False
    print("Probe sensor noise:", [bool(robot.noise_flag) for robot in sim.bot], flush=True)
    return sim
pb.SimInterface = make_sim
sys.argv = [str(task), *forwarded]
runpy.run_path(str(task), run_name="__main__")
'''


def z_grid(z_min, z_max, z_step):
    # [Z probe change] Decimal生成网格，避免0.1浮点累积造成漏端点。
    low, high, step = map(lambda value: Decimal(str(value)), (z_min, z_max, z_step))
    if step <= 0 or high < low:
        raise ValueError("Require z_step > 0 and z_max >= z_min")
    quotient = (high - low) / step
    if quotient != quotient.to_integral_value():
        raise ValueError("z_max - z_min must be an integer multiple of z_step")
    return [float(low + i * step) for i in range(int(quotient) + 1)]


def finite(value):
    number = float(value)
    return number if math.isfinite(number) else None


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def summarize(rows, heights, args):
    held = [r for r in rows if r["held_success"]]
    first_hit = [r for r in rows if r["reached_once"]]
    # [Z probe change] 只报告已测离散点，不假定成功区间连续或所有x/y具有同一边界。
    transitions = []
    for left, right in zip(rows, rows[1:]):
        if left["status"] != "process_error" and right["status"] != "process_error" and left["held_success"] != right["held_success"]:
            transitions.append({"z_left_m": left["target_z"], "z_right_m": right["target_z"],
                                "held_success_left": left["held_success"], "held_success_right": right["held_success"]})
    return {
        "state": "completed" if len(rows) == len(heights) else "running",
        "fixed_target_xy_m": args.xy,
        "planned_z_values_m": heights,
        "processed_targets": len(rows), "planned_targets": len(heights),
        "held_successes": len(held), "reached_once_count": len(first_hit),
        "process_errors": sum(r["status"] == "process_error" for r in rows),
        "successful_tested_z_values_m": [r["target_z"] for r in held],
        "lowest_successful_tested_z_m": min((r["target_z"] for r in held), default=None),
        "highest_successful_tested_z_m": max((r["target_z"] for r in held), default=None),
        "grid_transitions_for_refinement": transitions,
        "sensor_noise_mode": args.sensor_noise,
        "duration_limit_s": args.duration,
        "hold_seconds_required": args.hold_seconds,
        "reach_tolerance_m": .01,
        "scope": "Empirical completion of task_3 at fixed target X/Y and configured initial pose; not a global workspace bound",
        "note": "The expert changes its base standoff with target Z. Successful scan endpoints do not establish a physical limit.",
    }


def make_plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid = [r for r in rows if r["status"] != "process_error"]
    if not valid:
        return
    fig, (ax, time_ax) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    z = [r["target_z"] for r in valid]
    best = [max(r["min_error_m"] * 100, 1e-4) for r in valid]
    final = [max(r["final_error_m"] * 100, 1e-4) for r in valid]
    ax.plot(z, best, "o-", label="Minimum EE error", color="#287cc0")
    ax.plot(z, final, "s--", label="Final EE error", color="#8d65aa")
    ax.axhline(1, color="#cd454d", linestyle=":", label="1 cm tolerance")
    ax.set_yscale("log")
    ax.set(ylabel="EE error (cm, log scale)", title="Expert target-height probe")
    ax.grid(alpha=.25);ax.legend()
    colors = ["#208f79" if r["held_success"] else "#dc8a24" if r["reached_once"] else "#d0444c" for r in valid]
    time_ax.scatter(z, [r["end_time_s"] for r in valid], c=colors, s=65)
    time_ax.set(xlabel="Target Z (m)", ylabel="Episode end time (s)")
    time_ax.grid(alpha=.25)
    fig.text(.5, .015, "Green: reached and held | Orange: reached once, not held | Red: no arrival", ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(output / "z_probe.png", dpi=180)
    plt.close(fig)


def write_report(rows, summary, output):
    lines = ["# 专家目标高度探索", "", f"固定目标x、y：{summary['fixed_target_xy_m']}m。",
             "保持配置中的初始位置、朝向和机械臂姿态；本次只改变目标z。底盘实际停车位置会随专家的高度相关standoff规则变化。", "",
             "这里测量当前专家策略在指定初始化和目标平面位置下能否完成任务，不是机械臂的全局几何可达范围。失败也可能受时间上限、专家规划或控制参数影响。", "",
             f"噪声模式：{summary['sensor_noise_mode']}；力矩脉冲关闭；误差阈值1cm；连续保持{summary['hold_seconds_required']}秒。",
             "沿用task_3的8秒最短运行时间；达到并保持条件满足后才结束。首次到达也单独记录。", "",
             "| z m | 曾到达1cm内 | 完成连续保持 | 最小误差 cm | 最终误差 cm | 首次到达 s | 结束 s | 终止原因 |",
             "|---:|---|---|---:|---:|---:|---:|---|"]
    def number(value, scale=1):
        return "—" if value is None else f"{value*scale:.4f}"
    for row in rows:
        lines.append(f"| {row['target_z']:.3f} | {'是' if row['reached_once'] else '否'} | {'是' if row['held_success'] else '否'} | {number(row['min_error_m'],100)} | {number(row['final_error_m'],100)} | {number(row['first_reach_time_s'])} | {number(row['end_time_s'])} | {row['termination_reason']} |")
    lines += ["", f"已完成{summary['processed_targets']}/{summary['planned_targets']}个目标，其中{summary['held_successes']}个满足连续保持，{summary['reached_once_count']}个曾进入1cm范围。",
              f"成功的已测z值：{summary['successful_tested_z_values_m']}。", "",
              "若最低或最高扫描点仍成功，应继续向外扫描；若相邻点发生成功／失败变化，可缩小步长复测。不能只根据成功点最小值和最大值断言它们之间全都可达。", "",
              "![高度探索结果](z_probe.png)", "",
              "本目录是探索数据，未自动并入任何训练、验证或测试划分。后续扩充数据时应固定目标分组，避免同一目标的重复轨迹跨集合。"]
    (output / "探索结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xy", type=float, nargs=2, default=[0., 1.1], metavar=("X", "Y"))
    parser.add_argument("--z-min", type=float, default=.1)
    parser.add_argument("--z-max", type=float, default=1.3)
    parser.add_argument("--z-step", type=float, default=.1)
    parser.add_argument("--duration", type=float, default=40., help="Simulation seconds per target")
    parser.add_argument("--hold-seconds", type=float, default=1.)
    parser.add_argument("--sensor-noise", choices=["off", "config"], default="off")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wall-timeout", type=float, default=300.)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Show targets without starting simulations")
    args = parser.parse_args()
    numbers = [*args.xy, args.z_min, args.z_max, args.z_step, args.duration, args.hold_seconds, args.wall_timeout]
    if not all(math.isfinite(value) for value in numbers) or args.seed < 0:
        parser.error("Numeric values must be finite; seed must be nonnegative")
    if min(args.duration, args.hold_seconds, args.wall_timeout) <= 0:
        parser.error("duration, hold-seconds and wall-timeout must be positive")
    try:
        heights = z_grid(args.z_min, args.z_max, args.z_step)
    except ValueError as error:
        parser.error(str(error))
    task = Path(__file__).resolve().with_name("robotnik_arm_task_3.py")
    config_path = task.parent / "configs/robotnik_arm.json"
    control_path = task.with_name("robotnik_arm_control.py")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    step = float(config["sim"]["time_step"])
    max_steps = int(round(args.duration / step))
    if max_steps < 1 or not math.isclose(max_steps*step, args.duration, rel_tol=0, abs_tol=1e-8):
        parser.error("duration must be a positive multiple of the simulation time step")
    print(f"Z probe: xy={args.xy}; {len(heights)} heights={heights}", flush=True)
    if args.dry_run:
        return
    roboenv = task.parents[3] / "RoboEnv"
    output = args.output_dir or roboenv / "expert_z_probe" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir();(output / "logs").mkdir()
    sources = [task, config_path, control_path]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    atomic_json(output / "probe_config.json", {"xy": args.xy, "heights_m": heights,
        "sensor_noise_mode": args.sensor_noise, "torque_noise_std_nm": 0., "seed": args.seed,
        "max_steps": max_steps, "time_step_s": step, "duration_s": args.duration,
        "hold_seconds": args.hold_seconds, "expert_minimum_episode_seconds": 8.,
        "task_script": str(task), "source_sha256": hashes, "robot_config_snapshot": config,
        "purpose": "workspace_height_exploration_not_training_split"})
    env = os.environ.copy()
    env.update(ROBOENV_HEADLESS="1", ROBOENV_MAX_STEPS=str(max_steps), PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    rows = []
    fields = ["index", "target_x", "target_y", "target_z", "status", "reached_once", "held_success",
              "min_error_m", "final_error_m", "first_reach_time_s", "end_time_s", "navigation_done_time_s",
              "continuous_hold_s", "termination_reason", "rollout_file", "log_file", "error"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields);writer.writeheader();stream.flush()
        for index, z in enumerate(heights, 1):
            for path, digest in hashes.items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                    raise RuntimeError(f"Source changed during sweep: {path}")
            case_dir = output / "episodes" / f"case_{index:03d}"
            case_dir.mkdir()
            log = output / "logs" / f"case_{index:03d}.log"
            target = [*args.xy, z]
            print(f"[{index}/{len(heights)}] Target: {target}", flush=True)
            command = [sys.executable, "-B", "-u", "-c", WORKER_CODE, str(task), args.sensor_noise, str(args.seed),
                       "--target", *map(str, target), "--noise-std", "0", "--hold-seconds", str(args.hold_seconds),
                       "--seed", str(args.seed), "--output-dir", str(case_dir)]
            row = dict.fromkeys(fields)
            row.update(index=index, target_x=args.xy[0], target_y=args.xy[1], target_z=z,
                       status="process_error", reached_once=False, held_success=False,
                       termination_reason="process_error", log_file=str(log), error="")
            try:
                with log.open("w", encoding="utf-8") as logfile:
                    result = subprocess.run(command, env=env, cwd=task.parent, stdout=logfile, stderr=subprocess.STDOUT,
                                            timeout=args.wall_timeout, check=False)
                if result.returncode != 0:
                    raise RuntimeError(f"Expert process exited with code {result.returncode}; see log")
                files = list(case_dir.glob("expert_episode_*.npz"))
                if len(files) != 1:
                    raise RuntimeError(f"Expected one episode, found {len(files)}")
                with np.load(files[0], allow_pickle=False) as data:
                    if not np.allclose(data["target"], target, atol=1e-8, rtol=0):
                        raise RuntimeError("Saved target does not match requested target")
                    minimum = finite(data["best_error"].item())
                    final = finite(data["final_error"].item())
                    if minimum is None or final is None:
                        raise RuntimeError("Non-finite expert error")
                    held = bool(data["success"].item())
                    row.update(status="held_success" if held else "episode_failure", reached_once=minimum < .01,
                               held_success=held, min_error_m=minimum, final_error_m=final,
                               first_reach_time_s=finite(data["first_reach_time"].item()),
                               end_time_s=finite(data["final_timestamp"].item()),
                               navigation_done_time_s=finite(data["navigation_done_time"].item()),
                               continuous_hold_s=finite(data["continuous_hold_seconds"].item()),
                               termination_reason=str(data["termination_reason"].item()), rollout_file=str(files[0]))
            except (RuntimeError, ValueError, KeyError, OSError, subprocess.TimeoutExpired) as error:
                row["error"] = str(error)
            rows.append(row);writer.writerow(row);stream.flush()
            summary = summarize(rows, heights, args)
            atomic_json(output / "summary.json", summary)
            print(f"  {row['status']} | minimum={row['min_error_m']} m | final={row['final_error_m']} m", flush=True)
    make_plot(rows, output)
    write_report(rows, summary, output)
    print(f"Held successes: {summary['held_successes']}/{len(heights)}; reached once: {summary['reached_once_count']}")
    print("Successful tested z:", summary["successful_tested_z_values_m"])
    print("Results:", output)


if __name__ == "__main__":
    main()
