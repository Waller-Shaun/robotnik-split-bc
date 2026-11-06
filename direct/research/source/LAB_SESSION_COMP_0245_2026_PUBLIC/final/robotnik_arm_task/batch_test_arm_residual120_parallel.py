# [Residual120 change] 独立并行评估；不修改113残差版或120基础模型的原测试。
# [Residual change] 新副本，独立测试组合策略；保留144点val/156点test、种子、带噪观测与真实位置评分。
# [B2 change] 独立并行测试：多任务MLP，带噪输入，真实末端评分，绝不训练。
# [MLP noise change] 新副本：固定无GRU基线、默认配置噪声、真实位置评估、自适应并行。
# [Parallel change] 新副本：多目标并行；不修改串行程序、模型、种子或仿真控制。
# [B1 GRU change] 基于补采版batch副本，默认同一144点val；训练独立运行。
"""[Arm finetune change] 测试冻结底盘＋微调机械臂的组合；绝不启动训练。

Defaults: all 144 merged validation targets, 20 s limit, measurement noise from config; ground-truth EE scoring.
Every subprocess/error remains in the planned-target success-rate denominator.
No training, expert trajectory playback, or forced wheel stopping is performed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import ctypes
import threading
import time
import tempfile
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
from robotnik_bodyframe_features import BODYFRAME_FEATURE_VERSION
NAV_FEATURE_VERSION = None  # resolved by the short-lived preflight process


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
        "observed_reached_but_true_failed": sum(r.get("observed_reached_but_true_failed") is True for r in rows),
    }


def available_memory_bytes():
    """[Parallel change] Best-effort memory check; no extra dependency."""
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length",ctypes.c_ulong),("load",ctypes.c_ulong),
                        *[(n,ctypes.c_ulonglong) for n in ["total_physical","available_physical",
                            "total_page","available_page","total_virtual","available_virtual","extended"]]]
        status=MemoryStatus(); status.length=ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.available_physical)
    else:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])*1024
        except OSError:
            pass
    return None


def cpu_busy_fraction():
    """[MLP noise change] Windows全机CPU负载短采样；不依赖CIM/psutil。"""
    if os.name != "nt":
        try:
            return min(1.0, os.getloadavg()[0] / max(1, os.cpu_count() or 1))
        except (AttributeError, OSError):
            return 0.0
    def snapshot():
        idle, kernel, user = (ctypes.c_ulonglong() for _ in range(3))
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            return None
        return idle.value, kernel.value + user.value
    first = snapshot(); time.sleep(.25); second = snapshot()
    if first is None or second is None or second[1] <= first[1]:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (second[0]-first[0])/(second[1]-first[1])))


def resolve_finetuned_pair(roboenv, arm_run=None, nav_run=None):
    """[Parallel change] Verify original model binding, then release preflight RAM."""
    global NAV_FEATURE_VERSION
    with tempfile.TemporaryDirectory(prefix="robotnik_parallel_preflight_") as folder:
        result=Path(folder)/"pair.json"
        command=[sys.executable,"-B",str(Path(__file__).resolve()),"--resolve-pair",
                 "--roboenv",str(roboenv),"--result",str(result)]
        if arm_run is not None: command += ["--arm-run-dir",str(arm_run)]
        if nav_run is not None: command += ["--nav-run-dir",str(nav_run)]
        check=subprocess.run(command,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=120)
        if check.returncode:
            raise RuntimeError("Model preflight failed: "+(check.stderr+check.stdout)[-3000:])
        data=json.loads(result.read_text(encoding="utf-8"))
    NAV_FEATURE_VERSION=data["navigation_feature_version"]
    return Path(data["arm_run"]),Path(data["nav_run"]),data["arm_config"],data["nav_config"]


def preflight_main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--roboenv",type=Path,required=True)
    parser.add_argument("--result",type=Path,required=True)
    parser.add_argument("--arm-run-dir",type=Path)
    parser.add_argument("--nav-run-dir",type=Path)
    args=parser.parse_args(sys.argv[2:])
    from robotnik_arm_residual120_policy import resolve_finetuned_pair as original_resolve
    from robotnik_navigation_policy import NAV_FEATURE_VERSION as nav_version
    arm,nav,ac,nc=original_resolve(args.roboenv,args.arm_run_dir,args.nav_run_dir)
    save_json(args.result,{"arm_run":str(arm),"nav_run":str(nav),"arm_config":ac,"nav_config":nc,
                           "navigation_feature_version":nav_version})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    roboenv = Path(__file__).resolve().parents[3] / "RoboEnv"
    parser.add_argument("--nav-run-dir", type=Path, default=None,
                        help="Optional check: must match the frozen navigation run of the arm finetune")
    parser.add_argument("--arm-run-dir", type=Path, default=None,
                        help="Arm finetune result; default latest completed non-smoke run")
    # [Low Z test change] 默认144个val；也可分别查看原116点与新增28点。
    parser.add_argument("--split", choices=["test", "val"], default="val")
    parser.add_argument("--subset", choices=["all", "base", "supplement"], default="all")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Simulation seconds per target (default: 20)")
    parser.add_argument("--sensor-noise", choices=["off", "config"], default="config",
                        help="off overrides sensor noise at runtime; config uses RoboEnv config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional partial smoke run; omitted = full split")
    parser.add_argument("--output-dir", type=Path, default=None)
    # [Parallel change] auto respects available RAM; each child is one CPU-thread simulation.
    parser.add_argument("--workers", default="auto", help="auto: at most4, CPU-load/RAM-aware; or a positive integer")
    parser.add_argument("--wall-timeout", type=float, default=300.0,
                        help="Process timeout in wall-clock seconds; distinct from simulation horizon")
    parser.add_argument("--plan-only", action="store_true",
                        help="Read CPU/RAM and verify model/config; do not start any simulation")
    args = parser.parse_args()
    if (not math.isfinite(args.duration) or args.duration <= 0 or args.seed < 0
            or not math.isfinite(args.wall_timeout) or args.wall_timeout <= 0):
        parser.error("duration/timeout must be finite and positive; seed must be nonnegative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    memory_available = available_memory_bytes()
    # [MLP noise change] 运行时测量CPU忙碌程度与剩余内存；每个仿真单线程。
    cpu_busy = cpu_busy_fraction()
    cpu_available = max(1, int((os.cpu_count() or 1) * (1.0 - cpu_busy)))
    memory_slots = max(1, int((memory_available - 1536*1024**2)//(800*1024**2))) if memory_available else 2
    if args.workers == "auto":
        # 至多4进程：16GB机器留1.5GiB，预估每个子进程800MiB，并给系统留CPU余量。
        worker_limit = min(4, max(1, cpu_available//2), memory_slots)
    else:
        try:
            worker_limit = int(args.workers)
        except ValueError:
            parser.error("--workers must be auto or a positive integer")
        if worker_limit < 1:
            parser.error("--workers must be positive")

    # [Arm finetune change] 以机械臂微调结果为主目录，底盘固定为训练时绑定的原模型。
    run_dir, nav_run, arm_config, model_config = resolve_finetuned_pair(
        roboenv, args.arm_run_dir, args.nav_run_dir)
    script = Path(__file__).resolve().with_name("test_arm_residual120.py")
    pair_module = Path(__file__).resolve().with_name("robotnik_arm_residual120_policy.py")
    supplemental_pair_module = Path(__file__).resolve().with_name("arm_lowz_supplement_experiment.py")
    parent_pair_module = Path(__file__).resolve().with_name("arm_finetune_experiment.py")
    # [Body frame test change] 冻结共用特征转换及阶段控制器的哈希，记录依赖。
    feature_module = Path(__file__).resolve().with_name("robotnik_bodyframe_features.py")
    navigation_module = Path(__file__).resolve().with_name("robotnik_navigation_policy.py")
    controller_module = Path(__file__).resolve().with_name("robotnik_arm_control.py")
    checkpoint = run_dir / "best_model.pt"
    normalization = run_dir / "normalization.npz"
    manifest_path = run_dir / "data_split.json"
    model_config_path = run_dir / "config.json"  # [26D test change]
    robot_config_path = roboenv / "configs" / "robotnik_arm.json"
    for path in [script, feature_module, navigation_module, controller_module, checkpoint, normalization, manifest_path, model_config_path, robot_config_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("input_dim") != 23 or manifest.get("output_dim") != 11:
        raise ValueError("Expected a SOURCE manifest describing 23D NPZ files and 11D actions")
    # [26D test change] 划分清单仍描述23维源文件；config描述派生的26维网络输入。
    if (model_config.get("input_dim") != 8 or model_config.get("output_dim") != 4
            or model_config.get("input_feature_version") != NAV_FEATURE_VERSION):
        raise ValueError("Expected the new 8D navigation model")
    if {k:len(v) for k,v in manifest["targets"].items()} != {"train":698,"val":144,"test":156}:
        raise ValueError("Wrong merged 998-target split")
    arm_run=run_dir
    nav_files=[nav_run/n for n in ["best_model.pt","normalization.npz","config.json","data_split.json"]]
    available_targets = manifest["targets"][args.split]
    # [Low Z test change] 按原始索引选点和种子，避免单测子集时换掉随机种子。
    base_count = manifest["base_split_counts"][args.split]
    indices = [i for i in range(len(available_targets))
               if args.subset=="all" or (i<base_count if args.subset=="base" else i>=base_count)]
    if args.limit is not None:
        indices = indices[:args.limit]
    targets = [available_targets[i] for i in indices]
    values = np.asarray(targets, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid target list")
    if len({tuple(t) for t in targets}) != len(targets):
        raise ValueError("Duplicate target coordinates in the split")

    workers = min(worker_limit, len(targets))

    # [MLP noise change] 仅规划时不创建结果目录、不启动仿真。
    robot_snapshot = json.loads(robot_config_path.read_text(encoding="utf-8"))
    if args.sensor_noise == "config" and not all(robot_snapshot["robot_pybullet"]["noise_flag"]):
        raise ValueError("Enable noise_flag in RoboEnv/configs/robotnik_arm.json first")
    if args.plan_only:
        print(json.dumps({"arm_run":str(arm_run),"navigation_run":str(nav_run),
            "targets":len(targets),"split":args.split,"sensor_noise":args.sensor_noise,
            "config":str(robot_config_path),"noise":robot_snapshot["robot_pybullet"]["robot_noise"],
            "logical_cpus":os.cpu_count(),"cpu_busy_fraction":cpu_busy,
            "available_memory_gib":memory_available/1024**3 if memory_available else None,
            "workers":workers,"simulations_started":False},indent=2))
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output_dir or run_dir / "batch_tests" / f"parallel_residual120_noise_truth_{args.split}_{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "rollouts").mkdir()
    (output / "logs").mkdir()
    fixed_files = [Path(__file__).resolve(), script, feature_module, navigation_module, controller_module,
                   checkpoint, normalization, manifest_path, model_config_path, robot_config_path,pair_module,parent_pair_module,supplemental_pair_module,*nav_files]
    # [Residual change] 修正模型内部还包含原基础MLP；运行期间冻结其加载模块和父模型文件。
    fixed_files += [Path(__file__).with_name("robotnik_arm_aux_policy.py")]
    frozen_parent = Path(arm_config["source_model_run_dir"])
    fixed_files += [frozen_parent/n for n in ["best_model.pt","normalization.npz","config.json","data_split.json"]]
    # [Residual120 change] 保留数据采集模型113与冻结基础120的双重来源，评估期间核验上下文代码。
    fixed_files += [Path(__file__).with_name("recovery_residual120_context.py")]
    collection_policy = Path(arm_config["recovery_collection_policy_run_dir"])
    fixed_files += [collection_policy/n for n in ["best_model.pt","normalization.npz","config.json","data_split.json"]]
    fingerprints = {str(path): sha256(path) for path in fixed_files}
    conditions = {
        "run_dir": str(run_dir), "split": args.split,
        "execution_mode":"parallel_independent_subprocesses",
        "requested_workers":args.workers,"workers":workers,
        "logical_cpu_count":os.cpu_count(),"cpu_busy_fraction_at_start":cpu_busy,
        "available_memory_bytes_at_start":memory_available,
        "worker_memory_budget_mib":800,"reserved_memory_mib":1536,
        "child_numeric_threads":1,"gpu_inference":False,
        "each_target_seed_independent":True,"per_target_wall_timeout_seconds":args.wall_timeout,
        # [26D test change] 记录实际输入结构，不误读源清单的input_dim=23。
        "input_dim": 26, "source_input_dim": 23,
        "input_feature_version": BODYFRAME_FEATURE_VERSION,
        "navigation_input_dim":8,"navigation_feature_version":NAV_FEATURE_VERSION,
        "arm_run_dir":str(arm_run),"navigation_run_dir":str(nav_run),
        "arm_parent_run_dir":arm_config["source_model_run_dir"],
        "policy_architecture":"frozen_learned_nav8_residual120_arm26",
        "arm_finetuned":True,
        # [MLP noise change] 不读取GRU历史；真实末端位置只用于评分。
        "evaluation_position_source":"pybullet_link_world_frame",
        "model_position_source":"forward_kinematics_from_sensor_observations",
        "true_state_used_for_model_input":False,
        "expert_references_used_as_input":False,"auxiliary_output_applied_to_robot":False,
        "auxiliary_training_weight":arm_config["aux_weight"],
        "all_actions_model_generated":True,"expert_wheel_commands_applied":False,
        "target_position_frame": "base_yaw_aligned",
        "ee_error_frame": "base_yaw_aligned",
        "available_split_targets": len(available_targets), "selected_targets": len(targets),
        "partial_subset": len(targets) != len(available_targets),
        "subset":args.subset,"selected_target_indices":indices,
        "target_sources":["base" if i<base_count else "supplement" for i in indices],
        "duration_limit_seconds": args.duration,
        "sensor_noise_mode": args.sensor_noise,
        # [Anytime success change] Match single-episode success without the old 8s gate.
        "reach_tolerance_m": 0.01, "min_success_time_seconds": 0.0,
        "required_hold_seconds": 0.0,
        "success_definition": "first TRUE PyBullet EE distance < 0.01 m at any time within horizon; no minimum time or hold",
        "phase_source": "expert_navigation_rules_commands_discarded",
        "forced_parking": False, "action_limits": [3.0] * 4 + [30.0] * 7,
        "base_seed": args.seed,
        "target_seeds": ([manifest["evaluation_seeds"][args.split][i] for i in indices] if args.seed==42
                         else [args.seed+i for i in indices]),
        "targets": targets, "file_sha256": fingerprints,
        "robot_config_snapshot": json.loads(robot_config_path.read_text(encoding="utf-8")),
        "noise_override_note": "off disables measurement noise in memory; snapshot itself is not modified",
    }
    save_json(output / "evaluation_config.json", conditions)
    # [Parallel change] Each thread launches one ISOLATED Python simulation.
    # Only the parent thread writes summary/CSV; workers have unique NPZ/log paths.
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    rows = []
    fields = ["index", "target_source", "target_x", "target_y", "target_z", "success", "status", "termination_reason",
              "final_error_m", "min_error_m", "end_time_s", "phase_switch_time_s", "action_steps",
              "clipped_steps", "clipped_step_fraction", "seed", "return_code", "rollout_file", "log_file", "error",
              "wall_seconds", "observed_final_error_m", "observed_min_error_m", "observed_reached_but_true_failed"]
    stop = threading.Event()
    registry_lock = threading.Lock()
    active_children = {}
    batch_start = time.perf_counter()

    def update_summary(state):
        summary = summarize(rows, len(targets))
        summary.update({"state":state,"split":args.split,"partial_subset":conditions["partial_subset"],
                        "available_split_targets":len(available_targets),"output_dir":str(output),
                        "execution_mode":"parallel_independent_subprocesses","workers":workers,
                        "elapsed_wall_seconds":time.perf_counter()-batch_start,
                        "completed_success_rate_percent":100*summary["successes"]/len(rows) if rows else None,
                        "subset":args.subset,"by_target_source":{}})
        with registry_lock:
            summary["active_processes"]=len(active_children)
        for source in ["base","supplement"]:
            planned=conditions["target_sources"].count(source)
            if planned:
                summary["by_target_source"][source]=summarize([r for r in rows if r["target_source"]==source],planned)
        save_json(output/"summary.json",summary)
        return summary

    def save_rows():
        # [Parallel change] Finish order may vary; CSV remains in planned target order.
        temp=(output/"results.csv").with_suffix(".csv.tmp")
        with temp.open("w",newline="",encoding="utf-8-sig") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader();writer.writerows(sorted(rows,key=lambda r:r["index"]))
        temp.replace(output/"results.csv")

    def terminate_child(process):
        if process.poll() is None:
            try:
                process.terminate();process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
            except ProcessLookupError:
                pass

    def run_one(offset):
        if stop.is_set():
            return None
        if any(sha256(path)!=fingerprints[str(path)] for path in fixed_files):
            raise RuntimeError("Model/test/config changed; abort rather than mix experimental conditions")
        index=offset+1;target=targets[offset];seed=conditions["target_seeds"][offset]
        rollout=output/"rollouts"/f"episode_{index:03d}.npz"
        log=output/"logs"/f"episode_{index:03d}.log"
        row=dict.fromkeys(fields,"")
        row.update({"index":index,"target_source":conditions["target_sources"][offset],
                    "target_x":target[0],"target_y":target[1],"target_z":target[2],
                    "success":False,"status":"process_error","seed":seed,
                    "rollout_file":str(rollout),"log_file":str(log)})
        command=[sys.executable,"-B",str(script),"--nav-run-dir",str(nav_run),"--arm-run-dir",str(arm_run),
                 "--target",*map(str,target),"--headless","--duration",str(args.duration),
                 "--sensor-noise",args.sensor_noise,"--seed",str(seed),"--result-file",str(rollout)]
        process=None;started=time.perf_counter()
        try:
            with log.open("w",encoding="utf-8") as stream:
                with registry_lock:
                    if stop.is_set():
                        return None
                    process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
                    active_children[index]=process
                row["return_code"]=process.wait(timeout=args.wall_timeout)
            if stop.is_set():
                return None
            if row["return_code"]:
                raise RuntimeError(f"Child exited with code {row['return_code']}; see log")
            with np.load(rollout, allow_pickle=False) as result:
                # [26D test change] 确保子进程保存的是26维实际观测。
                if (int(result["input_dim"]) != 26 or result["observations"].ndim != 2
                        or result["observations"].shape[1] != 26):
                    raise ValueError("Child rollout does not contain 26D observations")
                # [Body frame test change] 同为26维也检查具体特征版本。
                if (str(result["input_feature_version"]) != BODYFRAME_FEATURE_VERSION
                        or str(result["target_position_frame"]) != "base_yaw_aligned"
                        or str(result["ee_error_frame"]) != "base_yaw_aligned"):
                    raise ValueError("Child rollout uses a different coordinate schema")
                if (int(result["navigation_input_dim"])!=8
                        or str(result["navigation_feature_version"])!=NAV_FEATURE_VERSION
                        or result["navigation_observations"].shape!=(len(result["actions"]),8)
                        or not bool(result["all_actions_model_generated"])
                        or bool(result["expert_wheel_commands_applied"])):
                    raise ValueError("Wrong learned-navigation plus learned-arm rollout")
                # [MLP noise change] 必须真实位置评分，且带噪估计单独保存。
                if (str(result["evaluation_position_source"]) != "pybullet_link_world_frame"
                        or bool(result["true_state_used_for_model_input"])
                        or result["observed_ee_positions"].shape != result["ee_positions"].shape):
                    raise ValueError("Wrong ground-truth scoring / sensor observation schema")
                distances = np.linalg.norm(result["ee_positions"] - np.asarray(target), axis=1)
                if not np.allclose(result["ee_errors"], distances, atol=1e-10, rtol=0):
                    raise ValueError("Stored errors differ from true EE distances")
                if bool(result["success"]) != bool(np.any(distances < 0.01)):
                    raise ValueError("Success does not follow anytime TRUE 1cm criterion")
                if args.sensor_noise == "config" and not result["sensor_noise_enabled"].all():
                    raise ValueError("Config measurement noise was not enabled")
                if (bool(result["expert_references_used_as_input"])
                        or bool(result["auxiliary_output_applied_to_robot"])):
                    raise ValueError("Auxiliary reference must never control the robot")
                if not np.allclose(result["target"], target, rtol=0, atol=1e-6):
                    raise ValueError("Rollout target differs from requested target")
                if (Path(str(result["arm_run"])).resolve()!=arm_run
                        or Path(str(result["navigation_run"])).resolve()!=nav_run
                        or not bool(result["arm_finetuned"])):
                    raise ValueError("Wrong frozen navigation / finetuned arm pair")
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
                            "observed_final_error_m": float(result["observed_ee_errors"][-1]),
                            "observed_min_error_m": float(np.min(result["observed_ee_errors"])),
                            "observed_reached_but_true_failed": bool(not success and np.any(result["observed_ee_errors"] < .01)),
                            "end_time_s": float(result["timestamps"][-1]),
                            "phase_switch_time_s": float(result["phase_switch_time"]),
                            "action_steps": action_steps, "clipped_steps": int(result["clipped_steps"]),
                            "clipped_step_fraction": int(result["clipped_steps"]) / action_steps if action_steps else 0.0})
        except subprocess.TimeoutExpired:
            if process is not None:
                terminate_child(process)
            row.update({"termination_reason":"process_wall_timeout","error":f"Wall-clock timeout: {args.wall_timeout}s"})
        except Exception as exc:
            row.update({"success":False,"status":"process_error","termination_reason":"process_error",
                        "error":f"{type(exc).__name__}: {exc}"})
        finally:
            with registry_lock:
                active_children.pop(index,None)
            row["wall_seconds"]=time.perf_counter()-started
        return None if stop.is_set() else row

    print(f"Parallel residual120 noise: workers={workers} (requested {args.workers}); targets={len(targets)}/{len(available_targets)}; "
          f"horizon={args.duration}s; sensor noise={args.sensor_noise}",flush=True)
    if memory_available is not None:
        print(f"Available RAM at start: {memory_available/1024**3:.2f} GiB; each simulation uses one CPU thread",flush=True)
    print(f"Results: {output}",flush=True)
    save_rows();update_summary("running")
    executor=ThreadPoolExecutor(max_workers=workers,thread_name_prefix="robotnik_test")
    pending={};next_offset=0;failed=False
    try:
        # [Parallel change] Bounded submission: no background batch of untracked children.
        while next_offset<len(targets) and len(pending)<workers:
            future=executor.submit(run_one,next_offset);pending[future]=next_offset;next_offset+=1
        while pending:
            done,_=wait(pending,timeout=.25,return_when=FIRST_COMPLETED)
            for future in sorted(done,key=lambda f:pending[f]):
                pending.pop(future)
                row=future.result()
                if row is not None:
                    rows.append(row);save_rows();summary=update_summary("running")
                    print(f"[{len(rows)}/{len(targets)} finished] target#{row['index']} "
                          f"{[row['target_x'],row['target_y'],row['target_z']]} | {row['status']} | "
                          f"error={row['final_error_m']}m | wall={row['wall_seconds']:.1f}s | "
                          f"successes={summary['successes']}/{len(rows)}",flush=True)
                if next_offset<len(targets):
                    new=executor.submit(run_one,next_offset);pending[new]=next_offset;next_offset+=1
    except BaseException:
        failed=True;stop.set()
        # Kill only the child processes launched by THIS batch, never another running test.
        with registry_lock:
            children=list(active_children.values())
        for process in children:
            terminate_child(process)
        for future in pending:
            future.cancel()
        update_summary("interrupted_or_aborted")
        raise
    finally:
        executor.shutdown(wait=True,cancel_futures=failed)
    if any(sha256(path)!=fingerprints[str(path)] for path in fixed_files):
        update_summary("interrupted_or_aborted")
        raise RuntimeError("Frozen file changed during evaluation")
    summary=update_summary("completed")
    print(f"\nSuccess rate: {summary['successes']}/{len(targets)} = {summary['success_rate_percent']:.2f}%")
    print(f"Process errors: {summary['process_errors']} (included in denominator)")
    print(f"Wall time: {summary['elapsed_wall_seconds']:.1f}s; workers={workers}")
    if conditions["partial_subset"]:
        print("Partial subset only; this is NOT the full test-set success rate.")
    print(f"Summary: {output/'summary.json'}")


if __name__=="__main__":
    if sys.argv[1:2]==["--resolve-pair"]:
        preflight_main()
    else:
        main()
