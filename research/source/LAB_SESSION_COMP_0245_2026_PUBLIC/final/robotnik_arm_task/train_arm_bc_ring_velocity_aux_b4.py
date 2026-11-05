# [B4 change] 只更换辅助标签为qd_des_arm；起点、网络、梯度门控、采样、主损失和早停与B3一致。
# [B4 inherited] 梯度方向门控辅助监督；gated/fixed/off同一起点对照，训练与测试分离。
"""[B4 inherited] 只训练：MLP力矩模仿 + 专家期望角速度辅助监督；无GRU，无仿真测试。

26维输入保持不变；训练预测11维旧动作（仅后7维参与控制损失）和7维辅助标签。
推理仅使用力矩路径，辅助参考从不作为输入。测试入口独立。
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import shutil
import time
import zipfile

import numpy as np
import torch

from robotnik_arm_aux_policy import resolve_finetuned_pair as resolve_baseline_pair
from robotnik_bodyframe_features import build_bodyframe_observations
from robotnik_arm_velocity_aux_policy import AuxiliaryArmPolicy, VARIANT, RUN_ROOT, FROZEN_FILES, sha256
from arm_aux_gradient_gate import (apply_gradient_update, diagnostic_vector, diagnostic_summary, GRADIENT_FIELDS)

# [B4 inherited] 从新带噪基线（无辅助继续训练最佳185轮，113/144）开始；不是从旧失败辅助组续训。
BASELINE = "bc_training_arm_ring_r08_aux_b2_control/run_20261104_025706_206181"
METRICS = ["arm_mse", "low_arm_mse", "other_arm_mse", "balanced_arm_mse",
           "aux_mse", "balanced_aux_mse", "aux_rmse_rad_s", "total_loss"]


def selected_files(manifest, split, limit):
    pairs = list(zip(manifest["splits"][split], manifest["targets"][split]))
    if limit is None:
        return pairs
    # [B2 check] 小样本检查覆盖旧/补采数据以及两类高度，不用于报告正式效果。
    selected = []
    for prefix in ["expert_data_23d_noise_ring/", "expert_data_23d_noise_ring_lowz_supplement/"]:
        for low in [True, False]:
            selected.append(next(p for p in pairs if p[0].startswith(prefix)
                                 and (p[1][2] <= .35+1e-6) == low))
    selected += [p for p in pairs if p not in selected][:limit-4]
    if len(selected) != limit:
        raise ValueError("Insufficient smoke files")
    return selected


def array_shape(path, name):
    with zipfile.ZipFile(path) as z, z.open(name+".npy") as stream:
        version = np.lib.format.read_magic(stream)
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0}
        shape, _, _ = readers[version](stream)
    return shape


def check_headers(manifest, split, limit=None):
    files = selected_files(manifest, split, limit)
    lengths = []
    for filename, _ in files:
        p = Path(manifest["data_dir"])/filename
        obs = array_shape(p, "observations")
        if len(obs) != 2 or obs[1] != 23 or obs[0] < 2:
            raise ValueError(f"Wrong observation schema: {p}")
        n = obs[0]
        for name, shape in [("actions", (n, 11)), ("qd_des_arm", (n, 7)),
                            ("ee_positions", (n, 3)), ("timestamps", (n,))]:
            if array_shape(p, name) != shape:
                raise ValueError(f"Unaligned auxiliary labels: {name}: {p}")
        lengths.append(n)
    return files, lengths


def load_dataset(manifest, split, stats, limit=None, aux_stats=None):
    files, lengths = check_headers(manifest, split, limit)
    n = sum(lengths)
    # [B2 speed] 预分配，数据只保留单帧，不复制成历史；稍后一次放入GPU。
    x = np.empty((n, 26), np.float32)
    y = np.empty((n, 7), np.float32)
    aux = np.empty((n, 7), np.float32)
    low = np.empty(n, bool)
    total = np.zeros(7, np.float64); square = total.copy()
    offset = 0
    for index, ((filename, target), size) in enumerate(zip(files, lengths), 1):
        with np.load(Path(manifest["data_dir"])/filename, allow_pickle=False) as d:
            obs = d["observations"].astype(np.float32)
            action = d["actions"].astype(np.float32)
            # [B4 label] 专家在当前帧计算的期望关节角速度，不是传感器实测qd。
            reference_velocity = d["qd_des_arm"].astype(np.float64)
            if (not np.isfinite(obs).all() or not np.isfinite(action).all()
                    or not np.isfinite(reference_velocity).all() or not np.isin(obs[:, 22], [0, 1]).all()):
                raise ValueError(f"Nonfinite data or invalid phase: {filename}")
            np.testing.assert_allclose(d["target"], target, atol=1e-6, rtol=0)
            if not np.all(np.diff(d["timestamps"]) > 0):
                raise ValueError("Non-increasing timestamps")
            # [B2 label] 采集器在同一帧计算参考、力矩并保存执行前观测；不平移/不取未来帧。
            # [B4 label] 直接取同帧qd_des，不减当前关节角/速度，也不做时间差分。
            auxiliary_target = reference_velocity
            sl = slice(offset, offset+size)
            x[sl] = (build_bodyframe_observations(obs, d["ee_positions"])-stats["x_mean"])/stats["x_std"]
            y[sl] = (action[:, 4:11]-stats["y_mean"][4:11])/stats["y_std"][4:11]
            aux[sl] = auxiliary_target
            low[sl] = obs[:, 21] <= .35+1e-6
            total += auxiliary_target.sum(0); square += np.square(auxiliary_target).sum(0)
        offset += size
        if index % 100 == 0 or index == len(files):
            print(f"{split}: {index}/{len(files)} trajectories loaded", flush=True)
    if not low.any() or low.all():
        raise ValueError("Both height groups required")
    if aux_stats is None:
        if split != "train":
            raise ValueError("Aux statistics may be fitted on TRAIN only")
        mean = total/n
        std = np.sqrt(np.maximum(square/n-mean**2, 0))
        aux_stats = {"mean": mean.astype(np.float32), "std": np.maximum(std, 1e-4).astype(np.float32)}
    # [B2 normalization] 各关节偏差分别标准化；验证沿用训练统计量，禁止验证泄漏。
    aux -= aux_stats["mean"]; aux /= aux_stats["std"]
    return x, y, aux, low, aux_stats, len(files)


def multitask_errors(action, reference, y, aux, weight):
    torque_error = (action[:, 4:11]-y).square().mean(1)
    auxiliary_error = (reference-aux).square().mean(1)
    return torque_error.mean()+weight*auxiliary_error.mean(), torque_error, auxiliary_error


class FrameStore:
    def __init__(self, x, y, aux, low, device):
        self.x, self.y, self.aux, self.low = [torch.as_tensor(v).to(device) for v in [x, y, aux, low]]


def run_epoch(model, data, indices, batch_size, aux_weight, aux_std, optimizer=None, gradient_mode="gated"):
    # [B4 inherited] 验证只算指标，不计算或更新梯度；门控只依赖训练批次。
    diagnostic_sums = torch.zeros(len(GRADIENT_FIELDS), dtype=torch.float64, device=data.x.device)
    model.train(optimizer is not None)
    sums = torch.zeros((2, 2), dtype=torch.float64, device=data.x.device)
    counts = torch.zeros(2, dtype=torch.int64, device=data.x.device)
    physical_aux_sum = torch.zeros((), dtype=torch.float64, device=data.x.device)
    with torch.set_grad_enabled(optimizer is not None):
        for start in range(0, len(indices), batch_size):
            ix = indices[start:start+batch_size]
            action, reference = model.forward_tasks(data.x[ix])
            loss, te, ae = multitask_errors(action, reference, data.y[ix], data.aux[ix], aux_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite multitask loss")
            if optimizer is not None:
                geometry = apply_gradient_update(model, te.mean(), ae.mean(), optimizer, aux_weight, gradient_mode)
                diagnostic_sums += diagnostic_vector(geometry)
            masks = torch.stack([data.low[ix], ~data.low[ix]])
            values = torch.stack([te.detach(), ae.detach()]).double()
            sums += values @ masks.double().T
            counts += masks.sum(1)
            physical_aux_sum += ((reference.detach()-data.aux[ix])*aux_std).double().square().mean(1).sum()
    sums = sums.cpu().numpy(); counts = counts.cpu().numpy()
    if np.any(counts == 0):
        raise ValueError("Each epoch must contain both groups")
    means = sums/counts[None, :]
    arm, aux = sums.sum(1)/counts.sum()
    result = {"arm_mse": float(arm), "low_arm_mse": float(means[0, 0]),
            "other_arm_mse": float(means[0, 1]), "balanced_arm_mse": float(means[0].mean()),
            "aux_mse": float(aux), "balanced_aux_mse": float(means[1].mean()),
            "aux_rmse_rad_s": float(np.sqrt(physical_aux_sum.cpu().item()/counts.sum())),
            "total_loss": float(arm+aux_weight*aux)}
    if optimizer is not None:
        result["gradient_diagnostics"] = diagnostic_summary(diagnostic_sums)
    return result


def main():
    roboenv = Path(__file__).resolve().parents[3]/"RoboEnv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-arm-run-dir", type=Path, default=roboenv/BASELINE)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--aux-weight", type=float, default=.01,
                        help="lambda for normalized auxiliary MSE; 0 gives matched torque-only control")
    # [B4 inherited] 门控为默认；fixed和off用于同起点、同预算对照。
    parser.add_argument("--gradient-mode", choices=["gated", "fixed", "off"], default="gated")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--samples-per-epoch", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-train-files", type=int)
    parser.add_argument("--max-val-files", type=int)
    parser.add_argument("--check-data-only", action="store_true", help="Validate train/val array headers without training")
    args = parser.parse_args()
    if (min(args.epochs, args.patience, args.batch_size, args.samples_per_epoch) < 1
            or not np.isfinite(args.aux_weight) or args.aux_weight < 0
            or not np.isfinite(args.learning_rate) or args.learning_rate <= 0
            or any(v is not None and v < 4 for v in [args.max_train_files, args.max_val_files])):
        parser.error("Invalid training settings; smoke file limits must be at least4")
    if args.gradient_mode == "off":
        args.aux_weight = 0.0  # [B4 inherited] off明确关闭所有辅助梯度。
    elif args.aux_weight == 0:
        parser.error("Use --gradient-mode off for the zero-auxiliary control")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    parent, nav, pc, nc = resolve_baseline_pair(roboenv, args.parent_arm_run_dir)
    manifest = json.loads((parent/"data_split.json").read_text(encoding="utf-8"))
    if args.check_data_only:
        result = {}
        for split, limit in [("train", args.max_train_files), ("val", args.max_val_files)]:
            files, lengths = check_headers(manifest, split, limit)
            result[split] = {"trajectories": len(files), "aligned_frames": sum(lengths)}
        print(json.dumps({"parent":str(parent),"aux_label":"qd_des_arm[t]",
                          "data":result,"training_started":False}, indent=2))
        return
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    else:
        torch.set_num_threads(min(4, torch.get_num_threads()))
    frozen = {str(folder/name):sha256(folder/name) for folder in [parent, nav] for name in FROZEN_FILES}
    def verify_frozen():
        for file, expected in frozen.items():
            if sha256(file) != expected:
                raise RuntimeError(f"Source model changed: {file}")
    checkpoint = torch.load(parent/"best_model.pt", map_location="cpu", weights_only=True)
    if checkpoint["config"] != pc:
        raise ValueError("Parent checkpoint/config mismatch")
    model = AuxiliaryArmPolicy(pc["hidden_dims"])
    # [B4 inherited] 继承新基线全部参数，并把未参与控制的辅助头统一重置为0；三个模式初始化一致。
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    torch.nn.init.zeros_(model.aux_head.weight)
    torch.nn.init.zeros_(model.aux_head.bias)
    with np.load(parent/"normalization.npz", allow_pickle=False) as d:
        stats = {k:d[k].copy() for k in ["x_mean", "x_std", "y_mean", "y_std"]}
    print(f"Device: {device}; parent MLP epoch={checkpoint['epoch']}; B4 mode={args.gradient_mode}; auxiliary weight={args.aux_weight}", flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    x, y, aux, low, aux_stats, tr_files = load_dataset(manifest, "train", stats, args.max_train_files)
    weights = torch.from_numpy(np.where(low, .5/low.sum(), .5/(~low).sum()))
    train_count, low_fraction = len(x), float(low.mean())
    train = FrameStore(x, y, aux, low, device)
    del x, y, aux, low
    xv, yv, av, lv, _, va_files = load_dataset(manifest, "val", stats, args.max_val_files, aux_stats)
    val_count, val_low_fraction = len(xv), float(lv.mean())
    val = FrameStore(xv, yv, av, lv, device)
    del xv, yv, av, lv
    count = min(args.samples_per_epoch, train_count)
    generator = torch.Generator().manual_seed(42)
    val_indices = torch.arange(val_count, device=device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    aux_std = torch.as_tensor(aux_stats["std"], device=device)
    smoke = args.max_train_files is not None or args.max_val_files is not None
    root = RUN_ROOT if args.gradient_mode == "gated" else RUN_ROOT+"_"+args.gradient_mode
    run = (args.output_dir or roboenv/root/datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")).resolve()
    run.mkdir(parents=True, exist_ok=False)
    for name in ["normalization.npz", "data_split.json"]:
        shutil.copyfile(parent/name, run/name)
    np.savez(run/"aux_normalization.npz", **aux_stats)
    config = dict(pc)
    config.update({"experiment_variant":VARIANT,"training_mode":"arm_only_gradient_gated_aux_navigation_frozen",
        "source_model_run_dir":str(parent),"source_model_epoch":int(checkpoint["epoch"]),
        "source_model_file_sha256":{n:sha256(parent/n) for n in FROZEN_FILES},
        "nav_run_dir":str(nav),"frozen_nav_file_sha256":{n:sha256(nav/n) for n in FROZEN_FILES},
        "navigation_weights_updated":False,"gru_used":False,"expert_references_used_as_input":False,
        "policy_architecture":"MLP26_shared256_256_128_action11_aux_desired_velocity7",
        "auxiliary_output_dim":7,"auxiliary_label":"qd_des_arm[t]",
        "auxiliary_label_units":"rad/s","auxiliary_label_alignment":"same_pre_action_frame_t_no_future",
        "auxiliary_normalization_fitted_on":"train_only_natural_frames",
        "aux_normalization_sha256":sha256(run/"aux_normalization.npz"),"aux_weight":args.aux_weight,
        "gradient_mode":args.gradient_mode,
        "shared_gradient_definition":"g_main + lambda * max(0, cosine(g_main,g_aux)) * g_aux in gated mode",
        "gradient_cosine_scope":"all_shared_trunk_parameters_excluding_both_output_heads",
        "gradient_cosine_is_detached":True,"zero_norm_gate":0.0,
        "auxiliary_head_update":"lambda*g_aux_head in gated/fixed; zero in off",
        "nominal_total_loss_is_not_gated_scalar_objective":True,
        "training_module_sha256":{"arm_aux_gradient_gate.py":sha256(Path(__file__).with_name("arm_aux_gradient_gate.py"))},
        "auxiliary_output_applied_to_robot":False,"trained_output_columns":list(range(4,11)),
        "selection_metric":"val_balanced_arm_mse","selection_includes_auxiliary_loss":False,
        "loss_type":"task_gradients_shared_cosine_gate_with_separate_heads",
        "sampling_method":"low_other_50_50_weighted_with_replacement","samples_per_epoch":count,
        "natural_low_training_fraction":low_fraction,"natural_low_validation_fraction":val_low_fraction,
        "train_samples":train_count,"val_samples":val_count,"train_trajectories":tr_files,"val_trajectories":va_files,
        "batch_size":args.batch_size,"max_epochs":args.epochs,"patience":args.patience,
        "learning_rate":args.learning_rate,"seed":42,"device":str(device),"torch_version":str(torch.__version__),
        "smoke_only":smoke,"optimizer_initialization":"new_Adam_for_multitask_experiment",
        "parameter_count":sum(p.numel() for p in model.parameters()),
        "inference_module_sha256":{n:sha256(Path(__file__).with_name(n)) for n in [
            "robotnik_arm_velocity_aux_policy.py","robotnik_arm_aux_policy.py",
            "robotnik_bodyframe_features.py","robotnik_navigation_policy.py","robotnik_arm_control.py"]}})
    (run/"config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copyfile(__file__, run/"training_source.py")
    shutil.copyfile(Path(__file__).with_name("arm_aux_gradient_gate.py"), run/"gradient_update_source.py")
    initial = run_epoch(model, val, val_indices, args.batch_size, args.aux_weight, aux_std)
    best, best_epoch, stale = initial["balanced_arm_mse"], 0, 0
    (run/"initial_validation.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
    def save(name, epoch, metrics):
        torch.save({"model_state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            "optimizer_state_dict":optimizer.state_dict(),"sampling_generator_state":generator.get_state(),
            "torch_rng_state":torch.get_rng_state(),
            "cuda_rng_state_all":torch.cuda.get_rng_state_all() if device.type=="cuda" else [],
            "epoch":epoch,"val_mse":metrics["arm_mse"],"val_loss":metrics["balanced_arm_mse"],
            "val_metrics":metrics,"config":config}, run/name)
    save("best_model.pt", 0, initial)
    print(f"Train frames={train_count:,}; val frames={val_count:,}; draws/epoch={count:,}\nOutput: {run}\nInitial: {initial}", flush=True)
    # [B4 inherited] 每轮训练梯度统计单独落盘；所有模式均输出，便于诊断。
    with (run/"history.csv").open("w", newline="", encoding="utf-8") as stream, \
            (run/"gradient_diagnostics.csv").open("w", newline="", encoding="utf-8") as grad_stream:
        grad_writer = csv.DictWriter(grad_stream, fieldnames=["epoch"]+GRADIENT_FIELDS)
        grad_writer.writeheader()
        writer = csv.writer(stream)
        writer.writerow(["epoch","seconds"]+[s+"_"+m for s in ["train","val"] for m in METRICS])
        for epoch in range(1, args.epochs+1):
            verify_frozen()
            start = time.perf_counter()
            indices = torch.multinomial(weights, count, replacement=True, generator=generator).to(device)
            tr = run_epoch(model, train, indices, args.batch_size, args.aux_weight, aux_std, optimizer, args.gradient_mode)
            gd = tr["gradient_diagnostics"]
            grad_writer.writerow({"epoch":epoch, **gd}); grad_stream.flush()
            va = run_epoch(model, val, val_indices, args.batch_size, args.aux_weight, aux_std)
            writer.writerow([epoch,time.perf_counter()-start]+[m[k] for m in [tr,va] for k in METRICS]); stream.flush()
            print(f"Epoch {epoch:03d} | train torque {tr['arm_mse']:.6f} | val torque {va['arm_mse']:.6f} | "
                  f"low {va['low_arm_mse']:.6f} | other {va['other_arm_mse']:.6f} | "
                  f"balanced {va['balanced_arm_mse']:.6f} | val aux {va['aux_mse']:.6f} | "
                  f"cos {gd['cosine_mean_valid']:.3f} | conflict {100*gd['conflict_fraction_valid']:.1f}% | "
                  f"aux active {100*gd['gate_active_fraction']:.1f}%", flush=True)
            if va["balanced_arm_mse"] < best:
                best, best_epoch, stale = va["balanced_arm_mse"], epoch, 0
                save("best_model.pt", epoch, va)
            else:
                stale += 1
            save("last_model.pt", epoch, va)
            if stale >= args.patience:
                print("Early stopping: balanced validation TORQUE MSE did not improve.", flush=True)
                break
    verify_frozen()
    (run/"training_finished.json").write_text(json.dumps({"best_epoch":best_epoch,"last_epoch":epoch,
        "best_balanced_validation_arm_mse":best,"smoke_only":smoke,"navigation_unchanged":True,
        "aux_weight":args.aux_weight,"gradient_mode":args.gradient_mode}, indent=2), encoding="utf-8")
    print(f"Best B4 epoch: {best_epoch}; balanced validation arm MSE: {best:.6f}\nResults: {run}", flush=True)
    print("Training only. Run batch_test_bc_split_nav_arm_velocity_aux_b4_parallel.py separately.", flush=True)


if __name__ == "__main__":
    main()
