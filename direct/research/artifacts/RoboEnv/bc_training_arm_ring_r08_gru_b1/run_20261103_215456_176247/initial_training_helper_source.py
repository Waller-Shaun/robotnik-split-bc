"""[B1 GRU change] Offline history-arm training only; no simulation/test run.

Same 998 split, parent MLP, normalization, torque labels, 50/50 height sampling,
Adam LR and selection metric as finetune_arm_bc_ring_lowz_supplement.py.
New: a finite causal history GRU branch, initially zero, predicts a correction.
No q_des labels/inputs, no recovery data collection, no forced wheel stopping.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from robotnik_bodyframe_features import BODYFRAME_FEATURE_VERSION, build_bodyframe_observations
from arm_finetune_experiment import resolve_finetuned_pair as resolve_parent_pair
from arm_lowz_supplement_experiment import validate_merged_manifest
from robotnik_arm_gru_policy import (HistoryArmPolicy, VARIANT, HISTORY_VERSION, RUN_ROOT,
                                     FROZEN_FILES, sha256)

SEED = 42
BATCH_SIZE = 4096
MAX_EPOCHS = 30
PATIENCE = 8
LEARNING_RATE = 1e-5
SAMPLES_PER_EPOCH = 500_000
LOW_MAX_Z = .35
HISTORY_LENGTH = 16
HISTORY_STRIDE_STEPS = 10  # [B1] 16 observations cover 150 ms at dt=1 ms.


def select_files(manifest, split, limit):
    pairs = list(zip(manifest["splits"][split], manifest["targets"][split]))
    if limit is None:
        return pairs
    # [B1] Small program checks include old/new sources and both height groups.
    if limit < 4:
        raise ValueError("Smoke file limits must be at least four")
    selected = []
    for source in ["expert_data_23d_noise_ring/", "expert_data_23d_noise_ring_lowz_supplement/"]:
        for low in [True, False]:
            selected.append(next(p for p in pairs if p[0].startswith(source)
                                 and (p[1][2] <= LOW_MAX_Z + 1e-6) == low))
    selected += [p for p in pairs if p not in selected][:limit - 4]
    if len(selected) != limit:
        raise ValueError("Not enough smoke trajectories")
    return selected


def load_episodes(manifest, split, stats, limit=None):
    """[B1] Store each frame ONCE, plus episode boundaries; do not expand N*K.

    A random training index is the end of a causal window. Older observations
    are gathered from the SAME episode, so random windows remain sequential.
    """
    xs, ys, groups, lengths = [], [], [], []
    sample_dt = None
    files = select_files(manifest, split, limit)
    for index, (filename, target) in enumerate(files, 1):
        with np.load(Path(manifest["data_dir"]) / filename, allow_pickle=False) as data:
            obs = data["observations"].astype(np.float32)
            actions = data["actions"]
            n = len(obs)
            if (obs.shape != (n, 23) or n < 2 or actions.shape != (n, 11)
                    or not np.isfinite(actions).all() or not np.isin(obs[:, 22], [0, 1]).all()):
                raise ValueError(f"Wrong observation/action schema: {filename}")
            np.testing.assert_allclose(data["target"], target, atol=1e-6, rtol=0)
            times = data["timestamps"].astype(np.float64)
            if times.shape != (n,) or not np.isfinite(times).all():
                raise ValueError(f"Invalid time alignment: {filename}")
            dt = float(np.median(np.diff(times)))
            if not np.allclose(np.diff(times), dt, atol=1e-9, rtol=0) or dt <= 0:
                raise ValueError("History assumes a constant per-frame sampling period")
            sample_dt = dt if sample_dt is None else sample_dt
            if not np.isclose(dt, sample_dt, atol=1e-9, rtol=0):
                raise ValueError("Trajectories use different sampling periods")
            x = build_bodyframe_observations(obs, data["ee_positions"])
            x = (x - stats["x_mean"]) / stats["x_std"]
            y = (actions[:, 4:11].astype(np.float32) - stats["y_mean"][4:11]) / stats["y_std"][4:11]
            xs.append(x); ys.append(y)
            groups.append(obs[:, 21] <= LOW_MAX_Z + 1e-6)
            lengths.append(n)
        if index % 100 == 0 or index == len(files):
            print(f"{split}: {index}/{len(files)} episodes loaded, causal boundaries retained", flush=True)
    x, y, low = np.concatenate(xs), np.concatenate(ys), np.concatenate(groups)
    if not low.any() or low.all():
        raise ValueError("Both height groups are required")
    return x, y, low, np.asarray(lengths, dtype=np.int64), float(sample_dt)


class WindowStore:
    """[B1 / speed] Gather windows in batches on the selected device.

    CUDA keeps the normalized frame arrays on GPU once; this avoids a Python
    DataLoader loop over millions of windows and per-batch CPU->GPU transfers.
    No sampling/control semantics are changed by this storage choice.
    """
    def __init__(self, x, y, low, lengths, device, history_length, history_stride):
        self.x = torch.as_tensor(x).to(device)
        self.y = torch.as_tensor(y).to(device)
        self.low = torch.as_tensor(low).to(device)
        self.n = len(x)
        if np.sum(lengths) != self.n or (lengths <= 0).any():
            raise ValueError("Invalid episode boundaries")
        self.ends = torch.as_tensor(np.cumsum(lengths), device=device)
        self.starts = torch.as_tensor(np.r_[0, np.cumsum(lengths)[:-1]], device=device)
        self.offsets = torch.arange(history_length - 1, -1, -1, device=device) * history_stride

    def batch(self, indices):
        episodes = torch.bucketize(indices, self.ends, right=True)
        earliest = self.starts[episodes]
        # No future frame and no previous-episode leakage, including episode starts.
        windows = torch.maximum(indices[:, None] - self.offsets[None, :], earliest[:, None])
        return self.x[windows], self.y[indices], self.low[indices]


def run_epoch(model, store, indices, batch_size, optimizer=None):
    training = optimizer is not None
    model.train(training)
    sums = torch.zeros(2, device=store.x.device, dtype=torch.float64)
    counts = torch.zeros(2, device=store.x.device, dtype=torch.int64)
    with torch.set_grad_enabled(training):
        for start in range(0, len(indices), batch_size):
            x, y, low = store.batch(indices[start:start + batch_size])
            error = (model(x) - y).square().mean(dim=1)
            loss = error.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite history-arm loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            # [B1 / speed] Accumulate on GPU; transfer epoch summaries only once.
            err = error.detach().double()
            mask = low.to(torch.float64)
            sums += torch.stack([(err * mask).sum(), (err * (1 - mask)).sum()])
            counts += torch.stack([low.sum(), (~low).sum()])
    sums, counts = sums.cpu().numpy(), counts.cpu().numpy()
    if np.any(counts == 0):
        raise ValueError("An epoch must include both height groups")
    group = sums / counts
    return {"arm_mse": float(sums.sum() / counts.sum()), "low_arm_mse": float(group[0]),
            "other_arm_mse": float(group[1]), "balanced_arm_mse": float(group.mean())}


def main():
    roboenv = Path(__file__).resolve().parents[3] / "RoboEnv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nav-run-dir", type=Path, default=roboenv / "bc_training_nav_ring_r08/run_20261103_022339_570160")
    parser.add_argument("--parent-arm-run-dir", type=Path, default=roboenv / "bc_training_arm_ring_r08_finetune/run_20261103_111655_849512")
    parser.add_argument("--split-path", type=Path, default=roboenv / "bc_training_arm_ring_r08_lowz_supplement/data_split.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--history-length", type=int, default=HISTORY_LENGTH,
                        help="1 is the same-capacity current-only ablation; default16 uses history")
    parser.add_argument("--history-stride-steps", type=int, default=HISTORY_STRIDE_STEPS)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-train-files", type=int, help="Program smoke check only; at least4")
    parser.add_argument("--max-val-files", type=int, help="Program smoke check only; at least4")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.history_length, args.history_stride_steps) < 1:
        parser.error("epochs, batch-size and history settings must be positive")
    if any(v is not None and v < 4 for v in [args.max_train_files, args.max_val_files]):
        parser.error("smoke file limits must be at least4")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable in this Python environment")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    else:
        torch.set_num_threads(min(4, torch.get_num_threads()))
    parent, nav, parent_config, nav_config = resolve_parent_pair(roboenv, args.parent_arm_run_dir, args.nav_run_dir)
    if parent_config["hidden_dims"] != [256, 256, 128] or parent_config["input_feature_version"] != BODYFRAME_FEATURE_VERSION:
        raise ValueError("B1 expects the same parent architecture/features as the B0 supplement run")
    manifest = json.loads(args.split_path.read_text(encoding="utf-8"))
    validate_merged_manifest(manifest, json.loads((nav / "data_split.json").read_text(encoding="utf-8")))
    parent_checkpoint = torch.load(parent / "best_model.pt", map_location="cpu", weights_only=True)
    if parent_checkpoint["config"] != parent_config:
        raise ValueError("Parent checkpoint/config differ")
    with np.load(parent / "normalization.npz", allow_pickle=False) as data:
        stats = {k: data[k].copy() for k in ["x_mean", "x_std", "y_mean", "y_std"]}
    if (stats["x_mean"].shape != (26,) or stats["y_mean"].shape != (11,)
            or stats["x_std"].shape != (26,) or stats["y_std"].shape != (11,)
            or not all(np.isfinite(v).all() for v in stats.values())
            or np.any(stats["x_std"] <= 0) or np.any(stats["y_std"] <= 0)
            or stats["x_mean"][22] != 0 or stats["x_std"][22] != 1):
        raise ValueError("Invalid inherited normalization")
    print(f"B1 Device: {device}; frozen nav; current MLP + causal GRU64 correction", flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    x, y, low, lengths, dt = load_episodes(manifest, "train", stats, args.max_train_files)
    weights = torch.from_numpy(np.where(low, .5 / low.sum(), .5 / (~low).sum()))
    train_samples = len(x)
    train_store = WindowStore(x, y, low, lengths, device, args.history_length, args.history_stride_steps)
    del x, y, low
    xv, yv, lowv, lengthsv, dtv = load_episodes(manifest, "val", stats, args.max_val_files)
    if not np.isclose(dt, dtv, atol=1e-9, rtol=0):
        raise ValueError("Train/validation history sampling periods differ")
    val_samples = len(xv)
    val_store = WindowStore(xv, yv, lowv, lengthsv, device, args.history_length, args.history_stride_steps)
    del xv, yv, lowv
    samples_per_epoch = min(SAMPLES_PER_EPOCH, train_samples)
    validation_indices = torch.arange(val_samples, device=device)
    sampling_generator = torch.Generator().manual_seed(SEED)
    # [B1] A current-only capacity control must not replace default temporal selection.
    output_root = RUN_ROOT if args.history_length > 1 else RUN_ROOT + "_current_only"
    run = (args.output_dir or roboenv / output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")).resolve()
    run.mkdir(parents=True, exist_ok=False)
    frozen_nav = {n: sha256(nav / n) for n in FROZEN_FILES}
    frozen_parent = {n: sha256(parent / n) for n in FROZEN_FILES}
    code_names = ["robotnik_arm_gru_policy.py", "robotnik_bodyframe_features.py", "robotnik_navigation_policy.py", "robotnik_arm_control.py"]
    config = {"experiment_variant": VARIANT, "training_mode": "arm_only_causal_history_navigation_frozen",
        "input_dim": 26, "source_input_dim": 23, "output_dim": 7, "output_action_columns": list(range(4, 11)),
        "input_feature_version": BODYFRAME_FEATURE_VERSION, "history_feature_version": HISTORY_VERSION,
        "target_position_frame": "base_yaw_aligned", "ee_error_frame": "base_yaw_aligned", "base_active_column": 22,
        "hidden_dims": [256, 256, 128], "encoder_dim": 64, "gru_hidden_dim": 64, "activation": "ReLU",
        "history_length": args.history_length, "history_stride_steps": args.history_stride_steps,
        "sample_dt_seconds": dt, "history_span_seconds": (args.history_length - 1) * args.history_stride_steps * dt,
        "history_padding": "repeat_episode_first", "history_hidden_state_carried": False,
        "history_reset": "new finite window, no state from previous episodes", "auxiliary_reference_loss": False,
        "teacher_references_used_as_input": False, "baseline_initialization": "same_parent_as_B0_supplement",
        "baseline_parameters_trainable": True, "gru_correction_zero_initialized": True,
        "navigation_weights_updated": False, "nav_run_dir": str(nav), "frozen_nav_file_sha256": frozen_nav,
        "source_model_run_dir": str(parent), "source_model_file_sha256": frozen_parent, "parent_best_epoch": parent_checkpoint["epoch"],
        "source_split_path": str(args.split_path.resolve()), "source_split_sha256": sha256(args.split_path),
        "source_normalization_sha256": sha256(parent / "normalization.npz"),
        "feature_source_sha256": {n: sha256(Path(__file__).with_name(n)) for n in code_names},
        "target_counts": manifest["target_counts"], "dataset_variant": manifest["dataset_variant"], "data_dir": manifest["data_dir"],
        "train_trajectories": len(lengths), "val_trajectories": len(lengthsv), "train_samples": train_samples, "val_samples": val_samples,
        "samples_per_epoch": samples_per_epoch, "sampling_method": "same_frame_endpoint_low_other_50_50_weighted_with_replacement",
        "loss_type": "mse_normalized_arm_torques_only", "selection_metric": "val_balanced_arm_mse",
        "validation_selection_definition": "0.5*low_arm_mse+0.5*other_arm_mse", "low_height_max_z_m": LOW_MAX_Z,
        "normalization_distribution": "inherited_parent_unchanged", "validation_all_frames": True,
        "batch_size": args.batch_size, "max_epochs": args.epochs, "learning_rate": LEARNING_RATE, "patience": PATIENCE,
        "seed": SEED, "device": str(device), "torch_version": str(torch.__version__), "dtype": "float32",
        "data_storage": "frame_arrays_on_training_device_windows_gathered_lazily",
        "smoke_only": args.max_train_files is not None or args.max_val_files is not None}
    model = HistoryArmPolicy().to(device)
    model.baseline.load_state_dict(parent_checkpoint["model_state_dict"], strict=True)
    config["parameter_count"] = sum(p.numel() for p in model.parameters())
    config["baseline_parameter_count"] = sum(p.numel() for p in model.baseline.parameters())
    (run / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(args.split_path, run / "data_split.json")
    shutil.copyfile(parent / "normalization.npz", run / "normalization.npz")
    shutil.copyfile(Path(__file__), run / "training_source.py")
    shutil.copyfile(Path(__file__).with_name("robotnik_arm_gru_policy.py"), run / "gru_policy_source.py")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"Output: {run}\nHistory: {args.history_length} x 26; spacing={args.history_stride_steps*dt:.4f}s; span={config['history_span_seconds']:.4f}s", flush=True)
    print(f"Train frames={train_samples:,}; val frames={val_samples:,}; sampled endpoints/epoch={samples_per_epoch:,}; parameters={config['parameter_count']:,}", flush=True)
    initial = run_epoch(model, val_store, validation_indices, args.batch_size)
    best, best_epoch, stale = initial["balanced_arm_mse"], 0, 0

    def save_best(epoch, metrics):
        torch.save({"model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "epoch": epoch, "val_mse": metrics["arm_mse"], "val_loss": metrics["balanced_arm_mse"],
            "val_metrics": metrics, "config": config}, run / "best_model.pt")

    save_best(0, initial)
    (run / "initial_validation.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
    metrics = ["arm_mse", "low_arm_mse", "other_arm_mse", "balanced_arm_mse"]
    print("Epoch000 (same parent outputs):", initial, flush=True)
    with (run / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "seconds"] + [s + "_" + m for s in ["train", "val"] for m in metrics])
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            indices = torch.multinomial(weights, samples_per_epoch, replacement=True, generator=sampling_generator).to(device)
            tr = run_epoch(model, train_store, indices, args.batch_size, optimizer)
            va = run_epoch(model, val_store, validation_indices, args.batch_size)
            writer.writerow([epoch, time.perf_counter() - start] + [v[m] for v in [tr, va] for m in metrics])
            stream.flush()
            print(f"Epoch {epoch:03d} | train arm MSE {tr['arm_mse']:.6f} | val arm MSE {va['arm_mse']:.6f} | low {va['low_arm_mse']:.6f} | other {va['other_arm_mse']:.6f} | balanced {va['balanced_arm_mse']:.6f}", flush=True)
            if va["balanced_arm_mse"] < best:
                best, best_epoch, stale = va["balanced_arm_mse"], epoch, 0
                save_best(epoch, va)
            else:
                stale += 1
            if stale >= PATIENCE:
                print("Early stopping: balanced validation arm MSE did not improve.", flush=True)
                break
    for folder, fingerprints in [(nav, frozen_nav), (parent, frozen_parent)]:
        if any(sha256(folder / n) != expected for n, expected in fingerprints.items()):
            raise RuntimeError("Frozen parent/navigation changed during B1 training")
    (run / "training_finished.json").write_text(json.dumps({"best_epoch": best_epoch, "last_epoch": epoch,
        "best_balanced_validation_arm_mse": best, "navigation_unchanged": True, "smoke_only": config["smoke_only"]}, indent=2), encoding="utf-8")
    print(f"Best B1 epoch: {best_epoch}; balanced validation arm MSE: {best:.6f}\nResults: {run}", flush=True)
    print("Training only. Run batch_test_bc_split_nav_arm_gru.py separately.", flush=True)


if __name__ == "__main__":
    main()
