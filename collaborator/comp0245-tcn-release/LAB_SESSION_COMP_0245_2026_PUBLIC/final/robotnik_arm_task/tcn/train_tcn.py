"""Train TCN or current-state MLP on shared frozen splits and BC settings.

Example (from the repository root):
    python LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task/tcn/train_tcn.py --smoke-test

Smoke runs use 8 training and 4 validation trajectories for 3 epochs by default.
They verify the pipeline; their losses are NOT full-set experimental results.
The test split is never loaded by this script.

--stop-weight 2 adds a stopped-phase wheel error term. The default 0 retains
the original TCN objective. It changes training only: no forced parking or
action substitution is introduced at inference. Sampling, architecture,
normalization, seeds and checkpoint selection by plain validation MSE stay
unchanged so the extra loss can be studied as a controlled ablation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tcn_dataset import DEFAULT_ROOT, TASK_DIR, TCNTrajectoryDataset
from tcn_model import TCNPolicy
from mlp_model import MLPPolicy


def compute_stats(arrays):
    """Merge float64 population moments; count each sampled frame only once.

    Do not compute moments from overlapping/padded windows, which would
    repeatedly count observations and overweight episode-start padding.
    """
    count = 0
    mean = np.zeros(arrays[0].shape[1], dtype=np.float64)
    m2 = np.zeros_like(mean)
    for array in arrays:
        n = len(array)
        batch_mean = array.mean(axis=0, dtype=np.float64)
        batch_m2 = array.var(axis=0, dtype=np.float64) * n
        delta = batch_mean - mean
        total = count + n
        m2 += batch_m2 + delta ** 2 * count * n / total
        mean += delta * n / total
        count = total
    std = np.sqrt(np.maximum(m2 / count, 0.0))
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class NormalizedWindows(Dataset):
    """Normalize each compact trajectory once using TRAINING statistics."""

    def __init__(self, source, stats):
        self.source = source
        self.features = [
            np.ascontiguousarray((a - stats["x_mean"]) / stats["x_std"], dtype=np.float32)
            for a in source.features
        ]
        self.actions = [
            np.ascontiguousarray((a - stats["y_mean"]) / stats["y_std"], dtype=np.float32)
            for a in source.actions
        ]
        if not all(np.isfinite(a).all() for a in self.features + self.actions):
            raise ValueError("Normalization produced non-finite values")
        self.offsets = np.arange(source.window_size - 1, -1, -1)
        # Read phase from ORIGINAL features, not normalized feature column 22.
        self.stopped = [a[:, 22] == 0.0 for a in source.features]
        self.stop_fraction = sum(int(a.sum()) for a in self.stopped) / len(source)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        trajectory, frame = self.source.locate(index)
        indices = np.maximum(frame - self.offsets, 0)
        return (
            torch.from_numpy(self.features[trajectory][indices]),
            torch.from_numpy(self.actions[trajectory][frame].copy()),
            torch.tensor(bool(self.stopped[trajectory][frame]), dtype=torch.bool),
        )


def compute_bc_loss(prediction, target, stopped, stop_fraction, stop_weight):
    """Original standardized MSE plus optional conditional stopped-wheel MSE.

    Multiplying by 4/11 gives the added term the same wheel-group scale as
    the original all-coordinate MSE. The conditional term is normalized by
    the dataset stop probability, providing an unbiased mini-batch estimate
    under the unchanged uniform sampler, including batches with no stops.
    Compare with expert labels in normalized space, not literal zero: a
    physical zero wheel speed usually has a nonzero normalized target.
    """
    squared = (prediction - target).square()
    loss = squared.mean()
    if stop_weight > 0 and stop_fraction > 0:
        stop_loss = squared[stopped, :4].sum() / (len(target) * 4 * stop_fraction)
        loss = loss + stop_weight * (4.0 / 11.0) * stop_loss
    return loss, squared


def run_epoch(model, loader, device, optimizer=None, stop_weight=0.0):
    """Report frame-weighted MSE, with separate wheel/arm diagnostics.

    Optional stopped-wheel loss is reported separately from the plain MSE.
    These are offline action errors, not closed-loop reaching success rates.
    """
    model.train(optimizer is not None)
    total = 0
    wheel_sse = arm_sse = stop_sse = 0.0
    stop_count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y, stopped in loader:
            x, y, stopped = x.to(device), y.to(device), stopped.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss, squared = compute_bc_loss(
                prediction, y, stopped, loader.dataset.stop_fraction, stop_weight
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite training/validation loss")
            if optimizer is not None:
                loss.backward()
                nnorm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0, error_if_nonfinite=True
                )
                if not torch.isfinite(nnorm).item():
                    raise FloatingPointError("Non-finite gradient norm")
                optimizer.step()
            error = squared.detach()
            wheel_sse += error[:, :4].sum().item()
            arm_sse += error[:, 4:].sum().item()
            stop_sse += error[stopped, :4].sum().item()
            stop_count += int(stopped.sum().item())
            total += len(y)
    mse = (wheel_sse + arm_sse) / (total * 11)
    stop_mse = stop_sse / (stop_count * 4) if stop_count else 0.0
    return {
        "mse": mse,
        "loss": mse + stop_weight * (4.0 / 11.0) * stop_mse,
        "wheel_mse": wheel_sse / (total * 4),
        "arm_mse": arm_sse / (total * 7),
        "stop_wheel_mse": stop_mse,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model", choices=("tcn", "mlp"), default="tcn",
                        help="Shared training settings; MLP uses only the current state")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stop-weight", type=float, default=0.0,
                        help="Extra stopped-wheel loss weight; 0 reproduces the original objective")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    epochs = args.epochs if args.epochs is not None else (3 if args.smoke_test else 150)
    if min(epochs, args.batch_size, args.window_size, args.stride, args.patience) < 1:
        parser.error("epochs, batch size, window size, stride and patience must be positive")
    if not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("lr must be positive and weight decay must be nonnegative")
    if not np.isfinite(args.stop_weight) or args.stop_weight < 0:
        parser.error("stop weight must be finite and nonnegative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Seeds improve repeatability, but MPS bit-for-bit determinism is not promised.
    selected = args.device
    if selected == "auto":
        selected = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(selected)
    root = args.root.resolve()
    print(f"Mode: {'SMOKE TEST (subset only)' if args.smoke_test else 'FULL'}; model: {args.model}; device: {device}", flush=True)
    print("Loading training trajectories...", flush=True)
    train_raw = TCNTrajectoryDataset(
        root, "train", args.window_size, args.stride,
        max_trajectories=8 if args.smoke_test else None,
    )
    print("Loading validation trajectories...", flush=True)
    val_raw = TCNTrajectoryDataset(
        root, "val", args.window_size, args.stride,
        max_trajectories=4 if args.smoke_test else None,
    )
    x_mean, x_std = compute_stats(train_raw.features)
    y_mean, y_std = compute_stats(train_raw.actions)
    stats = dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    train_set, val_set = NormalizedWindows(train_raw, stats), NormalizedWindows(val_raw, stats)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    if args.model == "tcn":
        architecture = dict(input_dim=26, channels=[64, 64, 128], kernel_size=3, dropout=0.1)
        policy_class, model_source = TCNPolicy, "tcn_model.py"
    else:
        architecture = dict(input_dim=26, hidden_dims=[256, 256, 176], dropout=0.1)
        policy_class, model_source = MLPPolicy, "mlp_model.py"
    model = policy_class(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    suffix = "smoke" if args.smoke_test else "full"
    result_folder = "bc_training_tcn_ring" if args.model == "tcn" else "bc_training_mlp_control_ring"
    run_dir = root / "RoboEnv" / result_folder / (
        f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}_{suffix}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "schema_version": 1, "mode": suffix, "seed": args.seed,
        "device": str(device), "architecture": architecture,
        "model_type": policy_class.__name__,
        "temporal_context": "strided_history" if args.model == "tcn" else "current_state_only",
        "effective_context_samples": args.window_size if args.model == "tcn" else 1,
        "comparison_note": "MLP vs TCN changes temporal context and encoder family; not equal-history architecture-only isolation.",
        "window_size": args.window_size, "stride": args.stride,
        "raw_dt_s": train_raw.raw_dt, "history_interval_s": args.stride * train_raw.raw_dt,
        "history_span_s": (args.window_size - 1) * args.stride * train_raw.raw_dt,
        "start_padding": "repeat_first_observation", "feature_version": train_raw.feature_version,
        "action_label_key": "actions", "action_order": "4 wheel speeds then 7 arm torques",
        "training_variant": "stop_weighted_bc" if args.stop_weight > 0 else "standard_bc",
        "loss": "MSE over 11 standardized coordinates + stop_weight*(4/11)*conditional stopped-wheel MSE",
        "stop_weight": args.stop_weight, "stop_phase": "raw base_active column 22 equals 0",
        "checkpoint_selection": "minimum plain validation MSE (unchanged)",
        "sampling": "uniform sampled frames; no phase balancing or low-height resampling",
        "train_stop_fraction": train_set.stop_fraction, "val_stop_fraction": val_set.stop_fraction,
        "normalization": "training sampled frames only, counted once before history padding",
        "max_epochs": epochs, "batch_size": args.batch_size,
        "learning_rate": args.lr, "weight_decay": args.weight_decay,
        "gradient_clip_norm": 1.0, "patience": args.patience,
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_trajectories": len(train_raw.filenames), "val_trajectories": len(val_raw.filenames),
        "train_windows": len(train_set), "val_windows": len(val_set),
        "test_evaluated": False, "timestamps_in_run_name": "UTC",
        "source_manifest_sha256": hashlib.sha256(train_raw.manifest_path.read_bytes()).hexdigest(),
        "software": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__},
    }
    source_paths = [Path(__file__), Path(__file__).with_name("tcn_model.py"), Path(__file__).with_name("mlp_model.py"),
                    Path(__file__).with_name("tcn_dataset.py"), TASK_DIR / "robotnik_bodyframe_features.py"]
    config["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    write_json(run_dir / "config.json", config)
    np.savez(run_dir / "normalization.npz", **stats)
    shutil.copy2(train_raw.manifest_path, run_dir / "data_split.json")
    write_json(run_dir / "used_trajectories.json", {"train": train_raw.filenames, "val": val_raw.filenames, "test": []})
    for source in source_paths:
        shutil.copy2(source, run_dir / source.name)

    print(f"Train: {len(train_raw.filenames)} trajectories / {len(train_set)} windows", flush=True)
    print(f"Val: {len(val_raw.filenames)} trajectories / {len(val_set)} windows; test: not loaded", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Model type: {policy_class.__name__}; parameters: {config['parameters']:,}; context samples: {config['effective_context_samples']}", flush=True)
    print(f"Stop weight: {args.stop_weight:g}; train stopped fraction: {train_set.stop_fraction:.2%}", flush=True)
    print("Inference unchanged: all wheel speeds remain network outputs.", flush=True)
    initial = run_epoch(model, val_loader, device, stop_weight=args.stop_weight)
    write_json(run_dir / "initial_validation.json", initial)
    print(f"Initial val MSE: {initial['mse']:.6f}", flush=True)
    best_mse, best_epoch, stale = float("inf"), 0, 0
    fields = ["epoch", "train_mse", "train_loss", "train_wheel_mse", "train_arm_mse", "train_stop_wheel_mse",
              "val_mse", "val_loss", "val_wheel_mse", "val_arm_mse", "val_stop_wheel_mse", "seconds"]
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            train_metrics = run_epoch(model, train_loader, device, optimizer, args.stop_weight)
            val_metrics = run_epoch(model, val_loader, device, stop_weight=args.stop_weight)
            row = {"epoch": epoch, "seconds": time.perf_counter() - started}
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            writer.writerow(row)
            stream.flush()
            print(f"Epoch {epoch:03d}/{epochs}: train={train_metrics['mse']:.6f} "
                  f"val={val_metrics['mse']:.6f} "
                  f"val_stop_wheel={val_metrics['stop_wheel_mse']:.6f} ({row['seconds']:.1f}s)", flush=True)
            if val_metrics["mse"] < best_mse:
                best_mse, best_epoch, stale = val_metrics["mse"], epoch, 0
                # Tensor-only state/normalization plus plain metadata permits
                # safe loading with torch.load(..., weights_only=True).
                torch.save({
                    "schema_version": 1, "model_type": policy_class.__name__, "config": config,
                    "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "normalization": {k: torch.from_numpy(v.copy()) for k, v in stats.items()},
                    "epoch": epoch, "validation": val_metrics,
                }, run_dir / "best_model.pt")
            else:
                stale += 1
            if stale >= args.patience:
                print("Early stopping on validation MSE.", flush=True)
                break

    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    restored = policy_class(**checkpoint["config"]["architecture"]).eval()
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    for key in stats:
        np.testing.assert_array_equal(checkpoint["normalization"][key].numpy(), stats[key])
    with torch.no_grad():
        check = restored(val_set[0][0].unsqueeze(0))
    if check.shape != (1, 11) or not torch.isfinite(check).all():
        raise RuntimeError("Checkpoint reload failed")
    write_json(run_dir / "training_finished.json", {
        "status": "completed", "mode": suffix, "epochs_completed": epoch,
        "model_type": policy_class.__name__,
        "best_epoch": best_epoch, "best_val_mse": best_mse,
        "best_val_stop_wheel_mse": checkpoint["validation"]["stop_wheel_mse"],
        "stop_weight": args.stop_weight, "checkpoint_selection": "plain_validation_mse",
        "initial_val_mse": initial["mse"], "checkpoint_reload_passed": True,
        "test_evaluated": False, "closed_loop_evaluated": False,
        "subset_only": args.smoke_test,
    })
    print(f"Best epoch: {best_epoch}; val MSE: {best_mse:.6f}", flush=True)
    print("PASS: checkpoint reload and normalization consistency", flush=True)
    print("SMOKE TEST COMPLETE: subset losses only, not final model results."
          if args.smoke_test else "TRAINING COMPLETE: closed-loop evaluation remains.", flush=True)


if __name__ == "__main__":
    main()
