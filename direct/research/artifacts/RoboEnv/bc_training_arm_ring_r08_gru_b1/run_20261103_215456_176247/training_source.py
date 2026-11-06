"""[B1 resume change] Continue a saved B1 model; never restart from parent MLP.

Default: best epoch55 -> total epoch100; patience15. Original B1 checkpoints did not save
Adam state, so their first continuation resets Adam and explicitly records it.
New best/last checkpoints save optimizer and RNG/sampler state for future use.
Training only: evaluate separately with the existing B1 parallel batch script.
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

from robotnik_arm_gru_policy import (RUN_ROOT, FROZEN_FILES, sha256,
                                     resolve_finetuned_pair, load_policy_from_checkpoint)
from robotnik_bodyframe_features import build_bodyframe_observations
from train_arm_bc_ring_gru import WindowStore, select_files, run_epoch


def episode_length(path):
    """[Resume / RAM change] Read only the array header before preallocation."""
    with zipfile.ZipFile(path) as archive, archive.open("observations.npy") as stream:
        version = np.lib.format.read_magic(stream)
        reader = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}.get(version)
        if reader is None:
            raise ValueError(f"Unsupported NPY version: {version}")
        shape, _, _ = reader(stream)
    if len(shape) != 2 or shape[1] != 23 or shape[0] < 2:
        raise ValueError(f"Invalid episode shape: {path}")
    return int(shape[0])


def load_preallocated(manifest, split, stats, limit=None):
    """[Resume / RAM change] Same normalized frames/boundaries, no list+concat peak.

    This avoids keeping two copies of the million-frame arrays in 16GB RAM.
    It changes allocation only, not the original training samples or labels.
    """
    files = select_files(manifest, split, limit)
    lengths = np.array([episode_length(Path(manifest["data_dir"]) / f) for f, _ in files], dtype=np.int64)
    total = int(lengths.sum())
    xs = np.empty((total, 26), dtype=np.float32)
    ys = np.empty((total, 7), dtype=np.float32)
    groups = np.empty(total, dtype=bool)
    offset = 0
    sample_dt = None
    for index, ((filename, target), n) in enumerate(zip(files, lengths), 1):
        with np.load(Path(manifest["data_dir"]) / filename, allow_pickle=False) as data:
            obs = data["observations"].astype(np.float32)
            actions = data["actions"]
            if (obs.shape != (n, 23) or actions.shape != (n, 11)
                    or not np.isfinite(actions).all() or not np.isin(obs[:, 22], [0, 1]).all()):
                raise ValueError(f"Wrong observation/action schema: {filename}")
            np.testing.assert_allclose(data["target"], target, atol=1e-6, rtol=0)
            times = data["timestamps"].astype(np.float64)
            if times.shape != (n,) or not np.isfinite(times).all():
                raise ValueError(f"Invalid timestamps: {filename}")
            dt = float(np.median(np.diff(times)))
            if dt <= 0 or not np.allclose(np.diff(times), dt, atol=1e-9, rtol=0):
                raise ValueError("History requires a constant positive timestep")
            sample_dt = dt if sample_dt is None else sample_dt
            if not np.isclose(dt, sample_dt, atol=1e-9, rtol=0):
                raise ValueError("Mixed episode sampling periods")
            x = build_bodyframe_observations(obs, data["ee_positions"])
            xs[offset:offset+n] = (x - stats["x_mean"]) / stats["x_std"]
            ys[offset:offset+n] = (actions[:, 4:11].astype(np.float32) - stats["y_mean"][4:11]) / stats["y_std"][4:11]
            groups[offset:offset+n] = obs[:, 21] <= .35 + 1e-6
        offset += int(n)
        if index % 100 == 0 or index == len(files):
            print(f"{split}: {index}/{len(files)} episodes loaded into preallocated arrays", flush=True)
    if not groups.any() or groups.all():
        raise ValueError("Both height groups are required")
    return xs, ys, groups, lengths, float(sample_dt)


def main():
    roboenv = Path(__file__).resolve().parents[3] / "RoboEnv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-run-dir", type=Path, default=roboenv / RUN_ROOT / "run_20261103_214030_030129")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--epochs", "--total-epochs", dest="total_epochs", type=int, default=100,
                        help="Total epoch number, NOT additional epochs: best epoch55 -> epoch56..100")
    # [Resume100 change] 显式覆盖旧检查点的patience=8，按本次要求使用15。
    parser.add_argument("--patience", type=int, default=15, help="Stop after this many consecutive non-improving validation epochs")
    parser.add_argument("--dry-run", action="store_true", help="Check checkpoint/settings only; do not load datasets or train")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=None, help="Default: inherit source batch size")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-train-files", type=int, help="Program smoke check only, at least4")
    parser.add_argument("--max-val-files", type=int, help="Program smoke check only, at least4")
    args = parser.parse_args()
    if any(v is not None and v < 4 for v in [args.max_train_files, args.max_val_files]):
        parser.error("Smoke file limits must be at least4")
    if args.patience < 1:
        parser.error("patience must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    source, nav, source_config, _ = resolve_finetuned_pair(roboenv, args.resume_run_dir)
    checkpoint_file = source / f"{args.checkpoint}_model.pt"
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    if checkpoint["config"] != source_config:
        raise ValueError("Resume checkpoint/config mismatch")
    start_epoch = int(checkpoint["epoch"])
    if args.total_epochs <= start_epoch:
        parser.error(f"Total epochs must be greater than source epoch {start_epoch}")
    # [Resume100 change] 只读核验入口；到这里尚未初始化模型/GPU或创建结果目录。
    if args.dry_run:
        print(json.dumps({"checkpoint":str(checkpoint_file),"checkpoint_epoch":start_epoch,
            "first_new_epoch":start_epoch+1,"max_epoch":args.total_epochs,
            "max_additional_epochs":args.total_epochs-start_epoch,"patience":args.patience,
            "optimizer_state_available":"optimizer_state_dict" in checkpoint,
            "sampler_state_available":"sampling_generator_state" in checkpoint,
            "batch_size":args.batch_size or source_config["batch_size"],
            "learning_rate":source_config["learning_rate"],"training_started":False},indent=2))
        return
    smoke = bool(source_config.get("smoke_only", False)) or args.max_train_files is not None or args.max_val_files is not None
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    seed = int(source_config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    else:
        torch.set_num_threads(min(4, torch.get_num_threads()))
    model = load_policy_from_checkpoint(checkpoint).to(device)
    learning_rate = float(source_config["learning_rate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_restored = "optimizer_state_dict" in checkpoint
    if optimizer_restored:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_mode = "restored_from_checkpoint"
    else:
        optimizer_mode = "reset_adam_source_did_not_save_state" if not smoke else "smoke_reset_adam"
    print(f"B1 resume: device={device}; source epoch={start_epoch}; target total={args.total_epochs}", flush=True)
    print(f"Loaded ALL trained MLP+GRU weights from {checkpoint_file}", flush=True)
    print(f"Optimizer: {optimizer_mode}; learning rate={learning_rate}", flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    with np.load(source / "normalization.npz", allow_pickle=False) as data:
        stats = {k: data[k].copy() for k in ["x_mean", "x_std", "y_mean", "y_std"]}
    manifest = json.loads((source / "data_split.json").read_text(encoding="utf-8"))
    x, y, low, lengths, dt = load_preallocated(manifest, "train", stats, args.max_train_files)
    if not np.isclose(dt, source_config["sample_dt_seconds"], atol=1e-9, rtol=0):
        raise ValueError("Resume training timestep differs from source")
    weights = torch.from_numpy(np.where(low, .5 / low.sum(), .5 / (~low).sum()))
    train_samples = len(x)
    train = WindowStore(x, y, low, lengths, device, source_config["history_length"], source_config["history_stride_steps"])
    del x, y, low
    xv, yv, lowv, lengthsv, dtv = load_preallocated(manifest, "val", stats, args.max_val_files)
    if not np.isclose(dt, dtv, atol=1e-9, rtol=0):
        raise ValueError("Train/val timestep mismatch")
    val_samples = len(xv)
    val = WindowStore(xv, yv, lowv, lengthsv, device, source_config["history_length"], source_config["history_stride_steps"])
    del xv, yv, lowv
    batch_size = args.batch_size or int(source_config["batch_size"])
    samples_per_epoch = min(int(source_config["samples_per_epoch"]), train_samples)
    sampling_generator = torch.Generator().manual_seed(seed)
    if not smoke and "sampling_generator_state" in checkpoint:
        sampling_generator.set_state(checkpoint["sampling_generator_state"])
        sampler_mode = "restored_from_checkpoint"
    elif not smoke:
        # [Resume] Reconstruct the original dedicated sampler's exact position.
        if train_samples != source_config["train_samples"]:
            raise ValueError("Cannot replay original sampler with different training frames")
        print(f"Replaying sampler RNG for {start_epoch} prior epochs (no training)...", flush=True)
        for _ in range(start_epoch):
            torch.multinomial(weights, samples_per_epoch, replacement=True, generator=sampling_generator)
        sampler_mode = "replayed_original_sampler_draws"
    else:
        sampler_mode = "smoke_seed_restart_on_subset"
    if optimizer_restored:
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    source_fingerprints = {n: sha256(source / n) for n in FROZEN_FILES}
    source_fingerprints[checkpoint_file.name] = sha256(checkpoint_file)
    nav_fingerprints = {n: sha256(nav / n) for n in FROZEN_FILES}
    output_root = RUN_ROOT if source_config["history_length"] > 1 else RUN_ROOT + "_current_only"
    run = (args.output_dir or roboenv / output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")).resolve()
    run.mkdir(parents=True, exist_ok=False)
    config = dict(source_config)
    config.update({"resumed_training": True, "resumed_from_run_dir": str(source),
        "resumed_from_checkpoint": checkpoint_file.name, "resumed_from_epoch": start_epoch,
        "resumed_source_file_sha256": source_fingerprints,
        "resume_initialization": "all_trained_B1_MLP_and_GRU_weights_loaded_strictly",
        "optimizer_state_restored": optimizer_restored, "optimizer_resume_mode": optimizer_mode,
        "sampler_resume_mode": sampler_mode, "future_checkpoints_include_optimizer_state": True,
        "early_stopping_counter_reset_on_resume": True, "max_epochs": args.total_epochs,
        "patience":args.patience,  # [Resume100 change] 写入实际使用的新早停设置。
        "additional_epochs_allowed": args.total_epochs - start_epoch,
        "train_trajectories": len(lengths), "val_trajectories": len(lengthsv),
        "train_samples": train_samples, "val_samples": val_samples, "samples_per_epoch": samples_per_epoch,
        "batch_size": batch_size, "device": str(device), "torch_version": str(torch.__version__),
        "source_split_path": str(source / "data_split.json"), "source_split_sha256": sha256(source / "data_split.json"),
        "data_loading_allocation": "preallocated_frame_arrays_same_samples",
        "training_helper_sha256": sha256(Path(__file__).with_name("train_arm_bc_ring_gru.py")),
        "smoke_only": smoke})
    (run / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ["data_split.json", "normalization.npz"]:
        shutil.copyfile(source / name, run / name)
    shutil.copyfile(Path(__file__), run / "training_source.py")
    shutil.copyfile(Path(__file__).with_name("train_arm_bc_ring_gru.py"), run / "initial_training_helper_source.py")
    shutil.copyfile(Path(__file__).with_name("robotnik_arm_gru_policy.py"), run / "gru_policy_source.py")
    shutil.copyfile(source / "history.csv", run / "previous_history.csv")
    validation_indices = torch.arange(val_samples, device=device)
    initial = run_epoch(model, val, validation_indices, batch_size)
    # [Resume] A full-data reload must reproduce the saved source metric.
    if not smoke and not np.isclose(initial["balanced_arm_mse"], checkpoint["val_loss"], atol=1e-7, rtol=1e-5):
        raise ValueError("Reloaded validation metric differs from source; stop before training")
    best, best_epoch, stale = initial["balanced_arm_mse"], start_epoch, 0
    print(f"Output: {run}\nInitial reloaded validation: {initial}", flush=True)
    print(f"Resume epochs: {start_epoch+1}..{args.total_epochs}; sampler={sampler_mode}", flush=True)

    def save_checkpoint(path, epoch, metrics):
        state = {"model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(), "sampling_generator_state": sampling_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(), "epoch": epoch,
            "val_mse": metrics["arm_mse"], "val_loss": metrics["balanced_arm_mse"],
            "val_metrics": metrics, "config": config}
        if device.type == "cuda":
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        torch.save(state, path)

    save_checkpoint(run / "best_model.pt", start_epoch, initial)
    save_checkpoint(run / "last_model.pt", start_epoch, initial)
    (run / "initial_validation.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
    metrics = ["arm_mse", "low_arm_mse", "other_arm_mse", "balanced_arm_mse"]
    with (run / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "seconds"] + [s + "_" + m for s in ["train", "val"] for m in metrics])
        for epoch in range(start_epoch + 1, args.total_epochs + 1):
            if any(sha256(source / n) != expected for n, expected in source_fingerprints.items()):
                raise RuntimeError("Original B1 run changed during continuation")
            started = time.perf_counter()
            indices = torch.multinomial(weights, samples_per_epoch, replacement=True, generator=sampling_generator).to(device)
            tr = run_epoch(model, train, indices, batch_size, optimizer)
            va = run_epoch(model, val, validation_indices, batch_size)
            writer.writerow([epoch, time.perf_counter() - started] + [v[m] for v in [tr, va] for m in metrics])
            stream.flush()
            print(f"Epoch {epoch:03d} | train arm MSE {tr['arm_mse']:.6f} | val arm MSE {va['arm_mse']:.6f} | low {va['low_arm_mse']:.6f} | other {va['other_arm_mse']:.6f} | balanced {va['balanced_arm_mse']:.6f}", flush=True)
            save_checkpoint(run / "last_model.pt", epoch, va)
            if va["balanced_arm_mse"] < best:
                best, best_epoch, stale = va["balanced_arm_mse"], epoch, 0
                save_checkpoint(run / "best_model.pt", epoch, va)
            else:
                stale += 1
            if stale >= args.patience:  # [Resume100 change] 连续15轮未改善才早停。
                print("Early stopping: balanced validation arm MSE did not improve.", flush=True)
                break
    for folder, fingerprints in [(source, source_fingerprints), (nav, nav_fingerprints)]:
        if any(sha256(folder / n) != expected for n, expected in fingerprints.items()):
            raise RuntimeError("Frozen source/navigation changed during resume")
    (run / "training_finished.json").write_text(json.dumps({"best_epoch": best_epoch, "last_epoch": epoch,
        "resume_start_epoch": start_epoch, "best_balanced_validation_arm_mse": best,
        "patience":args.patience,"max_epochs":args.total_epochs,
        "optimizer_state_restored": optimizer_restored, "navigation_unchanged": True,
        "smoke_only": smoke}, indent=2), encoding="utf-8")
    print(f"Best continued B1 epoch: {best_epoch}; balanced validation arm MSE: {best:.6f}\nResults: {run}", flush=True)
    print("Training only. Run batch_test_bc_split_nav_arm_gru_parallel.py separately.", flush=True)


if __name__ == "__main__":
    main()
