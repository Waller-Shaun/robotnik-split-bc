# [Wide finetune change] 从三隐藏层网络第135轮的最佳模型继续微调。
# 网络256/256/128、数据划分、采样倍率2、损失权重1:1:2均保持不变。
# 仅降低学习率到1e-5；沿用旧标准化；原模型记为epoch 0保护基线。
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler  # [Resample change]


# ---------- 路径和超参数 ----------
ROBOENV_DIR = Path(__file__).resolve().parents[3] / "RoboEnv"
# [23D change] Read the separate split manifest for 23D trajectories.
# [Noise dataset change] Dedicated manifest/results; preserve the clean baseline.
DATASET_DIR = ROBOENV_DIR / "expert_data_23d_noise"
# [Resample change] Reuse the fixed noise-data split, but isolate new results.
# [Wide finetune change] 固定来源三层模型；所有新结果写入独立目录。
PRETRAINED_RUN_DIR = ROBOENV_DIR / "bc_training_23d_noise_wide" / "run_20261030_112637_239667"
TRAINING_DIR = ROBOENV_DIR / "bc_training_23d_noise_wide_finetune"
SPLIT_PATH = PRETRAINED_RUN_DIR / "data_split.json"

SEED = 42
BATCH_SIZE = 4096
MAX_EPOCHS = 200
LEARNING_RATE = 1e-5  # [Wide finetune change] 原三层模型学习率的十分之一。
# [Weighted loss change] Training emphasis only; no forced parking at inference.
WHEEL_WEIGHT = 1.0
STOP_WEIGHT = 2.0
# [Resample change] Conditions use RAW physical units before normalization.
LOW_TARGET_MAX_Z = 0.4       # m; target z is observation column 21
NEAR_TARGET_MAX_ERROR = 0.05 # m; stored EE distance error
FOCUS_SAMPLE_WEIGHT = 2.0  # [Wide finetune change] 保持原三层模型的采样倍率。
PATIENCE = 15  # 连续多少轮验证损失未改善时停止


class BCPolicy(nn.Module):
    """输入当前状态和目标，输出车轮速度及关节力矩。"""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # [23D change] Last input is base_active (0 or 1).
            # [Architecture change] 原为两个128隐藏层，现在使用256、256、128。
            nn.Linear(23, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 11),
        )

    def forward(self, x):
        return self.network(x)


# [Resample change] No new policy inputs: EE error only selects training samples.
def build_sampling_weights(observations, ee_errors):
    if ee_errors.shape != (len(observations),):
        raise ValueError("ee_errors must align with observations")
    if not np.isfinite(ee_errors).all() or np.any(ee_errors < 0):
        raise ValueError("ee_errors must be finite and nonnegative")
    if not np.isfinite(FOCUS_SAMPLE_WEIGHT) or FOCUS_SAMPLE_WEIGHT < 1:
        raise ValueError("FOCUS_SAMPLE_WEIGHT must be finite and >= 1")
    focus = (
        (observations[:, 21] <= LOW_TARGET_MAX_Z + 1e-6)
        & (observations[:, 22] == 0.0)
        & (ee_errors <= NEAR_TARGET_MAX_ERROR)
    )
    return np.where(focus, FOCUS_SAMPLE_WEIGHT, 1.0).astype(np.float64)


def load_split(data_dir, filenames, return_sampling_weights=False):
    """将多条轨迹的时间步合并为训练样本。"""
    observations = []
    actions = []
    # [Resample change] Only training requests weights; validation stays natural.
    sampling_weights = []

    for filename in filenames:
        with np.load(data_dir / filename, allow_pickle=False) as data:
            # [23D change] Reject mixed 22D/23D files before training.
            obs = data["observations"].astype(np.float32)
            act = data["actions"]
            if obs.ndim != 2 or obs.shape[1] != 23:
                raise ValueError(f"{filename}: expected observations (N, 23), got {obs.shape}")
            if len(obs) == 0 or act.shape != (len(obs), 11):
                raise ValueError(f"{filename}: empty trajectory or mismatched actions")
            if not np.isin(obs[:, 22], [0.0, 1.0]).all():
                raise ValueError(f"{filename}: base_active must be 0 or 1")
            if return_sampling_weights:
                sampling_weights.append(build_sampling_weights(obs, data["ee_errors"]))
            observations.append(obs)
            # [Noise dataset change] Learn expert labels, not applied_actions.
            actions.append(
                data["actions"].astype(np.float32)
            )

    arrays = (
        np.concatenate(observations, axis=0),
        np.concatenate(actions, axis=0),
    )
    if return_sampling_weights:
        return (*arrays, np.concatenate(sampling_weights))
    return arrays


def compute_stats(array):
    # 用 float64 累计，减少大量样本计算均值、方差时的数值误差
    mean = array.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = array.std(axis=0, dtype=np.float64).astype(np.float32)

    # 常量特征不除以接近零的数
    std[std < 1e-6] = 1.0
    return mean, std


def make_loader(x, y, shuffle, use_cuda, sampling_weights=None):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    sampler = None
    stopped = x[:, 22] == 0.0
    # [Resample change] Compute phase probability under the ACTUAL sampling
    # distribution, so the extra stop-loss term retains its intended scale.
    if sampling_weights is not None:
        weights = np.asarray(sampling_weights, dtype=np.float64)
        if (weights.shape != (len(x),) or not np.isfinite(weights).all()
                or np.any(weights <= 0)):
            raise ValueError("Sampling weights must be positive, finite, and aligned")
        expected_stop_fraction = float(weights[stopped].sum() / weights.sum())
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights),
            num_samples=len(dataset),  # Same number of draws per epoch as baseline.
            replacement=True,
            generator=torch.Generator().manual_seed(SEED),
        )
    else:
        expected_stop_fraction = float(stopped.mean())
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=0,
        pin_memory=use_cuda,
    )
    # Last phase column remains binary after preprocessing.
    loader.expected_stop_fraction = expected_stop_fraction
    return loader


# [Weighted loss change] Inputs/actions are normalized; compare with expert
# labels, NOT literal zero (physical zero speed may normalize to nonzero).
def compute_bc_loss(prediction, target, observation, stop_fraction):
    squared_error = (prediction - target).square()
    wheel = squared_error[:, :4].mean()
    arm = squared_error[:, 4:].mean()
    stopped = observation[:, 22] == 0.0
    # Fixed dataset phase frequency gives an unbiased mini-batch estimate
    # of mean stopped-wheel error, including batches with no stopped samples.
    stop = (
        squared_error[stopped, :4].sum()
        / (len(target) * 4 * stop_fraction)
        if stop_fraction > 0 else prediction.sum() * 0.0
    )
    loss = arm + WHEEL_WEIGHT * wheel + STOP_WEIGHT * stop
    return loss, squared_error, stopped


def run_epoch(model, loader, device, optimizer=None):
    """[Weighted loss change] Optimize weighted loss and report exact epoch metrics."""
    training = optimizer is not None
    model.train(training)
    # [Resample change] Weighted probability for training, natural for validation.
    # Do not use the raw dataset phase ratio after changing sampling frequency.
    stop_fraction = loader.expected_stop_fraction
    wheel_sum = arm_sum = stop_sum = 0.0
    total_samples = stopped_samples = 0

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss, squared_error, stopped = compute_bc_loss(
                prediction, y, x, stop_fraction
            )
            if training:
                loss.backward()
                optimizer.step()

            error = squared_error.detach()
            wheel_sum += error[:, :4].sum().item()
            arm_sum += error[:, 4:].sum().item()
            stop_sum += error[stopped, :4].sum().item()
            total_samples += len(x)
            stopped_samples += int(stopped.sum().item())

    # Aggregate sums/counts across the WHOLE epoch. Training metrics describe
    # weighted draws; validation metrics still describe every natural sample.
    # Do not compare training MSE across samplers as if distributions were equal.
    wheel_mse = wheel_sum / (total_samples * 4)
    arm_mse = arm_sum / (total_samples * 7)
    stop_wheel_mse = stop_sum / (stopped_samples * 4) if stopped_samples else 0.0
    return {
        "loss": arm_mse + WHEEL_WEIGHT * wheel_mse + STOP_WEIGHT * stop_wheel_mse,
        "mse": (wheel_sum + arm_sum) / (total_samples * 11),
        "wheel_mse": wheel_mse,
        "arm_mse": arm_mse,
        "stop_wheel_mse": stop_wheel_mse,
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        torch.cuda.manual_seed_all(SEED)

    print(f"训练设备：{device}", flush=True)
    if use_cuda:
        print(f"显卡：{torch.cuda.get_device_name(0)}", flush=True)

    # [Wide finetune change] 开始读取大量数据前先检查来源文件。
    for filename in ("best_model.pt", "normalization.npz", "data_split.json"):
        if not (PRETRAINED_RUN_DIR / filename).is_file():
            raise FileNotFoundError(PRETRAINED_RUN_DIR / filename)
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"Run split_expert_data_noise.py first: {SPLIT_PATH}")
    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if manifest.get("input_dim") != 23 or manifest.get("output_dim") != 11:
        raise ValueError("Expected manifest input_dim=23 and output_dim=11")
    data_dir = Path(manifest["data_dir"])
    # [Noise dataset change] Refuse to silently train on the previous dataset.
    if data_dir.resolve() != DATASET_DIR.resolve():
        raise ValueError(f"Expected noise dataset {DATASET_DIR}, got {data_dir}")
    if not manifest["splits"]["train"] or not manifest["splits"]["val"]:
        raise ValueError("Training and validation splits must both be nonempty")

    train_files = manifest["splits"]["train"]
    val_files = manifest["splits"]["val"]

    print("正在读取训练集……", flush=True)
    # [Resample change] Compute priority from raw inputs and aligned EE errors.
    x_train, y_train, train_sampling_weights = load_split(
        data_dir, train_files, return_sampling_weights=True
    )
    boosted = train_sampling_weights > 1.0
    sampling_summary = {
        "boosted_train_samples": int(boosted.sum()),
        "raw_boosted_fraction": float(boosted.mean()),
        "expected_boosted_fraction": float(
            train_sampling_weights[boosted].sum() / train_sampling_weights.sum()
        ),
        "natural_stop_fraction": float((x_train[:, 22] == 0.0).mean()),
        "expected_sampled_stop_fraction": float(
            train_sampling_weights[x_train[:, 22] == 0.0].sum() / train_sampling_weights.sum()
        ),
    }
    print(f"Resampling summary: {sampling_summary}", flush=True)

    print("正在读取验证集……", flush=True)
    x_val, y_val = load_split(data_dir, val_files)

    print(f"训练样本数：{len(x_train):,}", flush=True)
    print(f"验证样本数：{len(x_val):,}", flush=True)

    # ---------- 标准化：只使用训练集统计量 ----------
    # [Wide finetune change] 必须沿用来源模型的标准化，不重新估计或改变输入尺度。
    with np.load(PRETRAINED_RUN_DIR / "normalization.npz", allow_pickle=False) as stats:
        x_mean = stats["x_mean"].copy()
        x_std = stats["x_std"].copy()
        y_mean = stats["y_mean"].copy()
        y_std = stats["y_std"].copy()
    for name, value, expected in (("x_mean", x_mean, (23,)), ("x_std", x_std, (23,)),
                                   ("y_mean", y_mean, (11,)), ("y_std", y_std, (11,))):
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f"Invalid source normalization: {name}")
    if np.any(x_std <= 0) or np.any(y_std <= 0) or x_mean[22] != 0 or x_std[22] != 1:
        raise ValueError("Source normalization must preserve binary phase and positive std")

    # [23D change] Preserve binary phase values through normalization.
    # These values are saved too, so inference uses the same transformation.
    x_mean[22] = 0.0
    x_std[22] = 1.0

    # 原地处理以减少额外内存占用
    x_train -= x_mean
    x_train /= x_std
    x_val -= x_mean
    x_val /= x_std

    y_train -= y_mean
    y_train /= y_std
    y_val -= y_mean
    y_val /= y_std

    train_loader = make_loader(
        # [Resample change] Sampler replaces shuffle; validation has no sampler.
        x_train, y_train, shuffle=False, use_cuda=use_cuda,
        sampling_weights=train_sampling_weights
    )
    val_loader = make_loader(
        x_val, y_val, shuffle=False, use_cuda=use_cuda
    )

    # 每次训练使用新目录，避免覆盖以前的结果
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    # [23D change] Keep the 22D baseline results intact.
    # [Noise dataset change] Each experiment gets a new independent run folder.
    run_dir = TRAINING_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    # 保存本次使用的数据划分，便于复现
    (run_dir / "data_split.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    config = {
        # [Weighted loss change] Log the objective for reproducibility.
        # [Wide finetune change] 记录来源，重建Adam状态；不引入自动学习率调度。
        "training_mode": "finetune",
        "experiment_variant": "three_hidden_layers_256_256_128_finetune",
        "source_split_path": str(SPLIT_PATH.resolve()),
        "pretrained_run_dir": str(PRETRAINED_RUN_DIR.resolve()),
        "pretrained_weights_loaded": True,
        "optimizer_state_restored": False,
        "epoch_zero_is_source_model": True,
        "learning_rate_scheduler": None,
        "loss_type": "arm_mse_plus_wheel_mse_plus_stop_wheel_mse",
        "wheel_weight": WHEEL_WEIGHT,
        "stop_weight": STOP_WEIGHT,
        "selection_metric": "val_loss",
        # [Noise dataset change] Keep architecture/loss fixed to isolate data changes.
        "dataset_variant": "config_measurement_noise_plus_hold",
        "data_dir": str(data_dir.resolve()),
        "action_label_key": "actions",
        "history_input": False,
        # [Resample change] Complete sampling specification; data/splits unchanged.
        "sampling_method": "WeightedRandomSampler",
        "sampling_replacement": True,
        "samples_per_epoch": len(x_train),
        "sampling_seed": SEED,
        "low_target_max_z_m": LOW_TARGET_MAX_Z,
        "near_target_max_error_m": NEAR_TARGET_MAX_ERROR,
        "focus_sample_weight": FOCUS_SAMPLE_WEIGHT,
        "focus_phase": "base_active=0",
        "sampling_summary": sampling_summary,
        "normalization_distribution": "reused_from_pretrained_run",
        "training_metrics_distribution": "weighted_samples",
        "validation_metrics_distribution": "unweighted_all_validation_samples",
        "train_trajectories": len(train_files),
        "val_trajectories": len(val_files),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        # [23D change] Document the input schema and phase preprocessing.
        "input_dim": 23,
        "base_active_column": 22,
        "base_active_standardized": False,
        "hidden_dims": [256, 256, 128],  # [Architecture change] 测试时需匹配此结构。
        "output_dim": 11,
        "activation": "ReLU",
        "device": str(device),
        "torch_version": str(torch.__version__),
        "train_samples": len(x_train),
        "val_samples": len(x_val),
    }

    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    np.savez(
        run_dir / "normalization.npz",
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )

    # [Wide finetune change] 严格加载来源三层网络的所有参数，不从随机权重开始。
    checkpoint = torch.load(PRETRAINED_RUN_DIR / "best_model.pt", map_location="cpu", weights_only=True)
    source_config = checkpoint["config"]
    if (source_config.get("hidden_dims") != [256, 256, 128]
            or source_config.get("input_dim") != 23 or source_config.get("output_dim") != 11
            or source_config.get("activation") != "ReLU"):
        raise ValueError("Source model must be the 23 -> 256 -> 256 -> 128 -> 11 ReLU policy")
    # 损失、重点采样条件保持一致，避免把其他改动混入学习率实验。
    expected_settings = {"wheel_weight": WHEEL_WEIGHT, "stop_weight": STOP_WEIGHT,
                         "focus_sample_weight": FOCUS_SAMPLE_WEIGHT,
                         "low_target_max_z_m": LOW_TARGET_MAX_Z,
                         "near_target_max_error_m": NEAR_TARGET_MAX_ERROR}
    for key, value in expected_settings.items():
        if source_config.get(key) != value:
            raise ValueError(f"Source/current setting mismatch: {key}")
    model = BCPolicy().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Source checkpoint contains non-finite parameters")
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Architecture: 23 -> 256 -> 256 -> 128 -> 11; parameters: {parameter_count:,}", flush=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # [Wide finetune change] 先用相同验证集评估原模型，作为epoch 0基线。
    # 即使后续都没有改善，新目录也有完整、可测试的原模型检查点。
    initial_metrics = run_epoch(model, val_loader, device)
    best_val_loss = initial_metrics["loss"]
    if not np.isfinite(best_val_loss):
        raise RuntimeError("Initial validation loss must be finite")
    config["source_checkpoint_epoch"] = checkpoint["epoch"]
    config["initial_validation_metrics"] = initial_metrics
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save({
        "model_state_dict": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
        "epoch": 0, "val_mse": initial_metrics["mse"], "val_loss": best_val_loss,
        "val_metrics": initial_metrics, "config": config,
    }, run_dir / "best_model.pt")
    print(f"Source checkpoint epoch: {checkpoint['epoch']}", flush=True)
    print(f"Epoch 000 | source val loss: {best_val_loss:.6f}", flush=True)
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"输出目录：{run_dir}", flush=True)

    with (run_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        # [Weighted loss change] Keep ordinary MSE for baseline comparison.
        metric_names = ["loss", "mse", "wheel_mse", "arm_mse", "stop_wheel_mse"]
        writer.writerow(
            ["epoch"] + [f"{split}_{key}" for split in ("train", "val") for key in metric_names]
        )

        # [Wide finetune change] 第0轮未训练，训练指标留空，仅记录来源验证表现。
        writer.writerow([0] + [""] * len(metric_names) + [initial_metrics[k] for k in metric_names])
        file.flush()
        for epoch in range(1, MAX_EPOCHS + 1):
            train_metrics = run_epoch(
                model, train_loader, device, optimizer
            )
            val_metrics = run_epoch(
                model, val_loader, device
            )

            # [Weighted loss change] Early stopping selects weighted validation loss.
            train_loss = train_metrics["loss"]
            val_loss = val_metrics["loss"]
            if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                raise RuntimeError("损失出现 NaN 或无穷值，训练已停止。")

            writer.writerow(
                [epoch] + [metrics[key] for metrics in (train_metrics, val_metrics) for key in metric_names]
            )
            file.flush()

            print(
                f"Epoch {epoch:03d} | "
                f"train loss: {train_loss:.6f} | "
                f"val loss: {val_loss:.6f} | "
                f"val MSE: {val_metrics['mse']:.6f} | "
                f"stop wheel MSE: {val_metrics['stop_wheel_mse']:.6f}",
                flush=True,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0

                # 将权重保存到 CPU，后续 CPU/GPU 均可加载
                cpu_state = {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                }

                torch.save(
                    {
                        "model_state_dict": cpu_state,
                        "epoch": epoch,
                        # [Weighted loss change] val_mse remains ordinary MSE
                        # for compatibility with the existing testing script.
                        "val_mse": val_metrics["mse"],
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                        "config": config,
                    },
                    run_dir / "best_model.pt",
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                print("验证损失连续未改善，提前结束训练。", flush=True)
                break

    print(f"\n最佳轮次：{best_epoch}")
    if best_epoch == 0:
        print("Best epoch 0: source model retained; fine-tuning did not improve validation loss.")
    print(f"Best weighted validation loss: {best_val_loss:.6f}")  # [Noise dataset change] Correct metric name.
    print(f"结果目录：{run_dir}")


if __name__ == "__main__":
    main()
