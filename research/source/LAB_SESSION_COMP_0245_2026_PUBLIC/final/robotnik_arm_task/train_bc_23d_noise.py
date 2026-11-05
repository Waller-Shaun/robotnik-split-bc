import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ---------- 路径和超参数 ----------
ROBOENV_DIR = Path(__file__).resolve().parents[3] / "RoboEnv"
# [23D change] Read the separate split manifest for 23D trajectories.
# [Noise dataset change] Dedicated manifest/results; preserve the clean baseline.
DATASET_DIR = ROBOENV_DIR / "expert_data_23d_noise"
TRAINING_DIR = ROBOENV_DIR / "bc_training_23d_noise"
SPLIT_PATH = TRAINING_DIR / "data_split.json"

SEED = 42
BATCH_SIZE = 4096
MAX_EPOCHS = 200
LEARNING_RATE = 1e-3
# [Weighted loss change] Training emphasis only; no forced parking at inference.
WHEEL_WEIGHT = 1.0
STOP_WEIGHT = 2.0
PATIENCE = 15  # 连续多少轮验证损失未改善时停止


class BCPolicy(nn.Module):
    """输入当前状态和目标，输出车轮速度及关节力矩。"""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # [23D change] Last input is base_active (0 or 1).
            nn.Linear(23, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 11),
        )

    def forward(self, x):
        return self.network(x)


def load_split(data_dir, filenames):
    """将多条轨迹的时间步合并为训练样本。"""
    observations = []
    actions = []

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
            observations.append(obs)
            # [Noise dataset change] Learn expert labels, not applied_actions.
            actions.append(
                data["actions"].astype(np.float32)
            )

    return (
        np.concatenate(observations, axis=0),
        np.concatenate(actions, axis=0),
    )


def compute_stats(array):
    # 用 float64 累计，减少大量样本计算均值、方差时的数值误差
    mean = array.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = array.std(axis=0, dtype=np.float64).astype(np.float32)

    # 常量特征不除以接近零的数
    std[std < 1e-6] = 1.0
    return mean, std


def make_loader(x, y, shuffle, use_cuda):
    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,  # Windows 下先使用最简单可靠的方式
        pin_memory=use_cuda,
    )


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
    phase = loader.dataset.tensors[0][:, 22]
    stop_fraction = float((phase == 0.0).float().mean())
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

    # Aggregate sums/counts across the WHOLE epoch, so validation metrics do
    # not depend on the phase mixture in individual batches.
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

    # [23D change] A new manifest must map the old target groups to new files.
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
    x_train, y_train = load_split(data_dir, train_files)

    print("正在读取验证集……", flush=True)
    x_val, y_val = load_split(data_dir, val_files)

    print(f"训练样本数：{len(x_train):,}", flush=True)
    print(f"验证样本数：{len(x_val):,}", flush=True)

    # ---------- 标准化：只使用训练集统计量 ----------
    x_mean, x_std = compute_stats(x_train)
    y_mean, y_std = compute_stats(y_train)

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
        x_train, y_train, shuffle=True, use_cuda=use_cuda
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
        "loss_type": "arm_mse_plus_wheel_mse_plus_stop_wheel_mse",
        "wheel_weight": WHEEL_WEIGHT,
        "stop_weight": STOP_WEIGHT,
        "selection_metric": "val_loss",
        # [Noise dataset change] Keep architecture/loss fixed to isolate data changes.
        "dataset_variant": "config_measurement_noise_plus_hold",
        "data_dir": str(data_dir.resolve()),
        "action_label_key": "actions",
        "history_input": False,
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
        "hidden_dims": [128, 128],
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

    model = BCPolicy().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")
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
    print(f"Best weighted validation loss: {best_val_loss:.6f}")  # [Noise dataset change] Correct metric name.
    print(f"结果目录：{run_dir}")


if __name__ == "__main__":
    main()
