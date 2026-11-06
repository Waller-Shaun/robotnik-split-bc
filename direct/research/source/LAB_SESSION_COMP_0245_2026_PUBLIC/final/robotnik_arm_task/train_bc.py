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
SPLIT_PATH = ROBOENV_DIR / "bc_training" / "data_split.json"

SEED = 42
BATCH_SIZE = 4096
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 15  # 连续多少轮验证损失未改善时停止


class BCPolicy(nn.Module):
    """输入当前状态和目标，输出车轮速度及关节力矩。"""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(22, 128),
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
            observations.append(
                data["observations"].astype(np.float32)
            )
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


def run_epoch(model, loader, device, optimizer=None):
    """有 optimizer 时训练，否则只评估。"""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            prediction = model(x)
            loss = nn.functional.mse_loss(prediction, y)

            if training:
                loss.backward()
                optimizer.step()

            # 按样本数加权，避免最后一个小批次影响平均值
            total_loss += loss.detach().item() * len(x)
            total_samples += len(x)

    return total_loss / total_samples


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

    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    data_dir = Path(manifest["data_dir"])

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
    run_dir = ROBOENV_DIR / "bc_training" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    # 保存本次使用的数据划分，便于复现
    (run_dir / "data_split.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    config = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "input_dim": 22,
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
        writer.writerow(["epoch", "train_mse", "val_mse"])

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = run_epoch(
                model, train_loader, device, optimizer
            )
            val_loss = run_epoch(
                model, val_loader, device
            )

            if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                raise RuntimeError("损失出现 NaN 或无穷值，训练已停止。")

            writer.writerow([epoch, train_loss, val_loss])
            file.flush()

            print(
                f"Epoch {epoch:03d} | "
                f"train MSE: {train_loss:.6f} | "
                f"val MSE: {val_loss:.6f}",
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
                        "val_mse": val_loss,
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
    print(f"最佳验证 MSE：{best_val_loss:.6f}")
    print(f"结果目录：{run_dir}")


if __name__ == "__main__":
    main()