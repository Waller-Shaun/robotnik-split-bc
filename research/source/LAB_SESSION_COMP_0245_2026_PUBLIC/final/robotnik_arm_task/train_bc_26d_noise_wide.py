# [26D change] 复制自 train_bc_23d_noise_wide.py；原始数据文件仍是23维，不重写NPZ。
# 读取时追加 target_xyz - ee_positions，得到26维输入；输出仍为11维。
# 保留1014目标划分、256/256/128隐藏层、损失、采样和超参数，单独保存结果。
# 从随机权重开始训练，不加载23维旧网络；没有增加历史输入或强制停车逻辑。
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
# [Extended dataset change] 切换到1014目标的独立清单，保留旧实验的冻结清单。
SOURCE_SPLIT_DIR = ROBOENV_DIR / "bc_training_23d_noise_extended"
SPLIT_PATH = SOURCE_SPLIT_DIR / "data_split.json"
# [26D change] 独立结果目录，避免与23维基线混放。
TRAINING_DIR = ROBOENV_DIR / "bc_training_26d_noise_wide_extended"
SOURCE_INPUT_DIM = 23  # [26D change] NPZ和原始划分清单中的维度不变。
INPUT_DIM = 26         # [26D change] 网络输入：原23维 + 三维末端位置误差。

SEED = 42
BATCH_SIZE = 4096
MAX_EPOCHS = 200
LEARNING_RATE = 1e-4  # [Architecture change] 沿用当前微调版学习率。
# [Weighted loss change] Training emphasis only; no forced parking at inference.
WHEEL_WEIGHT = 1.0
STOP_WEIGHT = 2.0
# [Resample change] Conditions use RAW physical units before normalization.
LOW_TARGET_MAX_Z = 0.4       # m; target z is observation column 21
NEAR_TARGET_MAX_ERROR = 0.05 # m; stored EE distance error
FOCUS_SAMPLE_WEIGHT = 2.0  # [Architecture change] 沿用当前微调版采样倍率。
PATIENCE = 15  # 连续多少轮验证损失未改善时停止


class BCPolicy(nn.Module):
    """输入当前状态和目标，输出车轮速度及关节力矩。"""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # [26D change] base_active仍在第22列，新增误差位于第23、24、25列。
            # [Architecture change] 原为两个128隐藏层，现在使用256、256、128。
            nn.Linear(INPUT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 11),
        )

    def forward(self, x):
        return self.network(x)


# [26D change] 采样仍使用原有标量ee_errors；输入新增的是三维有方向的误差。
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


def build_26d_observations(observations, ee_positions):
    """[26D change] 同一时间步：原23维 + [目标x-末端x, 目标y-末端y, 目标z-末端z]。

    ee_positions来自采集时的正运动学。测试时必须使用相同末端链接和坐标系，
    在应用动作前构造同样的特征；不能用执行动作后的末端位置。
    """
    obs = np.asarray(observations, dtype=np.float32)
    ee = np.asarray(ee_positions, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[1] != SOURCE_INPUT_DIM:
        raise ValueError(f"Expected source observations (N, 23), got {obs.shape}")
    if ee.shape != (len(obs), 3):
        raise ValueError(f"ee_positions must align with observations: got {ee.shape}")
    if not np.isfinite(obs).all() or not np.isfinite(ee).all():
        raise ValueError("Observations and ee_positions must be finite")
    # [26D change] 不改写源数据，保留原23列顺序，误差单位为米。
    ee_error_xyz = (obs[:, 19:22].astype(np.float64) - ee).astype(np.float32)
    result = np.concatenate((obs, ee_error_xyz), axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Derived 26D observations must be finite")
    return result


def load_split(data_dir, filenames, return_sampling_weights=False):
    """将多条轨迹的时间步合并为训练样本。"""
    observations = []
    actions = []
    # [Resample change] Only training requests weights; validation stays natural.
    sampling_weights = []

    for filename in filenames:
        with np.load(data_dir / filename, allow_pickle=False) as data:
            # [26D change] 源文件仍是23维；只在内存中构造26维输入。
            obs = data["observations"].astype(np.float32)
            act = data["actions"]
            if obs.ndim != 2 or obs.shape[1] != SOURCE_INPUT_DIM:
                raise ValueError(f"{filename}: expected observations (N, 23), got {obs.shape}")
            if len(obs) == 0 or act.shape != (len(obs), 11):
                raise ValueError(f"{filename}: empty trajectory or mismatched actions")
            if not np.isin(obs[:, 22], [0.0, 1.0]).all():
                raise ValueError(f"{filename}: base_active must be 0 or 1")
            if not np.isfinite(act).all():
                raise ValueError(f"{filename}: actions must be finite")
            if "ee_positions" not in data:
                raise ValueError(f"{filename}: ee_positions is required for 26D input")
            if return_sampling_weights:
                sampling_weights.append(build_sampling_weights(obs, data["ee_errors"]))
            observations.append(build_26d_observations(obs, data["ee_positions"]))
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
    # [26D change] 第22列阶段标志仍为二值；它不再是最后一列。
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

    # [23D change] A new manifest must map the old target groups to new files.
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"Run split_expert_data_noise_extended.py first: {SPLIT_PATH}")
    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    # [26D change] 清单描述磁盘上的23维文件，不应把原清单改成26维。
    if manifest.get("input_dim") != SOURCE_INPUT_DIM or manifest.get("output_dim") != 11:
        raise ValueError("Expected SOURCE manifest input_dim=23 and output_dim=11")
    # [Extended dataset change] 防止误用旧468目标清单；验证/测试不参与梯度更新。
    expected_counts = {"train": 709, "val": 152, "test": 153}
    actual_counts = {name: len(manifest["targets"][name]) for name in expected_counts}
    if actual_counts != expected_counts:
        raise ValueError(f"Expected extended 1014-target split, got {actual_counts}")
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
    # [26D change] 启动时明确打印派生后的维度，避免误用23维输入。
    print(f"输入维度：{x_train.shape[1]}；输出维度：{y_train.shape[1]}", flush=True)
    print("第23-25列：目标位置 - 当前末端位置，单位m（从0计列号）。", flush=True)

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

    # [26D change] 原样保存源清单：其input_dim=23描述NPZ，网络维度见config.json。
    (run_dir / "data_split.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    config = {
        # [Weighted loss change] Log the objective for reproducibility.
        # [Architecture change] 记录随机初始化，避免误认为是在旧模型上微调。
        "training_mode": "from_scratch",
        "experiment_variant": "26d_ee_error_three_hidden_layers_256_256_128_extended_z",
        "source_split_path": str(SPLIT_PATH.resolve()),
        "split_source_dir": str(SOURCE_SPLIT_DIR.resolve()),
        # [Extended dataset change] 保存目标计数，方便区分旧70与新152验证目标。
        "target_counts": actual_counts,
        "pretrained_weights_loaded": False,
        "loss_type": "arm_mse_plus_wheel_mse_plus_stop_wheel_mse",
        "wheel_weight": WHEEL_WEIGHT,
        "stop_weight": STOP_WEIGHT,
        "selection_metric": "val_loss",
        # [26D change] 数据和隐藏层/损失不变，仅新增三维输入特征。
        "dataset_variant": manifest["dataset_variant"],  # [Extended dataset change]
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
        "normalization_distribution": "unweighted_original_training_set",
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
        "input_dim": INPUT_DIM,
        # [26D change] 明确记录源数据与派生输入的区别，供后续测试构造相同特征。
        "source_input_dim": SOURCE_INPUT_DIM,
        "input_feature_version": "obs23_plus_target_minus_ee_xyz_v1",
        "source_observation_key": "observations",
        "ee_position_key": "ee_positions",
        "target_position_columns": [19, 20, 21],
        "ee_error_columns": [23, 24, 25],
        "ee_error_definition": "observations[:,19:22] - ee_positions; same pre-action timestep",
        "ee_error_units": "m",
        "ee_error_standardized": True,
        "ee_frame_name": "robot_j2s7s300_end_effector",
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

    # [Architecture change] 结构不同，随机初始化；不要加载旧的128层模型权重。
    model = BCPolicy().to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Architecture: {INPUT_DIM} -> 256 -> 256 -> 128 -> 11; parameters: {parameter_count:,}", flush=True)
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
