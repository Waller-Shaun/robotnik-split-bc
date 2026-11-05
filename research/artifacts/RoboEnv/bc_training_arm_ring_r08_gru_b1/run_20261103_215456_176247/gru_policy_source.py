"""[B1 GRU change] Shared architecture, causal history and checkpoint checks.

Inference-only helpers: importing this module never starts training/simulation.
The old current-state MLP is retained. A zero-initialized GRU correction adds
memory without discarding the pretrained arm. No expert reference is an input.
"""
from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from robotnik_bodyframe_features import BODYFRAME_FEATURE_VERSION
from arm_finetune_experiment import resolve_finetuned_pair as resolve_parent_pair
from arm_lowz_supplement_experiment import validate_merged_manifest

VARIANT = "B1_arm_gru_history_residual_v1"
HISTORY_VERSION = "causal_26d_oldest_to_current_repeat_episode_first_v1"
RUN_ROOT = "bc_training_arm_ring_r08_gru_b1"
FROZEN_FILES = ("best_model.pt", "normalization.npz", "config.json", "data_split.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CurrentStateMLP(nn.Module):
    """[B1] Same parameter names and layers as the 26D parent checkpoint."""
    def __init__(self, hidden_dims=(256, 256, 128)):
        super().__init__()
        dimensions = [26, *hidden_dims, 11]
        layers = []
        for i, (a, b) in enumerate(zip(dimensions, dimensions[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dimensions) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class HistoryArmPolicy(nn.Module):
    """[B1] (batch, history,26) -> seven normalized arm torques.

    Each finite window starts with a fresh GRU hidden state in training AND
    inference. There is no hidden state carried across windows or episodes.
    """
    def __init__(self, hidden_dims=(256, 256, 128), encoder_dim=64, gru_hidden_dim=64):
        super().__init__()
        self.baseline = CurrentStateMLP(hidden_dims)
        self.history_encoder = nn.Sequential(nn.Linear(26, encoder_dim), nn.ReLU())
        self.gru = nn.GRU(encoder_dim, gru_hidden_dim, batch_first=True)
        self.torque_correction = nn.Linear(gru_hidden_dim, 7)
        # [B1] Epoch zero is exactly the pretrained parent policy.
        nn.init.zeros_(self.torque_correction.weight)
        nn.init.zeros_(self.torque_correction.bias)

    def forward(self, history):
        if history.ndim != 3 or history.shape[-1] != 26 or history.shape[1] < 1:
            raise ValueError("Expected (batch,history_length,26)")
        current_torque = self.baseline(history[:, -1, :])[:, 4:11]
        _, hidden = self.gru(self.history_encoder(history))
        return current_torque + self.torque_correction(hidden[-1])


class CausalObservationHistory:
    """[B1] Exactly the same episode-clamped indices as training.

    Append one NORMALIZED observation per physics/control step. Then select
    t-(K-1)*stride,...,t-stride,t. Missing early observations repeat the FIRST
    observation of this episode, never the last observation of another episode.
    """
    def __init__(self, length, stride):
        if type(length) is not int or type(stride) is not int or min(length, stride) < 1:
            raise ValueError("History length/stride must be positive integers")
        self.length, self.stride = length, stride
        self.buffer = deque(maxlen=(length - 1) * stride + 1)

    def append(self, normalized_observation):
        observation = np.asarray(normalized_observation, dtype=np.float32)
        if observation.shape != (26,) or not np.isfinite(observation).all():
            raise ValueError("History requires a finite normalized 26D observation")
        self.buffer.append(observation.copy())
        states = list(self.buffer)
        return np.stack([states[max(0, len(states) - 1 - offset)]
                         for offset in range((self.length - 1) * self.stride, -1, -self.stride)])


def validate_config(config):
    if (config.get("experiment_variant") != VARIANT or config.get("input_dim") != 26
            or config.get("output_dim") != 7 or config.get("source_input_dim") != 23
            or config.get("input_feature_version") != BODYFRAME_FEATURE_VERSION
            or config.get("history_feature_version") != HISTORY_VERSION
            or config.get("target_position_frame") != "base_yaw_aligned"
            or config.get("ee_error_frame") != "base_yaw_aligned"
            or config.get("output_action_columns") != list(range(4, 11))
            or config.get("navigation_weights_updated") is not False
            or config.get("auxiliary_reference_loss") is not False
            or config.get("base_active_column") != 22
            or config.get("history_hidden_state_carried") is not False):
        raise ValueError("Expected the B1 causal-history arm checkpoint, without auxiliary supervision")
    for name in ("history_length", "history_stride_steps", "encoder_dim", "gru_hidden_dim"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"Invalid {name}")
    if config.get("hidden_dims") != [256, 256, 128]:
        raise ValueError("Expected unchanged 256/256/128 baseline layers")
    if not np.isfinite(config.get("sample_dt_seconds", np.nan)) or config["sample_dt_seconds"] <= 0:
        raise ValueError("Invalid history sample period")


def load_policy_from_checkpoint(checkpoint):
    config = checkpoint["config"]
    validate_config(config)
    model = HistoryArmPolicy(config["hidden_dims"], config["encoder_dim"], config["gru_hidden_dim"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("Nonfinite GRU checkpoint")
    return model


def resolve_finetuned_pair(roboenv, arm_run=None, nav_run=None):
    """[B1] Dedicated output root and strict frozen-baseline binding."""
    roboenv = Path(roboenv)
    if arm_run is None:
        runs = []
        for path in sorted((roboenv / RUN_ROOT).glob("run_*")):
            if not all((path / n).is_file() for n in ("training_finished.json", *FROZEN_FILES)):
                continue
            finished = json.loads((path / "training_finished.json").read_text(encoding="utf-8"))
            if not finished.get("smoke_only", False):
                runs.append(path)
        if not runs:
            raise FileNotFoundError("No completed B1 run. Train first or specify --arm-run-dir.")
        arm_run = runs[-1]
    arm_run = Path(arm_run).resolve()
    config = json.loads((arm_run / "config.json").read_text(encoding="utf-8"))
    validate_config(config)
    bound_nav = Path(config["nav_run_dir"]).resolve()
    if nav_run is not None and Path(nav_run).resolve() != bound_nav:
        raise ValueError("B1 must use its recorded frozen navigation model")
    for field, folder in [("frozen_nav_file_sha256", bound_nav),
                          ("source_model_file_sha256", Path(config["source_model_run_dir"]))]:
        if set(config[field]) != set(FROZEN_FILES):
            raise ValueError("Incomplete frozen checkpoint fingerprints")
        for name, expected in config[field].items():
            if sha256(folder / name) != expected:
                raise ValueError(f"Frozen parent/navigation changed: {folder / name}")
    parent, nav, _, nav_config = resolve_parent_pair(roboenv, config["source_model_run_dir"], bound_nav)
    if sha256(arm_run / "normalization.npz") != sha256(parent / "normalization.npz"):
        raise ValueError("B1 normalization differs from the same B0 starting model")
    if sha256(arm_run / "data_split.json") != config["source_split_sha256"]:
        raise ValueError("B1 frozen data split changed")
    for name, expected in config["feature_source_sha256"].items():
        if sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"B1 inference feature/model source changed: {name}")
    merged = json.loads((arm_run / "data_split.json").read_text(encoding="utf-8"))
    base = json.loads((nav / "data_split.json").read_text(encoding="utf-8"))
    validate_merged_manifest(merged, base)
    return arm_run, nav, config, nav_config
