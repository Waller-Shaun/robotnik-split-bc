"""Sanity-check the released split MLP in the SAME simulator loop as the TCN.

Reads immutable navigation/supplement-arm artifacts without path relocation.
Loads the original feature builders and policy loaders. No model training,
expert action execution, or wheel override occurs. These pretrained MLPs have
different training histories/features from the TCN; this is an integration
reference, not a controlled architecture-only comparison.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from test_tcn import rollout  # noqa: E402
from test_bc_split_nav_arm_lowz_supplement import load_policy_from_checkpoint  # noqa: E402
from robotnik_navigation_policy import load_navigation_policy, build_navigation_observation  # noqa: E402

ARTIFACTS = ROOT / "artifacts/RoboEnv"
NAV_RUN = ARTIFACTS / "bc_training_nav_ring_r08/run_20261103_022339_570160"
ARM_RUN = ARTIFACTS / "bc_training_arm_ring_r08_lowz_supplement/run_20261103_141232_983306"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_stats(folder, input_dim, output_dim):
    result = {}
    with np.load(folder / "normalization.npz", allow_pickle=False) as data:
        for key, width in (("x_mean", input_dim), ("x_std", input_dim),
                           ("y_mean", output_dim), ("y_std", output_dim)):
            value = data[key].copy()
            if value.shape != (width,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {key} in {folder}")
            if key.endswith("std") and np.any(value <= 0):
                raise ValueError(f"Invalid std in {folder}")
            result[key] = value
    return result


class ReleasedSplitMLP:
    policy_type = "released_split_nav8_arm26_MLP_reference"
    window_size = 1
    stride = 1

    def __init__(self):
        nav = torch.load(NAV_RUN / "best_model.pt", map_location="cpu", weights_only=True)
        arm = torch.load(ARM_RUN / "best_model.pt", map_location="cpu", weights_only=True)
        self.nav = load_navigation_policy(nav).eval()
        self.arm = load_policy_from_checkpoint(arm).eval()
        self.nav_stats = load_stats(NAV_RUN, 8, 4)
        self.arm_stats = load_stats(ARM_RUN, 26, 11)
        if (self.arm_stats["x_mean"][22] != 0 or self.arm_stats["x_std"][22] != 1
                or np.any(self.nav_stats["x_mean"][4:] != 0)
                or np.any(self.nav_stats["x_std"][4:] != 1)):
            raise ValueError("Released MLP categorical-feature normalization changed")
        # Verify the arm's declared frozen navigation binding by content, while
        # resolving portable paths to the immutable artifact tree explicitly.
        expected = arm["config"]["frozen_nav_file_sha256"]
        for filename in ("best_model.pt", "normalization.npz", "config.json", "data_split.json"):
            if sha256(NAV_RUN / filename) != expected[filename]:
                raise ValueError(f"Frozen navigation fingerprint mismatch: {filename}")
        # Check only tensors' feature schemas through the original loaders;
        # path relocation is neither needed nor performed for this reference.
        self.config = {"raw_dt_s": .001}
        self.run_dir = ARM_RUN
        self.epoch = arm["epoch"]
        self.val_mse = float(arm["val_mse"])
        self.manifest = json.loads((ARM_RUN / "data_split.json").read_text(encoding="utf-8"))
        self.provenance = {
            "kind": "integration_reference_only", "nav_run": str(NAV_RUN), "arm_run": str(ARM_RUN),
            "nav_epoch": nav["epoch"], "arm_epoch": arm["epoch"],
            "validation_metric_note": "Printed offline val MSE is released ARM-only MSE; not the TCN joint-action metric.",
            "training_difference": "Released MLP uses pretrained/finetuned arm, specialized nav features and sampling; not an architecture-only comparison.",
            "reference_script_sha256": sha256(Path(__file__).resolve()),
            "file_sha256": {str(folder / name): sha256(folder / name)
                            for folder in (NAV_RUN, ARM_RUN)
                            for name in ("best_model.pt", "normalization.npz", "config.json", "data_split.json")},
        }

    def reset(self):
        pass

    def predict(self, observation, **context):
        nav_x = build_navigation_observation(
            context["base_position"], context["bearing"], context["goal_position"],
            context["goal_bearing"], context["phase"],
        )
        with torch.inference_mode():
            nav_y = self.nav(torch.from_numpy(
                (nav_x - self.nav_stats["x_mean"]) / self.nav_stats["x_std"]
            ).unsqueeze(0))[0].numpy()
            arm_y = self.arm(torch.from_numpy(
                (observation - self.arm_stats["x_mean"]) / self.arm_stats["x_std"]
            ).unsqueeze(0))[0].numpy()
        wheels = nav_y * self.nav_stats["y_std"] + self.nav_stats["y_mean"]
        arm = arm_y * self.arm_stats["y_std"] + self.arm_stats["y_mean"]
        return np.concatenate((wheels, arm[4:11]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-index", type=int, default=79,
                        help="Default 79 is a height-1.0 target already in the fixed TCN diagnostic plan")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()
    if not 0 <= args.val_index < 144:
        parser.error("val index must be in [0, 143]")
    if not np.isfinite(args.duration) or args.duration < .001:
        parser.error("duration must be finite and >= 1 ms")
    torch.set_num_threads(1)
    controller = ReleasedSplitMLP()
    args.target = controller.manifest["targets"]["val"][args.val_index]
    args.seed = int(controller.manifest["evaluation_seeds"]["val"][args.val_index])
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.sensor_noise = "off"
    args.roboenv_dir = ROOT / "external/RoboEnv"
    args.result_file = ROOT / "RoboEnv/reference_mlp_checks" / (
        f"mlp_val{args.val_index:03d}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.npz"
    )
    print("Released split MLP integration reference; same physics loop and success criterion as TCN.", flush=True)
    print("Training/features differ: do not interpret this as an architecture-only comparison.", flush=True)
    print(f"Validation index: {args.val_index}; seed: {args.seed}", flush=True)
    rollout(args, controller)


if __name__ == "__main__":
    main()
