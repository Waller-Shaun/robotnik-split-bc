"""Trajectory-local history windows for the frozen Robotnik data splits.

This step returns UNNORMALIZED physical features/actions. Training-only
normalization will be added with the trainer. Labels are expert ``actions``,
not ``applied_actions``. Original NPZ files and split assignments are read-only.

With stride=10 and window_size=20, history samples are 10 ms apart and span
190 ms. Sampling the training data does not set the simulator control rate.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Support direct execution from any working directory and reuse the MLP's
# feature definition rather than duplicating its coordinate transform.
TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from robotnik_bodyframe_features import (  # noqa: E402
    BODYFRAME_FEATURE_VERSION,
    build_bodyframe_observations,
)

DEFAULT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST = Path("dataset_info/manifests/merged998/data_split.json")


class TCNTrajectoryDataset(Dataset):
    """Return (history [L,26], current action [11]) without future observations.

    Each trajectory is downsampled independently starting at its first frame.
    Histories shorter than L repeat the first observation on the left, matching
    the intended episode-start behavior at inference. Compact trajectories are
    stored once; overlapping windows are constructed only when requested.
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        split: str = "train",
        window_size: int = 20,
        stride: int = 10,
        max_trajectories: int | None = None,
        manifest: str | Path = DEFAULT_MANIFEST,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        if window_size < 1 or stride < 1:
            raise ValueError("window_size and stride must be positive")
        if max_trajectories is not None and max_trajectories < 1:
            raise ValueError("max_trajectories must be positive or None")

        self.root = Path(root).resolve()
        self.window_size = window_size
        self.stride = stride
        self.split = split
        self.feature_version = BODYFRAME_FEATURE_VERSION
        self.manifest_path = Path(manifest)
        if not self.manifest_path.is_absolute():
            self.manifest_path = self.root / self.manifest_path
        metadata = json.loads(self.manifest_path.read_text(encoding="utf-8"))

        # Inspect filenames only for disjointness; no held-out NPZ is loaded
        # unless its split is explicitly requested by the caller.
        names = metadata["splits"]
        for name in ("train", "val", "test"):
            if len(names[name]) != len(set(names[name])):
                raise ValueError(f"Duplicate trajectory in {name} split")
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if set(names[left]) & set(names[right]):
                raise ValueError(f"Overlapping trajectories: {left} and {right}")

        data_dir = Path(metadata["data_dir"].replace("@WORKSPACE@", str(self.root)))
        if not data_dir.is_absolute():
            data_dir = self.root / data_dir
        self.data_dir = data_dir.resolve()
        self.filenames = list(names[split])
        if max_trajectories is not None:
            self.filenames = self.filenames[:max_trajectories]
        if not self.filenames:
            raise ValueError(f"No trajectories in {split}")

        self.features: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.cumulative_lengths = [0]
        self.raw_dt = 0.001
        for filename in self.filenames:
            path = (self.data_dir / filename).resolve()
            if not path.is_relative_to(self.data_dir):
                raise ValueError(f"Trajectory escapes data directory: {filename}")
            with np.load(path, allow_pickle=False) as data:
                obs = data["observations"]
                act = data["actions"]
                ee = data["ee_positions"]
                times = data["timestamps"]
                n = len(obs)
                if n == 0 or obs.shape != (n, 23) or act.shape != (n, 11):
                    raise ValueError(f"{filename}: invalid observation/action shapes")
                if ee.shape != (n, 3) or times.shape != (n,):
                    raise ValueError(f"{filename}: unaligned positions/timestamps")
                if not all(np.isfinite(a).all() for a in (obs, act, ee, times)):
                    raise ValueError(f"{filename}: non-finite values")
                if not np.isin(obs[:, 22], (0.0, 1.0)).all():
                    raise ValueError(f"{filename}: base_active must be binary")
                if n > 1 and not np.allclose(
                    np.diff(times), self.raw_dt, rtol=1e-4, atol=1e-7
                ):
                    raise ValueError(f"{filename}: expected regular 1 ms timestamps")

                # Features and labels refer to the same pre-action frame.
                x = build_bodyframe_observations(obs[::stride], ee[::stride])
                y = np.array(act[::stride], dtype=np.float32, copy=True)
            self.features.append(np.ascontiguousarray(x, dtype=np.float32))
            self.actions.append(y)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(x))

        self._history_offsets = np.arange(window_size - 1, -1, -1)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def locate(self, index: int) -> tuple[int, int]:
        """Return trajectory index and downsampled frame index."""
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        trajectory = bisect_right(self.cumulative_lengths, index) - 1
        return trajectory, index - self.cumulative_lengths[trajectory]

    def __getitem__(self, index):
        trajectory, frame = self.locate(index)
        history_indices = np.maximum(frame - self._history_offsets, 0)
        history = self.features[trajectory][history_indices]
        label = self.actions[trajectory][frame].copy()
        return torch.from_numpy(history), torch.from_numpy(label)


def main():
    parser = argparse.ArgumentParser(description="Check TCN windows on training data only")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-trajectories", type=int, default=2)
    args = parser.parse_args()
    dataset = TCNTrajectoryDataset(
        root=args.root,
        window_size=args.window_size,
        stride=args.stride,
        max_trajectories=args.max_trajectories,
    )

    # Check episode-start padding, current-frame/action alignment and both
    # sides of every loaded trajectory boundary against its source NPZ.
    for trajectory, filename in enumerate(dataset.filenames):
        start = dataset.cumulative_lengths[trajectory]
        end = dataset.cumulative_lengths[trajectory + 1] - 1
        first, _ = dataset[start]
        np.testing.assert_array_equal(
            first.numpy(), np.repeat(dataset.features[trajectory][:1], args.window_size, axis=0)
        )
        with np.load(dataset.data_dir / filename, allow_pickle=False) as source:
            for index in (start, min(start + args.window_size, end), end):
                actual_trajectory, frame = dataset.locate(index)
                assert actual_trajectory == trajectory
                history, action = dataset[index]
                raw_frames = np.maximum(frame - dataset._history_offsets, 0) * args.stride
                expected = build_bodyframe_observations(
                    source["observations"][raw_frames], source["ee_positions"][raw_frames]
                )
                np.testing.assert_array_equal(history.numpy(), expected)
                np.testing.assert_array_equal(action.numpy(), source["actions"][frame * args.stride])

    x, y = next(iter(DataLoader(dataset, batch_size=32, shuffle=False)))
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    print(f"Split: {dataset.split}; trajectories loaded: {len(dataset.filenames)}")
    print(f"Feature version: {dataset.feature_version}")
    print(f"Windows: {len(dataset)}")
    print(f"Input batch: {tuple(x.shape)}; action batch: {tuple(y.shape)}")
    print(f"History interval: {args.stride * dataset.raw_dt:.3f} s")
    print(f"History span: {(args.window_size - 1) * args.stride * dataset.raw_dt:.3f} s")
    print("PASS: finite values, source alignment, causal history, episode boundaries, start padding")
    print("Features/actions are unnormalized; no training has been started.")


if __name__ == "__main__":
    main()
