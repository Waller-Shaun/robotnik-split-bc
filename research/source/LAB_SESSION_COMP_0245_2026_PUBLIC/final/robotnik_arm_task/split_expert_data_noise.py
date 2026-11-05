"""[Noise dataset change] Map new files to the original target-based splits.

Keep every trajectory, including episodes that did not finish the hold.
No original data, old split manifest or old model is modified.
"""
import json
from pathlib import Path

import numpy as np


ROBOENV_DIR = Path(__file__).resolve().parents[3] / "RoboEnv"
DATA_DIR = ROBOENV_DIR / "expert_data_23d_noise"
OUTPUT_PATH = ROBOENV_DIR / "bc_training_23d_noise" / "data_split.json"
BASELINE_PATH = (
    ROBOENV_DIR / "bc_training" / "run_20261023_135447_849619"
    / "data_split.json"
)


def target_key(target):
    return tuple(np.round(target, 6).tolist())


def main():
    if OUTPUT_PATH.exists():
        raise SystemExit(f"Keep existing fixed split: {OUTPUT_PATH}")
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    groups = {}
    unsuccessful_files = []
    for path in sorted(DATA_DIR.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            obs = data["observations"]
            if obs.ndim != 2 or obs.shape[1] != 23 or not len(obs):
                raise ValueError(f"{path.name}: expected nonempty (N,23) inputs")
            if data["actions"].shape != (len(obs), 11):
                raise ValueError(f"{path.name}: action dimensions do not match")
            if not np.isin(obs[:, 22], [0.0, 1.0]).all():
                raise ValueError(f"{path.name}: invalid phase flag")
            # Exclude mixed action-disturbance pilots by failing explicitly,
            # rather than silently mixing them with the sensor-noise experiment.
            noise_config = json.loads(str(data["noise_config_json"]))
            if noise_config["std_nm"] != 0.0:
                raise ValueError(f"{path.name}: torque-noise pilot found; separate it first")
            if not bool(data["success"]):
                unsuccessful_files.append(path.name)
            groups.setdefault(target_key(data["target"]), []).append(path.name)

    target_splits = {
        name: [target_key(t) for t in baseline["targets"][name]]
        for name in ("train", "val", "test")
    }
    all_targets = [t for targets in target_splits.values() for t in targets]
    if len(all_targets) != len(set(all_targets)):
        raise ValueError("Baseline contains overlapping or repeated targets")
    missing = set(all_targets) - set(groups)
    extra = set(groups) - set(all_targets)
    if missing or extra:
        raise ValueError(f"Target coverage differs. Missing={sorted(missing)}, extra={sorted(extra)}")
    file_splits = {
        name: [filename for target in targets for filename in groups[target]]
        for name, targets in target_splits.items()
    }
    manifest = {
        "seed": baseline["seed"],
        "selection": "all_trajectories",
        "dataset_variant": "config_measurement_noise_plus_hold",
        "baseline_split_path": str(BASELINE_PATH),
        "split_method": "reuse_baseline_target_groups",
        "data_dir": str(DATA_DIR),
        "input_dim": 23,
        "output_dim": 11,
        "base_active_column": 22,
        "action_label_key": "actions",
        "splits": file_splits,
        "targets": {name: [list(t) for t in targets] for name, targets in target_splits.items()},
        "unsuccessful_files_included": unsuccessful_files,
        "excluded_files": [],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, files in file_splits.items():
        print(f"{name}: {len(files)} trajectories, {len(target_splits[name])} targets")
    print(f"Unsuccessful hold episodes retained: {len(unsuccessful_files)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
