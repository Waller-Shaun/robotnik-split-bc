import json
from pathlib import Path

import numpy as np


ROBOENV_DIR = Path(__file__).resolve().parents[3] / "RoboEnv"
# [23D change] Read new data and write a separate manifest.
DATA_DIR = ROBOENV_DIR / "expert_data_23d"
OUTPUT_DIR = ROBOENV_DIR / "bc_training_23d"
# [23D change] Reuse target groups from the actual 22D baseline run.
BASELINE_SPLIT_PATH = (
    ROBOENV_DIR / "bc_training"
    / "run_20261023_135447_849619" / "data_split.json"
)
SPLIT_PATH = OUTPUT_DIR / "data_split.json"

ERROR_THRESHOLD = 1 # 单位 m


def main():
    # 防止不小心覆盖已经用于实验的数据划分
    if SPLIT_PATH.exists():
        raise SystemExit(
            f"划分文件已经存在：{SPLIT_PATH}\n"
            "请继续使用已有划分，不要在实验中反复重新划分。"
        )

    # 将相同目标的轨迹归为一组，确保不会跨集合
    groups = {}
    excluded_files = []

    for path in sorted(DATA_DIR.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            # [23D change] Prevent old 22D trajectories from entering this split.
            obs = data["observations"]
            if obs.ndim != 2 or obs.shape[1] != 23:
                raise ValueError(f"{path.name}: expected observations (N, 23)")
            if not np.isin(obs[:, 22], [0.0, 1.0]).all():
                raise ValueError(f"{path.name}: invalid binary phase flag")
            final_error = float(data["final_error"])
            target = tuple(
                np.round(data["target"], decimals=6).tolist()
            )

        if final_error >= ERROR_THRESHOLD:
            excluded_files.append(path.name)
            continue

        groups.setdefault(target, []).append(path.name)

    # [23D change] Map new filenames to the SAME target groups; do not reshuffle.
    baseline = json.loads(BASELINE_SPLIT_PATH.read_text(encoding="utf-8"))
    target_splits = {
        name: [tuple(np.round(t, 6).tolist()) for t in baseline["targets"][name]]
        for name in ("train", "val", "test")
    }
    baseline_targets = [t for ts in target_splits.values() for t in ts]
    if len(baseline_targets) != len(set(baseline_targets)):
        raise ValueError("Baseline has repeated targets across/within splits")
    missing = set(baseline_targets) - set(groups)
    extra = set(groups) - set(baseline_targets)
    if missing or extra:
        raise ValueError(
            f"Target coverage differs from baseline. Missing: {sorted(missing)}; "
            f"Extra: {sorted(extra)}. Finish collection/check filtering first."
        )

    file_splits = {
        name: [
            filename
            for target in split_targets
            for filename in groups[target]
        ]
        for name, split_targets in target_splits.items()
    }

    # 只保存清单，不复制或修改原始轨迹
    manifest = {
        "seed": baseline["seed"],
        # [23D change] Record where the fixed target assignment came from.
        "baseline_split_path": str(BASELINE_SPLIT_PATH),
        "split_method": "reuse_baseline_target_groups",
        "error_threshold_m": ERROR_THRESHOLD,
        "data_dir": str(DATA_DIR),
        # [23D change] Last column is base_active (0/1).
        "input_dim": 23,
        "base_active_column": 22,
        "output_dim": 11,
        "splits": file_splits,
        "targets": {
            name: [list(target) for target in split_targets]
            for name, split_targets in target_splits.items()
        },
        "excluded_files": excluded_files,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("数据划分完成：")
    for name in ("train", "val", "test"):
        print(
            f"{name}: "
            f"{len(target_splits[name])} 个目标，"
            f"{len(file_splits[name])} 条轨迹"
        )

    print(f"未纳入本次划分：{len(excluded_files)} 条")
    print(f"清单已保存：{SPLIT_PATH}")


if __name__ == "__main__":
    main()