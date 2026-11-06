"""[Low Z training change] 补采版的模型／清单核验，只有读取操作，不导入训练。"""
import json
import math
from pathlib import Path

from arm_finetune_experiment import sha256, resolve_finetuned_pair as resolve_parent_pair

VARIANT = "arm_lowz_continuous_supplement_replay_r08_v1"
MERGED_VARIANT = "ring_r08_base798_plus_continuous_lowz200_v1"
COUNTS = {"train": 698, "val": 144, "test": 156}
BASE_COUNTS = {"train": 558, "val": 116, "test": 124}


def validate_merged_manifest(manifest, base):
    if (manifest.get("dataset_variant") != MERGED_VARIANT
            or manifest.get("input_dim") != 23 or manifest.get("output_dim") != 11
            or manifest.get("target_counts") != COUNTS):
        raise ValueError("Expected the frozen 998-target supplemental manifest")
    seen = set()
    for split, count in COUNTS.items():
        files, targets = manifest["splits"][split], manifest["targets"][split]
        n = BASE_COUNTS[split]
        if len(files) != count or len(targets) != count or targets[:n] != base["targets"][split]:
            raise ValueError(f"Wrong split size or old target ownership: {split}")
        new_paths = [(Path(manifest["data_dir"]) / f).resolve() for f in files[:n]]
        old_paths = [(Path(base["data_dir"]) / f).resolve() for f in base["splits"][split]]
        if new_paths != old_paths:
            raise ValueError(f"Old trajectory paths/order changed: {split}")
        for target in targets:
            if len(target) != 3 or not all(math.isfinite(v) for v in target):
                raise ValueError("Nonfinite target")
            key = tuple(round(float(v), 6) for v in target)
            if key in seen:
                raise ValueError("Target appears in more than one episode/split")
            seen.add(key)
    if manifest["baseline_validation_seeds"][:116] != base["baseline_validation_seeds"]:
        raise ValueError("Old validation seeds changed")
    for split in ["val", "test"]:
        if len(manifest["evaluation_seeds"][split]) != COUNTS[split]:
            raise ValueError("Missing evaluation seeds")


def resolve_finetuned_pair(roboenv, arm_run=None, nav_run=None):
    if arm_run is None:
        root = Path(roboenv) / "bc_training_arm_ring_r08_lowz_supplement"
        runs = sorted(p for p in root.glob("run_*")
                      if (p / "training_finished.json").is_file() and (p / "best_model.pt").is_file()
                      and not json.loads((p / "training_finished.json").read_text(encoding="utf-8")).get("smoke_only", False)
                      and json.loads((p / "config.json").read_text(encoding="utf-8")).get("experiment_variant") == VARIANT)
        if not runs:
            raise FileNotFoundError(f"Train the supplemental arm first or pass --arm-run-dir: {root}")
        arm_run = runs[-1]
    arm_run = Path(arm_run).resolve()
    c = json.loads((arm_run / "config.json").read_text(encoding="utf-8"))
    if (c.get("experiment_variant") != VARIANT or c.get("training_mode") != "arm_only_finetune_navigation_frozen"
            or c.get("navigation_weights_updated") is not False or c.get("trained_output_columns") != list(range(4, 11))):
        raise ValueError("Expected the supplemental-data arm-only model")
    bound = Path(c["nav_run_dir"]).resolve()
    nav_run = Path(nav_run).resolve() if nav_run is not None else bound
    if nav_run != bound:
        raise ValueError("Navigation must remain the frozen baseline")
    expected_names = {"best_model.pt", "normalization.npz", "config.json", "data_split.json"}
    for field, folder in [("frozen_nav_file_sha256", nav_run),
                          ("source_model_file_sha256", Path(c["source_model_run_dir"]))]:
        if set(c[field]) != expected_names:
            raise ValueError("Incomplete frozen file hashes")
        for name, expected in c[field].items():
            if sha256(folder / name) != expected:
                raise ValueError(f"Frozen source changed: {folder / name}")
    # 原84.48%模型与原底盘绑定关系仍按旧核验函数检查，不改旧文件。
    parent, nav_run, parent_config, nav_config = resolve_parent_pair(roboenv, Path(c["source_model_run_dir"]), nav_run)
    if sha256(arm_run / "normalization.npz") != sha256(parent / "normalization.npz"):
        raise ValueError("Inherited arm normalization changed")
    if sha256(arm_run / "data_split.json") != c["source_split_sha256"]:
        raise ValueError("Supplemental arm manifest differs from training snapshot")
    merged = json.loads((arm_run / "data_split.json").read_text(encoding="utf-8"))
    base = json.loads((nav_run / "data_split.json").read_text(encoding="utf-8"))
    validate_merged_manifest(merged, base)
    return arm_run, nav_run, c, nav_config
