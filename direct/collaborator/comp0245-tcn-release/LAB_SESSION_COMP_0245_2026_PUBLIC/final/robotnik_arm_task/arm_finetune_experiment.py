"""[Arm finetune change] 只负责测试时选择和核对模型，不导入或启动训练。"""
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_finetuned_pair(roboenv, arm_run=None, nav_run=None):
    """微调模型绑定原底盘模型，避免无意中更换底盘后误判微调效果。"""
    if arm_run is None:
        root = Path(roboenv) / "bc_training_arm_ring_r08_finetune"
        runs = sorted(p for p in root.glob("run_*")
                      if (p / "training_finished.json").is_file()
                      and (p / "best_model.pt").is_file()
                      and not json.loads((p / "training_finished.json").read_text(encoding="utf-8")).get("smoke_only", False))
        if not runs:
            raise FileNotFoundError(f"No finished arm finetune in {root}; train first or pass --arm-run-dir")
        arm_run = runs[-1]
    arm_run = Path(arm_run).resolve()
    arm_config = json.loads((arm_run / "config.json").read_text(encoding="utf-8"))
    if (arm_config.get("training_mode") != "arm_only_finetune_navigation_frozen"
            or arm_config.get("navigation_weights_updated") is not False
            or arm_config.get("trained_output_columns") != list(range(4, 11))):
        raise ValueError("Expected an arm-only finetuned checkpoint")
    bound_nav = Path(arm_config["nav_run_dir"]).resolve()
    nav_run = Path(nav_run).resolve() if nav_run is not None else bound_nav
    if nav_run != bound_nav:
        raise ValueError("Use the same frozen navigation run recorded by this arm finetune")
    expected_files = arm_config["frozen_nav_file_sha256"]
    if set(expected_files) != {"best_model.pt", "normalization.npz", "config.json", "data_split.json"}:
        raise ValueError("Incomplete frozen navigation fingerprints")
    for name, expected in expected_files.items():
        if sha256(nav_run / name) != expected:
            raise ValueError(f"Frozen navigation changed: {name}")
    nav_config = json.loads((nav_run / "config.json").read_text(encoding="utf-8"))
    if Path(arm_config["source_model_run_dir"]).resolve() != Path(nav_config["arm_run_dir"]).resolve():
        raise ValueError("Wrong parent arm model for this frozen navigation baseline")
    if sha256(arm_run / "normalization.npz") != arm_config["source_normalization_sha256"]:
        raise ValueError("Arm normalization differs from inherited parent statistics")
    arm_split = json.loads((arm_run / "data_split.json").read_text(encoding="utf-8"))
    nav_split = json.loads((nav_run / "data_split.json").read_text(encoding="utf-8"))
    if arm_split != nav_split:
        raise ValueError("Arm and navigation evaluation splits differ")
    return arm_run, nav_run, arm_config, nav_config
