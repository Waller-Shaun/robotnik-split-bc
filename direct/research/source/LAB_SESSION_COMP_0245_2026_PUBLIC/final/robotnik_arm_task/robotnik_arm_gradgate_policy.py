"""[B3 change] 梯度门控只发生在训练中；部署结构仍为原MLP力矩路径。"""
from pathlib import Path
import json
import torch

from robotnik_arm_aux_policy import (AuxiliaryArmPolicy, FROZEN_FILES, sha256,
                                    resolve_finetuned_pair as resolve_parent_pair)

VARIANT = "B3_aux_shared_gradient_cosine_gate_v1"
RUN_ROOT = "bc_training_arm_ring_r08_gradgate_b3"


def load_policy_from_checkpoint(checkpoint):
    c = checkpoint["config"]
    if (c.get("experiment_variant") != VARIANT or c.get("input_dim") != 26
            or c.get("output_dim") != 11 or c.get("auxiliary_output_dim") != 7
            or c.get("expert_references_used_as_input") is not False
            or c.get("auxiliary_output_applied_to_robot") is not False
            or c.get("gru_used") is not False):
        raise ValueError("Expected B3 checkpoint, no reference input or auxiliary control")
    model = AuxiliaryArmPolicy(c["hidden_dims"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("Nonfinite B3 parameters")
    return model


def resolve_finetuned_pair(roboenv, arm_run=None, nav_run=None):
    if arm_run is None:
        runs = []
        for p in (Path(roboenv)/RUN_ROOT).glob("run_*"):
            if not (p/"training_finished.json").is_file() or not (p/"best_model.pt").is_file():
                continue
            c = json.loads((p/"config.json").read_text(encoding="utf-8"))
            if c.get("experiment_variant") == VARIANT and not c.get("smoke_only", True):
                runs.append(p)
        if not runs:
            raise FileNotFoundError("No completed gated B3 run; train separately or specify --arm-run-dir")
        arm_run = sorted(runs)[-1]
    arm_run = Path(arm_run).resolve()
    c = json.loads((arm_run/"config.json").read_text(encoding="utf-8"))
    if (c.get("experiment_variant") != VARIANT
            or c.get("training_mode") != "arm_only_gradient_gated_aux_navigation_frozen"
            or c.get("gradient_mode") not in ["gated", "fixed", "off"]
            or c.get("navigation_weights_updated") is not False):
        raise ValueError("Wrong B3 experiment schema")
    parent, nav, _, nc = resolve_parent_pair(roboenv, c["source_model_run_dir"], nav_run)
    if str(nav) != c["nav_run_dir"]:
        raise ValueError("Frozen navigation mismatch")
    for field, folder in [("source_model_file_sha256", parent), ("frozen_nav_file_sha256", nav)]:
        if set(c[field]) != set(FROZEN_FILES):
            raise ValueError("Incomplete parent/model binding")
        for name, expected in c[field].items():
            if sha256(folder/name) != expected:
                raise ValueError(f"Source changed: {folder/name}")
    for name in ["normalization.npz", "data_split.json"]:
        if sha256(arm_run/name) != sha256(parent/name):
            raise ValueError(f"B3 changed inherited {name}")
    if sha256(arm_run/"aux_normalization.npz") != c["aux_normalization_sha256"]:
        raise ValueError("Aux statistics changed")
    for name, expected in c["inference_module_sha256"].items():
        if sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"Inference source changed: {name}")
    return arm_run, nav, c, nc
