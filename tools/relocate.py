"""Relocate metadata without changing tensors, normalization, splits, or policies.

The four checkpoint folders form a DAG. Rebuild hashes in parent-first order;
never bypass the original inference-time integrity checks.
"""
import hashlib
import json
from pathlib import Path
import re

RUNS = [
    "bc_training_26d_noise_ring_bodyframe/run_20261102_184221_743102",
    "bc_training_nav_ring_r08/run_20261103_022339_570160",
    "bc_training_arm_ring_r08_finetune/run_20261103_111655_849512",
    "bc_training_arm_ring_r08_lowz_supplement/run_20261103_141232_983306",
]
SOURCE_PREFIX = "F:/UCL/COMP0245 Labs"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def replace_paths(obj, destination):
    if isinstance(obj, dict):
        return {replace_paths(k, destination): replace_paths(v, destination) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_paths(v, destination) for v in obj]
    if isinstance(obj, str):
        # Match only known workspace prefixes, leaving ordinary strings alone.
        for prefix in (SOURCE_PREFIX, "@WORKSPACE@"):
            normalized = obj.replace("\\", "/")
            if prefix in normalized:
                return normalized.replace(prefix, destination)
    return obj


def relocate(tree, workspace, portable=False):
    """tree is a RoboEnv-shaped copy, never the user's source experiment tree."""
    import torch
    tree = Path(tree)
    destination = "@WORKSPACE@" if portable else Path(workspace).resolve().as_posix()
    for path in sorted(tree.rglob("*.json")):
        write(path, replace_paths(read(path), destination))

    def local(path):
        text = str(path).replace("\\", "/")
        marker = destination + "/RoboEnv/"
        if not text.startswith(marker):
            raise ValueError(f"Cannot locate dependency inside release workspace: {path}")
        return tree / text[len(marker):]

    # Provenance hashes with files available in this curated snapshot.
    for path in sorted(tree.rglob("*.json")):
        obj = read(path)
        for field, hash_field in [("source_manifest", "source_manifest_sha256"),
                                  ("baseline_split_path", "baseline_manifest_sha256")]:
            if field in obj and hash_field in obj:
                try:
                    target = local(obj[field])
                except ValueError:
                    continue
                if target.is_file():
                    obj[hash_field] = sha(target)
        write(path, obj)
    plan_path = tree / "expert_data_23d_noise_ring_lowz_supplement/collection_plan.json"
    if plan_path.is_file():
        plan = read(plan_path)
        plan["base_split_sha256"] = sha(tree / RUNS[1] / "data_split.json")
        write(plan_path, plan)

    for rel in RUNS:
        folder = tree / rel
        config = read(folder / "config.json")
        for path_field, hashes_field in [
            ("arm_run_dir", "arm_file_sha256"),
            ("nav_run_dir", "frozen_nav_file_sha256"),
            ("source_model_run_dir", "source_model_file_sha256"),
        ]:
            if hashes_field in config:
                parent = local(config[path_field])
                config[hashes_field] = {name: sha(parent / name) for name in config[hashes_field]}
        if "source_split_sha256" in config:
            config["source_split_sha256"] = sha(local(config["source_split_path"]))
        if "source_normalization_sha256" in config:
            config["source_normalization_sha256"] = sha(local(config["source_model_run_dir"]) / "normalization.npz")
        write(folder / "config.json", config)
        checkpoint_path = folder / "best_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint = replace_paths(checkpoint, destination)
        checkpoint["config"] = config
        torch.save(checkpoint, checkpoint_path)
