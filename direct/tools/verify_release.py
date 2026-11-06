"""Check script integrity, source schema, checkpoint pairs, and optional real datasets."""
import argparse
import ast
import json
from pathlib import Path
import sys

import numpy as np
import torch
from relocate import RUNS, read, sha

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task"
sys.path.insert(0, str(CODE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--data", action="store_true", help="Check all real NPZ paths, targets, shapes and finite values; may take minutes")
    args = parser.parse_args()
    provenance = read(ROOT / "artifacts/provenance.json")
    for record in provenance["source_files"]:
        path = ROOT / record["file"]
        ast.parse(path.read_text(encoding="utf-8"))
        assert sha(path) == record["sha256"], path
    with np.load(ROOT / "examples/expert_sample.npz", allow_pickle=False) as d:
        assert d["observations"].shape[1] == 23
        assert d["actions"].shape == (len(d["observations"]), 11)
        assert d["ee_positions"].shape == (len(d["observations"]), 3)
        from robotnik_bodyframe_features import build_bodyframe_observations
        x = build_bodyframe_observations(d["observations"], d["ee_positions"])
        assert x.shape[1] == 26 and np.isfinite(x).all()
    tree = ROOT / ("RoboEnv" if args.runtime else "artifacts/RoboEnv")
    for rel in RUNS:
        folder = tree / rel
        checkpoint = torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True)
        assert checkpoint["config"] == read(folder / "config.json")
        assert all(torch.isfinite(v).all() for v in checkpoint["model_state_dict"].values())
    if args.runtime:
        from arm_lowz_supplement_experiment import resolve_finetuned_pair
        from demo_robotnik_bc import resolve_demo_pair, load_policy_from_checkpoint
        from robotnik_navigation_policy import load_navigation_policy
        arm, nav, c, nc = resolve_finetuned_pair(tree, tree / RUNS[3])
        resolve_demo_pair(tree, "previous")
        resolve_demo_pair(tree, "supplement")
        am = load_policy_from_checkpoint(torch.load(arm / "best_model.pt", map_location="cpu", weights_only=True))
        nm = load_navigation_policy(torch.load(nav / "best_model.pt", map_location="cpu", weights_only=True))
        assert am(torch.zeros(1, 26)).shape == (1, 11)
        assert nm(torch.zeros(1, 8)).shape == (1, 4)
        manifest = read(arm / "data_split.json")
        keys = set()
        for split, targets in manifest["targets"].items():
            for t in targets:
                key = tuple(round(float(v), 6) for v in t)
                assert key not in keys
                keys.add(key)
        assert len(keys) == 998
        if args.data:
            for split, files in manifest["splits"].items():
                for name, target in zip(files, manifest["targets"][split]):
                    with np.load(Path(manifest["data_dir"]) / name, allow_pickle=False) as d:
                        np.testing.assert_allclose(d["target"], target, atol=1e-6, rtol=0)
                        n = len(d["observations"])
                        assert d["observations"].shape == (n, 23)
                        assert d["actions"].shape == (n, 11)
                        assert d["ee_positions"].shape == (n, 3)
                        assert all(np.isfinite(d[k]).all() for k in ["observations", "actions", "ee_positions"])
        print("Original pair resolvers, normalization inheritance, 998 target owners, and model forwards: OK")
    print("Release source hashes, sample schema, and checkpoint configs: OK")


if __name__ == "__main__":
    main()
