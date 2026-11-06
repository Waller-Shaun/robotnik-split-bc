"""Expose path arguments for original splitting logic; never overwrite frozen lists."""
import argparse
from pathlib import Path
import sys
import zipfile

import numpy as np
from relocate import read, write, sha

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task"
sys.path.insert(0, str(CODE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["full", "r08", "merged"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--collection-config", type=Path)
    parser.add_argument("--full-manifest", type=Path)
    parser.add_argument("--base-split", type=Path)
    parser.add_argument("--supplement-dir", type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    runtime = ROOT / "RoboEnv"
    if args.stage == "full":
        import split_expert_data_ring as original
        original.DATA_DIR = (args.data_dir or runtime / "expert_data_23d_noise_ring").resolve()
        original.COLLECTION_PATH = (args.collection_config or runtime / "expert_collection_logs/ring_20261102_011739_685582/collection_config.json").resolve()
        original.BASELINE_PATH = runtime / "bc_training_26d_noise_wide_extended/run_20261101_150934_422124/data_split.json"
        sys.argv = [str(CODE / "split_expert_data_ring.py"), "--output-path", str(out / "data_split.json")]
        original.main()
    elif args.stage == "r08":
        import split_expert_data_ring as original
        source = (args.full_manifest or runtime / "bc_training_26d_noise_ring/data_split.json").resolve()
        full = read(source)
        reference = read(runtime / "bc_training_nav_ring_r08/data_split.json")
        seeds = {tuple(t): s for t, s in zip(reference["targets"]["val"], reference["baseline_validation_seeds"])}
        result = dict(full)
        result.update(splits={}, targets={}, sample_counts={}, target_counts={}, trajectory_counts={},
            minimum_horizontal_radius_m=.8, maximum_horizontal_radius_m=2., center_xy_m=[0., 0.],
            dataset_variant="config_measurement_noise_plus_hold_ring_r08_to_2m_7_heights",
            selection="horizontal_radius_at_least_0p8_preserve_original_split",
            source_manifest=str(source), source_manifest_sha256=sha(source),
            navigation_input_dim=8, derived_input_dim=26)
        kept = set()
        for split in ["train", "val", "test"]:
            pairs = [(f, t) for f, t in zip(full["splits"][split], full["targets"][split])
                     if round(float(np.linalg.norm(t[:2])), 4) >= .8]
            result["splits"][split] = [f for f, _ in pairs]
            result["targets"][split] = [t for _, t in pairs]
            result["target_counts"][split] = len(pairs)
            result["trajectory_counts"][split] = len(pairs)
            total = 0
            for f, _ in pairs:
                with zipfile.ZipFile(Path(full["data_dir"]) / f) as archive:
                    total += original.array_header(archive, "observations")[0][0]
                kept.add(f)
            result["sample_counts"][split] = total
        if result["target_counts"] != {"train": 558, "val": 116, "test": 124}:
            raise ValueError("Expected the same fixed 931-target layout and ownership")
        result["baseline_validation_seeds"] = [seeds[tuple(t)] for t in result["targets"]["val"]]
        for field in ["unsuccessful_files_included", "never_reached_files_included"]:
            result[field] = [f for f in full.get(field, []) if f in kept]
        # Remove parent-wide height/radius summaries; the selected targets are authoritative.
        for field in ["target_counts_by_height", "target_counts_by_radius", "preserved_shared_targets"]:
            result.pop(field, None)
        out.mkdir(parents=True)
        write(out / "data_split.json", result)
        print("Created", out / "data_split.json", result["target_counts"])
    else:
        import merge_expert_data_ring_lowz as original
        original.BASE_SPLIT_REL = (args.base_split or runtime / "bc_training_nav_ring_r08/data_split.json").resolve()
        original.SUPPLEMENT_REL = (args.supplement_dir or runtime / "expert_data_23d_noise_ring_lowz_supplement").resolve()
        manifest, audit = original.build_manifest(runtime)
        out.mkdir(parents=True)
        write(out / "data_split.json", manifest)
        write(out / "merge_audit.json", audit)
        (out / "split_report.md").write_text(original.report_text(manifest), encoding="utf-8")
        print("Created", out / "data_split.json", manifest["target_counts"])


if __name__ == "__main__":
    main()
