"""[Low Z merge] 合并原r>=0.8m的798条和200条补采，仅新建清单，不移动NPZ。"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROBOENV_DIR = Path(__file__).resolve().parents[3] / "RoboEnv"
BASE_SPLIT_REL = "bc_training_nav_ring_r08/run_20261103_022339_570160/data_split.json"
SUPPLEMENT_REL = "expert_data_23d_noise_ring_lowz_supplement"
OUTPUT_REL = "bc_training_arm_ring_r08_lowz_supplement"
SPLITS = ("train", "val", "test")


def read_json(path):
    return Path(path).read_text(encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(target):
    values = np.asarray(target, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Invalid target")
    return tuple(round(float(x), 6) for x in values)


def build_manifest(roboenv):
    roboenv = Path(roboenv).resolve()
    base_path = roboenv / BASE_SPLIT_REL
    supplement_dir = roboenv / SUPPLEMENT_REL
    plan_path = supplement_dir / "collection_plan.json"
    base = json.loads(read_json(base_path))
    plan = json.loads(read_json(plan_path))
    if {s: len(base["splits"][s]) for s in SPLITS} != {"train": 558, "val": 116, "test": 124}:
        raise ValueError("Expected the frozen 798-target base split")
    if plan["base_split_sha256"] != sha256(base_path):
        raise ValueError("Supplement was collected against another base split")
    source_hashes = {str(p): sha256(p) for p in [base_path, plan_path]}
    owners = {}
    for entry in plan["entries"]:
        target = key(entry["target"])
        if target in owners or entry["proposed_split"] not in SPLITS:
            raise ValueError("Invalid supplement target plan")
        owners[target] = entry
    if len(owners) != 200:
        raise ValueError("Expected 200 supplemental targets")
    groups = defaultdict(set)
    for entry in owners.values():
        groups[entry["xy_group"]].add(entry["proposed_split"])
    if len(groups) != 50 or any(len(s) != 1 for s in groups.values()):
        raise ValueError("Supplement XY groups cross split boundaries")

    files = {s: [] for s in SPLITS}
    targets = {s: [] for s in SPLITS}
    steps = Counter()
    by_source = {name: Counter() for name in ["base", "supplement"]}
    steps_by_source = {name: Counter() for name in ["base", "supplement"]}
    provenance = {}
    height_counts = {s: Counter() for s in SPLITS}
    radius_counts = {s: Counter() for s in SPLITS}
    bad_hold, never_reached = [], []
    seen_targets, seen_paths = set(), set()

    def append(path, target, split, source, entry=None):
        path = path.resolve()
        relative = path.relative_to(roboenv).as_posix()
        expected = key(target)
        if expected in seen_targets or path in seen_paths:
            raise ValueError(f"Duplicate target/file: {path}")
        with np.load(path, allow_pickle=False) as d:
            actual = key(d["target"])
            if actual != expected or int(d["input_dim"]) != 23:
                raise ValueError(f"Wrong target/schema: {path}")
            n = len(d["observations"])
            if (n <= 0 or d["observations"].shape != (n, 23)
                    or d["actions"].shape != (n, 11) or d["ee_positions"].shape != (n, 3)):
                raise ValueError(f"Misaligned observation/action/EE data: {path}")
            held = bool(d["success"])
            reached = float(d["best_error"]) < .01
            if source == "supplement" and (not held or float(d["continuous_hold_seconds"]) < 1 - 1e-9):
                raise ValueError(f"Supplement has not completed the promised hold: {path}")
        seen_targets.add(expected); seen_paths.add(path)
        files[split].append(relative); targets[split].append([float(v) for v in target])
        steps[split] += n; by_source[source][split] += 1; steps_by_source[source][split] += n
        height_counts[split][f"{float(target[2]):.6f}"] += 1
        radius = float(np.linalg.norm(np.asarray(target[:2]) - np.asarray(base["center_xy_m"])))
        if radius < .8 - 1e-5 or radius > 2. + 1e-5:
            raise ValueError(f"Out-of-range target: {target}")
        radius_counts[split][f"{radius:.6f}"] += 1
        provenance[relative] = {"source": source, "split": split, "time_steps": n,
                                "xy_group": entry["xy_group"] if entry else None}
        if not held: bad_hold.append(relative)
        if not reached: never_reached.append(relative)

    # [Low Z merge] 每组先放旧记录，顺序、坐标及归属完全保留，后面追加新记录。
    for split in SPLITS:
        if len(base["splits"][split]) != len(base["targets"][split]):
            raise ValueError("Base filenames and target coordinates are not aligned")
        for name, target in zip(base["splits"][split], base["targets"][split]):
            append(Path(base["data_dir"]) / name, target, split, "base")
    new_files = {}
    for path in sorted(supplement_dir.glob("expert_episode_*.npz")):
        with np.load(path, allow_pickle=False) as d:
            target = key(d["target"])
        if target not in owners or target in new_files:
            raise ValueError(f"Unexpected or duplicate supplement: {path}")
        new_files[target] = path
    if set(new_files) != set(owners):
        raise ValueError("Supplement is incomplete or contains unplanned targets")
    for entry in plan["entries"]:
        append(new_files[key(entry["target"])], entry["target"], entry["proposed_split"], "supplement", entry)
    counts = {s: len(files[s]) for s in SPLITS}
    if counts != {"train": 698, "val": 144, "test": 156}:
        raise ValueError(f"Wrong merged counts: {counts}")
    # 保留原116点验证种子；新28点另用固定种子，不把旧基线误写成144点成绩。
    old_seeds = base["baseline_validation_seeds"]
    if len(old_seeds) != 116:
        raise ValueError("Expected 116 frozen baseline validation seeds")
    val_seeds = old_seeds + [10042 + i for i in range(28)]
    test_seeds = [42 + i for i in range(124)] + [20042 + i for i in range(32)]
    manifest = {
        "dataset_variant": "ring_r08_base798_plus_continuous_lowz200_v1",
        "split_method": "preserve_base_target_owners_append_supplement_xy_group_split",
        # 两个原始数据目录均不动；相对路径从RoboEnv开始，旧加载方式仍可解析。
        "data_dir": str(roboenv), "path_convention": "splits paths are relative to data_dir; each includes source dataset folder",
        "input_dim": 23, "derived_input_dim": 26, "output_dim": 11,
        "base_active_column": 22, "action_label_key": "actions", "ee_position_key": "ee_positions",
        "splits": files, "targets": targets, "target_counts": counts, "trajectory_counts": counts,
        "sample_counts": dict(steps), "total_targets": 998, "total_samples": sum(steps.values()),
        "source_counts": {k: dict(v) for k, v in by_source.items()},
        "source_sample_counts": {k: dict(v) for k, v in steps_by_source.items()},
        "source_manifests": {"base": str(base_path), "supplement_plan": str(plan_path)},
        "source_manifest_sha256": source_hashes,
        "center_xy_m": base["center_xy_m"], "minimum_horizontal_radius_m": .8, "maximum_horizontal_radius_m": 2.,
        "base_heights_m": [.2, .35, .5, .75, 1., 1.15, 1.3], "supplement_height_range_m": [.2, .5],
        "target_counts_by_height": {s: dict(sorted(v.items())) for s, v in height_counts.items()},
        "target_counts_by_radius": {s: dict(sorted(v.items())) for s, v in radius_counts.items()},
        "base_split_counts": {"train": 558, "val": 116, "test": 124},
        "supplement_split_counts": {"train": 140, "val": 28, "test": 32},
        "preserved_base_targets": 798, "supplement_xy_groups": 50,
        "supplement_xy_group_owners": {g: next(iter(s)) for g, s in sorted(groups.items())},
        "grouping_note": "XY grouping applies to supplemental targets; base target-level ownership remains unchanged.",
        "baseline_validation_seeds": val_seeds, "evaluation_seeds": {"val": val_seeds, "test": test_seeds},
        "original_validation_prefix_length": 116, "original_test_prefix_length": 124,
        "unsuccessful_files_included": bad_hold, "never_reached_files_included": never_reached,
        "retention_policy": "preserve all base files; no success/error filtering; supplemental hold was checked",
        "original_ring_files_not_selected": 133,
        "training_or_evaluation_started": False,
    }
    for path, expected in source_hashes.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Source manifest changed while building merge: {path}")
    audit = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "source_manifest_sha256": source_hashes, "file_provenance": provenance,
             "all_files_found": True, "all_targets_unique": True, "old_target_owners_preserved": True,
             "supplement_xy_group_owners_preserved": True,
             "source_npz_moved_copied_or_modified": False, "old_manifests_modified": False,
             "base_files_without_hold_success_retained": bad_hold}
    return manifest, audit


def report_text(manifest):
    lines = ["# 低高度补采合并清单", "", "原r≥0.8m清单798条＋补采200条＝998条；原931条完整环状数据仍保留，133条近半径目标不加入本清单。", "",
             "| 集合 | 原记录 | 新补采 | 合计 | 时间步 |", "|---|---:|---:|---:|---:|"]
    for s in SPLITS:
        lines.append(f"| {s} | {manifest['base_split_counts'][s]} | {manifest['supplement_split_counts'][s]} | {manifest['target_counts'][s]} | {manifest['sample_counts'][s]:,} |")
    lines += ["", "## 文件路径与使用", "", "`data_dir` 指向RoboEnv，`splits`中每个路径包含来源子目录：", "", "```python",
              "path = Path(manifest['data_dir']) / manifest['splits']['train'][0]", "```", "",
              "NPZ仍为23维观测＋11维动作，带对齐的ee_positions，可派生底盘朝向坐标系26维输入。原文件不移动、不复制、不覆盖。", "",
              "每组前面保留旧目标及其顺序，后面追加补采目标；旧验证116点的种子也保留。新增目标按50个XY组划分，每组4个高度保持同组。此XY分组规则不追溯改写原798条的目标归属。", "",
              "验证集现在为144点、测试集为156点；旧116点成功率只用于旧子集对照，不能直接当成完整144点成绩。", "",
              "本次仅合并清单，未改训练／测试脚本。原微调脚本绑定旧清单和558/116/124数量，后续需使用接入本清单的新训练副本；不能把新JSON覆盖进旧模型目录后直接测试。", "",
              f"原记录中未完成保持的文件继续保留：{len(manifest['unsuccessful_files_included'])}条；其中从未到达1cm的{len(manifest['never_reached_files_included'])}条。本次未新增成功阈值筛选。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROBOENV_DIR / OUTPUT_REL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite existing merge directory: {output}")
    manifest, audit = build_manifest(ROBOENV_DIR)
    print("Merged targets:", manifest["target_counts"])
    print("Time steps:", manifest["sample_counts"])
    print("Source files and target owners preserved; 998 unique targets.")
    if args.dry_run:
        print("Dry run: no files written or changed.")
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / "data_split.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "merge_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "split_report.md").write_text(report_text(manifest), encoding="utf-8")
    print("Created:", output / "data_split.json")


if __name__ == "__main__":
    main()
