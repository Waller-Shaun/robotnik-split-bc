"""Extend the frozen 23D target split to the 0.1--1.3 m height grid.

[Extended split change] Preserve every old target assignment and retain all
episodes. Split only new targets, stratified by height, with a fixed seed.
No old manifest, training run, or episode is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


ROBOENV_DIR = Path(__file__).resolve().parents[3] / "RoboEnv"
DATA_DIR = ROBOENV_DIR / "expert_data_23d_noise"
# [Extended split change] 来源为训练时冻结的468目标清单，不读取可变的临时划分。
BASELINE_PATH = ROBOENV_DIR / "bc_training_23d_noise_wide_finetune" / "run_20261031_025012_026265" / "data_split.json"
OUTPUT_PATH = ROBOENV_DIR / "bc_training_23d_noise_extended" / "data_split.json"
SEED = 42
SPLITS = ("train", "val", "test")
RATIOS = np.array([.70, .15, .15])


def target_key(target):
    values = np.asarray(target, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Target must contain three finite coordinates")
    return tuple(round(float(v), 6) for v in values)


def array_header(archive, name):
    # [Extended split change] 只读数组头核对形状，不重新解压数百万行观测。
    with archive.open(name + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            reader = np.lib.format.read_array_header_1_0
        elif version == (2, 0):
            reader = np.lib.format.read_array_header_2_0
        else:
            raise ValueError(f"Unsupported NPY header version: {version}")
        return reader(stream)


def scan_episodes():
    groups, lengths, unsuccessful = {}, {}, []
    for path in sorted(DATA_DIR.glob("expert_episode_*.npz")):
        with zipfile.ZipFile(path) as archive:
            shape, _, _ = array_header(archive, "observations")
            action_shape, _, _ = array_header(archive, "actions")
            if len(shape) != 2 or shape[1] != 23 or shape[0] <= 0 or action_shape != (shape[0], 11):
                raise ValueError(f"{path.name}: invalid observation/action shape")
        with np.load(path, allow_pickle=False) as data:
            if int(data["input_dim"].item()) != 23 or int(data["base_active_column"].item()) != 22:
                raise ValueError(f"{path.name}: incompatible input schema")
            if json.loads(data["noise_config_json"].item())["std_nm"] != 0:
                raise ValueError(f"{path.name}: torque-noise pilot found in the formal dataset")
            key = target_key(data["target"])
            if not bool(data["success"].item()):
                unsuccessful.append(path.name)
        groups.setdefault(key, []).append(path.name)
        lengths[path.name] = int(shape[0])
    return groups, lengths, unsuccessful


def allocate_new_targets(new_targets, rng):
    # [Extended split change] 用最大余数法决定新增目标总配额：546 -> 382/82/82。
    exact = len(new_targets) * RATIOS
    totals = np.floor(exact).astype(int)
    for index in np.argsort(-(exact - totals), kind="stable")[:len(new_targets)-int(totals.sum())]:
        totals[index] += 1
    heights = sorted({target[2] for target in new_targets})
    buckets = [sorted(t for t in new_targets if t[2] == z) for z in heights]
    quotas = np.array([np.floor(len(bucket) * RATIOS).astype(int) for bucket in buckets])
    residual_rows = np.array([len(bucket) for bucket in buckets]) - quotas.sum(axis=1)
    deficits = totals - quotas.sum(axis=0)
    # 当前7个新增高度各78点：每层基础54/11/11，余下2点分给不同集合。
    # 随机安排每层遗漏的一个集合，使各层近似70/15/15且全局配额精确。
    if not np.all(residual_rows == 2) or np.any(deficits < 0) or np.any(deficits > len(buckets)):
        raise ValueError("Unexpected grid: this allocator expects equal 78-target new-height strata")
    omitted_counts = len(buckets) - deficits
    omitted = np.repeat(np.arange(3), omitted_counts)
    if len(omitted) != len(buckets):
        raise ValueError("Height allocation does not match total quotas")
    rng.shuffle(omitted)
    for i, skip in enumerate(omitted):
        quotas[i] += 1
        quotas[i, skip] -= 1
    assert np.array_equal(quotas.sum(axis=0), totals)
    result = {name: [] for name in SPLITS}
    for bucket, quota in zip(buckets, quotas):
        shuffled = [bucket[i] for i in rng.permutation(len(bucket))]
        offset = 0
        for name, count in zip(SPLITS, quota):
            result[name].extend(shuffled[offset:offset+int(count)])
            offset += int(count)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    output = args.output_path.resolve()
    report_path = output.with_name("split_report.md")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing split or report: {output.parent}")
    baseline_bytes = BASELINE_PATH.read_bytes()
    baseline = json.loads(baseline_bytes.decode("utf-8-sig"))
    if baseline.get("input_dim") != 23 or baseline.get("output_dim") != 11:
        raise ValueError("Baseline must be a 23-input/11-output manifest")
    old_targets = {name: [target_key(t) for t in baseline["targets"][name]] for name in SPLITS}
    old_owner = {}
    for name in SPLITS:
        for target in old_targets[name]:
            if target in old_owner:
                raise ValueError("Baseline contains overlapping target groups")
            old_owner[target] = name
    groups, lengths, unsuccessful = scan_episodes()
    expected = {(x/10, y/10, z/10) for x in range(-6, 7) for y in range(9, 15) for z in range(1, 14)}
    if set(groups) != expected:
        raise ValueError(f"Grid differs: missing={sorted(expected-set(groups))}, extra={sorted(set(groups)-expected)}")
    if not set(old_owner).issubset(groups):
        raise ValueError("Old targets are missing from the extended dataset")
    if len(old_owner) != 468:
        raise ValueError("Expected the frozen 468-target baseline")
    for name in SPLITS:
        for filename in baseline["splits"][name]:
            if filename not in lengths:
                raise ValueError(f"Missing old episode: {filename}")
    added = allocate_new_targets(sorted(set(groups) - set(old_owner)), np.random.default_rng(SEED))
    # 原目标顺序保留在前；新增目标追加，便于单独复核旧目标的表现。
    target_splits = {name: old_targets[name] + added[name] for name in SPLITS}
    file_splits = {name: [filename for t in target_splits[name] for filename in groups[t]] for name in SPLITS}
    all_targets = [t for group in target_splits.values() for t in group]
    all_files = [f for group in file_splits.values() for f in group]
    assert len(all_targets) == len(set(all_targets)) == 1014
    assert len(all_files) == len(set(all_files)) == len(lengths)
    for name in SPLITS:
        assert set(old_targets[name]).issubset(target_splits[name])
        assert set(baseline["splits"][name]).issubset(file_splits[name])
    sample_counts = {name: sum(lengths[f] for f in file_splits[name]) for name in SPLITS}
    height_counts = {str(z): {name: sum(t[2] == z for t in target_splits[name]) for name in SPLITS}
                     for z in sorted({t[2] for t in groups})}
    manifest = {
        "seed": SEED, "selection": "all_trajectories",
        "dataset_variant": "config_measurement_noise_plus_hold_z_extended_0p1_to_1p3",
        "baseline_split_path": str(BASELINE_PATH),
        "baseline_manifest_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "split_method": "preserve_old_target_groups_and_stratify_new_targets_by_height",
        "new_target_ratios": dict(zip(SPLITS, RATIOS.tolist())),
        "data_dir": str(DATA_DIR), "input_dim": 23, "output_dim": 11,
        "base_active_column": 22, "action_label_key": "actions",
        "splits": file_splits,
        "targets": {name: [list(t) for t in target_splits[name]] for name in SPLITS},
        "added_targets": {name: [list(t) for t in added[name]] for name in SPLITS},
        "preserved_target_counts": {name: len(old_targets[name]) for name in SPLITS},
        "added_target_counts": {name: len(added[name]) for name in SPLITS},
        "target_counts": {name: len(target_splits[name]) for name in SPLITS},
        "trajectory_counts": {name: len(file_splits[name]) for name in SPLITS},
        "sample_counts": sample_counts, "target_counts_by_height": height_counts,
        "unsuccessful_files_included": unsuccessful, "excluded_files": [],
        "old_target_order_preserved": True,
    }
    lines = ["# 扩展数据集划分结果", "", "旧468个目标及其文件的集合归属保持不变，仅划分新增546个目标。按高度分层，随机种子42，全部轨迹保留。", "",
             "| 集合 | 保留旧目标 | 新增目标 | 总目标 | 轨迹文件 | 时间步样本 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in SPLITS:
        lines.append(f"| {name} | {len(old_targets[name])} | {len(added[name])} | {len(target_splits[name])} | {len(file_splits[name])} | {sample_counts[name]:,} |")
    lines += ["", "## 各高度目标数量", "", "| z m | train | val | test | 总计 |", "|---:|---:|---:|---:|---:|"]
    for z, counts in height_counts.items():
        lines.append(f"| {z} | {counts['train']} | {counts['val']} | {counts['test']} | {sum(counts.values())} |")
    lines += ["", f"来源冻结清单：`{BASELINE_PATH}`。", f"新清单：`{output}`。", "",
              "按目标分组，各集合无重复、无交叉，覆盖全部1014个目标。每个新增高度的78个目标均分配到三个集合。", "",
              f"未满足专家连续保持结束条件但仍保留的文件数：{len(unsuccessful)}。未因最终误差或success标志筛选。", "",
              "旧实验的清单、模型、标准化参数及原始NPZ均未改动。旧70个验证目标和71个测试目标仍是扩展集合的子集，可继续单独用于同目标对照。", "",
              "训练脚本不会自动改用新清单；下一次训练需显式指向本清单。扩展后的验证集／测试集目标数不同，不应直接与旧70／71目标成功率混为同一评估。"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name in SPLITS:
        print(f"{name}: old={len(old_targets[name])}, new={len(added[name])}, total={len(target_splits[name])}, samples={sample_counts[name]:,}")
    print(f"Unsuccessful hold episodes retained: {len(unsuccessful)}")
    print("Saved:", output)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
