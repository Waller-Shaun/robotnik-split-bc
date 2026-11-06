"""[Ring split change] 基于split_expert_data_noise_extended.py的环状数据副本。

按目标分组，全部轨迹保留；每个高度93/20/20目标，兼顾各半径覆盖。
与旧1014目标重合的8个坐标继承旧归属；不把旧NPZ加入本次数据集。
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

ROBOENV_DIR = Path(__file__).resolve().parents[3] / "RoboEnv"
DATA_DIR = ROBOENV_DIR / "expert_data_23d_noise_ring"
BASELINE_PATH = (ROBOENV_DIR / "bc_training_26d_noise_wide_extended"
                 / "run_20261101_150934_422124" / "data_split.json")
COLLECTION_PATH = (ROBOENV_DIR / "expert_collection_logs"
                   / "ring_20261102_011739_685582" / "collection_config.json")
OUTPUT_PATH = ROBOENV_DIR / "bc_training_26d_noise_ring" / "data_split.json"
SEED = 42
SPLITS = ("train", "val", "test")
RATIOS = np.array([.70, .15, .15])
HEIGHT_QUOTAS = np.array([93, 20, 20])  # 133个位置约70/15/15；7层共651/140/140。


def target_key(target):
    values = np.asarray(target, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Target must contain three finite coordinates")
    return tuple(round(float(v), 6) for v in values)


def array_header(archive, name):
    # 沿用原脚本的数组头检查，无需把近千万行观测全部读入内存。
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
    groups, lengths, unsuccessful, unreached = {}, {}, [], []
    for path in sorted(DATA_DIR.glob("expert_episode_*.npz")):
        with zipfile.ZipFile(path) as archive:
            shape, _, _ = array_header(archive, "observations")
            action_shape, _, _ = array_header(archive, "actions")
            ee_shape, _, _ = array_header(archive, "ee_positions")
            if (len(shape) != 2 or shape[1] != 23 or shape[0] <= 0
                    or action_shape != (shape[0], 11) or ee_shape != (shape[0], 3)):
                raise ValueError(f"{path.name}: invalid observation/action/EE shape")
        with np.load(path, allow_pickle=False) as data:
            if int(data["input_dim"]) != 23 or int(data["base_active_column"]) != 22:
                raise ValueError(f"{path.name}: incompatible input schema")
            if json.loads(str(data["noise_config_json"]))["std_nm"] != 0:
                raise ValueError(f"{path.name}: unexpected torque-pulse pilot")
            key = target_key(data["target"])
            if not bool(data["success"]):
                unsuccessful.append(path.name)
            if float(data["best_error"]) >= .01:
                unreached.append(path.name)
        groups.setdefault(key, []).append(path.name)
        lengths[path.name] = int(shape[0])
    return groups, lengths, unsuccessful, unreached


def radius_key(target, center):
    # [Ring split change] 目标键保留6位坐标会引入微米级径向舍入差；
    # 半径按4位归类，远小于0.2m环间距，避免将同一圆环误分成多个半径。
    return round(float(np.linalg.norm(np.array(target[:2]) - center)), 4)


def allocate_ring_targets(groups, old_owner, center, layout):
    """[Ring split change] 按高度配额、半径分层，层内随机选目标，保留旧归属。"""
    rng = np.random.default_rng(SEED)
    heights = sorted({t[2] for t in groups})
    if len(heights) != 7:
        raise ValueError("Expected seven heights")
    # 圆心每层只有1点：7个圆心目标分为5/1/1，验证和测试也包含圆心目标。
    center_owners = np.array([0] * 5 + [1, 2])
    rng.shuffle(center_owners)
    result = {name: [] for name in SPLITS}
    for h_index, z in enumerate(heights):
        buckets = [sorted(t for t in groups if t[2] == z and radius_key(t, center) == round(r, 6))
                   for r, _ in layout]
        sizes = np.array([len(b) for b in buckets])
        if sizes.tolist() != [n for _, n in layout] or int(sizes.sum()) != 133:
            raise ValueError(f"Unexpected radius coverage at z={z}")
        quotas = np.floor(sizes[:, None] * RATIOS).astype(int)
        center_row = next(i for i, (r, _) in enumerate(layout) if r == 0)
        quotas[center_row, center_owners[h_index]] += 1
        remaining = HEIGHT_QUOTAS - quotas.sum(axis=0)
        residual = sizes - quotas.sum(axis=1)
        # 先分配每个半径尚缺的2个名额，再分配1个；同等级次序和并列集合随机。
        order = sorted(rng.permutation(len(buckets)), key=lambda i: -int(residual[i]))
        for i in order:
            available = list(range(3))
            for _ in range(int(residual[i])):
                highest = max(remaining[s] for s in available)
                if highest <= 0:
                    raise ValueError("Cannot satisfy height quotas")
                choices = [s for s in available if remaining[s] == highest]
                s = int(rng.choice(choices))
                quotas[i, s] += 1
                remaining[s] -= 1
                available.remove(s)
        if np.any(remaining) or not np.array_equal(quotas.sum(axis=0), HEIGHT_QUOTAS):
            raise ValueError("Height quotas do not match")
        for bucket, quota in zip(buckets, quotas):
            fixed = [[t for t in bucket if old_owner.get(t) == name] for name in SPLITS]
            needed = quota - np.array([len(v) for v in fixed])
            if np.any(needed < 0):
                raise ValueError("Preserved old target ownership exceeds radius quota")
            fresh = [t for t in bucket if t not in old_owner]
            shuffled = [fresh[i] for i in rng.permutation(len(fresh))]
            offset = 0
            for s, name in enumerate(SPLITS):
                count = int(needed[s])
                result[name].extend(fixed[s] + shuffled[offset:offset + count])
                offset += count
            assert offset == len(shuffled)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    output = args.output_path.resolve()
    report = output.with_name("split_report.md")
    if output.exists() or report.exists():
        raise FileExistsError(f"Keep existing frozen split/report: {output.parent}")
    baseline_bytes = BASELINE_PATH.read_bytes()
    baseline = json.loads(baseline_bytes.decode("utf-8-sig"))
    if baseline.get("input_dim") != 23 or baseline.get("output_dim") != 11:
        raise ValueError("Baseline must describe 23D source observations and 11D actions")
    old_owner = {}
    for name in SPLITS:
        for t in baseline["targets"][name]:
            k = target_key(t)
            if k in old_owner:
                raise ValueError("Baseline target groups overlap")
            old_owner[k] = name
    collection = json.loads(COLLECTION_PATH.read_text(encoding="utf-8"))
    center = np.array(collection["center_xy_m"], dtype=float)
    layout = collection["radius_angle_counts"]
    expected = {target_key(t) for t in collection["selected_targets"]}
    groups, lengths, unsuccessful, unreached = scan_episodes()
    if len(expected) != 931 or set(groups) != expected:
        raise ValueError(f"Ring coverage differs: missing={sorted(expected-set(groups))}, extra={sorted(set(groups)-expected)}")
    target_splits = allocate_ring_targets(groups, old_owner, center, layout)
    file_splits = {name: [f for t in target_splits[name] for f in groups[t]] for name in SPLITS}
    all_targets = [t for v in target_splits.values() for t in v]
    all_files = [f for v in file_splits.values() for f in v]
    assert len(all_targets) == len(set(all_targets)) == 931
    assert len(all_files) == len(set(all_files)) == len(lengths)
    shared = set(groups) & set(old_owner)
    for name in SPLITS:
        assert all(old_owner[t] == name for t in target_splits[name] if t in shared)
    sample_counts = {name: sum(lengths[f] for f in file_splits[name]) for name in SPLITS}
    height_counts = {str(z): {name: sum(t[2] == z for t in target_splits[name]) for name in SPLITS}
                     for z in sorted({t[2] for t in groups})}
    radius_counts = {str(r): {name: sum(radius_key(t, center) == round(r, 6) for t in target_splits[name])
                            for name in SPLITS} for r, _ in layout}
    manifest = {
        "seed": SEED, "selection": "all_trajectories",
        "dataset_variant": "config_measurement_noise_plus_hold_ring_0_to_2m_7_heights",
        "split_method": "stratify_by_height_and_radius_preserve_shared_old_target_owner",
        "target_ratios": dict(zip(SPLITS, RATIOS.tolist())),
        "baseline_split_path": str(BASELINE_PATH),
        "baseline_manifest_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "collection_config_path": str(COLLECTION_PATH),
        "data_dir": str(DATA_DIR),
        # [Ring split change] 清单描述原NPZ的23维；训练读取时追加误差得到26维。
        "input_dim": 23, "derived_input_dim": 26, "output_dim": 11,
        "base_active_column": 22, "action_label_key": "actions", "ee_position_key": "ee_positions",
        "splits": file_splits,
        "targets": {name: [list(t) for t in target_splits[name]] for name in SPLITS},
        "target_counts": {name: len(target_splits[name]) for name in SPLITS},
        "trajectory_counts": {name: len(file_splits[name]) for name in SPLITS},
        "sample_counts": sample_counts, "target_counts_by_height": height_counts,
        "target_counts_by_radius": radius_counts,
        "preserved_shared_targets": {name: [list(t) for t in target_splits[name] if t in shared] for name in SPLITS},
        "unsuccessful_files_included": unsuccessful, "never_reached_files_included": unreached,
        "excluded_files": [],
    }
    lines = ["# 环状数据集划分结果", "", "931个目标全部保留，按目标分组，高度和半径分层，随机种子42。",
             "同目标的所有轨迹进入同一集合；与旧数据重合的8个目标继承旧归属，仅使用新的环状NPZ。", "",
             "| 集合 | 目标数 | 轨迹数 | 时间步样本 |", "|---|---:|---:|---:|"]
    for name in SPLITS:
        lines.append(f"| {name} | {len(target_splits[name])} | {len(file_splits[name])} | {sample_counts[name]:,} |")
    for title, counts in [("各高度目标数量", height_counts), ("各半径目标数量", radius_counts)]:
        lines += ["", "## " + title, "", "| 高度或半径 m | train | val | test |", "|---:|---:|---:|---:|"]
        for value, c in counts.items():
            lines.append(f"| {value} | {c['train']} | {c['val']} | {c['test']} |")
    lines += ["", f"保留未完成连续保持的轨迹：{len(unsuccessful)}条，其中未曾进入1cm的{len(unreached)}条；没有按误差或成功标志筛选。",
              "每个高度93/20/20目标，总数651/140/140，约70/15/15。各半径在每个高度的配额由比例向下取整后补齐，角度随机抽取。",
              "每层圆心只有一个目标，七层圆心目标按5/1/1分组，不复制坐标凑齐集合。", "",
              "input_dim=23描述磁盘源观测，derived_input_dim=26描述建议网络输入；不能把清单的原始维度改为26。",
              "后续26维训练副本需指向本清单，重新计算训练集标准化，不加载旧23/26维归一化统计。", "",
              f"源数据：`{DATA_DIR}`。", f"冻结旧清单：`{BASELINE_PATH}`。", f"新清单：`{output}`。"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name in SPLITS:
        print(f"{name}: {len(target_splits[name])} targets, {len(file_splits[name])} trajectories, {sample_counts[name]:,} samples")
    print(f"Retained unsuccessful hold episodes: {len(unsuccessful)}; never reached: {len(unreached)}")
    print("Saved:", output)
    print("Report:", report)


if __name__ == "__main__":
    main()
