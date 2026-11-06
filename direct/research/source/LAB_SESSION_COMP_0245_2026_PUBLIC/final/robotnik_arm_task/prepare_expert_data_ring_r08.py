"""[R08 split change] 只新建半径>=0.8m清单，保留原文件、分组及旧模型。"""
import json
import hashlib
from pathlib import Path

import numpy as np


def prepare(roboenv):
    source = roboenv / "bc_training_26d_noise_ring/data_split.json"
    output = roboenv / "bc_training_nav_ring_r08"
    destination = output / "data_split.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    original = json.loads(source.read_text(encoding="utf-8"))
    baseline_dir = roboenv / "bc_training_26d_noise_ring_bodyframe/run_20261102_184221_743102/batch_tests/val_20261102_230205_714031"
    import csv
    with (baseline_dir/"results.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    key = lambda t: tuple(round(float(v), 6) for v in t)
    baseline = {key([r["target_"+c] for c in "xyz"]): r for r in rows}
    manifest = dict(original)
    manifest.update({"selection": "horizontal_radius_at_least_0p8_preserve_original_split",
                     "dataset_variant": "config_measurement_noise_plus_hold_ring_r08_to_2m_7_heights",
                     "source_manifest": str(source), "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                     "minimum_horizontal_radius_m": .8, "maximum_horizontal_radius_m": 2., "center_xy_m": [0.,0.],
                     "navigation_input_dim": 8, "derived_input_dim": 26,
                     "splits": {}, "targets": {}, "source_split_counts": {k:len(v) for k,v in original["splits"].items()},
                     "baseline_validation_dir": str(baseline_dir)})
    for split in ["train", "val", "test"]:
        owner = {key(t):t for t in original["targets"][split]}
        files, targets = [], []
        for filename in original["splits"][split]:
            with np.load(Path(original["data_dir"])/filename, allow_pickle=False) as d:
                target = d["target"]
            if key(target) not in owner:
                raise ValueError(f"Wrong original target ownership: {filename}")
            if round(float(np.linalg.norm(target[:2])), 4) >= .8:
                files.append(filename); targets.append(owner[key(target)])
        manifest["splits"][split], manifest["targets"][split] = files, targets
    assert {k:len(v) for k,v in manifest["splits"].items()} == {"train":558,"val":116,"test":124}
    manifest["baseline_validation_seeds"] = [int(baseline[key(t)]["seed"]) for t in manifest["targets"]["val"]]
    wins = sum(baseline[key(t)]["success"] == "True" for t in manifest["targets"]["val"])
    manifest["baseline_model_validation"] = {"successes":wins,"targets":116,"success_rate_percent":100*wins/116}
    retained = {f for v in manifest["splits"].values() for f in v}
    for name in ["unsuccessful_files_included", "never_reached_files_included"]:
        manifest[name] = [f for f in original.get(name,[]) if f in retained]
    output.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    report = "# 半径≥0.8m独立清单\n\n保留798条，train/val/test=558/116/124；原始NPZ和分组不改。\n\n"
    report += "新底盘只使用558个训练目标；复用的手臂模型此前使用原环状651个训练目标（含近距离训练目标），没有使用留出的验证／测试目标。\n\n"
    report += f"原模型在同一116目标上为{wins}/116={100*wins/116:.2f}%，作为开发验证基线。\n"
    (output/"split_report.md").write_text(report,encoding="utf-8")
    print("Created:",destination)
    print("Targets:",{k:len(v) for k,v in manifest["targets"].items()})
    return destination


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[3]/"RoboEnv")
