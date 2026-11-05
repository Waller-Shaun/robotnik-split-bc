# COMP0245: frozen hybrid TCN robot controller

Current model release, 2026-11-04. The default is the validated **TCN**, not the older two-MLP demonstration. The matched structured MLP and an earlier direct-torque baseline are also included.

This is a **hybrid controller**: learned arm movement field/gravity, a reference integrator, fixed PD tracking (100/30), and a frozen learned navigation network. It is not an end-to-end torque-only TCN and does not execute expert IK online. TCN history is cleared between episodes.

## Recorded prospective test

| Condition | Distinct new targets | Repeats | Success |
|---|---:|---:|---:|
| Eight native noise scales = 0.001 | 500 | 3 | **1486/1500 = 99.07%** |
| Same targets, noise off | 500 | 1 | **493/500 = 98.60%** |

Target-cluster, height-stratified bootstrap 95% interval for noisy success: approximately **98.4%-99.6%**. These are 500 targets, not 1500 independent targets. Three-repeat all-success targets: 490/500. Do not report best-of-three as 100% single-trial success or claim noise improves performance.

Protocol: horizontal distance from origin 0.8-2 m, height 0.2-1.3 m, fixed initial pose, 20 simulation seconds maximum, 1 ms physics/control timestep, anytime **true** end-effector error <1 cm, no hold requirement. `_cov` values are used as Gaussian **standard deviations** by this pinned RoboEnv implementation. Sensor noise is not an external-force perturbation.

All original weights, normalization and required inference source files are copied byte-for-byte. New portable launchers select model paths relative to this folder; embedded historical absolute paths are provenance only. Raw training data, full rollout archives, Python environments and upstream meshes are not bundled. Inference does not require the training dataset.

## Setup

Run all commands **inside this release folder**, even if it is a subfolder of the GitHub repository. Install Python, Git and Pixi first.

```bash
python tools/setup_dependencies.py
pixi install --manifest-path external/RoboEnv/pixi.toml -e ml --locked
pixi run --manifest-path external/RoboEnv/pixi.toml -e ml --locked python tools/verify_release.py --runtime
```

The setup script clones the original RoboEnv repository and its pinned submodule; it does not copy or alter another installation. The locked upstream ML environment supports Apple Silicon macOS, Linux and Windows; Intel macOS ML support is not promised by that lock. Initial dependency installation needs network access.

## Test your target (-0.9, 1.2, 1.1) m

```bash
pixi run --manifest-path external/RoboEnv/pixi.toml -e ml --locked python run_model.py --target -0.9 1.2 1.1 --seed 137
```

This opens the live viewer with the screenshot noise profile. Add `--headless` for no window, `--noise-profile off` for the historical clean condition, or `--model mlp` for the structured MLP. Default maximum duration is 20 simulation seconds, not a wall-clock guarantee. New outputs are written under `outputs/`; the frozen models are never overwritten.

## Repeat the shared target benchmark

```bash
# Quick pipeline check only; NOT a full benchmark success rate:
pixi run --manifest-path external/RoboEnv/pixi.toml -e ml --locked python run_batch.py --workers 2 --limit 2

# All 500 targets x 3 predeclared noise seeds:
pixi run --manifest-path external/RoboEnv/pixi.toml -e ml --locked python run_batch.py --workers 4

# Clean comparison: same 500 targets, one seed per target:
pixi run --manifest-path external/RoboEnv/pixi.toml -e ml --locked python run_batch.py --workers 4 --noise-profile off
```

`results/shared_test_targets.json` is the common target/seed/protocol file to share with a collaborator. The new runner preserves the original control and scoring loops. It writes a fresh result directory, retains failures, and records process errors in the denominator. Results may differ across platforms or dependency versions; the recorded 99.07% belongs to the original audited run, not an unexecuted rerun.

## Contents

- `models/tcn/`, `models/mlp/`, `models/baseline/`: frozen checkpoints, normalization, splits, configuration and training history.
- `artifacts/RoboEnv/`: byte-identical frozen navigation and historical parent artifacts/configuration.
- `LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task/`: original required inference/training source and tests.
- `results/`: recorded per-trial outcomes, audit, shared targets and failed-repair decisions. Historical paths in these records refer to the original local raw archive, which is not uploaded.
- `dataset_info/`: original fixed split definitions, dataset documentation and checksums; no full raw dataset.
- `release_manifest.json`: exact file hashes, model hashes and dependency pins.
- `docs/`: current model results, limitations and provenance.

## Retraining and limitations

Retraining needs the original 1131 trajectory dataset, including recorded auxiliary arm labels. Obtain it separately from your group; no public download link is fabricated. Restore its two data directories under `RoboEnv/`, preserving names and the fixed `merged998` manifest (698 train /144 validation /156 test). Then run:

```bash
pixi run --manifest-path external/RoboEnv/pixi.toml -e ml --locked python LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task/tcn/train_structured_arm.py --model tcn --normalization robust
```

The old main test was later used to diagnose repair candidates; it cannot be reused as an unseen test for future tuned models. No repair candidate was adopted. The unused reserve target list is intentionally not published in this release.

No claim is made for arbitrary initial poses, obstacles, stronger noise, external pushes, persistent holding, real robots or repeated training seeds. Compare architectures only with shared navigation, supervision, features, tracking controller and evaluation protocol. This repository is code/model evidence, not a completed written coursework report or a guarantee of submission compliance.

See [provenance and licensing](docs/PROVENANCE.md). Keep this coursework repository private unless you have permission to publish the teaching materials and derived code.
