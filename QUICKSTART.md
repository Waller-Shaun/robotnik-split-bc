# Quick start: inspect, run, train, evaluate

Run commands from the extracted `COMP0245_Code` directory. Use a short extraction path on Windows, for example `C:\COMP0245_Code`, because the historical research archive has long paths. Python3.11+ is suitable for the standard-library check tools; actual simulation uses the supplied Pixi ML lock. Git and [Pixi](https://pixi.sh/) are needed for a new runtime.

## 1. Inspect the bundle without installing simulation dependencies

```powershell
python -X utf8 submission.py verify
python -X utf8 submission.py results
python -X utf8 submission.py --help
```

`verify` checks the submission hashes, original-copy hashes, Python syntax and JSON. It imports no experiment modules, does not require Torch/PyBullet and does not start training, simulation or network activity. `results` counts existing result records; it does not reproduce raw-trajectory scoring.

## 2. Set up one shared pinned simulation runtime

Windows PowerShell:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
python -X utf8 structured/baseline/tools/setup_dependencies.py
pixi install --manifest-path structured/baseline/external/RoboEnv/pixi.toml -e ml --locked
pixi run --manifest-path structured/baseline/external/RoboEnv/pixi.toml -e ml --locked python -X utf8 structured/tools/verify_experiment.py --runtime
```

The setup script downloads the upstream RoboEnv at the recorded commit and initializes its simulator submodule. It refuses a different pre-existing checkout instead of silently replacing it. It does not train or simulate. Internet is required for Git/Pixi installation. On Linux/macOS, use `export PYTHONUTF8=1` and `export PYTHONIOENCODING=utf-8` rather than PowerShell assignments.

Pinned upstream revisions:

- RoboEnv: `7bb49439677c2ac111347b40e997025ce87bff50`
- simulation_and_control: `df5850526affcc33b25b15d3096c4d21b614bacd`

The dispatcher shares this installed runtime between the fixed-TCN, dynamic-PD and patrol examples. It preserves each model's original configuration/normalization bindings. Setup has not been executed during ZIP preparation.

Explicit path options are resolved relative to the terminal's current directory before the dispatcher changes a child's working directory. Absolute paths also work. The new shared asset tree has not been used to rerun the archived patrol trajectories; the recorded missions remain historical results, not a guarantee of identical cross-machine physics.

## 3. Fixed-PD TCN: one target with a viewer

```powershell
pixi run --manifest-path structured/baseline/external/RoboEnv/pixi.toml -e ml --locked python -X utf8 submission.py fixed-tcn --target -0.9 1.2 1.1 --seed 137
```

Add `--headless` to disable the viewer. Duration is at most 20 simulated seconds; noise defaults to the published `friend` profile. For the original clean comparator, add `--noise-profile off`.

Outputs: fresh `.npz` trajectories and JSON/timing metadata under `structured/baseline/outputs/`. The original `structured/baseline/run_model.py` contains the complete CLI. No weights are updated.

## 4. Learned dynamic PD: the same target

```powershell
pixi run --manifest-path structured/baseline/external/RoboEnv/pixi.toml -e ml --locked python -X utf8 submission.py dynamic-pd --target -0.9 1.2 1.1 --seed 137
```

Add `--headless` for no viewer, or `--variant fixed` for the unchanged parent through the same dynamic-experiment demo interface. Both variants retain the published noise profile. The gain head is selected epoch 17; the shared rollout's parent model epoch message refers to the frozen motion model.

Outputs: new `.npz` demo files under `structured/outputs/demos/`; dynamic runs also save `.gains.json` summaries. Neither the original results nor checkpoints are overwritten.

## 5. Continuous patrol with the frozen D3 direct-torque policy

```powershell
pixi run --manifest-path structured/baseline/external/RoboEnv/pixi.toml -e ml --locked python -X utf8 submission.py patrol --num-targets 5 --target-seed 1266108476 --height-range 0.5 1.3
```

The patrol dispatcher supplies `structured/baseline/external/RoboEnv` as the asset/output root unless `--roboenv-root PATH` is explicitly supplied. Model weights, normalization and the default noisy config come from `direct/artifacts/patrol120/`; the external RoboEnv config is not overwritten.

Useful options: `--target X Y Z` repeated for explicit coordinates, `--route-order given`, `--headless`, `--radius-range 0.8 2`, `--sensor-noise off`, and `--duration-per-target 20`. See `direct/patrol/README.md` for all parameters.

Outputs under the selected RoboEnv root: `patrol_validation_120/patrol_timestamp/`, including route plan JSON/PNG, per-target traces, CSV, summary and actual-path plot. Physical state continues between goals. Default5cm/1s is an application holding criterion, not the 1 cm single-target benchmark. This is D3 patrol, not a prospective validation of the TCN on continuous missions.

## 6. Re-run the published fixed/dynamic paired benchmark

This explicitly starts many simulations. It does not train.

```powershell
# Pipeline subset: 2 goals x3 repeats x2 variants; NOT a full success rate.
pixi run --manifest-path structured/baseline/external/RoboEnv/pixi.toml -e ml --locked python -X utf8 submission.py replay --workers 2 --limit 2

# Full recorded 500-goal paired protocol; choose workers to fit RAM.
pixi run --manifest-path structured/baseline/external/RoboEnv/pixi.toml -e ml --locked python -X utf8 submission.py replay --workers 4
```

Fresh results go to `structured/outputs/replays/`. Published results remain untouched. Four workers are an example, not a guarantee that every machine has enough RAM. Published goals are visible, so replay is reproducibility checking, not a new unseen evaluation for tuned candidates.

## 7. Training and data collection are separate, optional work

Running the supplied pretrained policies needs no original training dataset. Retraining requires separately restoring the dataset and using a NEW output directory:

```powershell
pixi run --manifest-path structured/baseline/external/RoboEnv/pixi.toml -e ml --locked python -X utf8 submission.py retrain-gain --data-root "PATH_TO_PROJECT_CONTAINING_ROBOENV" --output "NEW_GAIN_RUN_DIRECTORY" --smoke-test
```

Remove `--smoke-test` only for a deliberately requested full new training run. The portable trainer fixes cache-dependent initialization; it creates a new experiment and does not inherit the recorded model's score. Its archived implementation selects MPS or CPU; no claim is made that it automatically selects CUDA on Windows.

Fixed-TCN training code: `direct/collaborator/comp0245-tcn-release/LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task/tcn/train_structured_arm.py`. Its original dataset layout is described in that release. Direct-torque collection/splitting/training code is mapped in `REPORT_CODE_MAP.md`.

Do not launch `structured/dynamic_pd/bootstrap.py`, the original `dynamic_pd.train` or original `dynamic_pd.evaluate` as first-time portable entry points: their author-workspace checks include historical absolute paths and the original ZIP. They are retained for provenance. The portable `tools/retrain_gain.py` and `tools/replay_benchmark.py` are the supported new-workspace entry points.

The old `direct/tools/prepare_workspace.py` prepares only its original baseline workspace. It must not be used to mass-rewrite research or structured-model metadata. See `direct/docs/BASELINE_RELEASE_20261103.md` when deliberately restoring that earlier pipeline.
