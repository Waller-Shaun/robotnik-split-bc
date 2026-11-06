# State-dependent learned P/D: independent experiment

This folder is separate from the submitted frozen release and GitHub version.
Nothing here replaces the old 99.07% result, which belongs to a different
target set. See `EXPERIMENT_REPORT.md` for this completed paired study.

**Completed 2026-11-05:** 500 previously unused goals x3 shared noise seeds,
1500 trials per model. Fixed P/D: **1495/1500 = 99.67%**; learned dynamic P/D:
**1492/1500 = 99.47%**. Paired difference −0.20 percentage points, clustered
95% interval [−0.80, +0.27]; no demonstrated success-rate improvement.
All 3000 raw archives and 9000 sampled true-state FK positions passed audit;
10 regression tests passed. The old release is not replaced. See
`RESULTS_CN.md`, `FINAL_RESULTS.json`, `RAW_ARCHIVE_AUDIT.json` and
`COURSEWORK_CHECK.md`. This is a completed code/model experiment, not a
finished written coursework report or instructor approval of a hybrid design.

Method: a frozen supervised TCN motion/gravity network and frozen learned
navigation are extended with a state-conditioned neural gain head. Seven P
and seven D gains are learned using expert **joint torques**, not fabricated
variable-gain labels. Initial gains are exactly 100/30. P is bounded to
80..120 and D to 24..36; a 10 ms causal gain filter limits abrupt changes.
The full predictor emits four wheel commands and seven joint torques.

The joint-torque map retains a disclosed PD structure and a reference
integrator. It is a structured/hybrid supervised policy, not an unrestricted
direct-torque TCN. No claim is made that a learned PD head automatically
resolves the coursework interpretation; instructor approval may still be
necessary. No expert IK or analytical gravity oracle is called at inference.

Only the new gain head is trained. Motion/navigation weights, feature
normalization, contact physics, action limits, noise and reach criterion stay
unchanged. Recorded expert reference positions are used for the OFFLINE
torque loss only; online references come from the policy's own integration.
This teacher-forcing mismatch is a limitation, not hidden online supervision.
The expert gains were constant, so learned gain values are not uniquely
identifiable physical parameters and need not outperform the teacher.

## Run the trained model (no original dataset needed)

Run these commands **from this experiment directory**. Python, Git and Pixi
must be installed. Do not run `bootstrap.py` in an already prepared bundle.

```bash
python baseline/tools/setup_dependencies.py
pixi install --manifest-path baseline/external/RoboEnv/pixi.toml -e ml --locked
pixi run --manifest-path baseline/external/RoboEnv/pixi.toml -e ml --locked python tools/verify_experiment.py --runtime
pixi run --manifest-path baseline/external/RoboEnv/pixi.toml -e ml --locked python tools/run_tests.py
pixi run --manifest-path baseline/external/RoboEnv/pixi.toml -e ml --locked python -m dynamic_pd.demo --target -0.9 1.2 1.1 --seed 137
```

The last command opens a viewer. Add `--headless` for no window or
`--variant fixed` to compare the unchanged old policy. The noise profile
remains enabled in both variants. New demo files appear under `outputs/demos/`;
no training or model updates happen during inference.
The shared rollout's `Policy: epoch ...; offline val MSE ...` line describes
the **frozen parent motion model**, not the newly trained gain head. The gain
head's selected epoch and validation loss are recorded separately in
`runs/full/training_finished.json` (selected epoch 17 for this bundle).

After the original final test has finished, replay its published paired
benchmark on another computer without the original author's workspace:

```bash
# Pipeline check: only 2 goals, NOT a final success rate:
pixi run --manifest-path baseline/external/RoboEnv/pixi.toml -e ml --locked python tools/replay_benchmark.py --workers 2 --limit 2
# Full 500 goals x3 repeats x2 variants, fresh outputs only:
pixi run --manifest-path baseline/external/RoboEnv/pixi.toml -e ml --locked python tools/replay_benchmark.py --workers 6
```

Replays write fresh `outputs/replays/` directories and never replace the
recorded final score. These published targets are no longer an unseen test
for subsequent model tuning. Platform/library differences can change results.

## New portable retraining run (separate results)

With the same original dataset restored under a directory containing
`RoboEnv/`, a fresh training run can be created on another machine without
the author's absolute paths:

```bash
pixi run --manifest-path baseline/external/RoboEnv/pixi.toml -e ml --locked python tools/retrain_gain.py --data-root "/path/to/original_dataset_project" --output "/path/to/new_gain_run" --smoke-test
# Omit --smoke-test and use another NEW output directory for full training.
pixi run --manifest-path baseline/external/RoboEnv/pixi.toml -e ml --locked python -m unittest discover -s tools -p test_retrain_gain.py -v
```

This never overwrites the recorded head. It reuses the same supervised loss,
frozen encoder and original train/validation split, but resets the training
seed **after** feature preparation to avoid the cache-hit/cache-miss RNG
initialization dependence in the archived author-workspace trainer. The
archived training source/checkpoint is retained exactly for result provenance;
the portable run is a NEW experiment, not a promise of identical weights or
the recorded success rate. Load its checkpoint explicitly with
`DynamicPDController(checkpoint=...)` in a separate study. Do not replace
`runs/full/best_model.pt` or reuse published final targets as unseen data
when tuning a new model.

## Original experiment audit

The following are the **original author's workspace commands**, executed
inside the locked ML runtime. They require the read-only original dataset,
release/ZIP and source paths recorded in `protocol.json`/`local_paths.json`.
They deliberately reject overwriting existing runs or changing archived
sources, and are not the portable inference instructions above. A new study
must use a fresh experiment folder, fresh model selection and fresh final
targets, rather than reuse final outcomes for tuning.

```bash
python -m unittest dynamic_pd.test_dynamic_pd -v
python -m dynamic_pd.train --smoke-test
python -m dynamic_pd.train
python -m dynamic_pd.evaluate development --workers 6
python -m dynamic_pd.evaluate final --workers 6
python -m dynamic_pd.summarize
python tools/audit_raw.py
python tools/plot_results.py
```

Bootstrap copies the 118-file frozen release byte-for-byte into `baseline/`;
it does not copy a Python environment, model meshes or the raw dataset.
Training reads the original dataset without modifying it; `local_paths.json`
records that local read-only dataset location. Retraining elsewhere requires
the same 1131-trajectory dataset and a correctly set dataset root.
`bootstrap.py` is a one-time preparation utility for a new folder adjacent
to the author's original project and release, not an installation step for
the supplied trained bundle. Do not edit frozen source hashes to bypass
the original experiment audit.

Training uses original 698 train /144 validation trajectories. Selection is
by offline validation loss; the 100 known development targets are a closed-loop
diagnostic, not the final score. Final evaluation uses the previously unused
500 reserve targets, each with 3 predeclared noise seeds, paired against the
original fixed-gain model: 1500 episodes per model. Targets are not filtered
for expert/model reachability. Errors remain in the denominator. The final
test may only run after checkpoint and source hashes are frozen.

Noise: native eight Gaussian getter standard deviations 0.001; fixed initial
state; 20 s simulated horizon; 1 ms physics/control; anytime TRUE end-effector
error below 1 cm; no extra hold requirement. Estimated noisy state goes to
the policy, true state to scoring only. New histories, references and gain
filter state are reset each episode. True scoring positions/errors and actions
are retained at the full 1 ms rate; large diagnostic observations/robot states
are stored at 10 ms because disk space is limited. Scoring/control are NOT
downsampled. Applied/raw gain traces are recorded at 10 ms.

Inspired by gain-aware policy analysis in
[Tune to Learn](https://arxiv.org/abs/2604.02523), and state-dependent
impedance learning in [SCAPE](https://proceedings.mlr.press/v164/kim22b.html).
This is a custom supervised gain-head experiment, **not** a reproduction of
SCAPE/VICES reinforcement learning. Bounds/filter/regularizers are empirical
design constraints, not a closed-loop stability proof. Report failed results
and limitations rather than relaxing the score or substituting old results.

## Code sources / what changed

`baseline/` is the byte-identical previous release (including its source,
provenance, dependency commits and existing learned models); see its
`docs/PROVENANCE.md`. The upstream simulator and submodule are fetched by Git
at the recorded commits, not bundled as modified third-party code. Its native
noise getters, contact physics, action limits and scoring loop are unchanged.

New custom code is in `dynamic_pd/`: the gain head, expert-torque dataset
adapter, supervised trainer, state-reset/filter integration, paired evaluator,
unit tests and clustered result audit. `tools/` adds portable hash checks and
fresh-output benchmark replay. Training and simulation use the pinned upstream
environment. No claim is made that these files reproduce a paper's full method.
The code was prepared with AI assistance; you must review/understand it and
make any disclosure required by your course. See `COURSEWORK_CHECK.md` for
the exact submission boundary and remaining report/instructor decisions.
