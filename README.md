# COMP0245: Learning-Based Mobile Manipulation

**English** | [简体中文](README.zh-CN.md)

**Group:** Mingzhou Li, Shuchang Zhang, Boyan Li — University College London.  
**Prepared:** 6 November 2026.  
**Task:** supervised mobile-manipulator position reaching, plus a separate continuous-inspection application.

This is the smaller assignment upload bundle. It retains all report-related source code, main runnable model bundles and recorded evidence; optional historical research weights and raw recovery arrays are omitted. It does **not** train a new model or rerun any experiment during preparation. Original experiment sources, weights, normalization and records are copied byte-for-byte; only this top-level guide, dispatcher and static verification tools are new.

Start here:

- [QUICKSTART.md](QUICKSTART.md): installation, model demos, batch replay, expected outputs.
- [REPORT_CODE_MAP.md](REPORT_CODE_MAP.md): report experiment IDs mapped to source, models and results.
- [SOURCES_AND_MODIFICATIONS.md](SOURCES_AND_MODIFICATIONS.md): external code, group modifications and attribution.
- [BUNDLE_SCOPE.md](BUNDLE_SCOPE.md): included/omitted material, historical-path limitations and verification boundaries.
- `python -X utf8 submission.py verify`: hashes, Python syntax and JSON checks; no simulator, training or network.
- `python -X utf8 submission.py results`: recount the supplied fixed/dynamic-TCN result records; no new evaluation.

## Repository branches

| Branch | Contents |
|---|---|
| [`main`](https://github.com/Waller-Shaun/robotnik-split-bc/tree/main) | Current assignment submission: direct-torque research, structured TCN/fixed and dynamic PD, D3 multi-target patrol, models and recorded evidence |
| [`baseline`](https://github.com/Waller-Shaun/robotnik-split-bc/tree/baseline) | The previous `main`, preserved unchanged at commit `5073b83768a4c4931dbe146558f059d8ab2d70b7` |
| [`research/experiments-20261103-05`](https://github.com/Waller-Shaun/robotnik-split-bc/tree/research/experiments-20261103-05) | Earlier complete research archive, including optional historical weights/recovery fragments |
| [`feature/multi-target-patrol-120`](https://github.com/Waller-Shaun/robotnik-split-bc/tree/feature/multi-target-patrol-120) | Earlier standalone D3 patrol release |

Clone the default branch to inspect the current submission:

```bash
git clone https://github.com/Waller-Shaun/robotnik-split-bc.git
cd robotnik-split-bc
python -X utf8 submission.py verify
python -X utf8 submission.py results
```

Simulation installation and explicit run commands are in [QUICKSTART.md](QUICKSTART.md). Cloning or static verification does not start training or simulation. [GITHUB_PUBLICATION.md](GITHUB_PUBLICATION.md) identifies the assignment ZIP and the branch-preservation policy.

## Directory structure

```text
COMP0245_Code/
  README.md, QUICKSTART.md, REPORT_CODE_MAP.md
  SOURCES_AND_MODIFICATIONS.md, BUNDLE_SCOPE.md
  submission.py                     explicit command dispatcher
  submission_tools/                 read-only integrity/result checks
  direct/                           original direct-torque and patrol release
    LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task/
                                    initial baseline collection/split/training/testing
    research/source/.../robotnik_arm_task/
                                    later GRU, auxiliary, recovery, residual, RL and Transformer code
    research/artifacts/RoboEnv/      historical configs, normalization and results; PT omitted
    research/data/                  recovery manifests and diagnostics; raw arrays omitted
    patrol/                         portable D3 continuous-inspection application
    artifacts/patrol120/            original D3 arm/navigation weights, statistics and config
    collaborator/comp0245-tcn-release/
                                    original fixed-TCN release and its training source
    environment/, tools/, docs/      original baseline environment and preparation instructions
  structured/                       original independent dynamic-PD experiment bundle
    baseline/                       immutable fixed-TCN release, with its own model and provenance
    dynamic_pd/                     learned gain head, dataset, training, controller and evaluation
    tools/                          portable setup checks, benchmark replay and gain retraining
    runs/full/                      selected epoch-17 gain head and training records
    outputs/development/, outputs/final/
                                    per-trial records, not raw rollout archives
    FINAL_RESULTS.json, METHOD.md, README.md
  SOURCE_INVENTORY.json              provenance and hashes of copied files
  SUBMISSION_MANIFEST.json           hashes of the actual submission files
  VALIDATION.json                    packaging checks and their limits
```

`direct/` and `structured/` intentionally retain their internal directory layouts. Their apparently duplicate baseline models are needed by their original path and integrity checks. Do not merge or rename their internal files, rewrite archived hashes, or substitute one model's normalization for another's.

## The three runnable applications

1. **Fixed-PD structured TCN, single target.** A learned movement field and gravity estimate generate motion references; an integrator and fixed joint PD100/30 generate arm torques. Frozen learned navigation supplies wheel velocities. This is a disclosed hybrid controller, not unrestricted direct-torque regression.
2. **State-dependent-PD structured TCN, single target.** The same frozen motion/navigation models plus a learned 4,590-parameter head. Seven P and seven D requests stay within P80–120 and D24–36, with a causal gain filter. Only the gain head was trained in this ablation.
3. **D3 recovery-MLP multi-target patrol.** Frozen learned navigation and direct arm-torque policies retain physical state between visits. Relative parking, visit-order planning and a task monitor extend the single-goal skill. This application does **not** use the TCN-PD arm and does not perform object grasping or obstacle avoidance.

Use the explicit commands in [QUICKSTART.md](QUICKSTART.md). Importing the dispatcher or asking for `--help` does not start a simulation. Training and evaluation are separate commands. No script automatically starts training after a demo/test.

## Recorded results and interpretation

| Study | Protocol / data | Recorded outcome |
|---|---|---|
| P0 direct-torque published baseline | 144 validation goals, clean / noisy | 111/144 / 106/144 |
| C0 torque-only continuation | same 144 noisy validation goals | 113/144 |
| D3 expert-recovery fine-tuning | same noisy validation; separate noisy test | 120/144; 132/156 |
| R2 supervised residual on D3 | same noisy validation only | 122/144; no completed 156-goal test claimed |
| Fixed-TCN primary study | 500 new goals ×3 noise seeds | 1486/1500 (99.07%); clean493/500 |
| Fixed-PD paired reserve control | another 500 goals ×3 noise seeds | 1495/1500 (99.67%) |
| Learned-PD paired reserve candidate | SAME reserve goals/seeds | 1492/1500 (99.47%); paired change −0.20 percentage points, interval includes zero |
| Continuous patrol | individual development missions | successful and failed missions retained; no pooled independent mission success rate |

Single-goal success is **true PyBullet end-effector error <1cm at any time within20s**, without a holding requirement. The nominal noise protocol uses eight native Gaussian getter standard deviations0.001. These are measurement-noise scales, not force disturbances. Patrol instead switches after **observed error <5cm continuously for1s**, with true holding evaluated independently. These scores must not be interchanged.

The group does not attribute the complete TCN-hybrid improvement solely to temporal encoding or PD: features, supervision and control structure changed together. Fixed100/30 came from the demonstration-generating expert, not a demonstrated optimum. Dynamic gains reduced offline torque MSE without a verified success-rate improvement. Failed RL and partial CNN-Transformer runs are preserved as negative/incomplete studies, not full benchmark wins.

The old156-goal dataset, the primary500-goal test and the reserve500-goal study are different protocols. Published test targets are no longer unseen data for subsequent tuning. Hardware/runtime differences may change rerun outcomes.

## Code/data availability

Main runnable model checkpoints and small source-release examples are included for convenience. The 61 historical research PT checkpoints and 102 raw recovery NPZ arrays are omitted; their configurations, source snapshots, split/collection manifests and result records remain. The original multi-GB 1,131-trajectory dataset, full raw evaluation rollouts, robot meshes, simulator checkout and Python/Pixi environments are also omitted. See `OMITTED_OPTIONAL_ARTIFACTS.json` for exact exclusions. The complete backup ZIP preserves the optional artifacts; the upload package uses its own scoped verifier, not the original full-archive completeness test. No private Word files or draft report PDF are included; submit the final report separately.

Archived research sources retain some historical machine-specific paths. Their exact configurations/snapshots support inspection; they are **not** all promised to run after a single path substitution. Portable main demos and gain retraining/replay are identified separately. See [BUNDLE_SCOPE.md](BUNDLE_SCOPE.md).

All dependencies are installed by the user into the pinned upstream environment; preparation of this ZIP did not install dependencies, access the network, train or simulate. Do not run the archived one-time bootstrap on this prepared bundle. Start with the supported setup instructions.
