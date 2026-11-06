# Report-to-code and evidence map

This map distinguishes a historical reference model from an initialization checkpoint and separates complete evaluations from partial/negative studies. It describes existing records, not new experiments.

Path prefixes below:

- **S** = `direct/research/source/LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task/`
- **A** = `direct/research/artifacts/RoboEnv/`
- **T** = `direct/collaborator/comp0245-tcn-release/LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task/tcn/`

## Data collection, splitting and features

| Report content | Source location | Supporting material |
|---|---|---|
| Expert arm/navigation control and PD100/30 | S / `robotnik_arm_control.py` | Explicit `ArmControllerParams`; external course source acknowledged |
| Expert state/action/reference recording | S / `robotnik_arm_task_3.py` | NPZ observation/action, EE, quaternion and reference fields; task3 also supports bounded torque pulses/holding |
| Original collection and ring probing | S / `collect_expert.py`, `collect_expert_ring.py`, `collect_expert_ring_probe.py`, `collect_expert_z_probe.py` | Original reports and configurations retained in A |
| Ring931 and radius>=0.8 subset798 | S / `split_expert_data_ring.py` | Frozen split manifests in A and both structured baseline copies |
| Low-height supplement200 and merged998 | S / `collect_expert_ring_lowz_supplement.py`, `merge_expert_data_ring_lowz.py` | 698/144/156 split; supplement horizontal groups remain in one split |
| Body-yaw relative target/error features | S / `robotnik_bodyframe_features.py` | Direct arm26 retains world base position/yaw; do not claim complete invariance |
| Navigation features, phases and small MLP | S / `robotnik_navigation_policy.py`, `train_nav_bc_ring_r08.py` | Frozen nav run `bc_training_nav_ring_r08/run_20261103_022339_570160` |

The full base dataset is not in the ZIP. Source collection scripts preserve their original working layouts; follow the dataset documentation and explicitly configure a new data root when collecting/retraining. Do not expect archived Mac or F-drive paths to exist on a new machine.

## Direct-torque optimization studies

Each A run retains its `config.json`, normalization, split, history, source snapshots, `ARCHIVE_README.md` and available per-trial records. The 61 historical research PT files are omitted from this upload variant; main runnable model bundles remain. Its own source snapshots are more authoritative than later default settings in S.

| ID / study | Main training/method code in S | Evaluation code in S | Recorded run / result |
|---|---|---|---|
| P0 published baseline | `finetune_arm_bc_ring_lowz_supplement.py`; earlier pipeline in `direct/LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task/` | `test_bc_split_nav_arm_lowz_supplement.py`, `test_bc_split_nav_arm_mlp_noise.py` and batch variants | A / `bc_training_arm_ring_r08_lowz_supplement/run_20261103_141232_983306`: clean111/144; noisy106/144 |
| C0 torque-only continuation | `train_arm_bc_ring_aux_b2.py`; run configuration has `aux_weight=0` | `test_bc_split_nav_arm_aux_b2.py`, `batch_test_bc_split_nav_arm_aux_b2_parallel.py` | A / `bc_training_arm_ring_r08_aux_b2_control/run_20261104_025706_206181`: noisy113/144 |
| B1 GRU | `train_arm_bc_ring_gru.py`, `resume_arm_bc_ring_gru.py`, `robotnik_arm_gru_policy.py` | `test_bc_split_nav_arm_gru.py`, serial/parallel GRU batch scripts | A / `bc_training_arm_ring_r08_gru_b1/`: first112/144 and extended113/144, clean; other continuation checkpoints not independently ranked |
| B2 reference auxiliary | `train_arm_bc_ring_aux_b2.py`, corresponding auxiliary policy/data | Auxiliary single/parallel evaluators | A / `bc_training_arm_ring_r08_aux_b2/`: offline experiments; no complete standalone144-point gain claimed |
| B3 gradient gating | `train_arm_bc_ring_gradgate_b3.py`, `arm_aux_gradient_gate.py` | `test_bc_split_nav_arm_gradgate_b3.py`, parallel batch | A / `bc_training_arm_ring_r08_gradgate_b3/run_20261104_034810_928084`:113/144 |
| B4 velocity auxiliary | `train_arm_bc_ring_velocity_aux_b4.py` and policy/data | `test_bc_split_nav_arm_velocity_aux_b4.py`, parallel batch | A / `bc_training_arm_ring_r08_aux_velocity_b4/run_20261104_041750_567925`:109/144 |
| Recovery collection | `collect_arm_recovery.py`, `diagnose_arm_recovery.py`, `recovery_dataset.py` | Paired diagnostics/collection export | `direct/research/data/expert_recovery_data/collection_20261104_112547_770894/`:collection manifest records102 expert-only held continuations,233150 frames; raw recovery arrays omitted; diagnostic outcomes are not pure-model success |
| D3 recovery fine-tuning | `finetune_arm_recovery.py` | `batch_test_arm_recovery_parallel.py`, shared auxiliary single evaluator | A / `bc_training_arm_ring_r08_recovery/run_20261104_152933_829681`: best138; noisyval120/144; cleanval119/144; noisytest132/156 |
| R1 supervised residual on C0 | `train_arm_residual.py`, residual policy | `test_arm_residual.py`, parallel batch | A / `bc_training_arm_ring_r08_residual/run_20261104_171630_415142`:118/144 |
| R2 supervised residual on D3 | `train_arm_residual120.py`, `recovery_residual120_context.py`, residual policy | `test_arm_residual120.py`, parallel batch | A / `bc_training_arm_ring_r08_residual120/run_20261104_174723_237778`:122/144 validation only |
| TD3 residual | `train_arm_rl_residual.py`, RL env/policy | `test_arm_rl_residual.py`, parallel batch | A / `bc_training_arm_ring_r08_rl_td3/run_20261104_200609_974210`:20k interaction steps; no complete144-goal improvement |
| Conservative TD3 | `train_arm_rl_residual_conservative.py`, env/policy/validation | Conservative single/parallel evaluators | A / `bc_training_arm_ring_r08_rl_td3_conservative/run_20261104_205012_830783`: selected step0 zero-residual parent |
| CNN-Transformer | `train_arm_bc_cnn_transformer.py`, corresponding data/policy | `test_bc_split_nav_arm_cnn_transformer.py`, parallel batch | A / `bc_training_arm_ring_r08_cnn_transformer/run_20261104_221010_025438`: selected86;2/38 of144 before interruption, not full-set success |

P0 and the first GRU share an earlier parent `bc_training_arm_ring_r08_finetune/run_20261103_111655_849512`; the GRU was not initialized from P0's final checkpoint. Differences between their final scores do not isolate history as the only changed factor. C0/D3 and fixed/dynamic-PD comparisons must also retain their own stated controls and limits.

Full historical indexes: `direct/research/MODEL_INDEX.md`, `direct/research/model_index.json`, `direct/RESEARCH_ARCHIVE.md` and run-specific `ARCHIVE_README.md` files. The original `direct/research/verify_archive.py` requires the omitted historical checkpoints and raw recovery arrays. Use the top-level `submission.py verify` for this scoped upload; use the complete backup for the original full-archive completeness check.

## Structured reference control and gain learning

| Component | Code | Model/result evidence |
|---|---|---|
| Fixed reference/PD architecture | T / `structured_arm.py`, `test_structured_arm.py` | `structured/baseline/models/tcn/`; fixed TCN SHA `eb1defbcf8ff787c897d8a202d2c0c22be59e2d1b9c259546ccc607722c4ca1e` |
| Original supervised structured training | T / `train_structured_arm.py` | Original698/144 training/validation split; reference-velocity and reconstructed-gravity supervision plus teacher-forced torque term; no online recorded references |
| Structured-MLP comparator | Same T architecture/trainer with `--model mlp` | Supplied model, but no completed matched prospective closed-loop result claimed |
| Fixed-TCN primary evaluation | `structured/baseline/run_model.py`, `run_batch.py` | `structured/baseline/results/recorded_primary/friend/results.json`:1486/1500; `off/results.json`:493/500 |
| Dynamic gain network | `structured/dynamic_pd/model.py`, `controller.py` | P80–120,D24–36;128→32→14;4590 trainable parameters; original motion/navigation frozen |
| Gain data and historical training | `structured/dynamic_pd/dataset.py`, `train.py` | `structured/runs/full/`:selected17, original configs/history; uses expert references only offline |
| Portable new gain retraining | `structured/tools/retrain_gain.py` | Explicit data root and new output; fixes cache-dependent initialization; new experiment, no inherited score |
| Paired fixed/dynamic evaluation | `structured/dynamic_pd/evaluate.py` (historical), `structured/tools/replay_benchmark.py` (portable replay) | `structured/outputs/final/results.json`, `protocol.json`, `FINAL_RESULTS.json`:fixed1495/1500,dynamic1492/1500;500 distinct goals×3 seeds |
| Gain reset/filter and method checks | `structured/dynamic_pd/test_dynamic_pd.py`, `structured/tools/test_retrain_gain.py`, `structured/tools/verify_experiment.py` | Historical audits, manifests and portable verification; bounds/filter are not a stability proof |

The original fixed release is deliberately retained inside both source components. The unified demo uses `structured/baseline` so fixed and dynamic candidates can share one installed upstream environment. No original model/source bytes are rewritten.

## Continuous multi-target inspection

| Component | Path |
|---|---|
| Runnable entry | `direct/patrol/test_patrol_120.py` |
| Frozen original-byte D3 and navigation loader | `direct/patrol/patrol_checkpoint_bundle.py`; `direct/artifacts/patrol120/manifest.json` |
| Relative per-leg parking | `direct/patrol/patrol_parking_geometry.py` |
| Visit order: exact up to8, approximate above8 | `direct/patrol/patrol_route_planner.py` |
| Random targets and planned display | `direct/patrol/patrol_target_sampling.py`, `patrol_route_display.py` |
| Observed task switch and independent true holding | `direct/patrol/patrol_task_state.py` and entry |
| Existing mission records | `direct/patrol/results/index.json` and dated folders; report missionA is `patrol_20261105_122839_050084`, missionB is `patrol_20261105_123130_010104` |
| Static algorithm/display checks | `direct/patrol/check_patrol_*.py`, `verify_bundle.py` |

The report's patrol is a separate D3 application, not a continuous-mission test of the high-success TCN. It preserves actual robot state between visits, optimizes an estimated parking-distance objective, and scores true5cm holding separately from the observed switch. Mission failures and unattempted goals remain visible; no pooled independent mission success rate is claimed.
