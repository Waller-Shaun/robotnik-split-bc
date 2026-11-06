# Implemented method and reproducible design choices

## Scope of this ablation

This experiment changes only arm tracking gains. The previous causal TCN
motion/gravity model and learned base-navigation network are frozen. It is
not a new MLP-versus-TCN comparison, not a whole-policy retraining run, and
not reinforcement learning. Both paired conditions receive identical targets
and noise seeds and use the same observation/action/physics pipeline.
The base wheel driver's lower-level servo gains (400/30), hand-coded phase
switching and height-dependent navigation standoff rule remain fixed. They
are shared components, not outputs learned by the new arm-gain head. Returned
expert wheel speeds from the phase helper are discarded, not used as actions.

The gain network is `128 -> Linear(32) -> SiLU -> Linear(14) -> tanh`, with
4,590 trainable parameters. Its input is the last frozen causal TCN hidden
feature from exactly 20 frames separated by 10 ms. Input normalization is
fitted on training features only. The existing TCN uses left padding at the
start of each episode and never consumes a subsequent frame. The nominal
convolutional receptive field is larger than the supplied history; offline
feature extraction therefore constructs the same exact 20-frame windows as
online inference rather than convolving a whole trajectory.

For joints j=1..7, the raw requested gains are

```
P_requested[j] = 100 + 20*tanh(network_output[j])
D_requested[j] =  30 +  6*tanh(network_output[j+7])
```

The last linear layer starts at zero, making the initial policy exactly the
fixed-P/D parent. The bounds are an empirical local ±20% design constraint,
not learned physical truth or a stability certificate. A causal first-order
filter updates applied gains every 1 ms with `alpha=.001/(.010+.001)`.
All 14 gains, histories, references, traces and counters reset between goals.

## Supervised training, not fabricated gain labels

Expert trajectories provide robot state, joint velocity, reference position
and applied joint torque. For each sampled frame, frozen networks predict
velocity `v_hat` and gravity compensation `g_hat`. The differentiable
training prediction is

```
tau_hat = clip(P*(q_expert_reference-q_measured)
               + D*(v_hat-qd_measured) + g_hat, -30, 30)
```

The target is the recorded seven-dimensional expert **torque**, not a set of
invented state-varying P/D labels. The loss combines normalized torque MSE,
`0.02*mean(tanh_output**2)` deviation regularization and
`0.02*mean((tanh_output_current-tanh_output_previous)**2)` temporal
regularization. Previous frames are within the same episode; episode starts
repeat their first frame. This is a local regularizer on raw requests, not
a claim of optimizing the entire filtered closed-loop trajectory.

The original 698 training and 144 validation trajectories are fixed. Frames
are sampled at 10 ms for learning, with low/other-height groups balanced
50:50 and near-target reaching samples weighted 3x. These weights mean the
weighted training loss is not directly comparable to the unweighted full
validation loss. No final target or rollout enters optimization, early
stopping or normalization. The 156 old held-out trajectories are not used
in this new gain training either.

AdamW: seed 42, learning rate 0.001, weight decay 0.00001, batch size 4096,
200,000 weighted frame draws per epoch, cosine decay to 0.0001, at most
20 epochs, gradient norm clipping at 1, patience 6. Selection uses full
offline validation loss including fixed regularizers. The actual selected
epoch and training trajectory manifest are in `runs/full/`.
The `validation_gain_*` diagnostics in `training_finished.json` use the first
20,000 validation frames, not the entire validation dataset. Model selection
still uses all validation frames. Final gain ranges in `FINAL_RESULTS.json`
are accumulated over all 1,500 dynamic-gain closed-loop trials.

## Online torque output

Online, no recorded expert reference is available or read. The same parent
policy integrates its learned joint velocity into `q_reference`; the same
joint limits, ±0.15 rad reference lead constraint and navigation-phase arm
hold remain. The final predicted torque is

```
tau = P_filtered*(q_reference-q_measured)
      + D_filtered*(v_hat-qd_measured) + g_hat
```

The unchanged rollout applies the original actuator limits. Four learned
wheel commands are concatenated with seven arm torques. No expert inverse
kinematics, analytical gravity oracle or ground-truth state is fed to the
policy. True state is used only for independent scoring.

The offline expert-reference versus online integrated-reference mismatch is
a real limitation. Constant expert gains also make the learned P/D
non-identifiable: multiple gain choices can fit similar torques, and some
joints may barely vary. A smaller validation MSE alone is not evidence of
better closed-loop success or more physically correct gains.

## Independent paired final test

The 100 already known development targets are diagnostic. Final targets are
the previously untouched reserve: 500 unique goals, radius from origin
0.8–2.0 m and z=0.2–1.3 m, stratified by height and radius with angular
jitter. Negative x and y are allowed. This is an annular task distribution,
not every point in a rectangular XYZ box. Goals were not filtered by model
or expert success. `DATA_SEPARATION_AUDIT.json` verifies separation from
all 1,131 historical goals.

Each final goal uses three predeclared unique noise seeds. Fixed/dynamic
conditions share each seed. Eight native Gaussian getter noise standard
deviations are 0.001, matching the configured interpretation of the friend's
`_cov` fields; these are not force disturbances. The initial robot pose,
contacts, phase switching, 20 s simulated horizon, 1 ms control and
anytime TRUE end-effector distance <0.01 m are unchanged. No persistence
or hold time is required. This is a reach criterion, not stable holding.

There are 1,500 trials per model, but only 500 independent target clusters.
Report clustered height-stratified bootstrap intervals and the paired
dynamic-minus-fixed interval, not 1,500 independent-target binomial trials.
Process failures remain in denominators. Candidate source, weights, protocol
and training config were frozen before any final result. No final failure
may be removed or used to tune this reported candidate.

Full-rate true end-effector positions/errors, estimated end-effector
positions, timestamps and raw/applied actions are retained for scoring audits.
Large diagnostic observations/robot states and gain traces are stored at
10 ms to limit disk usage, without changing live control or scoring. The
small submission bundle contains aggregate/per-trial results and provenance;
the full raw archive remains in the local experiment directory.

## Paper inspiration and limits

- Gain-aware policy analysis: [Tune to Learn](https://arxiv.org/abs/2604.02523).
- Learned impedance: [SCAPE](https://proceedings.mlr.press/v164/kim22b.html).
- Variable-impedance action spaces: [VICES](https://arxiv.org/abs/1906.08880).

These sources motivate examining gains, but none establishes that this
particular supervised gain head will improve this robot or be accepted by
the course. This custom experiment is not a reproduction of their full
methods or a claim that their reinforcement-learning results automatically
transfer to behavior cloning. No real-robot safety guarantee is made.

## Retraining reproducibility correction

The original author-workspace trainer seeds before frozen feature preparation.
Constructing frozen networks on a cache miss consumes RNG whereas a cache hit
does not, so exact head initialization can depend on cache state. Its source
and the actual evaluated checkpoint remain archived unchanged: the reported
paired test concerns that exact frozen checkpoint, not an assertion that all
retraining runs will yield identical weights. Reproducing the author workflow
requires a fresh cache and the recorded workspace/data layout.

`tools/retrain_gain.py` supplies a separate portable fresh-run entry point with
an explicit data root/output directory and reseeding AFTER feature preparation.
It does not bypass the original experiment audit, overwrite selected weights,
or inherit their success rate. A dedicated cache-RNG unit test and two small
training/reload checks passed. New training outputs require their own model
selection and a new independent final evaluation. Platform/MPS numerical
differences and training-seed variability are still not eliminated.
