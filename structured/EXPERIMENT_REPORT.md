# State-dependent P/D: final independent paired experiment

| Model | New targets | Noise repeats | Success / trials | Success rate |
|---|---:|---:|---:|---:|
| Fixed P=100, D=30 | 500 | 3 | 1495/1500 | 99.67% |
| Learned state-dependent P/D | 500 | 3 | 1492/1500 | 99.47% |

Dynamic minus fixed: -0.200 percentage points;
target-cluster stratified bootstrap 95% interval: [-0.8, 0.26666666666666666].
Paired rescues: 3; paired regressions: 6.
Process errors: fixed 0, dynamic 0 (included in denominators).

This is the originally unused reserve: 500 new goals x3 predeclared noise seeds,
not 1500 independent goals. Shared fixed initial pose, 20 s simulation horizon,
1 ms physics/control, true EE error <1 cm anytime, no hold requirement. Same
native eight noise standard deviations 0.001, same simulator/contact settings.
All histories, integrated references and learned-gain filter state reset each
episode. All full-rate true scoring trajectories/actions were audited; large
diagnostic sensor/robot-state arrays were stored at 10 ms, not at every tick.
No test trajectory was used
to train or select the gain head. No final failures were removed.

## What was learned

A 128->32->14 neural head predicts bounded P in 80..120 and D in 24..36 from
frozen causal TCN features. Training used expert torque supervision on the
original 698 train /144 validation split, with deviation and temporal
regularization. Motion/gravity/navigation networks stayed frozen. A 10 ms
gain filter was applied online. Exact source, model and configuration hashes
were frozen before final outcomes. `FINAL_RESULTS.json` reports per-joint
gain ranges/variation and every final failure.

## What this does NOT establish

This is still a disclosed structured/hybrid controller with a PD torque map
and reference integrator; learning gains is not proof of instructor acceptance.
The coursework clearly requests supervised state/target->robot-command
regression; this implementation trains on expert final torques and its full
predictor returns wheel commands and torques, but its structured formulation
should be explicitly described and confirmed if the instructor requires an
unrestricted direct-torque regressor. No stability theorem or real-robot
safety guarantee is claimed. Bounds, filtering and tests are empirical checks.

Expert gains were constant. Learned P/D are compensatory policy parameters,
not uniquely identified physical gains. Offline expert reference positions
are used for the training loss only, whereas online references are integrated
from learned velocity; this teacher-forcing gap remains. Gain uncertainty,
multiple training seeds, arbitrary initial poses, obstacles and external
pushes were not comprehensively evaluated. Paper inspiration is not a full
reproduction of SCAPE/VICES, which use different tasks/learning methods.

The old 99.07% is a different target set's historical result, not the score
of this dynamic-gain model. The old ZIP/GitHub release was not overwritten.
Do not claim dynamic gains improve performance if the paired confidence
interval includes zero. Do not silently adopt the candidate if it regresses.
