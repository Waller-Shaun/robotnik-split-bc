# Sources, group code and modifications

This is a COMP0245 group project by Mingzhou Li, Shuchang Zhang and Boyan Li. The simulation engine and the original expert controller are external/course resources. We do not present them as newly implemented by the group.

| Component | Source | Use/modification in this work |
|---|---|---|
| Task and demonstration-generating expert | [UCL course public lab repository](https://github.com/VModugno/LAB_SESSION_COMP_0245_2026_PUBLIC/tree/main/final/robotnik_arm_task) | Retained expert controller for offline demonstrations; added recording, phase/context features, collection grids/rings, holding and recovery collection |
| RoboEnv and robot assets | [VModugno/RoboEnv](https://github.com/VModugno/RoboEnv) | Pinned upstream environment and assets; fetched separately by setup, not claimed as group code |
| PyBullet interface and dynamics wrapper | [VModugno/simulation_and_control](https://github.com/VModugno/simulation_and_control) | Upstream submodule; existing SimInterface/PinWrapper used for simulation and observed FK |
| Learning framework | [PyTorch](https://pytorch.org/) | Networks, autograd and optimizers; not a custom learning framework |
| Direct-torque baseline and experiments | Group code, archived in `direct/` | Feature construction; split navigation/arm policies; data splitting; weighted/resampled training; temporal/auxiliary variants; model-visited expert recovery; supervised/RL residuals; evaluations |
| Structured reference TCN | Boyan Li's group contribution, preserved in original release | Learned movement field/gravity, reference integrator, disclosed PD tracking, prospective single-goal evaluation |
| Dynamic PD | Boyan Li's group contribution, preserved in `structured/` | Bounded state-dependent gain head and supervised expert-torque loss; paired reserve evaluation and audit tools |
| Multi-target inspection | Shuchang Zhang's group contribution in `direct/patrol/` | Relative per-leg parking, persistent physical state, route ordering, random target parameters, task monitors and displays |
| Final report integration | Mingzhou Li, with group discussion/results | Report writing and integration; the report's contribution statement gives the final group allocation |
| Submission packaging | New top-level guides, manifest/check tools and dispatcher | Organizes existing results; changes no original model tensor, normalization, score or controller source |

Fixed arm P=100 and D=30 are inherited from `ArmControllerParams` in the expert controller. They were not learned in the fixed-PD study and are not established as optimal. Dynamic P80–120 and D24–36 bounds and the10ms filter are manually designed; the gain head's weights are learned. Wheel-driver400/30 gains, phase rules, reference limits and gripper setup remain explicit shared components. No claim is made that every control parameter is learned.

The original source code contains comments and historical provenance; it is not rewritten into a new unified style. `direct/research/artifacts/` stores run-specific source snapshots, which should be preferred over later generic defaults when interpreting a historical run. Existing licenses/source notices and original method/literature acknowledgements are preserved. No new blanket license is imposed on course or upstream code.

Method inspiration and limits are documented in `direct/research/docs/文献与方法来源.md`, `structured/METHOD.md` and the report. Citing DAgger, residual RL, temporal convolution, SCAPE, VICES or gradient-similarity work does not assert a complete reproduction. In particular, the gain head is supervised learning, not a reproduction of SCAPE/VICES RL.

Code preparation used AI assistance. The group remains responsible for understanding, reviewing and disclosing the work as required by the course. These tools and code archives do not constitute instructor approval of a hybrid design or replace the group's final report.

Upstream pins: RoboEnv `7bb49439677c2ac111347b40e997025ce87bff50`; simulation_and_control `df5850526affcc33b25b15d3096c4d21b614bacd`. Exact dependency information is retained in each original release manifest and Pixi lock.
