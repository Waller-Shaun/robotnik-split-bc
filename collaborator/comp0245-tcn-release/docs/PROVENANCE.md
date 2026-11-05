# Provenance, authorship and excluded material

This release packages the current local COMP0245 robot-controller experiments. It does not claim that the teaching template or upstream simulator was authored by the student.

- Original shared project: https://github.com/Waller-Shaun/robotnik-split-bc . Original dataset and parent MLP/navigation metadata are preserved under `artifacts/` and `dataset_info/`.
- Simulator: https://github.com/VModugno/RoboEnv , pinned at `7bb49439677c2ac111347b40e997025ce87bff50`.
- Simulation submodule: https://github.com/VModugno/simulation_and_control , pinned at `df5850526affcc33b25b15d3096c4d21b614bacd`.
- Current extension: temporal arm encoder, disclosed structured hybrid control, matched MLP, native noise protocol with separate true-state scoring, fast live viewer and prospective evaluation.

No new blanket open-source license is imposed on inherited teaching/source material. Respect each upstream repository's license and the university's coursework sharing rules; a private repository does not itself grant redistribution rights. Review coursework rules before making this public.

Current TCN checkpoint SHA-256:

`eb1defbcf8ff787c897d8a202d2c0c22be59e2d1b9c259546ccc607722c4ca1e`

Checkpoints, normalization, inference source, data splits and historical results are not rewritten to alter outcomes. The portable entry points resolve current model folders locally, rather than trusting embedded historical absolute paths. Source-copy records and exact package/dependency hashes are in `release_manifest.json`.

Excluded: full training trajectory dataset, full raw evaluation trajectories/logs, local Python/Pixi environments, third-party Git checkouts/meshes, temporary PDF images, coursework PDF, cached bytecode, rejected repair model weights and access credentials. Original files remain in the local working project. This release contains outcomes and an audit summary, not the full raw archive necessary to repeat every historical offline audit without further data.

The unused reserve target list is not included. Rejected recovery experiments are documented as unsuccessful attempts; none replaces the frozen deployed model.
