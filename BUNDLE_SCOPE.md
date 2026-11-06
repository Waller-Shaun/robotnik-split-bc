# Scope and limits of this submission

## Included

- All source code, guides, configurations, split manifests, result records and main runnable models from the curated direct/research/patrol publication; historical research PT files and recovery NPZ arrays are omitted as listed below.
- The independent dynamic-PD bundle in its original layout, including its immutable fixed-TCN baseline.
- Demonstration collection, split/merge, feature, model, training, recovery, test, plotting and patrol code relevant to the report.
- Original configurations, normalization, split manifests, main model checkpoints, training histories, result records and method/source documentation.
- Metadata, diagnostics and code for the 102-trajectory recovery collection, plus the small release examples. The raw 102 NPZ arrays are in the complete backup only.
- New English overview, supported entry-point dispatcher, report-to-code map and submission integrity tools.

Original source copies are byte-identical to the input release trees, as recorded by `SOURCE_INVENTORY.json`. The separate `SUBMISSION_MANIFEST.json` lists what is actually in this ZIP. Duplicated baseline subtrees are deliberate: relative paths and original manifests rely on them.

## Not included

- 61 historical research PT checkpoints and 102 raw recovery NPZ arrays (exact paths/hashes in `OMITTED_OPTIONAL_ARTIFACTS.json`). Main demo models are retained.

- The full original 1,131-trajectory base dataset and any additional local data expansions not in the curated archive.
- Full raw single-target or patrol evaluation rollouts. Per-trial summaries and historical audit reports are retained.
- Python/Pixi environments, simulator Git checkouts/submodules, robot meshes, feature caches, Git internals, user credentials or private documents.
- The draft report PDF and its review notes. Submit the final report/contribution statement separately.
- Recent local TCN-patrol/test adapters that are not the D3 patrol experiments described in this report. The supported submitted patrol remains the frozen D3 application.

The coursework states that data and pretrained weights need not be included. We retain main models and recovery metadata for inspection/convenience; the optional historical weights and raw recovery subset remain in the complete backup.

## Portable execution versus historical archives

The top-level dispatcher supports fixed-TCN, dynamic-PD and D3-patrol demos, paired replay and portable gain retraining. Their original model code is retained unchanged.

Historical direct-torque experiments in `direct/research/` retain original absolute `F:/...` paths, parent-checkpoint relationships and hashes. They support verification and code inspection. Recreating every historical run requires the matching dataset, initialization, environment and explicit relocation; the package does not promise one-command retraining of all experiments. Do not bypass source/hash checks by editing expected hashes.

The original dynamic-PD author-workspace trainer/evaluator checks Mac workspace files and the old ZIP. Portable `structured/tools/retrain_gain.py` and `structured/tools/replay_benchmark.py` are provided for a new machine. Their outputs are new runs, not replacements for the selected checkpoint or inherited performance claims.

Installing the shared upstream environment requires internet and is a user action. It was not done while packaging. On Windows use UTF8 (`python -X utf8` or `PYTHONUTF8=1`) because some frozen source-release code uses the platform-default text encoding.

The D3 local historical URDF and the teammate's locked asset manifest have different byte hashes; this packaging task did not establish whether the difference is only asset paths/formatting. Sharing an asset root is a supported loading layout, not proof of bitwise or physical trajectory equivalence to the old patrol environment. The original model/configuration bytes and source hashes remain available for inspection.

## Validation performed while packaging

`VALIDATION.json` records the actual checks. These include hashes for the actually included source files, Python AST parsing, JSON parsing, unchanged dynamic release integrity, recorded-count checks, dispatcher checks and ZIP member checks. The original full research archive verification passed for the complete backup; it is not applicable to this reduced upload because 163 optional binary files are omitted. None establishes a new success rate. No training, data collection or simulation was started.

Original historical audits of raw trajectories remain clearly identified as historical. The package does not contain enough raw evaluation data to independently rerun every such audit without obtaining those archives or generating fresh rollouts. Recounting JSON/CSV scores is not equivalent to re-auditing physical truth positions.

## Submission location

The prepared ZIP is intended as the code attachment for the assignment. It has not been uploaded to Moodle or GitHub by this preparation task. The final report should cite the submission's component folders or supported commands, rather than only saying that dynamic-PD code is supplied separately.


## GitHub main variant

The GitHub main variant adds bilingual README and publication/Git metadata to the assignment ZIP contents. Its submission manifest is regenerated for this actual tree; original source hashes are unchanged. Historical ZIP validation records remain historical. See `GITHUB_PUBLICATION.md` for the original ZIP checksum and preserved baseline reference.
