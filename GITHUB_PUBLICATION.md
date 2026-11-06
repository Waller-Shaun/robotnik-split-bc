# GitHub assignment publication

This `main` tree contains the files prepared for `COMP0245_Assignment_Submission_20261106.zip`, with an English default README, a Chinese README, Git metadata rules and a manifest updated for the actual repository contents. The ZIP itself is not duplicated in Git.

The previous `main` is preserved unchanged as `baseline`, pointing to `5073b83768a4c4931dbe146558f059d8ab2d70b7`. The research and standalone patrol branches remain available. The new main commit descends from the previous main; no existing main history is rewritten.

Original experiment sources, models, normalization, split files and recorded results are unchanged. Root documentation/metadata differs from the ZIP; `SOURCE_INVENTORY.json` continues to verify the original included source files, while `SUBMISSION_MANIFEST.json` verifies this GitHub tree. Old `VALIDATION.json` records refer to the ZIP preparation, not a claim of running new experiments during publication.

The 61 optional historical research checkpoints and102 raw recovery NPZ arrays remain omitted as listed in `OMITTED_OPTIONAL_ARTIFACTS.json`; their complete earlier archive is on the research branch. Main runnable models are included. Full raw datasets, environments and evaluation rollouts are not bundled.

Assignment ZIP SHA256: `2c5b4c0099692769e4fbec2d3030cfe51f16d96e5c0f7bcd9f6e1ca9d88dd087`.

Read-only checks:

```bash
python -X utf8 submission.py verify
python -X utf8 submission.py results
```

See `README.md` / `README.zh-CN.md` and `QUICKSTART.md` for explicit installation and run commands. No training, simulation, collection or dependency installation was started while preparing this publication.
