# Agent Instructions

This is a lean research repository for a paper.

Principle: simplicity over complexity. Do not add abstractions, frameworks, or files unless they directly help run, reproduce, compare, or explain experiments.

Read first:

1. `README.md`
2. `docs/code_map.md`
3. `docs/methods.md`
4. `docs/experiment_log.md`
5. `docs/paper_map.md`

Core commands:

- `make test`
- `make check`
- `make smoke`
- `make prepare-minipile`
- `make calibrate-pythia-14m`
- `make baseline`
- `make plots`

Rules:

- Do not invent scientific claims, datasets, models, metrics, or results.
- Use `TODO:` placeholders when information is missing.
- Keep all repository files, documentation, result records, plot labels, and generated outputs in English.
- Pythia experiments are pretraining runs: use the Pythia architecture with `model.initialization: random`; do not load released checkpoint weights unless the user explicitly requests a continuation/fine-tuning run.
- Activation pressure methods currently target `mlp_hiddens` first. Keep naive pressure and Adam-step orthogonal pressure separate in configs, metrics, and docs.
- Every experiment should have a config.
- Configs, result folders, run folders, and figures should use sequential prefixes such as `01-baseline.yaml`, `001-<timestamp>-<id>`, and `01-results-summary.pdf`.
- Smoke and calibration/pretrain runs must save `config.yaml` and a `status: running` manifest at launch, preserve that launch Git provenance, and publish the terminal `status: completed` manifest only after metrics and predictions are durable; escaping exceptions must leave `status: failed`. Data preparation, diagnostic, and clipping workflows still use the legacy end-of-run writer until their separate migration.
- Completed runs should save config, metrics, predictions, and manifest. Statusless historical runs remain valid when their core artifact envelope is coherent.
- Every paper figure should be regenerable from saved results.
- Treat figures as paper artifacts: avoid misleading axis truncation, use colorblind-safe colors, show sample size or uncertainty when relevant, and keep shared plotting style centralized in `src/paper_exp/plot_style.py`; family-specific loaders, reductions, and renderers belong in focused modules such as `src/paper_exp/plot_report04.py`.
- Pytest scratch directories such as `pytest_tmp*` may be created during testing, but must always be cleaned from the worktree after testing finishes; delete them or archive them outside the repository when retention is useful.
- Prefer one clear script over many clever abstractions.
- If adding complexity, explain why it is necessary.
