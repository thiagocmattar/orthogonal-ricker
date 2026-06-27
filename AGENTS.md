# Agent Instructions

This is a lean research repository for a paper.

Principle: simplicity over complexity. Do not add abstractions, frameworks, or files unless they directly help run, reproduce, compare, or explain experiments.

Read first:

1. `README.md`
2. `docs/methods.md`
3. `docs/experiment_log.md`
4. `docs/paper_map.md`

Core commands:

- `make test`
- `make smoke`
- `make prepare-minipile`
- `make calibrate-pythia-14m`
- `make baseline`
- `make plots`

Rules:

- Do not invent scientific claims, datasets, models, metrics, or results.
- Use `TODO:` placeholders when information is missing.
- Every experiment should have a config.
- Configs, result folders, run folders, and figures should use sequential prefixes such as `01-baseline.yaml`, `001-<timestamp>-<id>`, and `01-results-summary.pdf`.
- Every run should save its config, metrics, predictions, and manifest.
- Every paper figure should be regenerable from saved results.
- Treat figures as paper artifacts: avoid misleading axis truncation, use colorblind-safe colors, show sample size or uncertainty when relevant, and keep plotting style centralized in `src/paper_exp/plots.py`.
- Prefer one clear script over many clever abstractions.
- If adding complexity, explain why it is necessary.
