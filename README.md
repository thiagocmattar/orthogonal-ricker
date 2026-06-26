# Paper Experiment Harness

This repository is a lean, auditable harness for reproducible paper experiments. It is intentionally small: configs define runs, commands create saved artifacts, and plots are regenerated from saved results.

## Current Status

- Smoke runs work locally and write reproducible result artifacts.
- Plot generation works from saved smoke results.
- The real Hugging Face baseline is a `TODO:` until the model, dataset, task, and metric are provided.

## Install

```bash
make install
```

## Commands

```bash
make test      # run tests
make smoke     # run a tiny local sanity check
make baseline  # run configs/01-baseline.yaml once TODOs are filled in
make plots     # regenerate figures from saved results
```

## Numbering Convention

Use sequential prefixes for files and directories that accumulate over time:

```text
configs/01-baseline.yaml
configs/02-ablation-name.yaml
results/01-baseline/001-<timestamp>-<short_id>/
figures/01-results-summary.pdf
```

The number is for human ordering. Run directories still include a timestamp and short id for uniqueness.

## Smoke Run

```bash
make smoke
```

This creates a timestamped result directory under:

```text
results/<config_id>/<run_id>/
```

Example:

```text
results/01-baseline/001-20260626-173041-33e1187b/
```

Each run writes `config.yaml`, `metrics.json`, `manifest.json`, and `predictions.jsonl`. The manifest records `config_id`, `run_id`, and `run_sequence`.

## Baseline Run

```bash
make baseline
```

The baseline config is `configs/01-baseline.yaml`. It currently contains `TODO:` placeholders and the runner fails fast instead of inventing a model, dataset, task, metric, or result.

## Regenerate Plots

```bash
make plots
```

Plots are generated from saved result directories and written to `figures/`. The default summary figure is `figures/01-results-summary.pdf`, with a PNG copy when run through `make plots`.

See `figures/README.md` for paper-quality plotting standards. Figures should be honest, reproducible from saved results, and ready for direct paper inclusion.

## Add A New Experiment

1. Copy `configs/01-baseline.yaml` to a new numbered config file, such as `configs/02-ablation-name.yaml`.
2. Set `experiment_name`, model, data, evaluation, seed, and output fields.
3. Run the relevant command.
4. Add the result path to `docs/experiment_log.md`.
5. Map any paper claim or figure in `docs/paper_map.md`.

## Known TODOs

- `TODO:` choose the Hugging Face model.
- `TODO:` choose the dataset and split.
- `TODO:` define the evaluation metric and task.
- `TODO:` implement the real baseline once the above are known.
- `TODO:` add ablation and scale-ladder configs when the method is defined.
