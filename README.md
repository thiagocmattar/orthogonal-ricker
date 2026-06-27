# Paper Experiment Harness

This repository is a lean, auditable harness for reproducible paper experiments. It is intentionally small: configs define runs, commands create saved artifacts, and plots are regenerated from saved results.

## Current Status

- Smoke runs work locally and write reproducible result artifacts.
- The first concrete path is Pythia-14M on MiniPile.
- Data preparation tokenizes MiniPile into a local cache before calibration or paper runs.
- Plot generation works from saved results.

## Install

```bash
make install
```

For GPU calibration on this Windows/CUDA 12.8 machine, verify PyTorch sees CUDA:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it reports CPU-only PyTorch, install the CUDA 12.8 wheel:

```bash
python -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

## Commands

```bash
make test      # run tests
make smoke     # run a tiny local sanity check
make prepare-minipile      # tokenize the MiniPile smoke subset locally
make calibrate-pythia-14m  # short throughput calibration on Pythia-14M
make baseline  # placeholder until the calibrated 14M baseline budget is chosen
make plots     # regenerate figures from saved results
```

## Numbering Convention

Use sequential prefixes for files and directories that accumulate over time:

```text
configs/01-pythia-14m-minipile-smoke.yaml
configs/02-pythia-14m-minipile-baseline.yaml
results/01-pythia-14m-minipile-smoke/001-<timestamp>-<short_id>/
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
results/01-pythia-14m-minipile-smoke/001-20260626-173041-33e1187b/
```

Each run writes `config.yaml`, `metrics.json`, `manifest.json`, and `predictions.jsonl`. The manifest records `config_id`, `run_id`, and `run_sequence`.

## MiniPile Preparation

```bash
make prepare-minipile
```

This uses `configs/01-pythia-14m-minipile-smoke.yaml`, downloads MiniPile through Hugging Face datasets, tokenizes the configured subset with the Pythia tokenizer, and writes a local int32 token cache under `data/tokenized/`.

Paper runs should use local cached/tokenized data rather than streaming so token counts, ordering, and runtime are auditable.

## Pythia-14M Calibration

```bash
make calibrate-pythia-14m
```

This runs a short training calibration from the local token cache and records `tokens_per_second`, wall-clock seconds, tokens seen, loss, and peak GPU memory. Use these measurements to update wall-clock estimates before choosing the full baseline budget.

The calibration result directory includes:

```text
config.yaml
metrics.json
manifest.json
predictions.jsonl
events.jsonl
checkpoints/final/
```

`events.jsonl` and `predictions.jsonl` contain the same step-level event stream for now:

- `train` events: step, estimated epoch, tokens seen, train loss, learning rate, gradient norm, weight norm, step wall time.
- `validation` events: step, estimated epoch, tokens seen, validation loss, validation batches, validation tokens.

`metrics.json` stores the summary values: final/mean train loss, final/best validation loss, tokens/sec, train and total wall-clock time, peak GPU memory, final norms, and final checkpoint size.

Latest measured calibration:

```text
result: results/01-pythia-14m-minipile-smoke/008-20260627-110058-1442200f/
tokens/sec: 117,421
peak GPU memory: 5,785.7 MB
final train loss: 6.0995
final validation loss: 4.9626
final checkpoint: 26.84 MB
estimated Pythia-14M MiniPile pass: ~3.5 hours raw training-loop time
```

## Baseline Run

```bash
make baseline
```

The baseline config is `configs/02-pythia-14m-minipile-baseline.yaml`. The baseline runner is intentionally blocked until calibration establishes a practical run budget.

## Regenerate Plots

```bash
make plots
```

Plots are generated from saved result directories and written to `figures/`. The default summary figure is `figures/01-results-summary.pdf`, with a PNG copy when run through `make plots`.

See `figures/README.md` for paper-quality plotting standards. Figures should be honest, reproducible from saved results, and ready for direct paper inclusion.

## Add A New Experiment

1. Copy an existing numbered config to a new numbered config file, such as `configs/03-ablation-name.yaml`.
2. Set `experiment_name`, model, data, evaluation, seed, and output fields.
3. Run the relevant command.
4. Add the result path to `docs/experiment_log.md`.
5. Map any paper claim or figure in `docs/paper_map.md`.

## Known TODOs

- `TODO:` run MiniPile tokenization and 14M calibration.
- `TODO:` use measured throughput to choose the baseline training budget.
- `TODO:` implement the full baseline run after calibration.
- `TODO:` add ablation configs when the method is defined.
- `TODO:` consider scaling within Pythia up to 160M only after the 14M path is reproducible.
