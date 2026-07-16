# Paper Experiment Harness

This repository is a lean, auditable harness for reproducible paper experiments. It is intentionally small: configs define runs, commands create saved artifacts, and plots are regenerated from saved results.

Author: Thiago Mattar <thiagocmattar@gmail.com>

## Current Status

- The current target is Pythia-14M architecture pretraining on MiniPile.
- Model weights are randomly initialized. We do not load Pythia checkpoint weights for baseline pretraining runs.
- Earlier local runs that loaded pretrained Pythia weights were removed from the experiment log, result folders, and paper map.
- Data preparation tokenizes MiniPile into a local cache before calibration or paper runs.
- Repository files, documentation, result records, plot labels, and generated outputs are kept in English.

## Code Navigation

See [`docs/code_map.md`](docs/code_map.md) for module ownership, CLI-to-artifact
routes, config ownership, and surgical recipes for adding methods, activation
sites, diagnostics, and paper figures. See [`configs/README.md`](configs/README.md)
before creating or promoting an experiment config. See
[`docs/plotting.md`](docs/plotting.md) for the Report 04 visual contract,
plotting-module boundaries, and the parity checks required for figure changes.

## Install

```bash
make install
```

For GPU calibration on this Windows/CUDA machine, verify PyTorch sees CUDA:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Commands

```bash
make test      # run lightweight unit tests
make check     # read-only config, run-artifact, and paper-reference checks
make smoke     # run a tiny harness sanity check
make prepare-minipile      # tokenize the configured MiniPile subset locally
make calibrate-pythia-14m  # random-init Pythia-14M pretraining calibration
make pretrain-pythia-14m-full-10min  # 10-minute full-MiniPile pretraining checkpoint
make pressure-smoke-ricker-naive
make pressure-smoke-l1-naive
make pressure-smoke-orthogonal-ricker
make pressure-smoke-orthogonal-l1
make pressure-short-all  # four short full-MiniPile pressure checks
make baseline  # blocked until the pretraining budget is chosen
make plots     # regenerate figures from saved results
make plot-report04  # strict Report 04 suite + deterministic input provenance
```

`make check` reports warnings for incomplete run directories and reused figure
prefixes but exits nonzero only for errors. Use
`python -m paper_exp.cli check --strict` as a release gate when warnings must
also fail.

## Human-facing Notes

Documents written for human reading live under `docs/humans/`.

Current notes:

- `docs/humans/01-pythia-14m-architecture.md`: Pythia-14M architecture, block equations, pseudocode, and activation-pressure hook sites.
- `docs/humans/02-fixed-step-pressure-screen.md`: fixed-step Pythia-14M MiniPile pressure screen, metrics table, figures, and interpretation boundary.
- `docs/humans/03-research-agenda-to-publication.md`: internal research agenda that separates observations, hypotheses, candidate claims, and publication-stage evidence requirements.
- `docs/humans/04-attention-sparsification-paths.md`: selected post-QKV Q/K/V ReLU placement experiment, implementation contract, launch matrix, diagnostics, and deferred probability-threshold hypothesis.

## Model Initialization

For this project, "Pythia-14M" means the Pythia-14M architecture, not the released checkpoint weights.

Configs must make this explicit:

```yaml
model:
  provider: huggingface
  name: pythia-14m-random
  architecture: EleutherAI/pythia-14m-deduped
  revision: main
  initialization: random
```

The harness builds the model with `AutoConfig.from_pretrained(model.architecture)` followed by `AutoModelForCausalLM.from_config(...)`. This loads the architecture/config only and initializes weights randomly.

## Activation Pressure

The harness supports the first activation-pressure methods on Pythia MLP hidden activations:

- `ricker_naive`: adds Ricker pressure directly to task loss.
- `l1_naive`: adds activation L1 pressure directly to task loss.
- `orthogonal_ricker`: AdamW moments see task gradients only, then a projected Ricker correction is applied after the AdamW step.
- `orthogonal_l1`: same Adam-step orthogonal correction using activation L1 pressure.

The initial site target is `mlp_hiddens`, implemented with hooks on each Pythia layer's MLP activation module. The harness logs pressure loss, task/pressure gradient interference metrics, Adam-step projection metrics for orthogonal methods, and near-zero activation mass.

Post-hoc clipping frontiers can be run with:

```bash
python -m paper_exp.cli clip-sweep --run-dir <checkpoint-run-dir> --thresholds 0,0.001,0.01,0.03 --eval-batches 2
```

and plotted with:

```bash
python -m paper_exp.cli plot-clipping-frontier --run-dir <clipping-sweep-run-dir> --output figures/02-pythia-14m-minipile-clipping-frontier-smoke.pdf --png
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

## Results

Completed run directories normally contain:

```text
config.yaml
metrics.json
manifest.json
predictions.jsonl
events.jsonl          # calibration/pretrain only
checkpoints/final/    # when enabled
```

Smoke and calibration/pretrain runs use the first explicit run lifecycle. At
launch they save an immutable `config.yaml` snapshot and a manifest with
`status: running`. The manifest records the model architecture, random
initialization, config path, command, seed, dataset, Python version, package
versions, and Git commit/dirty state as observed at launch. Completion writes
metrics and predictions before atomically publishing `status: completed` and
`finished_at` in the manifest. An escaping exception is re-raised after the
manifest is updated to `status: failed`, `finished_at`, and its exception type
and message; metrics and predictions may therefore be absent from failed runs.

Statusless historical runs remain supported as completed runs when their core
config, metrics, manifest, and predictions envelope is coherent. Data
preparation, activation/weight histogram diagnostics, activation propagation,
and clipping sweeps still use the legacy end-of-run artifact writer; they will
receive the same lifecycle in a separate migration.

For calibration/pretrain runs, `predictions.jsonl` currently stores the same
training and validation event history represented by `events.jsonl`; it does
not contain generated-token predictions. The metric
`calibration/wall_seconds_train` measures the timed training interval including
validation performed inside that interval. Validation duration is also
reported separately, while `calibration/wall_seconds_total` additionally
includes final checkpoint work but not the final artifact writes.

First valid full-MiniPile random-init checkpoint:

```text
config: configs/03-pythia-14m-minipile-random-full-10min.yaml
result: results/03-pythia-14m-minipile-random-full-10min/003-20260627-142522-7fc1e76f/
figure: figures/01-pythia-14m-minipile-random-full-10min-diagnostics.pdf
tokens seen: 86,245,376
final train loss: 7.6701
final validation loss: 7.5450
```

First pressure smoke checks completed for all four method variants on the 128-document MiniPile smoke subset. These are plumbing checks only.

First post-hoc clipping smoke frontier:

```text
result: results/03-pythia-14m-minipile-random-full-10min-clipping-sweep/002-20260627-150326-6a61b34d/
figure: figures/02-pythia-14m-minipile-clipping-frontier-smoke.pdf
```

Fixed-step activation-pressure screen:

```text
configs: configs/12-pythia-14m-minipile-adamw-fixed-2048.yaml through configs/48-pythia-14m-minipile-orthogonal-l1-fixed-2048-w5.yaml
tokens/run: 134,217,728
summary: docs/humans/02-fixed-step-pressure-screen.md
figures: figures/05-pythia-14m-pressure-fixed-2048-summary.pdf through figures/20-pythia-14m-pressure-fixed-2048-high-pressure-clipping-frontiers.pdf
```

Post-LayerNorm ReLU pressure comparison:

```text
training configs: configs/98-pythia-14m-minipile-post-layernorm-relu-adamw-full-pass.yaml, configs/99-pythia-14m-minipile-post-layernorm-relu-orthogonal-l1-full-pass-w5.yaml, configs/103-pythia-14m-minipile-post-layernorm-relu-orthogonal-ricker-full-pass-w1-c0p05-s0p05.yaml, and configs/104-pythia-14m-minipile-post-layernorm-relu-l1-naive-full-pass-w5.yaml
validation diagnostics: configs/100-pythia-14m-minipile-post-layernorm-relu-input-histograms.yaml through configs/102-pythia-14m-minipile-post-layernorm-relu-activation-propagation.yaml, plus full-cache site-specific and joint clipping sweeps
report: report/04-2026-07-11-post-layernorm-relu-ol1-comparison/04-2026-07-11-post-layernorm-relu-ol1-comparison.pdf
figures: figures/79-pythia-14m-minipile-post-layernorm-relu-learning-diagnostics.pdf through figures/90-pythia-family-three-relu-model-compute-ceilings.pdf
```

This one-seed test applies ReLU after both branch LayerNorms in every transformer block, keeps the MLP-hidden ReLU, and leaves the final LayerNorm unchanged. Pressure is applied only to `attention_inputs`, `mlp_inputs`, and `mlp_hiddens`. The tested settings are OR weight 1 with `c = sigma = 0.05` and step budget 0.5, L1N weight 5, and OL1 weight 5 with step budget 0.5.

Config `102` compares AdamW, OR, L1N, and OL1 by counting exact zeros at every attention, MLP, and residual boundary and the corresponding logical zero-input products in QKV, QK, PV, attention output, W1, and W2 matmuls. It covers all 692,224 evaluated validation tokens and excludes future causal-mask entries from attention-core denominators.

## Known TODOs

- `TODO:` choose the longer MiniPile pretraining budget for the full ablation.
- `TODO:` repeat key candidates over multiple seeds.
- `TODO:` consider scaling within the Pythia family up to 160M only after the 14M random-init path is reproducible.
