# Paper Experiment Harness

This repository is a lean, auditable harness for reproducible paper experiments. It is intentionally small: configs define runs, commands create saved artifacts, and plots are regenerated from saved results.

## Current Status

- The current target is Pythia-14M architecture pretraining on MiniPile.
- Model weights are randomly initialized. We do not load Pythia checkpoint weights for baseline pretraining runs.
- Earlier local runs that loaded pretrained Pythia weights were removed from the experiment log, result folders, and paper map.
- Data preparation tokenizes MiniPile into a local cache before calibration or paper runs.

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
make smoke     # run a tiny harness sanity check
make prepare-minipile      # tokenize the configured MiniPile subset locally
make calibrate-pythia-14m  # random-init Pythia-14M pretraining calibration
make pretrain-pythia-14m-full-10min  # 10-minute full-MiniPile pretraining checkpoint
make baseline  # blocked until the pretraining budget is chosen
make plots     # regenerate figures from saved results
```

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

Each run writes:

```text
config.yaml
metrics.json
manifest.json
predictions.jsonl
events.jsonl
checkpoints/final/
```

`manifest.json` records the model architecture, random initialization, config path, command, seed, dataset, Python version, package versions, and git commit when available.

First valid full-MiniPile random-init checkpoint:

```text
config: configs/03-pythia-14m-minipile-random-full-10min.yaml
result: results/03-pythia-14m-minipile-random-full-10min/003-20260627-142522-7fc1e76f/
figure: figures/01-pythia-14m-minipile-random-full-10min-diagnostics.pdf
tokens seen: 86,245,376
final train loss: 7.6701
final validation loss: 7.5450
```

## Known TODOs

- `TODO:` choose the MiniPile pretraining budget.
- `TODO:` add ablation configs when the method is defined.
- `TODO:` consider scaling within the Pythia family up to 160M only after the 14M random-init path is reproducible.
