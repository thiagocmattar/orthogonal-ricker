# Configs

Each experiment should have a numbered YAML config. Start from the closest existing config, copy it, and change only the fields needed for the experiment.

During exploration, treat new configs as local/ignored working artifacts unless
they are deliberately promoted. A completed exploratory run does not by itself
make its config part of the public reproducibility surface. Promote a config
only when it is selected for a method, result, ablation, diagnostic, or paper
figure: review it, remove unintended `TODO:` values, preserve its sequential
name, add it explicitly with `git add -f configs/<selected-config>.yaml`, and
map it in `docs/experiment_log.md` and `docs/paper_map.md` where relevant. The
final open-source repository should
contain the selected configs needed to reproduce its documented evidence, not
every local search point.

For pretraining runs, keep model architecture and weight initialization explicit:

```yaml
model:
  provider: huggingface
  name: pythia-14m-random
  architecture: EleutherAI/pythia-14m-deduped
  revision: main
  initialization: random
```

`architecture` is the Hugging Face config source. `initialization: random` means the harness does not load released checkpoint weights.

Use sequential names:

```text
01-pythia-14m-minipile-smoke.yaml
02-pythia-14m-minipile-baseline.yaml
03-ablation-name.yaml
```

The run harness uses the config stem as the result experiment id, for example `configs/01-pythia-14m-minipile-smoke.yaml` writes under `results/01-pythia-14m-minipile-smoke/`.

## Config Kinds and Starting Points

The CLI command chooses the workflow. `evaluation.metric` documents the
measurement but does not dispatch the config automatically. Copy a config of
the same kind and architecture; do not assemble one from unrelated examples.

| Config kind | Canonical starting example | Main distinguishing section |
| --- | --- | --- |
| Harness smoke, data preparation, or short calibration | `01-pythia-14m-minipile-smoke.yaml` | `data`, `tokenizer`, `preprocessing`, `training`, `validation` |
| Random-initialized pretraining | `03-pythia-14m-minipile-random-full-10min.yaml` | `training`, `validation`, `checkpoint` |
| Naive pressure pretraining | `104-pythia-14m-minipile-post-layernorm-relu-l1-naive-full-pass-w5.yaml` | `activation_pressure.method: l1_naive` |
| Adam-step orthogonal pressure pretraining | `103-pythia-14m-minipile-post-layernorm-relu-orthogonal-ricker-full-pass-w1-c0p05-s0p05.yaml` | `activation_pressure.method: orthogonal_ricker` plus `step_budget` |
| Activation histogram diagnostic | `100-pythia-14m-minipile-post-layernorm-relu-input-histograms.yaml` | `activation_histograms` |
| Weight histogram diagnostic | `61-pythia-14m-minipile-full-pass-high-pressure-weight-histograms.yaml` | `weight_histograms` |
| Exact-zero/zero-product propagation diagnostic | `102-pythia-14m-minipile-post-layernorm-relu-activation-propagation.yaml` | `activation_propagation` |
| Fixed-step pressure sweep | `12-pythia-14m-minipile-adamw-fixed-2048.yaml` and adjacent generated configs | `sweep`; matrix source is `src/paper_exp/sweeps.py` |
| Post-hoc clipping sweep | Source training run's saved `config.yaml`; settings are normally CLI arguments | Optional `activation_clipping` plus `--thresholds`, `--quantiles`, `--rms-multipliers`, and `--sites` |

The Report 04 training and diagnostic configs (`98` through `104`) are useful
references for that specific Three-ReLU architecture and comparison. Do not
copy its architecture fields into a stock-Pythia run accidentally.

## Field Ownership

| Fields or section | Primary owner | Meaning |
| --- | --- | --- |
| `experiment_name`, `model`, `data`, `evaluation`, `run`, `output` | `config.py` validates the shared minimum; each workflow consumes its subset | Common experiment identity and run envelope |
| `model.architecture`, `model.revision`, `model.initialization`, `model.hidden_act`, `model.post_layernorm_relu` | `calibration.py`, `modeling.py` | Architecture construction and explicit scientific modifications |
| `data`, `tokenizer`, `preprocessing` | `data.py` | Dataset/tokenizer identity, token cache shape, and cache reuse |
| `training`, `validation`, `checkpoint` | `calibration.py` | Optimizer loop, evaluation schedule, and final checkpoint policy |
| `activation_pressure` | `activation_pressure.py`, `activations.py`, `calibration.py` | Method, sites, weight, Ricker parameters, step budget, and logged thresholds |
| `sweep` | `sweeps.py`; copied to the training manifest by `calibration.py` | Human grouping, role, and planned budget for a sweep member |
| `activation_clipping` | `clipping.py`, `activations.py` | Post-hoc clipping mode and sites; CLI values may construct this section |
| `activation_histograms` | `activation_histograms.py` | Sites, histogram range/bins, thresholds, and selected source runs |
| `weight_histograms` | `weight_histograms.py` | Parameter scope, histogram range/bins, and selected source runs |
| `activation_propagation` | `activation_propagation.py` | Selected source runs for exact-zero and zero-product measurement |

See `docs/code_map.md` for the corresponding CLI handlers and artifacts.

Large tokenized datasets can be reused across configs with:

```yaml
preprocessing:
  output_dir: data/tokenized
  cache_id: 03-pythia-14m-minipile-random-full-10min
```

Use `cache_id` only when the dataset, split, tokenizer, block size, EOS handling, and max document limits match the referenced cache. The run still writes results under the current config id.

Required fields:

- `experiment_name`
- `model.provider`
- `model.name`
- `model.architecture`
- `model.initialization`
- `data.name`
- `data.split`
- `evaluation.metric`
- `run.seed`
- `run.max_examples`
- `output.dir`

Use `TODO:` placeholders until the paper-specific model, dataset, metric, or method is known.
