# Configs

Each experiment should have a numbered YAML config. Start from the closest existing config, copy it, and change only the fields needed for the experiment.

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
