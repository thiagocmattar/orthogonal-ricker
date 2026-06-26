# Configs

Each experiment should have a numbered YAML config. Start from `01-baseline.yaml`, copy it, and change only the fields needed for the experiment.

Use sequential names:

```text
01-baseline.yaml
02-ablation-name.yaml
03-scale-ladder-small.yaml
```

The run harness uses the config stem as the result experiment id, for example `configs/01-baseline.yaml` writes under `results/01-baseline/`.

Required fields:

- `experiment_name`
- `model.provider`
- `model.name`
- `data.name`
- `data.split`
- `evaluation.metric`
- `run.seed`
- `run.max_examples`
- `output.dir`

Use `TODO:` placeholders until the paper-specific model, dataset, metric, or method is known.
