# Configs

`00-smoke.yaml` is a minimal infrastructure check. It is not a scientific
experiment, calibration recipe, baseline, or template for paper settings.

Definitive configs begin at `01` only after the user supplies and reviews
[`docs/experiment_plan.md`](../docs/experiment_plan.md). Until then, do not add
or launch scientific configs.

## Config Lifecycle

For every plan-authorized experiment:

1. Copy the closest config of the same workflow kind after such a config exists.
2. Change only fields required by the plan cell.
3. Use the next unique sequential prefix and a lowercase hyphenated name.
4. Remove unresolved `TODO:` values.
5. Validate and review the full config.
6. Add and commit it before launch.
7. Treat it as immutable after the first attempt begins.

A scientific change, different seed, changed budget, or changed diagnostic
source set gets a new config. An infrastructure-only retry creates a new run
attempt under the unchanged config.

Examples of the naming contract:

```text
configs/01-baseline.yaml
configs/02-method-name.yaml
configs/03-method-name-seed-1.yaml
```

The config stem becomes the result-group ID:

```text
configs/01-baseline.yaml
results/01-baseline/001-<timestamp>-<short-id>/
```

## Required Shared Fields

Every config contains:

- `experiment_name`;
- `model.provider`;
- `model.name`;
- `model.architecture`;
- `model.initialization`;
- `data.name`;
- `data.split`;
- `evaluation.metric`;
- `run.seed`;
- `output.dir`.

Only the infrastructure smoke config uses `run.max_examples`.

Pretraining configs use:

```yaml
model:
  provider: huggingface
  name: TODO
  architecture: TODO
  revision: TODO  # exact immutable 40- to 64-character commit
  initialization: random
```

`architecture` names the architecture/config source. `initialization: random`
means the harness constructs a new model and does not load released checkpoint
weights.

Training/data preflight also requires exact immutable `data.revision` and
`tokenizer.revision` commits; explicit text column, document limits, cache ID,
block size, EOS policy, and overwrite policy; explicit validation scope; all
optimizer/batch/precision/budget fields; checkpoint flags; independent model
and data-order seeds; and the complete activation-pressure section. Use `null`
only where the schema makes absence explicit, such as no wall-time cap, no
validation partition, or no learned-threshold learning-rate multiplier.
Release configs use the portable relative paths `output.dir: results` and
`preprocessing.output_dir: data/tokenized`; launch preflight rejects external
or machine-specific artifact roots.

Use `TODO:` only while drafting an unlaunched config. The runner rejects TODOs.

## Field Ownership

| Section | Primary owner | Meaning |
| --- | --- | --- |
| `experiment_name`, `model`, `data`, `evaluation`, `run`, `output` | `config.py` and the selected workflow | Identity and common envelope |
| `tokenizer`, `preprocessing` | `data.py` | Tokenizer, block construction, and cache identity |
| `training`, `validation`, `checkpoint` | `calibration.py` | Optimizer loop, evaluation, and saved state |
| `model.post_layernorm_relu`, `model.post_layernorm_gate`, `model.mlp_hidden_gate`, `model.post_qkv_relu` | `modeling.py`, `calibration.py` | Explicit architecture interventions |
| `activation_pressure` | `activation_pressure.py`, `activations.py`, `calibration.py` | Method, sites, weight, geometry, and monitoring thresholds |
| `activation_histograms` | `activation_histograms.py` | Sites, bins, thresholds, and pinned sources |
| `weight_histograms` | `weight_histograms.py` | Parameter scope, bins, and pinned sources |
| `activation_propagation` | `activation_propagation.py` | Pinned sources and exact-zero/product measurement |
| `activation_clipping` | `clipping.py`, `activations.py` | Clipping mode and sites; normally derived from a saved source run plus CLI arguments |

The CLI command selects the workflow. `evaluation.metric` documents intent but
does not dispatch the config.

## Reproducibility Fields

The definitive plan must specify independent model-initialization and
data-order seeds when coupled reproducibility is required. Training schedule and
validation partition hashes belong in manifests or specialized artifacts.

Reusing a token cache is valid only when dataset, split, revision, tokenizer,
block size, EOS policy, document limits, and partition contract match. Record
the cache metadata/hash; a shared path is not proof of equivalence.

## Method Separation

Use distinct method identifiers:

- `none` for a monitor-only optimizer control;
- `ricker_naive` and `l1_naive` for direct auxiliary-loss pressure;
- `orthogonal_ricker` and `orthogonal_l1` for task-only AdamW followed by the
  projected pressure correction.

Orthogonal methods require an explicit `step_budget`. Sites and all numerical
pressure parameters are explicit; the harness chooses no scientific default. See
[`docs/methods.md`](../docs/methods.md) before adding method fields.

## Diagnostic Sources

A diagnostic config names exact source config IDs and run IDs. Do not use
wildcards or "latest completed" selection. The diagnostic must record source
manifest status, checkpoint identity, validation coverage, and specialized
artifact schema described in [`docs/diagnostics.md`](../docs/diagnostics.md).

Post-hoc clipping is derived from one exact checkpoint run. Its CLI arguments
must still be captured in the derived run config/manifest and artifact.

## Multiple Pretraining Configs

When more than one pretraining config is ready, pass the complete ordered list
to one sequential runner:

```bash
paper-exp run-configs \
  --config configs/01-baseline.yaml \
  --config configs/02-method-name.yaml
```

Never start separate pretraining runners or parallel GPU jobs. Report first-run
and full-queue ETCs before launch and use `paper-exp run-status` for read-only
progress updates. Data preparation, calibration, and diagnostics remain
single-config workflows until an explicit sequential contract is implemented.
