# Configs

`00-smoke.yaml` is a minimal infrastructure check. It is not a scientific
experiment, calibration recipe, baseline, or template for paper settings.

[`exp-plan-v0.md`](../exp-plan-v0.md) is a structural preview, not launch
authorization. Definitive configs begin only after the final plan replaces and
reviews [`docs/experiment_plan.md`](../docs/experiment_plan.md).
While that document says `Plan status: placeholder`, do not create a numbered
scientific config or case runner. The schema examples below document runnable
field combinations only; they do not authorize or recommend an experiment.

Configs are grouped by launch tranche. The folder name exactly matches its
case runner:

```text
runners/NN-<phase>-<tranche>.py
configs/NN-<phase>-<tranche>/
```

## Config Lifecycle

For every plan-authorized experiment:

1. Copy the closest config of the same workflow kind after such a config exists.
2. Change only fields required by the plan cell.
3. Put it in the matching runner folder.
4. Use the next globally unique config prefix and a lowercase hyphenated name.
5. Remove unresolved `TODO:` values.
6. Validate and review the full config.
7. Add and commit it before launch.
8. Treat it as immutable after the first attempt begins.

A scientific change, different seed, changed budget, or changed diagnostic
source set gets a new config. An infrastructure-only retry creates a new run
attempt under the unchanged config.

Examples of the naming contract:

```text
configs/NN-<phase>-<tranche>/CCC-<case>.yaml
```

`NN` is a two-digit launch prefix. `CCC` is a three-digit config prefix that is
globally unique and sequential across launch folders. Phase IDs such as `a1`
and `b1` preserve the plan mapping without coupling the parent runner to a
particular plan version.

The config stem becomes the result-group ID:

```text
configs/NN-<phase>-<tranche>/CCC-<case>.yaml
results/CCC-<case>/001-<timestamp>-<short-id>/
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
  topology_id: A0
  site_gate: null
```

`architecture` names the architecture/config source. `initialization: random`
means the harness constructs a new model and does not load released checkpoint
weights. `A0` has no active site gates, requires `site_gate: null`, and keeps
the stock GELU at `h`. This is a schema illustration, not a scientific default.

`model.topology_id` selects only the active gate ports. The exact supported IDs
are `A0`, `A1-H`, `A2`, `A3`, `A4-Q`, `A4-K`, `A4-V`, `A5-QK-PRE`,
`A5-QK-POST`, `A6-PRE`, and `A6-POST`. Their port sets are authoritative in
[`docs/methods.md`](../docs/methods.md). In particular, `A2` means `m` + `h`,
while `A4-Q` and `A4-K` mean the POST-RoPE `q_post` and `k_post` ports.

Every non-`A0` topology requires an explicit `model.site_gate`. For example,
the ReLU form of `A2` is represented by:

```yaml
model:
  topology_id: A2
  site_gate:
    operator: relu
```

The other supported operators are `one_sided_threshold` and
`symmetric_threshold`; each requires an explicit finite nonnegative `kappa`.
The topology ID never implies the operator or `kappa`. Optimizer and
`activation_pressure` settings remain separate and must be supplied by the
reviewed plan.

Training/data preflight also requires exact immutable `data.revision` and
`tokenizer.revision` commits; explicit text column, document limits, cache ID,
block size, EOS policy, and overwrite policy; explicit validation scope; all
optimizer/batch/precision/budget fields; checkpoint flags; independent model
and data-order seeds; and the complete activation-pressure section. Use `null`
only where the schema makes absence explicit, such as no wall-time cap or no
validation partition.
Release configs use the portable relative paths `output.dir: results` and
`preprocessing.output_dir: data/tokenized`; launch preflight rejects external
or machine-specific artifact roots.

Use `TODO:` only while drafting an unlaunched config. The runner rejects TODOs.

## Field Ownership

[`docs/code_map.md`](../docs/code_map.md) is the single module-ownership map.
Config validation belongs to `config.py`; each workflow owns the meaning and
execution of its specific section. The CLI command selects the workflow.
`evaluation.metric` documents intent but does not dispatch a config.

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
- `l1_naive` for direct L1 auxiliary-loss pressure;
- `orthogonal_l1` for task-only AdamW followed by the projected L1 pressure
  correction.

Orthogonal methods require an explicit `step_budget`. Sites and all numerical
pressure parameters are explicit; the harness chooses no scientific default.
`activation_pressure.sites` uses only `a`, `m`, `h`, `q_pre`, `k_pre`,
`q_post`, `k_post`, and `v`, and is not inferred from `model.topology_id`.
See [`docs/methods.md`](../docs/methods.md) before adding method fields.

## Diagnostic Sources

A diagnostic config names exact source config IDs and run IDs. Do not use
wildcards or "latest completed" selection. The diagnostic must record source
manifest status, checkpoint identity, validation coverage, and specialized
artifact schema described in [`docs/diagnostics.md`](../docs/diagnostics.md).

Post-hoc clipping is derived from one exact checkpoint run. Its CLI arguments
must still be captured in the derived run config/manifest and artifact.

## Launch Runners

Each plan-defined tranche gets one thin case runner. It declares only the
ordered config paths and delegates execution to `paper_exp.runner.run_launch`:

```bash
python runners/NN-phase-tranche.py
```

Screening, selection, and promotion are separate numeric launches when later
configs depend on earlier evidence. The parent validates the whole tranche,
holds one lock, and executes one config at a time. See
[`runners/README.md`](../runners/README.md).
