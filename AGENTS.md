# Agent Instructions

This is a lean research repository intended to accompany a paper. Simplicity
comes first: add code, abstractions, files, and checks only when they directly
help run, reproduce, compare, or explain an experiment.

## Read First

1. `README.md`
2. `docs/code_map.md`
3. `docs/methods.md`
4. `docs/experiment_log.md`
5. `docs/paper_map.md`

Then read the document that owns the task:

- experiment design or configs: `docs/experiment_plan.md` and
  `configs/README.md`;
- measurements: `docs/diagnostics.md`;
- launches, ETC, monitoring, or recovery: `docs/runbook.md`;
- figures: `docs/plotting.md`;
- artifact status and completeness: `results/README.md`.

`exp-plan-v0.md` is a structural preview only. The reviewed
`docs/experiment_plan.md` is the authority for datasets, models, budgets,
seeds, comparisons, promotion rules, diagnostics, and paper outputs. Its
current status is a placeholder, so do not create or launch scientific configs
until the user supplies and reviews the definitive plan.

## Scientific Boundaries

- Do not invent scientific claims, inputs, metrics, thresholds, or results.
  Use `TODO:` where information is genuinely missing.
- Keep repository files, documentation, config values, result records, plot
  labels, and generated outputs in English.
- Pythia experiments are pretraining runs with
  `model.initialization: random`. Do not load released checkpoint weights
  unless the user explicitly requests continuation or fine-tuning.
- The only activation-pressure methods are `none`, `l1_naive` (L1), and
  `orthogonal_l1` (OL1). Keep naive loss pressure and post-AdamW OL1 separate
  in configs, metrics, plots, and prose.
- Fixed one-sided and symmetric threshold gates are explicit architecture
  interventions. Do not add retired or adaptive method families unless a
  future reviewed plan explicitly requires a new implementation.
- Pressure targets must be explicit; current work targets `mlp_hiddens` first.
- Treat changes to activation sites, gates, optimizers, data partitions, or
  model construction as scientific changes, not cleanup.
- Distinguish exact zeros, near-zero mass, logical product opportunities, and
  measured runtime speedups.

## Launch Structure

- Every experiment has an immutable numbered config.
- Each plan-defined tranche has one thin case runner named
  `runners/NN-<phase>-<tranche>.py` and one matching config folder.
- A case runner contains only the ordered config paths and calls the single
  parent, `paper_exp.runner.run_launch`.
- Definitive configs are named `CCC-<case>.yaml`; prefixes are globally unique
  and sequential. Results use `CCC-<case>/001-<timestamp>-<id>/`.
- Run every definitive tranche through its case runner, including a tranche
  with one config. Never launch case runners in parallel.
- The parent validates the complete tranche, holds one lock, executes configs
  serially, and stops on the first failure. Keep scientific selection and
  phase-specific behavior out of the parent.
- Before a launch, report first-run and full-tranche ETCs, projected completion
  time, evidence, assumptions, and uncertainty. Follow `docs/runbook.md` for
  status and recovery; monitoring is read-only.

## Artifacts and Figures

- Mutating scientific commands require a reviewed plan, a clean committed
  checkout, and the experiment lock.
- At launch, save `config.yaml` and a `status: running` manifest with Git
  provenance. Preserve that provenance through the attempt.
- Publish `status: completed` only after required metrics, predictions, events,
  and checkpoints are durable. Escaping exceptions must leave
  `status: failed`.
- Select diagnostic sources by exact config, run, and checkpoint identity.
  Statusless historical runs remain valid when their core envelope is coherent.
- Every paper figure must be regenerable from pinned saved results.
- Keep plotting isolated under `src/paper_exp/plots/`: shared presentation in
  `style.py`, export mechanics in `export.py`, and family-specific loading,
  reduction, labels, axes, and rendering in focused modules.
- Avoid misleading axis truncation; use colorblind-safe redundant encodings and
  show sample size or uncertainty when relevant.

## Working Practice

- Use `docs/code_map.md` to edit the smallest owning module. Avoid compatibility
  shims for code removed from this definitive workflow.
- Prefer one clear implementation over speculative frameworks or guardrails.
  Explain any necessary complexity.
- Preserve unrelated user work and never edit the execution checkout while a
  runner is active.
- Keep datasets, caches, results, logs, figures, build products, and temporary
  files out of Git.
- Keep tests focused on scientific mathematics, serialization, lifecycle, and
  public workflow contracts. Do not duplicate implementation details in tests.
- Remove `.pytest_tmp*` and `pytest_tmp*` scratch directories after testing.

Core verification commands are:

```bash
make test
make check
make smoke
```

Use `make prepare-data`, `make calibrate`, and `make plot` only with explicit
arguments described in the README and runbook.
