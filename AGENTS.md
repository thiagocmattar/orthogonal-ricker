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

- experiment design or configs: `docs/experiment_plan.md`,
  `docs/experimental-design/README.md`, and `experiments/README.md`;
- measurements: `docs/diagnostics.md`;
- launches, ETC, monitoring, or recovery: `docs/runbook.md`;
- RunPod provisioning, SSH, persistence, cost control, or teardown: load the
  installed `runpod` router skill first, then follow
  `docs/runbook.md#runpod-operations`;
- figures: `docs/plotting.md`;
- artifact status and completeness: `experiments/README.md`.

`docs/experiment_plan.md` and the exact normative components it lists are the
authority for datasets, models, budgets, seeds, comparisons,
promotion rules, diagnostics, and paper outputs. All eleven A1 configs/runs
`001`–`011` are accepted historical evidence and must never be rerun. The
predeclared rule selected `lr_14m = 6.4e-2` from config
`008-a1-lr-6p4e-2`, run `001-20260826-190546-4df1c441`, checkpoint
`checkpoints/final`, at the fixed 400M-token horizon. This is the best tested
setting for one seed, not a global, horizon-independent, or convergence claim.
The exact A2 groups `[A2-relu-control, A2-l1-screen]` are reviewed at design
commit `3a4b047b1f4712d07b32314461913aae09cc46a7`. Training configs `012`–`017`
and diagnostics `018`–`019` are completed immutable evidence and must not be
rerun. Diagnostic config `020` is materialized and its implementation is ready,
but the diagnostic is unattempted. Non-evidence production-shaped local timing
is permitted; definitive diagnostic execution remains separately
approval-gated. Use
`docs/experimental-design/cases.yaml` and
`docs/experimental-design/run-reuse.md` to ensure one physical config per
scientific condition and seed.

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
- Pressure targets must be explicit; current work targets `h` first.
- Treat changes to activation sites, gates, optimizers, data partitions, or
  model construction as scientific changes, not cleanup.
- Distinguish exact zeros, near-zero mass, logical product opportunities, and
  measured runtime speedups.

## Launch Structure

- Every experiment has an immutable numbered config.
- Each plan-defined tranche has one scaffold named
  `experiments/NN-<phase>-<tranche>/` with exactly `run/`, `raw/`, and `figs/`
  as its owned directories.
- The tracked `run/` directory contains one thin `runner.py` and every
  immutable config for the tranche. Never scatter tranche runners or configs
  into repository-level directories.
- A case runner contains the ordered config paths and calls the single parent,
  `paper_exp.runner.run_launch`. Only a reviewed bounded execution exception
  may add explicit tracked operational authorization metadata. A1's dormant
  worker authorization covers only a superseded `001`–`008` launch shape; it
  did not cover configs `009`–`011` or authorize worker-slot execution.
- Definitive configs are named `CCC-<case>.yaml`; prefixes are globally unique
  and sequential. Raw attempts use
  `raw/CCC-<case>/001-<timestamp>-<id>/` inside the owning scaffold.
- Run every definitive tranche through its case runner, including a tranche
  with one config. Never launch case runners in parallel.
- The parent validates the complete tranche and holds one lock. It executes
  configs serially by default. The completed original three-cell A1 launch
  historically used one coordinator and two distinct homogeneous A40 slots;
  configs `004`–`011` ran serially, with `006`–`008` completing on one A40
  from clean commit `d410572` and `009`–`011` also completing serially on one
  A40 through the plain case-runner invocation with no worker slots.
  Multiple case runners, same-GPU packing, heterogeneous slots, and multi-Pod
  dispatch remain forbidden. Keep
  scientific selection and phase-specific behavior out of the parent.
- Before a launch, report first-run and full-tranche ETCs, projected completion
  time, evidence, assumptions, and uncertainty. Follow `docs/runbook.md` for
  status and recovery; wait for explicit launch approval, and keep monitoring
  read-only.

## Artifacts and Figures

- Treat one `experiments/NN-<phase>-<tranche>/` scaffold as the ownership
  boundary for its tracked launch recipe, ignored raw attempts, and ignored
  generated figures. Keep `run/`, configs, runners, and directory keepers in
  Git; keep generated `raw/` and `figs/` payloads out of Git.
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

- Treat each coherent edit or migration step as its own checkpoint: run the
  verification proportional to that step and commit it before starting the
  next step. Keep commits small and do not combine unrelated changes.
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
