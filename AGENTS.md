# Agent Instructions

This is a lean research repository intended to accompany a paper. Favor direct,
auditable implementations over abstractions that do not materially improve
reproduction, comparison, or explanation.

## Read First

1. `README.md`
2. `docs/experiment_plan.md`
3. `docs/methods.md`
4. `docs/diagnostics.md`
5. `docs/runbook.md`
6. `docs/plotting.md`
7. `docs/code_map.md`

The reviewed experiment plan is the authority for scientific questions,
models, datasets, budgets, seeds, comparisons, promotion rules, diagnostics,
and paper outputs. The current plan is a placeholder. Until the user supplies
and reviews the definitive plan, do not create or launch scientific configs.

## Core Commands

- `make install`
- `make test`
- `make check`
- `make smoke`
- `make prepare-data CONFIG=configs/<file>.yaml`
- `make calibrate CONFIG=configs/<file>.yaml`
- `make pretrain CONFIG=configs/<file>.yaml`
- `make run-configs CONFIGS="configs/01.yaml configs/02.yaml"`
- `make run-status STATE=run-logs/runner-state.json`
- `make plot KIND=<kind> RUN_DIR=<run-dir> OUTPUT=<figure.pdf> [PNG=1]`

## Scientific Rules

- Do not invent scientific claims, datasets, models, metrics, thresholds,
  results, or plan details. Use `TODO:` when information is missing.
- Keep repository files, documentation, configs, metrics, labels, and generated
  outputs in English.
- Pythia pretraining uses the Pythia architecture with
  `model.initialization: random`. Loading released checkpoint weights is a
  continuation or fine-tuning run and requires explicit user authorization.
- Keep naive pressure and Adam-step orthogonal pressure separate in configs,
  metrics, plots, and prose.
- Treat architecture changes, activation-site changes, gate changes, and
  optimizer changes as scientific interventions, not cleanup.
- Every experiment and diagnostic has a numbered config unless the documented
  workflow is explicitly derived from a saved source run, such as a clipping
  sweep.
- Assign sequential prefixes to definitive configs, result groups, run
  attempts, and figures. `00-smoke.yaml` is infrastructure-only; definitive
  experiment numbering starts at `01`.
- Once launched, a config is immutable. A scientific change gets a new config;
  an infrastructure-only retry gets a new run attempt under the same config.

## Multiple Pretraining Runs and ETC

When more than one pretraining run is to be launched, pass the complete ordered
config list to one `paper-exp run-configs` invocation. It must execute one child
at a time. Do not hand-launch separate runners, start parallel GPU jobs, or
split the list across terminals. Non-pretraining workflows currently run one at
a time; do not create a multi-run tranche for them until the reviewed plan adds
an explicit sequential execution contract.

Before launch, tell the user:

- the ordered config list and run count;
- the estimated time to completion (ETC) for the first run;
- the ETC for the complete queue;
- the projected local completion time;
- the throughput evidence, assumptions, and uncertainty behind the estimate.

If no defensible same-hardware estimate exists, run the plan-approved
calibration first. Never silently omit ETC.

When the user asks for a status update, run `paper-exp run-status` and inspect
the exact runner state, current run manifest and events, exact owned process/GPU
state when available, and free disk space. Report:

- completed, active, failed, and remaining runs;
- current config and optimizer step versus plan;
- latest finite task and pressure metrics relevant to the method;
- recent throughput;
- updated current-run and full-queue ETCs with projected finish times;
- any change in assumptions or confidence.

Monitoring is read-only. Never terminate a process by name, guessed PID,
elapsed-time heuristic, or GPU-owner inference. Only the runner that owns the
exact child handle may terminate it.

## Artifact and Lifecycle Rules

- Every mutating scientific command requires `Plan status: reviewed`, a clean
  committed checkout, and the exclusive experiment lock.
- At launch, every retained workflow writes the immutable `config.yaml` snapshot and a
  `status: running` manifest with launch Git provenance.
- Publish `status: completed` only after required metrics, predictions, events,
  and checkpoints are durable.
- Escaping exceptions must leave `status: failed` with `finished_at` and failure
  details.
- Never rewrite a failed or interrupted attempt to manufacture completion.
- Select diagnostic source checkpoints by exact config and run identity.
- A completed run must contain the artifact envelope documented in
  `results/README.md`.
- Every paper figure must be regenerable from pinned saved artifacts and have
  deterministic input provenance.

## Plotting Rules

- Read only saved artifacts. Plotting must not train, re-evaluate, or silently
  substitute a newer run.
- Keep loading and pinning, pure reductions, rendering, and export as distinct
  boundaries.
- Use shared presentation rules in `src/paper_exp/plot_style.py` and mechanical
  layout/export helpers in `src/paper_exp/plot_api.py`.
- Keep family-specific cohorts, reductions, labels, axes, and renderers in one
  focused module defined by the experiment plan.
- Avoid misleading axis truncation, use colorblind-safe redundant encodings,
  show sample size and uncertainty when relevant, and distinguish logical
  compute opportunities from measured speedups.

## Repository Hygiene

- Preserve unrelated user changes in a dirty worktree.
- Do not edit the execution checkout while a runner is active.
- Pytest scratch directories such as `.pytest_tmp_run` or `pytest_tmp*` must be
  removed after testing finishes.
- Keep token caches, results, logs, generated figures, and temporary files out
  of Git.
- Prefer one clear script over many clever abstractions. Explain unavoidable
  complexity.
- Run focused tests while iterating, then `make test` and `make check` before
  handoff.
