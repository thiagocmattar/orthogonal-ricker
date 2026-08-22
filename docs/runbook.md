# Experiment Runbook

This is the operational contract for definitive experiments. The reviewed
[`experiment_plan.md`](experiment_plan.md) is the scientific authority.
[`../exp-plan-v0.md`](../exp-plan-v0.md) currently informs structure only.

## 1. Launch Gate

Scientific work remains blocked while the definitive plan says:

```text
Plan status: placeholder
```

The infrastructure-only `configs/00-smoke.yaml` is exempt. A scientific launch
requires a reviewed plan, committed configs and runner, and a clean checkout.
Once attempted, a config is immutable. A scientific change gets a new config;
an infrastructure retry gets a new run attempt under the same config.

## 2. Turn the Plan Into Launches

The plan's phases do not map one-to-one to processes. A screen, a
review-dependent selection, and a promotion are separate launches because the
later config list does not exist until earlier evidence is reviewed.

Use one numeric launch ID and keep the phase/tranche in the name:

```text
runners/NN-<phase>-<tranche>.py
configs/NN-<phase>-<tranche>/

configs/NN-<phase>-<tranche>/CCC-<case>.yaml
```

Runner numbers define launch order. Config numbers are globally unique and
define run order across folders. Do not reserve or create later launch files
before their inputs and promotion rules are known.

Each case runner contains only an ordered `CONFIGS` tuple and a call to
`paper_exp.runner.run_launch`. Shared mechanics belong in the parent; science
belongs in configs and the reviewed plan.

## 3. Preflight and ETC

Before a launch, verify:

- exact tranche and dependencies in the reviewed plan;
- ordered config list, with no duplicates or unresolved `TODO:` values;
- matched seeds and data ordering where the plan requires them;
- model, data, tokenizer, optimizer, schedule, budget, and validation fields;
- `model.initialization: random` for Pythia pretraining;
- cache metadata and available disk/GPU memory;
- current branch, Git SHA, clean checkout, and absence of another launch;
- focused tests, `make test`, and `make check`.

Before starting, report the estimated duration and local completion time for
the first run and the complete tranche. State the throughput evidence,
assumptions, and uncertainty. If no defensible same-hardware estimate exists,
run the plan-approved calibration first.

`calibrate` means a short throughput estimate. A scientific learning-rate
screen, including a future plan phase named "LR calibration," is a normal
training tranche and must use a case runner.

## 4. Launch

Single-config data preparation and throughput calibration commands remain
available:

```bash
make prepare-data CONFIG=configs/<launch-id>/<config>.yaml
make calibrate CONFIG=configs/<launch-id>/<config>.yaml
```

Run every definitive training tranche through its case runner, including a
tranche with only one config:

```bash
python runners/NN-phase-tranche.py
```

The parent runner:

- requires the runner and configs to be tracked;
- requires the matching config folder;
- validates the full ordered config list before starting;
- holds one repository launch lock;
- executes one config at a time;
- stops immediately on the first escaping failure.

Never start two case runners in parallel. Data preparation and diagnostics run
one config at a time unless the final plan explicitly adds a serial runner for
that workflow.

## 5. Monitor Without Mutation

Inspect the active run's `manifest.json` and `events.jsonl`, plus the exact
runner process, GPU state, and free disk space. Report:

- completed, active, failed, and remaining configs in the case runner;
- active config, run directory, step, tokens seen, and latest event time;
- latest finite task and method-specific metrics;
- recent throughput;
- updated current-run and full-tranche ETCs;
- stale events, nonfinite values, low disk, or other risks.

Monitoring must not rewrite artifacts or terminate a process by name, guessed
PID, GPU ownership, or elapsed time.

## 6. Completion, Failure, and Retry

After each run:

1. Read the terminal manifest.
2. Verify config/run identity, launch provenance, completed budget, and finite
   required metrics.
3. Verify `config.yaml`, `manifest.json`, `metrics.json`, `predictions.jsonl`,
   `events.jsonl`, and required checkpoints.
4. Run method-specific checks before using the run as diagnostic or paper input.
5. Record the evidence and its limitations in `docs/experiment_log.md`.

Preserve failed attempts and valid negative results. Never overwrite an
attempt or rewrite it to manufacture completion. A retry creates a new run
directory. A changed scientific input creates a new config.

## 7. Handoff

A live handoff states the Git SHA, exact case runner, ordered configs,
completed/active/remaining counts, active run and process, latest metrics,
current-run and full-tranche ETCs, failures, and next plan-authorized action.
The receiving agent verifies files and processes; chat text is not evidence.
