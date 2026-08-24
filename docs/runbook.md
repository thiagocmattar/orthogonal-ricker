# Experiment Runbook

This is the operational contract for definitive experiments. The reviewed
[`experiment_plan.md`](experiment_plan.md) is the scientific authority.
The modular proposal and physical-case reuse rules are indexed in
[`experimental-design/README.md`](experimental-design/README.md).

## 1. Launch Gate

Scientific work remains blocked while the definitive plan says:

```text
Plan status: placeholder
```

The infrastructure-only
`experiments/00-infrastructure-smoke/run/00-smoke.yaml` is exempt. A
scientific config requires its case group in the reviewed scope. A launch also
requires a committed scaffold recipe, clean checkout, calibrated ETC, and
explicit launch approval.
Once attempted, a config is immutable. A scientific change gets a new config;
an infrastructure retry gets a new run attempt under the same config.

## 2. Turn the Plan Into Launches

The plan's phases do not map one-to-one to processes. A screen, a
review-dependent selection, and a promotion are separate launches because the
later config list does not exist until earlier evidence is reviewed.

Before allocating a config, resolve its case group and condition fingerprint
under [`experimental-design/cases.yaml`](experimental-design/cases.yaml) and
[`experimental-design/run-reuse.md`](experimental-design/run-reuse.md). Reuse
an existing matching config/seed; a later stage is a consumer, not a duplicate
owner. The group must appear on `experiment_plan.md`'s raw
`Reviewed case groups:` line.

Use one numeric scaffold ID and keep the phase/tranche in the name:

```text
experiments/NN-<phase>-<tranche>/
  run/
    runner.py
    CCC-<case>.yaml
  raw/
  figs/
```

Scaffold numbers define launch order. Config numbers are globally unique and
define case order across scaffolds. Do not reserve or create later scaffold
recipes before their inputs and promotion rules are known.

Each case runner contains only an ordered `CONFIGS` tuple and a call to
`paper_exp.runner.run_launch`. Shared mechanics belong in the parent; science
belongs in configs and the reviewed plan.

## 3. Preflight and ETC

Before a launch, verify:

- exact tranche and dependencies in the reviewed plan;
- catalog membership, one config per condition fingerprint/seed, and every
  declared exact or functional-equivalence alias;
- ordered config list, with no duplicates or unresolved `TODO:` values;
- matched seeds and data ordering where the plan requires them;
- model, data, tokenizer, optimizer, schedule, budget, and validation fields;
- `model.initialization: random` for Pythia pretraining;
- cache metadata and available disk/GPU memory;
- current branch, Git SHA, clean checkout, and absence of another launch;
- focused tests, `make test`, and `make check`.

Before starting, report the estimated duration and local completion time for
the first run and the complete tranche. State the throughput evidence,
assumptions, and uncertainty, then wait for explicit launch approval. If no
defensible same-hardware estimate exists, run the plan-approved calibration
first.

`calibrate` means a 600-second production-shaped timing sample on the launch
hardware, with setup, training, validation, diagnostics, and checkpoint costs
reported separately. Its operational duration cap must not change the
immutable scientific config or truncate definitive pretraining; this remains
blocked until workboard item `OPS-03` is implemented. A scientific
learning-rate screen is a normal training tranche and must use a case runner.

## 4. Launch

Single-config data preparation and throughput calibration commands remain
available:

```bash
make prepare-data CONFIG=experiments/<scaffold>/run/<config>.yaml
make calibrate CONFIG=experiments/<scaffold>/run/<config>.yaml
```

Run every definitive training tranche through its case runner, including a
tranche with only one config:

```bash
python experiments/NN-phase-tranche/run/runner.py
```

The parent runner:

- requires the runner and configs to be tracked;
- requires every config to be a direct sibling of the scaffold runner;
- requires each config's `output.dir` to name that scaffold's exact `raw/`;
- validates the full ordered config list before starting;
- holds one repository launch lock;
- rechecks all existing attempt states under that lock;
- scopes resume decisions to `mode: pretrain`; preparation, calibration, and
  diagnostic attempts cannot satisfy or block a training case;
- reuses exactly one coherent completed pretraining attempt for an unchanged
  config;
- retries coherent failed pretraining attempts only with the explicit
  `--retry-failed` recovery flag;
- aborts before mutation on running, statusless, inconsistent, or ambiguous
  pretraining state;
- executes one config at a time;
- stops immediately on the first escaping failure.

Never start two case runners in parallel. Data preparation and diagnostics run
one config at a time unless the final plan explicitly adds a serial runner for
that workflow.

Concurrent execution is planned but remains blocked by workboard items
`CLOUD-01`, `OPS-05`, and `OPS-06`. The reviewed concurrent design must retain
one authoritative case-runner coordinator and one complete-tranche preflight,
then dispatch distinct immutable configs to bounded isolated subprocess/Pod/GPU
slots with exact-once claims and disjoint attempt roots. Multiple independent
case runners remain forbidden. On a worker failure, the coordinator stops
admitting new configs, lets already-running siblings publish terminal state,
and exits nonzero; it never kills workers in a way that strands `running`
manifests. A multi-Pod coordinator cannot rely on this repository's local lock.

Before enabling that mode, `OPS-06` must demonstrate actual concurrent GPU
workers, device/output isolation, durable artifact collection, completion
reuse, failure draining, and resource teardown using infrastructure-only smoke
inputs. Same-GPU process packing is allowed only if the `OPS-04` measurements
show a useful aggregate-throughput gain for the frozen physical batch.

## 5. Monitor Without Mutation

Inspect the active run's `manifest.json` and `events.jsonl`, plus the exact
runner process, GPU state, and free disk space. Report:

- completed, active, failed, and remaining configs in the case runner;
- active scaffold, config, run directory, step, tokens seen, and latest event
  time;
- latest finite task and method-specific metrics;
- recent throughput;
- updated current-run and full-tranche ETCs;
- stale events, nonfinite values, low disk, or other risks.

Monitoring must not rewrite artifacts or terminate a process by name, guessed
PID, GPU ownership, or elapsed time.

## 6. Completion, Failure, and Retry

After each run:

1. Read the terminal manifest.
2. Verify scaffold/config/run identity, launch provenance, completed budget,
   and finite required metrics.
3. Verify `config.yaml`, `manifest.json`, `metrics.json`, `predictions.jsonl`,
   `events.jsonl`, and required checkpoints.
4. Run method-specific checks before using the run as diagnostic or paper input.
5. Record terminal status, reviewed case class, evidence status, and
   limitations in `docs/experiment_log.md`.

Preserve failed attempts and valid negative results. Never overwrite an
attempt or rewrite it to manufacture completion. A retry creates a new run
directory. A changed scientific input creates a new config.

After recording a reviewed infrastructure failure and its recovery action,
restart the same unchanged case runner explicitly:

```bash
python experiments/NN-phase-tranche/run/runner.py --retry-failed
```

Without that flag, a coherent failed attempt stops before the lock or any new
attempt. With it, the parent skips coherent completed configs and resumes in
config order. The flag is the operator's attestation that the exact failed
attempt was reviewed and recorded as infrastructure; the runner does not infer
that classification from the experiment log. Never use it for a terminal
scientific failure, edit the runner list to remove earlier cases, rerun a
completed config manually, or create a replacement config/seed. Multiple
completed attempts, a running or statusless attempt, changed config snapshot,
symlink/non-directory state, or inconsistent artifacts stop recovery for
read-only review.

## 7. Handoff

A live handoff states the Git SHA, exact scaffold and case runner, ordered configs,
completed/active/remaining counts, active run and process, latest metrics,
current-run and full-tranche ETCs, failures, and next plan-authorized action.
The receiving agent verifies files and processes; chat text is not evidence.
