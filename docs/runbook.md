# Experiment Runbook

This runbook defines the operational contract for definitive experiments. The
reviewed [`experiment_plan.md`](experiment_plan.md) supplies the scientific
content. If the plan and this runbook conflict, stop and ask the user rather
than weakening either contract implicitly.

## 1. Launch Gate

Do not launch scientific work while `experiment_plan.md` declares
`Plan status: placeholder`. `configs/00-smoke.yaml` is the only current config
and is infrastructure-only. After scientific review, the plan owner must set
the raw status line to exactly `Plan status: reviewed`; commands enforce this
gate.

For a scientific config to become launchable:

1. The relevant plan cell, controls, budget, seeds, diagnostics, and acceptance
   rule are explicit.
2. The config uses the next sequential prefix and contains no unresolved
   `TODO:` value.
3. The config passes validation and matches its registered plan cell.
4. The config is reviewed, added to Git, and committed before launch.
5. The execution checkout is clean at the committed launch SHA.

Once launched, the config is immutable. A scientific change requires a new
config. An infrastructure-only retry may reuse the config but must create a new
run attempt and preserve the prior attempt.

## 2. Preflight

Before starting a single run or queue, verify:

- current branch, Git SHA, and clean working tree;
- exact ordered config list and absence of duplicates;
- model architecture, revision, and `initialization: random`;
- dataset/tokenizer revisions and cache metadata/hash;
- training schedule, seed identities, budget, and checkpoint requirements;
- validation partition and expected token count;
- method/site names and naive-versus-orthogonal routing;
- sufficient disk space for events, artifacts, checkpoints, and temporary
  atomic writes;
- expected accelerator, driver/runtime, precision support, and free memory;
- absence of another active experiment runner or ambiguous training process;
- required focused tests, `make test`, and `make check`.

Do not edit the execution checkout or run auxiliary model-loading diagnostics on
the same accelerator while a queue is active.

## 3. Estimate Completion Before Launch

Tell the user the estimated time to completion (ETC) before starting work. The
launch message must state:

- ordered configs and total run count;
- estimated duration and projected local finish time for the first run;
- estimated duration and projected local finish time for the full queue;
- estimate basis, such as a same-model/same-hardware calibration;
- material assumptions and an uncertainty range or confidence label.

Prefer evidence in this order:

1. a completed matched config on the same hardware and software stack;
2. a plan-approved calibration with the same model, sequence length, batch
   shape, precision, and relevant method overhead;
3. a closely matched measured run with an explicit adjustment;
4. a conservative bound from `max_wall_seconds`.

For a step-based run with measured seconds per optimizer step `r`, current step
`s`, and planned steps `S`:

```text
current remaining duration = r * max(S - s, 0)
```

For a queue, add the current remainder and each pending config's estimated
duration. Use per-config rates when method or architecture overhead differs.
When both a step estimate and a remaining wall-time cap exist, the runner uses
the smaller bound.

If no defensible rate or wall-time bound exists, run the approved calibration
first. Never silently omit ETC or present an unsupported point estimate as
precise.

## 4. Launch One Config

After preflight and user-facing ETC reporting:

```bash
make pretrain CONFIG=configs/01-example.yaml
# or
paper-exp pretrain --config configs/01-example.yaml
```

Direct pretraining is for exactly one config. If two or more configs are ready,
use the sequential runner instead.

All mutating scientific commands acquire the same exclusive experiment lock
and require a clean committed checkout. A direct command refuses to start while
the sequential runner owns the lock. The runner passes a one-time lock token to
the exact pretraining child it owns; users must not supply that internal token.

## 5. Launch Multiple Configs

Every multi-run pretraining launch uses one runner invocation with the full
ordered config list:

```bash
make run-configs CONFIGS="configs/01-example.yaml configs/02-example.yaml"
```

Equivalent package command:

```bash
paper-exp run-configs \
  --config configs/01-example.yaml \
  --config configs/02-example.yaml \
  --state run-logs/runner-state.json \
  --logs-dir run-logs
```

The runner contract is:

- preflight the complete ordered list;
- require a clean Git tree;
- hold one exclusive runner lock;
- start one pretraining child at a time;
- write atomic runner state and separate stdout/stderr logs;
- skip only an already completed attempt whose saved config and terminal
  envelope verify exactly;
- refuse unresolved, malformed, or still-running prior attempts;
- verify a child's completed envelope before advancing;
- stop the queue on the first child or verification failure.

Do not hand-launch multiple terminals, start one runner per config, or run
independent GPU jobs in parallel. Sequential execution with one owner is a
scientific comparability and process-ownership requirement.

The current runner executes pretraining configs only. Calibration, data
preparation, and diagnostics run one config at a time; do not create a batch of
those workflows until the reviewed plan defines and the code implements a
single-owner sequential path.

## 6. Monitor Without Mutation

Read current runner progress with:

```bash
make run-status STATE=run-logs/runner-state.json
# or
paper-exp run-status --state run-logs/runner-state.json
```

`run-status` reads and refreshes a copy of the state from saved artifacts; it
does not modify runner state or run artifacts. It reports completed configs,
the active config, step progress, remaining duration, and ETC when measurable.

At milestone updates, also inspect read-only evidence relevant to the active
method:

- exact runner and owned child process state;
- GPU utilization and memory;
- current run directory and manifest status;
- latest event timestamp, optimizer step, tokens seen, and finite task loss;
- pressure loss and gradient/update diagnostics when enabled;
- learned-threshold range, transition mass, and collapse flags when applicable;
- recent throughput and free disk space.

An unchanged metric is not itself a failure. A missing process with a
`status: running` manifest requires artifact inspection; do not relaunch or
rewrite the manifest blindly.

Only the runner that created and still owns the exact child process handle may
terminate it. External monitors must not use process-name matching, guessed
PIDs, parent-PID inference, GPU-owner inference, or elapsed-time heuristics to
stop Python processes.

## 7. Status Requests and Updated ETC

Whenever the user asks for status, inspect current evidence rather than quoting
an earlier update. Report:

- runner state and whether its recorded process is alive;
- completed/total configs and names of failed or remaining configs;
- active config and run directory;
- optimizer step/planned steps, percentage, tokens seen, and latest event time;
- recent throughput and relevant latest finite metrics;
- updated remaining duration and projected finish time for the active run;
- updated remaining duration and projected finish time for the full queue;
- the rate window or prior runs used and any change in confidence;
- any risk such as stale events, low disk, nonfinite metrics, or cap binding.

The built-in runner currently estimates seconds per step from observed elapsed
time and completed steps, and averages available per-run rates for pending
items. Treat the displayed ETC as an estimate, not a guarantee. Recompute or
qualify it when warmup, validation, checkpoint writes, method overhead, or a
hardware change makes the historical rate unrepresentative.

## 8. Terminal Verification

After each process exits:

1. Read the terminal manifest first.
2. Verify its config/run identity and launch provenance.
3. Compare completed steps and tokens with the config and stopping rule.
4. Verify `config.yaml`, `manifest.json`, `metrics.json`,
   `predictions.jsonl`, and `events.jsonl`.
5. Verify the final checkpoint and optimizer state when required.
6. Verify finite final validation loss and the exact validation coverage.
7. Reload learned gates and optimizer state when continuation is part of the
   contract.
8. Run method-specific checks, `make test`, and `make check`.
9. Assign `valid`, `provisional`, or `invalid` evidence status with a reason.
10. Update `docs/experiment_log.md`; update `docs/paper_map.md` only for pinned
    paper evidence.

Diagnostics may begin only after their source run and required checkpoint have
passed terminal verification.

## 9. Failures and Retries

- Preserve the failed attempt, terminal message, logs, config snapshot, and any
  durable artifacts.
- Never overwrite a checkpoint, metrics file, predictions file, or terminal
  manifest.
- Never rewrite `running` to `failed` or `completed` after the fact merely to
  simplify bookkeeping.
- Determine whether the failure is scientific, numerical, infrastructure, or
  artifact-publication related.
- A scientific input change requires a new config.
- An infrastructure-only correction may create a new attempt for the same
  immutable config after exact process exit and artifact state are verified.
- Use a new runner-state path for a reviewed retry; never reuse a failed queue
  state as if it were fresh.
- Do not call an unfavorable but valid endpoint an execution failure.

If process ownership, source artifacts, or retry comparability is ambiguous,
stop and ask the user.

## 10. Agent Handoff

A live-run handoff must state:

- Git SHA and dirty-tree state;
- exact runner state/log paths and runner identity;
- ordered configs and completed/active/remaining counts;
- active config, run directory, process identity, step, and latest metrics;
- current-run and full-queue ETCs with estimate basis;
- failed, provisional, or blocked items;
- the next action allowed by the plan.

The receiving agent independently verifies the state, process, manifests, and
events. Chat text is not completion evidence.
