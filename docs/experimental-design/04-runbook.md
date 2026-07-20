# 04. Campaign Runbook and Provenance

## 1. Sources of Truth

Authority is field-specific:

- `manifest.json` is authoritative for lifecycle, timestamps, run id, and
  launch provenance;
- the saved run `config.yaml` is authoritative for the scientific inputs that
  actually launched;
- durable metrics, events, predictions, and diagnostic artifacts are
  authoritative for measured values;
- `run-registry.yaml` and `config-registry.yaml` index those artifacts and
  decisions;
- narrative docs and reports interpret only pinned evidence.

The registries are indices, not replacements for artifacts. Never change a
failed manifest to make a registry row look complete, and never let a summarized
manifest field override the saved scientific config snapshot.

## 2. Config Allocation

Only one agent allocates config prefixes at a time.

1. Read `next_config_prefix` and verify it against `rg --files configs`.
2. Search both registries for the stable `design_id`.
3. Copy the closest config and make the smallest scientific changes.
4. Use the next unused contiguous prefix. Do not reserve a number by creating
   an empty config and do not leave intentional gaps.
5. Keep the filename stem under roughly 72 characters to protect Windows atomic
   temporary paths. Put full factor detail in the registry.
6. Validate the config and add a config-registry record with status `ready`.
7. Increment `next_config_prefix` in the same commit.
8. Commit the config and registry before launch. A launched config is immutable.

Compact examples:

```text
121-s1-p14m-a3-aw-lr3em5-s0.yaml
145-s1-p14m-a6post-gpm-k0p1-or.yaml
```

Approximate numbering windows are organizational hints, not reservations:

| Prefix range | Intended use |
| --- | --- |
| `121--349` | Engineering pilots plus S1 training and conditional controls |
| `350--399` | S1 propagation/histogram diagnostics |
| `400--599` | 14M token-promotion and confirmation configs |
| `600+` | Pythia-family scale training and diagnostics |

If a prior phase ends early, the next phase starts at the next unused prefix.

## 3. Stable Ids and Registry Semantics

`design_id` identifies a planned scientific cell independently of its eventual
config number. `run_id` identifies one execution attempt. The relationship is:

```text
one design_id -> one scientific config -> one or more infrastructure attempts
```

A scientific field change, new seed, or new token budget gets a new config and
design id. An infrastructure-only retry may reuse the immutable config and gets
a new run id. The registry `attempt` equals the manifest `run_sequence`; it is
not manually renumbered.

Allowed config transitions are
`planned|blocked -> ready -> active`, `active -> ready` after a retryable
infrastructure failure, and `active -> closed` only after accepting a canonical
run or deciding that no retry will occur. An unlaunched cell may become
`cancelled` with a reason.

If `canonical_run_id` is set, exactly one run row must have `canonical: true`,
its run id must equal the pointer, and its evidence must be `valid` or an
explicitly accepted `provisional`. No other run for that config may be
canonical.

Required config-registry fields include design/config ids and paths, phase,
block, matched control/pair/promotion family, model, steps/tokens, separate
model-initialization and data-order seeds plus the schedule scheme and hash,
architecture/gate factors, optimizer/LR/batch factors, pressure factors,
statuses, canonical run, and notes. The file's `field_order`
lists the complete schema.

Required run-registry fields include attempt, execution mode, and result path, lifecycle and
evidence status, timestamps and Git provenance, completed budget, validation
endpoint, runtime, diagnostic paths, compute/zero summaries, failure details,
external artifact inventory, execution backend and hardware/image provenance,
cost and resume provenance when applicable, and review date.

All tracked filesystem paths are repository-relative and use `/` separators.
Absolute machine paths belong only in transient logs, not registries.

Registry values:

- config status: `planned`, `blocked`, `ready`, `active`, `closed`, `cancelled`;
- decision status: `pending`, `control`, `screened_out`, `promote_tokens`,
  `promote_seed`, `promote_model`, `paper_candidate`, `superseded`;
- release tier: `history`, `reproduction`, `paper`;
- evidence: `unreviewed`, `valid`, `provisional`, `invalid`.

Harness manifests use `running -> completed|failed`. Historical registry rows
may also use `statusless_complete`, `event_stream`, `partial`, or
`inconsistent`. Do not write those historical categories into new manifests.
`statusless_complete` means there is no explicit manifest status and the core
artifact envelope is coherent.

## 4. Preflight

From a clean committed tree:

```text
make test
make check
git status --short
nvidia-smi
```

Then verify:

- config initialization is random;
- expected steps, effective batch, tokens, both seeds, schedule hash,
  validation partition, and
  pressure sites match the registry;
- RN/OR or L1N/OL1 pairs share pressure family, weight, sites, and, for
  Ricker, `c/sigma`; the orthogonal member additionally uses the fixed
  `step_budget: 0.5` and orthogonal update route;
- OR/OL1 primary cells have `step_budget: 0.5`;
- data-order and validation-partition hashes match their paired controls;
- no existing run for the design/config is active;
- enough disk space exists for the final checkpoint and atomic writes;
- the config filename and output path remain below the tested Windows path
  envelope.

Do not use `run-pressure-sweep` for this campaign. That command is hard-coded to
regenerate the historical configs `12--48`.

For cloud runs, the additional readiness, parity, storage, pricing, and secrets
checks in `05-runpod-cloud.md` are part of preflight. The current harness has no
cloud launcher, exact-resume command, or verified artifact uploader; do not
represent those workflows as available until their TODO gates are implemented
and tested.

## 5. Launch

Use the standard pretraining entry point:

```text
python -m paper_exp.cli pretrain --config configs/<config-id>.yaml
```

At launch:

1. verify the harness saved `config.yaml` and a `status: running` manifest;
2. verify launch Git SHA and `git_dirty: false`;
3. add a run-registry row with `lifecycle_status: running` and
   `evidence_status: unreviewed`;
4. set config status to `active`;
5. record the exact log path and expected completion time.

Commit the launch registry update before launching another config. This returns
the tree to clean state and preserves `git_dirty: false` for the next launch.

For a prespecified local tranche of independent pretraining cells, use the
durable serial queue instead of hand-launching each child:

```text
python -m paper_exp.cli run-pretrain-queue \
  --config configs/<first>.yaml \
  --config configs/<second>.yaml \
  --state-path run-logs/<tranche>-queue.json \
  --logs-dir run-logs/<tranche>
```

Tranche mode is valid only when every queued config and its `ready` registry
record are committed together before the queue starts. The queue launches one
blocking child at a time, checks for a clean tree and any existing running
pretrain before every child, verifies the terminal artifact envelope, and
stops on the first inconsistency or failure. Its ignored atomic state file and
logs are orchestration aids; result manifests and artifacts remain
authoritative. Reusing a failed queue state is forbidden: inspect the terminal
artifacts and use a new state path for an explicit retry.

Process ownership is exact, not inferred. While a queue is active, its
blocking child handle is the sole authority for that training process. External
monitors are read-only: they must not use `Stop-Process`, `taskkill`, process
name matching, parent-PID guesses, GPU-owner guesses, or elapsed-time heuristics
to terminate Python processes. Only the code that created and still owns an
exact process handle may terminate it. If ownership cannot be proven, leave the
process untouched and inspect queue state, manifests, and logs. Do not run
auxiliary model-loading audits concurrently with a live local training queue.

Do not edit the queue's execution checkout while it is active. If engineering
continues concurrently, run the tranche from a separate clean Git worktree at
the committed launch SHA. Ensure `PYTHONPATH` resolves to that worktree's
`src`, rather than an editable installation from a different dirty checkout,
and route writable data/results to the intended durable storage. After the
tranche ends, reconcile all child manifests into both registries in one
set-level commit before launching its numbered diagnostic. This bounded
exception replaces per-child launch/terminal registry commits; it does not
relax immutable configs, clean launch provenance, or terminal review.

Do not launch the next pressure member if its matched control failed preflight.
Independent cells may run in parallel only when GPU memory, data-cache access,
and registry allocation remain unambiguous.

On RunPod, matched methods use the same GPU SKU, image digest, precision, and
cache hash. Parallel Pods use separate Git worktrees and writable result paths.
Record the live rate before launch and never put credentials or network access
details in tracked artifacts.

## 6. Monitoring

During a run, check without mutating artifacts:

- process and GPU activity;
- completed step and recent event timestamp;
- finite task/pressure losses;
- OR/OL1 raw/final update ratios and cap binding;
- learned-threshold range, transition mass, and collapse flags;
- projected completion time;
- free disk space.

An unchanged metric is not itself failure. A missing process with a still-running
manifest requires artifact inspection; do not relaunch blindly.

## 7. Terminal Verification

After the process exits:

1. read the terminal manifest first;
2. verify completed steps/tokens against the config;
3. verify `config.yaml`, `metrics.json`, `predictions.jsonl`, `events.jsonl`, and
   the final checkpoint are durable where required;
4. verify final validation token count and finite loss;
5. verify learned ATG parameters reload exactly when applicable;
6. run `make check` and any method-specific tests;
7. update the run registry with the manifest status and measured endpoints;
8. mark evidence `valid`, `provisional`, or `invalid`, with a reason;
9. select at most one canonical run; close the config only if that run is
   accepted or no retry will occur, otherwise return it to `ready`;
10. commit terminal registry updates before another launch, or reconcile the
    full completed tranche in one set-level commit under the queue protocol;
11. only then launch its numbered validation diagnostics.

Training envelope:

```text
configs/<config-id>.yaml
results/<config-stem>/<run-id>/
  config.yaml
  manifest.json
  metrics.json
  predictions.jsonl
  events.jsonl
  checkpoints/final/
```

Propagation diagnostics save `activation_propagation.json`; histogram
diagnostics save their existing histogram artifacts. Clipping remains a
CLI-derived workflow indexed by source run and output path rather than a fake
training config.

## 8. Failures and Retries

- Preserve every failed attempt and its message.
- If scientific inputs change, allocate a new config.
- If only infrastructure changes, reuse the config, create a new run attempt,
  and retain the old directory.
- Never overwrite a checkpoint, metrics file, or terminal manifest.
- Never call a scientifically weak endpoint an execution failure.
- A failed run may be `provisional` only when its durable artifacts support the
  stated metric and the exception is explicit.
- Config `119` is the precedent: the full endpoint exists, but the manifest
  remains failed because an atomic predictions path exceeded Windows limits.

Before a retry, record `failure_type`, the exact failure message, the proposed
infrastructure correction, and whether numerical comparability is preserved.
If an infrastructure interruption leaves a `status: running` manifest, retain
that attempt byte-for-byte and register it as invalid and noncanonical; never
rewrite the manifest to manufacture a terminal state. A retry is eligible only
after an independently reviewed authorization is committed. That authorization
binds the failed run id, attempt inventory, predecessor queue identity and hash,
logs, terminated process identity when relevant, immutable scientific config,
and the next expected attempt number. Launch from a clean commit, restart from
step zero, and use new queue-state and log paths. A second interruption requires
a new adjudication and another unique recovery queue; authorization is never
implicitly reusable. Chained recovery is currently fail-closed rather than
automatic: do not relaunch until the additional attempt has been preserved and
the new recovery lineage is implemented and reviewed.

## 9. Diagnostics and Standard Handoff

Every valid S1 training run receives a lightweight numbered
selection-partition propagation/product diagnostic. Every promoted run also
receives complete-cache propagation. Histograms and clipping are run only for
predeclared representative cohorts.

When the user asks for results, return:

| Method | Val. loss | `R_block` | `R_model` | `z_a` | `z_m` | `z_h` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

Add architecture, gate parameters, `U_arch`, Q/K/V/context zeros, seed count,
validation-token count, and evidence status when relevant. Exact-zero values
must come from the full named validation diagnostic, not the last training
minibatch.

## 10. Handoff Between Agents

A concise handoff must state:

- current phase and next config prefix;
- last completed and currently active design ids;
- active process/log/run paths and ETC;
- any failed or provisional rows;
- which engineering blockers remain;
- which registry rows were changed;
- the next eligible cell and its matched-control dependency;
- current Git SHA and dirty-tree status.

During a local launch set, check and report on a milestone cadence rather than
at every 100 steps. The default is one update about every 4--5 minutes, plus
run completion or failure. Each update states the current config, step and
latest loss, followed by the global launch-set view: completed/total runs,
runs remaining, and ETC for both the current run and the full set.

The receiving agent independently verifies process state and manifests. A chat
message is not sufficient evidence that a run completed.

Current handoff: S1-B3 `t2-l1-flanks` is in prelaunch state with configs
`257--264` materialized in exact queue order. Diagnostic `256` is closed,
valid, and canonical. Prefix `265` remains reserved for the pooled diagnostic
and must not be materialized until all eight tranche runs are terminal,
audited, and reconciled.

Prepared local runner: `C:\tmp\osp-s1-b3-t2-runner` at launch commit
`23247f860d474718bf888a2a11ad2f9132059912`. Its eight result junctions and
token-cache junction are verified. Queue state
`run-logs\s1-b3-t2-l1-flanks-257-264-queue.json` and child-log directory
`run-logs\s1-b3-t2-l1-flanks-257-264\` must remain absent until an approved
launch. The largest dependency-valid campaign below the 12-hour cap is T2
followed, only after reconciliation and diagnostic `265`, by `t3-rk-weight`
configs `266--273`. Measured-equivalent training is approximately 7 h 57 min;
the conservative envelope including both pooled diagnostics and commit gates is
10 h 55 min. T4 is excluded because T2+T3+T4 exceeds 12 hours before gates.

For the explicitly approved unattended T2+T3 launch, use
`scripts/run_s1_b3_t2_t3_campaign.py`. It is one fail-closed owner process
around two separate immutable queues: T2, reconciliation and diagnostic `265`,
then T3 and diagnostic `274`. Between queues it verifies the emitted bundle
hashes and expected registry delta, rejects any registered scientific hard
failure flag, runs the required checks, and commits each clean gate boundary.
Its ignored atomic state is
`run-logs/s1-b3-t2-t3-sequential-controller.json`. It never retries an attempt
or kills an owned child at a wall-clock deadline; 10 h 55 min is the
conservative ETC, not a forced timeout.

## 11. Open-Source and Archive Policy

1. Track every launched campaign config before launch, including negative and
   later screened-out cells.
2. Never renumber or delete configs `1--120`; current reports reference them.
3. Do not erase failed or partial attempts. Record evidence and decision status
   instead.
4. `release_tier: history` preserves the search trajectory;
   `reproduction` identifies the promotion chain; `paper` identifies final
   reported evidence.
5. Retain S1 checkpoints until propagation and promotion are decided. Rejected
   checkpoints may later move to external archival storage only after a stable
   URI and deterministic SHA-256 inventory manifest exist. Store the inventory
   location and its own hash as `artifact_manifest_uri` and
   `artifact_manifest_sha256`. Scalar artifacts and manifests remain.
6. Any archive relocation is an explicit migration that verifies hashes
   atomically. The original repository-relative `result_path` is immutable;
   record the new location in `artifact_uri`. It is never routine cleanup.
7. Before release, backfill historical configs/runs into a frozen inventory,
   add artifact URIs/hashes, run strict integrity checks, and verify every paper
   figure from pinned saved results.

This policy keeps failures, negative results, and successful promotion chains
auditable without forcing all large checkpoints into Git.
