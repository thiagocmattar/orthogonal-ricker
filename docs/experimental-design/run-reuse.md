# Physical-Run Reuse Contract

The goal is one physical training case for each scientific condition and seed,
regardless of how many stages consume it. The exact planned groups and aliases
live in [`cases.yaml`](cases.yaml); this file defines how agents materialize and
reuse them. Attempt recovery belongs only to
[`../runbook.md`](../runbook.md#6-completion-failure-and-retry).

## Three Identities

| Identity | Meaning | Authority |
| --- | --- | --- |
| Condition fingerprint | Canonical scientific factors, including seed | `cases.yaml` plus the immutable config snapshot |
| Config ID | One immutable recipe owning that fingerprint | Tracked `experiments/NN-*/run/CCC-*.yaml` |
| Attempt ID | One execution of that config | `raw/CCC-*/NNN-*/manifest.json` |

A retry is another attempt under the same config. A changed scientific field,
including seed or budget, is a new condition/config. A later stage that needs
an existing condition is a consumer, not a new owner.

## Fingerprint Rule

Canonicalize the complete validated training config using
`cases.yaml:condition_fingerprint`, excluding only its exact `exclude_paths`.
Two conditions with the same fingerprint are duplicates even if their stage,
case name, or intended paper role differs.

Resolve every `decision:` reference and `TODO:` before hashing. Serialize a
schema-versioned canonical JSON object with sorted keys, normalized numeric
values, and repository-relative POSIX paths. Store the resulting fingerprint
and reviewed training-implementation identity in the immutable config. The
attempt manifest preserves both plus the complete config SHA-256. Exact Git
provenance remains a separate manifest field.

The seed is part of the fingerprint. Adding seeds 1 and 2 never reruns seed 0.
Changing from `lr-400m` to `full-pass-wrap` is a new condition. Adding a
post-hoc diagnostic to a saved compatible final checkpoint is not new
training.

The training-implementation identity is behavioral rather than a Git SHA.
Exact Git provenance is still mandatory in every attempt. A new dormant method
path may retain the identity only when it cannot affect the earlier config's
active path. Before reusing selected A1 A0 evidence from a later code revision,
record an unchanged-active-path check covering random initialization,
forward/loss, task gradients, optimizer/schedule, data order, validation, and
checkpoint semantics. If that check is absent or fails, the identity changes
and cross-revision reuse stops for review; never rerun A1 silently to conceal
the incompatibility.

## Allocation Procedure

Before creating a config:

1. Resolve the case group and every required value from `decisions.md`.
2. Expand only the physical cells needed by the current tranche; preserve
   conceptual aliases without configs.
3. Compute each condition fingerprint and scan all tracked scientific configs.
4. If a fingerprint exists, pin that config as the source; do not copy,
   relabel, or rerun it.
5. If no fingerprint exists, allocate the next global config number in the
   owning tranche and write the group ID and fingerprint into the config.
6. Verify that each runner lists only its new owning configs. Reused configs
   from another tranche are inputs to decisions/diagnostics, never members of
   the later runner.

No config allocation is allowed until automated catalog/fingerprint validation
in [`workboard.md`](workboard.md) is closed and the case group is listed in the
reviewed scope of [`../experiment_plan.md`](../experiment_plan.md).

## Catalog Validation Invariants

Before materialization or launch, automated validation must establish that:

- every conceptual cell resolves to exactly one primary case, exact alias, or
  reviewed functional-equivalence alias;
- no two primary cases have the same condition fingerprint;
- exact aliases are acyclic, resolve to an equal fingerprint, and never
  increase sample size;
- each functional-equivalence alias names a reviewed rule, passes its declared
  equivalence test, and never increases sample size;
- each config fingerprint equals the fingerprint derived from its complete
  normalized content; and
- a runner owns only newly materialized physical configs, while every reused
  input is pinned in analysis by exact tranche, config, run, and checkpoint.

## Predeclared Reuse Map

| Primary physical case | Consumers that must reuse it |
| --- | --- |
| Selected A1 `A0`, seed 0 | B1 control and B2 component baseline |
| A2 neutral ReLU-only (`pressure: none`) by seed | A2/A3 zero-pressure curve anchor and B2 ReLU component |
| A2 L1 at `lambda_B2` by seed | B2 L1-only component when the selected lambda matches |
| A3 OL1 at `lambda_B2`, seed 0 | B2 OL1-only component |
| B1 threshold point, seed 0 | B2 threshold-only component for the same family/kappa |
| B1 topology `A2`, one-sided, `kappa = 0`, seed 0 | All six symmetric-attention `kappa = 0` conceptual anchors |
| B2 combined winner, seed 0 | Seed-0 member of the B2 headline cohort |
| C2 `A0`, ReLU-only, and matching L1 cases | C3 component controls for the same model/seed |
| Any seed-0 screen point later promoted | Seed-0 member of the promoted cohort; add only seeds 1/2 |

Exact-fingerprint reuse never crosses a different model, seed, budget, LR,
data-order hash, precision, optimizer recipe, topology/threshold, pressure
setting, or required training artifact policy. The only exception is a
catalogued functional-equivalence alias after its named acceptance contract
passes review.

## Where Tracking Lives

- Planned groups and semantic aliases: `cases.yaml`.
- Open work and stage readiness: `workboard.md`.
- Physical-case ownership and fingerprint: the tracked immutable config,
  indexed by scanning all tracked configs.
- Live/terminal attempt state: the exact saved manifest.
- Accepted evidence and limitations: `docs/experiment_log.md`.
- Frozen upstream choices: `decisions.md`.
- Paper consumers: `docs/paper_map.md`.

Do not create a second manual per-attempt checklist. It will drift from the
manifest and makes duplicate execution more likely.
