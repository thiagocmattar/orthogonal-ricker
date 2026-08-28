# Definitive Experiment Plan Manifest

Plan status: placeholder
Reviewed design commit: none
Reviewed case groups: []

> **Current state:** All eleven A1 configs `001`–`011` are completed,
> eligible, valid, immutable evidence and must never be rerun. Applying the
> predeclared rule—lowest finite final selection loss across the exact
> eleven-cell cohort at 400M tokens—froze `lr_14m = 6.4e-2` from config
> `008-a1-lr-6p4e-2`, run `001-20260826-190546-4df1c441`, checkpoint
> `checkpoints/final`, with final selection loss `4.0587728086270785`.
> Configs `009`–`011` completed serially on one A40. This is the best tested
> setting for seed 0 at the fixed 400M-token horizon, not a global,
> horizon-independent, or convergence claim. A1 remains historical-only.
> A2 configs `012`–`017` and diagnostic `018` are also completed, accepted,
> immutable evidence and must not be rerun.

There is currently no active reviewed group. The plan is a placeholder while
the exact five-cell seed-0 `A3-ol1-screen` candidate is reviewed. The candidate
uses topology `A1-H`, h-only OL1 at lambda `{0.1, 0.5, 1, 2, 5}`, and proposed
fixed `ol1_step_budget = 0.1`; it reuses the accepted A2 control and L1
evidence. No A3 config may be materialized until the user approves the
committed [`A3 review packet`](experimental-design/a3-review-packet.md) and
this manifest records that reviewed SHA and group.

Completed A1 and A2 configs remain tracked only because the experiment log
indexes exact coherent completed runs whose immutable config snapshots match.
That preservation does not reactivate either group or authorize
materialization, retry, or launch.

The [`A1 formal review packet`](experimental-design/a1-review-packet.md) was
approved at design commit `54be534f383001b4af3d3b43597e135d4ca6653d`.
That historical approval permitted its three immutable configs and the now
completed original tranche only. The two separately reviewed single-cell
boundary extensions and the reviewed three-cell high-LR extension are also
complete; their immutable review and launch packets remain historical
provenance and authorize no rerun. The completed second three-cell extension
is owned by
[`a1-very-high-lr-extension-review-packet.md`](experimental-design/a1-very-high-lr-extension-review-packet.md)
and was reviewed at exact design commit
`2320d542b14926315a17e873afac2d41a40d6814` before materialization; configs
`009`–`011` then completed serially on one A40.

This file remains the sole launch-status authority. The proposed scientific
design is split into focused components under
[`experimental-design/`](experimental-design/README.md) so agents read and edit
one owning document at a time.

## Normative Components When Reviewed

- [`experimental-design/protocol.md`](experimental-design/protocol.md): shared
  data/model, optimization, budget, validation, and checkpoint settings.
- [`experimental-design/cases.yaml`](experimental-design/cases.yaml): exact
  grids, seed allocations, physical case groups, and reuse aliases.
- [`experimental-design/run-reuse.md`](experimental-design/run-reuse.md):
  fingerprint, allocation, and cross-stage reuse contract.
- [`experimental-design/phases/a-pressure.md`](experimental-design/phases/a-pressure.md),
  [`b-threshold.md`](experimental-design/phases/b-threshold.md), and
  [`c-scale.md`](experimental-design/phases/c-scale.md): stage dependencies,
  selection rules, and stop rules.
- [`experimental-design/outputs.md`](experimental-design/outputs.md): required
  evidence, paper outputs, and claim limits.
- [`experimental-design/decisions.md`](experimental-design/decisions.md):
  reviewed upstream selections used to materialize dependent cases.

The method and measurement contracts in [`methods.md`](methods.md) and
[`diagnostics.md`](diagnostics.md), launch procedure in
[`runbook.md`](runbook.md), and scaffold/config contract in
[`../experiments/README.md`](../experiments/README.md) remain independently
authoritative.

Non-normative navigation, executive, workboard, and manuscript notes are
listed in the [`experimental-design` index](experimental-design/README.md).

## Review and Change Rule

Review is incremental: only case groups listed on the raw
`Reviewed case groups:` line are in scope. To review a group:

1. Resolve its design/implementation workboard blockers. Same-hardware
   calibration and ETC are later per-launch checks, not review blockers.
2. Ensure its case groups have no unresolved decision reference or unapproved
   `TODO:` value.
3. Review the applicable normative components at one committed Git SHA.
4. Record the full 40-character lowercase design SHA and the exact group IDs
   as a one-line YAML list on the raw lines above, for example
   `Reviewed case groups: [A1-lr-screen]`.
5. Change the status line to exactly:

```text
Plan status: reviewed
```

After review, materialize only the listed groups: validate fingerprints, reuse
matching tracked configs, and allocate immutable config numbers only for new
physical cases. Then perform calibration/ETC and return for explicit launch
approval.

Any scientific edit affecting a reviewed group resets this file to
`Plan status: placeholder`, `Reviewed design commit: none`, and
`Reviewed case groups: []` until the new scope is reviewed. Recording a
predeclared upstream selection is a normative edit and follows the same rule.

While status is `placeholder`, do not create scientific scaffolds/configs,
prepare scientific caches, calibrate, pretrain, run scientific diagnostics, or
produce paper evidence. The infrastructure-only smoke remains exempt and is
not a scientific template or result.
