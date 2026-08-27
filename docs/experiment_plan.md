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
> horizon-independent, or convergence claim. Recording the result-backed
> decision reset this manifest to placeholder with no reviewed case groups;
> later groups require a new exact-SHA review before materialization or launch.

The proposed next review scope is exactly `[A2-relu-control, A2-l1-screen]`:
one seed-0 ReLU-only control and h-only L1 at lambda
`{0.1, 0.5, 1, 2, 5}`, all under the unchanged full-pass protocol and current
L1 implementation. Added-seed
replication of the full response is outside this scope. If the later final
winner cohort is reviewed and completed, its matching components may supply
only a selected-lambda ReLU/L1 contrast at added seeds; they do not replicate
the full grid. The user confirmed this scientific matrix on 2026-08-27 and
`OPS-08` is closed; formal review now awaits approval of the final exact
candidate SHA. This proposal is not review or launch authority while the raw
status lines above remain `placeholder`.

Completed A1 configs remain tracked as immutable historical evidence outside
the proposed active A2 scope only when an exact indexed coherent completed run
matches their config snapshot. That preservation does not reactivate A1 or
authorize materialization, retry, or launch.

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
