# Definitive Experiment Plan Manifest

Plan status: placeholder
Reviewed design commit: none
Reviewed case groups: []

> **Launch gate:** scientific configuration allocation, calibration, and
> launches are blocked.

The committed [`A1 formal review packet`](experimental-design/a1-review-packet.md)
is ready for review. Approval must name its exact Git SHA; only then may this
manifest list `A1-lr-screen` as reviewed and permit materialization of its
three configs. Formal plan approval is not RunPod spending or definitive-launch
approval.

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
