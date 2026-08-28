# Experimental Design Workspace

> This index does not authorize calibration, configuration allocation, or
> launch. Current review status and exact case-group scope live only in
> [`../experiment_plan.md`](../experiment_plan.md).

This package splits the working plan by ownership so an agent reads only the
material needed for its task. It is the design workspace for the lean
Pythia-14M discovery and Pythia-70M/410M replication program.

## Start Here

| Task | Read |
| --- | --- |
| Advisor review | [`executive.md`](executive.md) |
| Reviewed A2 post-hoc clipping frontier | [`a2-clipping-review-packet.md`](a2-clipping-review-packet.md) |
| Backlog A3 scientific candidate | [`a3-review-packet.md`](a3-review-packet.md) |
| Reviewed six-run A2 plan | [`a2-review-packet.md`](a2-review-packet.md) |
| Accepted A2 calibration and ETC | [`a2-calibration-packet.md`](a2-calibration-packet.md) |
| Original three-cell A1 review | [`a1-review-packet.md`](a1-review-packet.md) |
| Historical `4e-3` A1 design review | [`a1-boundary-extension-review-packet.md`](a1-boundary-extension-review-packet.md) |
| Historical `4e-3` definitive launch review | [`a1-boundary-extension-launch-packet.md`](a1-boundary-extension-launch-packet.md) |
| Historical `8e-3` A1 design review | [`a1-8e-3-boundary-extension-review-packet.md`](a1-8e-3-boundary-extension-review-packet.md) |
| Historical `8e-3` definitive launch review | [`a1-8e-3-launch-packet.md`](a1-8e-3-launch-packet.md) |
| Completed three-cell high-LR extension | [`a1-high-lr-extension-review-packet.md`](a1-high-lr-extension-review-packet.md) |
| Historical high-LR launch record | [`a1-high-lr-launch-packet.md`](a1-high-lr-launch-packet.md) |
| Reviewed very-high-LR extension | [`a1-very-high-lr-extension-review-packet.md`](a1-very-high-lr-extension-review-packet.md) |
| Completed very-high-LR launch | [`a1-very-high-lr-launch-packet.md`](a1-very-high-lr-launch-packet.md) |
| Historical A1 calibration and ETC | [`a1-calibration-packet.md`](a1-calibration-packet.md) |
| Shared scientific configuration | [`protocol.md`](protocol.md) |
| Stage design or selection | [Phase A](phases/a-pressure.md), [B](phases/b-threshold.md), or [C](phases/c-scale.md), plus [`cases.yaml`](cases.yaml) |
| Config allocation or reuse | [`cases.yaml`](cases.yaml), [`run-reuse.md`](run-reuse.md), and [`../../experiments/README.md`](../../experiments/README.md) |
| Diagnostics or paper output | [`outputs.md`](outputs.md) and [`../diagnostics.md`](../diagnostics.md) |
| Implementation work | [`workboard.md`](workboard.md) and [`../code_map.md`](../code_map.md) |
| Launch, recovery, or ETC | [`../runbook.md`](../runbook.md) |
| Record a selected value | [`decisions.md`](decisions.md) |

A1 configs `001`–`011` are completed, immutable evidence and must never be
rerun. The factor-two extension at `1.28e-1`, `2.56e-1`, and `5.12e-1`
completed without improving on config `008-a1-lr-6p4e-2`, which the final
eleven-cell lowest-loss rule selects for dependent experiments. The complete
evidence cohort is indexed by [`../experiment_log.md`](../experiment_log.md)
and its execution record is
[`a1-very-high-lr-launch-packet.md`](a1-very-high-lr-launch-packet.md). The
eleven-cell scope was reviewed at design commit
`2320d542b14926315a17e873afac2d41a40d6814`, configs `009`–`011` were
materialized at commit `1fd0914068531d1c05e047f95352fabee3e3b04a`, and the
launch packet was approved at commit
`2f8f55cd57b25828fea10f015992b0824b28f3a6`.

A2 pretraining and diagnostics `018`–`019` are complete: the seed-0 ReLU
control, five h-only L1 cells, and accepted activation diagnostics are
immutable evidence. The joint six-site post-hoc clipping design is reviewed,
config `020` is materialized, and its implementation is ready. Non-evidence
local timing calibration completed at implementation commit `904bdc4`, with a
conservative full-run ETC of about 2 minutes 15 seconds. Definitive diagnostic
execution remains separately approval-gated. A3 is deferred in the backlog.
Its packet and implementation create no materialization authority; no A3 config
may be created until a future explicit review is recorded in
[`../experiment_plan.md`](../experiment_plan.md).

## One Fact, One Owner

| Information | Authoritative owner |
| --- | --- |
| Launch status and reviewed component list | [`../experiment_plan.md`](../experiment_plan.md) |
| Site/topology vocabulary and L1/OL1 mathematics | [`../methods.md`](../methods.md) |
| Metric definitions and artifact contracts | [`../diagnostics.md`](../diagnostics.md) |
| Model/data pins, optimizer, schedule, budget, and validation protocol | [`protocol.md`](protocol.md) |
| Exact conceptual grids, physical case groups, seeds, and reuse aliases | [`cases.yaml`](cases.yaml) |
| Stage questions, dependencies, and decision rules | [Phase A](phases/a-pressure.md), [B](phases/b-threshold.md), and [C](phases/c-scale.md) |
| Figures, tables, evidence levels, and permitted claims | [`outputs.md`](outputs.md) |
| Open implementation/review work and acceptance criteria | [`workboard.md`](workboard.md) |
| Frozen upstream selections | [`decisions.md`](decisions.md) |
| Attempt status | Saved `manifest.json`; indexed by [`../experiment_log.md`](../experiment_log.md) |
| Reviewed case class and evidence status | [`../experiment_log.md`](../experiment_log.md), using [`protocol.md`](protocol.md) and [`../diagnostics.md`](../diagnostics.md) |
| Paper evidence | [`../paper_map.md`](../paper_map.md) |
| Paper-only rationale | [`manuscript-notes.md`](manuscript-notes.md) |

Summaries may repeat a value for readability, but agents must resolve any
mismatch in favor of the owner above. Do not copy attempt status, scalar
results, or decision values into multiple documents.

## Stage Graph

```text
A1 -> A2 -> A3
      |
      +----> B1

A3 + B1 -> B2
C1 + A2 -> C2
C2 + B2 -> C3
```

- [Phase A](phases/a-pressure.md): 14M learning rate, L1 spillover, and OL1.
- [Phase B](phases/b-threshold.md): 14M threshold placement and combined OL1.
- [Phase C](phases/c-scale.md): 70M/410M learning-rate and replication runs.

Only an upstream decision recorded in `decisions.md` may materialize a
dependent case group. A stage may stop under its predeclared null/failure rule;
downstream groups then remain unmaterialized rather than being replaced.
