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
| Formal A1 review | [`a1-review-packet.md`](a1-review-packet.md) |
| Shared scientific configuration | [`protocol.md`](protocol.md) |
| Stage design or selection | [Phase A](phases/a-pressure.md), [B](phases/b-threshold.md), or [C](phases/c-scale.md), plus [`cases.yaml`](cases.yaml) |
| Config allocation or reuse | [`cases.yaml`](cases.yaml), [`run-reuse.md`](run-reuse.md), and [`../../experiments/README.md`](../../experiments/README.md) |
| Diagnostics or paper output | [`outputs.md`](outputs.md) and [`../diagnostics.md`](../diagnostics.md) |
| Implementation work | [`workboard.md`](workboard.md) and [`../code_map.md`](../code_map.md) |
| Launch, recovery, or ETC | [`../runbook.md`](../runbook.md) |
| Record a selected value | [`decisions.md`](decisions.md) |

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
