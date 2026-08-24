# Frozen Decision Register

This register stores only values needed to materialize dependent case groups.
It does not copy result tables. Each result-backed decision must pin exact
configs/runs/diagnostics in [`../experiment_log.md`](../experiment_log.md).

| Decision ID | State | Value | Exact evidence pointer | Rule required before freezing |
| --- | --- | --- | --- | --- |
| `ol1_step_budget` | unresolved | `TODO:` | `TODO:` reviewed design record | Independent scientific review before A3; not selected by smoke calibration |
| `lr_14m` | unresolved | `TODO:` | `TODO:` selected config and accepted attempt | A1 rule and exact terminal case set |
| `lambda_B2` | unresolved | `TODO:` | `TODO:` experiment-log cohort | A3 matched L1/OL1 Pareto rule |
| `b1_family` | unresolved | `TODO:` | `TODO:` experiment-log cohort | B1 positive-kappa family rule |
| `b2_frontier` | unresolved | `TODO:` | `TODO:` exact case/attempt set | All valid nondominated B2 combined points |
| `b2_winner` | unresolved | `TODO:` | `TODO:` selected config and accepted attempt | Lowest-loss eligible point on `b2_frontier` |
| `lr_70m` | unresolved | `TODO:` | `TODO:` selected config and accepted attempt | C1 per-size LR rule |
| `lr_410m` | unresolved | `TODO:` | `TODO:` selected config and accepted attempt | C1 per-size LR rule |
| `confirmation_release` | unresolved | `TODO:` | `TODO:` frozen analysis record | All planned cohorts, labels, analysis, and exclusions frozen |

`b2_frontier` is a structured value containing the frozen family recipe and
ordered selected points, including `kappa_values`. `b2_winner` is a complete
recipe: topology, attention form, kappa, pressure method/sites, lambda, and
OL1 step budget. Downstream groups may dereference these fields but may not
retune them.

## Update Rule

1. Complete the upstream stage and required diagnostics.
2. Apply only its predeclared rule; do not substitute an informal preference.
3. Record the chosen value plus exact evidence identities in the experiment
   log, then point this row to them. A selected training source must identify
   `tranche_id`, `config_id`, `run_id`, and checkpoint; a scalar alone is not a
   reusable decision.
4. Reset `docs/experiment_plan.md` to `Plan status: placeholder`, commit the
   normative change, and request review for the newly unblocked groups.
5. Materialize configs only after those exact groups enter the reviewed scope.

A null/failed stage records `stopped` with its exact reason. It does not receive
a fallback value unless a new reviewed design revision changes the rule.
