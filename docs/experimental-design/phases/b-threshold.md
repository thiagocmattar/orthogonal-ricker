# Phase B — Pythia-14M Threshold and Combined Frontier

> Exact topology/form/kappa cells and aliases are owned by
> [`../cases.yaml`](../cases.yaml). Shared threshold semantics are in
> [`../protocol.md`](../protocol.md) and [`../../methods.md`](../../methods.md).

All Phase B training uses the frozen `lr_14m`, `full-pass-wrap`, and seed 0 for
the complete screens.

## B1 Threshold Ablation

**Question:** How do placement, attention-threshold form, and threshold value
change the validation-loss/`R_model` frontier?

**Prerequisite:** `lr_14m`. A2 must have a terminal interpretation, but B1's
grid is not selected from A2. Describe it as spillover-motivated, never
spillover-guided.

**Cases:** `B1-threshold-screen`. The catalog expands the eight topology
variants into 56 conceptual cells represented by 50 physical cases. Six
symmetric-attention `kappa = 0` cells reuse the topology `A2`, one-sided,
`kappa = 0` case because their attention operations are identities.

**Completeness:** classify all 50 physical cases before selection. Every finite
case needs valid final loss and `R_model`; a terminal scientific collapse is a
resolved dominated cell, while unresolved infrastructure state blocks the
decision.

**Family selection:**

1. Form the global Pareto set over the reused `A0` control and unique valid B1
   cases, minimizing final selection loss and maximizing `R_model`.
2. A family is one topology plus one attention form; form is `N/A` for
   `A1-H` and topology `A2`.
3. Use `kappa = 0` only as a plotted baseline. Score every family on its three
   positive-kappa points so the shared symmetric identity cannot favor several
   families.
4. Select the family with the most positive-kappa global-frontier points. Break
   ties by larger `R_model` span, lower mean loss over those points, topology
   order in the catalog, then symmetric before one-sided.

**Stop rule:** if no positive-kappa threshold case is on the global frontier,
record a null B1 result and stop B2/C3.

**Produces:** decision `b1_family` and the complete exploratory threshold
frontier.

## B2 Selected Threshold Plus OL1

**Question:** Does adding OL1 to the selected B1 family improve its
validation-loss/logical-opportunity frontier relative to its components?

**Prerequisites:** `b1_family`, `lambda_B2`, and `ol1_step_budget`.

**Cases:** `B2-combined-screen` evaluates every catalogued kappa in the
selected family. `B2-winner-confirmation` adds seeds 1 and 2 only to the final
six-component comparison and reuses every matching seed-0/control case.

**Frontier rule:** after every combined kappa cell is resolved, form the valid
combined Pareto set. Freeze all nondominated points as `b2_frontier`. Select
`b2_winner` as its lowest-loss point with `R_model > 0`; ties favor higher
`R_model`, then lower kappa.

**Stop rule:** if no valid combined point has `R_model > 0`, report a null B2
result and stop C3.

**Component interpretation:** the six conditions are `A0`, ReLU-only,
L1-only at `lambda_B2`, OL1-only at `lambda_B2`, threshold-only, and
threshold+OL1. There is no threshold+L1 case, so B2 cannot compare OL1 with L1
inside the selected threshold topology.

If the winner contains a nonidentity attention threshold, describe it only as
a selected multi-site recipe **containing** attention thresholding. Without a
matched FFN-threshold+OL1 case, do not attribute the improvement specifically
to the attention component.

**Produces:** decisions `b2_frontier` and `b2_winner` plus the three-seed
component comparison.
