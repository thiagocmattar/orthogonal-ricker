# Phase A — Pythia-14M Pressure Study

> Exact physical case groups and seeds are in
> [`../cases.yaml`](../cases.yaml). Shared settings are in
> [`../protocol.md`](../protocol.md).

A1 uses `lr-400m`: 400,031,744 input tokens and 1,526 optimizer updates. A2 and
A3 retain `full-pass-wrap`.

## A1 Learning Rate

**Question:** Which peak learning rate should every Pythia-14M comparison use?

**Cases:** `A1-lr-screen`.

**Decision rule:** after all three cases have a reviewed terminal
classification, select the eligible case with the lowest final selection loss;
an exact tie favors the lower LR. If no LR is eligible, stop A2, A3, B1, and
B2 for review.

**Produces:** decision `lr_14m`. Report the complete tuning table and make no
sparsity claim from A1. The decision selects the best tested LR for Pythia-14M
at this fixed horizon; it does not establish a horizon-independent optimum.

## A2 L1 Spillover

**Question:** As h-only L1 pressure increases, how do the targeted `h` and
untargeted `a`, `m`, `q_post`, `k_post`, and `v` distributions change?

**Prerequisite:** `lr_14m`.

**Cases:** `A2-relu-control`, `A2-l1-screen`, `A2-relu-confirmation`, and
`A2-spillover-confirmation`. The full lambda response is a seed-0 screen. The
zero-pressure anchor has `pressure: none`; only positive lambdas use
`pressure: l1_naive`. The fixed central contrast—ReLU-only versus L1 at the
median nonzero grid value—uses seeds 0, 1, and 2 independently of Phase B.

**Primary comparison:** within each seed, subtract the matched ReLU-only
condition from the L1 condition at each named site. Report validation loss,
exact-zero mass, near-zero mass, RMS, and the predeclared distributions in
[`../outputs.md`](../outputs.md).

**Completion rule:** the full seed-0 curve requires every screen cell to have a
terminal classification. The central contrast is confirmed only with equal
valid seed sets; missing or failed seeds are never replaced.

**Produces:** the 14M spillover response and the predeclared three-seed central
contrast. B1 remains predeclared and may proceed after a null A2 result, but its
interpretation becomes a general multi-site threshold study.

## A3 OL1 Robustness

**Question:** At matched pressure weights, how do L1 and OL1 differ in final
loss versus achieved `n_h(0.01)`, and what conflict/projection behavior
accompanies the difference?

**Prerequisites:** `lr_14m` and reviewed decision `ol1_step_budget`.

**Cases:** `A3-ol1-screen` contains only positive-lambda OL1 cases. Its
zero-pressure curve anchor reuses `A2-relu-control` rather than encoding an
OL1 case with lambda zero or training again.

**Required evidence:** matched L1/OL1 final loss and `n_h(0.01)` plus
all-optimizer-boundary conflict, projection, eligibility, and trust-budget
counters. Complete lambda curves are exploratory seed-0 evidence.

**Decision rule:** consider only nonzero lambdas with valid matched A2-L1 and
A3-OL1 cases. Form the OL1 Pareto set minimizing final selection loss and
maximizing `n_h(0.01)`. Choose the point with lowest loss, then higher
`n_h(0.01)`, then lower lambda.

**Stop rule:** if no matched nonzero point is valid, do not define
`lambda_B2`; stop B2 and C3.

**Produces:** decision `lambda_B2` and the OL1 mechanism summary.
