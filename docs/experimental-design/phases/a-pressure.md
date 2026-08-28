# Phase A — Pythia-14M Pressure Study

> Exact physical case groups and seeds are in
> [`../cases.yaml`](../cases.yaml). Shared settings are in
> [`../protocol.md`](../protocol.md).

A1 uses `lr-400m`: 400,031,744 input tokens and 1,526 optimizer updates. A2 and
A3 retain `full-pass-wrap`.

## A1 Learning Rate

**Question:** Which peak learning rate should every Pythia-14M comparison use?

**Cases:** `A1-lr-screen`.

**Decision rule:** after all eleven cases have a reviewed terminal
classification, select the eligible case with the lowest final selection loss;
an exact tie favors the lower LR. If no LR is eligible, stop A2, A3, B1, and
B2 for review.

**Produces:** decision `lr_14m`. Report the complete tuning table and make no
sparsity claim from A1. The decision selects the best tested LR for Pythia-14M
at this fixed horizon; it does not establish a horizon-independent optimum.
The final extension contains exactly three factor-two cells above `8e-3`:
`1.6e-2`, `3.2e-2`, and `6.4e-2`. Because validation loss still decreased at
that boundary, the user directed one further factor-two extension containing
exactly `1.28e-1`, `2.56e-1`, and `5.12e-1`. Reuse all eight completed cells;
do not rerun them. This eleven-cell design authorizes no further automatic
extension. If `5.12e-1` is selected, report that it remains the upper tested
boundary.

## A2 L1 Spillover

**Question:** As h-only L1 pressure increases, how do the targeted `h` and
untargeted `a`, `m`, `q_post`, `k_post`, and `v` distributions change?

**Prerequisite:** `lr_14m`.

**Cases:** `A2-relu-control` and `A2-l1-screen`. A2 is a seed-0 discovery
screen with one `pressure: none` anchor and five `pressure: l1_naive` points
at lambda `{0.1, 0.5, 1, 2, 5}`. The zero-pressure anchor is not encoded as
an L1 case with lambda zero.

**Primary comparison:** subtract the seed-0 ReLU-only condition from each L1
condition at every named site. Report validation loss, exact-zero mass,
and near-zero mass; use the already saved RMS and predeclared distributions as
supporting evidence in
[`../outputs.md`](../outputs.md). The response is directional single-seed
evidence and carries no seed-uncertainty claim.

**Completion rule:** all six seed-0 cells require a terminal classification.
Missing, infrastructure-failed, or unresolved cells block the A2 response;
scientific failures remain visible grid outcomes and are never replaced.

**Produces:** the 14M seed-0 spillover response. Full-grid added-seed
replication is not planned in the current program.
If the final B2 winner cohort is run, its independently initialized seed-1/2
ReLU-only and selected-L1 components may provide an added-seed contrast without
duplicating a physical condition, but only at the selected lambda.
B1 remains predeclared and may proceed after a null A2 result, but its
interpretation becomes a general multi-site threshold study.

### Proposed post-hoc clipping frontier

**Question:** Do the six trained A2 conditions expose different model-wide
quality–logical-opportunity tradeoffs when small operands are set exactly to
zero after training?

**Sources:** reuse only the accepted final checkpoints for configs `012`–`017`.
No training condition is added or rerun.

**Intervention:** jointly apply the existing absolute clipping rule
`abs(x) <= t -> 0` at `[a, m, h, q_post, k_post, v]`, in every layer, for
`t = {0, 0.01, 0.03, 0.10, 0.30}`. Use one common cutoff across sites, the
complete selection partition, and actual-operand logical-product measurement
at every point. The exact source identities, row contract, output, and claim
limits are frozen in
[`../a2-clipping-review-packet.md`](../a2-clipping-review-packet.md).

**Produces when reviewed and completed:** a 30-point seed-0 post-hoc frontier
reporting validation loss and observed `R_model`, plus within-checkpoint changes
relative to each sweep's `t = 0` row. This is descriptive model-wide
thresholdability evidence, not causal attribution to spillover and not measured
speedup.

## A3 OL1 Robustness

**Question:** At matched pressure weights, how do L1 and OL1 differ in final
loss versus achieved `n_h(0.01)`, and what conflict/projection behavior
accompanies the difference?

**Prerequisites:** `lr_14m` and reviewed decision `ol1_step_budget`.

**Cases:** `A3-ol1-screen` contains only positive-lambda OL1 cases at the
matched A2 grid `{0.1, 0.5, 1, 2, 5}`. Its zero-pressure curve anchor reuses
`A2-relu-control` rather than encoding an OL1 case with lambda zero or training
again.

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
