# Phase C — Pythia-70M and Pythia-410M Replication

> Exact physical case groups and seed additions are in
> [`../cases.yaml`](../cases.yaml). No intervention hyperparameter is retuned by
> model size.

## C1 Learning Rate by Size

**Question:** Which peak learning rate should each larger model use?

**Cases:** `C1-lr-screens`; each case uses `lr-400m` and seed 0.

**Decision rule:** independently for each size, wait until all three LR cells
have reviewed terminal classifications, then select the eligible case with the
lowest final selection loss; an exact tie favors the lower LR.

**Stop rule:** if a size has no eligible LR, stop that size's C2/C3 branch. Do
not borrow another model's LR or make a complete cross-scale persistence claim.

**Produces:** decisions `lr_70m` and `lr_410m`.

## C2 Spillover Replication

**Question:** Does the predeclared L1 spillover response recur at 70M and 410M?

**Prerequisites:** the relevant per-model LR and the completed A2 design.

**Cases:** `C2-dense-controls`, `C2-relu-controls`, `C2-l1-screens`,
`C2-relu-confirmation`, and `C2-spillover-confirmation`. C1's `A0` cases
cannot be reused because their budget is `lr-400m`; C2 uses `full-pass-wrap`.
Each zero-pressure response anchor has `pressure: none`; only positive lambdas
use `pressure: l1_naive`.

**Evidence:** show the complete lambda response as exploratory seed-0 evidence.
The same fixed ReLU-only versus L1 contrast used in A2 receives seeds 0, 1, and
2 at both larger sizes. Absolute lambda values are transported unchanged.

**Completion rule:** a full response requires every seed-0 grid cell to be
resolved. A cross-size central contrast requires equal valid seed sets at all
three sizes; missing/unfavorable seeds are not replaced.

**Produces:** the matched site/layer spillover vector for 14M, 70M, and 410M.

## C3 Frontier Replication

**Question:** Does the selected B2 frontier persist across scale without
retuning the intervention?

**Prerequisites:** C2, `b1_family`, `lambda_B2`, `b2_frontier`, `b2_winner`,
`ol1_step_budget`, and the per-model LRs.

**Cases:** `C3-ol1-controls` creates one OL1-only case per larger model at
seed 0. `C3-frontier-replication` transports the B2 topology, attention form,
kappa cohort, lambda, and step budget literally; only threshold-only and
threshold+OL1 vary by kappa. `C3-winner-confirmation` adds seeds 1 and 2 to the
same six-component winner comparison. Matching C2 `A0`, ReLU-only, and
L1-only cases are consumers, not new training runs.

**Evidence:** report the complete transported frontier as exploratory seed-0
evidence. Only the selected six-component comparison has `n = 3`. An absolute
kappa or lambda may realize a different effect at a larger size; that is the
replication question and must not be tuned away.

**Completion rule:** claim frontier persistence only if the complete frozen
cohort and required matched controls are valid at all three sizes. Otherwise
report the exact partial or failed replication.

**Produces:** the cross-size validation-loss/`R_model` comparison and final
component table.
