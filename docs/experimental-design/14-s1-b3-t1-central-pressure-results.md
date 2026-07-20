# S1-B3 T1 Central Pressure Results

## Status and measurement contract

S1-B3 tranche `t1-central` closed on 2026-07-20 with canonical diagnostic run
`001-20260720-122359-6fe7e068`. Configs `248--255` completed 2,048 optimizer
steps and 134,217,728 training tokens from one clean random-initialization
commit. Diagnostic `256` evaluated every checkpoint over the frozen selection
partition: 152 complete sequences, 38 batches, and 311,296 model-input tokens.

Exact zero means direct numeric equality to zero with no tolerance. Counts are
pooled over all evaluated tokens and all six layers rather than averaging batch
percentages. `R_block` and `R_model` are logical-product fractions, not measured
kernel speedups.

## Complete-selection endpoints

| Config | Architecture | Method | Validation loss | `R_block` | `R_model` | `z_a` | `z_m` | `z_h` | `z_Q^g` | `z_K^g` | `z_V^g` |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 248 | A3 | L1N, `w=1` | 7.053376 | 15.3315% | 4.5921% | 37.3285% | 32.5999% | 46.7538% | -- | -- | -- |
| 249 | A3 | OL1, `w=1` | 7.020024 | 15.6535% | 4.6886% | 40.9819% | 32.2215% | 46.6469% | -- | -- | -- |
| 250 | A6-POST | L1N, `w=1` | 7.053119 | 65.4273% | 19.5970% | 38.0028% | 30.5970% | 42.5982% | 89.7396% | 89.7407% | 82.1489% |
| 251 | A6-POST | OL1, `w=1` | 7.038947 | 59.0716% | 17.6933% | 51.0139% | 51.0446% | 46.0148% | 88.0619% | 88.0772% | 44.7550% |
| 252 | A3 | RN, `w=.3,c=.1,sigma=.1` | 7.100854 | 27.2427% | 8.1598% | 68.4789% | 68.3487% | 71.0439% | -- | -- | -- |
| 253 | A3 | OR, `w=.3,c=.1,sigma=.1` | 7.046164 | 26.2739% | 7.8697% | 66.2891% | 65.7635% | 68.4886% | -- | -- | -- |
| 254 | A6-POST | RN, `w=.3,c=.1,sigma=.1` | 7.096174 | 79.0287% | 23.6710% | 63.6988% | 63.1609% | 64.7599% | 85.5484% | 84.8322% | 89.5112% |
| 255 | A6-POST | OR, `w=.3,c=.1,sigma=.1` | 7.039297 | 58.6207% | 17.5583% | 51.0989% | 51.1503% | 45.9562% | 83.7126% | 82.5401% | 44.9094% |

Every percentage above was independently recomputed from integer counts. Each
method has 36 matmul records. The common block-product denominator is
857,085,050,880 and the LM-head denominator is 2,004,407,549,952 products.

## Prespecified matched contrasts

Second method minus first:

| Match | Delta validation loss | Delta `R_block` | Delta `R_model` |
| --- | ---: | ---: | ---: |
| A3 OL1 minus L1N | -0.033352 | +0.3220 pp | +0.0965 pp |
| A6-POST OL1 minus L1N | -0.014172 | -6.3557 pp | -1.9037 pp |
| A3 OR minus RN | -0.054690 | -0.9687 pp | -0.2902 pp |
| A6-POST OR minus RN | -0.056877 | -20.4079 pp | -6.1127 pp |

The orthogonal methods lower validation loss relative to their matched naive
methods in all four T1 pairs. That does not imply greater sparsity: the largest
counterexample is A6-POST OR, whose `R_model` is 6.1127 percentage points below
matched RN, driven in part by `z_V^g` falling by 44.6018 points.

A6-POST exposes substantially more targetable products than A3. Relative to
the matched A3 row, its `R_model` gain is +15.0049 pp for L1N, +13.0047 pp for
OL1, +15.5111 pp for RN, and +9.6886 pp for OR.

## Screening and handoff

All eight runs passed nonfinite, universal-collapse, loss-instability,
sparsity-evaporation, and step-budget checks. The preregistered `C4-BUDGET`
stability control is triggered: the orthogonal cap bound on 52.38% of logged
updates for A3 OL1/OR and 100% for A6-POST OL1/OR. No logged final update ratio
exceeded the 0.5 budget.

These are single-seed screening results. The closure performs no winner
selection or promotion. After diagnostic `256`, executable-core completion is
90/132, S1-B3 is 8/40 complete, and `t2-l1-flanks` is the next eligible
tranche.

Evidence:

- diagnostic artifact:
  `results/256-s1-b3-t1-central-selection-propagation/001-20260720-122359-6fe7e068/activation_propagation.json`;
- artifact SHA-256:
  `7555228761c26ae8ac5ea0bab3d17001b3a3cfc40f43bdc0737443a6594ec574`;
- diagnostic launch commit: `ee6c61284bceb7bb2eaa7132cd7842597893694e`.
