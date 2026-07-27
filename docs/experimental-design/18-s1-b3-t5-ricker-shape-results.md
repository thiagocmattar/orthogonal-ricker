# S1-B3 T5 Ricker Shape Results

## Status and measurement contract

S1-B3 tranche `t5-rk-shape` closed on 2026-07-27 with canonical diagnostic
run `001-20260727-095931-6f265e69`. Configs `284--291` completed 2,048
optimizer steps and 134,217,728 training tokens from clean launch commit
`e918f755cd313a68d312e42728df152a3706870c`. Diagnostic `292` evaluated all
eight checkpoints over 152 complete sequences, 38 batches, and 311,296 frozen
selection tokens.

Exact zero means direct numeric equality to zero with no tolerance. Counts are
pooled over all tokens and six layers. `R_block` and `R_model` are logical
scalar-product fractions, not measured kernel speedups.

## Complete-selection endpoints

| Config | Architecture | Method | `(w,c,sigma)` | Validation loss | `R_block` | `R_model` | `z_a` | `z_m` | `z_h` | `z_Q^g` | `z_K^g` | `z_V^g` |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 284 | A3 | RN | `(.3,.1,.05)` | 7.101430 | 28.9629% | 8.6751% | 74.0103% | 73.7823% | 73.5070% | -- | -- | -- |
| 285 | A3 | OR | `(.3,.1,.05)` | 7.051758 | 27.5192% | 8.2427% | 69.8646% | 69.1338% | 71.1561% | -- | -- | -- |
| 286 | A6-POST | RN | `(.3,.1,.05)` | 7.094460 | 80.8997% | 24.2314% | 66.7653% | 66.0610% | 66.1507% | 87.6522% | 86.7244% | 91.4694% |
| 287 | A6-POST | OR | `(.3,.1,.05)` | 7.037077 | 58.0644% | 17.3917% | 51.0217% | 51.0778% | 45.9737% | 82.6560% | 84.3385% | 44.3856% |
| 288 | A3 | RN | `(.3,.1,.2)` | 7.125618 | 27.0002% | 8.0872% | 65.0974% | 65.0740% | 75.1570% | -- | -- | -- |
| 289 | A3 | OR | `(.3,.1,.2)` | 7.044072 | 25.3802% | 7.6020% | 61.9102% | 61.7819% | 69.4965% | -- | -- | -- |
| 290 | A6-POST | RN | `(.3,.1,.2)` | 7.112029 | 80.2346% | 24.0322% | 61.5481% | 61.3911% | 67.7233% | 88.7321% | 88.7463% | 93.6715% |
| 291 | A6-POST | OR | `(.3,.1,.2)` | 7.040086 | 59.5872% | 17.8478% | 51.1759% | 51.2312% | 46.1429% | 88.1140% | 88.0591% | 45.9837% |

## Matched shape response

Wider `sigma=.2` minus narrower `sigma=.05`; sparsity differences are
percentage points:

| Architecture / method | Delta validation loss | Delta `R_block` | Delta `R_model` |
| --- | ---: | ---: | ---: |
| A3 RN | +0.024187 | -1.9627 pp | -0.5879 pp |
| A3 OR | -0.007686 | -2.1390 pp | -0.6407 pp |
| A6-POST RN | +0.017568 | -0.6651 pp | -0.1992 pp |
| A6-POST OR | +0.003009 | +1.5227 pp | +0.4561 pp |

At A3, widening `sigma` lowers attention- and MLP-input zero rates for both
methods, and neither hidden response recovers the lost compute opportunity.
A6-POST RN shifts zeros toward the hidden and Q/K/V gates while its branch-input
and post-PV context zero rates fall; `R_model` therefore remains nearly
unchanged. A6-POST OR keeps branch and hidden zero rates nearly fixed and raises
Q/K/V zero rates, yielding a small `R_model` increase with nearly unchanged
loss. No common `sigma` dominates across method and architecture.

## Orthogonality and pressure-cap boundary

OR minus matched RN:

| Architecture | `sigma` | Delta validation loss | Delta `R_block` | Delta `R_model` |
| --- | ---: | ---: | ---: | ---: |
| A3 | .05 | -0.049672 | -1.4437 pp | -0.4324 pp |
| A6-POST | .05 | -0.057383 | -22.8352 pp | -6.8397 pp |
| A3 | .2 | -0.081546 | -1.6200 pp | -0.4852 pp |
| A6-POST | .2 | -0.071943 | -20.6474 pp | -6.1844 pp |

OR has lower validation loss at every matched point but substantially lower
A6-POST value/hidden sparsity. The non-invalidating `C4-BUDGET` control
triggered: the `0.5` cap bound 22/42 logged updates for config `285`, 42/42 for
`287`, 28/42 for `289`, and 42/42 for `291`; no final update-ratio violation
occurred. The A6-POST OR contrasts are therefore fully budget-constrained and
the A3 OR contrasts partially constrained. They do not identify an
unconstrained orthogonal-geometry response. The primary step budget remains
`0.5`; this closure does not tune it.

All eight runs passed nonfinite, step-budget-violation, universal-collapse,
loss-instability, and sparsity-evaporation checks. This remains a single-seed
2,048-step selection screen; closure performs no ranking, winner selection, or
promotion.

Evidence:

- diagnostic artifact:
  `results/292-s1-b3-t5-rk-shape-selection-propagation/001-20260727-095931-6f265e69/activation_propagation.json`;
- artifact SHA-256:
  `6d20614d188e59b0ed64351af1b8f643a411e726a20a86d4b4e086150cf03513`;
- diagnostic launch commit: `6e6f392276b4908000cd24ae196d80ab2dc322b5`.
