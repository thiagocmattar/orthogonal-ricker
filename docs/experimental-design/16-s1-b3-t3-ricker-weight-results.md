# S1-B3 T3 Ricker Weight Results

## Status and measurement contract

S1-B3 tranche `t3-rk-weight` closed on 2026-07-22 with canonical diagnostic
run `001-20260722-125811-41dc3d92`. Configs `266--273` completed 2,048
optimizer steps and 134,217,728 training tokens from clean launch commit
`1f8ea98060bb08fc05589ffb6ea908e86b4143c3`. Diagnostic `274` evaluated all
eight checkpoints over 152 complete sequences, 38 batches, and 311,296 frozen
selection tokens.

Exact zero means direct numeric equality to zero with no tolerance. Counts are
pooled over all tokens and six layers. `R_block` and `R_model` are logical
scalar-product fractions, not measured kernel speedups.

## Complete-selection endpoints

The central `w=.3` rows from T1 are included to expose the full Ricker weight
ladder at fixed `(c,sigma)=(.1,.1)`.

| Config | Architecture | Method | Validation loss | `R_block` | `R_model` | `z_a` | `z_m` | `z_h` | `z_Q^g` | `z_K^g` | `z_V^g` |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 266 | A3 | RN, `w=.1` | 7.036829 | 24.1342% | 7.2288% | 61.0695% | 60.8057% | 62.3785% | -- | -- | -- |
| 267 | A3 | OR, `w=.1` | 7.031206 | 24.3185% | 7.2840% | 61.4057% | 60.9340% | 63.2889% | -- | -- | -- |
| 252 | A3 | RN, `w=.3` | 7.100854 | 27.2427% | 8.1598% | 68.4789% | 68.3487% | 71.0439% | -- | -- | -- |
| 253 | A3 | OR, `w=.3` | 7.046164 | 26.2739% | 7.8697% | 66.2891% | 65.7635% | 68.4886% | -- | -- | -- |
| 270 | A3 | RN, `w=1` | 7.201225 | 29.2868% | 8.7721% | 71.9950% | 72.1313% | 78.9373% | -- | -- | -- |
| 271 | A3 | OR, `w=1` | 7.058829 | 27.5472% | 8.2511% | 69.4915% | 69.2104% | 71.5554% | -- | -- | -- |
| 268 | A6-POST | RN, `w=.1` | 7.058684 | 73.0573% | 21.8824% | 56.8154% | 56.4108% | 55.2706% | 85.9614% | 85.0215% | 82.3470% |
| 269 | A6-POST | OR, `w=.1` | 7.039259 | 58.5844% | 17.5474% | 51.0905% | 51.1496% | 45.9813% | 83.7301% | 82.4395% | 44.8617% |
| 254 | A6-POST | RN, `w=.3` | 7.096174 | 79.0287% | 23.6710% | 63.6988% | 63.1609% | 64.7599% | 85.5484% | 84.8322% | 89.5112% |
| 255 | A6-POST | OR, `w=.3` | 7.039297 | 58.6207% | 17.5583% | 51.0989% | 51.1503% | 45.9562% | 83.7126% | 82.5401% | 44.9094% |
| 272 | A6-POST | RN, `w=1` | 7.162485 | 83.1256% | 24.8981% | 68.0290% | 67.6695% | 73.5885% | 85.1788% | 84.6247% | 92.2115% |
| 273 | A6-POST | OR, `w=1` | 7.039091 | 58.6154% | 17.5567% | 51.0903% | 51.1464% | 46.0160% | 83.8215% | 82.4982% | 44.8812% |

## Weight response and matched orthogonality effect

OR minus RN at each weight:

| Architecture | Weight | Delta validation loss | Delta `R_block` | Delta `R_model` |
| --- | ---: | ---: | ---: | ---: |
| A3 | .1 | -0.005623 | +0.1843 pp | +0.0552 pp |
| A3 | .3 | -0.054690 | -0.9688 pp | -0.2901 pp |
| A3 | 1 | -0.142396 | -1.7396 pp | -0.5210 pp |
| A6-POST | .1 | -0.019425 | -14.4729 pp | -4.3350 pp |
| A6-POST | .3 | -0.056877 | -20.4080 pp | -6.1127 pp |
| A6-POST | 1 | -0.123394 | -24.5102 pp | -7.3414 pp |

From `w=.1` to `w=1`, RN gains 1.5433 `R_model` points on A3 and 3.0157
points on A6-POST, while validation loss worsens by 0.1644 and 0.1038. OR
limits those loss increases to 0.0276 on A3 and effectively zero on A6-POST,
but its sparsity gains are only 0.9671 and 0.0093 points. The A6-POST OR
result is weight-insensitive because the 0.5 step-budget cap binds on every
logged update at all three weights.

Thus weight is an effective sparsity control for RN but has an increasingly
poor quality tradeoff. Orthogonalization protects quality, especially for the
six-gate architecture, while also limiting the realizable sparsity response.
The prespecified T4 basin test remains necessary to separate Ricker-shape
effects from weight and cap saturation; this single-seed screen does not select
a winner.

All eight T3 runs passed nonfinite, step-budget-violation, universal-collapse,
loss-instability, and sparsity-evaporation checks. The non-invalidating
`C4-BUDGET` control triggered.

Evidence:

- T3 diagnostic artifact:
  `results/274-s1-b3-t3-rk-weight-selection-propagation/001-20260722-125811-41dc3d92/activation_propagation.json`;
- T3 artifact SHA-256:
  `6808617feff987af16a1f4ca7d8bd3be38f4127dd5db0d2cc6ded0c7afcf835f`;
- T3 diagnostic launch commit: `b81f2cc24ebfa282d693a1171110bf7ded903578`;
- T1 central diagnostic artifact:
  `results/256-s1-b3-t1-central-selection-propagation/001-20260720-122359-6fe7e068/activation_propagation.json`;
- T1 artifact SHA-256:
  `7555228761c26ae8ac5ea0bab3d17001b3a3cfc40f43bdc0737443a6594ec574`;
- T1 diagnostic launch commit: `ee6c61284bceb7bb2eaa7132cd7842597893694e`.
