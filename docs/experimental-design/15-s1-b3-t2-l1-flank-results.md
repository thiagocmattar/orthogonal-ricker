# S1-B3 T2 L1 Flank Results

## Status and measurement contract

S1-B3 tranche `t2-l1-flanks` closed on 2026-07-21 with canonical diagnostic
run `001-20260721-111619-2544b63c`. Configs `257--264` completed 2,048
optimizer steps and 134,217,728 training tokens from one clean random-init
commit. Diagnostic `265` pooled exact-zero and logical-product counts over the
frozen selection partition: 152 complete sequences, 38 batches, and 311,296
model-input tokens.

Exact zero means direct numeric equality to zero with no tolerance. `R_block`
and `R_model` are logical-product fractions, not measured kernel speedups.

## Complete-selection endpoints

| Config | Architecture | Method | Validation loss | `R_block` | `R_model` | `z_a` | `z_m` | `z_h` | `z_Q^g` | `z_K^g` | `z_V^g` |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 257 | A3 | L1N, `w=.15` | 7.024206 | 16.9342% | 5.0722% | 43.2080% | 41.9932% | 44.1735% | -- | -- | -- |
| 258 | A3 | OL1, `w=.15` | 7.014213 | 17.1335% | 5.1319% | 42.8177% | 41.4209% | 46.4338% | -- | -- | -- |
| 259 | A6-POST | L1N, `w=.15` | 7.040482 | 62.6631% | 18.7691% | 46.9099% | 45.9854% | 45.4514% | 89.7265% | 89.7488% | 60.6292% |
| 260 | A6-POST | OL1, `w=.15` | 7.038983 | 59.0772% | 17.6950% | 50.9849% | 51.0209% | 46.0480% | 88.0539% | 88.0505% | 44.7691% |
| 261 | A3 | L1N, `w=5` | 7.175777 | 17.3019% | 5.1823% | 33.3981% | 28.6574% | 67.4412% | -- | -- | -- |
| 262 | A3 | OL1, `w=5` | 7.029094 | 15.7169% | 4.7076% | 46.1786% | 27.8205% | 47.5948% | -- | -- | -- |
| 263 | A6-POST | L1N, `w=5` | 7.095620 | 71.1768% | 21.3191% | 39.8045% | 26.3673% | 54.3075% | 89.6642% | 89.6670% | 96.0864% |
| 264 | A6-POST | OL1, `w=5` | 7.038695 | 59.0781% | 17.6953% | 51.0306% | 51.0663% | 46.0308% | 88.0733% | 88.0983% | 44.7496% |

## Matched contrasts and interpretation

Orthogonal minus naive at the same architecture and weight:

| Match | Delta validation loss | Delta `R_block` | Delta `R_model` |
| --- | ---: | ---: | ---: |
| A3, `w=.15` | -0.009993 | +0.1993 pp | +0.0597 pp |
| A6-POST, `w=.15` | -0.001499 | -3.5859 pp | -1.0741 pp |
| A3, `w=5` | -0.146683 | -1.5850 pp | -0.4747 pp |
| A6-POST, `w=5` | -0.056925 | -12.0987 pp | -3.6238 pp |

The high L1 weight damages the naive method's validation loss, especially on
A3. OL1 limits that damage, but the protection does not generally increase
logical-product sparsity. A6-POST OL1 is effectively unchanged between the two
weights (`R_model` 17.6950% versus 17.6953%), consistent with a binding
orthogonal update cap rather than a useful weight-response curve.

All eight runs passed nonfinite, step-budget-violation, universal-collapse,
loss-instability, and sparsity-evaporation checks. The preregistered
`C4-BUDGET` control triggered and is non-invalidating. These are single-seed
screening results; no promotion was performed by the closure.

Evidence:

- diagnostic artifact:
  `results/265-s1-b3-t2-l1-flanks-selection-propagation/001-20260721-111619-2544b63c/activation_propagation.json`;
- artifact SHA-256:
  `15e055362e25fc47263d4e62f55b8095bf05be6c140ef7d58dc33bd6ee4ca596`;
- diagnostic launch commit: `72804d044192ad0abad25708cd3e1889a4f315bd`.
