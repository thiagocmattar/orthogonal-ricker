# S1-B3 T4 Ricker Basin Results

## Status and measurement contract

S1-B3 tranche `t4-rk-basin` closed on 2026-07-23 with canonical diagnostic
run `001-20260723-101136-b0907704`. Configs `275--282` completed 2,048
optimizer steps and 134,217,728 training tokens from clean launch commit
`360686d4e60dafe410640161211d3011ad53dd20`. Diagnostic `283` evaluated all
eight checkpoints over 152 complete sequences, 38 batches, and 311,296 frozen
selection tokens.

Exact zero means direct numeric equality to zero with no tolerance. Counts are
pooled over all tokens and six layers. `R_block` and `R_model` are logical
scalar-product fractions, not measured kernel speedups.

## Complete-selection endpoints

| Config | Architecture | Method | `c = sigma` | Validation loss | `R_block` | `R_model` | `z_a` | `z_m` | `z_h` | `z_Q^g` | `z_K^g` | `z_V^g` |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 275 | A3 | RN | .05 | 7.120422 | 27.9144% | 8.3610% | 70.7436% | 70.7162% | 71.6817% | -- | -- | -- |
| 276 | A3 | OR | .05 | 7.033385 | 26.1906% | 7.8447% | 66.2172% | 65.6303% | 68.0922% | -- | -- | -- |
| 277 | A6-POST | RN | .05 | 7.109157 | 77.5721% | 23.2347% | 63.4899% | 63.1705% | 66.4910% | 81.1500% | 80.6330% | 83.6357% |
| 278 | A6-POST | OR | .05 | 7.035138 | 56.8176% | 17.0182% | 51.0637% | 51.1342% | 45.9905% | 74.9154% | 74.6553% | 44.2205% |
| 279 | A3 | RN | .5 | 7.079639 | 24.4541% | 7.3246% | 71.0138% | 70.1388% | 47.8272% | -- | -- | -- |
| 280 | A3 | OR | .5 | 7.055496 | 24.4623% | 7.3270% | 70.5935% | 69.4270% | 48.9117% | -- | -- | -- |
| 281 | A6-POST | RN | .5 | 7.071631 | 70.1420% | 21.0092% | 66.0218% | 64.8307% | 45.2386% | 89.4950% | 89.4834% | 68.8280% |
| 282 | A6-POST | OR | .5 | 7.040369 | 60.4289% | 18.0999% | 51.8254% | 51.8071% | 46.3233% | 89.6876% | 89.6384% | 47.7067% |

## Orthogonality and basin response

OR minus matched RN; sparsity differences are percentage points:

| Architecture | `c = sigma` | Delta validation loss | Delta `R_block` | Delta `R_model` |
| --- | ---: | ---: | ---: | ---: |
| A3 | .05 | -0.087037 | -1.7238 pp | -0.5163 pp |
| A6-POST | .05 | -0.074018 | -20.7545 pp | -6.2165 pp |
| A3 | .5 | -0.024143 | +0.0082 pp | +0.0025 pp |
| A6-POST | .5 | -0.031262 | -9.7130 pp | -2.9093 pp |

Large basin minus small basin:

| Architecture / method | Delta validation loss | Delta `R_block` | Delta `R_model` |
| --- | ---: | ---: | ---: |
| A3 RN | -0.040784 | -3.4604 pp | -1.0365 pp |
| A3 OR | +0.022111 | -1.7283 pp | -0.5177 pp |
| A6-POST RN | -0.037526 | -7.4301 pp | -2.2255 pp |
| A6-POST OR | +0.005230 | +3.6113 pp | +1.0817 pp |

The wider basin improves RN validation loss but reduces its compute
opportunity, mainly through lower MLP-hidden and, on A6-POST, value-gate zero
rates. OR protects quality at every matched point. Its sparsity cost is largest
for narrow-basin A6-POST, while A3 at `.5` retains essentially the same
`R_model` as RN. No common basin dominates across method and architecture.

All eight runs passed nonfinite, step-budget-violation, universal-collapse,
loss-instability, and sparsity-evaporation checks. The non-invalidating
`C4-BUDGET` control triggered. OR cap binding was 26/42 updates for A3 `.05`,
42/42 for A6-POST `.05`, 11/42 for A3 `.5`, and 42/42 for A6-POST `.5`.
Consequently, the two A6-POST comparisons describe budget-constrained OR, not
an unconstrained geometry response.

T4 changes `c` and `sigma` together and therefore cannot identify which
parameter drives the hidden/value redistribution. The prespecified next
tranche, `t5-rk-shape`, holds `c=.1` fixed and varies `sigma` over `{.05,.2}`.
This closure performs no ranking, winner selection, or promotion.

Evidence:

- diagnostic artifact:
  `results/283-s1-b3-t4-rk-basin-selection-propagation/001-20260723-101136-b0907704/activation_propagation.json`;
- artifact SHA-256:
  `e7eedc1ee14d36c9006f111be5bb4beeef14378faac083c55c608bdc96206296`;
- diagnostic launch commit: `7eb84a99cf9d034527df0e7534cba1c7b50aaa34`.
