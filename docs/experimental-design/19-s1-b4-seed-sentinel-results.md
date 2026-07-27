# S1-B4 Seed-1 Sentinel Results and S1 Closure

## Status and scope

S1-B4 closed on 2026-07-27. Configs `293--302` completed 2,048 optimizer
steps and 134,217,728 training tokens from clean launch commit
`2eae610b745237bfef2d0d816f861b4787fd3559`. The fail-stop serial queue
completed 10/10 runs with no failure in 3 h 40 min 09 s.

Diagnostic `303`, run `001-20260727-190455-ba5f3286`, evaluated all ten
canonical checkpoints over 152 complete sequences, 38 batches, and 311,296
frozen selection tokens. Its launch commit is
`b78fdcfd286e320645235c735b1d4c5ae371b702`; the
`activation_propagation.json` SHA-256 is
`2f2c24b31f953f58e1406aa2013992bff5526713f848ccc0c5da9e8c6393b4d5`.

Each sentinel changes model-initialization and data-order seeds together from
`0/0` to `1/1` relative to one exact seed-0 source. All other scientific
fields are fixed. The required seed-1 schedule hash is
`e3a2079b78a7816ae995c4289aa5946f28677ce50861b346605d42ca167e23a9`.

This is a two-point sensitivity check, not confirmation. It cannot separate
initialization from data-order effects or estimate population variance,
standard error, or a confidence interval.

## Training evidence

All rows use random-initialized Pythia-14M, model LR `3e-5`, 100 warmup steps,
sequence length 2,048, micro-batch 4, and accumulation 8.

| Config | Architecture / gate | Method and parameters | Canonical run | Train loss | Val. loss | Wall (h) | Tokens/s | Peak GPU (MiB) |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 293 | A0 | AdamW | `001-20260727-103125-a502cb9d` | 7.060050 | 7.058010 | 0.283 | 131,607 | 5,996 |
| 294 | A1-H | AdamW | `001-20260727-104830-b2c79809` | 7.014004 | 7.009003 | 0.281 | 132,789 | 5,948 |
| 295 | A3 | AdamW | `001-20260727-110526-adbfda90` | 7.029867 | 7.025085 | 0.294 | 126,810 | 5,996 |
| 296 | A6-PRE | AdamW | `001-20260727-112309-21d851fb` | 7.040618 | 7.034675 | 0.332 | 112,434 | 6,032 |
| 297 | A6-POST | AdamW | `001-20260727-114307-3f6d5070` | 7.058127 | 7.052760 | 0.330 | 112,940 | 6,056 |
| 298 | A6-POST, fixed G+ POST-QKV | AdamW, `kappa=.10` | `001-20260727-120300-10c2d667` | 7.058067 | 7.052987 | 0.345 | 108,163 | 6,074 |
| 299 | A6-POST, fixed Gpm POST-QKV | AdamW, `kappa=.10` | `001-20260727-122346-50b420c0` | 7.023873 | 7.023411 | 0.345 | 108,018 | 6,074 |
| 300 | A6-POST, learned Gpm POST-QKV ABS PLS | AdamW, `kappa_init=.10`, `tau=.03`, TLRM 10 | `001-20260727-124434-2205f2c0` | 7.023800 | 7.022480 | 0.349 | 106,882 | 6,273 |
| 301 | A6-POST ordinary ReLU | L1N, weight 1, all active gates | `001-20260727-130534-9c33bb64` | 7.080377 | 7.070321 | 0.597 | 62,399 | 6,296 |
| 302 | A6-POST ordinary ReLU | OR, `(w,c,sigma)=(.3,.1,.1)`, budget .5 | `001-20260727-134130-86a95dc4` | 7.063389 | 7.058295 | 0.500 | 74,635 | 6,987 |

The summed training wall time is 3.655 h. Dense PyTorch tokens/s reflects
method and instrumentation overhead; it is not a sparse-kernel speed result.

## Complete-selection endpoint handoff

Exact zero means direct numeric equality `x == 0`, with no tolerance. Counts
are integer-pooled over all complete selection examples, tokens, six layers,
and applicable tensor coordinates. `z_q`, `z_k`, and `z_v` are explicit
gate-output zero rates. An em dash means that the gate is absent.

`R_block` and `R_model` are logical scalar-product opportunity fractions, not
measured FLOP or wall-clock savings. `U_arch` divides the observed fraction by
the maximum allowed by that row's gate topology.

| Method | Config | Val. loss | `R_block` | `R_model` | `U_arch` | `z_a` | `z_m` | `z_h` | `z_q` | `z_k` | `z_v` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW / A0 | 293 | 7.058010 | <0.0001% | <0.0001% | -- | -- | -- | -- | -- | -- | -- |
| AdamW / A1-H | 294 | 7.009003 | 6.9653% | 2.0863% | 48.7704% | -- | -- | 48.7704% | -- | -- | -- |
| AdamW / A3 | 295 | 7.025085 | 19.2260% | 5.7586% | 48.9526% | 50.8583% | 50.9109% | 45.5650% | -- | -- | -- |
| AdamW / A6-PRE | 296 | 7.034675 | 41.1285% | 12.3190% | 41.1285% | 50.8771% | 51.0073% | 45.8418% | 18.2067% | 27.5220% | 44.4643% |
| AdamW / A6-POST | 297 | 7.052760 | 46.2464% | 13.8519% | 46.2464% | 50.9002% | 50.9768% | 45.8316% | 24.3058% | 35.7392% | 44.3114% |
| AdamW / fixed G+ POST-QKV, `kappa=.10` | 298 | 7.052987 | 56.9606% | 17.0610% | 56.9606% | 50.7134% | 50.8284% | 46.2639% | 38.4049% | 51.4421% | 66.7622% |
| AdamW / fixed Gpm POST-QKV, `kappa=.10` | 299 | 7.023411 | 39.9648% | 11.9704% | 39.9648% | 50.6236% | 50.7410% | 45.6930% | 8.8618% | 17.1225% | 48.8398% |
| AdamW / learned Gpm POST-QKV ABS PLS | 300 | 7.022480 | 44.1557% | 13.2257% | 44.1557% | 50.6223% | 50.7620% | 45.7339% | 13.3089% | 23.3329% | 55.1003% |
| L1N / A6-POST, weight 1 | 301 | 7.070321 | 66.0220% | 19.7752% | 66.0220% | 42.9649% | 34.7610% | 41.1071% | 89.7075% | 89.6969% | 81.1911% |
| OR / A6-POST, `(.3,.1,.1)` | 302 | 7.058295 | 58.6300% | 17.5611% | 58.6300% | 50.9730% | 51.0257% | 46.0041% | 83.8539% | 83.4854% | 45.3855% |

Every row uses 311,296 validation tokens and has `valid` evidence status.

## Paired seed sensitivity

`Delta_seed` is seed 1 minus seed 0. Differences in `R_model` and `U_arch` are
percentage points and are computed from unrounded fractions.

| Design | S0 / S1 configs | Loss S0 | Loss S1 | `Delta_seed` loss | `R_model` S0 | `R_model` S1 | `Delta_seed R_model` | `U_arch` S0 | `U_arch` S1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 AdamW | 123 / 293 | 7.049132 | 7.058010 | +0.008879 | <0.0001% | <0.0001% | <0.0001 pp | -- | -- |
| A1-H AdamW | 124 / 294 | 6.988746 | 7.009003 | +0.020258 | 2.0765% | 2.0863% | +0.0097 pp | 48.5426% | 48.7704% |
| A3 AdamW | 125 / 295 | 7.013099 | 7.025085 | +0.011986 | 5.7972% | 5.7586% | -0.0386 pp | 49.2804% | 48.9526% |
| A6-PRE AdamW | 126 / 296 | 7.016448 | 7.034675 | +0.018227 | 12.2187% | 12.3190% | +0.1003 pp | 40.7937% | 41.1285% |
| A6-POST AdamW | 127 / 297 | 7.032478 | 7.052760 | +0.020282 | 13.6425% | 13.8519% | +0.2094 pp | 45.5472% | 46.2464% |
| Fixed G+ POST-QKV | 169 / 298 | 7.034079 | 7.052987 | +0.018908 | 16.9122% | 17.0610% | +0.1489 pp | 56.4635% | 56.9606% |
| Fixed Gpm POST-QKV | 193 / 299 | 7.016880 | 7.023411 | +0.006530 | 11.8750% | 11.9704% | +0.0954 pp | 39.6462% | 39.9648% |
| Learned Gpm POST-QKV | 228 / 300 | 7.016171 | 7.022480 | +0.006309 | 13.0874% | 13.2257% | +0.1383 pp | 43.6941% | 44.1557% |
| A6-POST L1N, weight 1 | 250 / 301 | 7.053119 | 7.070321 | +0.017202 | 19.5970% | 19.7752% | +0.1781 pp | 65.4273% | 66.0220% |
| A6-POST OR, `(.3,.1,.1)` | 255 / 302 | 7.039297 | 7.058295 | +0.018998 | 17.5583% | 17.5611% | +0.0028 pp | 58.6207% | 58.6300% |

All ten seed-1 losses are higher by 0.006309--0.020282. Absolute
`R_model` shifts are at most 0.2094 pp; non-A0 `U_arch` shifts are at most
0.6992 pp. Loss and compute endpoints have different units and scales, so
these ranges do not establish that one endpoint is relatively less
seed-sensitive or imply a general variance ratio.

## Within-seed contrast replication

Each effect is candidate minus its matched parent, computed separately at each
seed. A6-versus-A3 rows are composite because B4 does not repeat the direct
A5 parent.

| Contrast | Loss effect S0 | Loss effect S1 | Change in effect | `R_model` effect S0 | `R_model` effect S1 | Change in effect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1-H - A0 | -0.060386 | -0.049007 | +0.011379 | +2.0765 pp | +2.0863 pp | +0.0097 pp |
| A3 - A1-H | +0.024353 | +0.016081 | -0.008272 | +3.7207 pp | +3.6724 pp | -0.0483 pp |
| A6-PRE - A3 (composite) | +0.003349 | +0.009591 | +0.006242 | +6.4215 pp | +6.5603 pp | +0.1389 pp |
| A6-POST - A3 (composite) | +0.019379 | +0.027676 | +0.008296 | +7.8452 pp | +8.0932 pp | +0.2480 pp |
| A6-POST - A6-PRE | +0.016030 | +0.018085 | +0.002055 | +1.4238 pp | +1.5329 pp | +0.1091 pp |
| Fixed G+ `kappa=.10` - A6-POST | +0.001601 | +0.000227 | -0.001374 | +3.2697 pp | +3.2092 pp | -0.0606 pp |
| Fixed Gpm `kappa=.10` - A3 | +0.003781 | -0.001674 | -0.005455 | +6.0778 pp | +6.2118 pp | +0.1340 pp |
| Learned - fixed Gpm | -0.000709 | -0.000931 | -0.000221 | +1.2124 pp | +1.2553 pp | +0.0428 pp |
| L1N weight 1 - A6-POST AdamW | +0.020640 | +0.017561 | -0.003080 | +5.9546 pp | +5.9233 pp | -0.0313 pp |
| OR central - A6-POST AdamW | +0.006819 | +0.005534 | -0.001284 | +3.9158 pp | +3.7092 pp | -0.2067 pp |

Every listed `R_model` effect keeps its sign. Nine of ten loss effects also
keep their sign. The only loss sign reversal is fixed Gpm versus A3:
`+0.003781` at seed 0 and `-0.001674` at seed 1. Its compute effect remains
positive at both seeds.

Learned Gpm retains a small lower-loss, higher-`R_model` effect versus fixed
Gpm at both seeds. L1N and OR both retain higher `R_model` and higher loss than
A6-POST AdamW. B4 contains no seed-1 RN or OL1 row, so it cannot test the seed
sensitivity of OR-RN or OL1-L1N orthogonalization contrasts.

Most site-zero shifts are small. The clear redistribution exception is L1N:
from seed 0 to seed 1, `z_a` and `z_m` rise by 4.9621 and 4.1640 pp while
`z_h` falls by 1.4911 pp; its Q/K/V gate rates move by less than 1 pp. OR is
more stable, with every displayed site moving by less than 1 pp.

## Safety and mechanism checks

- All ten runs pass nonfinite, universal-collapse, loss-instability,
  sparsity-evaporation, and step-budget-violation checks.
- Config `300` reconstructs all 18 per-layer/site threshold parameters. It has
  no nonfinite or frozen threshold steps, and its last three logged events have
  nonzero gradient, `rho`, and `kappa` motion. Final `kappa` spans
  0.104702--0.193033, with mean 0.139657.
- Config `302` reaches the OR `step_budget=.5` cap on all logged pressure
  updates (`orthogonal_cap_binding_fraction=1.0`) without violating the cap.
  This is a non-invalidating stability trigger, not evidence for retuning the
  primary budget.

## Closure and next gate

The applied registries give an executable S1 census of 132/132: B0 20/20,
B1 36/36, B2 26/26, B3 40/40, and B4 10/10. A dated local closure audit
reported all 36 mandatory scientific diagnostics `closed_valid` and zero
repository-integrity errors. Registry closure is commit
`5ddc0ca1e650f90c1035e14a3b7d6f69384d0205`, and the next unused config
prefix is `304`.

No ranking or promotion was performed during B4 closure. The next action is a
registered review of the predeclared conditional-control triggers. In
particular, `C4-BUDGET` is triggered by cap binding in B3 and reinforced by
config `302`; it remains a stability control rather than a tuning axis. Do not
launch a conditional cell or freeze an S2 panel until that review is recorded.
