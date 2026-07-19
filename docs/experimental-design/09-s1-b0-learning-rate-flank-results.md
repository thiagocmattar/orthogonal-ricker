# S1-B0 Learning-Rate Flank Results

## 1. Status and Scope

The S1-B0 learning-rate set is complete. Configs `135--144` provide AdamW
flanks at `1e-5` and `1e-4` for the five central `3e-5` architecture controls
in configs `123--127`. All ten flank runs reached 2,048 optimizer steps from
clean launch commits, saved durable final checkpoints, and passed terminal
review. Config `145` then measured their complete-selection exact-zero and
logical-product endpoints.

This is one-seed, fixed-2,048-step evidence. It supports learning-rate
comparisons only within the same architecture. It does not support a global
architecture ranking, promotion cutoff, or paper-level conclusion.

## 2. Evidence Contract

All 15 runs use random initialization, AdamW without activation pressure,
model/data-order seeds `0/0`, 100 warmup steps, sequence length 2,048,
micro-batch 4, gradient accumulation 8, and 134,217,728 training tokens. They
share schedule hash
`db4fa092d7092d29edc3bf1e2005af69f4a92b8bf6e6d88cbb6a166e12be02fc`.
Only model LR changes within each architecture triplet.

Diagnostics `128` and `145` use schema v3 and the same frozen `selection`
partition. Each checkpoint is evaluated on all 152 complete sequences: 38
batches and 311,296 tokens, with 443 trailing cache tokens excluded. The
partition hash is
`ffc857a6f0771929dd75c93bc17729de98a692f3a175ac5742cc9d101ff4ea47`.

An exact zero is the numeric comparison `x == 0`, with no tolerance. Counts
are pooled over every evaluated sequence, token, layer, and applicable
feature/head coordinate; percentages are pooled count ratios rather than
averages of batch or layer rates. Future causal attention positions are
excluded. An em dash denotes an absent gate.

`R_block` is the directly counted exact-zero scalar-product fraction across
QKV, valid-causal QK/PV, Wo, W1, and W2. `R_model` keeps that numerator and
adds the dense LM head to the denominator. The maxima are topology ceilings,
and `U_arch = R_model / R_model_max`. These are logical opportunities, not
measured sparse-kernel speedups. All endpoint values below are percentages.

## 3. Flank Run Evidence

`Delta loss` is relative to the same architecture at LR `3e-5`.

| Config | Architecture | LR | Canonical run | Train loss | Selection loss | Delta loss | Wall (min) | Tokens/s | Peak GPU (MiB) |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 135 | A0 | `1e-5` | `001-20260718-210843-fd04a711` | 8.397521 | 8.351036 | +1.301904 | 16.61 | 134,667 | 5,996 |
| 136 | A1-H | `1e-5` | `001-20260718-212731-d350d541` | 8.430215 | 8.386519 | +1.397773 | 16.62 | 134,631 | 5,948 |
| 137 | A3 | `1e-5` | `001-20260718-214625-4c3a46fa` | 8.443986 | 8.400148 | +1.387049 | 18.29 | 122,281 | 5,996 |
| 138 | A6-PRE | `1e-5` | `001-20260718-220621-c6efb587` | 8.432227 | 8.387391 | +1.370943 | 20.77 | 107,717 | 6,032 |
| 139 | A6-POST | `1e-5` | `001-20260718-222908-79d618fa` | 8.432246 | 8.387435 | +1.354957 | 20.72 | 107,949 | 6,056 |
| 140 | A0 | `1e-4` | `001-20260718-225228-b8972e8d` | 5.865663 | 5.938874 | -1.110258 | 16.53 | 135,298 | 5,996 |
| 141 | A1-H | `1e-4` | `001-20260718-231051-25bf322c` | 5.796514 | 5.874738 | -1.114008 | 16.55 | 135,195 | 5,948 |
| 142 | A3 | `1e-4` | `001-20260718-232919-f14b6c84` | 5.842091 | 5.917684 | -1.095415 | 18.20 | 122,921 | 5,996 |
| 143 | A6-PRE | `1e-4` | `001-20260718-234920-d1bc5ad6` | 5.857260 | 5.935388 | -1.081060 | 20.77 | 107,686 | 6,032 |
| 144 | A6-POST | `1e-4` | `001-20260719-001147-1cf26b02` | 5.990906 | 6.063204 | -0.969275 | 20.72 | 107,936 | 6,056 |

The flank set consumed 3.096 serial GPU-hours. Dense throughput is stable
across LR within each topology; the slower A3/A6 paths remain implementation
overhead rather than realized sparse acceleration.

## 4. Within-Architecture LR Triplets

| Architecture | Configs (`1e-5/3e-5/1e-4`) | Loss `1e-5` | Delta vs `3e-5` | Loss `3e-5` | Loss `1e-4` | Delta vs `3e-5` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A0 | `135/123/140` | 8.351036 | +1.301904 | 7.049132 | 5.938874 | -1.110258 |
| A1-H | `136/124/141` | 8.386519 | +1.397773 | 6.988746 | 5.874738 | -1.114008 |
| A3 | `137/125/142` | 8.400148 | +1.387049 | 7.013099 | 5.917684 | -1.095415 |
| A6-PRE | `138/126/143` | 8.387391 | +1.370943 | 7.016448 | 5.935388 | -1.081060 |
| A6-POST | `139/127/144` | 8.387435 | +1.354957 | 7.032478 | 6.063204 | -0.969275 |

## 5. Exact-Zero Endpoints

`z_Q_QK`, `z_K_QK`, and `z_V_PV` are the actual downstream operands, not
substitutes derived from gate marginals. Values below `0.0001%` are reported
as such rather than rounded to zero.

| Architecture | LR | `z_a` | `z_m` | `z_h` | `z_Q_gate` | `z_K_gate` | `z_V_gate` | `z_Q_QK` | `z_K_QK` | `z_V_PV` | `z_context_Wo` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | `1e-5` | — | — | — | — | — | — | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| A0 | `3e-5` | — | — | — | — | — | — | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| A0 | `1e-4` | — | — | — | — | — | — | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| A1-H | `1e-5` | — | — | 46.0126 | — | — | — | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| A1-H | `3e-5` | — | — | 48.5426 | — | — | — | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| A1-H | `1e-4` | — | — | 56.1435 | — | — | — | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| A3 | `1e-5` | 50.4903 | 50.5532 | 44.6664 | — | — | — | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| A3 | `3e-5` | 51.1209 | 51.1795 | 46.0010 | — | — | — | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| A3 | `1e-4` | 50.9056 | 51.2374 | 54.6698 | — | — | — | <0.0001 | <0.0001 | <0.0001 | <0.0001 |
| A6-PRE | `1e-5` | 50.6918 | 50.7804 | 44.4624 | 33.8610 | 41.3799 | 43.8869 | 27.9895 | 36.3441 | 43.8869 | 4.5384 |
| A6-PRE | `3e-5` | 51.0024 | 51.1477 | 45.8668 | 15.9551 | 28.6805 | 43.4604 | 12.0649 | 24.3230 | 43.4604 | 1.0253 |
| A6-PRE | `1e-4` | 50.7339 | 51.3145 | 55.0876 | 11.0622 | 17.7997 | 52.2219 | 7.3507 | 14.4865 | 52.2219 | 0.4085 |
| A6-POST | `1e-5` | 50.6896 | 50.7853 | 44.4554 | 36.1387 | 44.5052 | 43.9294 | 36.1387 | 44.5052 | 43.9294 | 4.5238 |
| A6-POST | `3e-5` | 50.9964 | 51.0899 | 45.9191 | 22.6102 | 34.9734 | 43.6140 | 22.6102 | 34.9734 | 43.6140 | 1.0230 |
| A6-POST | `1e-4` | 50.3957 | 51.0548 | 54.5857 | 17.1210 | 24.0230 | 52.6207 | 17.1210 | 24.0230 | 52.6207 | 0.3477 |

## 6. Compute Endpoints

`Delta R_model` is relative to the same architecture at LR `3e-5`, in
percentage points. The topology ceilings are invariant across LR, as required.

| Architecture | LR | `R_block` | `R_block_max` | `R_model` | `R_model_max` | `U_arch` | Delta `R_model` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | `1e-5` | ~0 | 0.0000 | ~0 | 0.0000 | — | ~0 |
| A0 | `3e-5` | ~0 | 0.0000 | ~0 | 0.0000 | — | ~0 |
| A0 | `1e-4` | ~0 | 0.0000 | ~0 | 0.0000 | — | ~0 |
| A1-H | `1e-5` | 6.5714 | 14.2817 | 1.9683 | 4.2777 | 46.0126 | -0.1082 |
| A1-H | `3e-5` | 6.9327 | 14.2817 | 2.0765 | 4.2777 | 48.5426 | 0.0000 |
| A1-H | `1e-4` | 8.0183 | 14.2817 | 2.4017 | 4.2777 | 56.1435 | +0.3251 |
| A3 | `1e-5` | 19.0072 | 39.2748 | 5.6931 | 11.7637 | 48.3954 | -0.1041 |
| A3 | `3e-5` | 19.3548 | 39.2748 | 5.7972 | 11.7637 | 49.2804 | 0.0000 |
| A3 | `1e-4` | 20.5780 | 39.2748 | 6.1636 | 11.7637 | 52.3950 | +0.3664 |
| A6-PRE | `1e-5` | 46.0662 | 100.0000 | 13.7979 | 29.9524 | 46.0662 | +1.5792 |
| A6-PRE | `3e-5` | 40.7937 | 100.0000 | 12.2187 | 29.9524 | 40.7937 | 0.0000 |
| A6-PRE | `1e-4` | 41.3154 | 100.0000 | 12.3749 | 29.9524 | 41.3154 | +0.1563 |
| A6-POST | `1e-5` | 49.5032 | 100.0000 | 14.8274 | 29.9524 | 49.5032 | +1.1849 |
| A6-POST | `3e-5` | 45.5472 | 100.0000 | 13.6425 | 29.9524 | 45.5472 | 0.0000 |
| A6-POST | `1e-4` | 45.5343 | 100.0000 | 13.6386 | 29.9524 | 45.5343 | -0.0039 |

## 7. Findings and Limitations

- LR `1e-5` is undertrained at this fixed step budget in every architecture:
  selection loss is 1.301904 to 1.397773 above the matched `3e-5` control.
- LR `1e-4` lowers the 2,048-step selection loss within every architecture by
  0.969275 to 1.114008. This is evidence of faster short-budget optimization,
  not a long-budget LR recommendation.
- LR changes where zeros occur. At `1e-4`, A6 Q/K zeros decrease while hidden
  and V zeros increase. Relative to `3e-5`, this redistribution changes
  `R_model` by only +0.1563 points for A6-PRE and -0.0039 for A6-POST.
- The poorly fitted `1e-5` A6 runs have the largest `R_model` values, gaining
  +1.5792 points for PRE and +1.1849 for POST while incurring more than +1.35
  loss. This is a quality--sparsity tradeoff, not a promotion signal.
- A1-H and A3 instead gain both hidden zeros and modest `R_model` at `1e-4`.
  These remain within-topology observations at one seed.

The 2,048-step budget failed the campaign's global rank-survival backtest.
Selection data must not be mixed with campaign confirmation, and logical zero
products do not imply wall-clock speedup. No pressure method, learned gate,
multiple seed, or longer token budget is tested here.

## 8. Handoff and Durable Sources

- E0 reproducibility controls: 2/2 complete.
- S1-B0: 20/22 scientific cells complete and 20/20 currently executable;
  the two post-PV context cells remain dependency-gated.
- Executable S1 core: 20/132 cells complete.
- Next unused config prefix: `146`.
- B1 remains blocked on positive-kappa one-sided-gate engineering; B2 remains
  blocked on the learned-ATG pilot.

The common `3e-5` slice remains the controlled campaign reference. Do not
silently replace it with `1e-4` in predeclared cells; any later tuned-LR path
must remain separate and matched within architecture. LR `1e-5` is preserved
as a negative short-budget control.

Canonical identities are in
[`config-registry.yaml`](config-registry.yaml) and
[`run-registry.yaml`](run-registry.yaml). Central endpoints come from
`results/128-s1-b0-p14m-architecture-anchors-selection-propagation/001-20260718-184159-f73e2c80/activation_propagation.json`;
flank endpoints come from
`results/145-s1-b0-p14m-lr-flanks-selection-propagation/001-20260719-003521-94d90e97/activation_propagation.json`.
