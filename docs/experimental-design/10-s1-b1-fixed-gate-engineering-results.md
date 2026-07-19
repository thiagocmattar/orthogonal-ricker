# 10. S1-B1 Fixed-Gate Engineering Results

## Decision

The fixed positive-threshold engineering gate passed on 2026-07-19. B1's 36
scientific AdamW cells may be materialized sequentially. These 128-step pilots
are plumbing evidence only: their validation losses do not select a gate
family, placement, scope, or threshold.

## Training pilots

Both runs used random-initialized Pythia-14M, AdamW at model LR `3e-5`, seeds
`0/0`, the same 128-step schedule, 8,388,608 training tokens, and the frozen
selection partition.

| Config | Architecture and gates | Train loss | Val. loss | Time (h) | Tokens/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 146 | A6-POST; fixed `G+`, `kappa=0.10`, at a/m/h/q/k/v | 10.13995 | 10.12764 | 0.02268 | 102,751 |
| 147 | A5-QK-PRE; ordinary a/m/h plus fixed `Gpm`, `kappa=0.10`, at q/k | 10.14757 | 10.13878 | 0.02155 | 108,132 |

Both completed 128/128 steps from clean commits, saved model and optimizer
state, and reloaded the exact intended gates in all six layers. Config 147
also reloaded with V absent and the final LayerNorm remained ungated.

## Complete-selection endpoints

Config 148 pooled integer exact-zero counts over all 152 complete 2,048-token
sequences in the frozen selection partition: 311,296 tokens across all six
layers. The final 443 cache tokens were excluded. An element is an exact zero
only when its computed tensor value compares equal to numeric zero; no
tolerance is used and percentages are not averages of batch percentages.

| Method | Val. loss | `R_block` | `R_model` | `z_a` | `z_m` | `z_h` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed `G+` A6-POST | 10.12764 | 70.8960% | 21.2350% | 53.0061% | 53.0278% | 74.1866% |
| Fixed `Gpm` A5-QK-PRE | 10.13878 | 36.7554% | 11.0091% | 49.9224% | 49.9353% | 48.5448% |

| Method | `z_q^gate` | `z_k^gate` | `z_v^gate` | `z_q^QK` | `z_k^QK` | `z_v^PV` | `z_context` | `U_arch` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed `G+` A6-POST | 70.7028% | 72.6783% | 70.1212% | 70.7028% | 72.6783% | 70.1212% | 24.3102% | 70.8960% |
| Fixed `Gpm` A5-QK-PRE | 45.5846% | 44.1723% | -- | 39.0327% | 37.8569% | 0.000001% | 0.000004% | 54.1699% |

For the PRE-RoPE pilot, RoPE preserved 44.56% and repopulated 55.44% of gated
zeros in rotary coordinates; pass-through coordinates preserved all gated
zeros. This is why `z_q^QK` and `z_k^QK` are below their gate-output rates.

Direct logical zero-product fractions were nonzero at every operation targeted
by each topology. For A6-POST they were 53.01% QKV, 91.48% QK, 70.12% PV,
24.31% Wo, 53.03% W1, and 74.19% W2. For A5-QK-PRE they were 49.92% QKV,
60.69% QK, 49.94% W1, and 48.54% W2. These are activation-side logical scalar
products, not measured sparse-kernel speedups.

## Acceptance and handoff

All five frozen criteria passed: focused tests, two finite clean pilots,
checkpoint reconstruction, pooled nonzero exact-zero/direct-product counts
without universal collapse, and engineering-only loss labeling. The canonical
evidence is:

- config 146 run `001-20260719-010308-764e5074`;
- config 147 run `001-20260719-010858-a4a15700`;
- config 148 run `001-20260719-011324-dd965e29`.

The next eligible block is B1's 36-cell fixed-threshold AdamW matrix. Continue
with the frozen 2,048-step feasibility/collapse and within-stratum policy; do
not globally rank cells or infer scientific conclusions from these pilots.
