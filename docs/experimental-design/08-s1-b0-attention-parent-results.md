# S1-B0 Attention Parent Results

## 1. Status and Scope

The second S1-B0 launch set is complete. Five Pythia-14M AdamW attention
parents reached 2,048 optimizer steps, and config `134` measured pooled
exact-zero propagation over their exact canonical checkpoints. All five
canonical training manifests report `status: completed` and clean launch
provenance. Each run has 134,217,728 training tokens, a durable final
checkpoint, 42 training events, and 10 complete-selection validation events.
The diagnostic is a clean, schema-v3 `statusless_complete` legacy workflow
whose five source manifests were all completed.

This is one-seed, short-budget S1 evidence. It supports feasibility and matched
within-stratum contrasts, not global ranking or paper-level confirmation.

## 2. Evidence Contract

All five training cells use random initialization, AdamW without activation
pressure, seeds `0/0`, LR `3e-5`, 100 warmup steps, sequence length 2,048,
micro-batch 4, and gradient accumulation 8. Their common schedule hash is
`db4fa092d7092d29edc3bf1e2005af69f4a92b8bf6e6d88cbb6a166e12be02fc`.

The diagnostic uses all 152 complete sequences in the frozen `selection`
partition: 311,296 evaluated tokens in 38 batches. It excludes 443 trailing
cache tokens. The source-document hash is
`ffc857a6f0771929dd75c93bc17729de98a692f3a175ac5742cc9d101ff4ea47`.

An exact zero is `x == 0` with no tolerance. Counts are integer sums pooled
over all evaluated sequences, tokens, six layers, and the relevant feature or
head coordinates; they are not averages of batch or layer percentages. Future
causal attention positions are excluded.

`R_block` is the fraction of direct scalar products with an exact-zero
activation operand across QKV, valid-causal QK, valid-causal PV, Wo, W1, and
W2. `R_model` keeps that numerator and adds the dense LM head to the
denominator. The corresponding `*_max` values set every active gate output to
zero, and `U_arch = R_model / R_model_max`. These are logical opportunities,
not measured kernel speedups. Gate outputs and downstream operands are kept
distinct because PRE-RoPE zeros can be repopulated. An em dash means that a
gate is absent, not that its measured zero rate is 0%.

## 3. Training Results

All five architectures have A3 as their predeclared direct parent, so the loss
delta below is both the A3 delta and `delta_L_arch` for this launch set.

| Config | Architecture | Canonical run | Train loss | Val. loss | Delta vs A3/direct parent | Wall (min) | Tokens/s | Peak GPU (MiB) |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 129 | A4-Q | `002-20260718-190411-3c1c6719` | 7.072230 | 7.021263 | +0.008164 | 19.17 | 116,700 | 6,020 |
| 130 | A4-K | `001-20260718-192601-b1795544` | 7.076086 | 7.024412 | +0.011313 | 19.49 | 114,792 | 6,020 |
| 131 | A4-V | `001-20260718-194713-d1dc3486` | 7.068373 | 7.016149 | +0.003050 | 19.35 | 115,619 | 6,008 |
| 132 | A5-QK-PRE | `001-20260718-200813-81b1dd63` | 7.067772 | 7.015856 | +0.002757 | 20.11 | 111,235 | 6,020 |
| 133 | A5-QK-POST | `001-20260718-203020-55473d96` | 7.081369 | 7.030639 | +0.017539 | 20.42 | 109,539 | 6,044 |

The five runs consumed 1.642 serial GPU-hours. The custom dense PyTorch paths
are 6.25% to 12.00% slower than A3; this is implementation overhead, not a
sparse-compute speed measurement.

## 4. Exact-Zero and Compute Endpoints

All values are percentages pooled over the complete selection partition.

| Config | Architecture | Val. loss | `z_a` | `z_m` | `z_h` | `z_Q_gate` | `z_K_gate` | `z_V_gate` |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 129 | A4-Q | 7.021263 | 51.1312 | 51.1729 | 45.9661 | 22.6942 | — | — |
| 130 | A4-K | 7.024412 | 51.1194 | 51.1491 | 45.8974 | — | 29.1944 | — |
| 131 | A4-V | 7.016149 | 51.0475 | 51.1994 | 45.9563 | — | — | 43.2056 |
| 132 | A5-QK-PRE | 7.015856 | 51.0695 | 51.1194 | 45.9961 | 15.5920 | 27.2178 | — |
| 133 | A5-QK-POST | 7.030639 | 51.1131 | 51.1350 | 45.9602 | 23.0982 | 34.2799 | — |

| Config | Architecture | `R_block` | `R_block_max` | `R_model` | `R_model_max` | `U_arch` |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 129 | A4-Q | 25.7342 | 67.8522 | 7.7080 | 20.3233 | 37.9269 |
| 130 | A4-K | 27.6605 | 67.8522 | 8.2850 | 20.3233 | 40.7658 |
| 131 | A4-V | 31.7177 | 71.4226 | 9.5002 | 21.3928 | 44.4085 |
| 132 | A5-QK-PRE | 28.1953 | 67.8522 | 8.4451 | 20.3233 | 41.5540 |
| 133 | A5-QK-POST | 32.9442 | 67.8522 | 9.8676 | 20.3233 | 48.5529 |

## 5. Attention and RoPE Detail

Gate zeros are distinct from the actual QK/PV operands. Product rates union
zero opportunities across both operands and therefore cannot be reconstructed
by multiplying marginal zero rates.

| Architecture | `z_Q_QK` | `z_K_QK` | `z_V_PV` | `z_context_Wo` | QKV products | QK products | PV products | Wo products |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A4-Q | 22.6942 | <0.0001 | <0.0001 | <0.0001 | 51.1312 | 22.3402 | <0.0001 | <0.0001 |
| A4-K | <0.0001 | 29.1944 | <0.0001 | <0.0001 | 51.1194 | 29.1313 | <0.0001 | <0.0001 |
| A4-V | <0.0001 | <0.0001 | 43.2056 | 0.7372 | 51.0475 | <0.0001 | 43.2091 | 0.7372 |
| A5-QK-PRE | 12.3039 | 23.1773 | <0.0001 | <0.0001 | 51.0695 | 30.9870 | <0.0001 | <0.0001 |
| A5-QK-POST | 23.0982 | 34.2799 | <0.0001 | <0.0001 | 51.1131 | 47.5986 | <0.0001 | <0.0001 |

For A5-QK-PRE, RoPE preserves 78.9116% of Q gate zeros and 85.1548% of K
gate zeros at the QK operands. Within rotary coordinates, it repopulates
83.8096% of Q zeros and 79.3972% of K zeros; pass-through-coordinate zeros are
fully preserved. POST gating occurs after RoPE, so A5-QK-POST gate-output and
QK-operand zero rates are identical.

## 6. Matched Architecture Contrasts

Compute deltas are percentage points. Every row shares the same seed,
schedule, token budget, and validation partition.

| Contrast | Question | Delta loss | Delta `R_block` | Delta `R_model` |
| --- | --- | ---: | ---: | ---: |
| A4-Q - A3 | Isolated POST Q gate | +0.008164 | +6.3794 | +1.9108 |
| A4-K - A3 | Isolated POST K gate | +0.011313 | +8.3057 | +2.4878 |
| A4-V - A3 | Isolated V/PV gate | +0.003050 | +12.3630 | +3.7030 |
| A5-QK-PRE - A3 | Joint PRE Q/K gates | +0.002757 | +8.8405 | +2.6479 |
| A5-QK-POST - A3 | Joint POST Q/K gates | +0.017539 | +13.5894 | +4.0704 |
| A5-QK-POST - A5-QK-PRE | Q/K placement | +0.014782 | +4.7489 | +1.4224 |
| A6-PRE - A5-QK-PRE | Add V under PRE | +0.000592 | +12.5984 | +3.7735 |
| A6-POST - A5-QK-POST | Add V under POST | +0.001840 | +12.6030 | +3.7749 |
| A6-POST - A6-PRE | Full QKV placement | +0.016030 | +4.7535 | +1.4238 |

## 7. Interpretation

- The isolated V gate gives the largest single-gate compute gain and the
  smallest single-gate loss delta: +3.7030 points of `R_model` at +0.003050
  loss versus A3. Its 43.21% V zeros reach PV, but context accumulation leaves
  only 0.74% exact zeros before Wo.
- PRE RoPE repopulates gate zeros: Q falls from 15.5920% at its gate to
  12.3039% at QK, and K falls from 27.2178% to 23.1773%. POST gains 1.4224
  `R_model` points over PRE, with +0.014782 loss in this short run.
- Adding V to the matched A5 parents changes loss by only +0.000592 (PRE) or
  +0.001840 (POST), while adding about 3.774 `R_model` points in both cases.
  These parent-child rows are the cleanest current evidence for V gating.
- The A5 POST joint-gate loss delta is close to the sum of isolated Q and K
  deltas: the residual interaction is -0.00194 loss. This is descriptive only.
- One seed and 2,048 steps cannot establish long-budget quality ordering.
  Selection evidence remains separate from confirmation, and logical products
  require sparse-kernel benchmarks before any speed claim.

## 8. Progress and Handoff

### Preserved Invalid Attempts

There were no scientific failures. Three infrastructure attempts are retained
as invalid, noncanonical history and contribute no endpoint evidence:

- Config `129` attempt 1, `001-20260718-190208-c6152824`, completed zero steps.
  Sandbox process containment ended the background child before the first
  training event; canonical retry 2 completed from the same immutable config.
- Config `131` attempt 2, `002-20260718-194735-f3cb784b`, was an accidental
  duplicate launched during an automatic-continuation handoff race. It was
  terminated after one step while canonical attempt 1 continued.
- Config `133` attempt 2, `002-20260718-203025-0a7597a7`, was the same class of
  handoff-race duplicate and was terminated after one step. Canonical attempt
  1 continued to completion.

### Campaign Progress

- E0 reproducibility controls: 2/2 complete.
- S1-B0 scientific cells: 10/22 complete, or 10/20 executable cells; two
  post-PV context cells remain dependency-gated.
- Full executable S1 core: 10/132 complete.
- Parent diagnostic: config `134`, run
  `001-20260718-205354-8821b592`, 20.84 seconds, schema v3, valid.

The diagnostic pins the exact canonical run IDs in the training table rather
than selecting by recency. Its source manifest records clean launch commit
`2deea54` and all five source manifests as completed.

## 9. Recommended Next Launch Set

The next B0 launch set is the 10 LR flanks for A0, A1-H, A3, A6-PRE, and
A6-POST at `1e-5` and `1e-4`. This preserves the five central `3e-5` controls
and changes only model LR. The cells are not yet materialized; the next unused
config prefix is `135`.

| Architecture | LR `1e-5` | LR `1e-4` | Purpose |
| --- | ---: | ---: | --- |
| A0 | 1 cell | 1 cell | Stock-model LR flank |
| A1-H | 1 cell | 1 cell | Hidden-ReLU LR flank |
| A3 | 1 cell | 1 cell | Three-ReLU LR flank |
| A6-PRE | 1 cell | 1 cell | Six-ReLU PRE LR flank |
| A6-POST | 1 cell | 1 cell | Six-ReLU POST LR flank |

The measured central-anchor runtimes project about 3.1 serial training hours
for these 10 cells. Budget approximately 3.5--4 hours locally including clean
launch commits, terminal reviews, registry updates, and the pooled endpoint
diagnostic.

Authoritative identities are in
[`config-registry.yaml`](config-registry.yaml) and
[`run-registry.yaml`](run-registry.yaml). Endpoint values come from
`results/134-s1-b0-p14m-attention-parents-selection-propagation/001-20260718-205354-8821b592/activation_propagation.json`.
