# S1-B0 Central Architecture Anchor Results

## 1. Status and Scope

The first S1 launch set is complete: five Pythia-14M AdamW architecture
anchors reached 2,048 optimizer steps, and one combined endpoint diagnostic
measured exact-zero propagation over the complete campaign selection
partition. All five training manifests report `status: completed`, clean launch
provenance, 134,217,728 training tokens, and durable final checkpoints. No
optimizer state was saved, as predeclared for these non-learned AdamW cells.

This is S1 feasibility and within-stratum evidence only. It must not be used
for a global method ranking or paper-level confirmation.

Main observations:

- A1-H has the lowest selection loss in this five-run set, but the result is
  one seed at the short S1 budget.
- Expanding the ordinary-ReLU topology raises observed `R_model` from 2.08% in
  A1-H to 5.80% in A3, 12.22% in A6-PRE, and 13.64% in A6-POST.
- PRE RoPE repopulates some Q/K gate zeros. POST therefore preserves more
  zeros at the QK operands and produces the larger QK zero-product fraction.
- The A6 matched parents have not run. Their direct architecture effects
  remain unavailable despite the descriptive comparisons shown here.

## 2. Evidence Contract

All training cells use random initialization, AdamW without activation
pressure, model/data-order seeds `0/0`, LR `3e-5`, 100 warmup steps, sequence
length 2,048, micro-batch 4, and gradient accumulation 8. They share training
schedule hash
`db4fa092d7092d29edc3bf1e2005af69f4a92b8bf6e6d88cbb6a166e12be02fc`.

Quality and sparsity endpoints use only the frozen `selection` partition,
whose source-document hash is
`ffc857a6f0771929dd75c93bc17729de98a692f3a175ac5742cc9d101ff4ea47`.
The cache contains 311,739 tokens; evaluation uses its 152 complete sequences,
or 311,296 tokens, and excludes 443 trailing tokens. The confirmation
partition was not inspected for this launch set.

An exact zero is the numeric comparison `x == 0`, with no tolerance. Counts
are integer sums pooled across all 38 validation batches, 311,296 tokens, six
layers, and the feature/head coordinates of each named tensor. Percentages are
pooled count ratios, not averages of batch or layer percentages. Future causal
attention positions are excluded. An em dash denotes an absent gate; it does
not mean a measured zero rate of 0%.

`R_block` is the pooled count of direct exact-zero scalar products in QKV, QK,
PV, Wo, W1, and W2 divided by all dense products in those block operations.
`R_model` retains that numerator and adds the dense LM head to the denominator.
The maxima are topology ceilings obtained by setting every active gate output
to zero, and `U_arch = R_model / R_model_max`. These are logical compute
opportunities, not measured sparse-kernel speedups.

## 3. Training Results

| Config | Architecture | Canonical run | Final train loss | Selection loss | Direct-parent delta | Wall (min) | Tokens/s | Peak GPU (MiB) |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 123 | A0 | `001-20260718-165523-9f3b9b91` | 7.10384 | 7.04913 | — | 17.51 | 127,771 | 5,996 |
| 124 | A1-H | `001-20260718-171535-c9b88552` | 7.04011 | 6.98875 | -0.06039 vs A0 | 16.63 | 134,502 | 5,948 |
| 125 | A3 | `001-20260718-173453-33ecfcac` | 7.06499 | 7.01310 | +0.02435 vs A1-H | 17.97 | 124,482 | 5,996 |
| 126 | A6-PRE | `001-20260718-175512-d98db007` | 7.06998 | 7.01645 | pending A5-QK-PRE | 20.61 | 108,541 | 6,032 |
| 127 | A6-POST | `001-20260718-181755-c71b5faf` | 7.08340 | 7.03248 | pending A5-QK-POST | 21.03 | 106,357 | 6,056 |

The five training runs consumed 1.563 serial GPU-hours. Each saved 42 training
and 10 complete-selection validation events; every best saved validation
endpoint is at step 2,048. The A6 custom attention path is slower in dense
PyTorch execution: throughput is 12.81% lower for PRE and 14.56% lower for
POST than A3. This is implementation overhead, not evidence of realized sparse
acceleration.

## 4. Exact-Zero and Compute Endpoints

All values below are percentages pooled over the complete selection
partition. The tiny A0 compute count is shown as approximately zero because A0
has no active gate and a zero topology ceiling; it comes from a handful of
incidental floating-point exact zeros.

| Architecture | Val. loss | `z_a` | `z_m` | `z_h` | `R_block` | `R_block_max` | `R_model` | `R_model_max` | `U_arch` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 7.04913 | — | — | — | ~0 | 0.0000 | ~0 | 0.0000 | — |
| A1-H | 6.98875 | — | — | 48.5426 | 6.9327 | 14.2817 | 2.0765 | 4.2777 | 48.5426 |
| A3 | 7.01310 | 51.1209 | 51.1795 | 46.0010 | 19.3548 | 39.2748 | 5.7972 | 11.7637 | 49.2804 |
| A6-PRE | 7.01645 | 51.0024 | 51.1477 | 45.8668 | 40.7937 | 100.0000 | 12.2187 | 29.9524 | 40.7937 |
| A6-POST | 7.03248 | 50.9964 | 51.0899 | 45.9191 | 45.5472 | 100.0000 | 13.6425 | 29.9524 | 45.5472 |

Observed A0 values before rounding are `R_block = 1.327e-8` and
`R_model = 3.975e-9` as fractions. A1-H's `U_arch` equals its hidden-gate zero
rate to displayed precision because W2 is its only targetable operation. A6 can theoretically reach
every block operation in this logical graph, but the dense LM head limits its
model-wide ceiling to 29.95% at Pythia-14M.

## 5. Attention Detail

`z_Q_gate`, `z_K_gate`, and `z_V_gate` measure explicit gate outputs.
`z_Q_QK`, `z_K_QK`, and `z_V_PV` measure the actual downstream operands. QK
and PV product rates union zero opportunities across both operands; they must
not be reconstructed by multiplying marginal zero rates.

| Architecture | `z_Q_gate` | `z_Q_QK` | `z_K_gate` | `z_K_QK` | `z_V_gate` | `z_V_PV` | `z_context_Wo` | QKV products | QK products | PV products | Wo products |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | — | ~0 | — | ~0 | — | ~0 | ~0 | 0.0000 | ~0 | ~0 | ~0 |
| A1-H | — | 0.0000 | — | ~0 | — | 0.0000 | ~0 | 0.0000 | ~0 | 0.0000 | ~0 |
| A3 | — | ~0 | — | ~0 | — | ~0 | ~0 | 51.1209 | ~0 | ~0 | ~0 |
| A6-PRE | 15.9551 | 12.0649 | 28.6805 | 24.3230 | 43.4604 | 43.4604 | 1.0253 | 51.0024 | 31.5573 | 43.4625 | 1.0253 |
| A6-POST | 22.6102 | 22.6102 | 34.9734 | 34.9734 | 43.6140 | 43.6140 | 1.0230 | 50.9964 | 48.0381 | 43.6208 | 1.0230 |

For A6-PRE, RoPE reduces Q exact zeros by 3.8902 percentage points and K
exact zeros by 4.3575 points. It preserves 75.62% of all Q gate zeros and
84.81% of all K gate zeros at the QK operands. Within rotary coordinates
specifically, it repopulates 88.12% of Q zeros and 78.90% of K zeros; the
pass-through-coordinate zeros are preserved. POST gating occurs after RoPE,
so its gate-output and QK-operand zero rates are identical.

Despite 43–44% exact-zero V operands, context accumulation leaves only about
1.02% exact zeros before Wo. V sparsity therefore creates a substantial direct
PV opportunity but mostly does not remain elementwise zero after the reduction.

## 6. Interpretation Boundary

- These are five one-seed AdamW architecture controls, not a pressure test.
- The 2,048-step budget supports feasibility, collapse detection, and matched
  within-stratum contrasts only. It failed the campaign's global rank-survival
  backtest.
- A1-H's lower short-run loss is an observation, not evidence that it will
  remain best at S2 or under additional seeds.
- A6-PRE is only 0.00335 above A3 in selection loss and A6-POST is 0.01938
  above A3, but neither is a direct architecture delta. Their prespecified
  A5-QK parents are absent.
- PRE versus POST differs by 0.01603 loss and 1.4238 percentage points of
  `R_model` in this seed. Treat this as a matched placement observation, not a
  final placement choice.
- Selection evidence must not be mixed with the disjoint confirmation
  partition. Reports 04--06 previously inspected the complete cache containing
  both partitions, so confirmation is campaign-disjoint rather than
  historically untouched.
- Logical zero products do not imply wall-clock speedup. Sparse-kernel
  benchmarks remain necessary.

## 7. Campaign Progress and Durable Sources

- E0 reproducibility controls: 2/2 complete.
- S1-B0 scientific cells: 5/22 complete, or 5/20 of the currently executable
  non-context cells.
- Executable S1 core: 5/132 scientific cells complete (3.8%).
- Endpoint diagnostics: 1 combined complete-selection diagnostic complete in
  21.68 seconds over all five checkpoints.
- Scientific failures in this launch set: none.
- Next unused config prefix after diagnostic config 128: `129`.

The authoritative identities are in
[`config-registry.yaml`](config-registry.yaml) and
[`run-registry.yaml`](run-registry.yaml). Endpoint values come from
`results/128-s1-b0-p14m-architecture-anchors-selection-propagation/001-20260718-184159-f73e2c80/activation_propagation.json`.
The diagnostic pins the exact canonical run IDs listed in the training table;
it does not select runs by recency.

## 8. Recommended Next Launch Set

Do not launch from this document. Materialize and review the following five
ordinary-ReLU AdamW cells as prefixes 129 through 133. They use the exact
contract of configs 123 through 127 and all have A3 as their matched control.

| Proposed prefix | Design ID | Added gate(s) over A3 | Q/K placement | Purpose |
| ---: | --- | --- | --- | --- |
| 129 | `S1-B0-ARCH-A4Q-LR3EM5-S0` | Q | POST | Isolate Q contribution |
| 130 | `S1-B0-ARCH-A4K-LR3EM5-S0` | K | POST | Isolate K contribution |
| 131 | `S1-B0-ARCH-A4V-LR3EM5-S0` | V | not applicable | Isolate V/PV contribution |
| 132 | `S1-B0-ARCH-A5QKPRE-LR3EM5-S0` | Q and K | PRE | Supply the A6-PRE parent |
| 133 | `S1-B0-ARCH-A5QKPOST-LR3EM5-S0` | Q and K | POST | Supply the A6-POST parent |

The V-only config must carry the validator-required harmless
`qk_placement: post_rope`, while its scientific registry placement remains not
applicable. Expected serial training time is approximately 90 to 105 minutes,
with the measured custom-attention audit giving 104.1 minutes. Budget about
1h55 to 2h05 from the first launch through the five per-run reviews and registry
commits, or 2h05 to 2h15 including the combined selection diagnostic and final
handoff. From the current unmaterialized state, allow approximately 2h20 for
config review, materialization, and the complete launch set.
Completing this set enables A4-Q/K/V versus A3, A5 PRE versus POST, and A6
versus matched A5 parent contrasts without introducing pressure or LR changes.
The context-gate rows remain blocked and should not be substituted.
