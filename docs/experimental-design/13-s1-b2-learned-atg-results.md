# S1-B2 Learned-ATG Scientific Results

## 1. Status and Scope

S1-B2 closed on 2026-07-20 with canonical diagnostic run
`001-20260720-062718-d9765000`. Its complete-selection propagation artifact
has SHA-256
`10db05c91c15540496271917f67035e92ba00f0abcfec386f6712ce3df78f45f`.

The block contains 26 one-seed, 2,048-step AdamW cells with learned Adaptive
Threshold Gates (ATGs) and no activation pressure. It crosses attention gate
family, Q/K placement, QK versus QKV scope, absolute versus RMS-relative
threshold scale, branch scope, and threshold-sharing granularity.

Main within-stratum observations are:

- All 26 canonical cells remained finite and avoided universal gate collapse.
- Against the declared fixed-`kappa=0.10` controls, every learned absolute row
  increased `R_model` by 0.2095--1.9074 percentage points, while validation-loss
  changes were mixed, from -0.000709 to +0.015464.
- RMS-relative thresholds had lower loss and lower `R_model` than absolute
  thresholds in all 11 learned pairs. These are learned-scale comparisons;
  RMS-relative rows have no fixed-RMS control.
- In the 16-cell attention factorial, `Gpm` had lower loss and lower `R_model`
  than `G+` in all eight matched comparisons. POST versus PRE and QKV versus QK
  did not have a uniform loss direction, although both increased `R_model` in
  all eight matched comparisons.
- These one-seed screen results do not define a global winner, cutoff, or
  promotion decision.

Closure accounting:

- S1-B2: 26/26 scientific cells complete;
- executable S1 core: 82/132 scientific cells complete (62.12%);
- canonical scientific failures: none;
- invalid attempts: one retained, noncanonical infrastructure attempt for
  config `241` (`001-20260720-022211-bb0dbb78`), followed by the reviewed fresh
  canonical retry `002-20260720-034639-7c90a013`;
- B2's next unused config prefix was `248`; S1-B3 tranche 1 subsequently
  registered configs `248--255`, with pooled diagnostic `256` reserved.

## 2. Design and Evidence Contract

The two learned gate families are

\[
G^+_\kappa(x)=x\,\mathbf 1[x\geq\kappa],\qquad
G^\pm_\kappa(x)=x\,\mathbf 1[|x|\geq\kappa].
\]

Each learned threshold is parameterized as `kappa = softplus(rho)` in FP32,
with no weight decay. The forward pass uses the hard mask above; only the
threshold-gradient path uses the soft surrogate. All cells use the registered
engineering revision `tau=0.03`, model LR `3e-5`, threshold-LR multiplier
`10`, threshold LR `3e-4`, and `kappa_init=0.10`. Per-layer/site (`PLS`) is the
default sharing rule. Absolute rows compare directly with `kappa`; RMS-relative
rows compare with `kappa` times a detached full-gate-tensor RMS statistic.

Attention-factorial and granularity rows retain ordinary ReLU at attention
input `a`, MLP input `m`, and MLP hidden `h`; their learned gates act only on
the selected Q/K(/V) sites. Branch rows learn every active gate: `h` for A1-H,
`a/m/h` for A3, and `a/m/h/q/k/v` for A6-POST. PRE applies learned Q/K gates
before RoPE; POST applies them after RoPE. V is gated after QKV splitting and
before PV. The final LayerNorm remains ungated.

| Sub-block | Configs | Cells | Factors |
| --- | --- | ---: | --- |
| Main learned attention factorial | `221--236` | 16 | family `{G+, Gpm}` x placement `{PRE, POST}` x scope `{QK, QKV}` x scale `{absolute, RMS-relative}` |
| Learned one-sided branch scope | `237--242` | 6 | topology `{A1-H, A3, A6-POST}` x scale `{absolute, RMS-relative}` |
| Threshold-sharing granularity | `243--246` | 4 | family `{G+, Gpm}` x sharing `{global, per-site}` for absolute POST-QKV; PLS counterparts are `227/228` |

All training cells use random-initialized Pythia-14M, model/data-order seeds
`0/0`, 100 warmup steps, sequence length 2,048, micro-batch 4, gradient
accumulation 8, 65,536 tokens/update, and 134,217,728 training tokens. Their
training-schedule hash is
`db4fa092d7092d29edc3bf1e2005af69f4a92b8bf6e6d88cbb6a166e12be02fc`.

Diagnostic `247` uses the exact 26 canonical source runs in config order and
the frozen `selection` partition, hash
`ffc857a6f0771929dd75c93bc17729de98a692f3a175ac5742cc9d101ff4ea47`.
It evaluates all 152 complete sequences in 38 batches: 311,296 tokens, with
443 trailing cache tokens excluded.

An exact zero is the numeric comparison `x == 0`, with no tolerance. Counts
are integer sums pooled across the complete selection partition, all six
layers, and every applicable tensor coordinate. Percentages are pooled count
ratios, not averages of batch, sequence, or layer percentages. Each width-128
site has 239,075,328 pooled elements; `h`, at width 512, has 956,301,312. An em
dash denotes an unavailable or absent gate, not a measured zero rate of 0%.

`R_block` is the directly counted fraction of scalar products with at least
one exact-zero activation operand across QKV, valid-causal QK, valid-causal PV,
Wo, W1, and W2. Its denominator is 857,085,050,880 products. `R_model` keeps
the same numerator and adds the dense LM head, for 2,861,492,600,832 products.
Future causal positions are excluded. The `*_max` values set every active gate
output to zero under the row's topology, and
`U_arch = R_block/R_block_max = R_model/R_model_max`. These are logical compute
opportunities, not measured sparse-kernel speedups.

## 3. Canonical Pooled Endpoint Handoff

### Diagnostic 247 standard result handoff

Complete-selection pooled endpoints from `247-s1-b2-learned-atg-selection-prop`. Percentages use direct integer counts over 311,296 validation tokens; exact zero means numeric equality to zero with no tolerance. `z_q`, `z_k`, and `z_v` denote gate outputs when available. Logical-product fractions are not measured kernel speedups.

| Method | Config | Val. loss | `R_block` | `R_model` | `U_arch` | `z_a` | `z_m` | `z_h` | `z_q` | `z_k` | `z_v` | Val. tokens | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AdamW / A5-QK-PRE / G+ | `221-s1-b2-p14m-atg-gplus-pre-qk-abs-pls-k010-s0` | 7.019547 | 35.1738% | 10.5354% | 51.8388% | 51.0912% | 51.1274% | 45.9414% | 38.9935% | 52.6268% | — | 311,296 | valid |
| AdamW / A5-QK-PRE / Gpm | `222-s1-b2-p14m-atg-gpm-pre-qk-abs-pls-k010-s0` | 7.014580 | 27.6864% | 8.2927% | 40.8040% | 51.1274% | 51.1876% | 46.0191% | 12.9337% | 26.1186% | — | 311,296 | valid |
| AdamW / A6-PRE / G+ | `223-s1-b2-p14m-atg-gplus-pre-qkv-abs-pls-k010-s0` | 7.021966 | 55.2071% | 16.5358% | 55.2071% | 50.8609% | 51.0408% | 46.0464% | 39.8935% | 53.1586% | 68.8680% | 311,296 | valid |
| AdamW / A6-PRE / Gpm | `224-s1-b2-p14m-atg-gpm-pre-qkv-abs-pls-k010-s0` | 7.017634 | 42.4204% | 12.7059% | 42.4204% | 50.9932% | 51.1030% | 46.0279% | 12.6265% | 24.2153% | 53.2277% | 311,296 | valid |
| AdamW / A5-QK-POST / G+ | `225-s1-b2-p14m-atg-gplus-post-qk-abs-pls-k010-s0` | 7.030978 | 38.8255% | 11.6292% | 57.2207% | 51.1123% | 51.1285% | 45.9467% | 41.5092% | 56.5226% | — | 311,296 | valid |
| AdamW / A5-QK-POST / Gpm | `226-s1-b2-p14m-atg-gpm-post-qk-abs-pls-k010-s0` | 7.014303 | 28.7619% | 8.6149% | 42.3891% | 51.1222% | 51.1822% | 45.9474% | 12.8725% | 24.5475% | — | 311,296 | valid |
| AdamW / A6-POST / G+ | `227-s1-b2-p14m-atg-gplus-post-qkv-abs-pls-k010-s0` | 7.034106 | 59.0054% | 17.6735% | 59.0054% | 50.9350% | 51.0605% | 46.0768% | 41.7534% | 57.8002% | 68.7503% | 311,296 | valid |
| AdamW / A6-POST / Gpm | `228-s1-b2-p14m-atg-gpm-post-qkv-abs-pls-k010-s0` | 7.016171 | 43.6941% | 13.0874% | 43.6941% | 50.9906% | 51.1043% | 46.0044% | 12.5392% | 23.6394% | 53.3243% | 311,296 | valid |
| AdamW / A5-QK-PRE / G+ | `229-s1-b2-p14m-atg-gplus-pre-qk-rms-pls-k010-s0` | 7.015944 | 31.9023% | 9.5555% | 47.0174% | 51.0802% | 51.1268% | 46.0030% | 27.9842% | 39.2651% | — | 311,296 | valid |
| AdamW / A5-QK-PRE / Gpm | `230-s1-b2-p14m-atg-gpm-pre-qk-rms-pls-k010-s0` | 7.012834 | 27.2654% | 8.1666% | 40.1835% | 51.1744% | 51.2303% | 46.0359% | 11.6200% | 22.3886% | — | 311,296 | valid |
| AdamW / A6-PRE / G+ | `231-s1-b2-p14m-atg-gplus-pre-qkv-rms-pls-k010-s0` | 7.013298 | 46.5759% | 13.9506% | 46.5759% | 50.9672% | 51.1200% | 45.9889% | 29.0797% | 42.4683% | 48.2788% | 311,296 | valid |
| AdamW / A6-PRE / Gpm | `232-s1-b2-p14m-atg-gpm-pre-qkv-rms-pls-k010-s0` | 7.013086 | 30.2461% | 9.0594% | 30.2461% | 51.1159% | 51.1832% | 45.9398% | 11.0047% | 22.1014% | 11.2611% | 311,296 | valid |
| AdamW / A5-QK-POST / G+ | `233-s1-b2-p14m-atg-gplus-post-qk-rms-pls-k010-s0` | 7.030412 | 35.6899% | 10.6900% | 52.5994% | 51.1077% | 51.1297% | 45.8999% | 32.1453% | 43.0633% | — | 311,296 | valid |
| AdamW / A5-QK-POST / Gpm | `234-s1-b2-p14m-atg-gpm-post-qk-rms-pls-k010-s0` | 7.012258 | 28.3407% | 8.4887% | 41.7683% | 51.1431% | 51.2038% | 46.0278% | 12.0690% | 22.6382% | — | 311,296 | valid |
| AdamW / A6-POST / G+ | `235-s1-b2-p14m-atg-gplus-post-qkv-rms-pls-k010-s0` | 7.029365 | 49.9407% | 14.9584% | 49.9407% | 50.9801% | 51.0798% | 45.9915% | 31.5632% | 44.9983% | 48.3286% | 311,296 | valid |
| AdamW / A6-POST / Gpm | `236-s1-b2-p14m-atg-gpm-post-qkv-rms-pls-k010-s0` | 7.011463 | 31.4316% | 9.4145% | 31.4316% | 51.1335% | 51.1957% | 45.9451% | 11.7861% | 22.2902% | 11.3503% | 311,296 | valid |
| AdamW / A1-H / G+ | `237-s1-b2-p14m-atg-gplus-a1h-abs-pls-k010-s0` | 7.000878 | 12.2110% | 3.6575% | 85.5012% | — | — | 85.5012% | — | — | — | 311,296 | valid |
| AdamW / A1-H / G+ | `238-s1-b2-p14m-atg-gplus-a1h-rms-pls-k010-s0` | 6.991361 | 8.8433% | 2.6488% | 61.9206% | — | — | 61.9206% | — | — | — | 311,296 | valid |
| AdamW / A3 / G+ | `239-s1-b2-p14m-atg-gplus-a3-abs-pls-k010-s0` | 7.042317 | 27.1770% | 8.1402% | 69.1971% | 54.9728% | 56.1023% | 92.9602% | — | — | — | 311,296 | valid |
| AdamW / A3 / G+ | `240-s1-b2-p14m-atg-gplus-a3-rms-pls-k010-s0` | 7.014485 | 23.0276% | 6.8973% | 58.6320% | 55.8162% | 58.3428% | 61.0331% | — | — | — | 311,296 | valid |
| AdamW / A6-POST / G+ | `241-s1-b2-p14m-atg-gplus-a6post-abs-pls-k010-s0` | 7.065146 | 65.0807% | 19.4932% | 65.0807% | 54.5836% | 55.8901% | 92.5833% | 39.4157% | 54.9363% | 67.2074% | 311,296 | valid |
| AdamW / A6-POST / G+ | `242-s1-b2-p14m-atg-gplus-a6post-rms-pls-k010-s0` | 7.032399 | 54.2715% | 16.2556% | 54.2715% | 56.0624% | 58.6736% | 60.8104% | 32.4710% | 46.2745% | 49.0013% | 311,296 | valid |
| AdamW / A6-POST / G+ | `243-s1-b2-p14m-atg-gran-gplus-post-qkv-abs-global-k010-s0` | 7.033863 | 59.2903% | 17.7589% | 59.2903% | 50.9408% | 51.0763% | 46.0737% | 40.7549% | 54.0014% | 72.3215% | 311,296 | valid |
| AdamW / A6-POST / G+ | `244-s1-b2-p14m-atg-gran-gplus-post-qkv-abs-site-k010-s0` | 7.033780 | 60.5779% | 18.1445% | 60.5779% | 50.9320% | 51.0659% | 46.0280% | 43.1121% | 61.0487% | 72.1907% | 311,296 | valid |
| AdamW / A6-POST / Gpm | `245-s1-b2-p14m-atg-gran-gpm-post-qkv-abs-global-k010-s0` | 7.016235 | 44.6610% | 13.3770% | 44.6610% | 50.9658% | 51.0880% | 46.0317% | 10.7396% | 20.9171% | 60.2668% | 311,296 | valid |
| AdamW / A6-POST / Gpm | `246-s1-b2-p14m-atg-gran-gpm-post-qkv-abs-site-k010-s0` | 7.016566 | 46.0143% | 13.7824% | 46.0143% | 50.9621% | 51.0888% | 46.0293% | 12.8401% | 26.2720% | 59.3000% | 311,296 | valid |

The closure helper records these endpoints without ranking methods or changing promotion decisions.

In this table, `z_q`, `z_k`, and `z_v` are explicit gate-output exact-zero
rates when those gates exist. They are not generic Q/K/V densities. In
particular, PRE-RoPE `z_q` and `z_k` can differ from the downstream QK operands
after RoPE. The standard handoff does not include `z_q_qk`, `z_k_qk`,
`z_v_pv`, or `z_context_wo`; those quantities remain separately labeled in the
reviewed propagation artifact.

## 4. Prespecified Matched Contrasts

Deltas are `learned minus control` within the exact matched rows below. Loss
deltas are scalar loss differences; `R_model` and `U_arch` deltas are
percentage-point differences. Unlike architectures are not pooled into a
global score.

### 4.1 Learned absolute thresholds versus fixed `kappa=0.10`

The fixed controls come from the reviewed B1 diagnostics; learned endpoints
come from diagnostic `247`. Granularity rows share the same constant fixed
control but also change learned-threshold sharing, so they are not pure
per-layer/site learning contrasts.

| Learned config | Fixed control | Contrast | Delta val. loss | Delta `R_model` | Delta `U_arch` |
| --- | ---: | --- | ---: | ---: | ---: |
| `221` | `151` | G+ PRE-QK, PLS | +0.000495 | +0.6594 pp | +3.2444 pp |
| `222` | `175` | Gpm PRE-QK, PLS | +0.000245 | +0.6887 pp | +3.3887 pp |
| `223` | `157` | G+ PRE-QKV, PLS | +0.000016 | +0.8380 pp | +2.7976 pp |
| `224` | `181` | Gpm PRE-QKV, PLS | -0.000111 | +1.0557 pp | +3.5246 pp |
| `225` | `163` | G+ POST-QK, PLS | +0.000080 | +0.5300 pp | +2.6079 pp |
| `226` | `187` | Gpm POST-QK, PLS | +0.000280 | +0.7847 pp | +3.8610 pp |
| `227` | `169` | G+ POST-QKV, PLS | +0.000027 | +0.7613 pp | +2.5418 pp |
| `228` | `193` | Gpm POST-QKV, PLS | -0.000709 | +1.2124 pp | +4.0479 pp |
| `237` | `205` | G+ A1-H, PLS | +0.006790 | +0.3076 pp | +7.1909 pp |
| `239` | `207` | G+ A3, PLS | +0.015464 | +0.2095 pp | +1.7812 pp |
| `241` | `209` | G+ A6-POST, PLS | +0.012486 | +0.8787 pp | +2.9336 pp |
| `243` | `169` | G+ POST-QKV, global | -0.000217 | +0.8467 pp | +2.8268 pp |
| `244` | `169` | G+ POST-QKV, per-site | -0.000299 | +1.2323 pp | +4.1143 pp |
| `245` | `193` | Gpm POST-QKV, global | -0.000646 | +1.5021 pp | +5.0148 pp |
| `246` | `193` | Gpm POST-QKV, per-site | -0.000315 | +1.9074 pp | +6.3680 pp |

All 15 learned rows increased `R_model` by 0.2095--1.9074 percentage points
and `U_arch` by 1.7812--7.1909 points. Loss moved down in six contrasts and up
in nine, spanning -0.000709 to +0.015464. The logical-compute direction is
consistent at this seed; the quality direction is not.

### 4.2 Absolute versus RMS-relative learned thresholds

| Matched factor | Pairs | Delta val. loss range (RMS - ABS) | Delta `R_model` range | Direction count |
| --- | ---: | ---: | ---: | --- |
| Attention family/place/scope | 8 | -0.008669 to -0.000565 | -3.6729 to -0.1261 pp | 8/8 lower loss; 8/8 lower `R_model` |
| Branch topology | 3 | -0.032747 to -0.009517 | -3.2376 to -1.0087 pp | 3/3 lower loss; 3/3 lower `R_model` |

RMS-relative rows have no fixed-RMS control. They support learned-scale
comparisons only; their ABS pair is not a learned-versus-fixed causal effect.

### 4.3 Threshold-sharing granularity

| Family | Global | Per-site | Per-layer/site | Loss ordering (low to high) | `R_model` ordering (low to high) |
| --- | ---: | ---: | ---: | --- | --- |
| G+ POST-QKV ABS | `243` | `244` | `227` | `244 < 243 < 227` | `227 < 243 < 244` |
| Gpm POST-QKV ABS | `245` | `246` | `228` | `228 < 245 < 246` | `228 < 245 < 246` |

Across sharing rules, the loss span is 0.000326 for `G+` and 0.000395 for
`Gpm`; the `R_model` span is 0.4710 and 0.6949 percentage points,
respectively. These are small one-seed descriptive differences, not a
population effect.

### 4.4 Factor-level descriptive checks

- `Gpm - G+`: loss was lower in 8/8 attention pairs, by 0.000211--0.018154;
  `R_model` was lower in 8/8, by 1.3889--5.5439 percentage points.
- `POST - PRE`: loss was lower in four pairs and higher in four, with deltas
  from -0.001623 to +0.016067; `R_model` increased in 8/8 by
  0.3221--1.1377 points.
- `QKV - QK`: loss was lower in three pairs and higher in five, with deltas
  from -0.002647 to +0.003128; `R_model` increased in 8/8 by
  0.8928--6.0443 points.
- Branch expansion was monotone from A1-H through A3 to A6-POST for both loss
  and `R_model`. From A1-H to A6-POST, ABS changed by +0.064268 loss and
  +15.8357 `R_model` points; RMS changed by +0.041038 and +13.6068 points.

## 5. Learned-Threshold Behavior and Safety

The generated pooled handoff does not contain threshold trajectories. The
following training facts come from the separately reviewed canonical
`events.jsonl`, `metrics.json`, and exact checkpoint/optimizer reconciliation;
they are not inferred from pooled zero rates.

| Quantity | Reviewed result |
| --- | --- |
| Model and optimizer reload | 26/26 passed exact saved/reloaded tensor and optimizer-state equality. Intended FP32 `rho` ownership and scope passed with 1, 3, 6, 12, 18, or 36 learned thresholds per run, zero threshold weight decay, and threshold LR `3e-4`. |
| Safety flags | 0/26 had nonfinite thresholds, persistent-frozen parameters, universal gate collapse, loss instability, sparsity evaporation, automatic-promotion blockers, or additional ATG review flags. |
| Minibatch zero telemetry | The final-three minus first-three median active-gate exact-zero fraction ranged from -0.8529 to +3.7886 percentage points: 15 runs decreased and 11 increased. This is training-minibatch telemetry, not pooled validation. |

Final learned thresholds, reduced over the exact reviewed parameter summaries,
were:

| Prespecified stratum | Runs | Thresholds | Final `kappa` min / weighted mean / max |
| --- | ---: | ---: | ---: |
| Attention ABS, PLS | 8 | 120 | 0.100111 / 0.144379 / 0.201986 |
| Attention RMS, PLS | 8 | 120 | 0.107422 / 0.158080 / 0.217821 |
| Branch ABS, PLS | 3 | 60 | 0.083935 / 0.131828 / 0.199700 |
| Branch RMS, PLS | 3 | 60 | 0.094237 / 0.154286 / 0.214337 |
| POST-QKV ABS, global | 2 | 2 | 0.132783 / 0.132853 / 0.132923 |
| POST-QKV ABS, per-site | 2 | 6 | 0.130567 / 0.155189 / 0.192357 |

Task-only learned thresholds have no explicit sparsity objective. Thresholds
may move toward an easier task solution, including toward lower sparsity;
threshold learning itself is not sparsity pressure.

## 6. Findings and Interpretation Boundary

- Learned ATGs were trainable under all 26 declared B2 conditions without a
  canonical failure or campaign collapse flag.
- Learned absolute thresholds consistently raised logical-compute opportunity
  relative to fixed `kappa=0.10`, but validation-loss changes were mixed.
- RMS normalization consistently traded lower logical sparsity for lower loss
  relative to the matched learned ABS rows.
- The attention factorial separated meaningful topology effects: `Gpm` was
  consistently less sparse and lower-loss than `G+`; POST and adding V
  consistently raised `R_model`, but did not consistently improve loss.
- Sharing granularity moved `R_model` by less than 0.7 percentage points and
  loss by less than 0.0004 within each family at this seed.
- Threshold and minibatch-zero trajectories were heterogeneous despite clean
  safety checks, which is consistent with task-only threshold learning rather
  than direct sparsity optimization.

Limitations:

- This is one seed and 2,048 steps. It supports feasibility/collapse checks and
  matched within-stratum observations, not global ranking, promotion cutoff,
  or paper-level confirmation.
- Every B2 scientific row is AdamW without activation pressure. B2 does not
  test L1N, OL1, RN, OR, pressure effects, or orthogonalization.
- Learned-versus-fixed causal contrasts are valid only for absolute-threshold
  rows with the declared B1 controls. RMS-relative rows remain unmatched
  exploratory scale-normalization evidence.
- Selection evidence is separate from the disjoint campaign-confirmation
  partition, which must remain uninspected until promotion decisions are
  frozen.
- Pooled gate zeros and logical zero products do not establish wall-clock
  speedup. Sparse-kernel benchmarks are required for runtime claims.
- The standard handoff's Q/K columns are gate outputs. PRE-RoPE downstream QK
  sparsity requires the separate operand counters in the propagation artifact.

## 7. Artifact Audit and Durable Sources

- Diagnostic config: `configs/247-s1-b2-learned-atg-selection-prop.yaml`.
- Canonical diagnostic run: `001-20260720-062718-d9765000`.
- Propagation artifact:
  `results/247-s1-b2-learned-atg-selection-prop/001-20260720-062718-d9765000/activation_propagation.json`.
- Propagation SHA-256:
  `10db05c91c15540496271917f67035e92ba00f0abcfec386f6712ce3df78f45f`.
- Generated standard table:
  `tmp/s1-b3-config-drafts/diagnostic-reconciliation-review/diagnostic-247/results-handoff.md`.
- Canonical source identities and pooled endpoints:
  [`run-registry.yaml`](run-registry.yaml).
- Config identities and B2 closure status:
  [`config-registry.yaml`](config-registry.yaml).
- Clean-tree S1 audit on 2026-07-20: zero invariant errors; B2 certified
  26/26 with diagnostic `247` `closed_valid`; executable-core progress 82/132.
  The overall audit remained incomplete only because B3/B4 were open.
- Repository integrity on the same clean state: zero errors. Known historical
  warnings remain retained for the later open-source backfill.

Diagnostic `247` pins exact canonical source run IDs rather than selecting by
recency. Its standard table is generated from attached pooled endpoints, not
from final-minibatch telemetry. The closure helper performed no method ranking
or promotion decision.

## 8. Handoff

B2 is closed. B3 is the next campaign block; tranche 1 registers configs
`248--255` followed by pooled diagnostic `256`. No B2 global winner is selected.
Complete matched panels advance only under the frozen S1 policy in
[`03-evaluation-and-promotion.md`](03-evaluation-and-promotion.md), with the
2,048-step rank-survival veto in
[`06-s1-budget-backtest.md`](06-s1-budget-backtest.md) kept explicit.
