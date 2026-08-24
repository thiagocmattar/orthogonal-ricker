# Experimental Plan — Executive Brief for Advisor Review

> **Status:** proposed, not approved. This brief and the
> [detailed proposal](exp-plan-v0.md) are review drafts. Only a reviewed
> [authoritative plan](docs/experiment_plan.md) can authorize scientific
> configurations, calibration, or launch.

## 1. Study in one paragraph

The study characterizes the trade-off between validation loss and activation
zero structure under activation pressure and explicit hard thresholds.
Discovery uses a randomly initialized 14M-parameter Pythia model; selected
settings are then frozen and tested at larger sizes. The opportunity metric
counts logical multiplications with an exactly zero activation operand; it is
**not** a measured speedup.

## 2. What is being changed and measured?

In each transformer block, the feed-forward network (FFN, also called the MLP)
expands the residual width, applies an activation function, then contracts it.
**The symbol h means only the expanded FFN activation-function output
immediately before the down-projection, in every block. It does not mean all FFN
activations.** At h, that function is stock GELU on the standard path and the
configured threshold on a thresholded path. Activation pressure always targets
only h. Every location below names the operand after any intervention at that
location.

### Activation locations

| Symbol | Exact location | Multiplication that consumes it |
| --- | --- | --- |
| a | Output of the attention-branch LayerNorm | Fused query/key/value projection |
| m | Output of the FFN-branch LayerNorm, before the FFN up-projection | FFN up-projection |
| h | Activation-function output between the FFN up- and down-projections: stock GELU or threshold output | FFN down-projection |
| q_pre, k_pre | Query and key immediately before rotary position encoding (RoPE) | Become operands of the query-key product after RoPE |
| q_post, k_post | Query and key immediately after RoPE | Actual operands of the query-key product |
| v | Value tensor from the query/key/value projection | Attention-probability-by-value product |

Pre- and post-RoPE locations are never merged: RoPE can turn coordinate-wise
zeros before rotation into nonzeros after rotation.

### Interventions

| Intervention | Definition |
| --- | --- |
| Standard path | No activation pressure or hard threshold; h uses stock GELU |
| L1 pressure | Optimize task loss plus λ times the mean absolute value of h; this encourages small values but does not guarantee exact zeros |
| OL1 pressure | Take the task-only AdamW step, remove a conflicting pressure component when conflict occurs, cap the correction with a step budget, then apply it; OL1 is conflict-aware, not unconditionally orthogonal |
| Symmetric hard threshold | During training and evaluation, retain x when abs(x) ≥ κ; otherwise output zero |
| One-sided hard threshold | During training and evaluation, retain x when x ≥ κ; otherwise output zero |
| Post-training zeroing sweep | Apply cutoffs only while evaluating a saved checkpoint; this is a diagnostic, not another training method |

Equality survives both hard-threshold rules. Hard thresholds create exact-zero
activations, but the current dense kernels still execute their multiplications.

### Primary outcomes

| Outcome | Definition and role |
| --- | --- |
| Validation loss | Held-out causal-language-modeling loss at a fixed input-token budget; lower is better |
| Exact-zero fraction z_s | Fraction of values exactly equal to zero at activation location s |
| Near-zero fraction n_s(0.01) | Fraction with abs(x) ≤ 0.01; primary pressure-response measure |
| Secondary near-zero fraction n_s(0.1) | Predeclared descriptive sensitivity measure |
| R_block | Fraction of declared multiplications inside measured transformer-block operations with an exactly zero activation operand |
| R_model | Same logical-opportunity fraction using the full declared model denominator, including dense operations outside the measured block where applicable |

Exact zeros, near-zero mass, logical multiplication opportunities, and measured
runtime are four different quantities. This study claims only what it directly
measures.

A condition is Pareto-optimal for two stated outcomes when no other eligible
condition has both no higher validation loss and no lower activation outcome,
with at least one strict improvement. The activation outcome is n_h(0.01) in A2
and R_model in B1, B2, and C.

## 3. Questions the study will answer

1. How does h-only pressure change validation loss and near-zero mass at h?
2. Does h-only pressure coincide with changes at untargeted FFN and attention
   locations, and how consistent are those changes across layers?
3. At a common pressure weight or comparable achieved h response, does OL1
   preserve validation quality better than L1, and which optimizer diagnostics
   accompany any difference?
4. Where should hard thresholds be inserted to improve the validation-loss
   versus logical-opportunity frontier?
5. At the same locations and cutoff, how do symmetric and one-sided hard
   thresholds differ?
6. Under frozen settings, does combining h pressure with hard thresholds improve
   the trade-off beyond either intervention alone?
7. Do frozen 14M findings replicate at larger sizes and a longer token budget?

Null, adverse, non-monotone, or method-divergent findings remain valid results.
A1 below is tuning only and cannot support an activation or efficiency claim.

## 4. Shared experimental protocol

| Component | Proposed choice |
| --- | --- |
| Model and initialization | Pythia-14M architecture, trained from random initialization; no released weights in the main study |
| Data | Pinned JeanKaddour/minipile cache: 1.492B cached tokens, sampled with replacement |
| Context and batch | 2,048 tokens; microbatch 4; 8-step accumulation; effective batch 32 sequences = 65,536 input tokens per update |
| Precision | BF16 CUDA computation with FP32 parameters and AdamW state |
| Optimizer | AdamW; β = (0.9, 0.95), ε = 1e-8, weight decay 0.1, no gradient clipping |
| Schedule | 260-update linear warmup, then constant learning rate |
| Budgets | 400M input tokens for screens; 1.7B for main comparisons; 3.4B for long-training robustness |
| Seeds | Seed 0 for screens; seeds 0, 1, and 2 for promoted comparisons |
| Checkpoints | Final model only; no intermediate or optimizer-state checkpoints |
| Diagnostics | Final-checkpoint pooled exact-zero, near-zero, and logical-opportunity measurements over the full declared partition; online values are monitoring only |
| Validation | Tune on 250 selection documents (311,296 input tokens); open the disjoint 250-document confirmation set (380,928 input tokens) once, after Phase A–D choices and analysis are frozen |
| Matching | Same-seed treatment and control must share initialization, data order, model, optimizer, batch, schedule, budget, and validation identities |
| Failure rule | Nonfinite or structurally invalid runs are reported but cannot be selected; unfavorable finite runs remain evidence; incomplete three-seed comparisons are not ranked |

Proposed Phase A model, tokenizer, data, cache, partition, and code identities
are pinned in the detailed proposal; license and ownership review remains an A1
prerequisite. Phase D and E identities remain unresolved.

## 5. Stage-by-stage design

### Phase A — Pressure at the FFN hidden activation h

| Stage | Question | Experiment and selection | Primary output |
| --- | --- | --- | --- |
| **A1: learning rate** | Which learning rate should every 14M experiment use? | Standard path, 1.7B tokens. Screen seed 0 at 1e-5, 3e-5, 1e-4, 3e-4, and 1e-3. Add seeds 1 and 2 for the two lowest-loss rates; freeze the lower three-seed mean. Exact ties favor the lower rate. | Appendix rate curve/table; no activation claim |
| **A2: pressure and spillover** | What changes when L1 or OL1 pressure acts only on h? | At the frozen rate and 1.7B tokens, test each method at λ = 0.1, 0.5, 1, 2, 5 with seed 0; reuse A1 controls. Over weights valid for both methods, form a loss/n_h(0.01) Pareto set within each method, take the union of represented weights, and promote at most its smallest, lower-middle, and largest weights. Add seeds 1 and 2. Diagnose h plus a, m, q_post, k_post, and v. | Figure 1 activation and quality effects |
| **A3: OL1 mechanism** | Which optimizer diagnostics accompany L1/OL1 differences? | No new training. Compare matched A2 runs at common weights and seeds, and against achieved n_h(0.01) or R_model; equal weights need not create equal corrections. | Mechanism table/appendix: conflict and projection frequencies, direction cosines, trust scaling, correction ratios, and paired method deltas |

**A2 estimates activation and quality effects; A3 characterizes optimizer
behavior using the same runs.**

### Phase B — Hard-threshold architecture

| Stage | Question | Experiment and selection | Primary output |
| --- | --- | --- | --- |
| **B1: placement** | Where should a hard threshold be applied? | Fix the symmetric rule. Screen the baseline plus 10 patterns below at κ = 0.03, 0.10, 0.30: 31 seed-0 runs at 400M. Retain families on the global loss/R_model Pareto frontier, ranked by number of frontier points then table order; keep at most three. Per family, promote its lowest-loss and highest-R_model cutoffs, deduplicated, to 1.7B with three seeds. If none qualifies, stop B2 and the threshold branch of C. | Figure 2 placement frontier; full screen in appendix |
| **B2: threshold rule** | For selected placement/cutoff pairs, how do symmetric and one-sided rules differ? | In B1-selected families, screen the one-sided rule on the same κ grid and reuse symmetric results. Keep cutoffs valid under both rules; within each family form a two-rule loss/R_model Pareto set and select the cutoffs of its lowest-loss and highest-R_model points. Promote both rules at each selected cutoff to 1.7B with three matched seeds. If no pair remains, B2 is inconclusive. | Proposed Figure 2 rule panel and paired result table |

**B1 changes placement with the rule fixed; B2 restricts attention to the
B1-selected families and compares rules at paired cutoffs.**

B1 evaluates these exact placement patterns; every listed location receives
the same symmetric rule and κ within a condition:

| Pattern | Thresholded activation locations |
| ---: | --- |
| 1 | h |
| 2 | m, h |
| 3 | a, m, h |
| 4 | a, m, h, q_post |
| 5 | a, m, h, k_post |
| 6 | a, m, h, v |
| 7 | a, m, h, q_pre, k_pre |
| 8 | a, m, h, q_post, k_post |
| 9 | a, m, h, q_pre, k_pre, v |
| 10 | a, m, h, q_post, k_post, v |

### Phase C — Final 14M intervention frontier

| Question | Experiment and selection | Primary output |
| --- | --- | --- |
| Are h pressure and the selected hard-threshold settings complementary? | Cross the dependency-selected threshold cohort with three pressure choices: none, L1, and OL1. The resulting six **family types** are no threshold/threshold × none/L1/OL1; there may be multiple conditions per type, and unthresholded controls are shared. Screen frozen combinations at 400M with seed 0; promote every eligible noncontrol on the global loss/R_model Pareto frontier to 1.7B with three seeds. | Figure 3 final 14M frontier and frontier-member table |

The saved-checkpoint post-training zeroing sweep is displayed with distinct
notation in Figure 3. It is a diagnostic, not a seventh training family.
Because the current rule promotes every eligible frontier condition, a numeric
promotion cap or total Phase C compute ceiling requires advisor approval before
configuration.

### Phase D — Frozen cross-size replication

Candidate sizes are Pythia 31M, 70M, 160M, and 410M. The discovery grid is not
repeated at each size.

| Stage | Question | Experiment and selection | Primary output |
| --- | --- | --- | --- |
| **D1: per-size learning rate** | Which rate makes each selected size a fair target? | At 400M tokens and seed 0, test one-third, one times, and three times the frozen 14M rate. Freeze the eligible rate with lowest final loss; ties favor the lower rate. A size with no eligible rate leaves D2/D3. | Appendix rate-by-size table |
| **D2: spillover across size** | Does the A2 activation pattern recur as width and depth change? | At 1.7B, compare the standard path with up to three frozen low-, middle-, and high-weight h-pressure settings. Labels refer to ordered weights, not effects. Method and seed cohorts remain to be frozen and must match within every claim. | Figure 4 exact-zero and near-zero changes by named location and size |
| **D3: intervention across size** | Does the frozen quality/opportunity frontier persist? | At 1.7B, compare the standard path, h pressure only, threshold only, quality-oriented combination, and aggressive combination. Carry placement, rule, κ, method, weight, and OL1 step budget from Phase C without retuning; use same-seed controls. | Figure 5 and scale table: loss, paired loss change, activation vector, absolute and paired R_model |

Phase D tests literal parameter transport, not a scaling law; scale-dependent
exceptions remain visible rather than being retuned away.

### Phase E — Robustness and scope

| Stage | Question | Proposed experiment | Status and report |
| --- | --- | --- | --- |
| **E1: longer training** | Do conclusions survive twice the main token budget? | Fresh 3.4B-token runs for the standard path, high h pressure, and selected combination at 14M and one larger size; match the 1.7B counterpart by seed, initialization, and data-order prefix. | Models, conditions, and seeds remain open. Proposed robustness table/appendix; a sixth main figure only if E1 becomes headline evidence. |
| **E2: pretrained adaptation** | Does the intervention transfer to a released checkpoint? | Supportive 300M-token adaptation comparison. | Blocked pending a separate checkpoint, optimizer, data, seed, contamination, implementation, and license protocol. Supplement only if approved. |

## 6. Planned paper and presentation outputs

This proposed output specification must be mirrored in the authoritative plan
before release.

| Output | Scientific message | Required evidence |
| --- | --- | --- |
| **A1 appendix figure and table** | The chosen 14M learning rate is the best predeclared candidate under the fixed recipe. | Complete five-rate screen, both promoted rates with three seeds, failures retained |
| **Figure 1 — pressure at h** | How L1 and OL1 change h, whether changes appear elsewhere, and how quality varies with achieved response. | Promoted A2 1.7B three-seed cohort; layer-by-weight maps; loss versus n_h(0.01) or R_model; representative h distribution with exact-zero atom separated |
| **A3 mechanism table/appendix** | Which conflict, projection, cosine, and trust-scale diagnostics accompany method differences. | Matched A2 events with complete optimizer-boundary counters |
| **Figure 2 — hard-threshold design** | Which placement/cutoff combinations lie on the best validation-loss/logical-opportunity trade-offs, and how the two rules differ. | Promoted B1/B2 1.7B paired cohorts; individual seeds, mean and sample SD; placement, rule, and κ labeled |
| **Figure 3 — final 14M frontier** | Whether pressure, hard thresholds, and their combination improve the quality/opportunity frontier. | Promoted Phase C cohort; six family types across selected threshold settings; post-training zeroing shown separately |
| **Figure 4 — activation changes across size** | Whether targeted and untargeted activation responses replicate by exact location. | Reviewed D2 multi-size matched cohort; paired location-level exact-zero and near-zero changes |
| **Figure 5 and main scale table** | Whether frozen interventions transport across sizes without retuning. | Reviewed D3 cohort; model-by-condition loss, paired loss change, activation vector, absolute and paired R_model |
| **Robustness output** | Stability at 3.4B tokens and, if authorized, transfer to a released checkpoint. | E1 robustness table/appendix; E2 supplementary table only |
| **Complete appendix** | Auditability, including negative and dominated evidence. | All screens and grids, individual seeds, learning curves, required activation distributions and diagnostics, exclusions, invalid runs, and selection/confirmation comparison |

Figures 1–5 are first rendered provisionally from promoted 1.7B multi-seed
selection evidence; their released versions use the one-time confirmation
evaluation of the frozen cohorts. Every multi-seed aggregate comparison shows
individual seeds, mean, sample SD, and sample size; the predeclared seed-0
representative distribution is labeled as such. Effects are same-seed paired.
Confirmation disagreement is disclosed and cannot trigger reselection.

## 7. Compute envelope and decisions requested

The following are gross input-token exposures under the current upper
promotion bounds:

| Stage | Gross exposure | Basis |
| --- | ---: | --- |
| A1 | 15.3B tokens | Five 1.7B screen runs plus four promoted-seed runs |
| A2 | Up to 37.4B tokens | Ten 1.7B seed-0 treatments plus up to 12 promoted-seed runs; controls reused |
| B1 | Up to 43.0B tokens | 31 × 400M screen plus up to six conditions × three seeds × 1.7B |
| B2 | Up to 64.8B tokens | Conservative bound before reuse: up to nine added 400M one-sided runs and 12 paired conditions × three seeds × 1.7B |
| C, D, E | To be bounded | Depend on upstream selections and unresolved model/seed scope |

A1 plus A2 is at most 52.7B tokens. The conservative gross total through B2 is
160.5B and overstates reusable work. Wall-clock ETCs require a 600-second
production-shaped training window plus separate setup, full-validation, and
checkpoint timings under the reviewed, committed A1 recipe. ETC and uncertainty
will be presented for approval before launch.

Advisor decisions requested:

1. Approve or revise the research questions, outcome definitions, and
   selection/confirmation policy.
2. Approve the full 1.7B-token A1 and A2 designs, or require a smaller tuning
   budget.
3. Freeze the OL1 step budget before A2.
4. Approve the B1 placement registry, both B2 threshold rules, and promotion
   bounds; set a Phase C promotion cap or compute ceiling.
5. Select the Phase D model sizes, method cohort, and seeds.
6. Decide whether E1 and E2 are core, supplementary, or deferred.
7. Complete license and identity review for the proposed model, tokenizer,
   dataset, and caches.
