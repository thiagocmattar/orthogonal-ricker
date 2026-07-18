# 03. Evaluation, Promotion, and Paper Outputs

## 1. Estimands Before Rankings

The campaign answers distinct questions with matched contrasts:

| Question | Contrast |
| --- | --- |
| What does a ReLU placement change? | AdamW architecture versus its matched AdamW parent at the same LR, seed, tokens, and data order |
| What does pressure change? | RN/OR/L1N/OL1 versus AdamW on the exact same architecture and active sites |
| What does orthogonalization change? | OR minus RN at identical `(weight,c,sigma)`; OL1 minus L1N at identical weight |
| What does RoPE placement change? | POST minus PRE with identical Q/K/V gates and all other fields fixed |
| What does signed preservation change? | `Gpm` minus `G+` at the same `kappa`, scope, and placement |
| What does threshold learning change? | learned versus fixed gate at matched family, initialization, scope, and scale definition |
| What changes with model size? | frozen method panel and normalized settings at matched token budgets, summarized separately at each size |

Nominal OR-RN and OL1-L1N contrasts compare the complete implemented
procedures. They do not by themselves isolate a pure geometric projection
effect if the procedures achieve different update magnitudes or sparsity.

`delta_L_arch` is reserved for the common-LR controlled slice. A
method-specific tuned result uses `delta_L_tuned_envelope` against independently
tuned same-architecture AdamW and is labeled non-causal. Every matched contrast
must use identical validation block ids; historical controls must be
re-evaluated on the campaign partitions before entering these deltas.

For architecture `A`, its predeclared parent `P(A)`, and pressure method `m`,
keep architecture and pressure effects separate:

\[
\begin{aligned}
\Delta L_{\rm arch}(A)&=L(A,\mathrm{AdamW})-L(P(A),\mathrm{AdamW}),\\
\Delta L_{\rm pressure}(A,m)&=L(A,m)-L(A,\mathrm{AdamW}),\\
\Delta R_{\rm arch}(A)&=R(A,\mathrm{AdamW})-R(P(A),\mathrm{AdamW}),\\
\Delta R_{\rm pressure}(A,m)&=R(A,m)-R(A,\mathrm{AdamW}).
\end{aligned}
\]

For a nonzero topology ceiling,
`delta_U_pressure = U(A,m)-U(A,AdamW)`. Here `R` denotes `R_model` unless an
operation or `R_block` is named explicitly.

## 2. Primary Endpoint Table

Every standard result handoff should be able to produce this schema:

| Architecture | Gate | Method | Parameters | Val. loss | `delta_L_arch` | `delta_L_pressure` | `z_a` | `z_m` | `z_h` | `z_Q_gate` | `z_K_gate` | `z_V_gate` | `z_Q_QK` | `z_K_QK` | `z_V_PV` | `R_block` | `R_model` | `delta_R_pressure` | `U_arch` |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

Use an em dash for an absent gate; never report an absent site as zero. `U_arch`
is also an em dash when the topology ceiling is zero, as for stock GELU. Include
config id, run id, seed, steps, tokens, validation partition/tokens, wall time,
tokens/s, and evidence status in the adjacent run table.

For the historical compact request "bring me back the results," return at
minimum:

```text
method, validation loss, R_block, R_model, z_a, z_m, z_h
```

Add Q/K/V, threshold, or context columns whenever those sites are present.

## 3. Exact-Zero Definition

Exact-zero metrics use integer counts, not histogram bins or an epsilon. For
site `s`:

\[
z_s=\frac{\sum_i \mathbf 1[x_{s,i}=0]}
{\sum_i 1}.
\]

The index pools all evaluated batches, tokens, layers, heads where applicable,
and coordinates for the named tensor. Save numerator and denominator, then
derive the percentage. For attention matrices, count only valid causal entries;
padding or masked future positions are not artificial zeros. Report per-layer
rates as well as the pooled value.

Distinguish these stages:

- `z_Q_gate`, `z_K_gate`, `z_V_gate`: outputs of the explicit gates;
- `z_Q_QK`, `z_K_QK`: actual operands entering QK after any RoPE operation;
- `z_V_PV`: actual V operand entering PV;
- optional `z_context_Wo`: context operand entering Wo.

PRE gate zeros can be mixed by RoPE, so gate-output zeros are not a substitute
for QK operand zeros. The paper endpoint diagnostic is pooled over the complete
named validation partition. S1 uses the realized token count of the
document-disjoint selection partition. After hyperparameters are frozen, the
primary confirmation quality claim uses the document-disjoint campaign-
confirmation partition; record its realized token count. Complete-cache loss is
a secondary endpoint because it contains the selection documents. Exact-zero/product
diagnostics may pool the full 692,224-token cache after selection is frozen.
Reports 04--06 already evaluated that complete cache, so the campaign-
confirmation half is not an historically untouched test set. A genuinely
held-out claim requires a new independent evaluation source. Online training
curves may use fewer fixed batches, but must be labeled separately.

Near-zero mass such as `abs(x)<=0.01` is a distribution diagnostic, not an
exact-zero result.

## 4. Compute Metrics

`R_block` is the number of directly counted avoidable scalar products divided by
all dense scalar products in the transformer blocks. `R_model` uses the same
numerator and adds the dense LM-head denominator. Counters must inspect actual
matmul operands and union overlapping opportunities; do not estimate QK or PV
avoidability by multiplying marginal zero percentages.

Report:

- per-operation zero-product fraction and numerator/denominator for QKV, QK,
  PV, Wo, W1, and W2;
- `R_block` and `R_model`;
- topology ceiling `R_block_max` and `R_model_max`;
- `U_arch = R_model/R_model_max`, undefined when `R_model_max=0`;
- `delta_R_model_arch` relative to matched AdamW;
- layerwise and normalized-depth decompositions.

The architecture formula and family ceiling table are in
`02-scaling-ladder.md`. Derive `L,d,T,V` and active gate topology from saved
configs/manifests. Do not infer them from an experiment label or use the
Report-05 hard-coded 14M constants in scaling figures.

## 5. Quality and Optimization Metrics

### Required for every training run

- selection/confirmation validation loss and perplexity;
- training and fixed-online-validation curves versus tokens;
- final step, total tokens, wall time, dense tokens/s, and peak memory;
- nonfinite, loss-instability, and gate-collapse indicators defined below;
- all activation-pressure losses and near-zero metrics already emitted by the
  harness.

### Naive pressure: L1N and RN

- task and pressure gradient norms;
- pressure/task gradient-norm ratio;
- gradient cosine and conflict frequency;
- raw and weighted pressure loss.

### Orthogonal pressure: OL1 and OR

- raw and final pressure-update ratios;
- applied scale;
- projection cosine before and after orthogonalization;
- fraction of logged steps at `step_budget=0.5`;
- eligible/skipped parameters and numerical fallbacks.

### Learned ATG

- `kappa` and `kappa/RMS(x)` per layer/site over time;
- threshold quantile in the pre-gate distribution;
- threshold gradient and update norms;
- transition-band mass for the soft backward estimator;
- gate zero rate, positive/negative survivor balance, and survivor RMS;
- `kappa -> 0`, all-zero, runaway, or frozen-threshold flags;
- exact threshold parameter and optimizer-group checkpoint round trip.

Task-only learned thresholds may shrink toward zero to recover signal. Do not
call threshold learning a sparsity objective unless it includes an explicit
active-mask/compute incentive. Such an incentive is a later method ablation,
not silently added to the ATG baseline.

## 6. Promotion Rule

Promotion is Pareto-based in validation loss and `R_model`; there is no single
loss/sparsity ratio.

1. Exclude invalid artifacts, nonfinite runs, universal gate collapse as
   operationalized below, and runs that did not reach their planned token
   budget from the primary ranking. Keep them in the registry as outcomes.
2. Compute Pareto sets within architecture/gate families, then compare the
   surviving families in absolute `R_model` and normalized `U_arch`.
3. Use `delta_L_arch` for the architecture cost versus its AdamW parent and
   `delta_L_pressure` for pressure cost versus same-topology AdamW.
4. Preserve at least one representative of the One-/Three-/Six-ReLU,
   PRE/POST, one-sided/signed, and fixed/learned contrasts when those families
   remain viable.
5. Promote Ricker settings only as matched RN+OR pairs and L1 settings only as
   matched L1N+OL1 pairs.
6. Retain a common-LR controlled slice even when a method-specific tuned LR is
   also reported.
7. Define the final quarter as logged steps `>=1536`. Universal collapse means
   every active gate site is at least 99.5% zero at each of the last three
   logged points. A loss-instability flag means median task loss over the last
   three final-quarter points exceeds the first three by more than 0.05; a
   sparsity-evaporation flag means pooled exact zeros fall by more than 10
   percentage points over the same comparison. Collapse is an exclusion;
   instability/evaporation block automatic promotion pending review, but do not
   erase the run. Freeze logging cadence before launch.
8. The working S1 safety guardrail is `delta_L_arch <= +0.10` versus the
   predeclared architecture parent, `delta_L_pressure <= +0.05` versus
   same-topology AdamW, and within-family Pareto membership, plus one of: at
   least `+1` percentage point `delta_R_pressure`; at least `+0.05`
   `delta_U_pressure`; or `delta_L_pressure <= -0.02` with no `R_model`
   regression. The normalized alternative
   reduces bias against low-ceiling architectures, while absolute `R_model`
   remains required for headline compute relevance. This is a permissive
   screening rule, not a paper claim. Any revision must be committed before S1
   endpoints are inspected.
9. S1/S2 select. S3 seeds 2--4 confirm. Do not tune after confirmation starts.

The campaign promotes panels, not one global winner. A low-ceiling topology can
be scientifically informative when `U_arch` is high, but the headline compute
claim also needs useful absolute `R_model`.

## 7. Uncertainty and Selection Bias

- S1 uses seed 0 plus ten seed-1 sentinels and is explicitly exploratory.
- S2 uses seeds 0 and 1 to assess rank survival from its same-cloud-stack
  step-2,048 milestone to step 8,192. Direct local-S1 versus cloud-S2 changes
  remain platform-sensitive secondary evidence.
- S3 uses fresh matched seeds 2, 3, and 4 after freezing the design; the headline
  panel should add seeds 5 and 6 if affordable.
- Family-scale endpoints use at least three matched seeds within each size.
- A matched seed requires identical data-order hashes and identical
  initialization for shared tensors. A common integer alone is insufficient.
  Plot every seed and report mean plus sample standard deviation and paired
  differences only for predeclared within-size contrasts. Across sizes, show
  separate seed distributions and summaries without implying pairing. With
  three seeds, emphasize effect sizes and consistency; do not use asymptotic
  p-values as the main evidence.
- Record all screened cells and all exclusion reasons. Do not report only the
  successful search path.
- Pre-register primary contrasts and final figure cohorts before S3. Once the
  confirmation partition is opened, further hypothesis generation belongs to a
  new discovery campaign.

## 8. Clipping and Distribution Diagnostics

Run a lightweight selection-partition propagation/product diagnostic for every
valid S1 cell; promotion requires its `R_block`, `R_model`, and `U_arch`.
Reserve complete-cache propagation, activation/weight histograms, and clipping
sweeps for promoted representatives.

Clipping frontiers must include:

- site-specific loss versus zeroing threshold;
- directly counted potentially avoidable model products versus loss;
- shared axes within an architecture comparison;
- thresholds zero, absolute, quantile, and RMS-relative where defined;
- frontier summaries at predeclared loss budgets, including zero loss change,
  `+0.01`, and `+0.05` validation loss.

Distribution plots keep the exact-zero atom separate and show density
conditional on nonzero values. Use both raw and RMS-normalized activation
distributions for cross-model comparison. Restrict LayerNorm, weight, learned
threshold, and detailed channel diagnostics to pinned representatives.

## 9. Figure and Table Suite

### S1 broad screen

1. Architecture/gate taxonomy with topology ceilings.
2. Validation loss versus `R_model`, faceted by architecture and gate family.
3. `delta_L_arch` versus `U_arch`.
4. LR, pressure-weight, Ricker-geometry, and threshold response maps; untested
   factorial cells remain visibly missing and are never interpolated.
5. Per-operation QKV/QK/PV/Wo/W1/W2 zero-product decomposition. Stacked
   components use the common `C_model` or `C_block` denominator so they sum to
   `R_model` or `R_block`; local operation rates are shown separately.
6. Learned `kappa`, exact-zero, and transition-band trajectories.
7. Pressure gradient/update ratios, cosines, and cap-binding frequency.

### Pythia-14M token scaling

1. Validation loss, `R_model`, and `U_arch` at the three measured token-rung
   endpoints; do not interpolate compute metrics without direct product
   diagnostics at intermediate milestones.
2. Pareto frontiers for the common promoted cohort at 2,048, 8,192, and 22,762
   steps, with the complete S1 cloud shown separately.
3. Common-cohort rank/Pareto survival from same-cloud-stack S2 step-2,048
   milestones to later budgets, labeled as selection-conditional; show the
   local-S1 bridge separately.
4. Exact-zero propagation heatmaps for the promoted method panels.
5. Site-specific and model-product clipping frontiers.

### Model scaling

1. Theoretical ceiling and observed `R_model` versus parameters.
2. `U_arch` and within-size matched validation-loss delta versus parameters.
3. Per-operation compute decomposition versus model size.
4. Site/layer zeros on normalized depth.
5. Quality-compute Pareto panels by size.
6. Logical opportunity versus measured sparse-kernel latency when available.

Main figures show selected cohorts; exhaustive curves and grids go in the
appendix. Tables must distinguish discovery, confirmation, provisional, and
invalid evidence.

## 10. Plotting Contract

- Every paper figure is vector PDF, sequentially named, and registered in
  `src/paper_exp/plot_catalog.py` and `docs/paper_map.md`.
- Use strict pinned cohorts for paper outputs. "Latest completed" is allowed
  only for exploratory previews.
- Preserve atomic staging, source/config/run hashes, and shared colorblind-safe
  style from `src/paper_exp/plot_style.py`.
- Put new screen and scale loaders/reductions/renderers in focused modules;
  do not expand Report 04/05 modules into a universal plotting file.
- Test pure reductions, topology ceiling derivation, matched-pair selection,
  and missing/duplicate cohort rejection.
- Show honest axes, validation-token counts, seed counts, and uncertainty where
  relevant.

## 11. Systems Claim Boundary

Logical product removal is necessary but not sufficient for speed. Before a
paper claims acceleration, add:

- mask structure and block occupancy per operation;
- sparse-kernel microbenchmarks on captured masks;
- dense-versus-sparse prefill latency, memory, and energy;
- end-to-end throughput with mask construction included;
- logical-product reduction versus realized speedup.

Until then, dense PyTorch tokens/s measures method overhead only. A lower
tokens/s value for a relufied or pressured run is not evidence that ReLU itself
accelerated or slowed an optimized sparse implementation.
