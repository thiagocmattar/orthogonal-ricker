# 02. Token and Model Scaling Ladder

## 1. Scaling Principle

Scale only after the previous rung has frozen its selection rule. The campaign
first increases training tokens at Pythia-14M, then transfers a fixed method
panel across the Pythia family through 410M. A larger run is confirmation, not a
new opportunity to search every hyperparameter again.

The transition is S1 core, only triggered conditional controls, candidate
freeze, RunPod qualification, and then S2/S3. The 182- or 184-cell S1 maximum is
not a mandatory intermediate rung. Cloud readiness, dated rates, expected
costs, and the operating procedure are defined in `05-runpod-cloud.md`.

All model-scale runs remain pretraining from random initialization on the same
MiniPile cache and sequence length. Released Pythia weights are never loaded.

## 2. Rungs

| Rung | Model | Steps | Training tokens | Selection use |
| --- | --- | ---: | ---: | --- |
| E0 | 14M | 128 | 8,388,608 | Engineering only; no scientific ranking |
| S1 | 14M | 2,048 | 134,217,728 | Broad discovery screen |
| S2 | 14M | 8,192 | 536,870,912 | Rank-survival and token-scaling screen |
| S3 | 14M | 22,762 | 1,491,730,432 | Frozen one-cache-pass-equivalent confirmation |
| S4 | 31M/70M/160M/410M | 2,048 | 134,217,728 | Per-scale LR and feasibility calibration |
| S5 | 31M/70M/160M/410M | 22,762 | 1,491,730,432 | Family-scale one-cache-pass-equivalent evidence |

The step counts assume 65,536 tokens per optimizer update. Hardware
micro-batching may change, but effective tokens per update and token order do
not.

## 3. Pythia-14M Token Promotion

### S2: 8,192 steps

Promote contrast-preserving families from S1:

1. `A3` as a complete Three-ReLU projection panel;
2. a matched ordinary Q/K PRE+POST pair as two complete method panels;
3. a matched fixed+learned ATG pair with the same family, scope, placement, and
   absolute `kappa`/initialization as two complete method panels;
4. stock `A0` AdamW.

Each complete panel is `{AdamW,RN,OR,L1N,OL1}` and reruns from scratch at model
initialization/data-order seeds 0 and 1 using frozen pressure and gate settings.
Every S2 run saves and evaluates a step-2,048 milestone on the same cloud GPU
SKU and image used through step 8,192. Primary rank/Pareto survival compares
these same-stack milestones; the local S1 endpoint is discovery/selection
evidence and its direct comparison with cloud S2 remains platform-confounded.
A learned ATG can enter a complete panel only after C6 supplies compatible
L1 and normalized-Ricker bridges; otherwise retain only its matched AdamW/L1
architecture evidence and do not call it a five-method promotion. The maximum
eligible design is `(5 + 10 + 10 + 1)*2 = 52` runs.

S2 can change which family advances, but it cannot reopen the full S1 grid. On
only the common promoted cells, record Spearman rank correlation with average
ranks for ties and Pareto survival between the cloud step-2,048 and step-8,192
milestones. This conditional correlation is selection-biased and cannot
describe the unpromoted S1 grid. Report the local-S1 comparison separately as a
platform-sensitive bridge. If the same-stack ordering is unstable, the correct
conclusion is that the short screen is an unreliable selector, not that the
latest endpoint is automatically better.

### S3: one-cache-pass-equivalent 14M token budget

Before inspecting S3 confirmation validation, freeze:

- `A3`;
- one selected ordinary attention architecture and its matched PRE/POST
  counterpart;
- one selected ATG architecture and its matched fixed/learned counterpart;
- one Ricker tuple and one L1 weight per promoted architecture;
- model LR, threshold settings, pressure scope, and all promotion contrasts.

The minimum confirmation per seed is a complete panel for `A3`, the selected
ordinary member, and the selected ATG member; AdamW-only matched counterparts
retain PRE/POST and fixed/learned architecture contrasts; stock is also run.
This is `5+5+1+5+1+1=18` cells per seed, or 54 runs for seeds 2, 3, and 4. If
pressure-by-placement or pressure-by-adaptivity is a primary claim, upgrade both
counterparts to complete panels before opening confirmation, yielding 78 runs.
If resources permit, add seeds 5 and 6 only for the headline ATG panel plus
stock: 12 further runs. The maximum S3 envelope is therefore 90.

No hyperparameter is retuned after viewing seeds 2--4. Any later revision is a
new, explicitly exploratory campaign.

## 4. Family Architecture and Learning-Rate References

The official Pythia table uses a native global batch of 2,097,152 tokens and
the following maximum learning rates. The `campaign reference LR` column is a
linear batch-size inference for 65,536 tokens per update:

\[
\eta_{\rm ref}=\eta_{\rm Pythia}\frac{65{,}536}{2{,}097{,}152}
=\frac{\eta_{\rm Pythia}}{32}.
\]

| Pythia size | Layers | Width `d` | Heads | Official native LR | Campaign reference LR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 14M | 6 | 128 | 4 | `1.0e-3` | `3.125e-5` (historical screen uses `3.0e-5`) |
| 31M | 6 | 256 | 8 | `1.0e-3` | `3.125e-5` |
| 70M | 6 | 512 | 8 | `1.0e-3` | `3.125e-5` |
| 160M | 12 | 768 | 12 | `6.0e-4` | `1.875e-5` |
| 410M | 24 | 1,024 | 16 | `3.0e-4` | `9.375e-6` |

Sources: the official [EleutherAI Pythia repository](https://github.com/EleutherAI/pythia),
the [Pythia paper](https://arxiv.org/abs/2304.01373), and the
[official model configs](https://github.com/EleutherAI/pythia/tree/main/models).

The inferred values are starting points, not official defaults for this smaller
batch. The current harness also differs from the native Pythia optimizer recipe
in beta2, weight decay, warmup, and schedule. The primary campaign keeps the
harness recipe fixed so architecture and pressure effects remain comparable to
configs `1--120`. A Pythia-recipe fidelity run is a separate robustness control,
not mixed into the main estimand.

## 5. Per-Scale LR Calibration

At each new size, run the selected architecture and all five methods for
`{0.5,1,2} * campaign reference LR` at S4: 15 cells per size. Use only the
selection-validation partition. Select the common LR from matched-architecture
AdamW alone using minimum selection loss, with the lower LR breaking an exact
tie; then apply that LR to all methods. A method-specific best-LR result may be
retained as a secondary, non-causal practical envelope, but it does not replace
the common-LR slice or enter `delta_L_arch`.

If a size requires a different effective batch because of systems limits,
recompute the LR candidate grid before launch and record that design revision.
Do not silently change both batch and LR.

## 6. Full Model-Scale Panel

Run all five family sizes: 14M, 31M, 70M, 160M, and 410M. S3 supplies the 14M
endpoint. S4 uses discovery seed 0. At each new size, S5 uses fresh matched
seeds 2, 3, and 4 and the minimum panel is:

| Cell family | Methods | Cells per seed |
| --- | --- | ---: |
| Stock architecture | AdamW | 1 |
| Three-ReLU `A3` anchor | AdamW | 1 |
| Selected ordinary attention placement | AdamW | 1 |
| Matched PRE/POST ordinary counterpart | AdamW | 1 |
| Selected ATG architecture | AdamW, RN, OR, L1N, OL1 | 5 |
| Matched fixed/learned ATG counterpart | AdamW | 1 |
| Total |  | **10** |

Use three matched seeds within each size: 30 full runs per new size, 120 across
31M--410M. At 70M, add RN/OR/L1N/OL1 to the `A3` anchor (12 more runs across
three seeds) to estimate one architecture-by-pressure interaction away from
14M. If resources permit, repeat that complete second panel at 160M and 410M;
label it an extension, not a required condition for the primary selected-ATG
scale curve.

Within-size matching requires identical data-order hashes and identical
initialization of every shared tensor, not merely the same seed integer. Runs at
different model sizes are not statistically paired.

The main transfer analysis freezes the 14M architecture, gate definition, and
normalized pressure settings, while using the predeclared per-scale common LR
selected from S4 AdamW. A fully frozen 14M-LR curve is a secondary robustness
control, not the primary family comparison. Absolute thresholds and
absolute Ricker `c,sigma` must not be transferred as if activation scale were
constant. Prefer RMS-relative parameters for the transfer result; if absolute
parameters are retuned, show both frozen and retuned curves.

## 7. Architecture-Specific Compute Ceilings

For width `d`, sequence length `T`, depth `L`, and vocabulary `V`, use
per-token dense scalar-product counts:

\[
C_{\rm block}=12d^2+d(T+1),\qquad
C_{\rm model}=LC_{\rm block}+dV.
\]

For the standard sites `a,m,h,q,k,v`, the union of all scalar products reachable
when the enabled gate outputs are entirely zero is

\[
\begin{aligned}
C_{\rm arch}^{\max}={}&3d^2 I_a+4d^2 I_m+4d^2 I_h\\
&+\frac{d(T+1)}{2}I_{q\lor k}
+\left[\frac{d(T+1)}{2}+d^2\right]I_v.
\end{aligned}
\]

Q and K are not additive: either all-zero operand removes QK. An all-zero V
removes PV and makes its context zero, exposing Wo. PRE and POST have the same
theoretical ceiling, but PRE realized products must use Q/K after RoPE. A
post-PV context gate requires a graph-union extension so its Wo opportunity is
not double counted with V.

The Wo term in the V-gate ceiling is a union/topological maximum attained when
V is entirely zero. It is not linearly attributable from marginal `z_V` under
partial sparsity; direct operand/product counters remain authoritative.

At `T=2048` and `V=50,304`, the theoretical ceilings are:

| Scale | One-ReLU `h`: block/model | Three-ReLU `a,m,h`: block/model | Q/K/V-only: block/model | Six-ReLU: block/model |
| --- | ---: | ---: | ---: | ---: |
| 14M | 14.28 / 4.28% | 39.27 / 11.76% | 60.73 / 18.19% | 100 / 29.95% |
| 31M | 20.00 / 7.58% | 54.99 / 20.85% | 45.01 / 17.07% | 100 / 37.92% |
| 70M | 25.00 / 12.35% | 68.74 / 33.97% | 31.26 / 15.45% | 100 / 49.42% |
| 160M | 27.27 / 19.87% | 74.99 / 54.65% | 25.01 / 18.22% | 100 / 72.88% |
| 410M | 28.57 / 24.93% | 78.57 / 68.55% | 21.43 / 18.70% | 100 / 87.25% |

Each entry is `R_block_max / R_model_max`. This confirms only the available
topological opportunity, not achieved sparsity. The scaling hypothesis must be
qualified: projection-gate model opportunity rises strongly with width and the
LM-head denominator is amortized, while the incremental Q/K/V-only block share
falls. Report absolute opportunity and architecture-normalized utilization:

\[
U_{\rm arch}=\frac{R_{\rm model}}
{R_{\rm model}^{\max}(\text{architecture})}.
\]

## 8. Hardware Calibration

Before committing each new model size:

1. find a micro-batch/accumulation pair that preserves 32 sequences per update;
2. record precision, gradient checkpointing, peak allocated GPU memory, and
   tokens/s;
3. verify the saved config and manifest report the same effective batch;
4. run a 10-step numerical and checkpoint smoke test;
5. estimate S4/S5 completion time from at least 128 steps, not by extrapolating
   the 14M throughput.

Before any scientific cloud cell, also pass the portable-cache, pinned-image,
resume, artifact-verification, stale-lifecycle, and local/cloud parity gates in
`05-runpod-cloud.md`. A matched panel uses one GPU SKU and image digest. Run
independent seeds on separate one-GPU Pods; multi-GPU execution is outside the
current scientific contract.

The 410M rung is blocked until the harness has tested gradient-checkpointing and,
if required, optimizer/offload support with correct checkpoint recovery. If one
sequence at micro-batch 1 cannot fit under the supported path, stop and record a
feasibility blocker; do not silently reduce effective tokens per update or the
training-token contract.

Dense PyTorch tokens/s is an implementation-overhead measurement. It is not an
expected sparse speedup, because the current dense kernels do not skip the
logical zero products counted by `R_model`.

## 9. Paper-Readiness Gates Beyond This Ladder

This ladder can support strong claims about random-initialized Pythia-family
pretraining on MiniPile. It does not by itself justify claims about all language
models or realized acceleration. Before submission, pre-register and complete:

- `TODO:` one independent data-distribution replication or a precisely scoped
  reason why the paper claim is MiniPile-specific;
- `TODO:` language-model quality evaluation beyond the repeatedly inspected
  selection loss, chosen before confirmation models are evaluated;
- `TODO:` sparse-kernel and end-to-end systems results at the achieved mask
  structures;
- `TODO:` a final literature audit and frozen primary-claim/figure list;
- `TODO:` release reproduction from a clean environment using only tracked
  configs, documented data preparation, and archived artifact hashes.

These gates are later stages. They are not reasons to broaden the current S1
matrix without control.
