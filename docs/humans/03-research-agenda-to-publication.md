# Research Agenda: From Pilot Evidence to Publication

Date: 2026-07-10

Purpose: turn the current Pythia-14M activation-pressure pilot into a falsifiable research program aimed at a strong publication in venues such as NeurIPS, ICLR, or TNNLS.

Status: internal planning document. Observations, hypotheses, and candidate paper claims are separated explicitly. Nothing marked as a hypothesis or candidate contribution should be presented as an established result.

Related repository documents:

- [Methods](../methods.md)
- [Experiment log](../experiment_log.md)
- [Paper map](../paper_map.md)
- [Fixed-step activation-pressure screen](02-fixed-step-pressure-screen.md)
- [ReLU site-scope comparison](../../report/03-2026-07-09-relu-site-scope-comparison/03-2026-07-09-relu-site-scope-comparison.pdf)

## Executive Decision

The strongest publication path is not three separate projects. It is one ordered program:

> **(3) establish whether local activation sparsification redistributes functional load across Transformer pathways; (2) use that mechanism to motivate a global, hardware-aware sparse-read architecture; and (1) retain orthogonal pressure only if it provides a reproducible optimization advantage under controlled conflict.**

The recommended `3 -> 2 -> 1` priority is:

| Priority | Direction | Role in the paper | Current verdict |
| --- | --- | --- | --- |
| `3` | Cross-layer and cross-pathway activation redistribution | Main scientific mechanism | Most promising, but current evidence shows coupling rather than causal compensation |
| `2` | Sparsity before expensive attention and MLP projections | Main architectural and systems intervention | Correct computational target, but plain ReLU after normalization is prior art |
| `1` | Naive versus orthogonal sparsity pressure | Conditional optimization contribution | Orthogonalization is active and helps transiently, but has no established endpoint advantage |

The candidate central thesis is:

> Local activation sparsification is not necessarily a local intervention. A Transformer may preserve task performance by redistributing functional load into other branches, layers, parameter scales, or dense operations. A global, hardware-aware activation budget should make this redistribution visible and prevent compute from escaping into unaccounted dense paths.

This thesis is not yet demonstrated. The rest of this document specifies what evidence would establish or falsify it.

Terminology rule:

- Use **cross-site coupling** or **activation redistribution** for the current distributional evidence.
- Use **functional compensation** only after causal branch-reliance evidence exists.
- Use **compute leakage** only after the displaced functional load is connected to measured or kernel-derived execution cost.

## Part I: Current Repository Status

### 1. Current Experimental Scope

The repository currently provides a lean, auditable pretraining harness with the following scope:

- Pythia-14M architecture initialized randomly.
- MiniPile training from a local token cache.
- FP32 parameters with bf16 autocast.
- AdamW task optimization.
- GELU and ReLU MLP activation experiments.
- Naive L1 and Ricker activation-pressure objectives.
- Adam-step orthogonal L1 and Ricker corrections.
- Pressure sites covering MLP hiddens, residual streams, and attention outputs.
- Fixed-step screens, one-cache-equivalent-pass runs, clipping frontiers, activation histograms, weight histograms, gradient diagnostics, and site-scope comparisons.
- Saved configs, metrics, predictions, manifests, events, and final checkpoints for each run.

The current program is a low-resource mechanism-discovery pilot. It is not a faithful reproduction of Pythia pretraining and is not yet a scale-valid demonstration of efficient language modeling.

### 2. What the Repository Already Does Well

| Dimension | Current strength | Research value |
| --- | --- | --- |
| Reproducibility | Runs are config-driven and save manifests, metrics, events, predictions, and checkpoints | Makes later scientific audits and matched reruns possible |
| Initialization semantics | Pythia architecture and released Pythia checkpoint weights are kept distinct | Avoids a major pretraining versus continuation confound |
| Method separation | Naive L1/Ricker and orthogonal L1/Ricker remain separate in configs and metrics | Prevents silent mixing of optimizer semantics |
| Breadth of pilot | Pressure magnitude, pressure shape, activation function, and site scope have been explored | Provides useful evidence for selecting the next falsification experiments |
| Internal diagnostics | Task/pressure gradient geometry, Adam-step projection, activation mass, clipping frontiers, and distributions are logged | Exposes optimization mechanisms that endpoint loss alone would hide |
| Artifact discipline | Figures are regenerated from saved results and plotting style is centralized | Provides a strong base for a paper-quality evidence package |

The engineering foundation is stronger than the current scientific claim. This is the correct asymmetry for a pilot repository: the next work should concentrate on causal design and scale rather than adding general-purpose infrastructure.

### 3. Strongest Current Empirical Signals

#### 3.1 ReLU MLP-only one-pass results

The completed ReLU MLP-only runs use one seed and one MiniPile token-cache-equivalent training budget.

| Method | Config | Final validation loss | Final exact-zero MLP mass |
| --- | ---: | ---: | ---: |
| AdamW | `77` | 4.8404 | 52.35% |
| Ricker naive | `78` | 4.8134 | 93.91% |
| Orthogonal Ricker | `79` | 4.8087 | 93.23% |
| L1 naive | `80` | **4.7967** | **95.35%** |
| Orthogonal L1 | `81` | 4.7981 | 93.02% |

Planning interpretation:

- ReLU creates an exact-zero substrate on which both L1 and Ricker pressure can produce very high dynamic sparsity.
- Naive L1 does not show an endpoint validation-loss penalty in this run; it has the best single endpoint.
- Orthogonal L1 reaches nearly the same endpoint with slightly lower exact sparsity.
- These differences are one-seed point estimates. They do not establish method ranking.

#### 3.2 Orthogonalization is active, but its clearest effect is transient

An audit of the existing event logs found the following for ReLU MLP-only L1:

- Naive L1 mean effective `||w g_pressure|| / ||g_task||`: approximately `0.145`.
- Naive L1 task-pressure conflict: `73.4%` of logged steps.
- Orthogonal L1 Adam-step-space conflict: approximately `100%` of logged steps.
- Orthogonal L1 mean applied correction/task-step ratio: approximately `0.205`.

Representative validation losses show early protection followed by convergence of the endpoints:

| Step | AdamW | L1 naive | Orthogonal L1 |
| ---: | ---: | ---: | ---: |
| 1,000 | 7.8868 | 7.9649 | **7.8986** |
| 5,000 | 5.9602 | 6.0008 | **5.9199** |
| 10,000 | 5.3952 | 5.3741 | **5.3382** |
| 20,000 | 4.9336 | 4.8921 | **4.8819** |
| 22,762 | 4.8404 | **4.7967** | 4.7981 |

Planning interpretation:

- The lack of endpoint separation is not explained by an inactive projection or non-conflicting gradients.
- Orthogonalization appears to protect the early learning curve.
- Naive L1 subsequently compensates and catches up by the current endpoint.
- One cache-equivalent pass is not asymptotic convergence. The correct current statement is about the observed endpoint, not final convergence.

`TODO:` turn this read-only event-log audit into a saved, reproducible analysis config/result/figure.

#### 3.3 Parameter-scale adaptation is visible

Checkpoint inspection produced the following ReLU MLP diagnostics:

| Model | MLP input-bias mean | Negative input biases | Up-row norm | Down-column norm |
| --- | ---: | ---: | ---: | ---: |
| AdamW | -0.00155 | 64.9% | 0.239 | 0.239 |
| L1 naive | -0.00857 | 94.9% | 0.242 | 0.303 |
| Orthogonal L1 | -0.00911 | 95.9% | 0.237 | 0.300 |

This is consistent with a parameterization escape route in which the model changes ReLU thresholds or activation scale while amplifying the following projection. For positive `alpha`, the ReLU path has the approximate scale symmetry

```text
V ReLU(Wx + b) = (V / alpha) ReLU(alpha Wx + alpha b).
```

Weight decay constrains but does not remove this freedom. Raw activation L1 can therefore fall without a proportional reduction in functional contribution or realized compute.

`TODO:` save the bias and paired up/down path-norm diagnostic as a versioned result artifact. Existing weight histograms omit the most important bias behavior.

#### 3.4 High exact sparsity appears mostly dynamic

A read-only full-validation diagnostic over 692,224 tokens found that naive ReLU L1 had:

- Only `5 / 3072` globally inactive MLP channels, or `0.16%`.
- Median active neurons per token by layer of `25, 10, 15, 18, 21, 26` out of 512.
- Median per-channel active rates by layer of `5.84%, 1.43%, 2.24%, 2.85%, 3.39%, 4.06%`.

This suggests that the observed 95% exact-zero mass is primarily token-dependent routing rather than deletion of whole neurons. This distinction is important: dynamic sparsity may support conditional computation, while static dead capacity may simply imply that the MLP could have been narrower.

`TODO:` reproduce this diagnostic through a committed config and save per-layer channel-frequency, support-turnover, and batch-union statistics before treating it as paper evidence.

#### 3.5 Cross-site density changes are visible

The ReLU site-scope analysis shows that MLP-only naive L1 changes full-validation histogram estimates approximately as follows:

| Site | AdamW near-zero mass | MLP-only naive L1 near-zero mass | Direction |
| --- | ---: | ---: | --- |
| MLP hiddens | 53.6% | 96.1% | Much sparser |
| Residual streams | 13.3% | 16.4% | Slightly sparser |
| Attention outputs | 51.4% | 37.8% | Denser |

The attention-output result is consistent with cross-pathway compensation. The residual result is not a simple conservation pattern, and other methods show different responses. The defensible current term is **cross-site coupling**, not a conservation law or proven compensation.

#### 3.6 Compute actionability of the current activation sites

The historical MLP+residual and all-site pressure experiments have pressured and measured activation sites that are useful for representation analysis but are not all positioned to avoid computation. Specifically, configs `65`-`66` and `90`-`93` include pressure on `attention_outputs` and `residual_streams`, while configs `70`-`73` and `86`-`89` include residual-stream pressure alongside MLP-hidden pressure. The associated clipping and histogram experiments also inspect these sites diagnostically.

| Repository site | Exact hook position | Can zeros avoid upstream computation? | Interpretation |
| --- | --- | --- | --- |
| `mlp_hiddens` | After the MLP activation and before `dense_4h_to_h` | **Yes, in principle:** a sparse kernel could skip rows/columns of the MLP down projection | Compute-actionable site, although the current harness still executes a dense linear layer |
| `attention_outputs` | Output of the complete GPT-NeoX attention module, after QKV projection, attention mixing, and attention output projection; immediately before residual addition | **No for the attention block:** all expensive attention operations have already executed | Diagnostic/regularization site; high sparsity here is not skipped attention compute |
| `residual_streams` | Input to the Transformer block, before the branch LayerNorms | **No in the current Pythia architecture:** LayerNorm mean subtraction generally turns residual zeros into dense normalized branch inputs | Diagnostic/regularization site; residual sparsity is not projection sparsity |
| `attention_inputs` | New post-LayerNorm/post-ReLU branch input immediately before fused QKV projection | **Yes, in principle:** the same sparse input can skip work in Q, K, and V projections | Compute-actionable site introduced for the post-LayerNorm ReLU configs |
| `mlp_inputs` | New post-LayerNorm/post-ReLU branch input immediately before `dense_h_to_4h` | **Yes, in principle:** sparse inputs can skip the MLP up projection | Compute-actionable site introduced for the post-LayerNorm ReLU configs |

Therefore:

> The historical aggregate "all-site sparsity" is not aggregate computational sparsity. In particular, high sparsity in `attention_outputs` or `residual_streams` does not imply that the corresponding attention or MLP FLOPs were avoided.

The post-LayerNorm ReLU configs `98` and `99` move the sparse interfaces to `attention_inputs` and `mlp_inputs`, while retaining `mlp_hiddens`. These are the correct locations for avoiding projection work in principle. However, these experiments still call dense PyTorch/Hugging Face linear operators. Until a sparse kernel or equivalent conditional execution is integrated, the zeros change representations but do not produce wall-clock acceleration.

#### 3.7 Where the largest ideal compute savings are located

For the local Pythia-14M architecture, `d_model = 128`, `d_ff = 512`, and there are six layers. Ignoring biases, normalization, attention score/value products, the output head, and kernel overhead, the dense linear MACs per token in one block are:

| Block operation | Dense MACs per token | Share of block linear MACs | Sparse input that could avoid them |
| --- | ---: | ---: | --- |
| Fused QKV projection | `3 * 128 * 128 = 49,152` | 25.0% | `attention_inputs` |
| Attention output projection | `128 * 128 = 16,384` | 8.3% | Attention context **before** the output projection, not current `attention_outputs` |
| MLP up projection | `128 * 512 = 65,536` | 33.3% | `mlp_inputs` |
| MLP down projection | `512 * 128 = 65,536` | 33.3% | `mlp_hiddens` |
| Total block linear work | `196,608` | 100.0% | All four actionable interfaces together |

If an actionable tensor has density `rho`, its ideal avoided MACs are approximately `(1 - rho)` times the MACs of its immediately downstream operation. This is only an upper-bound accounting model; sparse packing, mask construction, irregular access, and hardware occupancy reduce realized gains.

Within a Pythia block, the two highest-leverage individual interfaces are therefore:

1. The post-normalization MLP input before the up projection.
2. The MLP hidden after ReLU and before the down projection.

Each controls one third of the block's dense linear MACs. The post-normalization attention input controls the combined QKV projection, another one quarter.

Two larger whole-model considerations must also be explicit:

- **Long-context attention:** feature sparsity before QKV does not skip `QK^T`, softmax, or `AV`, whose cost grows quadratically with sequence length. Avoiding that work requires structured token-, head-, or attention-block sparsity before score/value computation.
- **Output vocabulary head:** Pythia-14M's `128 x 50,304` language-model head costs about `6.44 million` MACs per token, making it unusually important in this small, shallow model. Sparsifying the final post-normalization hidden input could in principle reduce this projection, but it is not currently targeted and would require its own quality and kernel study.

For a modern gated FFN, the largest compute-actionable feature interface is usually the normalized MLP input shared by the gate and up projections, because it controls approximately `2 * d_model * d_ff` MACs. The gated MLP hidden before the down projection controls another `d_model * d_ff`. At very long contexts, structured sequence-level sparse attention can exceed both because it reduces the quadratic attention operations rather than only their input projections.

### 4. What Current Evidence Does Not Establish

The repository does not yet establish that:

- Naive L1 is generally harmless to convergence.
- Orthogonal pressure is unnecessary or inferior.
- L1 is intrinsically better than Ricker at matched sparsity or matched update magnitude.
- Density changes imply redistribution of functional computation.
- MLP sparsity causes attention to become functionally more important.
- Near-zero residual or attention values can be skipped by an actual kernel.
- High activation sparsity produces training, prefill, or decoding speedups.
- Results persist across seeds, datasets, model sizes, modern gated architectures, or longer training.
- The current methods improve downstream capabilities at matched training or inference compute.

### 5. Current Scientific and Measurement Confounds

#### 5.1 Site-weight dilution

The current pressure implementation first averages each captured tensor and then averages across captured tensors. Consequently, with the same nominal pressure weight:

- MLP-only gives the MLP family the full coefficient.
- MLP plus residual gives the MLP family approximately half the coefficient.
- All-site pressure gives the MLP family approximately one third of the coefficient.

Scope comparisons therefore do not hold MLP pressure fixed. The all-site results cannot yet establish compensation or synergy at matched local intervention strength.

#### 5.2 Cross-site threshold comparability

An absolute threshold such as `|a| <= 0.01` does not have the same meaning for MLP hiddens, residual streams, normalized branch inputs, and attention outputs. These distributions have different scales, shapes, and downstream fan-out.

#### 5.3 Numerical density versus functional load

A denser attention output does not prove that attention performs more task-relevant work. Functional redistribution requires causal interventions such as branch ablation, activation patching, or loss/KL sensitivity.

#### 5.4 Numerical density versus executable compute

Continuous small weights and activations do not save compute unless the implementation skips them in a supported structure. Elementwise token-dependent sparsity may also become dense when masks are unioned across a batch or hardware tile.

#### 5.5 Orthogonal geometry

The current orthogonal method projects the pressure direction against the Adam momentum/preconditioned task direction, not the current task gradient. Orthogonality in Adam step space does not guarantee that the pressure correction is non-increasing for the current or held-out task loss.

#### 5.6 Pilot scale

The main evidence remains one-seed Pythia-14M pretraining on MiniPile for approximately one token-cache-equivalent pass, with random block sampling. This is appropriate for hypothesis generation, not a top-venue generalization claim.

### 6. Publication Readiness

| Area | Status | Missing evidence |
| --- | --- | --- |
| Reproducible harness | Strong | Preserve while adding only necessary metrics |
| Interesting pilot phenomenon | Partial | Multi-seed confirmation and saved dynamic-sparsity diagnostics |
| Optimization mechanism | Partial | Controlled decomposition of state isolation, projection, cap, and geometry |
| Causal compensation claim | Missing | Functional interventions and blocked escape-path experiments |
| Novel architecture | Missing | Plain post-normalization ReLU is prior art |
| Quality-compute advantage | Missing | Matched-compute comparisons against strong baselines |
| Systems result | Missing | Kernel integration and wall-clock measurements |
| Scale and generality | Missing | Modern gated models, larger scales, longer training, and downstream evaluation |
| Top-venue paper | Pre-paper stage | One major causal result plus one competitive method/systems result |

## Part II: Researcher Hypotheses

### Hypothesis 1: Why does naive L1 not hurt here?

Original intuition:

- Naive L1 appears as good as the baseline, reducing the visible advantage of orthogonalization.
- Orthogonalization was much clearer in another training pipeline.
- Ricker versus L1 also shows less separation than expected.

Verdict:

- The premise should be refined: naive L1 does hurt the early ReLU learning curve relative to orthogonal L1, but it catches up by the current endpoint.
- The difference between pipelines may arise from activation scale, loss normalization, bias use, model capacity, optimizer state, training duration, and the ratio of pressure to task updates.
- Nominal `lambda` values are not comparable across pipelines. Methods must be matched by realized exact sparsity and update magnitude.
- ReLU bias shifts and downstream weight amplification provide a plausible low-task-cost compensation mechanism.
- Orthogonal pressure should remain in the program, but its value must be demonstrated rather than assumed.

Research role:

> Direction 1 is an optimization-mechanism study and a possible supporting contribution. It should become the main contribution only if a corrected method reliably improves the quality-sparsity frontier across conflict regimes and model families.

### Hypothesis 2: Induce sparsity before attention and MLP projections

Original intuition:

- Insert ReLU after LayerNorm and pressure its output.
- Create exact zeros before QKV and MLP input projections.
- Move from sparsity after expensive computation toward sparsity that can prevent computation.

Verdict:

- The computational instinct is correct: expensive linear layers should receive a sparse input if the goal is to skip their work.
- The literal architecture is not novel. [ReLU Strikes Back](https://openreview.net/forum?id=osoWxY8q2E) inserted ReLU after LayerNorm/RMSNorm before attention and FFN projections as stage-2 relufication.
- [Q-Sparse](https://arxiv.org/abs/2407.10969) applies magnitude top-k before every linear projection, uses a straight-through estimator, and introduces block sparsity. Its ReLU ablation reports worse performance and declining sparsity in QKV/up/gate paths relative to top-k.
- [TEAL](https://openreview.net/forum?id=dGVZwyq5tV) applies input activation sparsity throughout the model, provides kernels, and reports real batch-1 decoding speedups.
- Plain ReLU removes negative coordinates rather than low-magnitude coordinates. This is a strong information bottleneck at residual width `d`, even when it is tolerable in an overcomplete FFN width.

Architectural clarification:

```text
x_dense remains on the residual path
s_att = SparseOperator(Norm_att(x_dense))
x_mid = x_dense + Attention(s_att)
s_mlp = SparseOperator(Norm_mlp(x_mid))
x_next = x_mid + MLP(s_mlp)
```

This produces a **sparse branch-read interface from a dense residual carrier**. It does not produce a sparse representation that remains sparse across layers:

- LayerNorm mean subtraction generally destroys inherited zeros.
- Dense QKV and MLP projections produce dense outputs.
- Attention mixing and residual addition densify representations.
- Sparse QKV inputs can reduce projection work, but they do not reduce `QK^T`, softmax, or `AV` work.

Research role:

> Direction 2 should become a globally budgeted, structured sparse-read architecture. ReLU after normalization is a mandatory baseline, not the proposed novelty.

### Hypothesis 3: Sparsity pressure causes cross-pathway compensation

Original intuition:

- Sparsifying the FFN may make residual and attention pathways less sparse.
- Layerwise activation and weight analysis may expose a redistribution mechanism missed by FFN-only work.
- Preventing this compensation may be necessary to translate local sparsity into global efficiency.

Verdict:

- This is the strongest scientific direction.
- Current density changes are suggestive but are not yet causal evidence of functional redistribution.
- Weight density is not the relevant efficiency object unless weights become exact and hardware-structured. Path norms, contribution-weighted activity, effective rank, and causal sensitivity are more informative.
- A close prior exists: [Sparsity Moves Computation](https://arxiv.org/abs/2605.09403) reports that sparse MoE routing shifts functional work from FFNs into attention in one-layer Transformers on algorithmic tasks. The open opportunity is to establish, quantify, and control this effect in deep pretrained language models.
- This hypothesis does not by itself explain why the very high FFN sparsity in *Sparser, Faster, Lighter Transformer Language Models* produces a much smaller whole-model speedup. Dense gate computation, dense attention, kernel overhead, and Amdahl's law already explain much of that gap. Compensation is more plausibly a barrier to further whole-model sparsification.

Research role:

> Direction 3 should provide the causal observation that motivates direction 2. It is potentially publishable as an analysis contribution only if it is demonstrated functionally, causally, across depth, and beyond the 14M pilot.

### Consolidated 3 -> 2 -> 1 Verdict

| Direction | Promise | Main risk | Required falsification |
| --- | --- | --- | --- |
| `3`: compensation | A non-local mechanism connecting sparsity training and Transformer computation | Density changes may be scale artifacts or non-functional | Show increased attention reliance and reduced FFN reliance at matched sparsity; block the attention escape route |
| `2`: global sparse reads | Converts the mechanism into a compute-relevant architecture | Crowded prior art and hardware-hostile masks | Beat ReLU-after-norm, Q-Sparse-style top-k, and uniform allocation on quality versus measured latency |
| `1`: orthogonal pressure | May protect task learning while enforcing a global budget | Current projection geometry may not protect task loss; endpoint gains may remain null | Isolate optimizer components and show a robust Pareto improvement at matched density/update ratio |

## Part III: Recommended 3 -> 2 -> 1 Research Program

### 1. Central Research Question

> When a Transformer is pressured to become sparse at one computational site, where does the displaced functional load go, and can a global hardware-aware activation budget force the model to allocate sparse computation efficiently across the whole network?

### 2. Candidate Paper Thesis

The target thesis, if supported, is:

> Local activation sparsification causes measurable redistribution of functional reliance across Transformer branches and depth. This compute leakage makes local activation density a poor predictor of end-to-end efficiency. A globally constrained, hardware-structured sparse-read Transformer controls this redistribution and improves the quality-versus-realized-latency frontier.

### 3. Candidate Contributions

#### Contribution C1: Causal cross-pathway redistribution

Establish a cross-site and cross-layer response map showing how interventions on MLP, attention, and branch-input sparsity change:

- Exact support density.
- Dynamic versus static sparsity.
- Branch contribution and ablation sensitivity.
- Attention statistics and residual update magnitude.
- Downstream path norms and parameter-scale compensation.
- Distribution of functional load across depth.

This contribution corresponds to direction 3.

#### Contribution C2: Global hardware-aware sparse-read architecture

Introduce a model/training objective that assigns a global compute budget across sparse branch inputs and other compute-relevant sites, using masks that an actual kernel can exploit.

A generic constrained objective is:

```text
minimize   E[L_task]
subject to sum_{layer,site} C_layer,site(mask_layer,site) <= B,
```

where `C` is a measured or kernel-derived execution cost, not raw near-zero mass.

Potential sparse operators to compare include:

- ReLU after normalization as a prior-art baseline.
- Shifted ReLU.
- Sign-preserving soft threshold: `sign(z) * ReLU(abs(z) - tau)`.
- Exact magnitude top-k.
- Grouped top-k.
- Block or `N:M` masks.
- Masks shared across Q/K/V and across MLP gate/up projections.
- Masks coherent over short token chunks when this improves tile or batch occupancy.

This contribution corresponds to direction 2. The sparse operator alone is unlikely to be novel; the global allocation, coupling-aware training, and measured execution frontier must carry the contribution.

#### Contribution C3: Conditional task-safe sparsity optimization

Determine when optimizer-state isolation and orthogonal pressure help enforce the global budget. A publishable contribution would require:

- Correctly defined task-safe geometry.
- A controlled decomposition of state isolation, preconditioning, projection, and trust caps.
- A phase diagram based on gradient conflict, pressure magnitude, model capacity, and training stage.
- Reproducible quality-sparsity or quality-latency gains across at least two meaningful regimes.

This contribution corresponds to direction 1. If it fails these conditions, it should remain an ablation rather than appear in the title or abstract.

### 4. Claims That Must Not Be Made Yet

- "Orthogonal pressure improves final language-model quality."
- "Naive L1 is harmless to convergence."
- "The Transformer conserves sparsity across pathways."
- "Denser attention outputs prove that attention performs more work."
- "ReLU after LayerNorm is a new architecture."
- "Elementwise activation zeros translate directly into GPU speedups."
- "The current Pythia-14M result predicts frontier-model behavior."
- "Cross-pathway compensation explains the existing speedup ceiling of prior FFN-only systems."

## Falsifiable Research Questions

### H3: Does local sparsification redistribute functional load?

**H3.1 Density response.** Pressuring one site changes activity distributions at unpressured sites after controlling for pressure magnitude and activation scale.

**Falsification:** cross-site effects disappear when site coefficients are held fixed, or they are inconsistent across seeds.

**H3.2 Functional response.** At matched MLP exact sparsity, the pressured model becomes less sensitive to FFN ablation and more sensitive to attention ablation than the dense baseline.

**Falsification:** density changes occur without a corresponding change in branch sensitivity, loss/KL contribution, or causal patching results.

**H3.3 Escape-route response.** Freezing or capacity-limiting attention increases the task cost of FFN sparsification if attention is an important compensation path.

**Falsification:** the quality-sparsity frontier is unchanged when the proposed escape route is blocked.

### H2: Can a global sparse-read budget prevent compute leakage?

**H2.1 Allocation.** A learned global allocation outperforms uniform per-layer/per-site sparsity at the same realized hardware cost.

**Falsification:** uniform top-k or a TEAL/Q-Sparse-style baseline matches or dominates the quality-cost frontier.

**H2.2 Structure.** Hardware-aligned group, block, or `N:M` masks retain sufficient quality to produce wall-clock gains.

**Falsification:** mask structure destroys the quality advantage or sparse execution remains slower than the dense baseline in the target regime.

**H2.3 Generality.** The allocation mechanism transfers across model size, evaluation data, and at least one modern gated architecture.

**Falsification:** the effect is specific to biased ReLU Pythia-14M or the MiniPile validation distribution.

### H1: When does orthogonal pressure help?

**H1.1 Conflict regime.** Orthogonal pressure improves early or final task learning when the pressure update is large and conflicts with a stable estimate of the task gradient.

**Falsification:** matched-update experiments show no benefit over state-isolated unprojected pressure or naive optimization.

**H1.2 Geometry.** Projection against a current, multi-batch, EMA, or held-out task gradient predicts actual loss changes better than projection against the Adam step direction.

**Falsification:** corrected geometries do not improve the one-step loss assay or downstream quality-sparsity frontier.

**H1.3 Schedule.** Orthogonal pressure is most useful early, after which naive pressure or task-only training can reach an equal or better endpoint.

**Falsification:** switching schedules provide no improvement over a single optimizer policy.

## Work Packages and Stage Gates

### WP0: Make the Pilot Scientifically Trustworthy

Objective: remove measurement and objective confounds before running more scope comparisons.

Required work:

- Add explicit site-family coefficients: `lambda_mlp`, `lambda_residual`, `lambda_attention`, and future pre-projection coefficients.
- Average within layers and then within site families so adding a site does not dilute existing coefficients.
- Log per-site task/pressure gradient norms, cosines, and applied update ratios.
- Add exact full-validation counters per site and layer.
- Add activation RMS, scale-normalized thresholds, and contribution-weighted activity such as `abs(a_i) * ||W[:, i]||` where well-defined.
- Save channel firing frequency, globally dead fraction, support turnover, per-token density, per-batch union density, and block/tile occupancy.
- Include MLP biases and paired up/down path norms in checkpoint diagnostics.
- Define deterministic paired data orders for method comparisons.
- Reproduce the highest-signal ReLU results over at least three seeds.

Deliverable:

- A saved measurement-validation experiment and one compact figure/table establishing that all site metrics and coefficients have the intended semantics.

Go/no-go gate:

- Do not interpret site-scope compensation unless the effect persists with fixed per-site coefficients and full-validation metrics.

### WP1: Establish or Falsify Causal Compensation

Objective: determine whether observed cross-site density changes correspond to redistribution of functional computation.

Core experiment set:

- Apply pressure to one site or one layer at a time.
- Construct a cross-site response matrix from intervention site to measured site.
- Measure time ordering: the targeted sparsification should precede the proposed off-target redistribution.
- Turn pressure on and off from a shared checkpoint to test whether the response is reversible.
- Compare MLP-only, attention-input-only, joint, and no-pressure conditions with the same initialization and data order.
- Run attention trainable versus frozen.
- Freeze the attention output projection separately.
- Compare narrowed and widened attention capacity where parameter matching is possible.
- Include dense narrow-FFN, static-mask, random-mask, shuffled-mask, and dynamic-mask controls.
- Include a scale-only regularizer that changes activation magnitude without explicitly targeting exact zeros.
- Measure validation loss/KL under FFN-output and attention-output ablations.
- Measure residual update norm relative to residual-stream norm.
- Use activation patching or another explicit causal intervention on selected layers.

Suggested response statistic:

```text
C[source -> target] = delta(functional_load_target) / delta(support_density_source).
```

The numerator must be a functional measure, not only numerical density.

Deliverable:

- A response heatmap plus causal branch-sensitivity figure across layers and seeds.

Go/no-go gate:

- Continue with compensation as the central thesis only if the functional shift is directionally consistent across seeds and is weakened or exposed when the proposed escape path is blocked.

### WP2: Prototype the Global Sparse-read Architecture

Objective: turn the mechanism into an intervention at compute-relevant interfaces.

Initial architecture:

```text
z_att = Norm_att(x)
s_att = SparseOperator_att(z_att)
q, k, v = QKV(s_att)
a = Attention(q, k, v)
x_mid = x + OutputProjection(a)

z_mlp = Norm_mlp(x_mid)
s_mlp = SparseOperator_mlp(z_mlp)
u, g = UpAndGate(s_mlp)
h = HiddenSparseOperator(Activation(g) * u)
x_next = x_mid + DownProjection(h)
```

Compute mapping:

| Sparse site | Relation to current hooks | Directly avoidable work | Work not avoided |
| --- | --- | --- | --- |
| Post-normalization attention input | New `attention_inputs` | Q, K, and V input-projection products | `QK^T`, softmax, and `AV` |
| Post-normalization MLP input | New `mlp_inputs` | Gate and up projections in a gated model; up projection in Pythia | Down projection unless the MLP hidden is also sparse |
| MLP hidden | Current `mlp_hiddens` | Down projection | Gate projection and any already-computed up projection |
| Attention context before output projection | Not the current `attention_outputs` hook | Attention output projection | Earlier attention computation |
| Complete attention-module output | Current `attention_outputs` | No material attention compute; only a following residual add | QKV, score/value attention, and output projection have already executed |
| Pre-normalization residual stream | Current `residual_streams` | No downstream projection in Pythia because LayerNorm densifies it | Branch normalization, QKV, attention, and MLP work |
| Token/head/block routing | Not currently implemented | Selected attention score/value blocks | Unselected dense projections unless jointly routed |
| Final normalized hidden state | Not currently targeted | Vocabulary projection / LM head | All preceding Transformer work |

Core comparisons:

- ReLU after normalization.
- Shifted ReLU.
- Sign-preserving threshold.
- Exact top-k with straight-through training.
- Block/group top-k.
- Fixed versus learned per-layer/site allocation.
- Local FFN-only versus global cost constraint.
- Naive versus state-isolated versus orthogonal budget enforcement.

Deliverable:

- A quality versus ideal-cost frontier on the pilot model, followed by a quality versus measured-latency frontier once a kernel-compatible mask is selected.

Go/no-go gate:

- Do not scale an operator that fails to beat strong prior-art baselines at matched realized density and quality on the pilot.

### WP3: Decompose and Correct Orthogonal Pressure

Objective: identify whether any part of the orthogonal method contributes independently of update magnitude or optimizer-state isolation.

Paired one-step assay from identical checkpoints and batches:

- Task-only AdamW.
- Naive joint objective.
- Task-only optimizer state plus unprojected pressure correction.
- Projection without trust cap.
- Projection with trust cap.
- Projection against current gradient.
- Projection against multi-batch, EMA, or held-out gradient.
- Standard constrained-optimization or gradient-surgery baselines with compatible semantics.

Measurements:

- Predicted first-order task-loss change.
- Actual task loss on the optimization batch.
- Actual loss on the next independent batches.
- Exact density change.
- Correction/task-step ratio.
- Optimizer-state divergence.

Additional ablations:

- Freeze or remove MLP input biases.
- Weight decay in `{0, 0.01, 0.1}`.
- Freeze or normalize down-projection columns.
- Orthogonal-early then naive, orthogonal-early then task-only, and constant-policy schedules.

Deliverable:

- An optimizer decomposition table and a phase diagram over conflict, update magnitude, and training stage.

Go/no-go gate:

- Promote orthogonal pressure to a paper contribution only if it dominates simpler state-isolated or naive alternatives on a matched quality-sparsity frontier in at least two regimes. Otherwise retain it as an informative negative result or ablation.

### WP4: Scale and Close the Systems Loop

Objective: demonstrate that the selected method survives modern architecture and produces real efficiency gains.

Scale ladder:

1. Pythia-14M for fast falsification and instrumentation.
2. A modern gated 100M-300M model for architecture selection and multi-seed experiments.
3. A 0.5B-1B confirmation model only after the smaller model passes the quality-cost gate.

`TODO:` select exact architectures, datasets, token budgets, and hardware after calibration. Do not claim them before configs exist.

Systems evaluation:

- Dense and sparse kernel microbenchmarks.
- End-to-end decoding latency at batch 1 and representative larger batches.
- Prefill latency across context lengths.
- Training step time if training acceleration is claimed.
- Memory traffic, peak memory, and energy where measurable.
- Packing, mask-selection, and dispatch overhead.
- Dense fallback behavior when sparse occupancy is unfavorable.
- Actual tile/block occupancy rather than elementwise density alone.

Deliverable:

- End-to-end quality-latency-memory frontiers on at least one target GPU, with clear decode and prefill regimes.

Go/no-go gate:

- A systems claim requires statistically stable wall-clock improvement after including all mask and packing overhead. Estimated FLOPs alone are insufficient.

### WP5: Build the Publication Evidence Package

Objective: turn the surviving mechanism and method into a review-resistant paper.

Required evidence:

- At least three seeds for decisions; use additional seeds for the final key claims when variance remains material.
- Matched initialization, data order, tokens, optimizer budget, and evaluation protocol.
- A tuned dense baseline.
- Strong contemporary sparsity baselines.
- Modern gated architecture evidence.
- Held-out language-model loss plus downstream evaluation appropriate to model scale.
- Confidence intervals or seed-level points on central figures.
- Actual wall-clock results for every efficiency claim.
- Saved configs/results for every paper table and figure.
- A limitations section covering mask irregularity, hardware specificity, training cost, and scale boundaries.

Deliverable:

- A complete paper map in which every claim points to saved experiments and every figure regenerates from repository artifacts.

## Experimental Evidence Contract

The following rules should govern all experiments intended to influence the paper:

1. Compare pressure methods at matched exact density and matched update magnitude, not only matched nominal weights.
2. Keep site coefficients explicit and unchanged when adding another site.
3. Separate exact zeros, near-zero mass, static dead channels, dynamic sparsity, and hardware-realizable sparsity.
4. Report per-layer and per-site results before aggregating.
5. Use full deterministic validation for endpoint distribution claims.
6. Save per-token, per-batch-union, and structured block occupancy for dynamic masks.
7. Distinguish functional load from activation magnitude through interventions.
8. Tune the dense baseline before interpreting small regularized improvements.
9. Use identical token and optimizer-step budgets for learning comparisons; separately report wall-clock training cost.
10. Evaluate decode, prefill, and training separately because their bottlenecks differ.
11. Require a config and saved artifacts for every experiment included in a decision table.
12. Treat single-seed differences of a few hundredths of validation loss as planning evidence only.

## Required Baselines and Novelty Boundary

| Baseline or prior work | What it already covers | What this project must add |
| --- | --- | --- |
| AdamW dense/ReLU baselines | Standard task training and natural ReLU sparsity | Properly tuned reference at every scale |
| Naive L1 and Ricker | Direct activation pressure | Matched-density/update comparisons and mechanism analysis |
| Current orthogonal variants | State-isolated Adam-step pressure correction | Correct geometry and reproducible advantage |
| [ReLU Strikes Back](https://openreview.net/forum?id=osoWxY8q2E) | ReLU replacement and ReLU after normalization before QKV/FFN projections | ReLU-after-norm must be treated as a baseline |
| [Q-Sparse](https://arxiv.org/abs/2407.10969) | Top-k before every linear projection, STE, block sparsity, scaling experiments | Better global allocation or quality-realized-compute frontier |
| [TEAL](https://openreview.net/forum?id=dGVZwyq5tV) | Training-free model-wide input sparsity, layer allocation, kernels, real decoding gains | Training-aware improvement that beats its quality-latency tradeoff |
| [Sparser, Faster, Lighter Transformer Language Models](https://arxiv.org/abs/2603.23198) | L1-induced gated-FFN hidden sparsity, custom kernels, measured gains, and FFN sparsity analysis across depth | Cross-pathway functional/parameter redistribution and compute beyond an FFN-hidden-only target |
| [Sparsity Moves Computation](https://arxiv.org/abs/2605.09403) | Causal FFN-to-attention redistribution in small one-layer algorithmic Transformers | Deep pretrained language models, layerwise response, and control of redistribution |
| [Spark Transformer](https://arxiv.org/abs/2506.06644) | Hard sparsity in FFN and attention with execution-oriented methods | Clear differentiation in allocation mechanism or training frontier |

Novelty boundary:

> `ReLU(Norm(x)) + L1`, layerwise density plots, or FFN-only activation pressure are not sufficient novel contributions. The paper must add causal evidence of non-local redistribution, a materially different global allocation/control mechanism, or a corrected optimization principle with strong empirical support.

Safe positioning sentence:

> Prior work induces sparsity locally or thresholds/routs activations model-wide and demonstrates efficiency, while controlled toy studies show that sparse FFN routing can redistribute computation to attention. What remains unestablished is whether continuous activation pressure in deep pretrained language models causes causal cross-pathway redistribution, how optimizer geometry affects that response, and whether a global hardware-weighted sparsity budget can prevent compute leakage and improve realized quality-latency.

`TODO:` rerun a formal literature audit before freezing the method and again before submission. This field is moving quickly.

## Publication Milestones

| Milestone | Evidence required | Status |
| --- | --- | --- |
| M0: trustworthy pilot | Fixed site coefficients, saved exact metrics, dynamic/static diagnostics, key three-seed reruns | Not complete |
| M1: causal mechanism | Cross-site response matrix, branch sensitivity, blocked escape-path result | Not started |
| M2: competitive method | Global sparse-read method beats local/uniform/prior-art baselines on quality-cost | Not started |
| M3: systems result | Kernel-compatible masks and end-to-end wall-clock gains | Not started |
| M4: scale/generalization | Modern gated 100M-300M evidence plus 0.5B-1B confirmation if justified | Not started |
| M5: submission package | Multi-seed claims, downstream evaluation, regenerated paper artifacts, limitations | Not started |

The project should not advance automatically from one milestone to the next. Each stage exists to avoid spending scale compute on a mechanism that has not survived inexpensive falsification.

## Immediate TODO Checklist

### Now: repair the evidence base

- `TODO:` implement explicit per-site-family pressure coefficients without changing existing result records.
- `TODO:` add tests for coefficient invariance when sites or layers are added.
- `TODO:` save the full-validation channel firing-frequency and dead-channel analysis.
- `TODO:` add per-token, batch-union, and block occupancy metrics.
- `TODO:` add MLP bias and paired up/down path-norm diagnostics.
- `TODO:` generate the saved early-learning-curve comparison for configs `77`, `80`, and `81`.
- `TODO:` run the paired one-step optimizer decomposition.
- `TODO:` run the MLP-bias freeze/removal experiment.
- `TODO:` repeat the key AdamW, naive L1, and orthogonal L1 ReLU runs over at least three seeds.

### Next: test the mechanism

- `TODO:` define functional-load metrics before launching new scope runs.
- `TODO:` run single-site and single-layer pressure interventions.
- `TODO:` build the cross-site response matrix.
- `TODO:` add FFN and attention branch-ablation evaluation.
- `TODO:` run attention trainable/frozen and output-projection-frozen controls.
- `TODO:` compare static, random, shuffled, and dynamic masks.
- `TODO:` decide whether the evidence supports the term "compensation" or only "coupling."

### Then: test the architectural response

- `TODO:` reproduce ReLU Strikes Back stage-2 relufication as a baseline.
- `TODO:` implement attention-input-only, MLP-input-only, and joint sparse-read variants.
- `TODO:` compare ReLU, shifted ReLU, sign-preserving threshold, top-k, and structured top-k at matched density.
- `TODO:` implement a global compute-budget controller with explicit per-site costs.
- `TODO:` compare uniform versus learned layer/site allocation.
- `TODO:` select a mask structure suitable for the target execution regime.

### Later: scale and publish

- `TODO:` select and calibrate a modern gated 100M-300M architecture.
- `TODO:` define the scale-appropriate training dataset, token budget, and downstream evaluation suite.
- `TODO:` integrate or implement the required sparse kernels.
- `TODO:` benchmark decoding, prefill, memory, and energy across relevant operating points.
- `TODO:` run the 0.5B-1B confirmation only after the smaller-scale gate passes.
- `TODO:` freeze paper claims, figures, and tables only after all key results are multi-seed and reproducible.

## Risks, Kill Criteria, and Fallback Contributions

### Risk 1: compensation disappears under controls

Kill criterion:

- Cross-site density changes vanish with fixed coefficients, or functional sensitivity does not move.

Response:

- Drop the compensation claim.
- Reframe the current effect as activation-distribution coupling or parameter-scale compensation.
- Continue only if the optimization or dynamic-routing result is independently strong.

### Risk 2: the architectural method loses to top-k or TEAL/Q-Sparse-style baselines

Kill criterion:

- The proposed operator or allocation is dominated at matched quality and measured latency.

Response:

- Do not claim a new efficient architecture.
- Preserve the causal analysis as a possible standalone paper if it is broad and strong.
- Use the winning prior-art sparsifier as the controlled intervention for studying redistribution.

### Risk 3: orthogonal pressure remains endpoint-neutral

Kill criterion:

- Corrected and matched experiments show no consistent advantage over state isolation or naive pressure.

Response:

- Demote orthogonal pressure to an ablation or negative result.
- Use the observed early protection to motivate scheduling only if it yields an actual frontier improvement.

### Risk 4: elementwise dynamic sparsity does not map to hardware

Kill criterion:

- Batch/tile union density is high, or sparse kernels remain slower after overhead.

Response:

- Move to structured or chunk-coherent masks.
- Narrow the systems claim to the regime where speedups are measured.
- Do not substitute estimated FLOPs for latency.

### Risk 5: the phenomenon is specific to Pythia-14M

Kill criterion:

- The effect reverses or disappears in a modern gated 100M-300M model.

Response:

- Do not scale further.
- Treat the Pythia result as a small-model case study rather than a frontier-AI claim.

### Fallback Contribution Ladder

From strongest to weakest:

1. Causal redistribution plus a global method with measured speedups.
2. Causal redistribution across deep language models without a new winning method.
3. A corrected orthogonal-budget optimizer with a broad empirical phase diagram.
4. Dynamic-versus-static sparsity and parameter-scale compensation analysis.
5. A well-documented negative result showing why local activation density fails to predict global efficiency.

Only the first three are plausible main-paper directions for the stated top-venue ambition without substantially broader evidence.

## Target Paper Shape

### Candidate working title

**Compute Leakage in Sparse Transformers: Global Activation Budgets for Efficient Language Models**

This title is provisional and should change if causal compensation or the global method does not survive.

### Paper question

> Does local activation sparsity remove computation, or does a Transformer learn to move functional load into other paths?

### Target contribution structure

1. Discover and causally validate cross-pathway redistribution under localized activation sparsification.
2. Introduce a global, structured sparse-read budget that accounts for pathway and layer coupling.
3. Demonstrate improved quality versus realized latency, with orthogonal pressure included only if it contributes independently.

### Core figures

1. Cross-site and cross-layer response matrix.
2. Causal FFN/attention reliance under matched local sparsity.
3. Global sparse-read architecture and explicit compute accounting.
4. Quality versus measured latency for dense, local, uniform-global, and learned-global methods.
5. Dynamic/static mask statistics and batch/tile occupancy.
6. Scale and architecture transfer.

### Core tables

1. Main language-model quality and sparsity results across seeds.
2. Optimizer decomposition and conflict-regime analysis.
3. End-to-end decode/prefill latency, memory, and energy.
4. Comparison with ReLU-after-norm, Q-Sparse-style, TEAL-style, and FFN-hidden baselines.

### Venue positioning

- NeurIPS/ICLR: strongest fit if the work combines a surprising causal mechanism, a clear method, scaling, and rigorous empirical evidence.
- TNNLS: plausible if the optimization or architectural method is technically developed and evaluated broadly, even with a less systems-centered narrative.
- Systems-oriented venue: consider only if kernel design and hardware evaluation become the dominant contribution.

## Interpretation Boundary

The current repository supports a serious research agenda, not a finished paper claim. Its strongest value is that it has exposed three nontrivial leads:

1. Orthogonal pressure is active and appears to protect early learning, even though naive L1 catches up at the current endpoint.
2. ReLU L1 produces mostly dynamic token-dependent sparsity and exploits bias/path-scale adaptation rather than simply deleting neurons.
3. Local MLP sparsification is accompanied by measurable changes in other activation families, motivating a causal study of cross-pathway redistribution.

The immediate scientific priority is to validate lead 3 under controlled coefficients and causal interventions. Only then should the project commit to a global sparse-read architecture, and only after that architecture survives strong baselines should substantial scale or kernel work begin.
