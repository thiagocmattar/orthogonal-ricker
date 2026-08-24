# Experimental Plan v0.2 — Lean Review Draft

> **Status:** proposal for scientific review. This file is not launch
> authority. Only a reviewed [definitive plan](docs/experiment_plan.md) may
> authorize implementation-dependent scientific configs, calibration, or
> launch.

This plan is organized around the working paper *Sparsity Spillover in
Transformers: Local Pressure, Model-Wide Consequences*. All numerical entries
below are proposed design choices, not findings.

## 1. Paper goal

The study asks whether pressure at one feed-forward activation is accompanied
by a nonlocal activation response, whether that response changes the
model-wide logical opportunity created by exact zeros, and whether selected
multi-site thresholding plus conflict-aware pressure improves the
validation-loss/logical-opportunity frontier.

The experimental program has five questions:

1. **Sparsity spillover:** as L1 pressure at the FFN hidden site `h` increases,
   how do `h` and untargeted FFN/attention activation distributions change?
2. **OL1 robustness:** at the same pressure weights, does OL1 reduce
   validation-loss sensitivity while preserving the activation response?
3. **Threshold design:** how do canonical site variant, attention-threshold
   form, and threshold value change the validation-loss versus `R_model`
   frontier?
4. **Combined intervention:** does adding OL1 to the selected threshold
   frontier improve it relative to pressure-only and threshold-only controls?
5. **Scale replication:** do the spillover response and selected frontier
   transport from Pythia-14M to Pythia-70M and Pythia-410M without retuning the
   intervention?

Null, adverse, heterogeneous, and non-monotone results remain valid. The plan
does not assume that spillover, symmetric-threshold superiority, OL1 benefit,
or cross-size persistence will be observed.

### Interpretation boundaries

- L1 can increase near-zero mass without creating exact zeros.
- An opposing change at an untargeted site is an associated nonlocal response;
  it is not evidence of functional compensation.
- `R_model` and `R_model^max` are logical zero-product opportunities. They are
  not measured FLOP reduction, latency, energy savings, or speedup.
- The experimental and paper label is always
  **validation-loss/logical-opportunity frontier** unless a later experiment
  directly measures sparse-kernel compute or runtime.

## 2. Canonical vocabulary and interventions

An activation tensor that feeds a matrix multiplication and is eligible for
sparsification is a **sparsification site**. This plan uses only the canonical
repository aliases:

| Site | Exact operand location | Group |
| --- | --- | --- |
| `a` | Attention-branch LayerNorm output before the fused QKV projection | Attention |
| `m` | FFN-branch LayerNorm output before the `W1` up-projection | FFN |
| `h` | FFN activation-function output between `W1` and `W2` | FFN |
| `q_post` | Query after RoPE, immediately before `QK^T` | Attention |
| `k_post` | Key after RoPE, immediately before `QK^T` | Attention |
| `v` | Value before the attention-probability-by-value product | Attention |

Each alias denotes the operand actually passed into the downstream matrix
multiplication: the unmodified value when no intervention is active, or the
ReLU/threshold output at that location. At `h`, ReLU or a hard threshold
**replaces GELU as `mlp.act`**; it is not applied after GELU.

Activation pressure targets **only `h` in every transformer block**. Stage A2
and Stage A3 use topology `A1-H` with the explicit `relu` operator, matching
the paper’s GELU-to-ReLU FFN intervention. The stock-Pythia control is topology
`A0` with GELU and no pressure.

Always qualify stage names and topology IDs: write **Stage A2** for the
experiment stage and **topology `A2`** for the `{m, h}` placement.

### Pressure methods

| Human name | Config method | Definition |
| --- | --- | --- |
| No pressure | `none` | Task loss only |
| L1 | `l1_naive` | Add `lambda * mean(abs(h))` directly to task loss |
| OL1 | `orthogonal_l1` | Take the task-only AdamW step, remove a conflicting component from the preconditioned L1 direction when needed, cap the correction with `step_budget`, then apply it |

OL1 is conflict-aware rather than unconditionally orthogonal. L1 and OL1 use
the same `h` target and pressure-weight grid.

### B1 site variants

Only POST-RoPE query/key variants are studied. They are the actual operands
immediately preceding `QK^T`, so the PRE/POST placement comparison is removed.

| Topology ID | Active sites |
| --- | --- |
| `A1-H` | `h` |
| `A2` | `m, h` |
| `A3` | `a, m, h` |
| `A4-Q` | `a, m, h, q_post` |
| `A4-K` | `a, m, h, k_post` |
| `A4-V` | `a, m, h, v` |
| `A5-QK-POST` | `a, m, h, q_post, k_post` |
| `A6-POST` | `a, m, h, q_post, k_post, v` |

For every B1/B2 condition:

- one global `kappa` applies to every active site;
- active FFN sites (`m` and `h`) always use the one-sided rule
  `x -> x if x >= kappa, else 0`;
- active attention sites (`a`, `q_post`, `k_post`, and `v`) use either the
  one-sided rule or the symmetric rule
  `x -> x if abs(x) >= kappa, else 0`;
- equality survives both threshold rules.

This mixed-form design is a **proposed extension**, not a currently supported
repository realization. The present methods contract applies one
`model.site_gate` operator to every active port. The definitive plan must first
approve per-group threshold forms and amend both the schema/code and
`docs/methods.md`; the topology IDs themselves continue to mean only the active
site sets in the table above.

At `kappa = 0`, a one-sided threshold is ReLU-like but remains the explicit
threshold operator; its equality-gradient convention differs from the
repository’s explicit `relu` operator. A symmetric attention threshold at
`kappa = 0` is the identity, while a one-sided attention threshold at
`kappa = 0` removes negative values.

## 3. Shared training protocol

All models are constructed from Pythia configurations with random weights.
Released checkpoint weights are not loaded.

### Proposed reproducibility identities

The locally verified 14M inputs remain the proposed base identities; they are
not authoritative until ownership and licenses are reviewed.

| Item | Proposed identity |
| --- | --- |
| Dataset | `JeanKaddour/minipile` at revision `18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0` |
| Training scope | 1,000,000 source documents; 1,491,711,416 cached tokens |
| 14M architecture | `EleutherAI/pythia-14m-deduped` at revision `7386d9a4ae45aef494a6e704910394def3037fc5` |
| 14M tokenizer | `EleutherAI/pythia-14m-deduped` at the same revision |
| Tokenization | Append EOS; store `int32` token IDs |
| Training cache | `03-pythia-14m-minipile-random-full-10min`; SHA-256 `da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c` |
| Validation split | First 500 validation documents; `shuffled_source_documents_half_v1`; partition seed `20260718` |
| Selection partition | 152 complete sequences; ordered-index SHA-256 `ffc857a6f0771929dd75c93bc17729de98a692f3a175ac5742cc9d101ff4ea47`; token SHA-256 `22bb7c27864f0e5941548c572d6c75b1b5ba6a4c13e4cd26f40f4de546c5cc19` |
| Confirmation partition | 186 complete sequences; ordered-index SHA-256 `8953a93f85c80a48d25fcacb7a0fbf44f6d9fd5b54037f92e01c5250f045ad99`; token SHA-256 `ee777ebdb8672b676ecfc05b2e7024c2f9446f8a9e46ac22b78e8a6c36f0890b` |

`TODO:` pin the dataset split/text-column names and accepted dataset, model,
and tokenizer licenses. Also pin exact architecture/tokenizer revisions and
cache compatibility for 70M and 410M. All three sizes must use the same
tokenizer and train/validation token identities unless a reviewed amendment
declares and justifies a mismatch.

### Model, batch, and peak-LR grids

| Model | Global batch size | Peak learning-rate sweep |
| --- | ---: | ---: |
| Pythia-14M | **262,144 input tokens/update** | `{5e-4, 1e-3, 2e-3}` |
| Pythia-70M | **262,144 input tokens/update** | `{5e-4, 1e-3, 2e-3}` |
| Pythia-410M | **262,144 input tokens/update** | `{1.5e-4, 3e-4, 6e-4}` |

At sequence length 2,048, the global batch is 128 sequences per optimizer
update. Physical microbatch size may differ by model and hardware, but

    micro_batch_size * gradient_accumulation_steps = 128

must hold. The physical decomposition is frozen per model after memory
calibration and cannot change the global batch or token budget.

### Fixed optimization recipe

| Hyperparameter | Plan value |
| --- | ---: |
| Optimizer implementation | PyTorch AdamW; one parameter group over all trainable parameters |
| `beta_1` | 0.9 |
| `beta_2` | 0.95 |
| `epsilon` | 1e-8 |
| Weight decay | 0.1 |
| Global gradient-norm clipping | 1.0 |
| Hidden dropout | 0 |
| Attention dropout | 0 |
| Sequence length | 2,048 |
| Precision | BF16 CUDA autocast; FP32 parameters and AdamW state |

The released Pythia GPT-NeoX configs specify Adam with weight decay 0.1. This
PyTorch harness uses AdamW with weight decay 0.1 and records that implementation
choice explicitly. BF16 is fixed across all three model sizes instead of
Pythia’s original FP16 dynamic-loss-scaling recipe. **Peak learning rate is the
only optimizer hyperparameter tuned.**

Global norm clipping applies to the gradient consumed by AdamW: the task
gradient for no-pressure and OL1 runs, and the combined task-plus-pressure
gradient for L1 runs. OL1’s separately computed pressure direction is governed
by `step_budget` rather than this Pythia task-gradient clip.

### Learning-rate schedule

Every run linearly warms from zero to the configured peak learning rate
`eta_max` over the first 1% of optimizer updates, then cosine-decays over the
remaining 99% to `0.1 * eta_max`. The warmup fraction, decay shape, and
minimum-to-peak ratio are fixed across all models and stages.

The integer warmup is

    warmup_steps = ceil(0.01 * max_steps)

so the 400M-token schedule uses 16 warmup updates and the proposed
complete-block-pass-plus-wrap schedule uses 57. The final scheduled update
reaches `0.1 * eta_max`.

This follows the released Pythia recipe: 1% linear warmup followed by cosine
decay to 10% of the peak.

### Training budgets and data order

The proposed MiniPile cache contains 1,491,711,416 tokens. With 2,048-token
sequences, it contains 728,374 complete blocks plus a 1,464-token tail.

| Purpose | Updates | Processed input tokens | Exact rule |
| --- | ---: | ---: | --- |
| C1 LR screen | 1,526 | 400,031,744 | First 1,526 updates of the same-seed complete-block permutation |
| Complete-block pass + wrap | 5,691 | 1,491,861,504 | Visit every complete block once, then repeat the first 74 blocks of the seeded permutation to fill the final global batch; exclude the 1,464-token tail |

This is one complete-block pass plus a 0.010% deterministic wrap, chosen
instead of dropping 54 complete blocks or changing the final global batch. For each seed,
all model sizes and treatments share the same complete-block permutation and
wrap. The definitive plan must pin the new schedule scheme and its hash.

## 4. Evidence policy

### Seeds and matching

- Every complete LR, lambda, and B1 threshold grid first runs with seed 0.
- Independently of B1/B2, add seeds 1 and 2 for the predeclared central
  spillover contrast: ReLU-only (`lambda = 0`) versus L1 at `lambda = 1`, at
  14M, 70M, and 410M. `lambda = 1` is the fixed median of the five nonzero
  grid values, chosen before outcomes; the full lambda curves remain seed-0
  screens.
- After B2 selects one quality-oriented winning recipe, add seeds 1 and 2 for
  the six-condition component cohort at 14M: `A0`, ReLU-only, L1-only at
  `lambda_B2`, OL1-only at `lambda_B2`, threshold-only, and threshold+OL1.
  Repeat the same cohort at 70M and 410M in C2/C3, reusing identical runs.
- Full seed-0 curves are exploratory and labeled `n = 1`. Only the predeclared
  spillover contrast and selected six-condition cohort can support three-seed
  claims; show their individual seeds, mean, sample SD, and `n = 3`.
- No failed or unfavorable seed may be silently replaced.

Within a seed, treatment and control share model initialization, data order,
global batch, optimizer, schedule, token budget, validation partition, and
diagnostic checkpoint.

### Validation and checkpoints

- The existing document-disjoint selection partition is used for LR selection
  and every dependency choice. Evaluation batch size is 4 sequences; every
  complete cached block is evaluated once in fixed order.
- During training, evaluate the complete selection partition deterministically
  at update 1, every 191 updates, and the final update. This is approximately
  every 50.07M training tokens: 9 evaluations for a 1,526-update C1 run and 31
  for a 5,691-update complete-block-pass-plus-wrap run. Intermediate values are
  monitoring only; decisions use the final selection measurement under the
  stage-specific rule.
- The existing confirmation partition is opened once after A–C recipes and
  analyses are frozen. Released headline figures use confirmation measurements;
  selection/confirmation disagreement is reported without reselection.
- Confirmation uses a generic saved-checkpoint validation command over every
  complete confirmation block; training-time validation or a post-hoc
  threshold-sweep reference cannot substitute for it.
- Save the final model checkpoint only. Intermediate and optimizer-state
  checkpoints are not required because every scientific run is fresh.

### Spillover measurements

For `s in {h, a, m, q_post, k_post, v}`, save pooled, count-first:

- exact-zero fraction `z_s = count(x == 0) / count(x)`;
- primary near-zero mass `n_s(0.01) = count(abs(x) <= 0.01) / count(x)`;
- secondary `n_s(0.1)`;
- layerwise activation histograms with the exact-zero atom separate;
- pooled RMS as the predeclared scalar distribution-width measure.

The primary spillover vector is the same-seed change from the ReLU-only
`A1-H` control in `n_s(0.01)` and RMS at every named site. A paper statement
that an attention distribution “broadens” requires an RMS increase plus the
corresponding distribution plot; a near-zero change alone is insufficient.

### Logical-opportunity measurements

For `A0` and every ReLU or threshold condition used in a frontier or results
table, compute `R_model(z)` from integer counts of logical scalar products with
an exact-zero activation operand, using the actual post-RoPE Q/K operands. This
includes ReLU-only, L1-only, and OL1-only controls, not just hard-threshold
conditions. Also compute `R_model^max` for each declared
model/topology/sequence length by setting every active site to fully sparse in
the same validated denominator. Headline comparisons use `T = 2,048`.

### Failure policy

Rank or select only after every planned cell in the relevant grid has a
reviewed terminal classification. Nonfinite loss, invalid artifacts,
schedule/hash mismatch, or missing required diagnostics makes a run
ineligible. A reproducible scientific collapse is retained as a terminal
adverse cell and contributes no frontier point; an unresolved infrastructure
failure blocks the stage decision. Finite adverse results remain valid and are
reported.

For A1/C1, select the lowest-loss eligible LR only after all three cells are
classified; if none is eligible, stop every dependent stage for that model.
For A3, a `lambda_B2` candidate requires valid matched Stage A2 L1 and Stage A3
OL1 results at the same nonzero lambda. For B1, every conceptual four-kappa
family must be resolved before family selection. For B2, all four combined
kappa cells must be resolved; valid cells define the frontier, while terminal
scientific collapses are shown and cannot be transported. C3 can test frontier
persistence only when the complete frozen cohort and its controls are valid at
all three sizes.

Infrastructure retries preserve the failed attempt and rerun the same
immutable config only after the cause is documented. Never retry away a
scientific failure; any scientific change requires a new config and reviewed
plan amendment.

## 5. Experiment matrix

The dependency graph is:

    Stage A1 -> Stage A2 -> Stage A3
                    |
                    +----> Stage B1

    Stage A3 + Stage B1 -> Stage B2
    Stage C1 + Stage A2 -> Stage C2
    Stage C2 + Stage B2 -> Stage C3

### Phase A — 14M pressure study

All Phase A scientific runs use one complete-block MiniPile pass plus the
74-block fixed-batch wrap.

| Stage | Question | Grid and controls | Decision/output |
| --- | --- | --- | --- |
| **Stage A1 — LR selection** | Which peak LR should Pythia-14M use? | `A0`, no pressure, seed 0, peak LR `{5e-4, 1e-3, 2e-3}` | Freeze the eligible LR with lowest final selection loss; exact ties favor the lower LR. Report the complete tuning table; make no sparsity claim. |
| **Stage A2 — L1 spillover** | How does h-only L1 pressure change targeted and untargeted activations? | Topology `A1-H` + explicit ReLU; `l1_naive` at `h`; seed-0 `lambda in {0, 0.1, 0.5, 1, 2, 5}`; reuse selected-LR `A0`; add seeds 1/2 at `lambda in {0, 1}` | Produce the exploratory 14M lambda-response and the independent three-seed ReLU-versus-L1 spillover contrast. |
| **Stage A3 — OL1 robustness** | At the same lambda, does OL1 reduce quality sensitivity and which optimizer diagnostics accompany it? | Same topology and seed-0 lambda grid as Stage A2; `orthogonal_l1`; reuse `lambda = 0` control | Report matched L1/OL1 loss versus lambda and achieved `n_h(0.01)`, plus conflict/projection/trust diagnostics. Among nonzero lambdas with valid matched L1 and OL1 results, freeze `lambda_B2` from the OL1 loss/`n_h(0.01)` Pareto set: lowest loss, then higher `n_h(0.01)`, then lower lambda. |

`TODO:` freeze one OL1 `step_budget` before Stage A3. It is a scientific
parameter and cannot be selected by smoke testing.

Stage B1 proceeds even if Stage A2 finds no spillover, because the placement
grid is predeclared. It is therefore **spillover-motivated**, not
spillover-guided. A null Stage A2 result changes its interpretation to a
general multi-site threshold-placement study.

### Phase B — 14M threshold and combined frontier

All Phase B training uses the frozen 14M LR, one complete-block pass plus the
74-block fixed-batch wrap, and seed 0 for the complete grid.

| Stage | Question | Grid and controls | Decision/output |
| --- | --- | --- | --- |
| **Stage B1 — threshold ablation** | How do placement, attention-threshold form, and threshold value change the logical-opportunity frontier? | No pressure; eight site variants; `kappa in {0, 0.03, 0.10, 0.30}`; FFN sites always one-sided; attention sites one-sided or symmetric; shared `A0` control | Produce the full validation-loss/`R_model` frontier and select one topology + attention-form family for B2. |
| **Stage B2 — threshold + OL1** | Does OL1 improve the selected Stage B1 family? | Apply `orthogonal_l1` at the single frozen `lambda_B2` to every kappa of the selected Stage B1 family; target `h`; reuse threshold-only Stage B1, L1-only Stage A2, OL1-only Stage A3, ReLU-only, and `A0` controls | Freeze the nondominated combined kappa points for Stage C3 and one quality-oriented winning recipe for the three-seed, six-condition component comparison. |

B1 has 56 conceptual threshold cells before the shared `A0` control:

- `A1-H` and topology `A2` have no attention-form factor:
  `2 topologies * 4 kappa = 8`;
- topology `A3` through `A6-POST` comprise six variants with two attention
  forms: `6 * 2 * 4 = 48`.

The six symmetric-attention cells at `kappa = 0` are functionally identical to
topology `A2` with a one-sided threshold at `kappa = 0`: their attention
operators are identities and `m/h` have the same threshold. Train that anchor
once and reuse it for those six cells. Thus B1 requires **50 unique threshold
runs**, not 56, and approximately **74.59B processed input tokens** at seed 0.
The reused result appears once in the global evidence set and never gains
weight from being represented in multiple conceptual families.

The Stage B1 family-selection rule is:

1. After all 50 unique runs have terminal classifications and every finite run
   has valid required diagnostics, form the global Pareto set over `A0` and
   unique valid B1 points, minimizing final selection loss and maximizing
   `R_model`. A terminal scientific collapse is a resolved, dominated cell.
2. A family is one topology plus one attention form; the form is `N/A` for
   `A1-H` and topology `A2`.
3. Each symmetric-attention family uses the single reused `A2`, `kappa = 0`
   result only as a plotted common baseline. To keep family scores comparable,
   family selection excludes `kappa = 0` for **every** family and uses only the
   three positive-kappa points.
4. Select the family with the most positive-kappa global-frontier points. Break
   ties by larger `R_model` span and then lower mean loss over those frontier
   points, followed by the topology-table order above, with symmetric before
   one-sided as the final tie-break.
5. If no positive-kappa threshold point is on the global frontier, report a
   null B1 result and stop B2/C3.

Stage B2 evaluates all four kappa values in the selected family. Its combined
Pareto points are frozen for Stage C3. The single quality-oriented winner is the
valid combined point on that Pareto set with lowest validation loss and
`R_model > 0`; ties favor higher `R_model` and then lower kappa. If A3 has no
valid matched nonzero OL1 Pareto point, Stage B2/C3 stop. If B2 has no valid
combined point with `R_model > 0`, report a null combined result and stop C3.

The plan does not include a threshold+L1 condition. Therefore Stage B2
supports the combined threshold+OL1 comparison against its components, but it
does not isolate OL1-versus-L1 behavior inside the selected threshold
architecture.
If the selected recipe has no nonidentity attention threshold (`A1-H`,
topology `A2`, or a symmetric-attention `kappa = 0` point), B2/C3 support only
an FFN threshold+OL1 result. Otherwise it may be described as a selected
multi-site threshold+OL1 recipe **containing** an attention threshold. Because
the plan has no matched FFN-threshold+OL1 version of that recipe, it cannot
attribute the combined improvement specifically to the attention component.

### Phase C — 70M and 410M replication

Only Pythia-70M and Pythia-410M are scaled. The intervention grids are not
retuned.

| Stage | Question | Grid and controls | Decision/output |
| --- | --- | --- | --- |
| **Stage C1 — per-size LR selection** | Which peak LR should each larger model use? | `A0`, no pressure, seed 0, 400,031,744 tokens; 70M grid `{5e-4, 1e-3, 2e-3}`; 410M grid `{1.5e-4, 3e-4, 6e-4}` | Freeze the lowest final selection-loss LR independently for each model; exact ties favor the lower LR. |
| **Stage C2 — spillover replication** | Does the Stage A2 L1 spillover response recur with scale? | At each frozen LR and one complete-block pass plus wrap: `A0` plus topology `A1-H` + ReLU with the unchanged Stage A2 lambda grid and h-only L1 pressure; add seeds 1/2 at `lambda in {0, 1}` | Report the exploratory seed-0 response at 14M/70M/410M and the predeclared three-seed central contrast. No lambda retuning by size. |
| **Stage C3 — frontier replication** | Does the selected Stage B2 frontier persist without intervention retuning? | At each frozen LR and one complete-block pass plus wrap: transport the Stage B2 topology, attention form, kappa cohort, `lambda_B2`, and OL1 `step_budget` literally. Run OL1-only, threshold-only, and threshold+OL1 matched components; reuse identical Stage C2 `A0`, ReLU-only, and L1-only runs and add any missing component seeds. | Report the cross-size validation-loss/`R_model` frontier. The Stage B2 winner and matched six-condition cohort receive seeds 1 and 2; the complete transported frontier remains seed 0 and exploratory. |

An absolute kappa or lambda may not have the same achieved effect at every
scale. That is the replication question; do not retune it away.

## 6. Paper outputs

The minimum main-paper package is three figures and one compact results table.

| Output | Evidence and required content |
| --- | --- |
| **Figure 1 — sparsity spillover** | Stage A2/C2 seed-0 lambda response for `h, a, m, q_post, k_post, v`; the three-seed `lambda = 0` versus `lambda = 1` contrast; paired changes in `n_s(0.01)` and RMS; layer structure; 14M/70M/410M comparison. The distribution panel is fixed in advance: Pythia-14M, deepest transformer layer, all six named sites as small multiples, `lambda = 0` versus `lambda = 1`, showing the three per-seed distributions and the exact-zero atom separately. |
| **Figure 2 — model-wide logical opportunity** | Validated `R_model` accounting by operation for ReLU and threshold conditions; `R_model^max` by model and Stage B1 topology at `T = 2,048`; observed Stage B1/B2/C3 operation-level contributions |
| **Figure 3 — intervention and mechanism** | Validation-loss/`R_model` views for `A0`, ReLU-only, L1-only at `lambda_B2`, OL1-only at `lambda_B2`, threshold-only, and threshold+OL1; Stage B1 and C3 frontiers; matched loss versus achieved `n_h(0.01)`; OL1 conflict and projection frequencies. Include a paired threshold-form panel comparing one-sided versus symmetric attention at every applicable topology and kappa for the same seed, with arrows for paired changes in loss and `R_model`. Label every curve/contrast as seed-0 exploratory or three-seed confirmed. There is no threshold+L1 condition, so this figure cannot claim OL1 beats L1 within the selected threshold architecture. |
| **Main results table** | One row per model and final matched condition: complete recipe, validation loss and paired change, `n_h(0.01)`, named spillover vector, absolute and paired `R_model`, seeds, mean/sample SD, and evidence status |

Appendix outputs:

- complete Stage A1/C1 LR tables and frozen rates;
- all Stage A2/A3 lambda and Stage B1/B2 kappa points, including dominated, adverse,
  failed, and invalid cases;
- full OL1 cosine, trust-scale, correction-ratio, and layerwise mechanism
  diagnostics beyond the main conflict/projection summaries;
- full per-site/layer distributions and `n_s(0.1)` sensitivity;
- formulas, integer counts, denominators, and coverage for `R_model` and
  `R_model^max`;
- selection-versus-confirmation comparison.

## 7. Implementation and review blockers

The simplified design intentionally requires several reviewed implementation
changes. Do not create configs until they are complete and tested.

1. **Complete-block sampler:** add the seeded without-replacement complete-block
   schedule with the 74-block deterministic wrap and schedule hash.
2. **Cosine schedule:** replace the current warmup-then-constant path with the
   exact 1% warmup/cosine-to-0.1 schedule and test endpoints.
3. **Gradient clipping:** add global norm clipping at 1.0 and record pre/post
   clip norms; test the L1/OL1 ordering specified above.
4. **Mixed threshold forms:** B1 requires one-sided FFN thresholds and an
   independently selected attention threshold form under one global kappa.
   The current single-operator `model.site_gate` schema cannot express this;
   amend the scientific contract in `docs/methods.md` as well as schema/code.
5. **Logical ceiling:** amend `docs/diagnostics.md` to define the serialized
   numerator, denominator, coverage, and model/topology/workload identity for
   `R_model^max`; implement and validate it for each actual case rather than
   reusing historical atlas values.
6. **Spillover width and histograms:** extend the named activation diagnostic
   with exact pooled second-moment/RMS accumulation and coverage.
   Freeze histogram bins/range before Stage A2.
7. **Reproducibility identities:** verify the proposed 14M pins and freeze the
   missing split/text-column and licenses; pin exact 70M/410M architecture and
   tokenizer revisions, cache compatibility, model-specific microbatch and
   accumulation, and all licenses.
8. **OL1:** freeze `step_budget` before Stage A3 and aggregate conflict,
   projection, and eligible-boundary numerators/denominators over every
   optimizer boundary, not only logging events.
9. **Validation:** implement/test the exact 191-update cadence and a generic
   saved-checkpoint confirmation command with exact source, partition,
   complete-block coverage, excluded-tail, and provenance checks.

After those items are incorporated into a reviewed
`docs/experiment_plan.md`, calibrate each model and each materially different
workload on the launch hardware. Use a 600-second production-shaped training
window plus separately timed setup, full validation, diagnostics, and final
checkpoint publication. Report first-run and tranche ETCs, projected
completion time, assumptions, and uncertainty before seeking launch approval.
Execution, monitoring, failure recovery, and immutable tranche ownership follow
[the runbook](docs/runbook.md) and [experiment scaffold contract](experiments/README.md).

## 8. Manuscript alignment and retained rationale

This program validates `R_model^max` only for Pythia-14M, 70M, and 410M at
`T = 2,048`. The current introduction's numerical examples for a 12B model and
`T = 50,000` are outside this plan and must be removed/reframed unless a
separately reviewed analytical scope validates them. The manuscript must also:

- describe B1 as spillover-motivated, because its sites are predeclared rather
  than selected from the Stage A2 outcome;
- if the winner contains a nonidentity attention threshold, describe the
  selected multi-site threshold+OL1 recipe as containing that intervention;
  do not attribute its improvement specifically to attention without adding a
  matched FFN-threshold+OL1 control;
- label full lambda/kappa/frontier curves as exploratory seed-0 evidence and
  reserve robustness language for the predeclared three-seed contrasts;
- treat directional statements that OL1 reduces sensitivity to lambda or that
  symmetric attention thresholds preserve more quality while one-sided
  thresholds expose more logical opportunity as exploratory unless those
  complete curves receive additional seeds;
- call `R_model` a logical opportunity, never removed/avoided compute or
  measured speedup.

Retain the following justification for later manuscript drafting, not as an
experimental result:

> We fix the global batch size across model scales and tune only the peak
> learning rate independently for each model. At a fixed training-token budget,
> changing batch size changes the number of optimizer updates and would
> therefore confound model scale with optimization trajectory; fixing the batch
> ensures that all models process the same number of tokens with the same
> number of updates. This follows the controlled-scaling philosophy of
> [Pythia](https://proceedings.mlr.press/v202/biderman23a.html) while using a
> smaller batch appropriate for the substantially smaller
> [MiniPile](https://arxiv.org/abs/2304.08442) corpus. We retune learning rate
> because optimal optimization hyperparameters are scale-dependent, with
> empirical scaling studies showing that the preferred learning rate decreases
> as training compute increases
> ([DeepSeek-AI, 2024](https://arxiv.org/abs/2401.02954)).

> We use 1% linear warmup followed by cosine decay to 10% of the peak learning
> rate, matching the established Pythia pretraining schedule. The schedule
> shape is fixed across model scales; only the peak learning rate is tuned.
