# Evidence and Paper Outputs

> Metric mathematics and artifact schemas are authoritative in
> [`../diagnostics.md`](../diagnostics.md). This file owns which evidence enters
> the paper and what it may support.

## Evidence Levels

- Complete grids use the catalog's `screen` seed set and are exploratory.
- A2 and C2 spillover responses currently use seed 0 only. They are
  directional evidence and carry no seed-uncertainty claim.
- The selected six-component winner uses the catalog's `confirmed` seed set.
  Show every individual seed plus mean, sample standard deviation, and `n`;
  never replace a failed or unfavorable seed.
- If the final winner cohorts supply matching added-seed ReLU/L1 components,
  they may also report the deferred selected-lambda spillover replication.
  This does not constitute replication of the full lambda response.
- Selection/confirmation disagreement is reported without reranking,
  retuning, or reopening the confirmation partition.

## Learning-Rate Screens

Report A1 as a complete eleven-cell table and curve with peak LR, final selection
loss, terminal classification, eligibility, exact config/run identity, and the
selected rate. Report a two-panel progress figure with validation loss and the
logged effective learning rate against cumulative training tokens for every
cell. Preserve timing/resource telemetry for every cell. If `5.12e-1` is
selected, label it as the upper tested boundary. A1 supports only a
best-tested-rate decision at the fixed 400M-token horizon; it does not support a
sparsity, convergence, full-pass, or horizon-independent optimum claim.

## Spillover Measurements

For each `s` in `{h, a, m, q_post, k_post, v}`, derive count-first from the
saved per-layer rows in `activation_histograms.json`:

- exact-zero fraction `z_s`;
- primary near-zero mass `n_s(0.01)`;
- secondary sensitivity `n_s(0.1)`;
- per-layer RMS and, when a compact site summary is needed, count-weighted
  pooled RMS;
- layerwise histograms with the exact-zero atom separate.

The current artifact schema already contains the integer threshold hits,
totals, finite counts, per-layer RMS values, histogram counts, underflow, and
overflow needed for these reductions; A2 requires no diagnostic-schema or
training change. The primary spillover vector is the within-seed change from
the ReLU-only `A1-H` control in `n_s(0.01)` at every named site. RMS is a
secondary activation-scale diagnostic, not the definition of spillover. A
statement that a distribution broadens requires an RMS increase and the
corresponding distribution panel; near-zero change alone is insufficient. An
opposing untargeted response is not evidence of functional compensation.

## Logical Opportunity

For `A0` and every ReLU/threshold condition used in a frontier or table,
compute `R_model` from integer zero-operand logical-product counts using the
actual post-RoPE Q/K operands. Compute `R_model^max` under the same declared
denominator for each model, topology, and `T = 2,048` workload.

`R_model` and `R_model^max` are logical opportunities, not removed FLOPs,
latency, energy savings, or measured speedup. The paper axis is always
**validation-loss/logical-opportunity frontier** unless a later reviewed
experiment directly measures sparse-kernel execution.

### Proposed A2 post-hoc clipping frontier

For each accepted A2 checkpoint and each common cutoff
`t = {0, 0.01, 0.03, 0.10, 0.30}`, jointly clip
`[a, m, h, q_post, k_post, v]` and report validation loss against observed
`R_model`. Use only each checkpoint's same-sweep `t = 0` row for paired loss
and `R_model` changes. The fixed-cohort figure contains an absolute panel and a
within-checkpoint delta panel; its companion table retains all 30 points and
the exact per-site/per-operation integer counts. Full details and limitations
are in [`a2-clipping-review-packet.md`](a2-clipping-review-packet.md).

## Main-Paper Package

| Output | Required content |
| --- | --- |
| Figure 1 — sparsity spillover | Seed-0 A2/C2 responses at `pressure: none` and lambda `{0.1, 0.5, 1, 2, 5}`; site/layer changes in near-zero mass and RMS across 14M/70M/410M. The required 14M distribution panel uses seed 0, the deepest transformer layer, all six named sites, ReLU-only versus lambda `1`, and separate exact-zero atoms. If final winner cohorts provide matching added seeds, a selected-lambda replication panel may be added without replacing the seed-0 response. |
| Figure 2 — model-wide logical opportunity | Validated operation-level `R_model`; `R_model^max` by actual model/topology at `T = 2,048`; observed B1/B2/C3 contributions. |
| Figure 3 — intervention and mechanism | Six matched components; B1/C3 frontiers; loss versus achieved `n_h(0.01)`; OL1 conflict/projection frequencies; paired one-sided-versus-symmetric points at every applicable matched topology/kappa/seed. |
| Main results table | One row per model and final matched condition: complete recipe, validation loss and paired change, `n_h(0.01)`, named spillover vector, absolute/paired `R_model`, seeds, uncertainty when replicated, and evidence status. |

Figure 3 cannot claim OL1 beats L1 inside the selected threshold topology
because no threshold+L1 case is planned. Paired threshold-form and complete
lambda-sensitivity curves remain single-seed directional evidence unless the
case catalog is reviewed to add seeds.

## Appendix Package

- complete eleven-cell A1 and complete C1 tuning tables and frozen rates;
- every lambda/kappa point, including dominated, adverse, failed, and invalid
  cases;
- full OL1 cosine, trust-scale, correction-ratio, and layerwise diagnostics;
- full per-site/layer distributions and `n_s(0.1)` sensitivity;
- the complete A2 post-hoc clipping grid, including dominated and adverse
  points, if its proposed design is reviewed and executed;
- `R_model`/`R_model^max` formulas, integer counts, denominators, and coverage;
- selection-versus-confirmation comparisons;
- exact config/run/diagnostic identities and regeneration commands.

## Release Boundary

Every main output must be regenerable from exact source identities in
[`../paper_map.md`](../paper_map.md). The current introduction's 12B-model and
`T = 50,000` ceiling examples are outside this program and must be removed or
supported by a separately reviewed analytical scope. B1 is
spillover-motivated, not selected from observed spillover. If the final recipe
contains a nonidentity attention threshold, say that it **contains** attention
thresholding; do not attribute its benefit specifically to attention without a
matched FFN-threshold+OL1 control.
