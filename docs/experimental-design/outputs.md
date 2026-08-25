# Evidence and Paper Outputs

> Metric mathematics and artifact schemas are authoritative in
> [`../diagnostics.md`](../diagnostics.md). This file owns which evidence enters
> the paper and what it may support.

## Evidence Levels

- Complete grids use the catalog's `screen` seed set and are exploratory.
- The fixed ReLU-only versus L1 spillover contrast and selected six-component
  winner use the catalog's `confirmed` seed set at all three model sizes.
- Show every individual seed plus mean, sample standard deviation, and `n`.
  Never replace a failed or unfavorable seed or compare unequal seed sets as a
  confirmed contrast.
- Selection/confirmation disagreement is reported without reranking,
  retuning, or reopening the confirmation partition.

## Learning-Rate Screens

Report A1 as a complete four-cell table and curve with peak LR, final selection
loss, terminal classification, eligibility, exact config/run identity, and the
selected rate. Preserve training/validation curves and timing/resource
telemetry for every cell. If `4e-3` is selected, label it as the upper tested
boundary. A1 supports only a best-tested-rate decision at the fixed 400M-token
horizon; it does not support a sparsity, convergence, full-pass, or
horizon-independent optimum claim.

## Spillover Measurements

For each `s` in `{h, a, m, q_post, k_post, v}`, save pooled count-first:

- exact-zero fraction `z_s`;
- primary near-zero mass `n_s(0.01)`;
- secondary sensitivity `n_s(0.1)`;
- pooled RMS;
- layerwise histograms with the exact-zero atom separate.

The primary spillover vector is the within-seed change from the ReLU-only
`A1-H` control in `n_s(0.01)` and RMS at every named site. A statement that a
distribution broadens requires an RMS increase and the corresponding
distribution panel; near-zero change alone is insufficient. An opposing
untargeted response is not evidence of functional compensation.

## Logical Opportunity

For `A0` and every ReLU/threshold condition used in a frontier or table,
compute `R_model` from integer zero-operand logical-product counts using the
actual post-RoPE Q/K operands. Compute `R_model^max` under the same declared
denominator for each model, topology, and `T = 2,048` workload.

`R_model` and `R_model^max` are logical opportunities, not removed FLOPs,
latency, energy savings, or measured speedup. The paper axis is always
**validation-loss/logical-opportunity frontier** unless a later reviewed
experiment directly measures sparse-kernel execution.

## Main-Paper Package

| Output | Required content |
| --- | --- |
| Figure 1 — sparsity spillover | Seed-0 A2/C2 lambda responses; three-seed fixed contrast; site/layer changes in near-zero mass and RMS across 14M/70M/410M. The distribution panel is fixed before results: 14M, deepest transformer layer, all six named sites, ReLU-only versus the fixed L1 contrast, with all three seeds and exact-zero atoms. |
| Figure 2 — model-wide logical opportunity | Validated operation-level `R_model`; `R_model^max` by actual model/topology at `T = 2,048`; observed B1/B2/C3 contributions. |
| Figure 3 — intervention and mechanism | Six matched components; B1/C3 frontiers; loss versus achieved `n_h(0.01)`; OL1 conflict/projection frequencies; paired one-sided-versus-symmetric points at every applicable matched topology/kappa/seed. |
| Main results table | One row per model and final matched condition: complete recipe, validation loss and paired change, `n_h(0.01)`, named spillover vector, absolute/paired `R_model`, seeds, mean/sample SD, and evidence status. |

Figure 3 cannot claim OL1 beats L1 inside the selected threshold topology
because no threshold+L1 case is planned. Paired threshold-form and complete
lambda-sensitivity curves remain single-seed directional evidence unless the
case catalog is reviewed to add seeds.

## Appendix Package

- complete four-cell A1 and complete C1 tuning tables and frozen rates;
- every lambda/kappa point, including dominated, adverse, failed, and invalid
  cases;
- full OL1 cosine, trust-scale, correction-ratio, and layerwise diagnostics;
- full per-site/layer distributions and `n_s(0.1)` sensitivity;
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
