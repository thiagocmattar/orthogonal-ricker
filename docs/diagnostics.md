# Diagnostics Contract

Diagnostics turn saved training artifacts into auditable measurements. This
document defines reusable metrics, aggregation, artifact schemas, and
interpretation boundaries. The definitive experiment plan decides which
diagnostics are required for each run.

## General Measurement Rules

- Evaluate only the exact config/run checkpoint named by the diagnostic input.
- Record source config ID, run ID, manifest status, checkpoint path, validation
  cache identity, and code provenance.
- Use deterministic validation ordering and document-disjoint partitions when
  the plan requires selection and confirmation data.
- Record integer numerators and denominators whenever a metric is a fraction.
- Pool counts first and divide once. Do not average per-batch or per-layer
  percentages unless that estimand is explicitly requested.
- Record the number of evaluated sequences, tokens, layers, seeds, and any
  excluded tail tokens or causal positions.
- Do not substitute training snapshots for full named diagnostics.

## Training and Validation Metrics

Every pretraining run records task loss independently of auxiliary pressure.
The standard event stream also carries the configured optimizer step, tokens
seen, learning rate, throughput, elapsed time, and validation measurements at
their configured cadence.

When pressure is enabled, record:

- `task_loss`;
- `pressure_loss`;
- `pressure_weight`;
- `weighted_pressure_loss`;
- `augmented_loss` for direct-loss methods.

For Adam-step orthogonal methods, `augmented_loss` is monitoring-only. AdamW
moments must still use task gradients alone.

Validation loss comparisons require the same checkpoint rule, token cache,
partition, token count, sequence length, precision policy, and evaluation
implementation. State every mismatch rather than presenting an unmatched
difference as a controlled effect.

## Gradient Diagnostics

At the same accumulated optimizer boundary, record task and pressure gradients
separately:

- task-gradient norm;
- raw pressure-gradient norm;
- pressure/task norm ratio;
- task-pressure dot product and cosine;
- conflict flag, defined by a negative dot product.

These raw-gradient metrics describe objective interaction before AdamW
preconditioning. They do not describe the final parameter displacement.

## Adam-Step Diagnostics

Orthogonal methods additionally record:

- task-direction norm;
- raw preconditioned pressure-direction norm;
- task-pressure dot product and cosine before projection;
- whether projection fired;
- dot product and cosine after projection;
- weighted raw pressure/task direction ratio;
- trust-budget scale;
- final pressure/task direction ratio;
- eligible and skipped parameter counts.

The interpretation must follow [`methods.md`](methods.md): the directions are
computed before per-group learning rates and exclude the separate decoupled
weight-decay displacement.

## Exact-zero and Near-zero Metrics

An exact zero is measured by direct floating-point equality:

```text
x == 0
```

Near-zero mass at threshold `epsilon` is:

```text
count(|x| <= epsilon) / count(x)
```

Always label `epsilon`. Exact zero and a near-zero threshold must never share a
name such as simply "sparsity." For a site/layer result, save both `zero_count`
or `threshold_hits` and `element_count`; derive the displayed percentage from
those integers.

If an architecture lacks a requested site, report `N/A`. Report `0%` only when
a compatible counter evaluated the site and observed a zero numerator.

## Activation Histograms

`activation_histograms.json` contains streamed distributions for explicitly
selected sites and source checkpoints. The artifact must record bin edges,
counts, underflow, overflow, total elements, threshold-hit counters, layer/site
identity, source identity, and validation coverage.

Exact-zero probability is a point mass, not a density bin. Plots must remove
the exact-zero count from the bin containing zero, display the atom separately,
and normalize the remaining density by the nonzero total. Never present a
bin-width-dependent spike as an exact-zero probability.

Histogram ranges must report underflow and overflow. A visually convenient
range does not authorize discarding mass silently.

## Weight Histograms

`weight_histograms.json` describes selected checkpoint tensors. Record the
exact parameter names or pattern, inclusion/exclusion rules for biases and
normalization parameters, bin edges, counts, out-of-range mass, total elements,
source run, and checkpoint.

Weight distributions are mechanism diagnostics. They do not by themselves
explain an activation effect or establish causality.

## Activation Propagation

`activation_propagation.json` follows exact zeros through configured activation
boundaries and counts logical scalar products with an exactly-zero activation
operand.

For Pythia causal language modeling, the supported logical-product accounting
can include:

- fused QKV projection;
- QK products over valid causal query/key pairs;
- PV products over valid causal query/key pairs;
- attention output projection;
- MLP up projection (`W1`);
- MLP down projection (`W2`).

Future causal-mask positions are excluded from QK, PV, and attention-core
denominators. The diagnostic uses the operands actually consumed by each
operation. In particular, PRE-RoPE query/key gate outputs cannot stand in for
post-RoPE QK operands.

Define the block opportunity as:

```text
R_block = zero-operand logical products in measured block operations
          / all logical products in those block operations
```

Define the model opportunity using the full declared model denominator,
including untargeted operations such as a dense output head when applicable:

```text
R_model = zero-operand logical products across measured model operations
          / all declared logical products in the model denominator
```

Both numerators and denominators depend on depth, width, sequence length,
attention implementation, vocabulary/output head, decoding regime, and gate
placement. Recompute them for the actual architecture; never reuse constants
from another model.

`R_block` and `R_model` are logical opportunities. Dense kernels still perform
the multiplications, so neither quantity is a measured speedup.

## Post-hoc Clipping Frontiers

`clipping_frontier.jsonl` stores one row per clipping setting. Each row records
the mode and cutoff, selected sites, validation loss, exact-zero counts, token
coverage, and optional logical-product counters.

Use the same sweep's zero-threshold row as the loss reference:

```text
delta_validation_loss(t) = validation_loss(t) - validation_loss(0)
```

This matters when clipping evaluation forces an implementation detail such as
eager attention to expose operands. Site-specific and joint clipping frontiers
answer different questions and must remain separate.

## Validation Reproducibility

When a plan divides validation documents into selection and confirmation
partitions, record:

- partition scheme and name;
- source-document count;
- partition seed;
- exact ordered document-index hash;
- token-cache metadata and hash;
- complete-block and excluded-tail token counts.

Training data-order schedules similarly record the schedule scheme, seed,
token-count and batch-shape inputs, plus a hash of the exact sampled block-start
array. A shared nominal seed without these identities is insufficient evidence
of matched data order.

## Standard Result Handoff

For a matched ReLU or threshold-gate comparison, the default compact table is:

| Method | Validation loss | `R_block` | `R_model` | `z_a` | `z_m` | `z_h` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

Here `z_a`, `z_m`, and `z_h` are pooled exact-zero fractions for attention
input, MLP input, and MLP hidden sites when those sites exist. Add explicitly
named columns for other active sites rather than remapping them silently.

Accompany the table with:

- architecture, method, gate, and pressure settings;
- exact config/run links and diagnostic artifact;
- validation-token and seed counts;
- matched deltas against the same-architecture optimizer control;
- matched orthogonal-versus-naive deltas when defined;
- uncertainty and evidence status;
- the statement that `R` values are logical opportunities, not measured
  speedups.

Display rounding must not alter saved values. Four decimals for validation loss
and two decimal places for percentages are suitable chat defaults, not storage
precision.

## Evidence Status

Use explicit evidence labels:

- `valid`: terminal artifacts and required diagnostics satisfy the reviewed
  plan and integrity checks;
- `provisional`: a precisely stated limitation permits only the named use;
- `invalid`: the artifact cannot support the intended comparison.

A scientifically unfavorable result can still be valid. An infrastructure
failure is not automatically scientifically unusable, but any provisional use
must identify the durable artifact and limitation without rewriting the source
manifest.
