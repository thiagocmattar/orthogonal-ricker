# Methods Contract

This document defines the reusable scientific behavior implemented by the
harness. It does not define the definitive experiment matrix, budgets, datasets,
seeds, or paper claims. Those belong only in the reviewed
[`experiment_plan.md`](experiment_plan.md).

## Model Construction

For a Pythia pretraining config, `model.architecture` names the Hugging Face
architecture/config source and `model.initialization: random` requires:

```python
config = AutoConfig.from_pretrained(model_architecture)
model = AutoModelForCausalLM.from_config(config)
```

Released checkpoint weights are not loaded. Loading them changes the scientific
task to continuation or fine-tuning and must be named and evaluated as such.

Randomly initialized model parameters are kept in FP32. A config may select a
lower-precision autocast mode for computation, but this is not permission to
store newly initialized parameters in a numerically unsafe dtype.

Pythia/GPT-NeoX uses parallel attention and MLP residual branches. For block
input `H_l`, both branches consume normalized views of the same input and the
block produces a residual sum. A named topology can gate branch inputs, the MLP
hidden activation, or Q/K/V ports; the final LayerNorm remains unchanged unless
a future reviewed plan and implementation explicitly add another path.

## Transformer Sites

A transformer site is an input-dependent tensor whose matrix dimensions feed a
large downstream matrix multiplication. Batch, head, and token axes may appear
as leading dimensions, but the alias denotes the exact operand port rather than
a broad conceptual region. These eight aliases are the complete supported
vocabulary:

| Alias | Exact tensor port | Shape | Large downstream matrix multiplication |
| --- | --- | --- | --- |
| `a` | Attention-branch LayerNorm output, or the gate output at that port | `[batch, seq, hidden]` | Fused `W_QKV` projection |
| `m` | MLP-branch LayerNorm output, or the gate output at that port | `[batch, seq, hidden]` | MLP `W1` up projection |
| `h` | Output of `gpt_neox.layers.N.mlp.act`: stock GELU or the configured gate | `[batch, seq, intermediate]` | MLP `W2` down projection |
| `q_pre` | Query port immediately before partial RoPE | `[batch, heads, tokens, head_width]` | `QK^T`, after partial RoPE |
| `k_pre` | Key port immediately before partial RoPE | `[batch, heads, tokens, head_width]` | `QK^T`, after partial RoPE |
| `q_post` | Query port immediately after partial RoPE | `[batch, heads, tokens, head_width]` | `QK^T` |
| `k_post` | Key port immediately after partial RoPE | `[batch, heads, tokens, head_width]` | `QK^T` |
| `v` | Value port from the fused QKV projection | `[batch, heads, tokens, head_width]` | `PV` attention-context product |

Aliases are exact in configs, artifacts, metrics, plots, and prose. Do not use
the retired broad names `mlp_hiddens`, `attention_inputs`, `mlp_inputs`,
`query_gate_outputs`, `key_gate_outputs`, `value_gate_outputs`,
`residual_streams`, or `attention_outputs` as synonyms.
For PRE-RoPE sites, captured gate-output zeros and actual QK-operand zeros are
different: RoPE can rotate a sparse pair into dense coordinates. Diagnostics
that claim QK opportunity must count the actual post-RoPE operands.

There is no default activation-pressure target. The reviewed plan and every
config must name each pressure or diagnostic site explicitly; an alias never
broadens silently.

## Activation Topologies

A topology ID specifies only the ports that have active site gates. It does not
specify the site-gate operator or `kappa`, optimizer, activation-pressure
method, pressure target, or pressure weight. Those are separate config fields
and experimental factors.

| Topology ID | Active gate ports | Placement note |
| --- | --- | --- |
| `A0` | none | Stock Pythia path, including stock GELU at `h`; `model.site_gate` is `null` |
| `A1-H` | `h` | MLP hidden port only |
| `A2` | `m`, `h` | MLP branch input and hidden ports only |
| `A3` | `a`, `m`, `h` | Both branch inputs and the MLP hidden port |
| `A4-Q` | `a`, `m`, `h`, `q_post` | Q is POST-RoPE |
| `A4-K` | `a`, `m`, `h`, `k_post` | K is POST-RoPE |
| `A4-V` | `a`, `m`, `h`, `v` | V port before `PV` |
| `A5-QK-PRE` | `a`, `m`, `h`, `q_pre`, `k_pre` | Q and K are PRE-RoPE |
| `A5-QK-POST` | `a`, `m`, `h`, `q_post`, `k_post` | Q and K are POST-RoPE |
| `A6-PRE` | `a`, `m`, `h`, `q_pre`, `k_pre`, `v` | Q and K are PRE-RoPE; V is also active |
| `A6-POST` | `a`, `m`, `h`, `q_post`, `k_post`, `v` | Q and K are POST-RoPE; V is also active |

`A4-Q` and `A4-K` retain concise IDs but always mean the POST-RoPE ports; there
are no implied PRE variants. To realize the requested ReLU form of `A2`, for
example, a config combines `model.topology_id: A2` with
`model.site_gate.operator: relu`. The ID `A2` alone means only `{m, h}`.

[`humans/110-pythia-14m-s1-topology-atlas.pdf`](humans/110-pythia-14m-s1-topology-atlas.pdf)
is a historical visual reference for the original topology rows, not the
normative registry. It predates `A2`; no `A2` logical-reach ceiling is defined
or inferred here. Its ceilings are logical-product reach calculations, not
measured runtime speedups.

## Pressure Aggregation

Let `A_j` denote one captured named tensor, normally one site in one layer. Each
pressure first averages over all scalar elements of `A_j`, then takes an
unweighted arithmetic mean over the captured tensors. Consequently, captured
tensors are equally weighted even when their widths differ. A future change to
element-count weighting is a scientific change.

Pressure methods use the task objective and pressure objective as distinct
quantities. They must be logged separately.

The pressure-method identifiers are intentionally limited to:

- `none`: capture and log configured activation sites without adding pressure;
- `l1_naive`: add the L1 activation objective directly to task loss;
- `orthogonal_l1`: apply L1 pressure after the task-only AdamW step using the
  conflict-removal and trust-budget geometry below.

## L1 Activation Pressure

The implemented L1 pressure is:

```text
L_1 = (1/J) sum_j mean_{a in A_j} |a|
```

The method identifier `l1_naive` optimizes:

```text
L_task + weight * L_1
```

It is distinct from weight regularization and from hard activation thresholding.

## Adam-Step Orthogonal Pressure

`orthogonal_l1` uses the L1 pressure source above but does not give pressure
gradients to AdamW's moment updates.

At an optimizer boundary:

1. Compute accumulated task gradients.
2. Compute pressure gradients separately.
3. Run the normal AdamW step using task gradients only. AdamW first and second
   moments therefore remain task-only.
4. Reconstruct the bias-corrected Adam task direction and precondition the
   current pressure gradient with AdamW's task second moment.
5. If the global dot product between those directions is negative, remove the
   conflicting component from the pressure direction.
6. Limit the weighted safe-pressure norm relative to the task-direction norm
   with `step_budget`.
7. Apply the pressure correction after the AdamW step.

For eligible parameters, the implementation constructs:

```text
d_task = m_hat_task / (sqrt(v_hat_task) + adam_eps)
d_pressure = g_pressure / (sqrt(v_hat_task) + adam_eps)
dot = <d_task, d_pressure>
```

The dot product and norms are global sums over all eligible parameters. When
`dot < 0` and the task direction has nonzero norm:

```text
d_safe = d_pressure - dot / (||d_task||^2 + eps) * d_task
```

Otherwise `d_safe = d_pressure`. Projection is conditional conflict removal,
not unconditional orthogonalization.

The raw correction ratio and trust scale are:

```text
raw_ratio = weight * ||d_safe|| / (||d_task|| + eps)
scale = min(1, step_budget / (raw_ratio + eps))
```

The post-AdamW correction for each optimizer group is:

```text
theta <- theta - learning_rate * weight * scale * d_safe
```

Important boundaries:

- The reconstructed task direction describes Adam's gradient direction. The
  separate decoupled weight-decay displacement is not part of the projection
  dot product.
- Direction norms and projection are computed before per-group learning rates
  are applied. With heterogeneous learning rates, this is not true joint
  parameter-update-space geometry.
- `step_budget` caps the final pressure/task direction ratio; it is not a
  sparsity target or convergence guarantee.

## Site-gate Operators

Every active port in a non-`A0` topology uses the same explicit
`model.site_gate`. The supported operators are `relu`,
`one_sided_threshold`, and `symmetric_threshold`. `relu` is the standard
elementwise ReLU and has no `kappa`. The two parameter-free hard-threshold
operators require an explicit `kappa`.

One-sided gate:

```text
G+_kappa(x) = x if x >= kappa, else 0
```

Signed-magnitude gate:

```text
Gpm_kappa(x) = x if |x| >= kappa, else 0
```

For threshold operators, `kappa` is finite, absolute, and nonnegative.
Equality survives. The mask is formed from a detached comparison: surviving
inputs have identity input gradient and rejected inputs have zero input
gradient. These gates produce exact zeros but do not make dense kernels skip
work.

`model.topology_id` chooses the active ports and `model.site_gate` chooses the
operator (and `kappa`, when required). `A0` requires `model.site_gate: null` and
keeps stock GELU at `h`. Optimizer and activation-pressure settings remain
independent of both fields.

## Post-hoc Activation Clipping

Clipping evaluates a saved checkpoint with selected activation elements set to
exact zero. Supported selection modes are:

- absolute threshold: zero values selected by an absolute cutoff;
- quantile: derive a cutoff from the requested activation quantile;
- RMS threshold: use `multiplier * RMS(A)` for the captured tensor and forward
  pass.

Clipping is an inference-time intervention on a saved model, not a training
method. Every frontier must include and use its own zero-threshold point as the
loss reference, because measurement settings such as eager attention can
differ from the training validation path.

## Interpretation Limits

- A differentiable pressure can change an activation distribution without
  creating exact zeros.
- Near-zero mass depends on a stated threshold and is not exact sparsity.
- Exact-zero activations create only potential logical product skips until an
  implementation actually exploits them.
- Logical operation counts are not FLOP/s, latency, energy, or wall-clock
  speedup measurements.
- A last training minibatch is not a pooled validation diagnostic.
- Method behavior from a smoke or calibration run is plumbing evidence only.
- Any quality, sparsity, scaling, or efficiency claim requires the comparisons,
  seeds, budgets, uncertainty, and promotion rules in the reviewed experiment
  plan.
