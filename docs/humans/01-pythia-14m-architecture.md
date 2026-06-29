# Pythia-14M Architecture and Activation Pressure Sites

This note is a human-facing reference for understanding where activation-sparsity pressures can be applied in the current harness.

Scope:

- Model family: Pythia / GPT-NeoX causal language model.
- Current architecture source: `EleutherAI/pythia-14m-deduped`.
- Current training mode in this repository: random initialization from the architecture config, not released checkpoint weights.
- Current implemented pressure site: MLP hidden activations, named `mlp_hiddens`.

This document describes architecture and implementation surfaces only. It does not claim that any sparsity pressure is scientifically effective.

## Exact Local Configuration

The current harness reads the architecture with:

```bash
python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('EleutherAI/pythia-14m-deduped'); print(c)"
```

Relevant fields:

| Symbol | Meaning | Pythia-14M value |
| ------ | ------- | ---------------- |
| `L` | transformer blocks | 6 |
| `d` | residual hidden width | 128 |
| `h` | attention heads | 4 |
| `d_head` | per-head width, `d / h` | 32 |
| `d_ff` | MLP hidden width | 512 |
| `T_max` | context length | 2048 |
| `V` | vocabulary size | 50304 |
| activation | MLP nonlinearity | GELU |
| residual type | GPT-NeoX parallel residual | enabled |
| attention dropout | dropout probability | 0.0 |
| hidden dropout | dropout probability | 0.0 |
| positional encoding | rotary embedding | RoPE |
| rotary fraction | fraction of each head using RoPE | 0.25 |
| rotary dimension | `0.25 * d_head` | 8 |
| tied embeddings | input/output embeddings shared | no |

Local instantiated parameter count:

| Component | Parameters |
| --------- | ---------- |
| input embedding + output projection | 12,877,824 |
| non-embedding transformer parameters | 1,189,888 |
| total trainable parameters | 14,067,712 |

## High-level Computation

Let:

- `X in {0, ..., V-1}^{B x T}` be a batch of token ids.
- `H_l in R^{B x T x d}` be the residual stream entering transformer block `l`.
- `A_l in R^{B x T x d_ff}` be the post-GELU MLP hidden activation in block `l`.

The model is autoregressive. For each sequence position `t`, it predicts token `x_{t+1}` from tokens `x_{<=t}`.

The task loss is standard next-token cross entropy:

```text
L_task(theta) =
  - (1 / N) * sum_{examples, t} log p_theta(x_{t+1} | x_{<=t})
```

where `N` is the number of predicted tokens contributing to the batch loss.

## Embedding and Output

The first operation maps token ids to residual vectors:

```text
H_0 = EmbedIn(X)
```

with shape:

```text
EmbedIn: [V, d] = [50304, 128]
H_0:     [B, T, d]
```

After the final transformer block:

```text
H_final = FinalLayerNorm(H_L)
logits = H_final * W_out^T
```

where:

```text
W_out: [V, d] = [50304, 128]
logits: [B, T, V]
```

The input embedding and output projection are not tied in this config.

## One Transformer Block

Pythia uses the GPT-NeoX block structure with parallel residual branches. In block `l`, attention and MLP are both computed from layer-normalized versions of the same incoming residual stream `H_l`, then added back together.

### Block Equations

Attention branch:

```text
U_l = LN_attn_l(H_l)

[Q_l, K_l, V_l] = U_l * W_qkv_l + b_qkv_l
Q_l, K_l, V_l in R^{B x T x h x d_head}

Q_l, K_l = RoPE(Q_l, K_l)

S_l = causal_softmax((Q_l K_l^T) / sqrt(d_head))
C_l = S_l V_l
O_l = concat_heads(C_l) * W_o_l + b_o_l
```

MLP branch:

```text
R_l = LN_mlp_l(H_l)
Z_l = R_l * W_1_l + b_1_l
A_l = GELU(Z_l)
M_l = A_l * W_2_l + b_2_l
```

Parallel residual update:

```text
H_{l+1} = H_l + O_l + M_l
```

Dropout modules exist in the implementation, but the current config has dropout probability `0.0`.

### Block Pseudocode

```text
function GPTNeoXBlock_l(H):
    U = input_layernorm_l(H)
    QKV = attention.query_key_value_l(U)
    Q, K, V = split_heads(QKV, heads=4, head_dim=32)
    Q, K = apply_rope(Q, K, rotary_dim=8)
    S = causal_softmax(Q @ transpose(K) / sqrt(32))
    O = attention.dense_l(concat_heads(S @ V))

    R = post_attention_layernorm_l(H)
    Z = mlp.dense_h_to_4h_l(R)
    A = GELU(Z)
    M = mlp.dense_4h_to_h_l(A)

    return H + O + M
```

The current pressure hook captures `A`.

## Concrete Module Map

For layer `l`, where `l = 0, ..., 5`, the relevant Transformers module names are:

| Tensor | Shape | Module / source | Notes |
| ------ | ----- | --------------- | ----- |
| `H_l` | `[B, T, 128]` | residual stream entering `gpt_neox.layers.l` | Not currently hooked |
| `U_l` | `[B, T, 128]` | `gpt_neox.layers.l.input_layernorm` output | Attention branch input |
| `QKV_l` | `[B, T, 384]` | `gpt_neox.layers.l.attention.query_key_value` output | Split into Q, K, V |
| `Q_l,K_l,V_l` | `[B, T, 4, 32]` | internal attention tensors | Not directly exposed by a simple module hook |
| `O_l` | `[B, T, 128]` | `gpt_neox.layers.l.attention.dense` output | Attention branch output |
| `R_l` | `[B, T, 128]` | `gpt_neox.layers.l.post_attention_layernorm` output | MLP branch input |
| `Z_l` | `[B, T, 512]` | `gpt_neox.layers.l.mlp.dense_h_to_4h` output | MLP preactivation |
| `A_l` | `[B, T, 512]` | `gpt_neox.layers.l.mlp.act` output | Current pressure site |
| `M_l` | `[B, T, 128]` | `gpt_neox.layers.l.mlp.dense_4h_to_h` output | MLP branch output |
| `H_{l+1}` | `[B, T, 128]` | residual after block add | Not currently hooked |

The current implementation in `src/paper_exp/activations.py` registers:

```text
gpt_neox.layers.{l}.mlp.act
```

and records the output as:

```text
mlp_hiddens.layer_{l}
```

This means the current pressure tensor is:

```text
A_l = GELU(Z_l) in R^{B x T x 512}
```

for every transformer block `l`.

The current sparsity notion is elementwise. A value such as `80%` exact-zero sparsity means that `80%` of scalar entries `A_l[b,t,j]` are zero across the measured layers, batches, token positions, and MLP hidden channels. It does not mean that the same `80%` of the 512 MLP hidden channels are zero for every batch item and every token position. Structured channel sparsity would require a separate channel-level metric or pressure.

## Why `mlp_hiddens` Is the First Pressure Site

The MLP hidden activation is a natural first target because:

- It is high-dimensional: `512` channels per token per layer, compared with `128` in the residual stream.
- It sits after the nonlinearity and before the MLP projection back to the residual stream.
- It has a stable module boundary in Transformers: `gpt_neox.layers.{l}.mlp.act`.
- It is less entangled with causal attention mechanics than Q/K/V or attention scores.
- It lets us measure near-zero mass directly on activations without changing model topology.

The pressure is applied to activations, not weights. A typical pressure over all MLP hidden sites is:

```text
P(A) = (1 / L) * sum_{l=0}^{L-1} P_l(A_l)
```

where `P_l` may be an L1 pressure, a Ricker pressure, or another scalar pressure over the entries of `A_l`. In the current implementation, both L1 and Ricker are applied to each scalar activation element and then averaged. The shared model weights receive gradients from this aggregate scalar pressure; the harness is not independently selecting or disabling whole activation channels.

## Current Pressure Objectives

### L1 Activation Pressure

For captured activations `A_l`, the L1 pressure is:

```text
P_L1(A) =
  (1 / L) * sum_l mean_{b,t,j} |A_l[b,t,j]|
```

The naive training objective is:

```text
L_total = L_task + lambda * P_L1(A)
```

### Ricker Activation Pressure

The current Ricker score for one scalar activation `a` is:

```text
r(a; c, sigma) =
  (1 - a^2 / c^2) * exp(-a^2 / (2 sigma^2))
```

The pressure is implemented as:

```text
P_Ricker(A) =
  1 - mean_{l,b,t,j} r(A_l[b,t,j]; c, sigma)
```

The naive training objective is:

```text
L_total = L_task + lambda * P_Ricker(A)
```

This pressure is intended to shape the activation distribution near zero. It should be interpreted empirically, not assumed to produce useful sparse computation by itself.

## Orthogonal Pressure Update

For orthogonal methods, AdamW should see only task gradients in its moments.

At each optimizer step:

```text
g_task     = grad_theta L_task
g_pressure = grad_theta P(A)
```

AdamW updates parameters using `g_task`. Then the pressure correction is applied after the AdamW step in Adam-preconditioned update space.

Pseudo-algorithm:

```text
for each training step:
    compute L_task and P(A)

    backprop L_task
    save g_task
    AdamW.step() using task gradients only

    compute pressure direction d_pressure in Adam step space
    compute task direction d_task in Adam step space

    if dot(d_task, d_pressure) < 0:
        d_pressure = d_pressure - projection of d_pressure onto d_task

    cap ||lambda * d_pressure|| / ||d_task|| by step_budget
    theta = theta - learning_rate * lambda * capped_d_pressure
```

The intent is to prevent the pressure correction from directly opposing the task update in the same Adam-step geometry. This is still a method hypothesis, not a guarantee of better training.

## Post-hoc Activation Clipping

Post-hoc clipping is different from training-time pressure.

Given a threshold `tau`, the clipping transform is:

```text
clip_tau(a) =
    0, if |a| <= tau
    a, otherwise
```

Applied to MLP hidden activations:

```text
A_l_clipped[b,t,j] = clip_tau(A_l[b,t,j])
```

This creates exact zeros and lets us measure a frontier:

```text
validation loss versus achieved exact-zero activation sparsity
```

This is the correct first diagnostic for asking whether near-zero activations can be zeroed without immediately damaging validation loss. It is not yet a hardware speedup measurement.

## Candidate Sites for Future Pressures

The table below separates currently implemented sites from plausible future sites.

| Site | Status | Tensor | Shape | Why it may matter | Main risk |
| ---- | ------ | ------ | ----- | ----------------- | --------- |
| MLP hidden activation | implemented | `A_l = GELU(Z_l)` | `[B, T, 512]` | Direct high-dimensional activation target before MLP compression | May reduce useful nonlinear features |
| MLP preactivation | future hook | `Z_l` | `[B, T, 512]` | Pressures the signal before GELU | Changes activation gating behavior more directly |
| MLP output | future hook | `M_l` | `[B, T, 128]` | Pressures the MLP contribution to the residual stream | Lower dimensional and closer to residual semantics |
| Attention output | future hook | `O_l` | `[B, T, 128]` | Pressures attention contribution to residual stream | May disturb routing and context aggregation |
| Q/K/V projections | future deeper hook | `Q_l,K_l,V_l` | `[B, T, 4, 32]` | Could target attention internals | Easy to break attention geometry |
| Residual stream | future hook | `H_l` or `H_{l+1}` | `[B, T, 128]` | Broadest representation-level pressure | Most likely to interfere with all downstream computation |
| LayerNorm outputs | future hook | `U_l`, `R_l` | `[B, T, 128]` | Normalized branch inputs may be easier to compare across layers | Can fight normalization dynamics |

Recommended progression:

1. Keep `mlp_hiddens` as the primary site until the baseline and pressure runs are stable.
2. Add layer-specific MLP hidden ablations before adding new tensor families.
3. Add `Z_l` or `M_l` only if the MLP hidden results justify a finer question.
4. Treat attention and residual-stream pressures as separate method variants, not minor implementation changes.

## What We Can Measure at `mlp_hiddens`

For each captured activation tensor `A_l`, the harness can measure:

```text
near_zero_mass(k) =
  count(|A_l| <= k) / number_of_entries(A_l)
```

Currently logged thresholds include:

```text
k = 0, 0.001, 0.01
```

For post-hoc clipping, exact sparsity is:

```text
exact_zero_sparsity =
  count(A_l_clipped == 0) / number_of_entries(A_l_clipped)
```

The important distinction is:

- Near-zero mass measures how much activation mass is close to zero.
- Exact-zero sparsity after clipping measures how much can be made exactly zero by a specified rule.
- Both are currently elementwise activation statistics over entries `A_l[b,t,j]`, not guarantees that fixed hidden dimensions are always inactive.
- Neither quantity alone proves speedup. Speedup requires sparse operators, routing, or explicit latency/FLOP instrumentation.

## References

- Pythia model card: <https://huggingface.co/EleutherAI/pythia-14m-deduped>
- GPT-NeoX architecture docs in Transformers: <https://huggingface.co/docs/transformers/main/model_doc/gpt_neox>
- Pythia paper: <https://arxiv.org/abs/2304.01373>
- Local hook implementation: `src/paper_exp/activations.py`
- Local pressure implementation: `src/paper_exp/activation_pressure.py`
