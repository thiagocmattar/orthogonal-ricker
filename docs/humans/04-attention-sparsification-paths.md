# Pythia Post-QKV ReLU Experiment: Q/K Placement Around RoPE

Date: 2026-07-16

Purpose: specify the next Pythia-14M MiniPile architecture experiment well enough that a fresh agent can implement, test, launch, diagnose, and report it without relying on prior conversation.

Status: the original PRE/POST ordinary-ReLU experiment is complete. Section 15 specifies the selected fixed symmetric-threshold POST follow-up. Every follow-up outcome remains a hypothesis until its runs complete.

Related repository documents:

- [Pythia-14M architecture](01-pythia-14m-architecture.md)
- [Methods](../methods.md)
- [Experiment log](../experiment_log.md)
- [Code map](../code_map.md)
- [Post-LayerNorm ReLU report](../../report/04-2026-07-11-post-layernorm-relu-ol1-comparison/04-2026-07-11-post-layernorm-relu-ol1-comparison.pdf)

## 1. Selected Experiment in One Page

The current Three-ReLU architecture creates exact zeros at:

```text
attention_inputs = ReLU(input_layernorm(H_l))
mlp_inputs       = ReLU(post_attention_layernorm(H_l))
mlp_hiddens      = ReLU(W1(mlp_inputs))
```

The first gate is the input to the fused QKV projection. Its zeros expose logical zero products inside QKV, but the dense biased projection produces effectively dense Q, K, and V tensors. The next experiment keeps all three existing gates and adds three separate, parameter-free ReLU modules after the fused QKV projection has been reshaped and split:

```text
query_relu
key_relu
value_relu
```

This produces a Six-ReLU architecture. The fused QKV linear layer must remain fused; "separate Q, K, and V" means separate gates and activation sites after the split, not three independently initialized Q/K/V linear layers.

Two matched architecture arms are required:

1. **PRE:** apply the Q and K ReLUs before standard partial RoPE; apply the V ReLU immediately after the split.
2. **POST:** apply standard partial RoPE first, then the Q and K ReLUs; apply the V ReLU at the same post-split location as PRE.

Each architecture is crossed with exactly three training methods:

- **AdamW:** architecture-only control, with no pressure;
- **OR:** orthogonal Ricker, weight 1.0, $c=\sigma=0.05$, step budget 0.5; and
- **OL1:** orthogonal L1, weight 5.0, step budget 0.5.

The primary grid therefore contains six full-pass runs:

```text
PRE  x {AdamW, OR, OL1}
POST x {AdamW, OR, OL1}
```

For clean attribution, OR and OL1 pressure only the three new Q/K/V gate outputs. The older `attention_inputs`, `mlp_inputs`, and `mlp_hiddens` ReLUs remain in the architecture but are not included in the pressure scalar for this screen. They are measured again in the full-validation propagation diagnostic. An all-six-site pressure experiment is deferred until PRE or POST has been selected.

This restriction is deliberate. The current pressure implementation averages one scalar mean per captured tensor. Expanding the pressure set from three site families to six would change the relative weighting of every old and new site at the same nominal pressure weight, while also mixing renewed QKV/$W_1$/$W_2$ pressure with the new QK/PV hypothesis. New-site-only pressure makes OR/OL1 minus same-placement AdamW the cleanest test of the added attention-core path.

Probability gating is not part of this experiment. Plain `ReLU(P)` is inactive because valid softmax probabilities are already nonnegative. A learnable positive-threshold probability gate remains a later TODO in Section 11.

## 2. Why This Is the Next Boundary

Figure 85 in Report 04 measures the current exact-zero propagation. For Three-ReLU OR, the QKV input exact-zero fraction is:

| Layer | Exact zeros at `attention_inputs` |
| ---: | ---: |
| 0 | 55.16% |
| 1 | 70.25% |
| 2 | 75.22% |
| 3 | 79.96% |
| 4 | 80.60% |
| 5 | 75.26% |
| Pooled over layers | 72.74% |

Thus the value above 80% is a layerwise value, not the six-layer pooled OR value. Config `106` pooled over 692,224 validation tokens shows that the existing gate is highly sparse while Q, K, and V after the projection path contain fewer than 0.00002% exact zeros.

The exact current boundary is:

```text
attention_inputs, sparse
        |
        v
fused biased QKV projection
        |
        v
Q0, K0, V0, effectively dense
        |
        +---- Q0, K0 -> partial RoPE -> QK -> softmax P
        |
        +---- V0 -----------------------------> PV
```

The current ReLU already creates a QKV input-product opportunity. The new post-QKV gates are additional gates: they do not save QKV work, but they can expose logical zero products in QK and PV.

At the architecture ceiling, if every existing and new gate operand were zero, the directly reachable operations would be QKV, QK, PV, $W_1$, and $W_2$. Only $W_O$ would remain untargeted within the block. Under the current causal-logical accounting at $T=2048$, this corresponds to a maximum 96.43% block opportunity and 28.88% model opportunity after including six blocks and the LM head. These are ceilings, not expected sparsity or speedups.

One measurement qualification matters. The current diagnostic records Q and K after RoPE and V directly after the QKV split. It does not separately record pre-RoPE Q/K. The new diagnostic must expose both sides of RoPE so that projection densification and RoPE mask changes can be separated.

## 3. Exact Current Attention Equations

### 3.1 Dimensions and indices

For Pythia-14M:

| Quantity | Symbol | Value |
| --- | ---: | ---: |
| Transformer layers | $L$ | 6 |
| Model width | $d$ | 128 |
| Attention heads | $H$ | 4 |
| Head width | $d_h=d/H$ | 32 |
| Rotary width per head | $d_R$ | 8 |
| Pass-through width per head | $d_h-d_R$ | 24 |
| Maximum sequence length | $T$ | 2,048 |
| RoPE base | $\theta$ | 10,000 |

Indices are $b$ for batch, $l$ for layer, $h$ for head, $t$ for query position, $s$ for key/value position, $k$ for model-width coordinate, and $r$ for head coordinate.

Attention and hidden dropout are zero in the current config. QKV and attention-output projections include learned biases. The model uses parallel residual branches.

### 3.2 Existing attention-input gate and fused QKV

Let $H_l\in\mathbb{R}^{B\times T\times d}$ be the residual stream entering block $l$. The current attention input is

$$
X_{bltk}
=
\operatorname{ReLU}\left(\operatorname{LN}^{\mathrm{att}}_l(H_l)_{btk}\right).
$$

Repository nomenclature is

```text
attention_inputs = X_l
```

The fused QKV projection is

$$
Y_l=X_lW_{QKV,l}^{\top}+b_{QKV,l},
$$

with stored weight shape $[384,128]$ and output shape $[B,T,384]$. After reshaping by head and splitting the last dimension:

$$
Q_l^0,K_l^0,V_l^0
\in
\mathbb{R}^{B\times H\times T\times d_h}.
$$

Elementwise,

$$
Q^0_{blhtr}
=
\sum_{k=0}^{d-1}X_{bltk}W^Q_{lhrk}+b^Q_{lhr},
$$

$$
K^0_{blhtr}
=
\sum_{k=0}^{d-1}X_{bltk}W^K_{lhrk}+b^K_{lhr},
$$

$$
V^0_{blhtr}
=
\sum_{k=0}^{d-1}X_{bltk}W^V_{lhrk}+b^V_{lhr}.
$$

A zero $X_{bltk}=0$ makes 384 input-weight products zero. It does not normally make a Q/K/V output coordinate zero because the other input coordinates and learned bias remain.

### 3.3 Standard partial RoPE

RoPE is a position-dependent rotation, not an additive positional embedding. It is applied only to Q and K. V is unchanged.

For pair index $j\in\{0,1,2,3\}$:

$$
\omega_j=10000^{-2j/8},
\qquad
\phi_{tj}=t\omega_j.
$$

For Q:

$$
Q^R_{blhtj}
=
Q^0_{blhtj}\cos\phi_{tj}
-Q^0_{blht,j+4}\sin\phi_{tj},
$$

$$
Q^R_{blht,j+4}
=
Q^0_{blht,j+4}\cos\phi_{tj}
+Q^0_{blhtj}\sin\phi_{tj}.
$$

K uses the same equations. Coordinates $r\in\{8,\ldots,31\}$ pass through unchanged. An all-zero rotary pair remains zero; an isolated zero paired with a nonzero value is generally repopulated. Position comes from `position_ids`, independently of the activation mask.

### 3.4 QK, softmax P, PV, and output projection

For each valid causal pair $s\le t$:

$$
S_{blhts}
=
\frac{1}{\sqrt{d_h}}
\sum_{r=0}^{d_h-1}Q_{blhtr}K_{blhsr}.
$$

After adding the causal mask, softmax gives

$$
P_{blhts}
=
\frac{\exp(S_{blhts})}
{\sum_{u=0}^{t}\exp(S_{blhtu})}.
$$

The context is

$$
C_{blhtr}
=
\sum_{s=0}^{t}P_{blhts}V_{blhsr}.
$$

After concatenating heads:

$$
O_l=C_l^{\mathrm{cat}}W_{O,l}^{\top}+b_{O,l}.
$$

Pythia's parallel residual update is

$$
H_{l+1}=H_l+O_l+M_l,
$$

where $M_l$ is independently computed by the MLP branch from the same $H_l$.

## 4. Architecture PRE: Q/K ReLU Before RoPE

The fused projection and split remain unchanged. Apply three distinct gates:

$$
Q^G_{blhtr}=\operatorname{ReLU}(Q^0_{blhtr}),
$$

$$
K^G_{blhtr}=\operatorname{ReLU}(K^0_{blhtr}),
$$

$$
V^G_{blhtr}=\operatorname{ReLU}(V^0_{blhtr}).
$$

Then apply standard RoPE to the gated Q/K tensors:

$$
Q^{\mathrm{QK}}_{blht:}=R_tQ^G_{blht:},
\qquad
K^{\mathrm{QK}}_{blht:}=R_tK^G_{blht:},
$$

and send V directly to PV:

$$
V^{\mathrm{PV}}=V^G.
$$

The actual attention operations are

$$
S_{blhts}
=
\frac{1}{\sqrt{d_h}}
\sum_rQ^{\mathrm{QK}}_{blhtr}K^{\mathrm{QK}}_{blhsr},
$$

$$
C_{blhtr}
=
\sum_{s=0}^{t}P_{blhts}V^G_{blhsr}.
$$

Pressure is applied to $Q^G$, $K^G$, and $V^G$, which are the outputs of the three new ReLU modules.

Expected structural behavior:

- On the 24 pass-through dimensions, gate zeros reach QK unchanged.
- On the eight rotary dimensions, an all-zero pair stays zero.
- An isolated zero inside an active rotary pair is generally repopulated.
- Standard RoPE relative-position geometry is preserved for the gated vectors:

$$
(R_tq)^\top(R_sk)=q^\top R_{s-t}k.
$$

- Q/K gate sparsity is therefore not the same as direct QK operand sparsity and both must be measured.

PRE is favored if it preserves validation quality and signed attention geometry while enough pass-through and pairwise zeros survive to make $R_{QK}^{0}$ material.

PRE is falsified as a useful direct QK path if pressure creates many gate zeros but RoPE repopulation leaves little direct QK zero-product opportunity.

## 5. Architecture POST: Q/K ReLU After RoPE

First apply ordinary RoPE to the dense split Q/K tensors:

$$
Q^R_{blht:}=R_tQ^0_{blht:},
\qquad
K^R_{blht:}=R_tK^0_{blht:}.
$$

Then apply the new Q/K gates:

$$
Q^{\mathrm{QK}}_{blhtr}=\operatorname{ReLU}(Q^R_{blhtr}),
$$

$$
K^{\mathrm{QK}}_{blhtr}=\operatorname{ReLU}(K^R_{blhtr}).
$$

V uses the same location as PRE:

$$
V^G_{blhtr}=\operatorname{ReLU}(V^0_{blhtr}),
\qquad
V^{\mathrm{PV}}=V^G.
$$

The Q/K gate outputs are now the exact operands of QK:

$$
S_{blhts}
=
\frac{1}{\sqrt{d_h}}
\sum_rQ^{\mathrm{QK}}_{blhtr}K^{\mathrm{QK}}_{blhsr}.
$$

Expected structural behavior:

- Every Q/K gate zero is a direct QK operand zero.
- Q/K gate exact-zero fractions equal their corresponding QK-input exact-zero fractions.
- Because both operands are nonnegative, every valid pre-mask dot-product score is nonnegative:

$$
S_{blhts}\ge0.
$$

- ReLU after RoPE does not preserve the standard RoPE relative-position identity.
- POST changes attention geometry as well as sparsity and therefore needs its own AdamW architecture control.

POST is favored if its direct $R_{QK}^{0}$ advantage over PRE is large enough to compensate for any validation-loss, score-distribution, or attention-entropy cost.

POST is disfavored if direct QK sparsity comes with disproportionate loss, head collapse, probability concentration, or unstable score scaling.

## 6. What PRE Versus POST Actually Changes

The two arms share:

- the same random initialization and fused QKV weights;
- the same existing Three-ReLU architecture;
- the same separate V gate;
- the same attention backend and output projection;
- the same training, validation, and checkpoint recipe; and
- the same new pressure-site aliases.

They differ only in whether the Q/K ReLUs are applied before or after partial RoPE. Since only eight of 32 dimensions per head use RoPE, PRE and POST are numerically identical on the 24 pass-through dimensions. At token position zero, RoPE is also the identity on the rotary dimensions.

In both arms, valid P is positive and $V^G$ is nonnegative, so the context is nonnegative. A context coordinate is exactly zero only when every causally available V value for that coordinate is zero, apart from numerical underflow. Measure any resulting $W_O$ input opportunity directly; do not add another context ReLU in this experiment because it would be an identity on the nonnegative context.

This comparison has two unavoidable scientific differences:

1. PRE pressures Q/K in the unrotated coordinate basis, while POST pressures Q/K in the position-rotated basis. L1 and Ricker pressure are elementwise and not rotation invariant.
2. PRE preserves standard RoPE geometry after gating, while POST guarantees direct scalar zeros but changes that geometry and makes both Q/K operands nonnegative.

These differences must be interpreted as the complete placement effect. This experiment cannot attribute a PRE/POST difference to scalar zero preservation alone.

## 7. Pressure Scope and Method Definitions

### 7.1 Stable proposed site aliases

Use these activation-pressure and capture aliases:

```text
query_gate_outputs
key_gate_outputs
value_gate_outputs
```

Their meaning is invariant across configs:

- `query_gate_outputs`: output of `attention.query_relu`;
- `key_gate_outputs`: output of `attention.key_relu`; and
- `value_gate_outputs`: output of `attention.value_relu`.

In PRE, Q/K gate outputs are before RoPE. In POST, Q/K gate outputs are after RoPE. V gate placement is identical in both.

The pressure scalar must use only these three aliases. The existing `attention_inputs`, `mlp_inputs`, and `mlp_hiddens` gates remain present but are excluded from the pressure loss in this screen.

For $\phi(X)$ equal to the element-mean Ricker or L1 pressure at one captured tensor, the intended equal-site/equal-layer aggregation is

$$
\mathcal{L}_{\mathrm{pressure}}
=
\frac{1}{3L}
\sum_{l=0}^{L-1}
\left[
\phi(Q_l^G)+\phi(K_l^G)+\phi(V_l^G)
\right].
$$

OR and OL1 still use one joint pressure scalar and one global projected correction. Separate aliases provide separate measurement and gate placement; they do not create three independent orthogonal updates.

### 7.2 AdamW control

```yaml
activation_pressure:
  enabled: true
  method: none
  sites:
    - query_gate_outputs
    - key_gate_outputs
    - value_gate_outputs
  weight: 0.0
```

This is the pure AdamW architecture control. ReLUs are active, but no pressure gradient or post-Adam correction is applied.

### 7.3 OR

```yaml
activation_pressure:
  enabled: true
  method: orthogonal_ricker
  sites:
    - query_gate_outputs
    - key_gate_outputs
    - value_gate_outputs
  weight: 1.0
  ricker_c: 0.05
  ricker_sigma: 0.05
  step_budget: 0.5
  eps: 1.0e-12
```

### 7.4 OL1

```yaml
activation_pressure:
  enabled: true
  method: orthogonal_l1
  sites:
    - query_gate_outputs
    - key_gate_outputs
    - value_gate_outputs
  weight: 5.0
  step_budget: 0.5
  eps: 1.0e-12
```

Only AdamW, OR, and OL1 are in scope. Without RN and L1N controls, this experiment cannot establish an orthogonalization effect. OR versus OL1 also changes both the penalty family and nominal weight, so that comparison is descriptive rather than a matched penalty comparison.

## 8. Proposed Config and Runtime Contract

### 8.1 Model config

Add one explicit architecture field:

```yaml
model:
  hidden_act: relu
  post_layernorm_relu: true
  post_qkv_relu:
    enabled: true
    query: true
    key: true
    value: true
    qk_placement: pre_rope  # or post_rope
```

The selected experiment always enables all three new gates. The individual booleans make checkpoint reconstruction and later ablations explicit; do not use them to silently broaden this six-run grid.

The complete `post_qkv_relu` mapping must be copied into the Hugging Face architecture config before model construction and saved in `checkpoints/final/config.json`. Checkpoint loading must reconstruct the same modules and placement from that saved field. ReLU has no parameters, so a missing reconstruction can otherwise load without a state-dict error while silently changing the architecture.

### 8.2 Planned config matrix

Assuming config `106` is still the highest prefix when implementation begins, reserve:

| Config | Architecture | Method |
| --- | --- | --- |
| `107-pythia-14m-minipile-post-qkv-relu-qk-pre-rope-adamw-full-pass.yaml` | PRE | AdamW |
| `108-pythia-14m-minipile-post-qkv-relu-qk-pre-rope-orthogonal-ricker-full-pass-w1-c0p05-s0p05.yaml` | PRE | OR |
| `109-pythia-14m-minipile-post-qkv-relu-qk-pre-rope-orthogonal-l1-full-pass-w5.yaml` | PRE | OL1 |
| `110-pythia-14m-minipile-post-qkv-relu-qk-post-rope-adamw-full-pass.yaml` | POST | AdamW |
| `111-pythia-14m-minipile-post-qkv-relu-qk-post-rope-orthogonal-ricker-full-pass-w1-c0p05-s0p05.yaml` | POST | OR |
| `112-pythia-14m-minipile-post-qkv-relu-qk-post-rope-orthogonal-l1-full-pass-w5.yaml` | POST | OL1 |
| `113-pythia-14m-minipile-post-qkv-relu-qk-placement-activation-propagation.yaml` | nine checkpoints | Full-validation diagnostic |

Verify the highest existing prefix immediately before creating files. If another config has been added, shift the entire block while preserving its order.

Config `113` should select the six new checkpoints plus existing configs `98` (Three-ReLU AdamW), `103` (Three-ReLU OR), and `99` (Three-ReLU OL1). The older checkpoints provide directly remeasured context under the extended diagnostic schema, with `N/A` for absent Q/K/V gates. They are not matched pressure-scope controls.

### 8.3 Matched training recipe

Copy the full-pass recipe from configs `98`, `103`, and `99`:

| Setting | Value |
| --- | ---: |
| Initialization | random |
| Seed | 0 |
| Training steps | 22,762 |
| Planned tokens | 1,491,730,432 |
| Block size | 2,048 |
| Micro-batch size | 4 |
| Gradient accumulation | 8 |
| Learning rate | $3\times10^{-5}$ |
| Warmup steps | 100 |
| AdamW betas | $(0.9,0.999)$ |
| AdamW epsilon | $10^{-8}$ |
| Weight decay | 0.01 |
| Parameters | float32 |
| Compute precision | bf16 autocast when supported |
| Validation | full deterministic cache |
| Evaluated validation tokens | 692,224 |
| Final checkpoint | required |

No released Pythia checkpoint weights are loaded.

## 9. Implementation Handoff

The architecture change is scientific code. Implement it locally; do not edit the installed Transformers package.

### 9.1 `src/paper_exp/config.py`

- Validate `model.post_qkv_relu` as a mapping when present.
- Require boolean `enabled`, `query`, `key`, and `value` fields.
- Require `qk_placement` to be exactly `pre_rope` or `post_rope`.
- Reject a placement when the feature is disabled rather than silently ignoring malformed fields.

### 9.2 `src/paper_exp/calibration.py`

- Extend `_apply_model_architecture_overrides` to copy `post_qkv_relu` into the Hugging Face architecture config before `AutoModelForCausalLM.from_config(...)`.
- Apply the runtime architecture modification after model construction, alongside `apply_post_layernorm_relu`.
- Preserve `model.initialization: random` and FP32 parameter initialization.

### 9.3 `src/paper_exp/modeling.py`

- Add `query_relu`, `key_relu`, and `value_relu` modules to every GPT-NeoX attention layer when enabled.
- Route PRE and POST through the same local attention-forward implementation. The only branch should be Q/K gate placement around RoPE.
- Keep the original fused `query_key_value` linear, head reshape/split, partial RoPE, cache update, selected attention backend, head concatenation, `attention.dense`, and dropout semantics.
- Keep V gating immediately after the QKV split in both arms.
- Preserve GPT-NeoX's actual fused layout: reshape `[B,T,3d]` to `[B,T,H,3d_h]`, transpose to `[B,H,T,3d_h]`, and only then split Q/K/V along the last dimension. Do not split the raw fused tensor into three contiguous width-$d$ slices.
- Cache the tensors with the architecture already applied: PRE caches RoPE-rotated gated K and gated V; POST caches gated post-RoPE K and gated V. Pretraining uses `use_cache: false`, but checkpoint reconstruction must not create incorrect generation behavior.
- Do not implement PRE with one hook and POST with a different attention backend; that would confound placement with execution code.
- Do not consume random numbers when registering parameter-free gates.
- Reapply the modification in `load_checkpoint_model` from the saved config.
- Guard against applying the modification twice.

If a local attention wrapper or bound forward method is used, test it against the installed Transformers version. The disabled/no-gate path must match the stock forward and gradients.

### 9.4 `src/paper_exp/activations.py`

- Add the three stable aliases from Section 7.
- Capture the outputs of the explicit ReLU modules, not inferred or reconstructed tensors.
- Record shape `[batch, heads, tokens, head_width]`.
- For PRE Q/K metadata, state that RoPE is the next operator.
- For POST Q/K metadata, state that QK is the next operator.
- For V metadata, state that PV is the next operator.

The generic L1/Ricker pressure code should not need a new pressure formula once these aliases resolve to tensors.

### 9.5 `src/paper_exp/activation_propagation.py`

Add diagnostic stages that make placement auditable:

```text
query_projection_output
key_projection_output
value_projection_output
query_gate_input
key_gate_input
value_gate_input
query_gate_output
key_gate_output
value_gate_output
query_qk_input
key_qk_input
value_pv_input
```

The schema must be stable across both arms and the older contextual checkpoints. For an absent gate, report `N/A`; do not silently substitute a raw tensor or report measured zero. In PRE, `query_gate_input`/`key_gate_input` are the projection outputs and `query_qk_input`/`key_qk_input` are after RoPE. In POST, the gate inputs are after RoPE and the gate outputs are the QK inputs. The actual QK and PV operands must drive zero-product counts. Do not infer QK opportunity from the gate-output fraction in PRE.

The existing diagnostic temporarily selects eager attention to expose P. Ensure that this diagnostic route still executes the configured post-RoPE gates; it must not bypass a gate implemented only inside a custom SDPA interface.

Continue excluding future causal-mask entries from P, QK, and PV denominators.

### 9.6 Required tests

Add focused tests before creating full-pass configs:

1. Config validation accepts both placements and rejects unknown values/types.
2. Disabled architecture has forward and backward parity with stock GPT-NeoX attention.
3. PRE applies Q/K ReLU before a controlled rotation and V ReLU after the split.
4. POST applies the controlled rotation before Q/K ReLU and uses the same V location.
5. The fused QKV linear weights and biases are unchanged; no separate projections are introduced.
6. Q/K/V capture aliases return the exact tensors used at the configured boundaries.
7. Pressure receives exactly 18 tensors: three sites across six layers.
8. PRE diagnostic distinguishes gate zeros from actual QK-input zeros after RoPE.
9. POST gate zeros equal actual QK-input zeros.
10. PV zero-product counting weights a V zero at key position $s$ across all valid queries $t\ge s$.
11. Checkpoint save/load reconstructs all gates and their placement.
12. Existing post-LayerNorm ReLU behavior and final LayerNorm exclusion remain unchanged.

Use a deterministic hand-computed rotary pair in the placement tests. A nonzero paired with a zero must demonstrate PRE repopulation, while POST must zero the negative rotated coordinate after rotation.

## 10. Preflight, Launch, and Completion Checks

### 10.1 Before full runs

Run:

```powershell
make test
make check
```

Also run one small forward/backward test for each placement and confirm:

- all six ReLU modules per block are present;
- Q/K/V pressure capture returns finite tensors;
- task and pressure gradients are finite;
- OR/OL1 projection diagnostics are emitted;
- POST valid QK scores are nonnegative before masking; and
- PRE and POST use the same attention backend.

If an empirical training smoke is needed, give it a new numbered config. Do not edit a planned full-pass config into a smoke run and then reuse its identity.

### 10.2 Launch commands

Create all six configs before launch. The exact commands are:

```powershell
python -m paper_exp.cli pretrain --config configs/107-pythia-14m-minipile-post-qkv-relu-qk-pre-rope-adamw-full-pass.yaml
python -m paper_exp.cli pretrain --config configs/110-pythia-14m-minipile-post-qkv-relu-qk-post-rope-adamw-full-pass.yaml
python -m paper_exp.cli pretrain --config configs/108-pythia-14m-minipile-post-qkv-relu-qk-pre-rope-orthogonal-ricker-full-pass-w1-c0p05-s0p05.yaml
python -m paper_exp.cli pretrain --config configs/111-pythia-14m-minipile-post-qkv-relu-qk-post-rope-orthogonal-ricker-full-pass-w1-c0p05-s0p05.yaml
python -m paper_exp.cli pretrain --config configs/109-pythia-14m-minipile-post-qkv-relu-qk-pre-rope-orthogonal-l1-full-pass-w5.yaml
python -m paper_exp.cli pretrain --config configs/112-pythia-14m-minipile-post-qkv-relu-qk-post-rope-orthogonal-l1-full-pass-w5.yaml
```

Recommended launch order:

1. PRE AdamW (`107`);
2. POST AdamW (`110`);
3. PRE OR (`108`);
4. POST OR (`111`);
5. PRE OL1 (`109`); and
6. POST OL1 (`112`).

Running both AdamW controls first detects an architecture-quality failure before spending four additional full passes on pressure runs.

Every launch must save the immutable `config.yaml` snapshot and `status: running` manifest before training. A successful run is complete only after metrics, predictions/events, final checkpoint, and `status: completed` manifest are durable.

### 10.3 Full-validation diagnostic

After all selected checkpoints complete, run:

```powershell
python -m paper_exp.cli activation-propagation --config configs/113-pythia-14m-minipile-post-qkv-relu-qk-placement-activation-propagation.yaml
```

The diagnostic must use all 338 complete validation blocks and 692,224 tokens for every checkpoint. It must record integer zero/total counts rather than averages of batch percentages.

## 11. Deferred P Hypothesis: Learnable-Threshold Dynamic ReLU

Probability sparsification is explicitly out of scope for configs `107` through `113`.

For every finite valid causal score, softmax gives

$$
P_{blhts}>0.
$$

Therefore

$$
\operatorname{ReLU}(P_{blhts})=P_{blhts},
$$

so a simple ReLU is not an active gate. L1 pressure on raw or renormalized P is also constant per row because valid probabilities are nonnegative and sum to one.

`TODO:` test a later learnable positive-threshold ReLU. One safe candidate is

$$
\tau_{blht}
=
\operatorname{sigmoid}(\eta_{lh})
\max_{s\le t}P_{blhts},
$$

$$
G_{blhts}
=
\operatorname{ReLU}(P_{blhts}-\tau_{blht}),
$$

$$
\widetilde P_{blhts}
=
\frac{G_{blhts}}
{\sum_{u=0}^{t}G_{blhtu}}.
$$

Here $\eta_{lh}$ is learnable per layer and head, while the threshold adapts to each probability row. Since `sigmoid` is less than one in exact arithmetic, at least one maximum-probability entry survives; an explicit all-zero-row safeguard is still required in finite precision.

This is a shifted/thresholded ReLU, not plain ReLU on P. Its pressure objective also requires separate design because L1 on normalized $\widetilde P$ remains constant. Do not implement this TODO as part of the selected Q/K/V experiment.

## 12. Required Result Handoff

Exact zero means direct numeric equality:

```text
value == 0
```

Counts must be pooled as integers over all evaluated validation tokens, layers, heads, and coordinates. QK and PV counts use only valid causal query-key pairs.

The default result table should be:

| Placement | Method | Validation loss | $R_{\mathrm{block}}$ | $R_{\mathrm{model}}$ | $R_{QK}^{0}$ | $R_{PV}^{0}$ | $z_a$ | $z_q^{G}$ | $z_q^{QK}$ | $z_k^{G}$ | $z_k^{QK}$ | $z_v^{PV}$ | $z_m$ | $z_h$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

Definitions:

- $z_q^G,z_k^G$: exact-zero fractions at Q/K gate outputs;
- $z_q^{QK},z_k^{QK}$: exact-zero fractions at the actual QK operands;
- $z_v^{PV}$: exact-zero fraction at the actual V operand of PV;
- $z_a,z_m,z_h$: exact-zero fractions at the existing attention-input, MLP-input, and MLP-hidden gates;
- $R_{QK}^{0}$: direct fraction of valid QK scalar products for which Q or K is exactly zero;
- $R_{PV}^{0}$: direct fraction of valid PV scalar products for which P or V is exactly zero;
- $R_{\mathrm{block}}$: logical zero-product fraction across QKV, valid-causal QK, valid-causal PV, $W_O$, $W_1$, and $W_2$; and
- $R_{\mathrm{model}}$: the corresponding six-block fraction with the final LM head in the denominator.

Use direct union counts:

$$
R_{QK}^{0}
=
\frac{
\sum_{b,l,h,t}\sum_{s=0}^{t}\sum_r
\mathbf{1}[Q_{blhtr}=0\;\lor\;K_{blhsr}=0]
}{B L H d_h T(T+1)/2},
$$

$$
R_{PV}^{0}
=
\frac{
\sum_{b,l,h,t}\sum_{s=0}^{t}\sum_r
\mathbf{1}[P_{blhts}=0\;\lor\;V_{blhsr}=0]
}{B L H d_h T(T+1)/2}.
$$

Do not replace these with $z_Q+z_K-z_Qz_K$ or $z_P+z_V-z_Pz_V$ unless explicitly labeled as an independence approximation.

Also report:

- PRE rotary zero preservation, repopulation, and creation rates, split into rotary eight and pass-through 24 dimensions;
- all-zero rotary-pair fraction;
- Q/K/V norms and distributions by layer;
- score mean, standard deviation, quantiles, and negative fraction;
- valid P entropy, maximum probability, and effective support by layer/head;
- context exact-zero fraction and the resulting direct $W_O$ zero-product opportunity;
- pressure, gradient, projection, and step-budget diagnostics; and
- tokens/s as observed dense execution only, not sparse speedup.

After the absolute table, report three matched comparisons:

1. PRE AdamW versus POST AdamW: placement architecture effect;
2. OR and OL1 versus the AdamW baseline with the same placement: pressure effect; and
3. PRE versus POST within the same method: placement effect under the same pressure family.

Configs `98`, `103`, and `99` are contextual Three-ReLU references. They are not matched controls because they lack the new gates and pressure a different site set.

## 13. Success, Falsification, and Interpretation Boundary

Architecture success requires PRE or POST AdamW to retain tolerable validation quality relative to existing Three-ReLU AdamW while exposing new QK/PV logical opportunities. `TODO:` predeclare an acceptable validation-loss delta before inspecting the endpoints; do not invent it after the runs.

Mechanism success requires OR or OL1 to increase direct $R_{QK}^{0}$, $R_{PV}^{0}$, and preferably $R_{\mathrm{model}}$ relative to the AdamW control with the same placement at a favorable validation-loss tradeoff.

Specific falsification outcomes:

- PRE gate sparsity rises but RoPE repopulates enough Q/K coordinates that $R_{QK}^{0}$ barely changes.
- POST creates larger direct QK opportunity but causes disproportionate validation loss, score distortion, or attention-head collapse.
- V gating or pressure produces little causally weighted $R_{PV}^{0}$.
- Pressure merely redistributes sparsity away from the older gates without improving full-model logical opportunity.
- Both AdamW architectures degrade substantially, indicating that simultaneous Q+K+V ReLU is too aggressive. The fallback is separate V-only, Q-only, and K-only architecture controls.

This remains a one-seed planning screen unless repeated. Logical zero products are not measured speedups. Dense SDPA can execute all nominal products even when operands are exactly zero. No runtime-efficiency claim is allowed without a zero-aware kernel benchmark.

## 14. Explicitly Deferred Work

- Learnable-threshold P gating from Section 11.
- Pair-structured or mask-preserving RoPE variants.
- Context ReLU before $W_O$.
- Q-only, K-only, and V-only runs unless the combined gate fails.
- Pressure across all six ReLU site families.
- RN and L1N controls.
- Post-hoc Q/K/V clipping frontiers, unless needed to interpret the six endpoints.
- Multi-seed confirmation and larger Pythia models.
- Sparse QK/PV/W_O kernel implementation and speed measurement.

## 15. Fixed Symmetric-Threshold POST Follow-up

The next two-run test keeps the existing POST topology but replaces only the three Q/K/V ReLU modules with

$$
s_{\kappa}(x)
=
\begin{cases}
x, & |x|\geq\kappa,\\
0, & |x|<\kappa,
\end{cases}
\qquad \kappa=0.1.
$$

This is a signed hard magnitude threshold: negative and positive survivors are preserved, equality survives, and zero means direct floating-point equality. It is not an ordinary ReLU, an RMS-normalized threshold, or a straight-through estimator. The attention-input, MLP-input, and MLP-hidden gates remain ordinary ReLUs.

The tested path in every layer is

$$
(Q^{\rm raw},K^{\rm raw},V^{\rm raw})=\operatorname{split}(W_{QKV}A_l+b_{QKV}),
$$

$$
(Q^{\rm rope},K^{\rm rope})=\operatorname{RoPE}(Q^{\rm raw},K^{\rm raw}),
$$

$$
Q^g=s_{0.1}(Q^{\rm rope}),\qquad
K^g=s_{0.1}(K^{\rm rope}),\qquad
V^g=s_{0.1}(V^{\rm raw}),
$$

followed by the unchanged QK, causal softmax, PV, and output-projection computations.

| Config | Method | Pressure scope | Fixed settings |
| --- | --- | --- | --- |
| `118` | AdamW monitor-only | none | POST Q/K/V $s_{0.1}$ |
| `119` | OR | Q/K/V gate outputs only | weight 1, $c=\sigma=0.05$, step budget 0.5 |

The primary question is whether signed magnitude thresholding preserves quality better than the matched ordinary POST ReLU while creating useful exact-zero Q/K/V operands. Compare config `118` with ordinary POST AdamW config `110`, and config `119` with ordinary POST OR config `111` and fixed-threshold config `118`.

Interpret OR carefully. With $c=\sigma=0.05$, the inward Ricker basin ends at $\sqrt{3}c\approx0.0866<\kappa$. Values masked by $s_{0.1}$ have zero gate gradient; surviving values begin outside that basin. The approved OR run is therefore a matched interaction control and may polarize survivors rather than directly move them across the threshold.

The later diagnostic should report the standard validation-loss, exact-zero, $R_{\rm block}$, and $R_{\rm model}$ table. To test the symmetry/energy rationale directly, also report Q/K/V pre-gate negative and positive mass, $\Pr(|x|<0.1)$, and retained squared-energy fraction.

- `TODO:` test learnable $\kappa$.
- `TODO:` add the matched OL1 run.
- `TODO:` design the full-validation propagation/distribution config after configs `118` and `119` complete; do not mutate pinned Report 05 diagnostics.
