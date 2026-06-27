# Activation Sites

The pressure and clipping methods operate on named activation tensors. A transfer should implement stable site names for the target architecture and log the resolved names in every run.

For a new architecture, define each site with this metadata:

- `name`: stable string used in configs, metrics, and reports.
- `module_path`: model module or hook location that emits the tensor.
- `role`: residual, MLP input, MLP hidden, MLP output, attention query/key/value, attention score/probability, convolution feature map, or another explicit role.
- `shape`: expected tensor shape convention, for example `[batch, seq, hidden]` or `[batch, channels, height, width]`.
- `downstream_operator`: the dense operation that could consume exact zeros if speedup is claimed.
- `train_eval_scope`: whether the site is captured during train, eval, post-hoc checkpoint sweeps, or inference only.

The names below are the conventions from this repository's transformer harness.

Residual stream sites:

- `embedding`
- `block_N`, for example `block_0`
- `blocks`, all block output residual streams
- `final_ln`
- `all`, residual stream alias for `embedding`, all block outputs, and `final_ln`

MLP sites:

- `mlp_inputs` / `V_l`
- `mlp_hiddens` / `G_l`
- `mlp_outputs` / `M_l`

Attention sites:

- `attention_inputs` / `U_l`
- `q`, `k`, `v`, `qkv`
- `attention_scores` / `a`
- `attention_probs` / `B_lm`
- `attention_values` / `V_lm`
- `attention`

Broad aliases:

- `all_sites`: residual stream sites plus MLP hidden activations.
- `everything` / `all_sparse_sites`: residual stream sites, MLP hidden activations, and attention `q/k/v`. The harness intentionally does not include attention scores/probabilities in this broad sparse-site alias.

Usual starting sites:

- For activation pressure, start with `mlp_hiddens` or `all_sites`.
- For hard clipping and speedup experiments, start with MLP hidden or MLP input-like surfaces that map clearly to a downstream linear layer.
- For language-model evidence, keep the site set fixed inside a suite. A new site set changes the comparison boundary.

Expected resistance:

- Residual-stream pressure can interfere with broad information flow and may hurt endpoint quality quickly.
- MLP hidden pressure is easier to connect to GEMV/GEMM sparsity and post-hoc thresholding.
- Attention probability/scores need extra care because masking them changes attention semantics directly.
