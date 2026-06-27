# Model Porting Contract

An agent can port the methods to a new model or architecture if the training loop exposes the interfaces below. Without these interfaces, the handoff is only a conceptual description.

Minimum model interface:

- A normal task loss, for example cross-entropy, regression loss, or sequence loss.
- A dictionary of differentiable activation tensors keyed by stable site names.
- The activation tensors must remain attached to the autograd graph for training-time pressure.
- A way to apply optional hard clipping to selected activation tensors during train, eval, and post-hoc checkpoint sweeps.
- Per-run logs containing selected sites, tensor shapes, pressure scalar, task loss, hard-zero fractions, and method parameters.

Activation capture options:

- Return activations directly from `forward(..., return_activations=True)`.
- Register module forward hooks that store tensors in an activation cache.
- Wrap individual submodules, for example MLP activation, attention projection, or convolution feature blocks.

Do not detach activation tensors before pressure computation. Detach only for logging, threshold selection, or quantile calculation.

Architecture mapping:

- Transformer MLP: target hidden activations after nonlinearity first; these map cleanly to MLP down-projection sparse GEMV/GEMM.
- Transformer residual stream: possible, but more invasive because every downstream block consumes the changed representation.
- Transformer attention `q/k/v`: possible, but speedup requires a sparse projection or attention-specific route.
- Attention scores/probabilities: treat as a separate method family because clipping or pressure changes attention semantics directly.
- CNNs and vision models: target feature maps or channel activations, but define whether sparsity is elementwise, channelwise, spatial, or blockwise before making compute claims.
- Mixture-of-experts or routed models: activation pressure is not routing unless explicit gates, top-k selection, expert skipping, or route instrumentation are present.

Optimizer requirements:

- Naive Ricker and naive L1 work with any differentiable optimizer because the pressure is part of the loss.
- Adam-step orthogonal Ricker and Adam-step orthogonal L1 require AdamW-compatible moments and a visible optimizer step boundary.
- For SGD, Lion, Adafactor, Shampoo, or other optimizers, define a new projection space. Do not call the result "Adam-step orthogonal" unless it uses AdamW's first and second moment preconditioner.

Training-loop requirements for Adam-step orthogonal methods:

1. Compute task loss and clone task gradients.
2. Compute pressure loss and clone pressure gradients.
3. Restore task gradients.
4. Execute one AdamW task-only `optimizer.step()`.
5. Apply the memoryless pressure correction using AdamW state from that task step.
6. Log projection dot products, raw ratio, final ratio, scale, and pressure kind.

Validation requirements:

- `weight = 0` must match dense AdamW within ordinary numerical tolerance.
- Naive pressure should produce nonzero pressure gradients on the selected sites.
- AdamW moments must not change when only the post-step pressure correction is varied.
- Projection should activate only when the Adam-step-space dot product is negative.
- After projection, the pressure direction dot product should be nonnegative up to numerical tolerance.
- The trust budget should cap `final_ratio <= step_budget` when a budget is set.
- Clipping should report achieved exact-zero fraction and validation loss for every threshold or quantile.

Claim boundaries:

- "Works for any architecture" means the math can be applied to any differentiable activation tensor once these interfaces exist. It does not mean the same sites, thresholds, weights, or speedup results transfer unchanged.
- Endpoint quality, sparsity frontier, and latency are separate claims and need separate controls.

