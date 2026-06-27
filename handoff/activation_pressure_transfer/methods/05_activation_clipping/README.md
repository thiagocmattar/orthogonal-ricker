# Activation Clipping

Activation clipping is a hard-zero mechanism. It is not an optimizer regularizer by itself; it changes the forward pass by replacing small activation values with exact zeros.

Modes:

```text
threshold: zero values with abs(a) <= threshold
quantile: compute a per-forward threshold at the requested abs-value quantile, then zero values below it
```

Repository implementation:

```python
def clip_activation_tensor(value, cfg):
    if cfg.mode == "threshold":
        return value.masked_fill(value.abs() <= cfg.threshold, 0.0)
    if cfg.mode == "quantile":
        threshold = value.detach().abs().reshape(-1).float().kthvalue(k).values
        return value.masked_fill(value.abs() <= threshold, 0.0)
```

Source:

- `src/lm_harness/model.py`: `ActivationClippingConfig`, `clip_activation_tensor`, model call sites.
- `src/lm_harness/train.py`: `activation_clipping_config_from_dict`, evaluation path.
- `src/lm_harness/checkpoint_eval.py`: post-hoc clipping sweeps.

Training/eval config skeleton:

```json
{
  "activation_clipping": {
    "enabled": true,
    "mode": "threshold",
    "sites": ["mlp_hiddens"],
    "threshold": 0.01
  }
}
```

Quantile config skeleton:

```json
{
  "activation_clipping": {
    "enabled": true,
    "mode": "quantile",
    "sites": ["mlp_hiddens"],
    "quantile": 0.90
  }
}
```

Usual thresholds:

- Diagnostic hard-zero fractions: `1e-3`, `1e-2`, and method-specific thresholds.
- Ricker/L1 clipping frontiers often need a sweep rather than one threshold, for example `0.0`, `0.001`, `0.003`, `0.01`, `0.03`, `0.05`, `0.08`, `0.10`.
- Quantile clipping is useful for fixed achieved sparsity probes, but it is harder to map to a static inference kernel than a fixed threshold.

Applicability:

- Threshold clipping is easiest to connect to fixed-threshold sparse kernels.
- It should target tensors whose zeros can be consumed by a sparse operator. MLP hidden activations are the cleanest starting point.
- Clipping residual streams or attention tensors can alter semantics more broadly and should not be mixed into an MLP-kernel speedup claim.

Expected result and resistance:

- Hard clipping can expose whether a trained model actually tolerates exact zeros.
- A checkpoint may have high near-zero mass but still lose quality at a practical threshold.
- A threshold that gives high achieved sparsity is useful only if the downstream sparse operation wins after indexing, masking, and launch overhead.

Post-hoc sweep recommendation:

1. Evaluate dense checkpoint with clipping disabled.
2. For each site group and threshold, run validation with clipping enabled.
3. Record validation loss, achieved hard-zero fraction, and throughput.
4. Send the best candidate shapes and achieved sparsity to the kernel benchmark.

