# L1 Naive

Naive activation L1 adds mean absolute activation magnitude directly to the task loss. In this repo, the L1 pressure source is implemented and used by the Adam-step orthogonal L1 mode; the naive transfer is the same pressure scalar without the Adam-step projection helper.

Math:

```text
p_l1 = mean_sites mean_elements abs(a)
L = L_task + w * p_l1
```

Implementation snippet:

```python
def activation_l1_pressure(acts, sites):
    return torch.stack([acts[name].float().abs().mean() for name in sites]).mean()


l1 = activation_l1_pressure(acts, sites)
loss = task_loss + weight * l1
loss.backward()
optimizer.step()
```

Repository source:

- `src/lm_harness/sparsity_alm.py`: `activation_l1_pressure`.
- `task_only_adamw_step_orthogonal_activation_l1` selects `activation_l1` as the direct pressure kind when task-safe gradients are enabled.

Parameters:

- `sparsity_weight`: L1 pressure weight. Short-horizon L1 sweeps included values from small `0.05`-style probes through multi-unit weights. Port with a logarithmic or coarse sweep rather than assuming one value.
- This method has one pressure term: `L_task + weight * mean(abs(a))`. Do not include ALM targets, ramp auxiliaries, or Ricker shape parameters in this simple variant.

Applicability:

- Works on any collected activation tensor.
- Most natural on post-nonlinearity MLP hiddens, because lower absolute magnitude directly supports later thresholding.
- Can also be used on residual streams, but that changes the representation everywhere downstream and should be studied separately.

Expected result and resistance:

- L1 tends to shrink magnitudes smoothly rather than create the sharper Ricker-style gap around zero.
- It can preserve endpoint quality better than aggressive Ricker in short runs, but may not produce as strong a high-threshold clipping frontier.
- Naive L1 can still interfere with task learning because the pressure gradient is part of the optimizer update and Adam moments.

Recommended controls:

- Dense AdamW.
- Adam-step orthogonal L1 with the same sites and weight.
- Ricker variants if the target is exact-zero tolerance after thresholding.
