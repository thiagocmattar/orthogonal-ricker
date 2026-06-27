# Ricker Naive

Naive Ricker adds a differentiable activation pressure directly to the training loss with no task-safety projection. It is useful as a comparator and stress test, not as the current recommended method.

Math:

```text
r(a; c, sigma) = (1 - a^2 / c^2) exp(-a^2 / (2 sigma^2))
s = mean_sites mean_elements r(a)
L = L_task + w * (1 - s)
```

Implementation snippet:

```python
def soft_zero_ricker(x, c, sigma):
    z = x.float()
    return ((1.0 - z.square() / (c * c)) * torch.exp(-z.square() / (2.0 * sigma * sigma))).mean()


ricker_soft = torch.stack([soft_zero_ricker(acts[name], c, sigma) for name in sites]).mean()
loss = task_loss + weight * (1.0 - ricker_soft)
loss.backward()
optimizer.step()
```

Repository source:

- `src/lm_harness/sparsity_alm.py`: `soft_zero_ricker`, `SparsityALM.augmented_loss`.

Parameters:

- `c_ricker`: zero-crossing radius and width of the positive central lobe. Usual tested values were around `0.03` to `0.08`, with common baseline value `0.05`. Transfer sweeps can extend to larger exploratory values such as `0.1`, `0.2`, and `0.3`; mark those as exploratory unless a matched suite has tested them.
- `sigma_ricker`: Gaussian envelope width. Usually tied to `c_ricker`; common baseline value `0.05`. A simple tied sweep is `c = sigma in {0.03, 0.05, 0.08, 0.1, 0.2, 0.3}`.
- `sparsity_weight`: direct pressure weight. Transfer with a small sweep; Ricker experiments commonly used around `0.5` to `3.0` depending on rung and site.

Shape notes:

- The Ricker score crosses zero at `abs(a) = c`. This zero crossing is set by `c`, not by `sigma`.
- The score is positive for `abs(a) < c`, negative for `abs(a) > c`, and approaches `0` from below in the far tail. The score does not become positive again after `sigma`.
- The pressure is `1 - r(a)`. Its local gradient has a negative trough at `abs(a) = sqrt(c^2 + 2 sigma^2)`. Outside that trough, the pressure-gradient direction reverses and can push already-large activations farther outward, although the exponential envelope may make that effect small.
- Larger `sigma` broadens the negative lobe and moves that gradient reversal farther from zero. Treat large `c = sigma` values such as `0.1` to `0.3` as explicit shape-sensitivity probes, not established defaults.

Applicability:

- Works on any collected activation tensor.
- Better first targets are MLP hidden activations and other tensors that can plausibly be thresholded before a linear operation.
- Broad residual-site pressure is more invasive and should be treated as a separate suite axis.

Expected result and resistance:

- Ricker can create a gap around zero and high threshold tolerance when tuned.
- Naive Ricker can also over-regularize the network, suppress learning, and produce premature convergence or high-loss plateaus.
- A Ricker-only improvement in near-zero activation mass is not speedup evidence without an explicit sparse operator or clipping path.

Recommended controls:

- Dense AdamW with the same model, data, seeds, LR, and token budget.
- Adam-step orthogonal Ricker with the same `c`, `sigma`, sites, and pressure weight.
