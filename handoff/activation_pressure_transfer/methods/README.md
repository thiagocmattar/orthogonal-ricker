# Method Map

This repository has two families of activation-sparsity mechanisms:

1. Training-time activation pressure: add or apply a differentiable pressure computed from intermediate activations.
2. Hard-zero activation clipping: explicitly set small activations to zero during a forward pass or during post-hoc checkpoint evaluation.

The four requested pressure variants differ along two axes:

| Method | Pressure scalar | Optimizer interaction | Projection |
| --- | --- | --- | --- |
| Ricker naive | `1 - mean(r(a))` | Added directly to training loss | None |
| L1 naive | `mean(abs(a))` | Added directly to training loss | None |
| Adam-step orthogonal Ricker | `1 - mean(r(a))` | AdamW moments see task gradient only; pressure is applied after AdamW step | Project only if pressure update conflicts with Adam task update |
| Adam-step orthogonal L1 | `mean(abs(a))` | AdamW moments see task gradient only; pressure is applied after AdamW step | Project only if pressure update conflicts with Adam task update |

Ricker score:

```text
r(a; c, sigma) = (1 - a^2 / c^2) exp(-a^2 / (2 sigma^2))
Ricker pressure = 1 - mean_sites mean_elements r(a)
```

L1 score:

```text
L1 pressure = mean_sites mean_elements abs(a)
```

Canonical baseline:

```json
{
  "sparsity_alm": {
    "enabled": true,
    "target_sparsity": 0.0,
    "small_activation_weight": 0.0,
    "sparsity_weight": 1.0,
    "c_ricker": 0.05,
    "sigma_ricker": 0.05,
    "task_safe_ricker_gradients": {
      "enabled": true,
      "mode": "task_only_adamw_step_orthogonal_ricker",
      "ricker_step_budget": 1.5
    }
  }
}
```

Orthogonality clarification:

In the Adam-step orthogonal modes, projection is not a generic "make activation pressure orthogonal to task loss" rule. AdamW first takes a normal task-only step, so AdamW first and second moments contain only task gradients. Then the pressure correction is converted into the AdamW-preconditioned update space and compared to AdamW's task update direction. Projection fires only when their dot product is negative.

```text
d_task = m_hat_task / (sqrt(v_hat_task) + adam_eps)
d_pressure = g_pressure / (sqrt(v_hat_task) + adam_eps)
dot = <d_task, d_pressure>

if dot < 0:
    d_pressure_safe = d_pressure - dot / (||d_task||^2 + eps) * d_task
else:
    d_pressure_safe = d_pressure
```

After projection, a trust budget limits the weighted correction norm relative to the task-step norm:

```text
raw_ratio = pressure_weight * ||d_pressure_safe|| / (||d_task|| + eps)
scale = min(1, step_budget / (raw_ratio + eps))
theta <- theta - lr * pressure_weight * scale * d_pressure_safe
```

This differs from `safe_combined_gradient`, which projects a raw gradient conflict before AdamW sees the combined gradient. The Adam-step orthogonal methods are the transfer target for current Ricker/L1 work.

Related but not main transfer targets:

- `safe_combined_gradient`: historical raw-gradient task-safe Ricker.
- `task_only_adamw_safe_ricker_step`: historical task-only, preconditioned-gradient comparator.
- `task_only_adamw_step_orthogonal_ricker_self_preconditioned`: Ricker-owned EMA second-moment preconditioner.
- `task_only_adamw_step_orthogonal_ricker_current_preconditioned`: current-gradient Ricker preconditioner.
- `small_activation_weight` / `small_activation_ramp`: auxiliary ramp pressure around a small threshold.

