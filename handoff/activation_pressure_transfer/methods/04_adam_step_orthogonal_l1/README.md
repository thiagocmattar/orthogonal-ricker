# Adam-Step Orthogonal L1

This method uses the same task-only Adam-step projection machinery as canonical Ricker, but the pressure scalar is direct activation L1:

```text
sparsity_alm.task_safe_ricker_gradients.mode = "task_only_adamw_step_orthogonal_activation_l1"
```

Math:

```text
p_l1 = mean_sites mean_elements abs(a)
g_task = grad_theta L_task
g_l1 = grad_theta p_l1

After AdamW task step:
d_task = m_hat_task / (sqrt(v_hat_task) + adam_eps)
d_l1 = g_l1 / (sqrt(v_hat_task) + adam_eps)
dot = <d_task, d_l1>

if dot < 0:
    d_l1_safe = d_l1 - dot / (||d_task||^2 + eps) * d_task
else:
    d_l1_safe = d_l1

raw_ratio = w * ||d_l1_safe|| / (||d_task|| + eps)
scale = min(1, budget / (raw_ratio + eps))
theta <- theta - lr * w * scale * d_l1_safe
```

Orthogonality clarification:

Projection fires only when the L1 pressure update conflicts with AdamW's task update direction after AdamW preconditioning. This is not a raw task-gradient check and not a check against the scalar task loss.

Repository source:

- `src/lm_harness/sparsity_alm.py`: `activation_l1_pressure` and direct pressure kind selection.
- `src/lm_harness/train.py`: routes this mode through the same helper as Adam-step orthogonal Ricker.
- `src/lm_harness/task_safe_gradients.py`: `apply_task_only_adamw_step_orthogonal_ricker_step`; the helper name is historical, but it accepts any pressure gradient sequence.

Config skeleton:

```json
{
  "sparsity_alm": {
    "enabled": true,
    "sites": ["mlp_hiddens"],
    "target_sparsity": 0.0,
    "rho": 0.0,
    "sparsity_weight": 0.5,
    "small_activation_weight": 0.0,
    "task_safe_ricker_gradients": {
      "enabled": true,
      "mode": "task_only_adamw_step_orthogonal_activation_l1",
      "ricker_step_budget": 0.5,
      "eps": 1e-12
    }
  }
}
```

Usual parameters and intervals:

- `sparsity_weight`: sweep from small values such as `0.05`, `0.1`, `0.25`, `0.5` through larger values if the task loss is robust.
- `ricker_step_budget`: same field name is reused for L1; common probes include `0.25`, `0.5`, `1.0`, `1.5`.
- `c_ricker` and `sigma_ricker`: irrelevant to the L1 pressure scalar, though the config object may still contain defaults.
- `sites`: start with `mlp_hiddens`; transfer residual and attention sites only as explicit suite axes.

Expected result and resistance:

- L1 shrinks activation magnitudes and may be gentler than Ricker for endpoint loss.
- It may not create as sharp a gap around zero as Ricker, so high-threshold exact-zero inference frontiers can be weaker.
- The orthogonal version avoids pressure gradients contaminating AdamW's task moments.

Diagnostics to preserve:

- `sparsity_alm_task_safe_ricker_pressure_kind = "activation_l1"` or equivalent.
- Mean activation L1 by site and global.
- Adam-step projection dot/ratio metrics.
- Thresholded validation loss and achieved activation sparsity sweeps.

