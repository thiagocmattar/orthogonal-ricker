# Adam-Step Orthogonal Ricker

This is the canonical Ricker method in the active harness:

```text
sparsity_alm.task_safe_ricker_gradients.mode = "task_only_adamw_step_orthogonal_ricker"
```

It is the default meaning of "task-only Ricker", "safe Ricker", and "Ricker baseline" in current work unless a historical comparator is explicitly requested.

Core idea:

- AdamW sees only task gradients.
- Ricker pressure gradients are computed separately.
- After the AdamW task step, a memoryless Ricker correction is applied.
- The correction is represented in AdamW's task preconditioned step space.
- Projection is applied only if the Ricker correction has a negative dot product with AdamW's task update direction.
- A trust budget caps the weighted correction norm relative to the task-step norm.

Math:

```text
p_ricker = 1 - mean_sites mean_elements r(a; c, sigma)
g_task = grad_theta L_task
g_ricker = grad_theta p_ricker

After AdamW task step:
d_task = m_hat_task / (sqrt(v_hat_task) + adam_eps)
d_ricker = g_ricker / (sqrt(v_hat_task) + adam_eps)
dot = <d_task, d_ricker>

if dot < 0:
    d_ricker_safe = d_ricker - dot / (||d_task||^2 + eps) * d_task
else:
    d_ricker_safe = d_ricker

raw_ratio = w * ||d_ricker_safe|| / (||d_task|| + eps)
scale = min(1, budget / (raw_ratio + eps))
theta <- theta - lr * w * scale * d_ricker_safe
```

The projection condition is global across eligible parameters in the repository helper. It is not applied per activation tensor, and it is not based on the scalar task loss. It checks conflict between the pressure update and the AdamW task update in Adam-step space.

Repository source:

- `src/lm_harness/train.py`: training-loop ordering and mode selection.
- `src/lm_harness/task_safe_gradients.py`: `apply_task_only_adamw_step_orthogonal_ricker_step`.
- `src/lm_harness/sparsity_alm.py`: Ricker pressure and config fields.
- `docs/sparsity_alm_ricker_handoff.md`: prior Ricker-specific handoff.

Config skeleton:

```json
{
  "sparsity_alm": {
    "enabled": true,
    "sites": ["mlp_hiddens"],
    "target_sparsity": 0.0,
    "rho": 0.0,
    "sparsity_weight": 1.0,
    "c_ricker": 0.05,
    "sigma_ricker": 0.05,
    "small_activation_weight": 0.0,
    "task_safe_ricker_gradients": {
      "enabled": true,
      "mode": "task_only_adamw_step_orthogonal_ricker",
      "ricker_step_budget": 1.5,
      "eps": 1e-12
    }
  }
}
```

Usual parameters and intervals:

- `c_ricker`: commonly `0.05`; sweep roughly `0.03`, `0.05`, `0.08`.
- `sigma_ricker`: commonly tied to `c_ricker`; sweep independently only when studying shape sensitivity.
- `sparsity_weight`: common probes around `0.5`, `1.0`, `2.0`, `3.0`.
- `ricker_step_budget`: common probes around `0.25`, `0.5`, `1.0`, `1.5`, `2.5`; the canonical Rung 2/Rung 3 baseline often used `1.5`.
- `sites`: keep fixed inside a suite. Start with `mlp_hiddens` or `all_sites`.

Expected result and resistance:

- This method was the strongest Ricker transfer target because AdamW endpoint learning is protected from pressure gradients entering Adam moments.
- It can produce high exact-zero thresholding tolerance, especially compared with naive pressure.
- It was not established as a seed-robust endpoint-quality win over tuned dense AdamW. Treat endpoint quality and sparsity frontier as separate claims.
- If `dot >= 0`, projection does nothing; the trust budget can still scale the correction.

Diagnostics to preserve:

- Dot before/after projection in Adam-step space.
- Projection count or projection-active flag.
- Raw and final pressure/task step ratio.
- Hard-zero fraction at relevant thresholds.
- Activation clipping sweep against the same checkpoint.

