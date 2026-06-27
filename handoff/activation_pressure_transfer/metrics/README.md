# Additional Measures

This folder specifies the extra measurements to carry with activation-pressure
methods. It intentionally does not restate base training metrics or default
diagnostics.

## Task And Pressure Separation

For every activation-pressure method, log the task objective and pressure
objective as separate quantities.

Recommended fields:

- `task_loss`: task cross-entropy used for quality comparison.
- `pressure_loss`: the scalar pressure being applied, for example Ricker
  pressure `1 - mean(r(a))` or activation L1 pressure `mean(abs(a))`.
- `pressure_weight`: configured pressure multiplier.
- `weighted_pressure_loss`: `pressure_weight * pressure_loss`.
- `augmented_loss`: `task_loss + weighted_pressure_loss` for direct-loss
  methods.

For Adam-step orthogonal methods, `augmented_loss` is only a monitoring
quantity. AdamW moments should still receive task gradients only, with the
pressure correction applied after the AdamW task step.

## Task And Pressure Gradient Metrics

Log task and pressure gradients separately at the same optimizer-step boundary.
Use the same accumulated batch boundary that produces the actual update.

Recommended fields:

- `task_gradient_norm`: L2 norm of the task-loss gradient.
- `pressure_gradient_norm`: L2 norm of the pressure-gradient source before any
  projection or trust-budget scaling.
- `pressure_to_task_gradient_norm_ratio`: `pressure_gradient_norm /
  task_gradient_norm`.
- `task_pressure_gradient_dot`: raw dot product between task and pressure
  gradients.
- `task_pressure_gradient_cosine`: cosine between task and pressure gradients.
- `pressure_conflict`: whether the task/pressure dot product is negative.

For Adam-step orthogonal methods, also monitor the update-space separation:

- `task_update_direction_norm`: norm of the AdamW task update direction.
- `pressure_update_direction_norm_raw`: norm of the pressure direction after
  AdamW second-moment preconditioning and before projection.
- `task_pressure_update_dot_before`: dot product before projection.
- `task_pressure_update_cosine_before`: cosine before projection.
- `pressure_update_projected`: whether projection fired.
- `task_pressure_update_dot_after`: dot product after projection.
- `task_pressure_update_cosine_after`: cosine after projection.
- `pressure_update_ratio_raw`: weighted pressure-update norm divided by task
  update norm before trust-budget scaling.
- `pressure_update_applied_scale`: trust-budget scale applied to the pressure
  update.
- `pressure_update_ratio_final`: final weighted pressure-update norm divided by
  task update norm after scaling.

## Target-Site Near-Zero Mass

At train time, measure target-site near-zero activation mass during evaluation
steps. Use the same monitored activation sites as the pressure method.

For each evaluation step and each site group, log:

- `target_site_near_zero_mass_k0`: mean indicator `abs(a) <= 0`.
- `target_site_near_zero_mass_k1e-3`: mean indicator `abs(a) <= 1e-3`.
- `target_site_near_zero_mass_k1e-2`: mean indicator `abs(a) <= 1e-2`.

Also log an aggregate over all target sites with the same three thresholds.
The `k1e-2` quantity matches the target-site near-zero mass definition used in
`report/20260530_task_only_ricker_method/main.tex`; the added `k0` and
`k1e-3` thresholds distinguish exact zeros from merely small activations.
