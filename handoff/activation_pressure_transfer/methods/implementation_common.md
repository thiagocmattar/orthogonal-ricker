# Common Implementation Notes

Core pressure functions:

```python
def ricker_score(x, c: float, sigma: float):
    z = x.float()
    return ((1.0 - z.square() / (c * c)) * torch.exp(-z.square() / (2.0 * sigma * sigma))).mean()


def ricker_pressure(activations, sites, c: float, sigma: float):
    return 1.0 - torch.stack([ricker_score(activations[name], c, sigma) for name in sites]).mean()


def activation_l1_pressure(activations, sites):
    return torch.stack([activations[name].float().abs().mean() for name in sites]).mean()
```

Direct pressure setup:

- The simple direct-pressure variants are exactly `task_loss + weight * pressure`.
- Do not include stochastic pressure gates, ALM targets, or ramp auxiliaries unless deliberately studying those as separate methods.
- Log both soft pressure and hard-zero fraction at one or more thresholds, for example `1e-3`, `1e-2`, and the intended inference threshold.
- Always log endpoint task loss/perplexity separately from sparsity metrics.

Adam-step orthogonal setup:

- Compute task loss and pressure loss separately.
- Backprop task loss first and save task gradients.
- Set parameter `.grad` to task gradients and call `optimizer.step()` so AdamW state is task-only.
- Backprop pressure loss separately to get pressure gradients.
- Apply the memoryless pressure correction after the AdamW step using AdamW's task state.

Minimal ordering sketch:

```python
optimizer.zero_grad(set_to_none=True)
task_loss.backward(retain_graph=True)
task_grads = [None if p.grad is None else p.grad.detach().clone() for p in params]

optimizer.zero_grad(set_to_none=True)
pressure_loss.backward()
pressure_grads = [None if p.grad is None else p.grad.detach().clone() for p in params]

for p, g in zip(params, task_grads):
    p.grad = None if g is None else g.clone()
optimizer.step()

apply_adam_step_orthogonal_pressure(
    optimizer,
    params,
    task_grads,
    pressure_grads,
    pressure_weight=weight,
    step_budget=budget,
)
```

Do not feed pressure gradients into AdamW moments for the Adam-step orthogonal methods. That separation is the point of the method.

Reference Adam-step orthogonal helper:

```python
import torch


def _optimizer_step_as_float(step):
    if torch.is_tensor(step):
        return float(step.detach().cpu().item())
    return float(step)


@torch.no_grad()
def apply_adam_step_orthogonal_pressure(
    optimizer,
    params,
    task_grads,
    pressure_grads,
    *,
    pressure_weight: float,
    step_budget: float | None,
    eps: float = 1e-12,
):
    """Apply memoryless pressure after a task-only AdamW step.

    Call order:
    1. Backprop task loss and clone `task_grads`.
    2. Backprop pressure loss and clone `pressure_grads`.
    3. Restore task grads and call `optimizer.step()`.
    4. Call this helper.

    This matches the harness behavior for Adam-step orthogonal Ricker/L1.
    Decoupled AdamW weight decay is applied by AdamW itself, but is not included
    in the projection direction below.
    """

    if len(params) != len(task_grads) or len(params) != len(pressure_grads):
        raise ValueError("params, task_grads, and pressure_grads must have equal length")
    if step_budget is not None and step_budget < 0.0:
        raise ValueError("step_budget must be non-negative or None")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    param_index = {id(p): i for i, p in enumerate(params)}
    corrections = []
    task_direction_sq = 0.0
    pressure_direction_sq = 0.0
    dot_before = 0.0
    skipped = 0

    for group in optimizer.param_groups:
        lr = float(group.get("lr", 0.0))
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        beta1 = float(beta1)
        beta2 = float(beta2)
        adam_eps = float(group.get("eps", 1e-8))
        amsgrad = bool(group.get("amsgrad", False))

        for p in group["params"]:
            i = param_index.get(id(p))
            if i is None or task_grads[i] is None or pressure_grads[i] is None:
                skipped += 1
                continue

            state = optimizer.state.get(p, {})
            if "exp_avg" not in state or "exp_avg_sq" not in state or "step" not in state:
                skipped += 1
                continue

            step = _optimizer_step_as_float(state["step"])
            if step <= 0.0:
                skipped += 1
                continue

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            if bias_correction1 <= 0.0 or bias_correction2 <= 0.0:
                skipped += 1
                continue

            exp_avg = state["exp_avg"].detach().to(device=p.device).float()
            exp_avg_sq = state["exp_avg_sq"]
            if amsgrad and "max_exp_avg_sq" in state:
                exp_avg_sq = state["max_exp_avg_sq"]
            denom = exp_avg_sq.detach().to(device=p.device).float()
            denom = denom.div(bias_correction2).sqrt().add_(adam_eps)

            d_task = exp_avg.div(bias_correction1).div(denom)
            d_pressure = pressure_grads[i].detach().to(device=p.device).float().div(denom)

            task_direction_sq += float(d_task.square().sum().item())
            pressure_direction_sq += float(d_pressure.square().sum().item())
            dot_before += float((d_task * d_pressure).sum().item())
            corrections.append((p, d_task, d_pressure, lr))

    should_project = dot_before < 0.0 and task_direction_sq > eps
    coeff = dot_before / (task_direction_sq + eps) if should_project else 0.0

    safe_corrections = []
    safe_direction_sq = 0.0
    dot_after = 0.0
    for p, d_task, d_pressure, lr in corrections:
        d_safe = d_pressure - coeff * d_task if should_project else d_pressure
        safe_direction_sq += float(d_safe.square().sum().item())
        dot_after += float((d_task * d_safe).sum().item())
        safe_corrections.append((p, d_safe, lr))

    task_norm = task_direction_sq**0.5
    safe_norm = safe_direction_sq**0.5
    raw_ratio = pressure_weight * safe_norm / (task_norm + eps)
    scale = 1.0
    if step_budget is not None and raw_ratio > 0.0:
        scale = min(1.0, step_budget / (raw_ratio + eps))

    if pressure_weight != 0.0 and scale != 0.0:
        for p, d_safe, lr in safe_corrections:
            p.add_(d_safe.to(device=p.device, dtype=p.dtype), alpha=-lr * pressure_weight * scale)

    return {
        "projected": should_project,
        "dot_before": dot_before,
        "dot_after": dot_after,
        "raw_ratio": raw_ratio,
        "final_ratio": raw_ratio * scale,
        "scale": scale,
        "eligible_parameters": len(corrections),
        "skipped_parameters": skipped,
        "raw_pressure_norm": pressure_direction_sq**0.5,
        "safe_pressure_norm": safe_norm,
        "task_norm": task_norm,
    }
```

Porting constraints:

- This helper assumes AdamW-compatible optimizer state with `exp_avg`, `exp_avg_sq`, and `step` per parameter. For fused, sharded, or custom optimizers, expose equivalent moments or reimplement the same equations inside that optimizer.
- If using mixed precision, unscale gradients before cloning `task_grads` and `pressure_grads`.
- With gradient accumulation, apply the pressure correction only on real optimizer steps and make task/pressure gradients correspond to the same accumulated batch boundary.
- With DDP/FSDP, compute the projection dot product over the globally reduced task and pressure gradients, not separate local shards, unless the optimizer itself is shard-local and the approximation is intentional.
