from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from paper_exp.activation_pressure import accumulate_grads
from paper_exp.activation_pressure import activation_near_zero_metrics
from paper_exp.activation_pressure import apply_adam_step_orthogonal_pressure
from paper_exp.activation_pressure import clone_grads
from paper_exp.activation_pressure import grad_metrics
from paper_exp.activation_pressure import pressure_loss


def _build_adamw_optimizer(
    *,
    torch: Any,
    model: Any,
    training: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    optimizer_config = _optimizer_config(training)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=optimizer_config["betas"],
        eps=optimizer_config["eps"],
        weight_decay=optimizer_config["weight_decay"],
    )
    return optimizer, optimizer_config


def _run_training_step(
    *,
    model: Any,
    optimizer: Any,
    params: list[Any],
    torch: Any,
    np: Any,
    train_tokens: Any,
    block_size: int,
    micro_batch_size: int,
    grad_accum: int,
    device: Any,
    dtype: Any,
    pressure_config: Any,
    activation_capture: Any,
    step: int,
    schedule_step: Any = None,
) -> dict[str, Any]:
    if pressure_config.enabled and activation_capture is None:
        raise ValueError("Activation pressure is enabled but no activation capture is registered.")

    if pressure_config.orthogonal:
        return _run_orthogonal_pressure_step(
            model=model,
            optimizer=optimizer,
            params=params,
            torch=torch,
            np=np,
            train_tokens=train_tokens,
            block_size=block_size,
            micro_batch_size=micro_batch_size,
            grad_accum=grad_accum,
            device=device,
            dtype=dtype,
            pressure_config=pressure_config,
            activation_capture=activation_capture,
            step=step,
            schedule_step=schedule_step,
        )

    optimizer.zero_grad(set_to_none=True)
    task_loss_total = 0.0
    pressure_loss_total = 0.0
    activation_metrics: dict[str, float] = {}
    task_grads_for_metrics: list[Any | None] = []
    pressure_grads_for_metrics: list[Any | None] = []
    pressure_active = pressure_config.applies_pressure

    for micro_step in range(grad_accum):
        if activation_capture is not None:
            activation_capture.clear()
        batch = _sample_batch(
            torch,
            np,
            train_tokens,
            block_size,
            micro_batch_size,
            device,
            starts=(None if schedule_step is None else schedule_step[micro_step]),
        )
        with _autocast_context(torch, device, dtype):
            output = model(input_ids=batch, labels=batch)
            task_loss = output.loss
            current_pressure_loss = (
                pressure_loss(torch, activation_capture.activations, pressure_config)
                if pressure_active
                else None
            )
            augmented_loss = (
                task_loss + pressure_config.weight * current_pressure_loss
                if current_pressure_loss is not None
                else task_loss
            )
        _require_finite_loss(torch, task_loss, f"task loss at step {step}")
        _require_finite_loss(torch, augmented_loss, f"training loss at step {step}")
        if pressure_active and current_pressure_loss is not None:
            task_grads_for_metrics = accumulate_grads(
                task_grads_for_metrics,
                torch.autograd.grad(
                    task_loss / grad_accum,
                    params,
                    retain_graph=True,
                    allow_unused=True,
                ),
            )
            pressure_grads_for_metrics = accumulate_grads(
                pressure_grads_for_metrics,
                torch.autograd.grad(
                    current_pressure_loss / grad_accum,
                    params,
                    retain_graph=True,
                    allow_unused=True,
                ),
            )
        (augmented_loss / grad_accum).backward()
        task_loss_total += float(task_loss.detach().cpu())
        if current_pressure_loss is not None:
            _require_finite_loss(torch, current_pressure_loss, f"pressure loss at step {step}")
            pressure_loss_total += float(current_pressure_loss.detach().cpu())
        if activation_capture is not None:
            activation_metrics = activation_near_zero_metrics(
                activation_capture.activations,
                pressure_config.log_thresholds,
            )

    task_grads = task_grads_for_metrics if pressure_active else clone_grads(params)
    pressure_grads = pressure_grads_for_metrics if pressure_active else [None for _ in params]
    step_metrics = grad_metrics(torch, task_grads, pressure_grads)
    optimizer.step()

    task_loss_mean = task_loss_total / grad_accum
    pressure_loss_mean = pressure_loss_total / grad_accum if pressure_active else None
    result = {
        "task_loss": task_loss_mean,
        "pressure/task_gradient_norm": step_metrics["pressure/task_gradient_norm"],
    }
    if pressure_active:
        result.update(step_metrics)
        result.update(
            {
                "pressure_loss": pressure_loss_mean,
                "pressure_weight": pressure_config.weight,
                "weighted_pressure_loss": pressure_config.weight * pressure_loss_mean,
                "augmented_loss": task_loss_mean + pressure_config.weight * pressure_loss_mean,
            }
        )
    if pressure_config.enabled:
        result.update(activation_metrics)
    return result


def _run_orthogonal_pressure_step(
    *,
    model: Any,
    optimizer: Any,
    params: list[Any],
    torch: Any,
    np: Any,
    train_tokens: Any,
    block_size: int,
    micro_batch_size: int,
    grad_accum: int,
    device: Any,
    dtype: Any,
    pressure_config: Any,
    activation_capture: Any,
    step: int,
    schedule_step: Any = None,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    task_loss_total = 0.0
    pressure_loss_total = 0.0
    pressure_grads: list[Any | None] = []
    activation_metrics: dict[str, float] = {}

    for micro_step in range(grad_accum):
        activation_capture.clear()
        batch = _sample_batch(
            torch,
            np,
            train_tokens,
            block_size,
            micro_batch_size,
            device,
            starts=(None if schedule_step is None else schedule_step[micro_step]),
        )
        with _autocast_context(torch, device, dtype):
            output = model(input_ids=batch, labels=batch)
            task_loss = output.loss
            current_pressure_loss = pressure_loss(
                torch,
                activation_capture.activations,
                pressure_config,
            )
        _require_finite_loss(torch, task_loss, f"task loss at step {step}")
        _require_finite_loss(torch, current_pressure_loss, f"pressure loss at step {step}")

        (task_loss / grad_accum).backward(retain_graph=True)
        new_pressure_grads = torch.autograd.grad(
            current_pressure_loss / grad_accum,
            params,
            allow_unused=True,
        )
        pressure_grads = accumulate_grads(pressure_grads, new_pressure_grads)
        task_loss_total += float(task_loss.detach().cpu())
        pressure_loss_total += float(current_pressure_loss.detach().cpu())
        activation_metrics = activation_near_zero_metrics(
            activation_capture.activations,
            pressure_config.log_thresholds,
        )

    task_grads = clone_grads(params)
    result = {
        "task_loss": task_loss_total / grad_accum,
        "pressure_loss": pressure_loss_total / grad_accum,
        "pressure_weight": pressure_config.weight,
        "weighted_pressure_loss": pressure_config.weight * pressure_loss_total / grad_accum,
        "augmented_loss": (
            task_loss_total / grad_accum
            + pressure_config.weight * pressure_loss_total / grad_accum
        ),
    }
    result.update(grad_metrics(torch, task_grads, pressure_grads))

    optimizer.step()
    result.update(
        apply_adam_step_orthogonal_pressure(
            optimizer,
            params,
            task_grads,
            pressure_grads,
            pressure_weight=pressure_config.weight,
            step_budget=pressure_config.step_budget,
            eps=pressure_config.eps,
        )
    )
    result.update(activation_metrics)
    return result


def _sample_batch(
    torch: Any,
    np: Any,
    tokens: Any,
    block_size: int,
    batch_size: int,
    device: Any,
    *,
    starts: Any = None,
) -> Any:
    if starts is None:
        max_start = len(tokens) - block_size - 1
        starts = np.random.randint(0, max_start, size=batch_size)
    if len(starts) != batch_size:
        raise ValueError(f"Expected {batch_size} batch starts, got {len(starts)}.")
    batch = np.stack([tokens[start : start + block_size] for start in starts])
    return torch.as_tensor(batch, dtype=torch.long, device=device)


def _autocast_context(torch: Any, device: Any, dtype: Any) -> Any:
    if dtype is not None and device.type == "cuda":
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def _require_finite_loss(torch: Any, loss: Any, label: str) -> None:
    if not bool(torch.isfinite(loss.detach()).item()):
        raise RuntimeError(f"Non-finite {label}.")


def _learning_rate_for_step(step: int, base_learning_rate: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return base_learning_rate
    return base_learning_rate * min(1.0, step / warmup_steps)


def _optimizer_config(training: dict[str, Any]) -> dict[str, Any]:
    name = str(training["optimizer"])
    if name != "adamw":
        raise ValueError(f"Unsupported optimizer: {name}")
    betas = training["adamw_betas"]
    if not isinstance(betas, list | tuple) or len(betas) != 2:
        raise ValueError("training.adamw_betas must contain exactly two values.")
    beta1 = float(betas[0])
    beta2 = float(betas[1])
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("training.adamw_betas values must be in [0, 1).")
    eps = float(training["adamw_eps"])
    if eps <= 0.0:
        raise ValueError("training.adamw_eps must be positive.")
    weight_decay = float(training["weight_decay"])
    if weight_decay < 0.0:
        raise ValueError("training.weight_decay must be non-negative.")
    return {
        "name": name,
        "betas": (beta1, beta2),
        "eps": eps,
        "weight_decay": weight_decay,
    }


def _set_optimizer_lr(optimizer: Any, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _global_grad_norm(model: Any) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        param_norm = parameter.grad.detach().float().norm(2).item()
        total += param_norm * param_norm
    return total**0.5


def _global_weight_norm(model: Any) -> float:
    total = 0.0
    for parameter in model.parameters():
        param_norm = parameter.detach().float().norm(2).item()
        total += param_norm * param_norm
    return total**0.5


def _mlp_weight_norm(model: Any) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if ".mlp." not in name or not name.endswith(".weight"):
            continue
        param_norm = parameter.detach().float().norm(2).item()
        total += param_norm * param_norm
    return total**0.5
