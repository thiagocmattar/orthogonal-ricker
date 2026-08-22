from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

PRESSURE_METHODS = {
    "none",
    "l1_naive",
    "orthogonal_l1",
}


@dataclass(frozen=True)
class ActivationPressureConfig:
    enabled: bool
    method: str
    sites: list[str]
    weight: float
    step_budget: float | None
    eps: float
    log_thresholds: tuple[float, ...]

    @property
    def pressure_kind(self) -> str:
        if self.method in {"l1_naive", "orthogonal_l1"}:
            return "activation_l1"
        return "none"

    @property
    def applies_pressure(self) -> bool:
        return self.enabled and self.method != "none"

    @property
    def orthogonal(self) -> bool:
        return self.method == "orthogonal_l1"


def activation_pressure_config(config: dict[str, Any]) -> ActivationPressureConfig:
    raw = config.get("activation_pressure")
    if not isinstance(raw, dict):
        raise ValueError("activation_pressure must be an explicit mapping.")
    required = ("enabled", "method", "sites", "weight", "step_budget", "eps", "log_thresholds")
    missing = [field for field in required if field not in raw]
    if missing:
        raise ValueError(
            "activation_pressure requires explicit fields: " + ", ".join(missing)
        )
    unknown = sorted(set(raw) - set(required))
    if unknown:
        raise ValueError(
            "activation_pressure contains unsupported fields: " + ", ".join(unknown)
        )
    if not isinstance(raw["enabled"], bool):
        raise ValueError("activation_pressure.enabled must be a boolean.")
    enabled = raw["enabled"]
    method = str(raw["method"])
    if method not in PRESSURE_METHODS:
        raise ValueError(f"Unknown activation pressure method: {method}")
    if not enabled and method != "none":
        raise ValueError("Disabled activation pressure must use method: none.")

    sites = raw["sites"]
    if not isinstance(sites, list) or not sites or not all(
        isinstance(site, str) and site.strip() for site in sites
    ):
        raise ValueError("activation_pressure.sites must be a non-empty list of names.")
    if len(set(sites)) != len(sites):
        raise ValueError("activation_pressure.sites must not contain duplicates.")

    weight = _finite_float(raw["weight"], "activation_pressure.weight")
    if weight < 0.0:
        raise ValueError("activation_pressure.weight must be non-negative.")
    if method == "none" and weight != 0.0:
        raise ValueError("activation_pressure.weight must be zero for method: none.")

    step_budget_value = raw["step_budget"]
    step_budget = (
        None
        if step_budget_value is None
        else _finite_float(step_budget_value, "activation_pressure.step_budget")
    )
    if method == "orthogonal_l1":
        if step_budget is None or step_budget <= 0.0:
            raise ValueError(
                "Orthogonal activation pressure requires a positive step_budget."
            )
    elif step_budget is not None:
        raise ValueError(
            "activation_pressure.step_budget applies only to orthogonal methods."
        )

    eps = _finite_float(raw["eps"], "activation_pressure.eps")
    if eps <= 0.0:
        raise ValueError("activation_pressure.eps must be positive.")
    raw_thresholds = raw["log_thresholds"]
    if not isinstance(raw_thresholds, list) or not raw_thresholds:
        raise ValueError("activation_pressure.log_thresholds must be a non-empty list.")
    log_thresholds = tuple(
        _finite_float(value, "activation_pressure.log_thresholds")
        for value in raw_thresholds
    )
    if any(value < 0.0 for value in log_thresholds):
        raise ValueError("activation_pressure.log_thresholds must be non-negative.")
    if tuple(sorted(set(log_thresholds))) != log_thresholds:
        raise ValueError(
            "activation_pressure.log_thresholds must be strictly increasing and unique."
        )

    return ActivationPressureConfig(
        enabled=enabled,
        method=method,
        sites=list(sites),
        weight=weight,
        step_budget=step_budget,
        eps=eps,
        log_thresholds=log_thresholds,
    )


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number.")
    return result


def pressure_loss(torch: Any, activations: dict[str, Any], cfg: ActivationPressureConfig) -> Any:
    if not cfg.enabled or cfg.method == "none":
        return None
    if cfg.pressure_kind == "activation_l1":
        return activation_l1_pressure(torch, activations)
    raise ValueError(f"Unsupported pressure kind: {cfg.pressure_kind}")


def activation_l1_pressure(torch: Any, activations: dict[str, Any]) -> Any:
    """Elementwise absolute activation value, averaged to one loss."""
    _require_activations(activations)
    return torch.stack([value.float().abs().mean() for value in activations.values()]).mean()


def activation_near_zero_metrics(activations: dict[str, Any], thresholds: tuple[float, ...]) -> dict[str, float]:
    if not activations:
        return {}

    metrics: dict[str, float] = {}
    total_count = 0
    total_hits = {threshold: 0 for threshold in thresholds}
    for name, value in activations.items():
        detached = value.detach().float().abs()
        count = detached.numel()
        total_count += count
        for threshold in thresholds:
            hits = int((detached <= threshold).sum().item())
            total_hits[threshold] += hits
            metrics[f"activation/{name}/near_zero_mass/{_threshold_name(threshold)}"] = hits / count

    for threshold in thresholds:
        metrics[f"activation/near_zero_mass/{_threshold_name(threshold)}"] = total_hits[threshold] / total_count
    return metrics


def clone_grads(params: list[Any]) -> list[Any | None]:
    return [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in params]


def accumulate_grads(existing: list[Any | None], new_grads: tuple[Any | None, ...]) -> list[Any | None]:
    if not existing:
        return [None if grad is None else grad.detach().clone() for grad in new_grads]
    if len(existing) != len(new_grads):
        raise ValueError("Gradient lists must have equal length.")
    for index, grad in enumerate(new_grads):
        if grad is None:
            continue
        if existing[index] is None:
            existing[index] = grad.detach().clone()
        else:
            existing[index].add_(grad.detach())
    return existing


def grad_metrics(torch: Any, task_grads: list[Any | None], pressure_grads: list[Any | None]) -> dict[str, float | bool]:
    task_sq = 0.0
    pressure_sq = 0.0
    dot = 0.0
    for task_grad, pressure_grad in zip(task_grads, pressure_grads):
        if task_grad is not None:
            task_sq += float(task_grad.detach().float().square().sum().item())
        if pressure_grad is not None:
            pressure_sq += float(pressure_grad.detach().float().square().sum().item())
        if task_grad is not None and pressure_grad is not None:
            dot += float((task_grad.detach().float() * pressure_grad.detach().float()).sum().item())
    task_norm = task_sq**0.5
    pressure_norm = pressure_sq**0.5
    cosine = dot / (task_norm * pressure_norm + 1e-12)
    return {
        "pressure/task_gradient_norm": task_norm,
        "pressure/pressure_gradient_norm": pressure_norm,
        "pressure/pressure_to_task_gradient_norm_ratio": pressure_norm / (task_norm + 1e-12),
        "pressure/task_pressure_gradient_dot": dot,
        "pressure/task_pressure_gradient_cosine": cosine,
        "pressure/pressure_conflict": dot < 0.0,
    }


@torch.no_grad()
def apply_adam_step_orthogonal_pressure(
    optimizer: Any,
    params: list[Any],
    task_grads: list[Any | None],
    pressure_grads: list[Any | None],
    *,
    pressure_weight: float,
    step_budget: float | None,
    eps: float = 1e-12,
) -> dict[str, float | bool | int]:
    if len(params) != len(task_grads) or len(params) != len(pressure_grads):
        raise ValueError("params, task_grads, and pressure_grads must have equal length")
    if step_budget is not None and step_budget < 0.0:
        raise ValueError("step_budget must be non-negative or None")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    param_index = {id(parameter): index for index, parameter in enumerate(params)}
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

        for parameter in group["params"]:
            index = param_index.get(id(parameter))
            if index is None or task_grads[index] is None or pressure_grads[index] is None:
                skipped += 1
                continue

            state = optimizer.state.get(parameter, {})
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

            exp_avg = state["exp_avg"].detach().to(device=parameter.device).float()
            exp_avg_sq = state["exp_avg_sq"]
            if amsgrad and "max_exp_avg_sq" in state:
                exp_avg_sq = state["max_exp_avg_sq"]
            denom = exp_avg_sq.detach().to(device=parameter.device).float()
            denom = denom.div(bias_correction2).sqrt().add_(adam_eps)

            d_task = exp_avg.div(bias_correction1).div(denom)
            d_pressure = pressure_grads[index].detach().to(device=parameter.device).float().div(denom)

            task_direction_sq += float(d_task.square().sum().item())
            pressure_direction_sq += float(d_pressure.square().sum().item())
            dot_before += float((d_task * d_pressure).sum().item())
            corrections.append((parameter, d_task, d_pressure, lr))

    should_project = dot_before < 0.0 and task_direction_sq > eps
    coeff = dot_before / (task_direction_sq + eps) if should_project else 0.0

    safe_corrections = []
    safe_direction_sq = 0.0
    dot_after = 0.0
    for parameter, d_task, d_pressure, lr in corrections:
        d_safe = d_pressure - coeff * d_task if should_project else d_pressure
        safe_direction_sq += float(d_safe.square().sum().item())
        dot_after += float((d_task * d_safe).sum().item())
        safe_corrections.append((parameter, d_safe, lr))

    task_norm = task_direction_sq**0.5
    safe_norm = safe_direction_sq**0.5
    raw_ratio = pressure_weight * safe_norm / (task_norm + eps)
    scale = 1.0
    if step_budget is not None and raw_ratio > 0.0:
        scale = min(1.0, step_budget / (raw_ratio + eps))

    if pressure_weight != 0.0 and scale != 0.0:
        for parameter, d_safe, lr in safe_corrections:
            parameter.add_(d_safe.to(device=parameter.device, dtype=parameter.dtype), alpha=-lr * pressure_weight * scale)

    return {
        "pressure/task_update_direction_norm": task_norm,
        "pressure/pressure_update_direction_norm_raw": pressure_direction_sq**0.5,
        "pressure/task_pressure_update_dot_before": dot_before,
        "pressure/task_pressure_update_cosine_before": dot_before / (task_norm * pressure_direction_sq**0.5 + eps),
        "pressure/pressure_update_projected": should_project,
        "pressure/task_pressure_update_dot_after": dot_after,
        "pressure/task_pressure_update_cosine_after": dot_after / (task_norm * safe_norm + eps),
        "pressure/pressure_update_ratio_raw": raw_ratio,
        "pressure/pressure_update_applied_scale": scale,
        "pressure/pressure_update_ratio_final": raw_ratio * scale,
        "pressure/eligible_parameters": len(corrections),
        "pressure/skipped_parameters": skipped,
    }


def _optimizer_step_as_float(step: Any) -> float:
    if hasattr(step, "detach"):
        return float(step.detach().cpu().item())
    return float(step)


def _threshold_name(threshold: float) -> str:
    if threshold == 0.0:
        return "k0"
    text = f"{threshold:.0e}".replace("-", "m").replace("+", "")
    return f"k{text}"


def _require_activations(activations: dict[str, Any]) -> None:
    if not activations:
        raise ValueError("No activations were captured for pressure computation.")
