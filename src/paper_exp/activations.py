from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from paper_exp.topology import SITE_SPECS
from paper_exp.topology import SUPPORTED_SITE_ALIASES


@dataclass(frozen=True)
class ActivationSite:
    alias: str
    name: str
    module_path: str
    tensor: str
    shape: str
    downstream_matmul: str
    operations_before_matmul: tuple[str, ...]


class ActivationCapture:
    def __init__(
        self,
        model: Any,
        sites: list[str],
        *,
        torch: Any,
        clipping: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.requested_sites = sites
        self.torch = torch
        self.clipping = clipping or {"enabled": False}
        self.activations: dict[str, Any] = {}
        self.site_metadata: list[ActivationSite] = []
        self._handles: list[Any] = []

    def __enter__(self) -> ActivationCapture:
        self.register()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.remove()

    def clear(self) -> None:
        self.activations.clear()

    def register(self) -> None:
        self.remove()
        self.site_metadata.clear()
        for alias in resolve_site_aliases(self.requested_sites):
            if alias == "a":
                self._register_branch_site(
                    alias="a",
                    gate_name="a_gate",
                    layernorm_name="input_layernorm",
                )
            elif alias == "m":
                self._register_branch_site(
                    alias="m",
                    gate_name="m_gate",
                    layernorm_name="post_attention_layernorm",
                )
            elif alias == "h":
                self._register_h_site()
            else:
                self._register_attention_site(alias)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _layers(self, alias: str) -> Any:
        layers = getattr(getattr(self.model, "gpt_neox", None), "layers", None)
        if layers is None:
            raise ValueError(f"Site {alias} capture currently supports GPTNeoX/Pythia models only.")
        return layers

    def _register_branch_site(
        self,
        *,
        alias: str,
        gate_name: str,
        layernorm_name: str,
    ) -> None:
        for index, layer in enumerate(self._layers(alias)):
            capture_module = getattr(layer, gate_name, None)
            module_name = gate_name
            if capture_module is None:
                capture_module = getattr(layer, layernorm_name, None)
                module_name = layernorm_name
            if capture_module is None:
                raise ValueError(f"Could not resolve site {alias} in layer {index}.")
            self._register_output_module(
                alias=alias,
                index=index,
                module=capture_module,
                module_path=f"gpt_neox.layers.{index}.{module_name}",
            )

    def _register_h_site(self) -> None:
        for index, layer in enumerate(self._layers("h")):
            capture_module = getattr(getattr(layer, "mlp", None), "act", None)
            if capture_module is None:
                raise ValueError(f"Could not resolve site h in layer {index}.")
            self._register_output_module(
                alias="h",
                index=index,
                module=capture_module,
                module_path=f"gpt_neox.layers.{index}.mlp.act",
            )

    def _register_attention_site(self, alias: str) -> None:
        from paper_exp.modeling import expose_attention_sites

        expose_attention_sites(self.model, torch=self.torch)
        for index, layer in enumerate(self._layers(alias)):
            attention = getattr(layer, "attention", None)
            capture_module = getattr(attention, f"{alias}_site", None)
            if capture_module is None:
                raise ValueError(
                    f"Could not resolve site {alias} in layer {index}; the model does not "
                    "support the canonical attention ports."
                )
            self._register_output_module(
                alias=alias,
                index=index,
                module=capture_module,
                module_path=f"gpt_neox.layers.{index}.attention.{alias}_site",
            )

    def _register_output_module(
        self,
        *,
        alias: str,
        index: int,
        module: Any,
        module_path: str,
    ) -> None:
        name = f"{alias}.layer_{index}"
        spec = SITE_SPECS[alias]
        self.site_metadata.append(
            ActivationSite(
                alias=alias,
                name=name,
                module_path=module_path,
                tensor=spec.tensor,
                shape=spec.shape,
                downstream_matmul=spec.downstream_matmul,
                operations_before_matmul=spec.operations_before_matmul,
            )
        )
        self._handles.append(module.register_forward_hook(self._make_hook(name, alias)))

    def _make_hook(self, name: str, alias: str) -> Any:
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            value = _first_tensor(output)
            if _site_clipping_enabled(self.clipping, alias, name):
                value = clip_activation_tensor(value, self.clipping, torch=self.torch)
                self.activations[name] = value
                return _replace_first_tensor(output, value)
            self.activations[name] = value
            return output

        return hook


def resolve_site_aliases(sites: list[str]) -> tuple[str, ...]:
    if not sites:
        raise ValueError("At least one transformer site must be configured.")
    if len(set(sites)) != len(sites):
        raise ValueError("Transformer site aliases must not contain duplicates.")
    for site in sites:
        if site not in SUPPORTED_SITE_ALIASES:
            raise ValueError(f"Unsupported transformer site alias: {site}")
    return tuple(sites)


def activation_exact_zero_counts(activations: dict[str, Any]) -> tuple[int, int]:
    zero_count = 0
    activation_count = 0
    for value in activations.values():
        detached = value.detach()
        zero_count += int((detached == 0).sum().item())
        activation_count += detached.numel()
    return zero_count, activation_count


def activation_exact_zero_counts_by_alias(activations: dict[str, Any]) -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = {}
    for name, value in activations.items():
        alias = name.split(".layer_", 1)[0]
        detached = value.detach()
        if alias not in counts:
            counts[alias] = [0, 0]
        counts[alias][0] += int((detached == 0).sum().item())
        counts[alias][1] += detached.numel()
    return {alias: (values[0], values[1]) for alias, values in counts.items()}


def _first_tensor(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                continue
        raise TypeError("Could not find tensor in activation hook output.")
    if hasattr(value, "detach"):
        return value
    raise TypeError(f"Unsupported activation hook value type: {type(value)!r}")


def _replace_first_tensor(value: Any, replacement: Any) -> Any:
    if hasattr(value, "detach"):
        return replacement
    if isinstance(value, tuple):
        replaced = False
        items = []
        for item in value:
            if replaced:
                items.append(item)
                continue
            try:
                items.append(_replace_first_tensor(item, replacement))
                replaced = True
            except TypeError:
                items.append(item)
        if not replaced:
            raise TypeError("Could not find tensor in activation hook output.")
        return tuple(items)
    if isinstance(value, list):
        replaced = False
        items = []
        for item in value:
            if replaced:
                items.append(item)
                continue
            try:
                items.append(_replace_first_tensor(item, replacement))
                replaced = True
            except TypeError:
                items.append(item)
        if not replaced:
            raise TypeError("Could not find tensor in activation hook output.")
        return items
    raise TypeError(f"Unsupported activation hook value type: {type(value)!r}")


def clip_activation_tensor(value: Any, cfg: dict[str, Any], *, torch: Any) -> Any:
    mode = cfg.get("mode", "threshold")
    if mode == "threshold":
        threshold = float(cfg.get("threshold", 0.0))
        return value.masked_fill(value.detach().abs() <= threshold, 0.0)
    if mode == "rms_threshold":
        multiplier = float(cfg["rms_multiplier"])
        if multiplier < 0.0:
            raise ValueError("activation_clipping.rms_multiplier must be non-negative.")
        detached = value.detach().float()
        rms = detached.square().mean().sqrt()
        threshold = multiplier * rms
        return value.masked_fill(detached.abs() <= threshold, 0.0)
    if mode == "quantile":
        quantile = float(cfg["quantile"])
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("activation_clipping.quantile must be between 0 and 1.")
        flat = value.detach().abs().reshape(-1).float()
        if flat.numel() == 0:
            return value
        k = max(1, min(flat.numel(), int(round(quantile * flat.numel()))))
        threshold = flat.kthvalue(k).values
        return value.masked_fill(value.detach().abs() <= threshold, 0.0)
    raise ValueError(f"Unknown activation clipping mode: {mode}")


def _site_clipping_enabled(cfg: dict[str, Any], alias: str, name: str) -> bool:
    if not cfg.get("enabled", False):
        return False
    sites = cfg.get("sites", [])
    return alias in sites or name in sites
