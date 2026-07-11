from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SUPPORTED_SITE_ALIASES = {
    "mlp_hiddens",
    "attention_inputs",
    "mlp_inputs",
    "residual_streams",
    "attention_outputs",
}


@dataclass(frozen=True)
class ActivationSite:
    name: str
    module_path: str
    role: str
    shape: str
    downstream_operator: str


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
        resolved = resolve_site_aliases(self.requested_sites)
        if "mlp_hiddens" in resolved:
            self._register_pythia_mlp_hiddens()
        if "attention_inputs" in resolved:
            self._register_pythia_attention_inputs()
        if "mlp_inputs" in resolved:
            self._register_pythia_mlp_inputs()
        if "residual_streams" in resolved:
            self._register_pythia_residual_streams()
        if "attention_outputs" in resolved:
            self._register_pythia_attention_outputs()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _register_pythia_mlp_hiddens(self) -> None:
        layers = getattr(getattr(self.model, "gpt_neox", None), "layers", None)
        if layers is None:
            raise ValueError("mlp_hiddens capture currently supports GPTNeoX/Pythia models only.")

        for index, layer in enumerate(layers):
            activation_module = getattr(getattr(layer, "mlp", None), "act", None)
            if activation_module is None:
                raise ValueError(f"Could not resolve MLP activation module for layer {index}.")

            name = f"mlp_hiddens.layer_{index}"
            self.site_metadata.append(
                ActivationSite(
                    name=name,
                    module_path=f"gpt_neox.layers.{index}.mlp.act",
                    role="mlp_hidden",
                    shape="[batch, seq, intermediate]",
                    downstream_operator=f"gpt_neox.layers.{index}.mlp.dense_4h_to_h",
                )
            )
            self._handles.append(activation_module.register_forward_hook(self._make_hook(name, "mlp_hiddens")))

    def _register_pythia_attention_inputs(self) -> None:
        layers = getattr(getattr(self.model, "gpt_neox", None), "layers", None)
        if layers is None:
            raise ValueError("attention_inputs capture currently supports GPTNeoX/Pythia models only.")

        for index, layer in enumerate(layers):
            capture_module = getattr(layer, "attention_input_relu", None)
            module_path = f"gpt_neox.layers.{index}.attention_input_relu"
            if capture_module is None:
                capture_module = getattr(layer, "input_layernorm", None)
                module_path = f"gpt_neox.layers.{index}.input_layernorm"
            if capture_module is None:
                raise ValueError(f"Could not resolve attention input module for layer {index}.")

            name = f"attention_inputs.layer_{index}"
            self.site_metadata.append(
                ActivationSite(
                    name=name,
                    module_path=module_path,
                    role="attention_input",
                    shape="[batch, seq, hidden]",
                    downstream_operator=f"gpt_neox.layers.{index}.attention.query_key_value",
                )
            )
            self._handles.append(capture_module.register_forward_hook(self._make_hook(name, "attention_inputs")))

    def _register_pythia_mlp_inputs(self) -> None:
        layers = getattr(getattr(self.model, "gpt_neox", None), "layers", None)
        if layers is None:
            raise ValueError("mlp_inputs capture currently supports GPTNeoX/Pythia models only.")

        for index, layer in enumerate(layers):
            capture_module = getattr(layer, "mlp_input_relu", None)
            module_path = f"gpt_neox.layers.{index}.mlp_input_relu"
            if capture_module is None:
                capture_module = getattr(layer, "post_attention_layernorm", None)
                module_path = f"gpt_neox.layers.{index}.post_attention_layernorm"
            if capture_module is None:
                raise ValueError(f"Could not resolve MLP input module for layer {index}.")

            name = f"mlp_inputs.layer_{index}"
            self.site_metadata.append(
                ActivationSite(
                    name=name,
                    module_path=module_path,
                    role="mlp_input",
                    shape="[batch, seq, hidden]",
                    downstream_operator=f"gpt_neox.layers.{index}.mlp.dense_h_to_4h",
                )
            )
            self._handles.append(capture_module.register_forward_hook(self._make_hook(name, "mlp_inputs")))

    def _register_pythia_residual_streams(self) -> None:
        layers = getattr(getattr(self.model, "gpt_neox", None), "layers", None)
        if layers is None:
            raise ValueError("residual_streams capture currently supports GPTNeoX/Pythia models only.")

        for index, layer in enumerate(layers):
            name = f"residual_streams.layer_{index}"
            self.site_metadata.append(
                ActivationSite(
                    name=name,
                    module_path=f"gpt_neox.layers.{index}",
                    role="residual_stream",
                    shape="[batch, seq, hidden]",
                    downstream_operator=f"gpt_neox.layers.{index}",
                )
            )
            self._handles.append(layer.register_forward_pre_hook(self._make_pre_hook(name, "residual_streams")))

    def _register_pythia_attention_outputs(self) -> None:
        layers = getattr(getattr(self.model, "gpt_neox", None), "layers", None)
        if layers is None:
            raise ValueError("attention_outputs capture currently supports GPTNeoX/Pythia models only.")

        for index, layer in enumerate(layers):
            attention_module = getattr(layer, "attention", None)
            if attention_module is None:
                raise ValueError(f"Could not resolve attention module for layer {index}.")

            name = f"attention_outputs.layer_{index}"
            self.site_metadata.append(
                ActivationSite(
                    name=name,
                    module_path=f"gpt_neox.layers.{index}.attention",
                    role="attention_output",
                    shape="[batch, seq, hidden]",
                    downstream_operator=f"gpt_neox.layers.{index} residual add",
                )
            )
            self._handles.append(attention_module.register_forward_hook(self._make_hook(name, "attention_outputs")))

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

    def _make_pre_hook(self, name: str, alias: str) -> Any:
        def hook(_module: Any, inputs: tuple[Any, ...]) -> Any:
            if not inputs:
                raise ValueError(f"Could not capture {name}: module received no positional inputs.")
            value = _first_tensor(inputs[0])
            if _site_clipping_enabled(self.clipping, alias, name):
                value = clip_activation_tensor(value, self.clipping, torch=self.torch)
                self.activations[name] = value
                return (value, *inputs[1:])
            self.activations[name] = value
            return None

        return hook


def resolve_site_aliases(sites: list[str]) -> set[str]:
    if not sites:
        raise ValueError("At least one activation site must be configured.")
    resolved: set[str] = set()
    for site in sites:
        if site == "all_sites":
            resolved.add("mlp_hiddens")
        elif site in SUPPORTED_SITE_ALIASES:
            resolved.add(site)
        else:
            raise ValueError(f"Unsupported activation site for this harness: {site}")
    return resolved


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
    return alias in sites or name in sites or "all_sites" in sites
