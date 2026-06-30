from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SUPPORTED_SITE_ALIASES = {"mlp_hiddens"}


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
            self._handles.append(activation_module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str) -> Any:
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            value = output
            if _site_clipping_enabled(self.clipping, "mlp_hiddens", name):
                value = clip_activation_tensor(value, self.clipping, torch=self.torch)
            self.activations[name] = value
            return value

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
