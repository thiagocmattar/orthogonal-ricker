"""Canonical transformer-site and activation-topology nomenclature."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any


SITE_ALIAS_ORDER = (
    "a",
    "m",
    "h",
    "q_pre",
    "k_pre",
    "q_post",
    "k_post",
    "v",
)
SUPPORTED_SITE_ALIASES = frozenset(SITE_ALIAS_ORDER)


@dataclass(frozen=True)
class TransformerSiteSpec:
    alias: str
    tensor: str
    shape: str
    downstream_matmul: str
    operations_before_matmul: tuple[str, ...] = ()


SITE_SPECS = {
    "a": TransformerSiteSpec(
        alias="a",
        tensor="attention branch LayerNorm or gate output",
        shape="[batch, seq, hidden]",
        downstream_matmul="fused W_QKV projection",
    ),
    "m": TransformerSiteSpec(
        alias="m",
        tensor="MLP branch LayerNorm or gate output",
        shape="[batch, seq, hidden]",
        downstream_matmul="MLP W1 up projection",
    ),
    "h": TransformerSiteSpec(
        alias="h",
        tensor="MLP hidden nonlinearity or gate output",
        shape="[batch, seq, intermediate]",
        downstream_matmul="MLP W2 down projection",
    ),
    "q_pre": TransformerSiteSpec(
        alias="q_pre",
        tensor="query port output before partial RoPE",
        shape="[batch, heads, tokens, head_width]",
        downstream_matmul="QK^T attention-score product",
        operations_before_matmul=("partial RoPE",),
    ),
    "k_pre": TransformerSiteSpec(
        alias="k_pre",
        tensor="key port output before partial RoPE",
        shape="[batch, heads, tokens, head_width]",
        downstream_matmul="QK^T attention-score product",
        operations_before_matmul=("partial RoPE",),
    ),
    "q_post": TransformerSiteSpec(
        alias="q_post",
        tensor="query port output after partial RoPE",
        shape="[batch, heads, tokens, head_width]",
        downstream_matmul="QK^T attention-score product",
    ),
    "k_post": TransformerSiteSpec(
        alias="k_post",
        tensor="key port output after partial RoPE",
        shape="[batch, heads, tokens, head_width]",
        downstream_matmul="QK^T attention-score product",
    ),
    "v": TransformerSiteSpec(
        alias="v",
        tensor="value port output",
        shape="[batch, heads, tokens, head_width]",
        downstream_matmul="PV attention-context product",
    ),
}


@dataclass(frozen=True)
class ActivationTopology:
    topology_id: str
    active_sites: tuple[str, ...]

    @property
    def qk_placement(self) -> str | None:
        if any(site in self.active_sites for site in ("q_pre", "k_pre")):
            return "pre_rope"
        if any(site in self.active_sites for site in ("q_post", "k_post")):
            return "post_rope"
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "topology_id": self.topology_id,
            "active_sites": list(self.active_sites),
            "qk_placement": self.qk_placement,
        }


_TOPOLOGY_ROWS = (
    ("A0", ()),
    ("A1-H", ("h",)),
    ("A2", ("m", "h")),
    ("A3", ("a", "m", "h")),
    ("A4-Q", ("a", "m", "h", "q_post")),
    ("A4-K", ("a", "m", "h", "k_post")),
    ("A4-V", ("a", "m", "h", "v")),
    ("A5-QK-PRE", ("a", "m", "h", "q_pre", "k_pre")),
    ("A5-QK-POST", ("a", "m", "h", "q_post", "k_post")),
    ("A6-PRE", ("a", "m", "h", "q_pre", "k_pre", "v")),
    ("A6-POST", ("a", "m", "h", "q_post", "k_post", "v")),
)

TOPOLOGY_ID_ORDER = tuple(topology_id for topology_id, _sites in _TOPOLOGY_ROWS)
SUPPORTED_TOPOLOGIES = {
    topology_id: ActivationTopology(topology_id=topology_id, active_sites=active_sites)
    for topology_id, active_sites in _TOPOLOGY_ROWS
}


def resolve_topology(topology_id: Any) -> ActivationTopology:
    if not isinstance(topology_id, str) or topology_id not in SUPPORTED_TOPOLOGIES:
        supported = ", ".join(TOPOLOGY_ID_ORDER)
        raise ValueError(
            f"Unsupported model.topology_id: {topology_id!r}. Expected one of: {supported}."
        )
    return SUPPORTED_TOPOLOGIES[topology_id]


def resolve_topology_and_gate(
    topology_id: Any,
    site_gate: Any,
) -> tuple[ActivationTopology, dict[str, Any] | None]:
    topology = resolve_topology(topology_id)
    if topology.topology_id == "A0":
        if site_gate is not None:
            raise ValueError("model.site_gate must be null for topology A0.")
        return topology, None
    if not isinstance(site_gate, Mapping):
        raise ValueError(
            f"model.site_gate must be an explicit mapping for topology {topology.topology_id}."
        )

    allowed = {"operator", "kappa"}
    extra = set(site_gate) - allowed
    if extra:
        fields = ", ".join(sorted(str(field) for field in extra))
        raise ValueError(f"model.site_gate contains unsupported fields: {fields}.")

    operator = site_gate.get("operator")
    if not isinstance(operator, str) or operator not in {
        "relu",
        "one_sided_threshold",
        "symmetric_threshold",
    }:
        raise ValueError(
            "model.site_gate.operator must be 'relu', 'one_sided_threshold', "
            "or 'symmetric_threshold'."
        )
    if operator == "relu":
        if "kappa" in site_gate:
            raise ValueError("model.site_gate.kappa must be omitted for operator: relu.")
        return topology, {"operator": "relu"}

    if "kappa" not in site_gate:
        raise ValueError(
            f"model.site_gate.kappa is required for operator: {operator}."
        )
    kappa = site_gate["kappa"]
    if (
        isinstance(kappa, bool)
        or not isinstance(kappa, (int, float))
        or not math.isfinite(float(kappa))
        or float(kappa) < 0.0
    ):
        raise ValueError("model.site_gate.kappa must be a finite non-negative number.")
    return topology, {"operator": operator, "kappa": float(kappa)}


def topology_for_runtime_sites(active_sites: list[str] | tuple[str, ...]) -> ActivationTopology:
    active = frozenset(active_sites)
    matches = [
        topology
        for topology in SUPPORTED_TOPOLOGIES.values()
        if frozenset(topology.active_sites) == active
    ]
    if len(matches) != 1:
        rendered = ", ".join(sorted(active)) or "none"
        raise ValueError(f"Active gate sites do not identify a supported topology: {rendered}.")
    return matches[0]


for _topology in SUPPORTED_TOPOLOGIES.values():
    unknown_sites = set(_topology.active_sites) - SUPPORTED_SITE_ALIASES
    if unknown_sites:
        raise RuntimeError(
            f"Topology {_topology.topology_id} contains unsupported sites: {sorted(unknown_sites)}"
        )
    if _topology.qk_placement == "pre_rope" and any(
        site in _topology.active_sites for site in ("q_post", "k_post")
    ):
        raise RuntimeError(f"Topology {_topology.topology_id} mixes PRE and POST Q/K sites.")
