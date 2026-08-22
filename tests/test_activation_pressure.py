from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from paper_exp.activation_pressure import activation_l1_pressure
from paper_exp.activation_pressure import activation_near_zero_metrics
from paper_exp.activation_pressure import activation_pressure_config
from paper_exp.activation_pressure import apply_adam_step_orthogonal_pressure
from paper_exp.activation_pressure import pressure_loss
from paper_exp.activations import ActivationCapture
from paper_exp.activations import activation_exact_zero_counts
from paper_exp.activations import activation_exact_zero_counts_by_alias
from paper_exp.activations import clip_activation_tensor
from paper_exp.activations import resolve_site_aliases
from paper_exp.cli import build_parser
from paper_exp.diagnostics.clipping_evaluation import _LogicalZeroProductAccumulator
from paper_exp.topology import SITE_ALIAS_ORDER
from paper_exp.topology import SITE_SPECS
from paper_exp.topology import SUPPORTED_SITE_ALIASES


def _pressure_config(
    *,
    method: str = "l1_naive",
    sites: list[str] | None = None,
    enabled: bool = True,
    weight: float | None = None,
    step_budget: float | None = None,
) -> dict[str, object]:
    if weight is None:
        weight = 0.0 if method == "none" else 1.0
    if step_budget is None and method == "orthogonal_l1":
        step_budget = 0.1
    return {
        "activation_pressure": {
            "enabled": enabled,
            "method": method,
            "sites": ["h"] if sites is None else sites,
            "weight": weight,
            "step_budget": step_budget,
            "eps": 1e-12,
            "log_thresholds": [0.0, 0.01],
        }
    }


def test_l1_pressure_is_an_unweighted_mean_of_per_tensor_means() -> None:
    activations = {
        "h.layer_0": torch.tensor([0.0, 2.0]),
        "a.layer_0": torch.tensor([10.0]),
    }

    l1 = activation_l1_pressure(torch, activations)

    assert torch.isfinite(l1)
    assert float(l1) == pytest.approx((1.0 + 10.0) / 2.0)


@pytest.mark.parametrize(
    ("method", "expected_pressure_kind", "orthogonal", "applies_pressure"),
    [
        ("none", "none", False, False),
        ("l1_naive", "activation_l1", False, True),
        ("orthogonal_l1", "activation_l1", True, True),
    ],
)
def test_only_canonical_pressure_methods_are_accepted(
    method: str,
    expected_pressure_kind: str,
    orthogonal: bool,
    applies_pressure: bool,
) -> None:
    cfg = activation_pressure_config(_pressure_config(method=method))

    assert cfg.method == method
    assert cfg.pressure_kind == expected_pressure_kind
    assert cfg.orthogonal is orthogonal
    assert cfg.applies_pressure is applies_pressure
    result = pressure_loss(torch, {"h.layer_0": torch.tensor([1.0])}, cfg)
    if applies_pressure:
        assert float(result) == pytest.approx(1.0)
    else:
        assert result is None


@pytest.mark.parametrize("site", SITE_ALIAS_ORDER)
def test_activation_pressure_accepts_every_canonical_site_alias(site: str) -> None:
    cfg = activation_pressure_config(_pressure_config(sites=[site]))

    assert cfg.sites == [site]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"method": "unsupported"}, "Unknown activation pressure method"),
        ({"sites": []}, "non-empty list"),
        ({"sites": ["h", "h"]}, "duplicates"),
        ({"sites": ["unknown"]}, "Unsupported transformer site alias"),
        ({"weight": -0.1}, "non-negative"),
        ({"log_thresholds": [0.01, 0.0]}, "strictly increasing"),
    ],
)
def test_activation_pressure_rejects_invalid_values(
    mutation: dict[str, object],
    message: str,
) -> None:
    config = _pressure_config()
    config["activation_pressure"].update(mutation)  # type: ignore[union-attr]

    with pytest.raises(ValueError, match=message):
        activation_pressure_config(config)


def test_activation_pressure_requires_exact_fields() -> None:
    missing = _pressure_config()
    del missing["activation_pressure"]["eps"]  # type: ignore[index]
    with pytest.raises(ValueError, match="eps"):
        activation_pressure_config(missing)

    extra = _pressure_config()
    extra["activation_pressure"]["unexpected"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="unexpected"):
        activation_pressure_config(extra)


def test_canonical_site_alias_catalog_is_exact_and_order_preserving() -> None:
    assert SUPPORTED_SITE_ALIASES == frozenset(SITE_ALIAS_ORDER)
    assert resolve_site_aliases(list(SITE_ALIAS_ORDER)) == SITE_ALIAS_ORDER


@pytest.mark.parametrize(
    ("sites", "message"),
    [
        ([], "At least one transformer site"),
        (["h", "h"], "duplicates"),
        (["unknown"], "Unsupported transformer site alias"),
    ],
)
def test_site_alias_resolution_is_strict(sites: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_site_aliases(sites)


@pytest.mark.parametrize(
    ("alias", "module_suffix"),
    [
        ("a", ".a_gate"),
        ("m", ".m_gate"),
        ("h", ".mlp.act"),
        ("q_pre", ".attention.q_pre_site"),
        ("k_pre", ".attention.k_pre_site"),
        ("q_post", ".attention.q_post_site"),
        ("k_post", ".attention.k_post_site"),
        ("v", ".attention.v_site"),
    ],
)
def test_activation_capture_uses_each_canonical_site_module(
    alias: str,
    module_suffix: str,
) -> None:
    model = _CanonicalCaptureModel()
    value = torch.tensor([[[-1.0, 2.0]]])

    with ActivationCapture(model, [alias], torch=torch) as capture:
        output = model.emit(alias, value)

    name = f"{alias}.layer_0"
    assert capture.activations[name].data_ptr() == output.data_ptr()
    assert len(capture.site_metadata) == 1
    metadata = capture.site_metadata[0]
    spec = SITE_SPECS[alias]
    assert metadata.alias == alias
    assert metadata.name == name
    assert metadata.module_path.endswith(module_suffix)
    assert metadata.tensor == spec.tensor
    assert metadata.shape == spec.shape
    assert metadata.downstream_matmul == spec.downstream_matmul
    assert metadata.operations_before_matmul == spec.operations_before_matmul


def test_activation_capture_registers_all_canonical_aliases_without_broadening() -> None:
    model = _CanonicalCaptureModel()

    with ActivationCapture(model, list(SITE_ALIAS_ORDER), torch=torch) as capture:
        outputs = {
            alias: model.emit(alias, torch.tensor([float(index)]))
            for index, alias in enumerate(SITE_ALIAS_ORDER)
        }

    assert tuple(site.alias for site in capture.site_metadata) == SITE_ALIAS_ORDER
    assert set(capture.activations) == {
        f"{alias}.layer_0" for alias in SITE_ALIAS_ORDER
    }
    for alias, output in outputs.items():
        assert capture.activations[f"{alias}.layer_0"].data_ptr() == output.data_ptr()


def test_a_and_m_capture_fall_back_to_layernorm_when_topology_has_no_gate() -> None:
    model = _CanonicalCaptureModel()
    layer = model.gpt_neox.layers[0]
    del layer.a_gate
    del layer.m_gate
    value = torch.tensor([[[-1.0, 2.0]]])

    with ActivationCapture(model, ["a", "m"], torch=torch) as capture:
        a_output = model.emit("a", value)
        m_output = model.emit("m", value)

    assert capture.activations["a.layer_0"].data_ptr() == a_output.data_ptr()
    assert capture.activations["m.layer_0"].data_ptr() == m_output.data_ptr()
    assert capture.site_metadata[0].module_path.endswith(".input_layernorm")
    assert capture.site_metadata[1].module_path.endswith(".post_attention_layernorm")


def test_activation_capture_clips_a_before_the_fused_projection() -> None:
    model = _CanonicalCaptureModel()
    value = torch.tensor([[[0.001, 2.0]]])

    with ActivationCapture(
        model,
        ["a"],
        torch=torch,
        clipping={
            "enabled": True,
            "mode": "threshold",
            "threshold": 0.01,
            "sites": ["a"],
        },
    ) as capture:
        output = model.emit("a", value)

    expected = torch.tensor([[[0.0, 2.0]]])
    assert torch.equal(capture.activations["a.layer_0"], expected)
    assert torch.equal(model.gpt_neox.layers[0].attention.query_key_value.last_input, expected)
    assert torch.equal(output, expected)


def test_clipping_produces_exact_zeros_and_near_zero_metrics() -> None:
    value = torch.tensor([-0.02, -0.001, 0.0, 0.003, 0.04])
    clipped = clip_activation_tensor(
        value,
        {"enabled": True, "mode": "threshold", "threshold": 0.003},
        torch=torch,
    )

    metrics = activation_near_zero_metrics({"h.layer_0": clipped}, (0.0, 0.01))

    assert torch.equal(clipped, torch.tensor([-0.02, 0.0, 0.0, 0.0, 0.04]))
    assert metrics["activation/near_zero_mass/k0"] == 0.6
    assert metrics["activation/near_zero_mass/k1em02"] == 0.6


def test_rms_threshold_clipping_uses_current_activation_scale() -> None:
    value = torch.tensor([0.1, 1.0, 2.0])
    clipped = clip_activation_tensor(
        value,
        {"enabled": True, "mode": "rms_threshold", "rms_multiplier": 1.0},
        torch=torch,
    )
    rms = value.float().square().mean().sqrt()

    assert torch.equal(clipped, value.masked_fill(value.abs() <= rms, 0.0))
    assert torch.equal(clipped, torch.tensor([0.0, 0.0, 2.0]))


def test_exact_zero_counts_are_integer_and_grouped_by_canonical_alias() -> None:
    activations = {
        "a.layer_0": torch.tensor([0.0, 1.0, 0.0]),
        "a.layer_1": torch.tensor([2.0, 0.0]),
        "h.layer_0": torch.tensor([0.0, 3.0]),
    }

    zero_count, activation_count = activation_exact_zero_counts(activations)
    grouped = activation_exact_zero_counts_by_alias(activations)

    assert (zero_count, activation_count) == (4, 7)
    assert grouped == {"a": (3, 5), "h": (1, 2)}


def test_logical_zero_product_summary_includes_dense_lm_head_denominator() -> None:
    accumulator = _LogicalZeroProductAccumulator()
    zero_counts = (1, 2, 3, 4, 5, 6)
    for name, zero_count in zip(accumulator.zero_counts, zero_counts, strict=True):
        accumulator.add(name, zero_count, 100)

    model = SimpleNamespace(
        get_output_embeddings=lambda: SimpleNamespace(
            weight=SimpleNamespace(shape=(3, 2))
        )
    )
    summary = accumulator.summary(model=model, total_tokens=4)

    assert summary["block_zero_product_count"] == 21
    assert summary["block_matmul_product_count"] == 600
    assert summary["lm_head_matmul_product_count"] == 24
    assert summary["model_matmul_product_count"] == 624
    assert summary["potentially_avoidable_block_matmul_fraction"] == 21 / 600
    assert summary["potentially_avoidable_model_matmul_fraction"] == 21 / 624


def test_clip_sweep_cli_can_request_actual_zero_product_measurement() -> None:
    args = build_parser().parse_args(
        [
            "clip-sweep",
            "--run-dir",
            "checkpoint",
            "--measure-zero-products",
            "--seed",
            "0",
        ]
    )

    assert args.measure_zero_products is True


def test_adam_step_orthogonal_pressure_projects_conflict_and_caps_ratio() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    task_loss = parameter.square().sum()
    task_loss.backward()
    task_grads = [parameter.grad.detach().clone()]
    optimizer.step()

    metrics = apply_adam_step_orthogonal_pressure(
        optimizer,
        [parameter],
        task_grads,
        [-task_grads[0]],
        pressure_weight=1.0,
        step_budget=0.1,
    )

    assert metrics["pressure/pressure_update_projected"] is True
    assert metrics["pressure/task_pressure_update_dot_before"] < 0.0
    assert metrics["pressure/task_pressure_update_dot_after"] >= -1e-8
    assert metrics["pressure/pressure_update_ratio_final"] <= 0.1 + 1e-8


class _CanonicalCaptureModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gpt_neox = SimpleNamespace(
            layers=torch.nn.ModuleList([_CanonicalCaptureLayer()])
        )

    def emit(self, alias: str, value: torch.Tensor) -> torch.Tensor:
        layer = self.gpt_neox.layers[0]
        if alias == "a":
            module = getattr(layer, "a_gate", layer.input_layernorm)
            return layer.attention.query_key_value(module(value))
        if alias == "m":
            module = getattr(layer, "m_gate", layer.post_attention_layernorm)
            return layer.mlp.dense_h_to_4h(module(value))
        if alias == "h":
            return layer.mlp.dense_4h_to_h(layer.mlp.act(value))
        return getattr(layer.attention, f"{alias}_site")(value)


class _CanonicalCaptureLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = torch.nn.Identity()
        self.post_attention_layernorm = torch.nn.Identity()
        self.a_gate = torch.nn.Identity()
        self.m_gate = torch.nn.Identity()
        self.mlp = _CanonicalCaptureMLP()
        self.attention = _CanonicalCaptureAttention()


class _CanonicalCaptureMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense_h_to_4h = _RecordingIdentity()
        self.act = torch.nn.Identity()
        self.dense_4h_to_h = _RecordingIdentity()


class _CanonicalCaptureAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_key_value = _RecordingIdentity()
        for alias in ("q_pre", "k_pre", "q_post", "k_post", "v"):
            setattr(self, f"{alias}_site", torch.nn.Identity())


class _RecordingIdentity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input: torch.Tensor | None = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.last_input = value
        return value
