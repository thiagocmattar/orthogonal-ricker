from __future__ import annotations

from pathlib import Path
from typing import Any


def apply_post_layernorm_relu(model: Any, *, torch: Any) -> Any:
    """Apply configured ReLUs after each GPT-NeoX block LayerNorm."""
    config = getattr(model, "config", None)
    if not bool(getattr(config, "post_layernorm_relu", False)):
        return model
    if getattr(model, "_post_layernorm_relu_applied", False):
        return model

    gpt_neox = getattr(model, "gpt_neox", None)
    layers = getattr(gpt_neox, "layers", None)
    if layers is None:
        raise ValueError("Configured model.post_layernorm_relu, but the model has no GPT-NeoX layers.")

    for layer in layers:
        input_layernorm = getattr(layer, "input_layernorm", None)
        post_attention_layernorm = getattr(layer, "post_attention_layernorm", None)
        if input_layernorm is None or post_attention_layernorm is None:
            raise ValueError(
                "Configured model.post_layernorm_relu, but a GPT-NeoX layer is missing a branch LayerNorm."
            )

        layer.attention_input_relu = torch.nn.ReLU()
        layer.mlp_input_relu = torch.nn.ReLU()
        input_layernorm.register_forward_hook(_relu_output_hook(layer.attention_input_relu))
        post_attention_layernorm.register_forward_hook(_relu_output_hook(layer.mlp_input_relu))

    model._post_layernorm_relu_applied = True
    return model


def load_checkpoint_model(auto_model: Any, checkpoint_path: str | Path, *, torch: Any) -> Any:
    model = auto_model.from_pretrained(checkpoint_path)
    return apply_post_layernorm_relu(model, torch=torch)


def _relu_output_hook(relu: Any) -> Any:
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        return relu(output)

    return hook
