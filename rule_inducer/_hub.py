from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn


def _materialize_lazy_example_proj_x(encoder: nn.Module) -> None:
    """Convert ``example_proj_x[0]`` from LazyLinear to Linear when shaped."""
    example_proj_x = getattr(encoder, "example_proj_x", None)
    if not isinstance(example_proj_x, nn.Sequential) or len(example_proj_x) == 0:
        return

    first = example_proj_x[0]
    if not isinstance(first, nn.LazyLinear):
        return

    try:
        weight = first.weight
        bias = first.bias
        in_features = int(weight.shape[1])
        out_features = int(weight.shape[0])
    except Exception:
        return

    new_layer = nn.Linear(in_features, out_features, bias=bias is not None).to(
        device=weight.device, dtype=weight.dtype
    )
    with torch.no_grad():
        new_layer.weight.copy_(weight)
        if bias is not None and new_layer.bias is not None:
            new_layer.bias.copy_(bias)

    example_proj_x[0] = new_layer
    setattr(encoder, "_example_proj_x_in_dim", in_features)


def _materialize_lazy_example_proj_x_from_state(
    encoder: nn.Module,
    state_dict: Mapping[str, Tensor],
    *,
    prefix: str = "literal_encoder.",
) -> None:
    """Create a Linear ``example_proj_x[0]`` using saved checkpoint shapes."""
    example_proj_x = getattr(encoder, "example_proj_x", None)
    if not isinstance(example_proj_x, nn.Sequential) or len(example_proj_x) == 0:
        return
    if not isinstance(example_proj_x[0], nn.LazyLinear):
        return

    weight = state_dict.get(f"{prefix}example_proj_x.0.weight")
    bias = state_dict.get(f"{prefix}example_proj_x.0.bias")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return

    new_layer = nn.Linear(
        int(weight.shape[1]), int(weight.shape[0]), bias=isinstance(bias, torch.Tensor)
    ).to(device=weight.device, dtype=weight.dtype)
    example_proj_x[0] = new_layer
    setattr(encoder, "_example_proj_x_in_dim", int(weight.shape[1]))


def materialize_rule_inducer_for_hub(model: nn.Module) -> None:
    """Run a tiny episode through RuleInducer so lazy layers are saveable."""
    literal_encoder = getattr(model, "literal_encoder", None)
    if isinstance(literal_encoder, nn.Module):
        _materialize_lazy_example_proj_x(literal_encoder)
        example_proj_x = getattr(literal_encoder, "example_proj_x", None)
        if not (
            isinstance(example_proj_x, nn.Sequential)
            and len(example_proj_x) > 0
            and isinstance(example_proj_x[0], nn.LazyLinear)
        ):
            return

    try:
        first_param = next(model.parameters())
        device = first_param.device
        dtype = first_param.dtype if first_param.is_floating_point() else torch.float32
    except StopIteration:
        device = torch.device("cpu")
        dtype = torch.float32

    was_training = model.training
    model.eval()
    with torch.no_grad():
        x_val = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], device=device, dtype=dtype)
        x_mask = torch.ones_like(x_val, dtype=torch.bool)
        y_val = torch.tensor([[[1.0], [0.0]]], device=device, dtype=dtype)
        y_mask = torch.ones_like(y_val, dtype=torch.bool)
        n_len = torch.tensor([2], device=device, dtype=torch.long)
        m_len = torch.tensor([2], device=device, dtype=torch.long)
        h_len = torch.tensor([1], device=device, dtype=torch.long)
        model(
            X_val=x_val,
            X_mask=x_mask,
            Y_val=y_val,
            Y_mask=y_mask,
            N_len=n_len,
            M_len=m_len,
            H_len=h_len,
        )
    if was_training:
        model.train()

    if isinstance(literal_encoder, nn.Module):
        _materialize_lazy_example_proj_x(literal_encoder)
