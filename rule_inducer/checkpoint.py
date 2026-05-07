from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from ._hub import _materialize_lazy_example_proj_x
from .model import LiteralFilmConfig, RuleInducer


def _extract_state_dict(payload: Mapping[str, object]) -> dict[str, Tensor]:
    for key in ("model_state_dict", "model", "state_dict"):
        cand = payload.get(key)
        if isinstance(cand, Mapping):
            return {str(k): v for k, v in cand.items() if isinstance(v, torch.Tensor)}
    raise KeyError(
        "Checkpoint does not contain a model state dict (expected one of: "
        "model_state_dict, model, state_dict)."
    )


def _extract_config(payload: Mapping[str, object]) -> dict[str, Any]:
    cfg = payload.get("config")
    if isinstance(cfg, Mapping):
        return {str(k): v for k, v in cfg.items()}
    args = payload.get("args")
    if isinstance(args, Mapping):
        return {str(k): v for k, v in args.items()}
    return {}


def _strip_module_prefix(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    out: dict[str, Tensor] = {}
    for k, v in state_dict.items():
        out[k[len("module.") :] if k.startswith("module.") else k] = v
    return out


def _get_int(config: Mapping[str, Any], *keys: str, default: int) -> int:
    for key in keys:
        if key in config and config[key] is not None:
            try:
                return int(config[key])
            except (TypeError, ValueError):
                continue
    return int(default)


def _get_float(config: Mapping[str, Any], *keys: str, default: float) -> float:
    for key in keys:
        if key in config and config[key] is not None:
            try:
                return float(config[key])
            except (TypeError, ValueError):
                continue
    return float(default)


def _get_bool(config: Mapping[str, Any], *keys: str, default: bool) -> bool:
    for key in keys:
        if key in config and config[key] is not None:
            return bool(config[key])
    return bool(default)


def _get_str_list(
    config: Mapping[str, Any], key: str, default: Sequence[str] | None
) -> list[str] | None:
    if key not in config or config[key] is None:
        return list(default) if default is not None else None
    val = config[key]
    if isinstance(val, str):
        return [val]
    if isinstance(val, Sequence):
        return [str(x) for x in val]
    return list(default) if default is not None else None


def _infer_literal_dims(
    state_dict: Mapping[str, Tensor], config: Mapping[str, Any]
) -> tuple[int, int]:
    embed_dim = _get_int(config, "literal_embed_dim", "embed_dim", default=128)
    hidden_dim = _get_int(config, "literal_hidden_dim", "hidden_dim", default=embed_dim)

    w_hidden = state_dict.get("literal_encoder.mlp.0.weight")
    if isinstance(w_hidden, torch.Tensor) and w_hidden.ndim == 2:
        hidden_dim = int(w_hidden.shape[0])

    w_embed = state_dict.get("literal_encoder.mlp.2.weight")
    if isinstance(w_embed, torch.Tensor) and w_embed.ndim == 2:
        embed_dim = int(w_embed.shape[0])
        hidden_dim = int(w_embed.shape[1])

    return int(embed_dim), int(hidden_dim)


def _infer_t_max(state_dict: Mapping[str, Tensor], config: Mapping[str, Any]) -> int:
    for key in ("t_max", "T_max", "max_slots"):
        if key in config and config[key] is not None:
            try:
                return int(config[key])
            except (TypeError, ValueError):
                pass

    t = state_dict.get("clause_composer.query_embed")
    if isinstance(t, torch.Tensor) and t.ndim >= 1:
        return int(t.shape[0])

    return _get_int(config, "max_clauses", default=8)


def _infer_k_max(config: Mapping[str, Any]) -> int:
    return _get_int(config, "k_max", "K_max", "max_literals", default=4)


def _infer_literal_film_config(
    state_dict: Mapping[str, Tensor], config: Mapping[str, Any]
) -> LiteralFilmConfig | None:
    enabled = _get_bool(config, "literal_film_enabled", default=False)
    if not enabled:
        enabled = any(k.startswith("literal_conditioner.") for k in state_dict.keys())
    if not enabled:
        return None
    return LiteralFilmConfig(
        enabled=True,
        mode=str(config.get("literal_film_mode", "full")),
        beta_init=str(config.get("literal_film_beta_init", "orthogonal")),
        beta_std=_get_float(config, "literal_film_beta_std", default=0.5),
        gamma_init=str(config.get("literal_film_gamma_init", "normal")),
        gamma_mean=_get_float(config, "literal_film_gamma_mean", default=1.0),
        gamma_std=_get_float(config, "literal_film_gamma_std", default=0.5),
    )


def load_model(checkpoint_path: str, device: torch.device) -> RuleInducer:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint payload must be a mapping.")

    config = _extract_config(payload)
    state_dict = _strip_module_prefix(_extract_state_dict(payload))

    literal_embed_dim, literal_hidden_dim = _infer_literal_dims(state_dict, config)
    t_max = _infer_t_max(state_dict, config)
    k_max = _infer_k_max(config)

    gate_mode = str(config.get("gate_mode", "sigmoid"))
    literal_film_cfg = _infer_literal_film_config(state_dict, config)

    model = RuleInducer(
        literal_embed_dim=literal_embed_dim,
        literal_hidden_dim=literal_hidden_dim,
        clause_hidden_dim=_get_int(config, "clause_hidden_dim", default=literal_embed_dim),
        T_max=t_max,
        K_max=k_max,
        gate_mode=gate_mode,
        clause_topk=config.get("clause_topk"),
        literal_add_posneg_cooc=_get_bool(
            config, "literal_add_posneg_cooc", default=True
        ),
        literal_example_content_keys=_get_bool(
            config, "literal_example_content_keys", default=True
        ),
        literal_example_x_bottleneck=_get_int(
            config, "literal_example_x_bottleneck", default=64
        ),
        mutual_exclusion_hard=_get_bool(config, "mutual_exclusion_hard", default=False),
        setmatch_hidden_dim=config.get("setmatch_hidden_dim"),
        setmatch_num_layers=_get_int(config, "setmatch_num_layers", default=3),
        setmatch_num_heads=_get_int(config, "setmatch_num_heads", default=4),
        setmatch_dropout=_get_float(config, "setmatch_dropout", default=0.1),
        clause_dropout=_get_float(config, "clause_dropout", default=0.0),
        clause_dropout_min_keep=_get_int(config, "clause_dropout_min_keep", default=1),
        literal_film_config=literal_film_cfg,
    ).to(device)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    _materialize_lazy_example_proj_x(model.literal_encoder)
    if missing_keys:
        print(f"Warning: missing keys in checkpoint: {missing_keys}", file=sys.stderr)
    if unexpected_keys:
        print(
            f"Warning: unexpected keys in checkpoint: {unexpected_keys}",
            file=sys.stderr,
        )

    model.eval()
    return model


__all__ = [
    "_extract_state_dict",
    "_extract_config",
    "_strip_module_prefix",
    "_get_int",
    "_get_float",
    "_get_bool",
    "_get_str_list",
    "_infer_literal_dims",
    "_infer_t_max",
    "_infer_k_max",
    "_infer_literal_film_config",
    "_materialize_lazy_example_proj_x",
    "load_model",
]
