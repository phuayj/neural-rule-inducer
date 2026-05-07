from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import torch
from huggingface_hub import PyTorchModelHubMixin, constants, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from torch import Tensor, nn
from torch.nn import functional as F

from ._hub import (
    _materialize_lazy_example_proj_x,
    _materialize_lazy_example_proj_x_from_state,
    materialize_rule_inducer_for_hub,
)

try:
    from torch.nn import LazyLinear  # type: ignore
except ImportError:  # pragma: no cover - fallback for older PyTorch
    LazyLinear = None  # type: ignore


@dataclass
class RuleInducerOutput:
    """Container for model output tensors."""

    R_pred: Tensor
    Lit_logits: Tensor
    Lit_probs: Tensor
    Clause_gate_logits: Tensor
    Clause_truth: Tensor
    Complexity_metrics: Dict[str, Tensor]
    Projected_states: Optional[Tensor] = None


@dataclass
class LiteralFilmConfig:
    """Configuration for clause-conditioned literal FiLM modulation."""

    enabled: bool = False
    mode: str = "full"  # "full" (gamma + beta), "additive" (beta only), "none"

    # Strong initialization settings (proven critical for clause diversity)
    beta_init: str = "orthogonal"  # "orthogonal", "normal", "zeros"
    beta_std: float = 0.5  # std for normal init
    gamma_init: str = "normal"  # "normal", "ones"
    gamma_mean: float = 1.0
    gamma_std: float = 0.5  # 5x larger than typical; breaks symmetry


def _coerce_literal_film_config(
    config: LiteralFilmConfig | Mapping[str, Any] | None,
) -> LiteralFilmConfig | None:
    if config is None or isinstance(config, LiteralFilmConfig):
        return config
    return LiteralFilmConfig(**dict(config))


def _build_length_mask(lengths: Tensor, max_len: int) -> Tensor:
    """Returns a boolean mask of shape [B, max_len] covering valid positions."""
    device = lengths.device
    range_tensor = torch.arange(max_len, device=device).unsqueeze(0)
    return range_tensor < lengths.unsqueeze(1)


def _apply_mutual_exclusion_hard(
    logits: Tensor,
    mask: Tensor,
    fill_value: float,
) -> Tensor:
    """
    Applies a hard mutual exclusion by suppressing the lower-scoring literal in each ± pair.
    """
    if logits.shape[-1] < 2:
        return logits
    half = logits.shape[-1] // 2
    if half == 0:
        return logits

    pos_slice = logits[..., :half]
    neg_slice = logits[..., half : half + half]
    pos_mask = mask[..., :half]
    neg_mask = mask[..., half : half + half]

    both_valid = pos_mask & neg_mask
    prefer_pos = pos_slice >= neg_slice
    keep_pos = (~both_valid & pos_mask) | (both_valid & prefer_pos)
    keep_neg = (~both_valid & neg_mask) | (both_valid & (~prefer_pos))

    suppress_pos = pos_mask & ~keep_pos
    suppress_neg = neg_mask & ~keep_neg

    if suppress_pos.any():
        fill_tensor = torch.full_like(pos_slice, fill_value)
        pos_slice = torch.where(suppress_pos, fill_tensor, pos_slice)
    if suppress_neg.any():
        fill_tensor = torch.full_like(neg_slice, fill_value)
        neg_slice = torch.where(suppress_neg, fill_tensor, neg_slice)

    logits[..., :half] = pos_slice
    logits[..., half : half + half] = neg_slice
    return logits


def _binary_entropy(prob: Tensor, eps: float) -> Tensor:
    prob = prob.clamp(min=eps, max=1.0 - eps)
    return -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())


class LiteralStatsEncoder(nn.Module):
    """Episode → literal summariser. Computes per-literal feature statistics."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 128,
        *,
        literal_add_posneg_cooc: bool = True,
        literal_example_content_keys: bool = True,
        literal_example_x_bottleneck: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.literal_add_posneg_cooc = literal_add_posneg_cooc
        self.literal_example_content_keys = bool(literal_example_content_keys)
        self.literal_example_x_bottleneck = int(literal_example_x_bottleneck)
        if self.literal_example_content_keys and self.literal_example_x_bottleneck <= 0:
            raise ValueError(
                "literal_example_x_bottleneck must be positive when enabling content keys."
            )

        self.feature_dim = self._infer_feature_dim()
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.example_proj_y = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        if self.literal_example_content_keys:
            self.example_proj_x = nn.Sequential(
                nn.LazyLinear(self.literal_example_x_bottleneck),
                nn.ReLU(),
                nn.Linear(self.literal_example_x_bottleneck, embed_dim),
            )
            self._example_proj_x_in_dim: int | None = None
        else:
            self.example_proj_x = None
            self._example_proj_x_in_dim = None
        self.example_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True
        )

    def _infer_feature_dim(self) -> int:
        dim = 13  # base handcrafted features
        if self.literal_add_posneg_cooc:
            dim += 5  # mean(abs), mean, per-class diffs
        return dim

    def forward(
        self,
        X_val: Tensor,
        X_mask: Tensor,
        Y_val: Tensor,
        Y_mask: Tensor,
        N_len: Tensor,
        M_len: Tensor,
        H_len: Tensor,
    ) -> Dict[str, Tensor | object]:
        """
        Computes per-literal embeddings conditioned on labels per head.

        Returns a dict with:
            literal_embeddings: [B, H_max, 2*N_max, D]
            literal_valid_mask: [B, 1, 2*N_max] (bool)
            head_mask: [B, H_max] (bool)
            example_mask: [B, M_max] (bool)
        """

        eps = 1e-6
        B, M_max, N_max = X_val.shape
        _, _, H_max = Y_val.shape
        device = X_val.device

        example_mask = _build_length_mask(M_len, M_max)
        head_mask = _build_length_mask(H_len, H_max)
        atom_mask = _build_length_mask(N_len, N_max)

        literal_mask_bool = torch.cat([atom_mask, atom_mask], dim=-1).unsqueeze(1)
        literal_mask_bool = literal_mask_bool.expand(B, H_max, 2 * N_max)
        literal_mask_float = literal_mask_bool.to(X_val.dtype)

        x_known_mask = torch.cat([X_mask, X_mask], dim=-1)
        x_known_mask = x_known_mask.unsqueeze(1).expand(B, H_max, M_max, 2 * N_max)
        x_known_mask_float = x_known_mask.to(X_val.dtype)

        literal_truth = torch.cat([X_val, 1.0 - X_val], dim=-1)
        literal_truth = literal_truth.unsqueeze(1).expand(B, H_max, M_max, 2 * N_max)
        literal_truth = literal_truth * x_known_mask_float

        y_mask = Y_mask.bool()
        y_mask_float = y_mask.to(X_val.dtype)
        example_mask_float = example_mask.to(X_val.dtype)
        total_examples = example_mask_float.sum(dim=1, keepdim=True).clamp(
            min=1.0
        )  # [B,1]

        # Positive/negative weights per head for normalisation.
        pos_examples = (Y_val * y_mask_float).sum(dim=1)  # [B,H]
        neg_examples = ((1.0 - Y_val) * y_mask_float).sum(dim=1)  # [B,H]
        pos_weight = pos_examples + eps
        neg_weight = neg_examples + eps

        pos_num = torch.einsum("bmh,bhml->bhl", Y_val * y_mask_float, literal_truth)
        pos_den = torch.einsum(
            "bmh,bhml->bhl", Y_val * y_mask_float, x_known_mask_float
        )
        pos_true_rate = pos_num / (pos_den + eps)
        pos_obs_rate = pos_den / (pos_weight.unsqueeze(-1))

        neg_num = torch.einsum(
            "bmh,bhml->bhl", (1.0 - Y_val) * y_mask_float, literal_truth
        )
        neg_den = torch.einsum(
            "bmh,bhml->bhl", (1.0 - Y_val) * y_mask_float, x_known_mask_float
        )
        neg_true_rate = neg_num / (neg_den + eps)
        neg_obs_rate = neg_den / (neg_weight.unsqueeze(-1))

        total_num = torch.einsum("bm,bhml->bhl", example_mask_float, literal_truth)
        total_den = torch.einsum("bm,bhml->bhl", example_mask_float, x_known_mask_float)
        total_true_rate = total_num / (total_den + eps)
        total_obs_rate = total_den / total_examples.unsqueeze(-1)

        pos_false_rate = 1.0 - pos_true_rate
        neg_false_rate = 1.0 - neg_true_rate
        total_false_rate = 1.0 - total_true_rate

        entropy = _binary_entropy(total_true_rate, eps)

        literal_sign = torch.cat(
            [
                torch.ones(N_max, device=device, dtype=X_val.dtype),
                torch.zeros(N_max, device=device, dtype=X_val.dtype),
            ],
            dim=0,
        )
        literal_sign = literal_sign.view(1, 1, 2 * N_max).expand(B, H_max, 2 * N_max)

        # Stack handcrafted stats into feature tensor.
        centered_truth = (
            literal_truth - total_true_rate.unsqueeze(2)
        ) * x_known_mask_float
        cooc = torch.einsum("bhml,bhmn->bhln", centered_truth, centered_truth)
        denom_examples = x_known_mask_float.sum(dim=2).clamp(min=1.0)
        cooc = cooc / denom_examples.unsqueeze(-1)
        cooc_mask = literal_mask_float.unsqueeze(-1) * literal_mask_float.unsqueeze(-2)
        cooc = cooc * cooc_mask
        L_total = literal_truth.shape[-1]
        eye = torch.eye(L_total, device=device, dtype=cooc.dtype).view(
            1, 1, L_total, L_total
        )
        cooc = cooc * (1.0 - eye)
        other_count = (cooc_mask.sum(dim=-1) - literal_mask_float).clamp(min=1.0)
        cooc_strength = (cooc.abs().sum(dim=-1) / other_count) * literal_mask_float

        feature_components = [
            pos_true_rate,
            pos_false_rate,
            pos_obs_rate,
            neg_true_rate,
            neg_false_rate,
            neg_obs_rate,
            total_true_rate,
            total_false_rate,
            total_obs_rate,
            entropy,
            literal_sign,
            torch.zeros_like(literal_sign),  # reserved slot for future signals
            cooc_strength,
        ]

        Y_heads = Y_val.permute(0, 2, 1)
        Y_mask_heads = Y_mask.permute(0, 2, 1)
        Y_mask_heads_float = Y_mask_heads.to(X_val.dtype)
        pos_mask = (Y_heads * Y_mask_heads_float).unsqueeze(-1)
        neg_mask = ((1.0 - Y_heads) * Y_mask_heads_float).unsqueeze(-1)
        has_pos = (pos_examples > 0.0).to(X_val.dtype).unsqueeze(-1)
        has_neg = (neg_examples > 0.0).to(X_val.dtype).unsqueeze(-1)

        centered_pos = (
            (literal_truth - pos_true_rate.unsqueeze(2)) * x_known_mask_float * pos_mask
        )
        centered_neg = (
            (literal_truth - neg_true_rate.unsqueeze(2)) * x_known_mask_float * neg_mask
        )
        pos_cooc = torch.einsum("bhml,bhmn->bhln", centered_pos, centered_pos)
        neg_cooc = torch.einsum("bhml,bhmn->bhln", centered_neg, centered_neg)
        pos_known = torch.einsum(
            "bhm,bhml->bhl", pos_mask.squeeze(-1), x_known_mask_float
        ).clamp(min=1.0)
        neg_known = torch.einsum(
            "bhm,bhml->bhl", neg_mask.squeeze(-1), x_known_mask_float
        ).clamp(min=1.0)
        pos_cooc = (pos_cooc / pos_known.unsqueeze(-1)) * cooc_mask * (1.0 - eye)
        neg_cooc = (neg_cooc / neg_known.unsqueeze(-1)) * cooc_mask * (1.0 - eye)

        if self.literal_add_posneg_cooc:
            pos_abs_mean = (
                (pos_cooc.abs().sum(dim=-1) / other_count)
                * literal_mask_float
                * has_pos
            )
            pos_mean = (
                (pos_cooc.sum(dim=-1) / other_count) * literal_mask_float * has_pos
            )
            neg_abs_mean = (
                (neg_cooc.abs().sum(dim=-1) / other_count)
                * literal_mask_float
                * has_neg
            )
            neg_mean = (
                (neg_cooc.sum(dim=-1) / other_count) * literal_mask_float * has_neg
            )
            diff_mean = (pos_mean - neg_mean) * literal_mask_float
            feature_components.extend(
                [pos_abs_mean, pos_mean, neg_abs_mean, neg_mean, diff_mean]
            )

        feature_stack = torch.stack(feature_components, dim=-1)
        feature_stack = feature_stack * literal_mask_float.unsqueeze(-1)
        features = torch.nan_to_num(feature_stack, nan=0.0, posinf=1.0, neginf=0.0)

        literal_embeddings = self.mlp(features)
        literal_embeddings = literal_embeddings * literal_mask_float.unsqueeze(-1)

        # Example-conditioned refinement
        example_feats = torch.stack(
            [
                Y_heads,
                1.0 - Y_heads,
                Y_mask_heads_float,
            ],
            dim=-1,
        )
        example_emb = self.example_proj_y(example_feats)
        example_emb = example_emb * Y_mask_heads_float.unsqueeze(-1)

        if self.literal_example_content_keys and self.example_proj_x is not None:
            content_input = literal_truth * literal_mask_float.unsqueeze(2)
            content_input = content_input.reshape(B * H_max * M_max, 2 * N_max)
            current_dim = content_input.shape[-1]
            first_layer = self.example_proj_x[0]
            is_lazy_linear = LazyLinear is not None and isinstance(
                first_layer, LazyLinear
            )
            if not is_lazy_linear:
                layer_in = int(getattr(first_layer, "in_features", current_dim))
                if self._example_proj_x_in_dim != layer_in:
                    self._example_proj_x_in_dim = layer_in
            target_dim = self._example_proj_x_in_dim
            if target_dim is None:
                target_dim = current_dim
            if current_dim < target_dim:
                pad = target_dim - current_dim
                content_input = F.pad(content_input, (0, pad))
            elif current_dim > target_dim:
                first_layer = self.example_proj_x[0]
                last_layer = self.example_proj_x[2]
                device_layer = (
                    first_layer.weight.device
                    if hasattr(first_layer, "weight")
                    else content_input.device
                )
                dtype_layer = (
                    first_layer.weight.dtype
                    if hasattr(first_layer, "weight")
                    else content_input.dtype
                )
                new_first = nn.Linear(
                    current_dim, self.literal_example_x_bottleneck
                ).to(device_layer, dtype_layer)
                new_last = nn.Linear(
                    self.literal_example_x_bottleneck, self.embed_dim
                ).to(device_layer, dtype_layer)
                with torch.no_grad():
                    if hasattr(first_layer, "weight"):
                        copy_dim = min(
                            first_layer.weight.shape[1], new_first.weight.shape[1]
                        )
                        new_first.weight[:, :copy_dim] = first_layer.weight[
                            :, :copy_dim
                        ]
                        if first_layer.bias is not None and new_first.bias is not None:
                            new_first.bias.copy_(first_layer.bias)
                    if hasattr(last_layer, "weight"):
                        new_last.weight.copy_(last_layer.weight)
                        if last_layer.bias is not None and new_last.bias is not None:
                            new_last.bias.copy_(last_layer.bias)
                self.example_proj_x = nn.Sequential(new_first, nn.ReLU(), new_last)
                target_dim = current_dim
            self._example_proj_x_in_dim = target_dim
            content_emb = self.example_proj_x(content_input)
            content_emb = content_emb.view(B, H_max, M_max, self.embed_dim)
            content_emb = content_emb * example_mask_float.unsqueeze(1).unsqueeze(-1)
            example_emb = example_emb + content_emb

        L_total = literal_embeddings.shape[2]
        bh = B * H_max
        queries = literal_embeddings.view(bh, L_total, self.embed_dim)
        keys = example_emb.view(bh, M_max, self.embed_dim)
        key_padding = (~example_mask.unsqueeze(1).expand(B, H_max, M_max)).reshape(
            bh, M_max
        )
        head_mask_flat = head_mask.view(bh)
        queries = queries * head_mask_flat.view(bh, 1, 1).to(queries.dtype)
        keys = keys * head_mask_flat.view(bh, 1, 1).to(keys.dtype)
        attn_out, _ = self.example_attn(
            queries,
            keys,
            keys,
            key_padding_mask=key_padding,
        )
        attn_out = attn_out.view(B, H_max, 2 * N_max, self.embed_dim)
        literal_embeddings = literal_embeddings + attn_out
        literal_embeddings = literal_embeddings * literal_mask_float.unsqueeze(-1)
        literal_embeddings = literal_embeddings * head_mask.unsqueeze(-1).unsqueeze(
            -1
        ).to(literal_embeddings.dtype)

        return {
            "literal_embeddings": literal_embeddings,
            "literal_valid_mask": literal_mask_bool[:, :1, :],
            "head_mask": head_mask,
            "example_mask": example_mask,
            "pos_true_rate": pos_true_rate,
        }


class ClauseLiteralConditioner(nn.Module):
    """Produces per-clause literal embeddings from shared embeddings via FiLM.

    FiLM (Feature-wise Linear Modulation) transforms shared literal embeddings
    [B, H, L, D] into per-clause embeddings [B, H, T, L, D] using learnable
    per-clause gamma (scale) and beta (shift) parameters.

    Critical: Strong initialization is essential to break symmetry and encourage
    clause diversity. Weak init (std=0.1) keeps gamma≈1, beta≈0, causing all
    clauses to learn identical patterns.
    """

    def __init__(
        self,
        embed_dim: int,
        T_max: int,
        config: LiteralFilmConfig,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.T_max = T_max
        self.config = config
        self.mode = config.mode if config.enabled else "none"

        if self.mode != "none":
            # Beta (shift): [T_max, embed_dim]
            self.clause_beta = nn.Parameter(torch.zeros(T_max, embed_dim))
            self._init_beta()

            if self.mode == "full":
                # Gamma (scale): [T_max, embed_dim]
                self.clause_gamma = nn.Parameter(torch.ones(T_max, embed_dim))
                self._init_gamma()

    def _init_beta(self) -> None:
        """Initialize beta with strong diversity-inducing values."""
        cfg = self.config
        if cfg.beta_init == "orthogonal":
            if self.T_max <= self.embed_dim:
                nn.init.orthogonal_(self.clause_beta, gain=1.0)
            else:
                # Fallback for T_max > embed_dim
                nn.init.normal_(self.clause_beta, std=cfg.beta_std)
        elif cfg.beta_init == "normal":
            nn.init.normal_(self.clause_beta, std=cfg.beta_std)
        # "zeros" is the default (already initialized)

    def _init_gamma(self) -> None:
        """Initialize gamma with high variance around 1.0."""
        cfg = self.config
        if cfg.gamma_init == "normal":
            nn.init.normal_(self.clause_gamma, mean=cfg.gamma_mean, std=cfg.gamma_std)
        # "ones" is the default (already initialized)

    def forward(self, literal_embeddings: Tensor) -> Tensor:
        """Transform shared literal embeddings to per-clause embeddings.

        Args:
            literal_embeddings: [B, H, L, D] shared literal embeddings

        Returns:
            [B, H, T, L, D] per-clause literal embeddings
        """
        B, H, L, D = literal_embeddings.shape

        # Expand: [B, H, L, D] -> [B, H, T, L, D]
        expanded = literal_embeddings.unsqueeze(2).expand(B, H, self.T_max, L, D)

        if self.mode == "none":
            return expanded.contiguous()

        # Apply FiLM: gamma * x + beta
        if self.mode == "full" and hasattr(self, "clause_gamma"):
            gamma = self.clause_gamma.view(1, 1, self.T_max, 1, D)
            expanded = gamma * expanded

        beta = self.clause_beta.view(1, 1, self.T_max, 1, D)
        expanded = expanded + beta

        return expanded.contiguous()


class ClauseComposerSetDecoder(nn.Module):
    """Set-based clause decoder using transformer-style cross attention."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int],
        T_max: int,
        K_max: int,
        *,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        mask_fill_value: float = -1e2,
        mutual_exclusion_hard: bool = False,
    ):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        if T_max < 0:
            raise ValueError("T_max must be non-negative")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.T_max = T_max
        self.K_max = K_max
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = float(dropout)
        self.mask_fill_value = mask_fill_value
        self.mutual_exclusion_hard = bool(mutual_exclusion_hard)

        self.literal_proj = (
            nn.Linear(embed_dim, hidden_dim)
            if embed_dim != hidden_dim
            else nn.Identity()
        )
        self.literal_feature = nn.Linear(embed_dim, hidden_dim)
        self.head_context_proj = nn.Linear(embed_dim, hidden_dim)
        self.query_embed = nn.Parameter(torch.empty(T_max, hidden_dim))
        if T_max > 0:
            nn.init.normal_(self.query_embed, mean=0.0, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=self.dropout,
            batch_first=False,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.logit_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.logit_bias = nn.Parameter(torch.zeros(1))
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.scale = hidden_dim**-0.5

    def forward(
        self,
        literal_embeddings: Tensor,
        literal_valid_mask: Tensor,
        head_mask: Tensor,
        *,
        literal_embeddings_per_clause: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        B, H_max, L, _ = literal_embeddings.shape
        dtype = literal_embeddings.dtype

        if self.T_max == 0:
            empty_logits = literal_embeddings.new_empty(B, H_max, 0, L)
            empty_gates = literal_embeddings.new_empty(B, H_max, 0)
            return {
                "Lit_logits": empty_logits,
                "Clause_gate_logits": empty_gates,
            }

        literal_mask = literal_valid_mask.unsqueeze(1).expand(B, H_max, self.T_max, L)
        mask_fill = (
            torch.finfo(dtype).min
            if torch.is_floating_point(literal_embeddings)
            else self.mask_fill_value
        )

        literal_float_mask = literal_valid_mask.to(dtype)
        valid_count = literal_float_mask.sum(dim=2, keepdim=True).clamp(min=1.0)
        head_summary = (literal_embeddings * literal_float_mask.unsqueeze(-1)).sum(
            dim=2
        ) / valid_count

        memory = self.literal_proj(literal_embeddings)
        memory = memory.view(B * H_max, L, self.hidden_dim).transpose(0, 1)
        # Expand mask to all heads before reshaping (literal_valid_mask is [B, 1, L]).
        literal_valid_mask_expanded = literal_valid_mask.expand(B, H_max, L)
        memory_key_padding = (~literal_valid_mask_expanded.reshape(B * H_max, L)).to(
            torch.bool
        )

        if self.query_embed.numel() == 0:
            tgt = literal_embeddings.new_empty(0, B * H_max, self.hidden_dim)
        else:
            tgt = self.query_embed.unsqueeze(1).expand(
                self.T_max, B * H_max, self.hidden_dim
            )

        context = self.head_context_proj(head_summary).view(B * H_max, self.hidden_dim)
        tgt = tgt + context.view(1, B * H_max, self.hidden_dim)

        decoder_out = self.decoder(
            tgt,
            memory,
            tgt_mask=None,
            memory_key_padding_mask=memory_key_padding,
        )
        clause_state = self.decoder_norm(decoder_out).transpose(0, 1)
        clause_state = clause_state.view(B, H_max, self.T_max, self.hidden_dim)

        projected = self.logit_proj(clause_state)  # [B, H, T, D]

        if literal_embeddings_per_clause is not None:
            if literal_embeddings_per_clause.ndim != 5:
                raise ValueError(
                    "literal_embeddings_per_clause must be [B,H,T,L,D], got "
                    f"{tuple(literal_embeddings_per_clause.shape)}"
                )
            if (
                literal_embeddings_per_clause.shape[0] != B
                or literal_embeddings_per_clause.shape[1] != H_max
                or literal_embeddings_per_clause.shape[2] != self.T_max
                or literal_embeddings_per_clause.shape[3] != L
            ):
                raise ValueError(
                    "literal_embeddings_per_clause must align with literal_embeddings on [B,H,T,L] dims, got "
                    f"literal_embeddings={tuple(literal_embeddings.shape)} "
                    f"literal_embeddings_per_clause={tuple(literal_embeddings_per_clause.shape)}"
                )

            # Per-clause literal features: [B,H,T,L,D]
            literal_features = self.literal_feature(literal_embeddings_per_clause)

            logits = (
                torch.einsum(
                    "bhtd,bhtld->bhtl",
                    projected,
                    literal_features,
                )
                * self.scale
            )
        else:
            # Backward compatible: [B,H,L,D]
            literal_features = self.literal_feature(literal_embeddings)

            logits = (
                torch.einsum(
                    "bhtd,bhld->bhtl",
                    projected,
                    literal_features,
                )
                * self.scale
            )
        logits = logits + self.logit_bias.view(1, 1, 1, 1)
        logits = logits.masked_fill(~literal_mask, mask_fill)

        if self.mutual_exclusion_hard:
            logits = _apply_mutual_exclusion_hard(logits, literal_mask, mask_fill)

        literal_probs = torch.sigmoid(logits)
        literal_probs = literal_probs * literal_mask.to(literal_probs.dtype)
        literal_probs = torch.nan_to_num(literal_probs, nan=0.0, posinf=0.0, neginf=0.0)
        literal_norm = literal_probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        if literal_embeddings_per_clause is not None:
            clause_literal_summary = (
                torch.einsum(
                    "bhtl,bhtld->bhtd",
                    literal_probs,
                    literal_embeddings_per_clause,
                )
                / literal_norm
            )
        else:
            clause_literal_summary = (
                torch.einsum("bhtl,bhld->bhtd", literal_probs, literal_embeddings)
                / literal_norm
            )
        clause_non_null = 1.0 - torch.prod(
            1.0 - literal_probs.clamp(min=0.0, max=1.0), dim=-1
        )

        head_repeat = head_summary.unsqueeze(2).expand(-1, -1, self.T_max, -1)
        gate_input = torch.cat(
            [
                head_repeat,
                clause_literal_summary,
                clause_state,
                clause_non_null.unsqueeze(-1),
            ],
            dim=-1,
        )
        clause_gate_logits = self.gate_mlp(gate_input).squeeze(-1)

        head_mask = head_mask.unsqueeze(-1)
        clause_gate_logits = clause_gate_logits.masked_fill(
            ~head_mask,
            torch.finfo(clause_gate_logits.dtype).min,
        )

        return {
            "Lit_logits": logits,
            "Clause_gate_logits": clause_gate_logits,
            "projected_clause_states": projected,
        }


class RuleAggregator(nn.Module):
    """Soft rule evaluation with product t-norm/t-conorm semantics."""

    def __init__(
        self,
        gate_mode: str = "sigmoid",
        *,
        clause_topk: Optional[int] = None,
        mask_fill_value: float = -1e2,
        clause_dropout: float = 0.0,  # Fraction of clauses to drop during training (0.0 = disabled)
        clause_dropout_min_keep: int = 1,  # Minimum number of clauses to keep
    ):
        super().__init__()
        self.gate_mode = gate_mode
        self.clause_dropout = float(clause_dropout)
        if not (0.0 <= self.clause_dropout <= 1.0):
            raise ValueError("clause_dropout must be in [0, 1].")
        self.clause_dropout_min_keep = int(clause_dropout_min_keep)
        if self.clause_dropout_min_keep < 1:
            raise ValueError("clause_dropout_min_keep must be >= 1.")

        self.clause_topk = clause_topk
        self.mask_fill_value = mask_fill_value

    def _reduce_literal_t_norm(self, values: Tensor) -> Tensor:
        """Aggregate literal truth values with the product t-norm."""
        return values.prod(dim=-1).clamp(min=0.0, max=1.0)

    def _reduce_clause_s_norm(self, values: Tensor) -> Tensor:
        """Aggregate clause truth values with the product s-norm (noisy-or)."""
        return 1.0 - torch.prod(1.0 - values.clamp(min=1e-6, max=1.0), dim=2)

    def forward(
        self,
        Lit_logits: Tensor,
        Clause_gate_logits: Tensor,
        X_val: Tensor,
        X_mask: Tensor,
        example_mask: Tensor,
        head_mask: Tensor,
        gumbel: bool = False,
        gate_mode_override: Optional[str] = None,
        clause_topk_override: Optional[int] = None,
    ) -> RuleInducerOutput:
        B, M_max, N_max = X_val.shape
        _, H_max, T_max = Clause_gate_logits.shape

        literal_truth = torch.cat([X_val, 1.0 - X_val], dim=-1)
        literal_known_mask = torch.cat([X_mask, X_mask], dim=-1)
        literal_truth = torch.where(
            literal_known_mask.bool(),
            literal_truth,
            torch.full_like(literal_truth, 0.5),
        )

        logits = Lit_logits
        if gumbel:
            uniform = torch.rand_like(logits)
            uniform = uniform.clamp_min(1e-8).clamp_max(1.0 - 1e-8)
            logistic_noise = torch.log(uniform) - torch.log(1.0 - uniform)
            logits = logits + logistic_noise

        literal_probs = torch.sigmoid(logits)
        literal_probs = torch.nan_to_num(literal_probs, nan=0.0, posinf=0.0, neginf=0.0)
        literal_probs = literal_probs * head_mask.to(literal_probs.dtype).unsqueeze(
            -1
        ).unsqueeze(-1)

        gate_mode = gate_mode_override or self.gate_mode
        if gate_mode not in {"topk", "sigmoid"}:
            raise ValueError(f"Unsupported gate mode override '{gate_mode}'.")
        if gate_mode == "topk":
            gate_logits = Clause_gate_logits
            if gumbel:
                uniform_gate = torch.rand_like(gate_logits)
                uniform_gate = uniform_gate.clamp_min(1e-8).clamp_max(1.0 - 1e-8)
                gumbel_gate = -torch.log(-torch.log(uniform_gate))
                gate_logits = gate_logits + gumbel_gate
            probs = torch.softmax(gate_logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            k_override = (
                clause_topk_override
                if clause_topk_override is not None
                else self.clause_topk
            )
            k = k_override if k_override is not None and k_override > 0 else 1
            k = max(1, min(k, gate_logits.shape[-1]))
            topk_values, topk_idx = gate_logits.topk(k, dim=-1)
            hard = torch.zeros_like(gate_logits)
            hard.scatter_(-1, topk_idx, 1.0)
            clause_gate = hard + (probs - probs.detach())
            clause_gate = clause_gate * head_mask.unsqueeze(-1).to(clause_gate.dtype)
            gate_expected = probs
        else:
            if gumbel:
                uniform_gate = torch.rand_like(Clause_gate_logits)
                logistic_noise_gate = torch.log(
                    uniform_gate.clamp_min(1e-8)
                ) - torch.log((1.0 - uniform_gate).clamp_min(1e-8))
                clause_gate = torch.sigmoid(Clause_gate_logits + logistic_noise_gate)
            else:
                clause_gate = torch.sigmoid(Clause_gate_logits)
            clause_gate = clause_gate.clamp(0.0, 1.0)
            gate_expected = clause_gate

        truth_exp = literal_truth.unsqueeze(1).unsqueeze(2)  # [B,1,1,M,L]
        literal_probs_exp = literal_probs.unsqueeze(3)  # [B,H,T,1,L]

        # Compute clause truth
        literal_component = 1.0 - literal_probs_exp + literal_probs_exp * truth_exp
        literal_component = literal_component.clamp(min=1e-6, max=1.0)

        clause_truth = self._reduce_literal_t_norm(literal_component)  # [B,H,T,M]
        clause_non_null = 1.0 - torch.prod(1.0 - literal_probs.clamp(0.0, 1.0), dim=-1)
        clause_truth = clause_truth * clause_non_null.unsqueeze(-1)

        clause_truth = clause_truth * clause_gate.unsqueeze(-1)

        # Clause dropout: randomly drop a subset of clauses during training so each clause
        # must be independently useful.
        clause_dropout_keep_fraction: Optional[Tensor] = None
        if self.training and self.clause_dropout > 0.0:
            B_cd, H_cd, T_cd, _M_cd = clause_truth.shape
            if T_cd > 0:
                keep_prob = 1.0 - float(self.clause_dropout)
                min_keep = max(1, int(self.clause_dropout_min_keep))
                min_keep = min(min_keep, int(T_cd))

                if min_keep >= T_cd:
                    clause_drop_mask = torch.ones(
                        (B_cd, H_cd, T_cd, 1),
                        device=clause_truth.device,
                        dtype=torch.bool,
                    )
                else:
                    clause_drop_mask = (
                        torch.rand(
                            (B_cd, H_cd, T_cd, 1),
                            device=clause_truth.device,
                        )
                        < keep_prob
                    )
                    kept_count = clause_drop_mask.sum(dim=2, keepdim=True)  # [B,H,1,1]
                    need_fix = kept_count < min_keep
                    if need_fix.any():
                        scores = torch.rand(
                            (B_cd, H_cd, T_cd),
                            device=clause_truth.device,
                        )
                        topk_idx = scores.topk(min_keep, dim=2).indices  # [B,H,K]
                        force_mask = torch.zeros(
                            (B_cd, H_cd, T_cd),
                            device=clause_truth.device,
                            dtype=torch.bool,
                        )
                        force_mask.scatter_(2, topk_idx, True)
                        force_mask = force_mask.unsqueeze(-1)
                        clause_drop_mask = clause_drop_mask | (
                            force_mask & need_fix.expand_as(clause_drop_mask)
                        )

                clause_dropout_keep_fraction = clause_drop_mask.to(
                    clause_truth.dtype
                ).mean()
                clause_truth = clause_truth * clause_drop_mask.to(clause_truth.dtype)

        clause_truth_clamped = clause_truth.clamp(min=0.0, max=1.0)
        rule_truth = self._reduce_clause_s_norm(clause_truth_clamped)
        rule_truth = rule_truth.permute(0, 2, 1)

        example_mask_expanded = example_mask.to(rule_truth.dtype).unsqueeze(-1)
        head_mask = head_mask.to(rule_truth.dtype)
        rule_truth = rule_truth * example_mask_expanded
        rule_truth = rule_truth * head_mask.unsqueeze(1)

        clause_truth = clause_truth.permute(0, 3, 1, 2)
        clause_truth = clause_truth * example_mask_expanded.unsqueeze(-1)
        clause_truth = clause_truth * head_mask.unsqueeze(1).unsqueeze(-1)

        head_mask_bool = head_mask.bool()
        literal_mass = literal_probs * head_mask.to(literal_probs.dtype).unsqueeze(
            -1
        ).unsqueeze(-1)
        expected_literals = literal_mass.sum(dim=-1)
        expected_clauses = gate_expected * head_mask.to(gate_expected.dtype).unsqueeze(
            -1
        )
        gate_probs_for_mask = gate_expected
        clause_presence_mask = (gate_probs_for_mask > 0.0) & head_mask_bool.unsqueeze(
            -1
        )
        literal_mean = literal_probs.mean(dim=-1)
        literal_std = literal_probs.std(dim=-1, unbiased=False)

        complexity = {
            "expected_literals_per_clause": expected_literals,
            "literal_activation_prob": literal_probs,
            "expected_active_clauses": expected_clauses,
            "clause_non_null_prob": clause_non_null,
            "gate_expected": gate_expected,
            "clause_presence_mask": clause_presence_mask,
        }
        if clause_dropout_keep_fraction is not None:
            complexity["clause_dropout_keep_fraction"] = clause_dropout_keep_fraction

        return RuleInducerOutput(
            R_pred=rule_truth,
            Lit_logits=Lit_logits,
            Lit_probs=literal_probs,
            Clause_gate_logits=Clause_gate_logits,
            Clause_truth=clause_truth,
            Complexity_metrics=complexity,
        )


class RuleInducer(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="rule-inducer",
    tags=[
        "rule-induction",
        "neuro-symbolic",
        "logic-programming",
        "ilp",
        "interpretability",
        "zero-shot",
        "pytorch",
    ],
    repo_url="https://github.com/phuayj/neural-rule-inducer",
    paper_url="https://arxiv.org/abs/2605.04916",
    license="mit",
    pipeline_tag="other",
):
    """High-level module that wires together encoder, composer, and aggregator."""

    def __init__(
        self,
        literal_embed_dim: int = 128,
        literal_hidden_dim: int = 128,
        clause_hidden_dim: Optional[int] = None,
        T_max: int = 4,
        K_max: int = 4,
        gate_mode: str = "sigmoid",
        clause_topk: Optional[int] = None,
        mask_fill_value: float = -1e9,
        setmatch_hidden_dim: Optional[int] = None,
        setmatch_num_layers: int = 3,
        setmatch_num_heads: int = 4,
        setmatch_dropout: float = 0.1,
        literal_film_config: LiteralFilmConfig | dict[str, Any] | None = None,
        literal_add_posneg_cooc: bool = True,
        literal_example_content_keys: bool = True,
        literal_example_x_bottleneck: int = 64,
        mutual_exclusion_hard: bool = False,
        clause_dropout: float = 0.0,
        clause_dropout_min_keep: int = 1,
    ):
        super().__init__()
        literal_film_config = _coerce_literal_film_config(literal_film_config)
        self.literal_encoder = LiteralStatsEncoder(
            embed_dim=literal_embed_dim,
            hidden_dim=literal_hidden_dim,
            literal_add_posneg_cooc=literal_add_posneg_cooc,
            literal_example_content_keys=literal_example_content_keys,
            literal_example_x_bottleneck=literal_example_x_bottleneck,
        )
        self.mutual_exclusion_hard = bool(mutual_exclusion_hard)
        set_hidden = setmatch_hidden_dim or clause_hidden_dim or literal_embed_dim
        self.clause_composer = ClauseComposerSetDecoder(
            embed_dim=literal_embed_dim,
            hidden_dim=set_hidden,
            T_max=T_max,
            K_max=K_max,
            num_layers=setmatch_num_layers,
            num_heads=setmatch_num_heads,
            dropout=setmatch_dropout,
            mask_fill_value=mask_fill_value,
            mutual_exclusion_hard=self.mutual_exclusion_hard,
        )

        # Initialize clause literal conditioner if enabled
        self.literal_film_config: LiteralFilmConfig | None = literal_film_config
        self.literal_conditioner: ClauseLiteralConditioner | None = None
        if literal_film_config is not None and literal_film_config.enabled:
            # Get T_max from the clause composer
            T_max_value = getattr(self.clause_composer, "T_max", T_max)
            self.literal_conditioner = ClauseLiteralConditioner(
                embed_dim=literal_embed_dim,
                T_max=T_max_value,
                config=literal_film_config,
            )

        self.aggregator = RuleAggregator(
            gate_mode=gate_mode,
            clause_topk=clause_topk,
            mask_fill_value=mask_fill_value,
            clause_dropout=clause_dropout,
            clause_dropout_min_keep=clause_dropout_min_keep,
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save Hub inference weights after materializing lazy projection layers."""
        model_to_save = self.module if hasattr(self, "module") else self
        materialize_rule_inducer_for_hub(model_to_save)
        state_dict = model_to_save.state_dict()

        try:
            from safetensors.torch import save_file

            save_file(state_dict, str(Path(save_directory) / constants.SAFETENSORS_SINGLE_FILE))
        except ImportError:
            torch.save(state_dict, Path(save_directory) / constants.PYTORCH_WEIGHTS_NAME)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs: Any,
    ) -> "RuleInducer":
        """Load Hub inference weights with LazyLinear materialized from saved shapes."""
        literal_film_config = model_kwargs.get("literal_film_config")
        if isinstance(literal_film_config, Mapping):
            model_kwargs["literal_film_config"] = LiteralFilmConfig(
                **dict(literal_film_config)
            )

        model = cls(**model_kwargs)
        model_file = cls._resolve_hub_weight_file(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
        )
        state_dict = cls._load_hub_state_dict(model_file, map_location=map_location)
        _materialize_lazy_example_proj_x_from_state(model.literal_encoder, state_dict)
        model.load_state_dict(state_dict, strict=strict)
        _materialize_lazy_example_proj_x(model.literal_encoder)
        model.to(torch.device(map_location))
        model.eval()
        return model

    @staticmethod
    def _resolve_hub_weight_file(
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        local_files_only: bool,
        token: str | bool | None,
    ) -> Path:
        if os.path.isdir(model_id):
            model_dir = Path(model_id)
            safetensors_path = model_dir / constants.SAFETENSORS_SINGLE_FILE
            if safetensors_path.exists():
                return safetensors_path
            return model_dir / constants.PYTORCH_WEIGHTS_NAME

        try:
            return Path(
                hf_hub_download(
                    repo_id=model_id,
                    filename=constants.SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            )
        except EntryNotFoundError:
            return Path(
                hf_hub_download(
                    repo_id=model_id,
                    filename=constants.PYTORCH_WEIGHTS_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            )

    @staticmethod
    def _load_hub_state_dict(
        model_file: Path,
        *,
        map_location: str,
    ) -> dict[str, Tensor]:
        if model_file.name == constants.SAFETENSORS_SINGLE_FILE:
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError(
                    "Loading model.safetensors requires the optional safetensors package."
                ) from exc
            return load_file(str(model_file), device=map_location)
        loaded = torch.load(
            model_file,
            map_location=torch.device(map_location),
            weights_only=True,
        )
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Hub weight file must contain a state dict: {model_file}")
        return {str(k): v for k, v in loaded.items() if isinstance(v, torch.Tensor)}

    def forward(
        self,
        X_val: Tensor,
        X_mask: Tensor,
        Y_val: Tensor,
        Y_mask: Tensor,
        N_len: Tensor,
        M_len: Tensor,
        H_len: Tensor,
        gumbel: bool = False,
        gate_mode_override: Optional[str] = None,
        clause_topk_override: Optional[int] = None,
    ) -> RuleInducerOutput:
        literals = self.literal_encoder(
            X_val=X_val,
            X_mask=X_mask,
            Y_val=Y_val,
            Y_mask=Y_mask,
            N_len=N_len,
            M_len=M_len,
            H_len=H_len,
        )

        literal_embeddings = literals["literal_embeddings"]
        B, H_max, L_total, _ = literal_embeddings.shape
        literal_valid_mask = literals["literal_valid_mask"]
        head_mask = literals["head_mask"]

        # Compute per-clause literal embeddings if FiLM is enabled
        literal_embeddings_per_clause: Tensor | None = None
        if self.literal_conditioner is not None:
            literal_embeddings_per_clause = self.literal_conditioner(literal_embeddings)

        composer_out = self.clause_composer(
            literal_embeddings=literal_embeddings,
            literal_valid_mask=literal_valid_mask,
            head_mask=head_mask,
            literal_embeddings_per_clause=literal_embeddings_per_clause,
        )

        agg_output = self.aggregator(
            Lit_logits=composer_out["Lit_logits"],
            Clause_gate_logits=composer_out["Clause_gate_logits"],
            X_val=X_val,
            X_mask=X_mask,
            example_mask=literals["example_mask"],
            head_mask=head_mask,
            gumbel=gumbel,
            gate_mode_override=gate_mode_override,
            clause_topk_override=clause_topk_override,
        )
        clause_summary_from_composer = composer_out.get("Clause_literal_summary")
        if clause_summary_from_composer is not None:
            agg_output.Complexity_metrics["clause_literal_summary"] = (
                clause_summary_from_composer
            )

        projected_clause_states = composer_out.get("projected_clause_states")
        if isinstance(projected_clause_states, torch.Tensor):
            agg_output.Complexity_metrics["projected_clause_states"] = (
                projected_clause_states
            )
            agg_output.Projected_states = projected_clause_states
        return agg_output

    @torch.no_grad()
    def export_program(
        self,
        output: RuleInducerOutput,
        N_len: Tensor,
        H_len: Tensor,
        M_len: Optional[Tensor] = None,
        clause_threshold: float = 0.5,
        literal_threshold: float = 0.5,
        merge_threshold: Optional[float] = None,
        max_clauses: Optional[int] = None,
        literal_topk: Optional[int] = None,
        literal_percentile: Optional[float] = None,
        literal_threshold_floor: float = 0.0,
        literal_negative_boost: float = 0.0,
        literal_negative_limit: Optional[int] = None,
        literal_negative_percentile: Optional[float] = None,
        literal_negative_threshold_floor: Optional[float] = None,
        literal_negative_keep_topk: Optional[int] = None,
        literal_negative_rank_boost: float = 0.0,
        literal_negative_priority: float = 1.0,
        min_literals: int = 0,
        export_strategy: str = "default",
        bridge_disable_positive_fallback: bool = False,
        return_diagnostics: bool = False,
        export_mode: str = "default",
        literal_dedup_threshold: Optional[float] = None,
        calibration: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Tensor | object]:
        """
        Converts soft literal selections into discrete rule representations.

        Args:
            literal_negative_boost: Amount to lower the literal threshold for candidate
                negative literals (before clamping by `literal_threshold_floor`).
            literal_negative_limit: Maximum number of negative literals allowed per
                exported clause when applying thresholding/top-k filtering. `None`
                keeps all negatives that satisfy the thresholds.
            literal_negative_percentile: Optional percentile used to set per-clause thresholds
                based only on negative literals.
            literal_negative_threshold_floor: Minimum allowable threshold for negative literals
                after percentile/boost adjustments (defaults to literal_threshold_floor).
            literal_negative_keep_topk: Ensure at least this many negative literals per clause
                are retained (before applying literal_negative_limit).
            literal_negative_rank_boost: Additive bonus applied when ranking negatives for
                top-k selection. Does not change probability thresholds; only influences which
                literals enter the candidate pool.
            export_mode: Preset bundle for export heuristics. ``"recall_first"`` lowers literal
                thresholds, encourages negative retention, and enables literal-level deduplication.
            literal_dedup_threshold: Optional literal-level Jaccard threshold applied after
                clause expansion to drop near-duplicate clauses (defaults depend on export_mode).
            calibration: Optional mapping with keys ``positive_scale``, ``positive_bias``,
                ``negative_scale``, and ``negative_bias`` used to affine-transform literal logits
                prior to thresholding (acts as a simple isotonic-style calibration).
        """

        lit_probs = output.Lit_probs
        Clause_gate_logits = output.Clause_gate_logits

        B, H_max, T_max, literal_dim = lit_probs.shape
        device = lit_probs.device
        N_max = literal_dim // 2

        clause_prob = torch.sigmoid(Clause_gate_logits)

        if literal_percentile is not None and not (
            0.0 <= float(literal_percentile) <= 1.0
        ):
            raise ValueError("literal_percentile must be in [0, 1]")
        if min_literals < 0:
            raise ValueError("min_literals must be non-negative")
        if literal_negative_boost < 0.0:
            raise ValueError("literal_negative_boost must be non-negative")
        if literal_negative_limit is not None and literal_negative_limit < 0:
            raise ValueError(
                "literal_negative_limit must be non-negative when provided"
            )
        if literal_negative_percentile is not None and not (
            0.0 <= float(literal_negative_percentile) <= 1.0
        ):
            raise ValueError("literal_negative_percentile must be in [0, 1]")
        if (
            literal_negative_threshold_floor is not None
            and literal_negative_threshold_floor < 0.0
        ):
            raise ValueError(
                "literal_negative_threshold_floor must be non-negative when provided"
            )
        if literal_negative_keep_topk is not None and literal_negative_keep_topk < 0:
            raise ValueError(
                "literal_negative_keep_topk must be non-negative when provided"
            )
        if literal_negative_rank_boost < 0.0:
            raise ValueError("literal_negative_rank_boost must be non-negative")
        if literal_negative_priority <= 0.0:
            raise ValueError("literal_negative_priority must be positive")
        bridge_disable_positive_fallback = bool(bridge_disable_positive_fallback)
        export_strategy_value = export_strategy.lower()
        if export_strategy_value not in {"default", "bridge_v1"}:
            raise ValueError(f"Unsupported export_strategy: {export_strategy}")

        export_mode_value = export_mode.lower()
        if export_mode_value not in {"default", "recall_first"}:
            raise ValueError(f"Unsupported export_mode: {export_mode}")
        recall_first_mode = export_mode_value == "recall_first"

        literal_percentile_value = literal_percentile
        literal_negative_percentile_value = literal_negative_percentile
        literal_topk_value = literal_topk
        literal_negative_keep_topk_value = literal_negative_keep_topk
        literal_negative_boost_value = float(literal_negative_boost)
        literal_negative_priority_value = float(literal_negative_priority)
        literal_threshold_floor_value = float(literal_threshold_floor)
        neg_threshold_floor_value = (
            float(literal_negative_threshold_floor)
            if literal_negative_threshold_floor is not None
            else literal_threshold_floor_value
        )
        literal_dedup_threshold_value = (
            float(literal_dedup_threshold)
            if literal_dedup_threshold is not None
            else None
        )

        if recall_first_mode:
            if literal_percentile_value is None:
                literal_percentile_value = 0.30
            literal_threshold_floor_value = min(literal_threshold_floor_value, 0.05)
            if literal_topk_value is None:
                literal_topk_value = 8
            if literal_negative_percentile_value is None:
                literal_negative_percentile_value = literal_percentile_value
            if literal_negative_keep_topk_value is None:
                literal_negative_keep_topk_value = 1
            if literal_dedup_threshold_value is None:
                literal_dedup_threshold_value = 0.9
            neg_threshold_floor_value = min(
                neg_threshold_floor_value, literal_threshold_floor_value
            )

        head_mask = _build_length_mask(H_len.to(Clause_gate_logits.device), H_max)
        clause_active = clause_prob >= clause_threshold
        clause_active = clause_active & head_mask.unsqueeze(-1)

        literal_indices = torch.arange(literal_dim, device=device).view(
            1, 1, 1, literal_dim
        )
        literal_cap = (N_len.to(device) * 2).view(B, 1, 1, 1)
        literal_valid_mask = literal_indices < literal_cap

        if calibration:
            lit_probs = lit_probs.clone()
            logits = torch.logit(lit_probs.clamp(min=1e-6, max=1.0 - 1e-6))
            pos_scale = float(calibration.get("positive_scale", 1.0))
            neg_scale = float(calibration.get("negative_scale", 1.0))
            pos_bias = float(calibration.get("positive_bias", 0.0))
            neg_bias = float(calibration.get("negative_bias", 0.0))
            pos_len = min(N_max, literal_dim)
            if pos_len > 0:
                logits[..., :pos_len] = logits[..., :pos_len] * pos_scale + pos_bias
            neg_start = N_max
            neg_len = min(N_max, max(literal_dim - neg_start, 0))
            neg_end = min(neg_start + neg_len, literal_dim)
            if neg_end > neg_start:
                logits[..., neg_start:neg_end] = (
                    logits[..., neg_start:neg_end] * neg_scale + neg_bias
                )
            lit_probs = torch.sigmoid(logits)

        lit_probs_masked = lit_probs * literal_valid_mask.to(lit_probs.dtype)

        literal_thresholds = torch.full(
            (B, H_max, T_max),
            float(literal_threshold),
            dtype=lit_probs.dtype,
            device=device,
        )
        adaptive_thresholds: Optional[Tensor] = None
        if literal_percentile_value is not None:
            adaptive_thresholds = torch.full_like(
                literal_thresholds, float(literal_threshold)
            )
            percent = float(literal_percentile_value)
            for b_idx in range(B):
                valid_mask = literal_valid_mask[b_idx, 0, 0]
                valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
                if valid_indices.numel() == 0:
                    continue
                for h_idx in range(H_max):
                    for t_idx in range(T_max):
                        clause_vec = lit_probs[b_idx, h_idx, t_idx, valid_indices]
                        if clause_vec.numel() == 0:
                            continue
                        quantile = torch.quantile(clause_vec, percent)
                        adaptive_thresholds[b_idx, h_idx, t_idx] = quantile
            literal_thresholds = torch.minimum(literal_thresholds, adaptive_thresholds)
        literal_thresholds = torch.clamp(
            literal_thresholds, min=literal_threshold_floor_value
        )

        neg_thresholds = literal_thresholds.clone()
        if literal_negative_percentile_value is not None:
            neg_thresholds = torch.full_like(
                literal_thresholds, float(literal_threshold)
            )
            percent_neg = float(literal_negative_percentile_value)
            for b_idx in range(B):
                valid_mask = literal_valid_mask[b_idx, 0, 0]
                valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
                if valid_indices.numel() == 0:
                    continue
                neg_indices = valid_indices[valid_indices >= N_max]
                if neg_indices.numel() == 0:
                    continue
                for h_idx in range(H_max):
                    for t_idx in range(T_max):
                        clause_vec = lit_probs[b_idx, h_idx, t_idx, neg_indices]
                        if clause_vec.numel() == 0:
                            continue
                        quantile_neg = torch.quantile(clause_vec, percent_neg)
                        neg_thresholds[b_idx, h_idx, t_idx] = quantile_neg
            neg_thresholds = torch.minimum(neg_thresholds, literal_thresholds)
        neg_thresholds = torch.clamp(neg_thresholds, min=neg_threshold_floor_value)

        literal_dedup_removed_total = 0
        coverage_metric_map: Dict[Tuple[int, int, int], Dict[str, float]] = {}
        head_coverage_summary: Dict[Tuple[int, int], float] = {}

        merge_stats: Optional[Dict[str, Tensor]] = None
        merge_threshold_value = merge_threshold
        if merge_threshold_value is not None and merge_threshold_value <= 0.0:
            merge_threshold_value = None
        max_clauses_value = (
            max_clauses if max_clauses is not None and max_clauses > 0 else None
        )
        if merge_threshold_value is not None or max_clauses_value is not None:
            clause_active_pre = clause_active.clone()
            eps = 1e-6
            if merge_threshold_value is not None:
                for b_idx in range(B):
                    head_limit = (
                        int(H_len[b_idx].item()) if b_idx < len(H_len) else H_max
                    )
                    head_limit = min(head_limit, H_max)
                    for h_idx in range(head_limit):
                        active_indices = torch.nonzero(
                            clause_active[b_idx, h_idx], as_tuple=False
                        ).flatten()
                        if active_indices.numel() < 2:
                            continue
                        gate_scores = clause_prob[b_idx, h_idx, active_indices]
                        order = torch.argsort(gate_scores, descending=True)
                        kept: list[int] = []
                        for order_idx in order.tolist():
                            clause_idx = int(active_indices[order_idx])
                            if not bool(clause_active[b_idx, h_idx, clause_idx]):
                                continue
                            current_vec = lit_probs_masked[b_idx, h_idx, clause_idx]
                            keep_clause = True
                            for kept_idx in kept:
                                kept_vec = lit_probs_masked[b_idx, h_idx, kept_idx]
                                intersection = (current_vec * kept_vec).sum()
                                union = (
                                    (current_vec + kept_vec - current_vec * kept_vec)
                                    .sum()
                                    .clamp(min=eps)
                                )
                                jaccard = intersection / union
                                if float(jaccard.item()) >= merge_threshold_value:
                                    clause_active[b_idx, h_idx, clause_idx] = False
                                    keep_clause = False
                                    break
                            if keep_clause:
                                kept.append(clause_idx)
            if max_clauses_value is not None:
                for b_idx in range(B):
                    head_limit = (
                        int(H_len[b_idx].item()) if b_idx < len(H_len) else H_max
                    )
                    head_limit = min(head_limit, H_max)
                    for h_idx in range(head_limit):
                        active_indices = torch.nonzero(
                            clause_active[b_idx, h_idx], as_tuple=False
                        ).flatten()
                        if active_indices.numel() <= max_clauses_value:
                            continue
                        gate_scores = clause_prob[b_idx, h_idx, active_indices]
                        order = torch.argsort(gate_scores, descending=True)
                        drop = active_indices[order[max_clauses_value:]]
                        clause_active[b_idx, h_idx, drop] = False

            clauses_before = clause_active_pre.sum(dim=-1).to(torch.float32)
            clauses_after = clause_active.sum(dim=-1).to(torch.float32)
            clauses_removed = (
                (clause_active_pre & ~clause_active).sum(dim=-1).to(torch.float32)
            )
            merge_stats = {
                "clauses_before": clauses_before,
                "clauses_after": clauses_after,
                "clauses_removed": clauses_removed,
            }

        requested_topk = (
            min(max(int(literal_topk_value), 0), literal_dim)
            if literal_topk_value is not None
            else None
        )
        topk_capacity = getattr(self.clause_composer, "K_max", 0)
        topk_slots = max(
            0,
            min(
                topk_capacity,
                literal_dim,
                requested_topk if requested_topk is not None else topk_capacity,
            ),
        )

        choice = torch.full(
            (B, H_max, T_max, self.clause_composer.K_max),
            -1,
            dtype=torch.int64,
            device=device,
        )
        literal_active = torch.zeros_like(choice, dtype=torch.bool)

        if topk_slots > 0:
            clause_coverage: Optional[Tensor]
            if output.Clause_truth is not None and output.Clause_truth.numel() > 0:
                clause_coverage = output.Clause_truth.mean(dim=1).to(lit_probs.dtype)
            else:
                clause_coverage = None

            neg_limit_global = (
                topk_slots
                if literal_negative_limit is None
                else min(int(literal_negative_limit), topk_slots)
            )
            neg_keep_global = (
                int(literal_negative_keep_topk_value)
                if literal_negative_keep_topk_value is not None
                else 0
            )
            min_literal_guard_raw = (
                min(int(min_literals), topk_slots) if min_literals > 0 else 0
            )
            min_literal_guard = min_literal_guard_raw
            neg_boost_value = literal_negative_boost_value

            for b_idx in range(B):
                atom_count = int(N_len[b_idx].item())
                if atom_count <= 0:
                    continue
                head_limit = int(H_len[b_idx].item()) if b_idx < len(H_len) else H_max
                head_limit = min(head_limit, H_max)
                pos_cap = min(atom_count, N_max)
                neg_cap = min(atom_count, max(0, literal_dim - N_max))
                for h_idx in range(head_limit):
                    for t_idx in range(T_max):
                        if not bool(clause_active[b_idx, h_idx, t_idx]):
                            continue

                        clause_probs = lit_probs[b_idx, h_idx, t_idx]
                        pos_scores = clause_probs[:pos_cap]
                        neg_scores = (
                            clause_probs[N_max : N_max + neg_cap]
                            if neg_cap > 0
                            else clause_probs.new_empty(0)
                        )

                        if pos_scores.numel() == 0 and neg_scores.numel() == 0:
                            continue

                        pos_sorted_scores, pos_sorted_idx = torch.sort(
                            pos_scores, descending=True
                        )
                        neg_sorted_scores, neg_sorted_idx = torch.sort(
                            neg_scores, descending=True
                        )

                        pos_threshold_val = float(
                            literal_thresholds[b_idx, h_idx, t_idx].item()
                        )
                        neg_threshold_base = float(
                            neg_thresholds[b_idx, h_idx, t_idx].item()
                        )
                        neg_threshold_effective = max(
                            neg_threshold_base - neg_boost_value,
                            neg_threshold_floor_value,
                        )
                        coverage_val = (
                            float(clause_coverage[b_idx, h_idx, t_idx].item())
                            if clause_coverage is not None
                            else 0.0
                        )
                        gate_val = float(clause_prob[b_idx, h_idx, t_idx].item())
                        neg_rank_scale = 1.0 + coverage_val + gate_val
                        neg_rank_scale = max(neg_rank_scale, 1e-6)

                        neg_limit_effective = min(neg_limit_global, neg_cap)
                        if neg_limit_effective < 0:
                            neg_limit_effective = 0
                        neg_keep_effective = min(
                            neg_keep_global, neg_limit_effective, neg_cap
                        )

                        allow_positive_fallback = not bridge_disable_positive_fallback
                        neg_selected: list[tuple[float, int]] = []
                        neg_fallback: list[tuple[float, int]] = []
                        if neg_limit_effective > 0 and neg_sorted_scores.numel() > 0:
                            neg_rank_scores = (
                                neg_sorted_scores
                                * neg_rank_scale
                                * literal_negative_priority_value
                            )
                            if literal_negative_rank_boost > 0.0:
                                neg_rank_scores = neg_rank_scores + float(
                                    literal_negative_rank_boost
                                )
                            neg_rank_list = list(
                                zip(
                                    neg_rank_scores.tolist(),
                                    neg_sorted_scores.tolist(),
                                    neg_sorted_idx.tolist(),
                                )
                            )
                            # Already sorted by neg_sorted_scores descending; ensure rank ordering matches.
                            neg_rank_list.sort(key=lambda entry: entry[0], reverse=True)

                            for _, score_val, idx_local in neg_rank_list:
                                if len(neg_selected) >= neg_limit_effective:
                                    break
                                idx_offset = int(idx_local)
                                if idx_offset < 0 or idx_offset >= neg_cap:
                                    continue
                                raw_index = N_max + idx_offset
                                if raw_index >= clause_probs.shape[0]:
                                    continue
                                if score_val >= neg_threshold_effective:
                                    neg_selected.append((score_val, raw_index))
                                elif score_val >= neg_threshold_floor_value:
                                    neg_fallback.append((score_val, raw_index))

                            if (
                                neg_keep_effective > 0
                                and len(neg_selected) < neg_keep_effective
                            ):
                                needed = neg_keep_effective - len(neg_selected)
                                for fallback_entry in neg_fallback:
                                    if len(neg_selected) >= neg_limit_effective:
                                        break
                                    neg_selected.append(fallback_entry)
                                    needed -= 1
                                    if needed <= 0:
                                        break

                            if (
                                neg_limit_effective > 0
                                and len(neg_selected) > neg_limit_effective
                            ):
                                neg_selected = neg_selected[:neg_limit_effective]

                        available_slots = max(topk_slots - len(neg_selected), 0)
                        pos_candidates = list(
                            zip(
                                pos_sorted_scores.tolist(),
                                [int(idx) for idx in pos_sorted_idx.tolist()],
                            )
                        )
                        primary_pos: list[tuple[float, int]] = []
                        fallback_pos: list[tuple[float, int]] = []
                        for score_val, idx_local in pos_candidates:
                            if idx_local < 0 or idx_local >= pos_cap:
                                continue
                            if score_val >= pos_threshold_val:
                                primary_pos.append((score_val, idx_local))
                            else:
                                fallback_pos.append((score_val, idx_local))

                        pos_selected: list[tuple[float, int]] = []
                        if available_slots > 0:
                            for entry in primary_pos:
                                if len(pos_selected) >= available_slots:
                                    break
                                pos_selected.append(entry)
                        total_selected = len(neg_selected) + len(pos_selected)
                        need_min_guard = (
                            min_literal_guard > 0 and total_selected < min_literal_guard
                        )
                        need_any_literal = total_selected == 0
                        if (
                            allow_positive_fallback
                            and len(pos_selected) < available_slots
                            and fallback_pos
                            and (need_min_guard or need_any_literal)
                        ):
                            needed = available_slots - len(pos_selected)
                            pos_selected.extend(fallback_pos[:needed])

                        min_required = min_literal_guard
                        if min_required > 0 and allow_positive_fallback:
                            current_total = len(neg_selected) + len(pos_selected)
                            if current_total < min_required and available_slots > len(
                                pos_selected
                            ):
                                additional_needed = min_required - current_total
                                used_pos_indices = {idx for _, idx in pos_selected}
                                remaining_candidates = (
                                    primary_pos[len(pos_selected) :] + fallback_pos
                                )
                                for candidate in remaining_candidates:
                                    if len(pos_selected) >= available_slots:
                                        break
                                    score_val, idx_local = candidate
                                    if idx_local in used_pos_indices:
                                        continue
                                    pos_selected.append(candidate)
                                    used_pos_indices.add(idx_local)
                                    additional_needed -= 1
                                    if additional_needed <= 0:
                                        break

                        selected_literals = neg_selected + [
                            (score, idx) for score, idx in pos_selected
                        ]
                        selected_literals.sort(key=lambda entry: entry[0], reverse=True)
                        selected_literals = selected_literals[:topk_slots]

                        for slot_idx in range(topk_slots):
                            if slot_idx < len(selected_literals):
                                score_val, raw_index = selected_literals[slot_idx]
                                choice[b_idx, h_idx, t_idx, slot_idx] = int(raw_index)
                                literal_active[b_idx, h_idx, t_idx, slot_idx] = True
                            else:
                                choice[b_idx, h_idx, t_idx, slot_idx] = -1
                                literal_active[b_idx, h_idx, t_idx, slot_idx] = False

        if self.clause_composer.K_max > 0:
            N_len = N_len.to(device)
            two_n = (2 * N_len).view(B, 1, 1, 1)
            N_max = literal_dim // 2
            choice_raw = choice

            invalid = choice_raw >= two_n
            choice_raw = torch.where(
                invalid, torch.full_like(choice_raw, -1), choice_raw
            )

            is_positive = (choice_raw >= 0) & (choice_raw < N_max)
            positive_invalid = is_positive & (choice_raw >= N_len.view(B, 1, 1, 1))
            choice = torch.where(
                positive_invalid, torch.full_like(choice_raw, -1), choice_raw
            )

            is_negative = (choice_raw >= N_max) & (choice_raw < two_n)
            neg_offset = N_len.view(B, 1, 1, 1)
            choice = torch.where(
                is_negative,
                neg_offset + (choice_raw - N_max),
                choice,
            )

        if (
            literal_dedup_threshold_value is not None
            and literal_dedup_threshold_value > 0.0
        ):
            dedup_threshold = float(literal_dedup_threshold_value)
            eps = 1e-6
            for b_idx in range(B):
                head_limit = int(H_len[b_idx].item()) if b_idx < len(H_len) else H_max
                head_limit = min(head_limit, H_max)
                for h_idx in range(head_limit):
                    active_indices = torch.nonzero(
                        clause_active[b_idx, h_idx], as_tuple=False
                    ).flatten()
                    if active_indices.numel() <= 1:
                        continue
                    gate_scores = clause_prob[b_idx, h_idx, active_indices]
                    order = torch.argsort(gate_scores, descending=True)
                    kept_signatures: list[Set[int]] = []
                    for ord_idx in order.tolist():
                        clause_idx = int(active_indices[ord_idx])
                        if not bool(clause_active[b_idx, h_idx, clause_idx]):
                            continue
                        slot_indices = choice[b_idx, h_idx, clause_idx]
                        slot_mask = literal_active[b_idx, h_idx, clause_idx]
                        selected_ids: list[int] = []
                        for lit_idx, is_active in zip(
                            slot_indices.tolist(), slot_mask.tolist()
                        ):
                            if not is_active:
                                continue
                            literal_id = int(lit_idx)
                            if literal_id >= 0:
                                selected_ids.append(literal_id)
                        if not selected_ids:
                            continue
                        literal_set = set(selected_ids)
                        drop_clause = False
                        for kept_set in kept_signatures:
                            union = len(literal_set | kept_set)
                            if union == 0:
                                continue
                            intersection = len(literal_set & kept_set)
                            jaccard = intersection / max(union, eps)
                            if jaccard >= dedup_threshold:
                                drop_clause = True
                                break
                        if drop_clause:
                            clause_active[b_idx, h_idx, clause_idx] = False
                            literal_active[b_idx, h_idx, clause_idx] = False
                            choice[b_idx, h_idx, clause_idx] = -1
                            literal_dedup_removed_total += 1
                        else:
                            kept_signatures.append(literal_set)

        literal_active = literal_active & (choice >= 0)

        literal_active = literal_active & head_mask.unsqueeze(-1).unsqueeze(-1)

        rule_clause_active = clause_active & head_mask.unsqueeze(-1)

        literal_counts = literal_active.sum(dim=-1)
        active_clause_mask = rule_clause_active

        export_stats: Optional[Dict[str, float]] = None

        result: Dict[str, Tensor | object] = {
            "Rule_lit_index": choice.to(torch.int32),
            "Rule_clause_active": rule_clause_active,
            "Rule_lit_active": literal_active,
        }

        if merge_stats is not None:
            result["merge_stats"] = merge_stats

        if return_diagnostics:
            threshold_mask = active_clause_mask
            thresholds_active = literal_thresholds[threshold_mask]
            literal_counts_active = literal_counts[threshold_mask]
            literal_counts_total = float(literal_counts_active.sum().item())
            active_clause_count = int(threshold_mask.sum().item())

            neg_mask = choice >= 0
            neg_mask = neg_mask & literal_active
            neg_mask = neg_mask & (choice >= N_len.view(B, 1, 1, 1))
            negative_literal_total = int(neg_mask.sum().item())
            literal_active_total = int(literal_active.sum().item())

            threshold_sum = (
                float(thresholds_active.sum().item())
                if thresholds_active.numel() > 0
                else 0.0
            )
            negative_fraction = (
                float(negative_literal_total) / float(max(literal_active_total, 1))
                if literal_active_total > 0
                else 0.0
            )
            export_stats = {
                "active_clause_count": float(active_clause_count),
                "literal_threshold_sum": threshold_sum,
                "literal_threshold_count": float(thresholds_active.numel()),
                "literal_threshold_min": (
                    float(thresholds_active.min().item())
                    if thresholds_active.numel() > 0
                    else None
                ),
                "literal_threshold_max": (
                    float(thresholds_active.max().item())
                    if thresholds_active.numel() > 0
                    else None
                ),
                "literal_count_total": float(literal_active_total),
                "literal_per_clause_sum": float(literal_counts_total),
                "literal_per_clause_count": float(literal_counts_active.numel()),
                "literal_per_clause_min": (
                    float(literal_counts_active.min().item())
                    if literal_counts_active.numel() > 0
                    else None
                ),
                "literal_per_clause_max": (
                    float(literal_counts_active.max().item())
                    if literal_counts_active.numel() > 0
                    else None
                ),
                "negative_literal_count_total": float(negative_literal_total),
                "negative_literal_fraction": negative_fraction,
                "literal_threshold_floor": float(literal_threshold_floor_value),
                "literal_threshold_base": float(literal_threshold),
                "literal_negative_boost": float(literal_negative_boost),
                "literal_negative_limit": (
                    float(literal_negative_limit)
                    if literal_negative_limit is not None
                    else None
                ),
                "literal_negative_percentile": (
                    float(literal_negative_percentile_value)
                    if literal_negative_percentile_value is not None
                    else None
                ),
                "literal_negative_threshold_floor": float(neg_threshold_floor_value),
                "literal_negative_keep_topk": (
                    float(literal_negative_keep_topk_value)
                    if literal_negative_keep_topk_value is not None
                    else None
                ),
                "literal_negative_rank_boost": float(literal_negative_rank_boost),
                "literal_dedup_threshold": (
                    float(literal_dedup_threshold_value)
                    if literal_dedup_threshold_value is not None
                    else None
                ),
                "literal_dedup_removed_total": float(literal_dedup_removed_total),
                "export_mode": export_mode_value,
                "calibration_applied": float(1.0 if calibration else 0.0),
            }
            if coverage_metric_map:
                gain_values = [entry["gain"] for entry in coverage_metric_map.values()]
                gain_norm_values = [
                    entry["gain_norm"] for entry in coverage_metric_map.values()
                ]
                export_stats["coverage_gain_sum"] = float(sum(gain_values))
                export_stats["coverage_gain_mean"] = float(sum(gain_values)) / max(
                    len(gain_values), 1
                )
                export_stats["coverage_gain_norm_mean"] = float(
                    sum(gain_norm_values)
                ) / max(len(gain_norm_values), 1)
            if head_coverage_summary:
                head_values = list(head_coverage_summary.values())
                export_stats["coverage_head_mean"] = float(sum(head_values)) / max(
                    len(head_values), 1
                )
                export_stats["coverage_head_count"] = float(len(head_values))
            debug_entries: list[dict[str, object]] = []
            active_positions = torch.nonzero(active_clause_mask, as_tuple=False)
            for pos in active_positions:
                b_idx, h_idx, t_idx = pos.tolist()
                n_atoms = int(N_len[b_idx].item())
                pos_probs = lit_probs[b_idx, h_idx, t_idx, :n_atoms]
                neg_probs = lit_probs[b_idx, h_idx, t_idx, N_max : N_max + n_atoms]
                top_limit = min(5, n_atoms) if n_atoms > 0 else 0
                pos_top: list[tuple[int, float]] = []
                neg_top: list[tuple[int, float]] = []
                if top_limit > 0:
                    pos_vals, pos_idx = torch.topk(pos_probs, k=top_limit)
                    neg_vals, neg_idx = torch.topk(neg_probs, k=top_limit)
                    pos_top = [
                        (int(idx.item()), float(val.item()))
                        for idx, val in zip(pos_idx.cpu(), pos_vals.cpu())
                    ]
                    neg_top = [
                        (int(idx.item()), float(val.item()))
                        for idx, val in zip(neg_idx.cpu(), neg_vals.cpu())
                    ]
                metric_entry = coverage_metric_map.get(
                    (int(b_idx), int(h_idx), int(t_idx))
                )
                coverage_info: Dict[str, float] = {}
                if metric_entry is not None:
                    coverage_info = {
                        "coverage_gain": metric_entry["gain"],
                        "coverage_gain_norm": metric_entry["gain_norm"],
                        "coverage_rank": metric_entry["rank"],
                        "coverage_total": metric_entry["coverage_total"],
                        "coverage_gate": metric_entry["gate"],
                    }
                debug_entries.append(
                    {
                        "batch": int(b_idx),
                        "head": int(h_idx),
                        "clause": int(t_idx),
                        "threshold": float(
                            literal_thresholds[b_idx, h_idx, t_idx].item()
                        ),
                        "neg_threshold": float(
                            neg_thresholds[b_idx, h_idx, t_idx].item()
                        ),
                        "pos_top": pos_top,
                        "neg_top": neg_top,
                        **coverage_info,
                    }
                )
            export_stats["clause_debug"] = debug_entries

        if export_stats is not None:
            result["export_stats"] = export_stats

        return result
