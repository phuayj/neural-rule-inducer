"""Counterfactual necessity losses for clause-based rule induction.

This module provides utilities for recomputing clause truth under literal flips,
estimating clause responsibility, and computing counterfactual necessity losses
with overlap/load-balance diagnostics.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
from torch import Tensor

__all__ = [
    "recompute_clause_truth",
    "compute_clause_responsibility",
    "coverage_overlap_loss",
    "clause_load_balance_loss",
    "compute_responsibility_weighted_cf_loss",
]

logger = logging.getLogger(__name__)


def recompute_clause_truth(
    lit_probs: Tensor,  # [B, H, T, 2N]
    literal_truth: Tensor,  # [B, M, 2N]
) -> Tensor:
    """
    Recompute clause truth from literal probabilities and literal truth values.

    clause_truth[b,h,t,m] = prod_l (1 - p_l + p_l * truth[b,m,l])

    Args:
        lit_probs: Literal selection probabilities [B, H, T, 2N].
        literal_truth: Literal truth values [B, M, 2N].

    Returns:
        Clause truth tensor with shape [B, H, T, M].
    """
    # lit_probs: [B, H, T, 2N]
    # literal_truth: [B, M, 2N]
    # Need to compute for all (clause, example) pairs

    # Expand for broadcasting:
    # lit_probs_exp: [B, H, T, 1, 2N]
    # truth_exp: [B, 1, 1, M, 2N]
    lit_probs_exp = lit_probs.unsqueeze(3)  # [B, H, T, 1, 2N]
    truth_exp = literal_truth.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, M, 2N]

    # literal_component = 1 - p + p * truth
    # When p=0: component=1 (literal doesn't matter)
    # When p=1, truth=1: component=1 (literal satisfied)
    # When p=1, truth=0: component=0 (literal violated)
    literal_component = 1.0 - lit_probs_exp + lit_probs_exp * truth_exp
    literal_component = literal_component.clamp(min=1e-6, max=1.0)

    # Product t-norm over literals
    clause_truth = literal_component.prod(dim=-1)  # [B, H, T, M]

    return clause_truth


def compute_clause_responsibility(
    clause_truth: Tensor,  # [B, H, T, M]
    *,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    """
    Compute per-clause responsibility weights for each example.

    Responsibility measures which clause "owns" each positive prediction:
    - w_t(x) = c_t(x) * ∏ₖ≠ₜ(1 - cₖ(x))  # active AND not redundant
    - r_t(x) = w_t(x) / (Σⱼ wⱼ(x) + ε)    # normalized

    Args:
        clause_truth: Per-clause truth values [B, H, T, M]
        eps: Small constant for numerical stability

    Returns:
        responsibility: Normalized responsibility weights [B, H, T, M]
        raw_weights: Unnormalized weights [B, H, T, M]
    """
    # Compute w_t(x) = c_t(x) * ∏ₖ≠ₜ(1 - cₖ(x))
    # This is: "clause t is active" AND "no other clause would have covered this"

    # Product of (1 - c_k) over all clauses
    # Using log-sum-exp trick for numerical stability
    log_complement = torch.log((1.0 - clause_truth).clamp(min=eps))  # [B, H, T, M]
    total_log_complement = log_complement.sum(dim=2, keepdim=True)  # [B, H, 1, M]

    # For clause t, we want ∏ₖ≠ₜ(1 - cₖ) = total_product / (1 - cₜ)
    # In log space: total_log - log(1 - cₜ)
    others_log_complement = total_log_complement - log_complement  # [B, H, T, M]
    others_complement = torch.exp(others_log_complement.clamp(max=20.0))  # [B, H, T, M]

    # w_t = c_t * ∏ₖ≠ₜ(1 - cₖ)
    raw_weights = clause_truth * others_complement  # [B, H, T, M]

    # Normalize across clauses
    weight_sum = raw_weights.sum(dim=2, keepdim=True) + eps  # [B, H, 1, M]
    responsibility = raw_weights / weight_sum  # [B, H, T, M]

    return responsibility, raw_weights


def coverage_overlap_loss(
    clause_truth: Tensor,  # [B, H, T, M]
    positive_mask: Tensor,  # [B, M, H] - which examples are positive
    *,
    eps: float = 1e-8,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Compute coverage overlap penalty: penalize multiple clauses covering same positive.

    L_overlap = E[Σₜ<ᵤ c_t(x) * c_u(x)] for positive examples

    Args:
        clause_truth: Per-clause truth values [B, H, T, M]
        positive_mask: Boolean mask for positive examples [B, M, H]
        eps: Numerical stability constant

    Returns:
        loss: Scalar overlap penalty
        metrics: Dict with diagnostic metrics
    """
    _, _, num_clauses, _ = clause_truth.shape

    # Permute positive_mask to [B, H, M]
    pos_mask = positive_mask.permute(0, 2, 1).float()  # [B, H, M]

    # Apply positive mask to clause truth
    clause_truth_pos = clause_truth * pos_mask.unsqueeze(2)  # [B, H, T, M]

    # Compute pairwise overlap: for each pair (t, u) where t < u
    # overlap[t,u] = sum_m c_t(m) * c_u(m) for positive m
    # This is clause_truth_pos @ clause_truth_pos.T but only upper triangle

    # Compute Gram matrix: [B, H, T, T]
    gram = torch.einsum("bhtm,bhum->bhtu", clause_truth_pos, clause_truth_pos)

    # Only count upper triangle (t < u) to avoid double counting
    triu_mask = torch.triu(
        torch.ones(num_clauses, num_clauses, device=gram.device, dtype=torch.bool),
        diagonal=1,
    )
    triu_mask = triu_mask.view(1, 1, num_clauses, num_clauses)

    overlap_sum = (gram * triu_mask.float()).sum()
    num_positives = pos_mask.sum().clamp(min=1.0)
    num_pairs = num_clauses * (num_clauses - 1) / 2

    loss = overlap_sum / (num_positives * num_pairs + eps)

    metrics = {
        "overlap_raw": overlap_sum.detach(),
        "overlap_per_positive": (overlap_sum / num_positives).detach(),
        "num_positives": num_positives.detach(),
    }

    return loss, metrics


def clause_load_balance_loss(
    responsibility: Tensor,  # [B, H, T, M]
    positive_mask: Tensor,  # [B, M, H]
    *,
    eps: float = 1e-8,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Compute entropy-based load balancing loss.

    Directly maximizes per-example responsibility entropy to encourage
    each example to be covered by multiple clauses, not just one dominant clause.

    Loss = -mean(entropy) so minimizing this maximizes entropy.

    Args:
        responsibility: Normalized responsibility weights [B, H, T, M]
        positive_mask: Boolean mask for positive examples [B, M, H]
        eps: Numerical stability constant

    Returns:
        loss: Negative mean entropy (minimize to maximize entropy)
        metrics: Dict with diagnostic metrics
    """
    _, _, num_clauses, _ = responsibility.shape

    # Permute positive_mask to [B, H, M]
    pos_mask = positive_mask.permute(0, 2, 1).float()  # [B, H, M]
    num_positives = pos_mask.sum().clamp(min=1.0)

    # Per-example entropy over clauses: H(r) = -sum_t r_t * log(r_t)
    # responsibility: [B, H, T, M], sum over T (clauses)
    entropy_per_example = -(responsibility * torch.log(responsibility + eps)).sum(
        dim=2
    )  # [B, H, M]

    # Apply positive mask and compute mean
    masked_entropy = entropy_per_example * pos_mask  # [B, H, M]
    mean_entropy = masked_entropy.sum() / num_positives

    # Loss is NEGATIVE entropy (minimize loss = maximize entropy)
    loss = -mean_entropy

    # Keep diagnostic metrics
    masked_resp = responsibility * pos_mask.unsqueeze(2)  # [B, H, T, M]
    num_pos_per_head = pos_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, H, 1]
    mean_resp = masked_resp.sum(dim=-1) / num_pos_per_head  # [B, H, T]
    cv = mean_resp.std(dim=-1) / (mean_resp.mean(dim=-1) + eps)

    # Max entropy for reference (log of num_clauses)
    max_entropy = torch.log(
        torch.tensor(
            num_clauses, device=responsibility.device, dtype=responsibility.dtype
        )
    )

    metrics = {
        "mean_entropy": mean_entropy.detach(),
        "max_entropy": max_entropy.detach(),
        "entropy_ratio": (mean_entropy / max_entropy).detach(),
        "responsibility_cv": cv.mean().detach(),
        "responsibility_max": mean_resp.max(dim=-1).values.mean().detach(),
        "responsibility_min": mean_resp.min(dim=-1).values.mean().detach(),
    }

    return loss, metrics


def compute_responsibility_weighted_cf_loss(
    X_val: Tensor,  # [B, M, N] original features
    X_mask: Tensor,  # [B, M, N] validity mask
    Y_val: Tensor,  # [B, M, H] ground truth
    Lit_probs: Tensor,  # [B, H, T, 2N] literal selection probs
    clause_truth: Tensor,  # [B, H, T, M] clause truth values (from model)
    *,
    necessity_weight: float = 1.0,
    spuriousness_weight: float = 1.0,
    overlap_weight: float = 0.1,
    balance_weight: float = 0.01,
    select_threshold: float = 0.3,
    num_cf_samples: int = 4,
    s_norm_temperature: float = 0.1,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Compute responsibility-weighted counterfactual necessity loss.

    Key difference from original: CF necessity is applied per-clause weighted
    by that clause's responsibility for each example, rather than to global pred.

    This fixes the credit assignment collapse where one dominant clause
    blocks gradient flow to other clauses.

    Args:
        X_val: Input features [B, M, N]
        X_mask: Feature validity mask [B, M, N]
        Y_val: Ground truth labels [B, M, H]
        Lit_probs: Literal selection probabilities [B, H, T, 2N]
        clause_truth: Pre-computed clause truth values [B, H, T, M]
        necessity_weight: Weight for necessity loss term
        spuriousness_weight: Weight for spuriousness loss term
        overlap_weight: Weight for coverage overlap penalty
        balance_weight: Weight for load balancing loss
        select_threshold: Threshold for considering a literal "selected"
        num_cf_samples: Number of literals to sample for counterfactuals
        s_norm_temperature: Temperature for s-norm (clause aggregation)

    Returns:
        loss: Combined loss tensor
        metrics: Dict with diagnostic metrics
    """
    device = X_val.device
    dtype = X_val.dtype
    B, M, N = X_val.shape
    _, _, num_clauses, _ = Lit_probs.shape  # L = 2N

    # Aggregate clause truth to prediction using s-norm for positive mask
    R_pred = torch.logsumexp(clause_truth / s_norm_temperature, dim=2)
    R_pred = s_norm_temperature * R_pred  # [B, H, M]
    R_pred = R_pred.permute(0, 2, 1).clamp(0, 1)  # [B, M, H]

    # Identify positive examples: Y > 0.5 and prediction > 0.5
    Y_binary = (Y_val > 0.5).float()  # [B, M, H]
    pred_binary = (R_pred > 0.5).float()  # [B, M, H]
    positive_mask = (Y_binary * pred_binary).bool()  # [B, M, H]

    num_positives = positive_mask.float().sum()
    if num_positives.item() == 0:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        return zero, {
            "necessity_loss": zero,
            "spuriousness_loss": zero,
            "overlap_loss": zero,
            "balance_loss": zero,
            "num_positive_examples": torch.tensor(0, device=device),
            "cf_selected_fraction": zero,
            "cf_selected_prob_mean": zero,
            "cf_ignored_prob_mean": zero,
            "cf_max_lit_prob_mean": zero,
            "responsibility_entropy": zero,
            "responsibility_max": zero,
        }

    literal_known_mask = torch.cat([X_mask, X_mask], dim=-1).bool()
    literal_unknown_fill = X_val.new_full(literal_known_mask.shape, 0.5)
    literal_truth = torch.cat([X_val, 1.0 - X_val], dim=-1)

    def build_counterfactual_literal_truth(feature_idx: Tensor) -> Tensor:
        # Multiple heads may target the same feature; parity captures net flips.
        flip_counts = torch.zeros(B, N, device=device, dtype=torch.int64)
        flip_counts.scatter_add_(
            1, feature_idx, torch.ones_like(feature_idx, dtype=torch.int64)
        )
        flip_mask = flip_counts.remainder(2).to(dtype)
        flip_mask = torch.cat([flip_mask, flip_mask], dim=-1).unsqueeze(1)
        literal_truth_cf = literal_truth + flip_mask * (1.0 - 2.0 * literal_truth)
        return torch.where(literal_known_mask, literal_truth_cf, literal_unknown_fill)

    # Get max literal selection prob per clause: [B, H, T]
    max_lit_prob = Lit_probs.max(dim=-1).values  # [B, H, T]
    max_lit_prob_mean = max_lit_prob.mean()

    # For necessity: sample selected literals (high prob)
    # For spuriousness: sample ignored literals (low prob)

    # Average Lit_probs across clauses for sampling: [B, H, 2N]
    lit_probs_avg_detached = Lit_probs.mean(dim=2).detach()  # Average over clauses

    selected_mask = lit_probs_avg_detached >= select_threshold
    ignored_mask = ~selected_mask
    selected_count = selected_mask.float().sum().clamp(min=1.0)
    ignored_count = ignored_mask.float().sum().clamp(min=1.0)
    selected_prob_mean = (
        lit_probs_avg_detached * selected_mask.float()
    ).sum() / selected_count
    ignored_prob_mean = (
        lit_probs_avg_detached * ignored_mask.float()
    ).sum() / ignored_count
    selected_fraction = selected_mask.float().mean()

    weight_floor = lit_probs_avg_detached.new_full(lit_probs_avg_detached.shape, 1e-6)

    # Sample indices for selected literals (weighted by prob)
    # Use Gumbel-top-k for differentiable sampling
    K = min(num_cf_samples, N)

    # --- Necessity loss: flip selected literals ---
    # Sample K literals with high selection probability
    sample_weights_sel = torch.where(
        selected_mask, lit_probs_avg_detached, weight_floor
    )
    gumbel_noise = -torch.empty_like(sample_weights_sel).exponential_().log()
    perturbed_sel = (
        sample_weights_sel.clamp(min=1e-6).log().clamp(min=-20) + gumbel_noise
    )
    _, selected_indices = perturbed_sel.topk(K, dim=-1)  # [B, H, K]

    # --- Spuriousness loss: flip non-selected literals ---
    sample_weights_ign = torch.where(
        ignored_mask, 1.0 - lit_probs_avg_detached, weight_floor
    )
    gumbel_noise_ign = -torch.empty_like(sample_weights_ign).exponential_().log()
    perturbed_ign = (
        sample_weights_ign.clamp(min=1e-6).log().clamp(min=-20) + gumbel_noise_ign
    )
    _, ignored_indices = perturbed_ign.topk(K, dim=-1)  # [B, H, K]

    # Responsibility weights per clause/example
    responsibility, raw_weights = compute_clause_responsibility(clause_truth, eps=1e-8)
    pos_mask = positive_mask.permute(0, 2, 1).float()  # [B, H, M]
    responsibility_pos = responsibility * pos_mask.unsqueeze(2)
    num_positives_clamped = num_positives.clamp(min=1.0)
    positive_mask_float = positive_mask.float()

    responsibility_entropy = (
        -(responsibility * torch.log(responsibility + 1e-8)).sum(dim=2).mean()
    )
    responsibility_max = responsibility.max(dim=2).values.mean()

    # Compute counterfactual clause truth for selected literals
    necessity_loss = torch.zeros((), device=device, dtype=dtype)
    responsibility_cf_mean = torch.zeros((), device=device, dtype=dtype)
    clause_truth_cf_mean = torch.zeros((), device=device, dtype=dtype)
    for k in range(K):
        # Get index of literal to flip: [B, H]
        flip_idx = selected_indices[:, :, k]  # [B, H]

        # Map to feature index (first N are positive, next N are negative)
        feature_idx = flip_idx % N  # [B, H]

        # Vectorized feature flips across heads (parity handles duplicates).
        literal_truth_cf = build_counterfactual_literal_truth(feature_idx)
        clause_truth_cf = recompute_clause_truth(Lit_probs, literal_truth_cf)

        # Responsibility-weighted clause truth on positives
        weighted_cf = (responsibility_pos * clause_truth_cf).sum()
        necessity_loss += weighted_cf
        responsibility_cf_mean += weighted_cf / num_positives_clamped
        clause_truth_cf_mean += (clause_truth_cf * pos_mask.unsqueeze(2)).sum() / (
            num_positives_clamped * num_clauses
        )

    necessity_loss = necessity_loss / (num_positives_clamped * K)
    responsibility_cf_mean = responsibility_cf_mean / K
    clause_truth_cf_mean = clause_truth_cf_mean / K

    # Compute counterfactual predictions for ignored literals
    spuriousness_loss = torch.zeros((), device=device, dtype=dtype)
    spurious_diff_mean = torch.zeros((), device=device, dtype=dtype)
    spurious_diff_pos_mean = torch.zeros((), device=device, dtype=dtype)
    for k in range(K):
        flip_idx = ignored_indices[:, :, k]  # [B, H]
        feature_idx = flip_idx % N

        # Vectorized feature flips across heads (parity handles duplicates).
        literal_truth_cf = build_counterfactual_literal_truth(feature_idx)
        clause_truth_cf = recompute_clause_truth(Lit_probs, literal_truth_cf)

        R_pred_cf = torch.logsumexp(clause_truth_cf / s_norm_temperature, dim=2)
        R_pred_cf = s_norm_temperature * R_pred_cf
        R_pred_cf = R_pred_cf.permute(0, 2, 1).clamp(0, 1)

        # For ignored literals, prediction should NOT change
        diff = (R_pred_cf - R_pred).abs()
        diff_pos_sum = (diff * positive_mask_float).sum()
        spuriousness_loss += diff_pos_sum
        spurious_diff_mean += diff.mean()
        spurious_diff_pos_mean += diff_pos_sum / num_positives_clamped

    spuriousness_loss = spuriousness_loss / (num_positives_clamped * K)
    spurious_diff_mean = spurious_diff_mean / K
    spurious_diff_pos_mean = spurious_diff_pos_mean / K

    # Overlap and load-balance penalties
    overlap_loss, overlap_metrics = coverage_overlap_loss(
        clause_truth=clause_truth,
        positive_mask=positive_mask,
    )
    balance_loss, balance_metrics = clause_load_balance_loss(
        responsibility=responsibility,
        positive_mask=positive_mask,
    )

    # Total loss
    total_loss = (
        necessity_weight * necessity_loss
        + spuriousness_weight * spuriousness_loss
        + overlap_weight * overlap_loss
        + balance_weight * balance_loss
    )

    metrics = {
        "necessity_loss": necessity_loss.detach(),
        "spuriousness_loss": spuriousness_loss.detach(),
        "overlap_loss": overlap_loss.detach(),
        "balance_loss": balance_loss.detach(),
        "mean_cf_responsibility_pred": responsibility_cf_mean.detach(),
        "mean_cf_clause_truth_selected": clause_truth_cf_mean.detach(),
        "mean_cf_pred_ignored": spurious_diff_mean.detach(),
        "mean_cf_pred_ignored_pos": spurious_diff_pos_mean.detach(),
        "num_positive_examples": num_positives.detach(),
        "cf_selected_fraction": selected_fraction.detach(),
        "cf_selected_prob_mean": selected_prob_mean.detach(),
        "cf_ignored_prob_mean": ignored_prob_mean.detach(),
        "cf_max_lit_prob_mean": max_lit_prob_mean.detach(),
        "responsibility_entropy": responsibility_entropy.detach(),
        "responsibility_max": responsibility_max.detach(),
        "responsibility_mean": responsibility.mean().detach(),
        "responsibility_sum_mean": responsibility.sum(dim=2).mean().detach(),
        "raw_responsibility_weight_mean": raw_weights.mean().detach(),
    }
    for metric_key, metric_value in overlap_metrics.items():
        metrics[f"overlap_{metric_key}"] = metric_value
    for metric_key, metric_value in balance_metrics.items():
        metrics[f"balance_{metric_key}"] = metric_value

    if logger.isEnabledFor(logging.DEBUG):
        positive_pred_mean = (
            (R_pred * positive_mask_float).sum() / num_positives_clamped
        ).detach()
        logger.debug(
            "Responsibility-weighted CF stats: positives=%d mean_pred=%.4f mean_pos_pred=%.4f "
            "necessity_loss=%.4f spuriousness_loss=%.4f overlap=%.4f balance=%.4f "
            "mean_cf_sel=%.4f mean_cf_ign=%.4f resp_entropy=%.4f resp_max=%.4f "
            "selected_frac=%.4f selected_prob=%.4f ignored_prob=%.4f max_lit_prob=%.4f",
            int(num_positives.item()),
            float(R_pred.mean().detach()),
            float(positive_pred_mean),
            float(necessity_loss.detach()),
            float(spuriousness_loss.detach()),
            float(overlap_loss.detach()),
            float(balance_loss.detach()),
            float(responsibility_cf_mean.detach()),
            float(spurious_diff_mean.detach()),
            float(responsibility_entropy.detach()),
            float(responsibility_max.detach()),
            float(selected_fraction.detach()),
            float(selected_prob_mean.detach()),
            float(ignored_prob_mean.detach()),
            float(max_lit_prob_mean.detach()),
        )

    return total_loss, metrics
