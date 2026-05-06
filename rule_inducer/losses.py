from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F


def slot_load_balance_loss(
    transport: Tensor,  # [T, K] or [B, H, T, K] transport assignment matrix
    gate_logits: Tensor,  # [B, H, T] or [T] clause gate logits
    eps: float = 1e-6,
) -> Tensor:
    """Switch-Transformer style load-balancing loss to encourage uniform slot usage.

    Computes: T * sum_t(u_t * f_t) where:
    - u_t = mean routing probability to slot t (from softmax of gate logits)
    - f_t = fraction of assignments to slot t (from transport)

    This loss is minimized when slots are equally used.
    """

    if gate_logits.ndim == 1:
        routing_probs = torch.softmax(gate_logits, dim=0)  # [T]
    else:
        routing_probs = torch.softmax(gate_logits, dim=-1).mean(dim=(0, 1))  # [T]

    if transport.ndim == 2:
        slot_usage = transport.sum(dim=1)  # [T]
        slot_usage = slot_usage / (slot_usage.sum() + eps)
    else:
        slot_usage = transport.sum(dim=-1).mean(dim=(0, 1))  # [T]
        slot_usage = slot_usage / (slot_usage.sum() + eps)

    T = routing_probs.size(0)
    loss = T * (routing_probs * slot_usage).sum()

    return loss


def slot_load_balance_cv2_loss(
    gate_logits: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """MoE-style load-balancing loss using CV² on per-example normalized slot usage.

    This produces much larger gradients than KL divergence when slots drift,
    because CV² is quadratic in the deviation from uniformity.

    Args:
        gate_logits: [..., T] gate logits (any leading dims, last dim = slots)
        eps: small constant for numerical stability

    Returns:
        Scalar loss value

    Reference: Switch Transformer (Fedus et al., 2021)
    """
    # Gate probabilities
    g = torch.sigmoid(gate_logits.float())  # [..., T]

    # Normalize per-example to get mixture weights (competition)
    w = g / (g.sum(dim=-1, keepdim=True) + eps)  # [..., T]

    # Compute "importance" per slot: average weight across all examples
    # Flatten all leading dims
    T = w.shape[-1]
    w_flat = w.reshape(-1, T)  # [N, T] where N = product of leading dims
    importance = w_flat.mean(dim=0)  # [T]

    # CV² loss: T * sum((I_t / mean(I) - 1)²)
    # This penalizes deviation from uniform importance
    mean_importance = importance.mean()
    cv2 = T * ((importance / (mean_importance + eps) - 1) ** 2).sum()

    return cv2


def slot_activation_balance_loss(
    gate_logits: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Activation balance loss to prevent slots from going permanently dead.

    Unlike the load-balance loss which operates on normalized weights,
    this directly penalizes variance in raw activation probabilities.

    Args:
        gate_logits: [..., T] gate logits
        eps: small constant for numerical stability

    Returns:
        Scalar loss value
    """
    g = torch.sigmoid(gate_logits.float())  # [..., T]

    # Average activation per slot across all examples
    T = g.shape[-1]
    g_flat = g.reshape(-1, T)  # [N, T]
    activation = g_flat.mean(dim=0)  # [T]

    # CV² on activations
    mean_activation = activation.mean()
    cv2 = T * ((activation / (mean_activation + eps) - 1) ** 2).sum()

    return cv2


def coverage_loss(
    predictions: Tensor,
    targets: Tensor,
    target_mask: Tensor,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
    margin_pos: Optional[float] = None,
    margin_neg: Optional[float] = None,
    example_weights: Optional[Tensor] = None,
) -> Tensor:
    """
    Computes masked coverage loss for rule predictions.

    Args:
        predictions: R_pred tensor [B, M_max, H_max].
        targets: Y_val tensor [B, M_max, H_max].
        target_mask: Y_mask tensor [B, M_max, H_max].
    """

    target_mask = target_mask.to(predictions.dtype)
    targets = targets.clamp(0.0, 1.0)
    preds = predictions.clamp(1e-6, 1 - 1e-6)
    logit_preds = torch.logit(preds, eps=1e-6)

    weights = torch.where(targets > 0.5, pos_weight, neg_weight)
    if example_weights is not None:
        weights = weights * example_weights.to(predictions.dtype)
    losses = F.binary_cross_entropy_with_logits(logit_preds, targets, reduction="none")
    coverage_mask = target_mask * weights
    coverage = (losses * coverage_mask).sum()
    normaliser = target_mask.sum().clamp(min=1.0)
    if example_weights is not None:
        normaliser = coverage_mask.sum().clamp(min=1.0)
    coverage = coverage / normaliser

    if margin_pos is not None or margin_neg is not None:
        if margin_pos is not None:
            pos_margin_violation = torch.relu(margin_pos - preds)
            pos_penalty = (pos_margin_violation * targets * target_mask).sum()
            pos_den = (targets * target_mask).sum().clamp(min=1.0)
            coverage = coverage + pos_penalty / pos_den

        if margin_neg is not None:
            neg_margin_violation = torch.relu(preds - margin_neg)
            neg_penalty = (neg_margin_violation * (1.0 - targets) * target_mask).sum()
            neg_den = ((1.0 - targets) * target_mask).sum().clamp(min=1.0)
            coverage = coverage + neg_penalty / neg_den

    return coverage


def max_margin_coverage_loss(
    clause_truth: Tensor,
    targets: Tensor,
    target_mask: Tensor,
    margin_pos: float = 0.7,
    margin_neg: float = 0.3,
) -> Tensor:
    """Max-margin coverage loss that only penalizes when the best clause is below/above margin.

    Unlike standard coverage loss which rewards spreading probability across clauses,
    this loss only cares about the MAX clause score per example.

    Args:
        clause_truth: Per-clause truth values [B, M, H, T] or [B, H, T, M]
        targets: Y_val tensor [B, M, H]
        target_mask: Y_mask tensor [B, M, H]
        margin_pos: Margin for positive examples (penalize if max < margin_pos)
        margin_neg: Margin for negative examples (penalize if max > margin_neg)

    Returns:
        Scalar loss tensor
    """

    # Handle different clause_truth layouts
    # We need [B, M, H, T] to take max over T (clause slots)
    if clause_truth.ndim != 4:
        raise ValueError(
            f"clause_truth must be 4D, got shape {tuple(clause_truth.shape)}"
        )

    # Check layout: if shape[1:3] matches targets shape[1:3], it's [B, M, H, T]
    # Otherwise assume [B, H, T, M] and permute
    if (
        clause_truth.shape[1] == targets.shape[1]
        and clause_truth.shape[2] == targets.shape[2]
    ):
        # Already [B, M, H, T]
        clause_truth_bmht = clause_truth
    else:
        # Assume [B, H, T, M], permute to [B, M, H, T]
        clause_truth_bmht = clause_truth.permute(0, 3, 1, 2).contiguous()

    # Take max over clause slots (T dimension) -> [B, M, H]
    max_clause_score, _ = clause_truth_bmht.max(dim=-1)

    # Clamp for numerical stability
    max_clause_score = max_clause_score.clamp(1e-6, 1 - 1e-6)

    # Compute masks
    mask = target_mask.to(max_clause_score.dtype)
    pos_mask = (targets > 0.5).to(max_clause_score.dtype) * mask
    neg_mask = (targets <= 0.5).to(max_clause_score.dtype) * mask

    # Positive examples: penalize if max_score < margin_pos
    pos_violation = torch.relu(margin_pos - max_clause_score)
    pos_loss = (pos_violation * pos_mask).sum()
    pos_count = pos_mask.sum().clamp(min=1.0)

    # Negative examples: penalize if max_score > margin_neg
    neg_violation = torch.relu(max_clause_score - margin_neg)
    neg_loss = (neg_violation * neg_mask).sum()
    neg_count = neg_mask.sum().clamp(min=1.0)

    # Combine (average separately then add)
    total_loss = pos_loss / pos_count + neg_loss / neg_count

    return total_loss
