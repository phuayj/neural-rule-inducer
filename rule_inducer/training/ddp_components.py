"""Components for distributed data-parallel training with advanced features."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from .protocols import EvalOutput, LossOutput
from .state import TrainState

__all__ = [
    "DDPTrainContext",
    "DDPScheduleManager",
    "DDPLossComputer",
    "DDPEvaluator",
    "get_scheduled_value",
]


# =============================================================================
# Schedule Utilities
# =============================================================================


def get_scheduled_value(
    schedule: Optional[List[Tuple[int, Any]]],
    step: int,
    default: Any,
) -> Any:
    """Get value from a step-based schedule."""

    if not schedule:
        return default
    current = default
    for boundary, candidate in schedule:
        if step >= boundary:
            current = candidate
        else:
            break
    return current


# =============================================================================
# Training Context
# =============================================================================


@dataclass
class DDPTrainContext:
    """Context for a single training step in DDP training."""

    step_idx: int
    total_steps: int

    coverage_scale: float = 1.0
    coverage_pos_weight: float = 1.0
    coverage_neg_weight: float = 1.0
    margin_pos: float = 0.0
    margin_neg: float = 0.0

    max_margin_coverage_enabled: bool = False
    max_margin_coverage_pos: float = 0.7
    max_margin_coverage_neg: float = 0.3
    max_margin_coverage_weight: float = 1.0

    slot_load_balance_weight: float = 0.0
    slot_balance_cv2_weight: float = 0.0
    slot_activation_balance_weight: float = 0.0

    cf_necessity_enabled: bool = False
    cf_necessity_weight: float = 0.1
    cf_necessity_spurious_weight: float = 0.1
    cf_necessity_select_threshold: float = 0.3
    cf_necessity_num_samples: int = 4
    cf_necessity_warmup_steps: int = 0
    cf_necessity_overlap_weight: float = 0.1
    cf_necessity_balance_weight: float = 0.01

    clause_topk_override: Optional[int] = None

    extra_config: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Schedule Manager
# =============================================================================


class DDPScheduleManager:
    """Manages schedule computations for DDP training."""

    def __init__(self, args: argparse.Namespace, total_steps: int) -> None:
        self.args = args
        self.total_steps = int(total_steps)

    def compute_context(
        self,
        step: int,
        batch: Dict[str, Any],
        device: torch.device,
    ) -> DDPTrainContext:
        del batch
        del device

        args = self.args
        total_steps = int(self.total_steps)

        coverage_scale = 1.0

        return DDPTrainContext(
            step_idx=int(step),
            total_steps=total_steps,
            coverage_scale=coverage_scale,
            coverage_pos_weight=float(getattr(args, "coverage_pos_weight", 1.0)),
            coverage_neg_weight=float(getattr(args, "coverage_neg_weight", 1.0)),
            margin_pos=float(getattr(args, "margin_pos", 0.0)),
            margin_neg=float(getattr(args, "margin_neg", 0.0)),
            max_margin_coverage_enabled=bool(
                getattr(args, "max_margin_coverage_enabled", False)
            ),
            max_margin_coverage_pos=float(
                getattr(args, "max_margin_coverage_pos", 0.7)
            ),
            max_margin_coverage_neg=float(
                getattr(args, "max_margin_coverage_neg", 0.3)
            ),
            max_margin_coverage_weight=float(
                getattr(args, "max_margin_coverage_weight", 1.0)
            ),
            slot_load_balance_weight=float(
                getattr(args, "slot_load_balance_weight", 0.0)
            ),
            slot_balance_cv2_weight=float(
                getattr(args, "slot_balance_cv2_weight", 0.0)
            ),
            slot_activation_balance_weight=float(
                getattr(args, "slot_activation_balance_weight", 0.0)
            ),
            cf_necessity_enabled=bool(getattr(args, "cf_necessity_enabled", False)),
            cf_necessity_weight=float(getattr(args, "cf_necessity_weight", 0.1)),
            cf_necessity_spurious_weight=float(
                getattr(args, "cf_necessity_spurious_weight", 0.1)
            ),
            cf_necessity_select_threshold=float(
                getattr(args, "cf_necessity_select_threshold", 0.3)
            ),
            cf_necessity_num_samples=int(getattr(args, "cf_necessity_num_samples", 4)),
            cf_necessity_warmup_steps=int(
                getattr(args, "cf_necessity_warmup_steps", 0)
            ),
            cf_necessity_overlap_weight=float(
                getattr(args, "cf_necessity_overlap_weight", 0.1)
            ),
            cf_necessity_balance_weight=float(
                getattr(args, "cf_necessity_balance_weight", 0.01)
            ),
            clause_topk_override=get_scheduled_value(
                getattr(args, "clause_topk_schedule", None),
                step,
                getattr(args, "clause_topk", None),
            ),
            extra_config={"args": args},
        )


# =============================================================================
# Loss Computer
# =============================================================================


class DDPLossComputer:
    """Loss computer for DDP training (best_v1 coverage + setmatch)."""

    def __init__(
        self,
        args: argparse.Namespace,
        schedule_manager: DDPScheduleManager,
        device: torch.device,
        model: nn.Module,
    ) -> None:
        self.args = args
        self.schedule_manager = schedule_manager
        self.device = device
        self.model = model

        self._cached_slot_load_balance_loss = torch.tensor(0.0, device=device)
        self._cached_slot_balance_cv2_loss = torch.tensor(0.0, device=device)
        self._cached_slot_activation_balance_loss = torch.tensor(0.0, device=device)

    def _normalize_outputs(self, outputs: Any) -> Dict[str, Any]:
        if isinstance(outputs, dict):
            return outputs
        if hasattr(outputs, "_asdict"):
            return outputs._asdict()  # type: ignore[return-value]
        if hasattr(outputs, "__dict__"):
            return dict(outputs.__dict__)
        return {"outputs": outputs}

    def _clause_truth_bhtm(
        self, clause_truth: Tensor, y_val: Tensor
    ) -> Optional[Tensor]:
        if clause_truth.ndim != 4:
            return None
        if (
            clause_truth.shape[1] == y_val.shape[1]
            and clause_truth.shape[2] == y_val.shape[2]
        ):
            return clause_truth.permute(0, 2, 3, 1).contiguous()
        if (
            clause_truth.shape[1] == y_val.shape[2]
            and clause_truth.shape[3] == y_val.shape[1]
        ):
            return clause_truth
        return None

    def __call__(
        self,
        outputs: Any,
        batch: Dict[str, Tensor],
        state: TrainState,
    ) -> LossOutput:
        step = int(state.step)

        ctx = self.schedule_manager.compute_context(step, batch, self.device)

        outputs_dict = self._normalize_outputs(outputs)
        if "R_pred" not in outputs_dict:
            raise ValueError("LossComputer requires outputs with 'R_pred'.")

        predictions = outputs_dict["R_pred"].float()
        lit_probs = outputs_dict.get("Lit_probs")
        gate_logits = outputs_dict.get("Clause_gate_logits")
        clause_truth = outputs_dict.get("Clause_truth")

        if not isinstance(gate_logits, Tensor):
            raise ValueError("LossComputer requires 'Clause_gate_logits' tensor.")
        if not isinstance(lit_probs, Tensor):
            raise ValueError("LossComputer requires 'Lit_probs' tensor.")

        y_val = batch["Y_val"].float()
        y_mask = batch.get("Y_mask")
        if not isinstance(y_mask, Tensor):
            y_mask = torch.ones_like(y_val)
        y_mask = y_mask.float()

        n_len = batch.get("N_len")
        if not isinstance(n_len, Tensor):
            raise ValueError("LossComputer requires batch['N_len'] tensor.")
        h_len = batch.get("H_len")
        if not isinstance(h_len, Tensor):
            h_len = torch.full(
                (predictions.shape[0],), predictions.shape[2], device=self.device
            )

        parts: Dict[str, Tensor] = {}
        logs: Dict[str, float] = {}

        from rule_inducer.losses import coverage_loss

        cov_loss = coverage_loss(
            predictions,
            y_val,
            y_mask,
            pos_weight=ctx.coverage_pos_weight,
            neg_weight=ctx.coverage_neg_weight,
            margin_pos=ctx.margin_pos,
            margin_neg=ctx.margin_neg,
        ).to(torch.float32)

        cov_loss_scaled = cov_loss * ctx.coverage_scale
        parts["coverage"] = cov_loss_scaled
        logs["coverage"] = float(cov_loss.item())
        logs["coverage_scale"] = float(ctx.coverage_scale)

        max_margin_loss = cov_loss.new_zeros(())
        if ctx.max_margin_coverage_enabled and isinstance(clause_truth, Tensor):
            from rule_inducer.losses import max_margin_coverage_loss

            max_margin_loss = max_margin_coverage_loss(
                clause_truth,
                y_val,
                y_mask,
                margin_pos=ctx.max_margin_coverage_pos,
                margin_neg=ctx.max_margin_coverage_neg,
            ).to(torch.float32)
            parts["max_margin_coverage"] = (
                max_margin_loss * ctx.max_margin_coverage_weight
            )
            logs["max_margin_coverage"] = float(max_margin_loss.item())

        gate_logits_f = gate_logits.float().clamp(min=-20.0, max=20.0)
        gate_probs = torch.sigmoid(gate_logits_f).clamp(min=1e-6, max=1.0 - 1e-6)

        slot_load_balance_loss = cov_loss.new_zeros(())
        if ctx.slot_load_balance_weight > 0.0:
            from rule_inducer.losses import (
                slot_load_balance_loss as compute_slot_load_balance,
            )

            slot_load_balance_loss = compute_slot_load_balance(
                gate_probs.unsqueeze(-1), gate_logits_f
            ).to(torch.float32)
            self._cached_slot_load_balance_loss = slot_load_balance_loss.detach()

        slot_balance_cv2_loss = self._cached_slot_balance_cv2_loss
        if ctx.slot_balance_cv2_weight > 0:
            from rule_inducer.losses import slot_load_balance_cv2_loss

            slot_balance_cv2_loss = slot_load_balance_cv2_loss(gate_logits_f).to(
                torch.float32
            )
            self._cached_slot_balance_cv2_loss = slot_balance_cv2_loss.detach()
        parts["slot_balance_cv2"] = slot_balance_cv2_loss * ctx.slot_balance_cv2_weight
        logs["slot_balance_cv2"] = float(slot_balance_cv2_loss.item())

        slot_activation_balance_loss_val = self._cached_slot_activation_balance_loss
        if ctx.slot_activation_balance_weight > 0:
            from rule_inducer.losses import (
                slot_activation_balance_loss as compute_slot_activation_balance_loss,
            )

            slot_activation_balance_loss_val = compute_slot_activation_balance_loss(
                gate_logits_f
            ).to(torch.float32)
            self._cached_slot_activation_balance_loss = (
                slot_activation_balance_loss_val.detach()
            )
        parts["slot_activation_balance"] = (
            slot_activation_balance_loss_val * ctx.slot_activation_balance_weight
        )
        logs["slot_activation_balance"] = float(slot_activation_balance_loss_val.item())

        parts["slot_load_balance"] = (
            slot_load_balance_loss * ctx.slot_load_balance_weight
        )
        logs["slot_load_balance"] = float(slot_load_balance_loss.item())

        cf_necessity_loss = None
        if ctx.cf_necessity_enabled:
            if not isinstance(clause_truth, Tensor):
                raise ValueError("cf_necessity_enabled requires Clause_truth tensor.")
            clause_truth_bhtm = self._clause_truth_bhtm(clause_truth, y_val)
            if clause_truth_bhtm is None:
                raise ValueError(
                    "Clause_truth shape is incompatible with cf necessity."
                )
            if step >= int(ctx.cf_necessity_warmup_steps):
                from rule_inducer.counterfactual_necessity import (
                    compute_responsibility_weighted_cf_loss,
                )

                x_val = batch["X_val"].float()
                x_mask = batch.get("X_mask")
                if not isinstance(x_mask, Tensor):
                    x_mask = torch.ones_like(x_val, dtype=torch.bool)
                cf_necessity_loss, cf_metrics = compute_responsibility_weighted_cf_loss(
                    X_val=x_val,
                    X_mask=x_mask.float(),
                    Y_val=y_val,
                    Lit_probs=lit_probs,
                    clause_truth=clause_truth_bhtm.float(),
                    necessity_weight=ctx.cf_necessity_weight,
                    spuriousness_weight=ctx.cf_necessity_spurious_weight,
                    overlap_weight=ctx.cf_necessity_overlap_weight,
                    balance_weight=ctx.cf_necessity_balance_weight,
                    select_threshold=ctx.cf_necessity_select_threshold,
                    num_cf_samples=ctx.cf_necessity_num_samples,
                )
                logs["cf_necessity_loss"] = float(cf_metrics["necessity_loss"].item())
                logs["cf_spuriousness_loss"] = float(
                    cf_metrics["spuriousness_loss"].item()
                )
                logs["cf_positive_examples"] = float(
                    cf_metrics["num_positive_examples"].item()
                )
                logs["cf_selected_fraction"] = float(
                    cf_metrics["cf_selected_fraction"].item()
                )
                logs["cf_responsibility_entropy"] = float(
                    cf_metrics["responsibility_entropy"].item()
                )

        if cf_necessity_loss is not None:
            parts["cf_necessity"] = cf_necessity_loss

        total_loss = (
            cov_loss_scaled
            + max_margin_loss * ctx.max_margin_coverage_weight
            + slot_load_balance_loss * ctx.slot_load_balance_weight
            + slot_balance_cv2_loss * ctx.slot_balance_cv2_weight
            + slot_activation_balance_loss_val * ctx.slot_activation_balance_weight
        )

        if cf_necessity_loss is not None:
            total_loss = total_loss + cf_necessity_loss

        logs["step"] = float(step)

        return LossOutput(total=total_loss, parts=parts, logs=logs)


# =============================================================================
# Evaluator
# =============================================================================


class DDPEvaluator:
    """Evaluator for DDP training with rule-matching metrics."""

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        self.args = args
        self.device = device

    def evaluate(
        self,
        model: nn.Module,
        loaders: Dict[str, Any],
        state: TrainState,
    ) -> EvalOutput:
        del state

        val_loader = loaders.get("val")
        if val_loader is None:
            return EvalOutput(metrics={}, artifacts={})

        model.eval()

        total_loss = 0.0
        total_count = 0.0
        correct = 0.0

        match_stats = None
        rule_exact_matches = 0
        rule_exact_total = 0

        gate_entropy_total = 0.0
        gate_entropy_denom = 0.0

        clause_count_total = 0.0
        clause_count_denom = 0.0
        literal_count_total = 0.0
        literal_count_denom = 0.0

        positive_ratio_total = 0.0
        positive_ratio_batches = 0

        clause_topk_override = getattr(self.args, "clause_topk", None)

        from rule_inducer.losses import coverage_loss
        from rule_inducer import RuleMatchStats, decode_program
        from rule_inducer.eval import compute_exact_match

        with torch.no_grad():
            for batch in val_loader:
                batch = self._move_batch_to_device(batch)

                output = model(
                    batch["X_val"],
                    batch.get("X_mask", torch.ones_like(batch["X_val"])),
                    batch["Y_val"],
                    batch.get("Y_mask", torch.ones_like(batch["Y_val"])),
                    batch["N_len"],
                    batch["M_len"],
                    batch["H_len"],
                    gumbel=False,
                    clause_topk_override=clause_topk_override,
                )

                output_dict = output if isinstance(output, dict) else output.__dict__
                r_pred = output_dict["R_pred"]
                gate_logits = output_dict["Clause_gate_logits"]
                complexity = output_dict.get("Complexity_metrics", {})

                cov = coverage_loss(
                    r_pred,
                    batch["Y_val"],
                    batch.get("Y_mask", torch.ones_like(batch["Y_val"])),
                    pos_weight=float(getattr(self.args, "coverage_pos_weight", 1.0)),
                    neg_weight=float(getattr(self.args, "coverage_neg_weight", 1.0)),
                    margin_pos=float(getattr(self.args, "margin_pos", 0.0)),
                    margin_neg=float(getattr(self.args, "margin_neg", 0.0)),
                )

                labels = (
                    batch["Y_mask"].sum().item()
                    if "Y_mask" in batch
                    else batch["Y_val"].numel()
                )
                total_loss += cov.item() * labels
                total_count += labels

                preds = (r_pred >= 0.5).float()
                y_mask = batch.get("Y_mask", torch.ones_like(batch["Y_val"]))
                correct += (
                    ((preds == batch["Y_val"]).float() * y_mask.float()).sum().item()
                )

                if batch.get("rules") is not None:
                    if match_stats is None:
                        match_stats = RuleMatchStats()
                    export_program = getattr(model, "export_program")
                    program = export_program(
                        output,
                        N_len=batch["N_len"],
                        H_len=batch["H_len"],
                        M_len=batch["M_len"],
                    )
                    program_cpu = {
                        k: (v.detach().cpu() if isinstance(v, Tensor) else v)
                        for k, v in program.items()
                    }
                    decoded = decode_program(
                        program_cpu,
                        batch["N_len"].detach().cpu(),
                        batch["H_len"].detach().cpu(),
                    )
                    target_rules = batch["rules"]
                    if isinstance(target_rules, list):
                        matches, total = compute_exact_match(decoded, target_rules)
                        rule_exact_matches += matches
                        rule_exact_total += total
                        match_stats.update(decoded, target_rules)

                gate_probs = torch.sigmoid(gate_logits.float()).clamp(1e-6, 1 - 1e-6)
                B, H, _ = gate_probs.shape
                head_mask = torch.arange(H, device=self.device).unsqueeze(0) < batch[
                    "H_len"
                ].unsqueeze(1)
                gate_mask = head_mask.unsqueeze(-1).to(gate_probs.dtype)
                gate_entropy = -torch.special.xlogy(
                    gate_probs, gate_probs
                ) - torch.special.xlogy(1 - gate_probs, 1 - gate_probs)
                gate_entropy_total += (gate_entropy * gate_mask).sum().item()
                gate_entropy_denom += gate_mask.sum().item()

                if isinstance(complexity, dict):
                    expected_clauses = complexity.get("expected_active_clauses")
                    if isinstance(expected_clauses, Tensor):
                        head_mask_f = head_mask.to(expected_clauses.dtype)
                        per_head_expected = (
                            expected_clauses * head_mask_f.unsqueeze(-1)
                        ).sum(dim=-1)
                        clause_count_total += per_head_expected.sum().item()
                        clause_count_denom += head_mask_f.sum().item()

                    expected_literals = complexity.get("expected_literals_per_clause")
                    if isinstance(expected_literals, Tensor):
                        clause_mask = (
                            head_mask.unsqueeze(-1)
                            .to(expected_literals.dtype)
                            .expand_as(expected_literals)
                        )
                        literal_count_total += (
                            (expected_literals * clause_mask).sum().item()
                        )
                        literal_count_denom += clause_mask.sum().item()

                if "Y_mask" in batch and "Y_val" in batch:
                    y_mask_batch = batch["Y_mask"].float()
                    y_val_batch = batch["Y_val"].float()
                    pos_mask = (y_val_batch > 0.5).float() * y_mask_batch
                    total_examples = y_mask_batch.sum().clamp(min=1.0)
                    positive_ratio_total += (pos_mask.sum() / total_examples).item()
                    positive_ratio_batches += 1

        model.train()

        metrics: Dict[str, float] = {}
        if total_count > 0:
            metrics["coverage_loss"] = total_loss / total_count
            metrics["accuracy"] = correct / total_count

        if gate_entropy_denom > 0:
            metrics["gate_entropy"] = gate_entropy_total / gate_entropy_denom

        if clause_count_denom > 0:
            metrics["expected_clauses"] = clause_count_total / clause_count_denom

        if literal_count_denom > 0:
            metrics["expected_literals"] = literal_count_total / literal_count_denom

        if positive_ratio_batches > 0:
            metrics["positive_ratio"] = positive_ratio_total / positive_ratio_batches

        if match_stats is not None:
            summary = match_stats.to_metrics()
            metrics["clause_precision"] = summary.get("clause_precision", 0.0)
            metrics["clause_recall"] = summary.get("clause_recall", 0.0)
            metrics["clause_f1"] = summary.get("clause_f1", 0.0)
            metrics["rule_exact_match"] = (
                rule_exact_matches / rule_exact_total if rule_exact_total > 0 else 0.0
            )

        return EvalOutput(metrics=metrics, artifacts={})

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, Tensor):
                moved[key] = value.to(self.device, non_blocking=True)
            else:
                moved[key] = value
        return moved
