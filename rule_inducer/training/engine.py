"""Training engine with DDP and AMP support."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .checkpoint import CheckpointManager, is_rank_zero
from .logging import MetricsLogger
from .protocols import Callback, EvalOutput, Evaluator, LossComputer, LossOutput
from .state import TrainState

__all__ = ["TrainingConfig", "TrainingEngine", "unwrap_model"]


@dataclass
class TrainingConfig:
    """Configuration for the training engine.

    Attributes:
        num_steps: Total number of optimizer steps.
        log_interval: Steps between metric logs.
        eval_interval: Steps between evaluations (0 to disable).
        checkpoint_interval: Steps between checkpoints (0 to disable).
        grad_clip: Max gradient norm (0 to disable).
        use_amp: Enable automatic mixed precision.
        grad_accumulation: Micro-batches to accumulate before stepping.
    """

    num_steps: int = 10000
    log_interval: int = 100
    eval_interval: int = 1000
    checkpoint_interval: int = 1000
    grad_clip: float = 1.0
    use_amp: bool = False
    grad_accumulation: int = 1


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the wrapped module when using DDP."""
    return model.module if isinstance(model, DDP) else model


class TrainingEngine:
    """Core training loop with optional evaluation and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_computer: LossComputer,
        train_loader: DataLoader,
        config: TrainingConfig,
        *,
        scheduler: Optional[Any] = None,
        evaluator: Optional[Evaluator] = None,
        val_loaders: Optional[Dict[str, DataLoader]] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        logger: Optional[MetricsLogger] = None,
        callbacks: Optional[Sequence[Callback]] = None,
        device: Optional[torch.device] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_computer = loss_computer
        self.train_loader = train_loader
        self.config = config
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.val_loaders = val_loaders or {}
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger or MetricsLogger(log_interval=config.log_interval)
        self.callbacks = list(callbacks) if callbacks else []
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.run_config = run_config

        self.scaler: Optional[torch.amp.GradScaler] = None
        if config.use_amp and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")

        self.state = TrainState()
        self._train_iter: Optional[Iterator[Dict[str, Any]]] = None
        self._accum_steps: int = 0

    def _get_train_iter(self) -> Iterator[Dict[str, Any]]:
        if self._train_iter is None:
            self._train_iter = self._infinite_train_iterator()
        return self._train_iter

    def _infinite_train_iterator(self) -> Iterator[Dict[str, Any]]:
        while True:
            sampler = getattr(self.train_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(int(self.state.epoch))

            for batch in self.train_loader:
                yield batch

            self.state.epoch += 1

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, Tensor):
                moved[key] = value.to(self.device, non_blocking=True)
            else:
                moved[key] = value
        return moved

    def _call_callbacks(self, method: str, **kwargs: Any) -> None:
        for callback in self.callbacks:
            fn = getattr(callback, method, None)
            if fn is not None:
                fn(self.state, **kwargs)

    def _grad_norm(self) -> float:
        norm_sq = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                norm_sq += param.grad.detach().norm(p=2).pow(2).item()
        return math.sqrt(norm_sq)

    def _normalize_outputs(self, outputs: Any) -> Dict[str, Any]:
        if isinstance(outputs, dict):
            return outputs
        if hasattr(outputs, "_asdict"):
            return outputs._asdict()  # type: ignore[return-value]
        if hasattr(outputs, "__dict__"):
            return dict(outputs.__dict__)
        return {"outputs": outputs}

    def train_step(self, batch: Dict[str, Any]) -> Tuple[Dict[str, float], bool]:
        self.model.train()

        grad_accumulation = max(int(self.config.grad_accumulation), 1)
        if self._accum_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)

        amp_enabled = self.scaler is not None

        def compute_loss(
            local_batch: Dict[str, Any],
        ) -> Tuple[LossOutput, Dict[str, Any]]:
            autocast_ctx = torch.amp.autocast("cuda") if amp_enabled else nullcontext()
            with autocast_ctx:
                outputs = self.model(
                    local_batch["X_val"],
                    local_batch.get(
                        "X_mask",
                        torch.ones_like(local_batch["X_val"], dtype=torch.bool),
                    ),
                    local_batch["Y_val"],
                    local_batch.get(
                        "Y_mask",
                        torch.ones_like(local_batch["Y_val"], dtype=torch.bool),
                    ),
                    local_batch["N_len"],
                    local_batch["M_len"],
                    local_batch["H_len"],
                    gumbel=True,
                )

                outputs_dict = self._normalize_outputs(outputs)
                loss_output = self.loss_computer(outputs_dict, local_batch, self.state)
            return loss_output, outputs_dict

        loss_output, outputs_dict = compute_loss(batch)
        loss_value = loss_output.total
        if not bool(torch.isfinite(loss_value).item()):
            raise RuntimeError(
                f"Non-finite loss detected at step {self.state.step}: {loss_value}"
            )
        loss_for_backward = loss_value / float(grad_accumulation)
        if self.scaler is not None:
            self.scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        self._accum_steps += 1
        accum_steps_now = int(self._accum_steps)

        stepped = False
        grad_norm = 0.0

        if self._accum_steps >= grad_accumulation:
            stepped = True
            if self.config.grad_clip > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = float(
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), float(self.config.grad_clip)
                    ).item()
                )
            else:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = float(self._grad_norm())

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self.optimizer.zero_grad(set_to_none=True)
            self._accum_steps = 0

        metrics: Dict[str, float] = {
            "loss": float(loss_output.total.detach().item()),
            "grad_norm": float(grad_norm),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "accum_steps": float(accum_steps_now),
            "stepped": 1.0 if stepped else 0.0,
            **loss_output.logs,
        }

        if self.scaler is not None:
            metrics["amp_scale"] = float(self.scaler.get_scale())

        for name, tensor in loss_output.parts.items():
            if isinstance(tensor, Tensor) and tensor.numel() == 1:
                metrics[name] = float(tensor.detach().item())

        pred = outputs_dict.get("R_pred")
        target = batch.get("Y_val")
        mask = batch.get("Y_mask")
        if isinstance(pred, Tensor) and isinstance(target, Tensor):
            pred_values = pred.detach().to(torch.float32)
            target_values = target.detach().to(torch.float32)
            if (
                isinstance(mask, Tensor)
                and mask.shape == target_values.shape
                and bool(mask.any())
            ):
                valid = mask.detach().to(torch.bool)
                pred_values = pred_values[valid]
                target_values = target_values[valid]

            if pred_values.numel() > 0 and target_values.numel() > 0:
                metrics["pred_mean"] = float(pred_values.mean().item())
                metrics["target_mean"] = float(target_values.mean().item())
                metrics["pred_example0"] = float(pred_values.reshape(-1)[0].item())
                metrics["target_example0"] = float(target_values.reshape(-1)[0].item())

        return metrics, stepped

    def evaluate(self) -> Optional[EvalOutput]:
        if self.evaluator is None or not self.val_loaders:
            return None

        self.model.eval()
        with torch.no_grad():
            output = self.evaluator.evaluate(
                unwrap_model(self.model), self.val_loaders, self.state
            )
        self.model.train()
        return output

    def fit(self, resume_from: Optional[str] = None) -> TrainState:
        if resume_from and self.checkpoint_manager:
            self.state = self.checkpoint_manager.load(
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                checkpoint_path=resume_from,
            )
            if is_rank_zero():
                self.logger.log_text(f"Resumed from step {self.state.step}")

        self._call_callbacks("on_train_start")

        train_iter = self._get_train_iter()
        start_step = self.state.step + 1
        end_step = self.config.num_steps + 1

        if is_rank_zero():
            self.logger.log_text(
                f"Starting training from step {start_step} to {self.config.num_steps}"
            )

        for step in range(start_step, end_step):
            self.state.step = step
            self._call_callbacks("on_step_start")

            while True:
                batch = next(train_iter)
                batch = self._move_batch_to_device(batch)
                metrics, stepped = self.train_step(batch)
                if stepped:
                    break

            metrics["step"] = float(step)

            self._call_callbacks("on_step_end", metrics=metrics)

            if self.logger.should_log(step) or step == 1:
                self.logger.log(step, metrics)

            if self.config.eval_interval > 0 and step % self.config.eval_interval == 0:
                eval_output = self.evaluate()
                if eval_output is not None:
                    self.logger.log(step, eval_output.metrics, prefix="val")

                    primary_metric = eval_output.metrics.get("accuracy")
                    if primary_metric is not None:
                        if (
                            self.state.best_metric is None
                            or primary_metric > self.state.best_metric
                        ):
                            self.state.best_metric = float(primary_metric)
                            self.state.best_step = step

                    self._call_callbacks("on_eval_end", eval_output=eval_output)

            if (
                self.checkpoint_manager is not None
                and self.config.checkpoint_interval > 0
                and step % self.config.checkpoint_interval == 0
            ):
                is_best = (
                    self.state.best_step is not None and self.state.best_step == step
                )
                self.checkpoint_manager.save(
                    self.state,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    config=self.run_config,
                    metrics=metrics,
                    is_best=is_best,
                )

        final_eval = self.evaluate()
        if final_eval is not None:
            self.logger.log(
                self.state.step,
                final_eval.metrics,
                prefix="final",
                force=True,
            )

        if self.checkpoint_manager is not None:
            is_best = (
                self.state.best_step is not None
                and self.state.best_step == self.state.step
            )
            self.checkpoint_manager.save(
                self.state,
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                config=self.run_config,
                metrics=final_eval.metrics if final_eval else None,
                is_best=is_best,
            )

        self._call_callbacks("on_train_end")

        if is_rank_zero():
            self.logger.log_text(f"Training complete. Final step: {self.state.step}")
            if self.state.best_metric is not None:
                self.logger.log_text(
                    "Best metric: "
                    f"{self.state.best_metric:.4f} at step {self.state.best_step}"
                )

        return self.state
