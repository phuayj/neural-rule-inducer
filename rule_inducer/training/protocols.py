"""Protocols and dataclasses for training components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, runtime_checkable

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .state import TrainState

__all__ = [
    "LossOutput",
    "EvalOutput",
    "LossComputer",
    "Evaluator",
    "Callback",
]


@dataclass
class LossOutput:
    """Output from a loss computation.

    Attributes:
        total: Loss tensor to backpropagate.
        parts: Individual loss components (for logging/inspection).
        logs: Scalar metrics pre-computed for logging.
    """

    total: Tensor
    parts: Dict[str, Tensor] = field(default_factory=dict)
    logs: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvalOutput:
    """Output from evaluation.

    Attributes:
        metrics: Dictionary of metric name -> scalar value.
        artifacts: Additional artifacts generated during evaluation.
    """

    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LossComputer(Protocol):
    """Protocol for computing training losses."""

    def __call__(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Tensor],
        state: TrainState,
    ) -> LossOutput:
        """Compute loss from model outputs and input batch."""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for evaluation helpers."""

    def evaluate(
        self,
        model: torch.nn.Module,
        loaders: Dict[str, DataLoader],
        state: TrainState,
    ) -> EvalOutput:
        """Run evaluation and return metrics/artifacts."""
        ...


@runtime_checkable
class Callback(Protocol):
    """Protocol for training callbacks/hooks."""

    def on_train_start(self, state: TrainState, **kwargs: Any) -> None:
        """Called before the first training step."""
        ...

    def on_step_start(self, state: TrainState, **kwargs: Any) -> None:
        """Called at the start of each training step."""
        ...

    def on_step_end(
        self, state: TrainState, metrics: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Called after each training step."""
        ...

    def on_eval_end(
        self, state: TrainState, eval_output: EvalOutput, **kwargs: Any
    ) -> None:
        """Called after an evaluation pass."""
        ...

    def on_train_end(self, state: TrainState, **kwargs: Any) -> None:
        """Called after training completes."""
        ...
