"""Training state and RNG helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

__all__ = ["TrainState", "capture_rng_state", "restore_rng_state"]


@dataclass
class TrainState:
    """Serializable training state used for checkpointing.

    Attributes:
        step: Current 1-based training step.
        epoch: Epoch counter for dataloader reseeding.
        best_metric: Best validation metric observed so far.
        best_step: Step at which the best metric was observed.
        schedules: Mutable schedule state (temperature, lambdas, etc.).
    """

    step: int = 0
    epoch: int = 0
    best_metric: Optional[float] = None
    best_step: Optional[int] = None
    schedules: Dict[str, Any] = field(default_factory=dict)

    def state_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "step": int(self.step),
            "epoch": int(self.epoch),
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "schedules": dict(self.schedules),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "TrainState":
        """Reconstruct a :class:`TrainState` from a checkpoint payload."""
        return cls(
            step=int(state.get("step", 0)),
            epoch=int(state.get("epoch", 0)),
            best_metric=state.get("best_metric"),
            best_step=state.get("best_step"),
            schedules=dict(state.get("schedules", {})),
        )


def capture_rng_state() -> Dict[str, Any]:
    """Capture RNG state for deterministic resume."""
    state: Dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG state from a checkpoint payload."""
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
