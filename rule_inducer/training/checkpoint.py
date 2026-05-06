"""Checkpoint management for training runs."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn

from .state import TrainState, capture_rng_state, restore_rng_state

logger = logging.getLogger(__name__)

__all__ = ["CheckpointManager", "is_rank_zero"]


def is_rank_zero() -> bool:
    """Return True if this process is rank 0 (or non-distributed)."""
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


class CheckpointManager:
    """Save and load checkpoints with rank-0 safety."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        *,
        keep_last_n: int = 3,
        save_best: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = int(keep_last_n)
        self.save_best = bool(save_best)

        if is_rank_zero():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        state: TrainState,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> Optional[Path]:
        """Save a checkpoint payload (rank 0 only)."""
        if not is_rank_zero():
            return None

        model_to_save = model.module if hasattr(model, "module") else model

        checkpoint: Dict[str, Any] = {
            "train_state": state.state_dict(),
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": capture_rng_state(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()
        if config is not None:
            checkpoint["config"] = config
        if metrics is not None:
            checkpoint["metrics"] = metrics

        step = int(state.step)
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step:07d}.pt"
        torch.save(checkpoint, checkpoint_path)

        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        shutil.copy2(checkpoint_path, latest_path)

        if is_best and self.save_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            shutil.copy2(checkpoint_path, best_path)

        self._cleanup_old_checkpoints()
        return checkpoint_path

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        *,
        checkpoint_path: Optional[Union[str, Path]] = None,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_rng: bool = True,
        map_location: Optional[str] = None,
    ) -> TrainState:
        """Load a checkpoint and restore training state."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pt"

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if map_location is None:
            map_location = "cpu"

        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.load_state_dict(checkpoint["model_state_dict"], strict=True)

        if (
            load_optimizer
            and optimizer is not None
            and "optimizer_state_dict" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if (
            load_scheduler
            and scheduler is not None
            and "scheduler_state_dict" in checkpoint
        ):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if load_rng and "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])

        return TrainState.from_state_dict(checkpoint.get("train_state", {}))

    def find_latest(self) -> Optional[Path]:
        """Return the path to the latest checkpoint if it exists."""
        latest = self.checkpoint_dir / "checkpoint_latest.pt"
        return latest if latest.exists() else None

    def _cleanup_old_checkpoints(self) -> None:
        """Keep only the most recent N checkpoints."""
        if self.keep_last_n <= 0:
            return

        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoints) <= self.keep_last_n:
            return

        to_remove = checkpoints[: -self.keep_last_n]
        for path in to_remove:
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("Failed to remove old checkpoint %s: %s", path, exc)
