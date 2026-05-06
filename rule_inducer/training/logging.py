"""Metrics logging utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TextIO, Union

import torch

__all__ = ["MetricsLogger", "is_rank_zero"]


def is_rank_zero() -> bool:
    """Return True if this process is rank 0 (or non-distributed)."""
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


class MetricsLogger:
    """JSONL metrics logger with rank-0 safety."""

    def __init__(
        self,
        log_path: Optional[Union[str, Path]] = None,
        log_interval: int = 1,
        console: bool = True,
        console_format: str = "step={step} loss={loss:.4f}",
    ) -> None:
        self.log_path = Path(log_path) if log_path else None
        self.log_interval = max(int(log_interval), 1)
        self.console = bool(console)
        self.console_format = str(console_format)
        self._file: Optional[TextIO] = None

        if is_rank_zero() and self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.log_path.open("a", encoding="utf-8")

    def should_log(self, step: int) -> bool:
        """Return True if metrics should be logged at this step."""
        return int(step) % int(self.log_interval) == 0

    def log(
        self,
        step: int,
        metrics: Dict[str, Any],
        *,
        prefix: str = "",
        force: bool = False,
    ) -> None:
        """Log metrics at a given step."""
        if not force and not self.should_log(step):
            return
        if not is_rank_zero():
            return

        if prefix:
            metrics = {f"{prefix}/{key}": value for key, value in metrics.items()}

        record = {
            "step": int(step),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **self._serialize_metrics(metrics),
        }

        if self._file is not None:
            self._file.write(json.dumps(record, sort_keys=True) + "\n")
            self._file.flush()

        if self.console:
            try:
                print(self.console_format.format(**record), flush=True)
            except KeyError:
                print(f"step={step} metrics={metrics}", flush=True)

    def log_text(self, message: str) -> None:
        """Log a text message to the console."""
        if is_rank_zero():
            print(message, flush=True)

    def close(self) -> None:
        """Close the log file handle if open."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @staticmethod
    def _serialize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metrics to JSON-serializable types."""
        serialized: Dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    value = float(value.item())
                else:
                    value = value.tolist()
            elif isinstance(value, (int, float, str, bool, type(None))):
                pass
            elif hasattr(value, "__float__"):
                value = float(value)
            else:
                value = str(value)
            serialized[key] = value
        return serialized
