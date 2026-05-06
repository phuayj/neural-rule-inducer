"""DDP-specific callbacks for training engine integration."""

from __future__ import annotations

import json
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, IO, Optional

from torch import Tensor

from .protocols import EvalOutput
from .state import TrainState

__all__ = [
    "SignalHandlerCallback",
    "DDPLoggingCallback",
]


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _jsonify_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, Tensor):
            if value.numel() == 1:
                result[key] = float(value.detach().cpu().item())
            else:
                result[key] = value.detach().cpu().tolist()
        elif isinstance(value, (int, float, str, bool)) or value is None:
            result[key] = value
        else:
            result[key] = str(value)
    return result


# =============================================================================
# 1. Signal Handler
# =============================================================================


@dataclass
class SignalHandlerCallback:
    """Graceful shutdown on SIGINT/SIGTERM."""

    rank: int = 0
    stop_requested: bool = field(default=False, init=False)

    _registered: bool = field(default=False, init=False)
    _previous_handlers: Dict[int, Any] = field(default_factory=dict, init=False)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        del frame
        if self.stop_requested:
            return
        self.stop_requested = True
        if self.rank == 0:
            try:
                name = signal.Signals(signum).name
            except Exception:
                name = str(signum)
            print(
                f"[SignalHandlerCallback] Received {name}. "
                "Stop requested; will exit after current step.",
                flush=True,
            )

    def on_train_start(self, state: TrainState, **kwargs: Any) -> None:
        del kwargs
        self.stop_requested = False

        if self._registered:
            state.schedules["stop_requested"] = bool(self.stop_requested)
            return

        try:
            self._previous_handlers[int(signal.SIGINT)] = signal.getsignal(
                signal.SIGINT
            )
            self._previous_handlers[int(signal.SIGTERM)] = signal.getsignal(
                signal.SIGTERM
            )
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
            self._registered = True
            if self.rank == 0:
                print(
                    "[SignalHandlerCallback] Registered SIGINT/SIGTERM handlers.",
                    flush=True,
                )
        except ValueError:
            if self.rank == 0:
                print(
                    "[SignalHandlerCallback] WARNING: could not register signal handlers "
                    "(not in main thread).",
                    flush=True,
                )

        state.schedules["stop_requested"] = bool(self.stop_requested)

    def on_step_start(self, state: TrainState, **kwargs: Any) -> None:
        del kwargs
        state.schedules["stop_requested"] = bool(self.stop_requested)

    def on_step_end(
        self, state: TrainState, metrics: Dict[str, Any], **kwargs: Any
    ) -> None:
        del metrics
        del kwargs
        state.schedules["stop_requested"] = bool(self.stop_requested)

    def on_eval_end(
        self, state: TrainState, eval_output: EvalOutput, **kwargs: Any
    ) -> None:
        del state
        del eval_output
        del kwargs

    def on_train_end(self, state: TrainState, **kwargs: Any) -> None:
        del kwargs
        state.schedules["stop_requested"] = bool(self.stop_requested)
        if not self._registered:
            return

        try:
            prev_int = self._previous_handlers.get(int(signal.SIGINT))
            prev_term = self._previous_handlers.get(int(signal.SIGTERM))
            if prev_int is not None:
                signal.signal(signal.SIGINT, prev_int)
            if prev_term is not None:
                signal.signal(signal.SIGTERM, prev_term)
        except ValueError:
            pass

    def should_stop(self) -> bool:
        return bool(self.stop_requested)


# =============================================================================
# 2. Logging
# =============================================================================


@dataclass
class DDPLoggingCallback:
    """Logging for DDP training (JSONL + console)."""

    log_dir: Optional[Path] = None
    log_interval: int = 10
    rank: int = 0

    jsonl_file: Optional[IO[str]] = field(default=None, init=False)
    start_time: float = field(default=0.0, init=False)

    def on_train_start(self, state: TrainState, **kwargs: Any) -> None:
        del kwargs
        self.start_time = time.time()
        if self.rank == 0 and self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            if self.jsonl_file is None:
                self.jsonl_file = open(
                    self.log_dir / "metrics.jsonl", "a", encoding="utf-8"
                )
                self.jsonl_file.write(
                    json.dumps(
                        {
                            "event": "train_start",
                            "timestamp": time.time(),
                            "resume_step": int(state.step),
                        }
                    )
                    + "\n"
                )
                self.jsonl_file.flush()

    def on_step_start(self, state: TrainState, **kwargs: Any) -> None:
        del state
        del kwargs

    def on_step_end(
        self, state: TrainState, metrics: Dict[str, Any], **kwargs: Any
    ) -> None:
        del kwargs
        if self.rank != 0:
            return
        interval = int(max(self.log_interval, 1))
        if int(state.step) % interval != 0:
            return

        elapsed = float(time.time() - float(self.start_time))

        record = {
            "step": int(state.step),
            "elapsed_s": elapsed,
            **_jsonify_metrics(metrics),
        }

        if self.jsonl_file is not None:
            self.jsonl_file.write(json.dumps(record, sort_keys=True) + "\n")
            self.jsonl_file.flush()

        loss = _as_float(metrics.get("loss"))
        lr = _as_float(metrics.get("lr"))
        grad_norm = _as_float(metrics.get("grad_norm"))

        summary = f"[train] step={state.step}"
        if loss is not None:
            summary += f" loss={loss:.6f}"
        if lr is not None:
            summary += f" lr={lr:.3e}"
        if grad_norm is not None:
            summary += f" grad_norm={grad_norm:.4f}"
        summary += f" elapsed_s={elapsed:.1f}"
        print(summary, flush=True)

    def on_eval_end(
        self, state: TrainState, eval_output: EvalOutput, **kwargs: Any
    ) -> None:
        del kwargs
        if self.rank != 0:
            return
        if self.jsonl_file is None:
            return
        record = {
            "event": "eval_end",
            "step": int(state.step),
            "elapsed_s": float(time.time() - float(self.start_time)),
            "metrics": dict(eval_output.metrics),
        }
        self.jsonl_file.write(json.dumps(record, sort_keys=True) + "\n")
        self.jsonl_file.flush()

    def on_train_end(self, state: TrainState, **kwargs: Any) -> None:
        del kwargs
        if self.jsonl_file is not None:
            self.jsonl_file.write(
                json.dumps(
                    {
                        "event": "train_end",
                        "step": int(state.step),
                        "elapsed_s": float(time.time() - float(self.start_time)),
                    }
                )
                + "\n"
            )
            self.jsonl_file.close()
            self.jsonl_file = None
