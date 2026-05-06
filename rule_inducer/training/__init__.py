"""Unified training infrastructure for Neural Rule Inducer."""

from .state import TrainState, capture_rng_state, restore_rng_state
from .checkpoint import CheckpointManager, is_rank_zero
from .logging import MetricsLogger
from .engine import TrainingEngine, TrainingConfig, unwrap_model
from .protocols import (
    Callback,
    Evaluator,
    EvalOutput,
    LossComputer,
    LossOutput,
)
from .ddp_components import (
    DDPTrainContext,
    DDPScheduleManager,
    DDPLossComputer,
    DDPEvaluator,
    get_scheduled_value,
)
from .ddp_callbacks import (
    SignalHandlerCallback,
    DDPLoggingCallback,
)

__all__ = [
    # State
    "TrainState",
    "capture_rng_state",
    "restore_rng_state",
    # Checkpoint
    "CheckpointManager",
    "is_rank_zero",
    # Logging
    "MetricsLogger",
    # Engine
    "TrainingEngine",
    "TrainingConfig",
    "unwrap_model",
    # Protocols
    "Callback",
    "Evaluator",
    "EvalOutput",
    "LossComputer",
    "LossOutput",
    # DDP schedule/context
    "DDPTrainContext",
    "DDPScheduleManager",
    "DDPLossComputer",
    "DDPEvaluator",
    "get_scheduled_value",
    # DDP callbacks
    "SignalHandlerCallback",
    "DDPLoggingCallback",
]
