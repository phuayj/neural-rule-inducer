"""Neural Rule Inducer - Differentiable DNF learning."""

from .data import (
    SyntheticEpisodeConfig,
    SyntheticEpisodeDataset,
    NPZEpisodeDataset,
    synthetic_episode_collate,
    Episode,
    CollatedEpisodeBatch,
)
from .model import RuleInducer, RuleInducerOutput, LiteralFilmConfig
from .eval import (
    RuleMatchStats,
    decode_program,
    evaluate_rules_on_examples,
    rule_precision_recall,
)

__all__ = [
    "RuleInducer",
    "RuleInducerOutput",
    "LiteralFilmConfig",
    "SyntheticEpisodeConfig",
    "SyntheticEpisodeDataset",
    "NPZEpisodeDataset",
    "synthetic_episode_collate",
    "Episode",
    "CollatedEpisodeBatch",
    "RuleMatchStats",
    "decode_program",
    "evaluate_rules_on_examples",
    "rule_precision_recall",
]
