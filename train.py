#!/usr/bin/env python3
"""Distributed training entry point for the Rule Inducer foundation model."""

from __future__ import annotations

import argparse
import atexit
import json
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import Dataset

try:  # PyTorch 2.x+
    from torch.nn.modules.lazy import LazyModuleMixin
except ImportError:  # pragma: no cover - fallback

    class LazyModuleMixin:  # type: ignore
        pass


from rule_inducer import (
    NPZEpisodeDataset,
    RuleInducer,
    SyntheticEpisodeConfig,
    SyntheticEpisodeDataset,
    synthetic_episode_collate,
)

from rule_inducer.model import LiteralFilmConfig


def _load_config_file(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    suffix = config_path.suffix.lower()
    data: Any
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                f"PyYAML is required to parse '{config_path.suffix}' configs but is not installed."
            ) from exc
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    else:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected top-level object in '{config_path}' to be a mapping."
        )
    return data


def _ensure_path(value: Any | None) -> Optional[Path]:
    if value is None or value == "":
        return None
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _prepare_schedule(value: Any) -> Optional[list[tuple[int, Any]]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        prepared: list[tuple[int, Any]] = []
        for entry in value:
            if isinstance(entry, dict):
                if "step" not in entry or "value" not in entry:
                    raise ValueError(
                        "Schedule dict entries must provide 'step' and 'value' keys."
                    )
                step = int(entry["step"])
                val = entry["value"]
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                step = int(entry[0])
                val = entry[1]
            else:
                raise ValueError(f"Unsupported schedule entry: {entry!r}")
            prepared.append((step, val))
        prepared.sort(key=lambda item: item[0])
        return prepared
    raise ValueError(f"Unsupported schedule container type: {type(value)!r}")


def _has_uninitialized_lazy_params(module: nn.Module) -> bool:
    if LazyModuleMixin is None:
        return False
    for submodule in module.modules():
        if isinstance(submodule, LazyModuleMixin):
            has_check = getattr(submodule, "has_uninitialized_params", None)
            if callable(has_check) and has_check():
                return True
    return False


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    The config file (JSON/YAML) is the single source of truth for all
    training/model/loss hyperparameters. The CLI only exposes runtime/
    operational overrides.
    """

    default_config: Dict[str, Any] = {
        # Dataset / loader defaults (also CLI-overridable).
        "train_manifest": None,
        "val_manifest": None,
        "manifest_format": "npz",
        "synthetic_only": False,
        "epochs": 50,
        "total_steps": 0,
        "batch_size": 32,
        "eval_batch_size": 32,
        "train_num_workers": 4,
        "eval_num_workers": 4,
        "train_pin_memory": True,
        "eval_pin_memory": True,
        "train_prefetch_factor": 2,
        "eval_prefetch_factor": 2,
        "output_dir": Path("runs/latest"),
        "force_overwrite": False,
        "resume": None,
        "seed": 0,
        "use_amp": False,
        "log_interval": 50,
        "eval_interval": 1000,
        "save_interval": 2000,
        # Synthetic data fallbacks (config-only).
        "train_episodes": 256,
        "val_episodes": 128,
        "synthetic_num_atoms": (6, 8),
        "synthetic_num_examples": (24, 32),
        "synthetic_max_clauses": 6,
        "synthetic_max_literals": 4,
        "spurious_env_enabled": False,
        "num_spurious_env_features": 0,
        "spurious_env_correlation": 0.7,
        "spurious_env_mode": "sign_flip",
        # Core optimisation defaults (config-only).
        "learning_rate": 3e-4,
        "weight_decay": 1e-2,
        "adam_betas": (0.9, 0.999),
        "grad_accumulation": 1,
        "warmup_steps": 0,
        # Core model defaults (config-only).
        "t_max": 4,
        "k_max": 4,
        "literal_embed_dim": 256,
        "literal_hidden_dim": 256,
        "clause_hidden_dim": 256,
        "gate_mode": "sigmoid",
        "clause_topk": 1,
        "clause_dropout": 0.0,
        "clause_dropout_min_keep": 1,
        # Loss defaults required by downstream components (config-only).
        "coverage_pos_weight": 1.0,
        "coverage_neg_weight": 1.0,
        "margin_pos": 0.7,
        "margin_neg": 0.3,
        "max_margin_coverage_enabled": False,
        "max_margin_coverage_pos": 0.7,
        "max_margin_coverage_neg": 0.3,
        "max_margin_coverage_weight": 1.0,
        "cf_necessity_enabled": False,
        "cf_necessity_weight": 0.1,
        "cf_necessity_spurious_weight": 0.1,
        "cf_necessity_select_threshold": 0.3,
        "cf_necessity_num_samples": 4,
        "cf_necessity_warmup_steps": 1000,
        # Internal bookkeeping.
        "_config": {},
    }

    prelim = argparse.ArgumentParser(add_help=False)
    prelim.add_argument("--config", type=Path, default=None)
    prelim_args, _ = prelim.parse_known_args(argv)

    config_data: Dict[str, Any] = {}
    if prelim_args.config is not None:
        config_data = _load_config_file(prelim_args.config)

    merged_defaults = dict(default_config)
    merged_defaults.update({k: v for k, v in config_data.items() if k != "config"})

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "REQUIRED. Path to a JSON/YAML config file containing all training/model/loss "
            "hyperparameters. Only runtime/operational overrides are accepted on the CLI."
        ),
    )

    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--train-manifest",
        type=Path,
        default=merged_defaults.get("train_manifest"),
        help="Optional override for training manifest path.",
    )
    data_group.add_argument(
        "--val-manifest",
        type=Path,
        default=merged_defaults.get("val_manifest"),
        help="Optional override for validation manifest path.",
    )
    data_group.add_argument(
        "--manifest-format",
        choices=["npz", "synthetic"],
        default=merged_defaults.get("manifest_format", "npz"),
        help="Dataset backend to use (npz or synthetic).",
    )
    data_group.add_argument(
        "--synthetic-only",
        action="store_true",
        default=bool(merged_defaults.get("synthetic_only", False)),
        help="Alias for --manifest-format synthetic (ignores manifest paths).",
    )
    data_group.add_argument(
        "--spurious-env-enabled",
        action=argparse.BooleanOptionalAction,
        default=bool(merged_defaults.get("spurious_env_enabled", False)),
        help="Enable multi-environment spurious correlation features in synthetic episodes.",
    )
    data_group.add_argument(
        "--num-spurious-env-features",
        type=int,
        default=int(merged_defaults.get("num_spurious_env_features", 0)),
        help="Number of spurious features that flip correlation across environments.",
    )
    data_group.add_argument(
        "--spurious-env-correlation",
        type=float,
        default=float(merged_defaults.get("spurious_env_correlation", 0.7)),
        help="Correlation strength for spurious env features in env1.",
    )
    data_group.add_argument(
        "--spurious-env-mode",
        choices=["sign_flip", "disappearing", "strength_variation"],
        default=str(merged_defaults.get("spurious_env_mode", "sign_flip")),
        help="How spurious env feature correlations change in env2.",
    )

    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument(
        "--output-dir",
        type=Path,
        default=merged_defaults.get("output_dir", Path("runs/latest")),
        help="Where to write checkpoints/metrics.",
    )
    runtime_group.add_argument(
        "--force-overwrite",
        action="store_true",
        default=bool(merged_defaults.get("force_overwrite", False)),
        help="Allow overwriting existing checkpoints in the output directory.",
    )
    runtime_group.add_argument(
        "--resume",
        type=Path,
        default=merged_defaults.get("resume", None),
        help="Checkpoint to resume from.",
    )
    runtime_group.add_argument(
        "--seed", type=int, default=int(merged_defaults.get("seed", 0))
    )
    runtime_group.add_argument(
        "--epochs",
        type=int,
        default=int(merged_defaults.get("epochs", 0)),
        help="Number of full passes over the training set.",
    )
    runtime_group.add_argument(
        "--total-steps",
        type=int,
        default=int(merged_defaults.get("total_steps", 0)),
        help="Override total steps for schedulers.",
    )

    loader_group = parser.add_argument_group("dataloader")
    loader_group.add_argument(
        "--batch-size",
        type=int,
        default=int(merged_defaults.get("batch_size", 32)),
        help="Per-device batch size.",
    )
    loader_group.add_argument(
        "--eval-batch-size",
        type=int,
        default=int(merged_defaults.get("eval_batch_size", 32)),
        help="Per-device batch size during evaluation.",
    )
    loader_group.add_argument(
        "--train-num-workers",
        type=int,
        default=int(merged_defaults.get("train_num_workers", 0)),
        help="Number of worker processes for the training DataLoader.",
    )
    loader_group.add_argument(
        "--eval-num-workers",
        type=int,
        default=int(merged_defaults.get("eval_num_workers", 0)),
        help="Number of worker processes for the evaluation DataLoader.",
    )
    loader_group.add_argument(
        "--train-pin-memory",
        action=argparse.BooleanOptionalAction,
        default=bool(merged_defaults.get("train_pin_memory", True)),
        help="Enable pinned memory for the training DataLoader.",
    )
    loader_group.add_argument(
        "--eval-pin-memory",
        action=argparse.BooleanOptionalAction,
        default=bool(merged_defaults.get("eval_pin_memory", True)),
        help="Enable pinned memory for the evaluation DataLoader.",
    )
    loader_group.add_argument(
        "--train-prefetch-factor",
        type=int,
        default=int(merged_defaults.get("train_prefetch_factor", 2)),
        help="Batches prefetched per worker for the training DataLoader (requires workers > 0).",
    )
    loader_group.add_argument(
        "--eval-prefetch-factor",
        type=int,
        default=int(merged_defaults.get("eval_prefetch_factor", 2)),
        help="Batches prefetched per worker for the evaluation DataLoader (requires workers > 0).",
    )

    log_group = parser.add_argument_group("logging")
    log_group.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=bool(merged_defaults.get("use_amp", False)),
        help="Enable mixed precision training.",
    )
    log_group.add_argument(
        "--log-interval",
        type=int,
        default=int(merged_defaults.get("log_interval", 50)),
    )
    log_group.add_argument(
        "--eval-interval",
        type=int,
        default=int(merged_defaults.get("eval_interval", 0) or 0),
    )
    log_group.add_argument(
        "--save-interval",
        type=int,
        default=int(merged_defaults.get("save_interval", 0) or 0),
    )
    model_override_group = parser.add_argument_group("model overrides")
    model_override_group.add_argument(
        "--clause-dropout",
        type=float,
        default=float(merged_defaults.get("clause_dropout", 0.0)),
        help="Clause dropout rate during training (0.0 = disabled)",
    )
    model_override_group.add_argument(
        "--clause-dropout-min-keep",
        type=int,
        default=int(merged_defaults.get("clause_dropout_min_keep", 1)),
        help="Minimum clauses to keep when using clause dropout",
    )

    loss_override_group = parser.add_argument_group("loss overrides")
    loss_override_group.add_argument(
        "--max-margin-coverage-enabled",
        action="store_true",
        default=bool(merged_defaults.get("max_margin_coverage_enabled", False)),
        help="Enable max-margin coverage loss (only penalizes best clause)",
    )
    loss_override_group.add_argument(
        "--max-margin-coverage-pos",
        type=float,
        default=float(merged_defaults.get("max_margin_coverage_pos", 0.7)),
        help="Margin for positive examples in max-margin coverage",
    )
    loss_override_group.add_argument(
        "--max-margin-coverage-neg",
        type=float,
        default=float(merged_defaults.get("max_margin_coverage_neg", 0.3)),
        help="Margin for negative examples in max-margin coverage",
    )
    loss_override_group.add_argument(
        "--max-margin-coverage-weight",
        type=float,
        default=float(merged_defaults.get("max_margin_coverage_weight", 1.0)),
        help="Weight for max-margin coverage loss",
    )
    # Counterfactual necessity arguments
    loss_override_group.add_argument(
        "--cf-necessity-enabled",
        action="store_true",
        default=bool(merged_defaults.get("cf_necessity_enabled", False)),
        help="Enable counterfactual necessity loss",
    )
    loss_override_group.add_argument(
        "--cf-necessity-weight",
        type=float,
        default=float(merged_defaults.get("cf_necessity_weight", 0.1)),
        help="Weight for necessity loss term",
    )
    loss_override_group.add_argument(
        "--cf-necessity-spurious-weight",
        type=float,
        default=float(merged_defaults.get("cf_necessity_spurious_weight", 0.1)),
        help="Weight for spuriousness loss term",
    )
    loss_override_group.add_argument(
        "--cf-necessity-select-threshold",
        type=float,
        default=float(merged_defaults.get("cf_necessity_select_threshold", 0.3)),
        help="Threshold for considering literal as selected",
    )
    loss_override_group.add_argument(
        "--cf-necessity-num-samples",
        type=int,
        default=int(merged_defaults.get("cf_necessity_num_samples", 4)),
        help="Number of counterfactual samples per example",
    )
    loss_override_group.add_argument(
        "--cf-necessity-warmup-steps",
        type=int,
        default=int(merged_defaults.get("cf_necessity_warmup_steps", 1000)),
        help="Steps before enabling counterfactual necessity loss",
    )
    cli_args, unknown = parser.parse_known_args(argv)
    if unknown:
        parser.error(
            "Unrecognized arguments: "
            + " ".join(unknown)
            + "\n\n"
            + "Only runtime/operational flags are supported on the CLI. "
            + "Put all model/loss/optimizer hyperparameters in the --config file."
        )

    merged = dict(merged_defaults)
    merged.update(vars(cli_args))

    args = argparse.Namespace(**merged)
    args._config = config_data

    for field in ("train_manifest", "val_manifest", "output_dir", "resume", "config"):
        if hasattr(args, field):
            coerced = _ensure_path(getattr(args, field))
            setattr(args, field, coerced)

    if args.output_dir is None:
        args.output_dir = Path("runs/latest")

    if getattr(args, "synthetic_only", False):
        args.manifest_format = "synthetic"
        args.train_manifest = None
        args.val_manifest = None

    if args.manifest_format == "npz" and (
        args.train_manifest is None or args.val_manifest is None
    ):
        raise ValueError(
            "train_manifest and val_manifest must be provided (in --config or via CLI overrides) "
            "when manifest_format is 'npz'."
        )

    for attr, raw_value in list(vars(args).items()):
        if attr.endswith("_schedule") and raw_value is not None:
            setattr(args, attr, _prepare_schedule(raw_value))

    return args


def setup_distributed() -> Dict[str, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:  # single-process fallback
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def cleanup_distributed() -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        if dist.get_world_size() > 1:
            dist.barrier()
    except RuntimeError:
        pass
    finally:
        dist.destroy_process_group()


atexit.register(cleanup_distributed)


def seed_everything(seed: int, rank: int) -> None:
    final_seed = seed + rank
    torch.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)


def build_datasets(
    args: argparse.Namespace,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if args.manifest_format == "npz":
        if args.train_manifest is None or args.val_manifest is None:
            raise ValueError(
                "train-manifest and val-manifest are required for npz format"
            )
        train_dataset = NPZEpisodeDataset(args.train_manifest)
        val_dataset = NPZEpisodeDataset(args.val_manifest)
    elif args.manifest_format == "synthetic":
        # Synthetic-only training: use the synthetic_* knobs from config/CLI.
        #
        # This makes it possible to sweep schema size (N), example count (M), and
        # rule complexity (K/L) directly from the CLI/configs.
        num_atoms = getattr(args, "synthetic_num_atoms", (4, 6))
        if isinstance(num_atoms, list):
            num_atoms = (int(num_atoms[0]), int(num_atoms[1]))

        num_examples = getattr(args, "synthetic_num_examples", (16, 24))
        if isinstance(num_examples, list):
            num_examples = (int(num_examples[0]), int(num_examples[1]))

        cfg = SyntheticEpisodeConfig(
            num_atoms=(int(num_atoms[0]), int(num_atoms[1])),
            num_examples=(int(num_examples[0]), int(num_examples[1])),
            num_heads=int(getattr(args, "synthetic_num_heads", 1)),
            max_clauses=int(getattr(args, "synthetic_max_clauses", 3)),
            max_literals=int(getattr(args, "synthetic_max_literals", 3)),
            spurious_env_enabled=bool(getattr(args, "spurious_env_enabled", False)),
            num_spurious_env_features=int(
                getattr(args, "num_spurious_env_features", 0)
            ),
            spurious_env_correlation=float(
                getattr(args, "spurious_env_correlation", 0.7)
            ),
            spurious_env_mode=str(getattr(args, "spurious_env_mode", "sign_flip")),
        )

        train_dataset = SyntheticEpisodeDataset(
            args.train_episodes, cfg, seed=args.seed
        )
        val_dataset = SyntheticEpisodeDataset(
            args.val_episodes, cfg, seed=args.seed + 1
        )
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(f"Unsupported manifest format: {args.manifest_format}")
    return train_dataset, val_dataset


def move_batch_to_device(
    batch: Dict[str, torch.Tensor | None], device: torch.device
) -> Dict[str, torch.Tensor | None]:
    out: Dict[str, torch.Tensor | None] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


# =============================================================================
# TrainingEngine wiring
# =============================================================================


def create_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    """Create RuleInducer model from args."""

    # Build FiLM config if enabled
    literal_film_cfg = None
    if getattr(args, "literal_film_enabled", False):
        literal_film_cfg = LiteralFilmConfig(
            enabled=True,
            mode=getattr(args, "literal_film_mode", "full"),
            beta_init=getattr(args, "literal_film_beta_init", "orthogonal"),
            beta_std=getattr(args, "literal_film_beta_std", 0.5),
            gamma_init="normal",
            gamma_mean=1.0,
            gamma_std=getattr(args, "literal_film_gamma_std", 0.5),
        )

    model = RuleInducer(
        literal_embed_dim=args.literal_embed_dim,
        literal_hidden_dim=args.literal_hidden_dim,
        clause_hidden_dim=args.clause_hidden_dim,
        T_max=args.t_max,
        K_max=args.k_max,
        gate_mode=args.gate_mode,
        clause_topk=args.clause_topk,
        literal_add_posneg_cooc=getattr(args, "literal_add_posneg_cooc", True),
        literal_example_content_keys=getattr(
            args, "literal_example_content_keys", True
        ),
        literal_example_x_bottleneck=getattr(args, "literal_example_x_bottleneck", 64),
        mutual_exclusion_hard=getattr(args, "mutual_exclusion_hard", True),
        clause_dropout=getattr(args, "clause_dropout", 0.0),
        clause_dropout_min_keep=getattr(args, "clause_dropout_min_keep", 1),
        setmatch_hidden_dim=getattr(args, "setmatch_hidden_dim", None),
        setmatch_num_layers=getattr(args, "setmatch_num_layers", 3),
        setmatch_num_heads=getattr(args, "setmatch_num_heads", 4),
        setmatch_dropout=getattr(args, "setmatch_dropout", 0.1),
        literal_film_config=literal_film_cfg,
    ).to(device)

    return model


def create_optimizer(
    model: nn.Module, args: argparse.Namespace, rank: int
) -> torch.optim.Optimizer:
    """Create optimizer over trainable parameters."""

    optimizer_params: Iterable[nn.Parameter] | list[dict[str, object]]
    optimizer_params = [p for p in model.parameters() if p.requires_grad]

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        if frozen_params > 0:
            print(
                f"[Optimizer] Parameters: {trainable_params:,} trainable, {frozen_params:,} frozen",
                flush=True,
            )

    return torch.optim.AdamW(
        optimizer_params,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        betas=tuple(args.adam_betas),
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer, args: argparse.Namespace, total_steps: int
):
    """Create learning rate scheduler."""

    warmup_iters = int(args.warmup_steps) if args.warmup_steps > 0 else int(total_steps)
    start_factor = (
        max(1e-8, 1.0 / max(warmup_iters, 1)) if args.warmup_steps > 0 else 1.0
    )
    return torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=float(start_factor),
        total_iters=int(max(warmup_iters, 1)),
    )


def create_callbacks(
    args: argparse.Namespace,
    rank: int,
    run_dir: Path,
    schedule_manager: Any,
    train_dataset: torch.utils.data.Dataset,
) -> list:
    """Create training callbacks."""

    from rule_inducer.training import (
        SignalHandlerCallback,
        DDPLoggingCallback,
    )
    from rule_inducer.training.state import TrainState

    callbacks: list = [SignalHandlerCallback(rank=rank)]

    # Keep DDPScheduleManager in sync with TrainState schedules.
    class _ScheduleSyncCallback:
        def __init__(self, manager: Any) -> None:
            self.manager = manager
            self._train_dataset = train_dataset

        def _sync(self, state: TrainState) -> None:
            if isinstance(self._train_dataset, SyntheticEpisodeDataset):
                self._train_dataset.set_current_step(int(state.step))

        def on_train_start(self, state: TrainState, **kwargs: Any) -> None:
            del kwargs
            self._sync(state)

        def on_step_start(self, state: TrainState, **kwargs: Any) -> None:
            del kwargs
            self._sync(state)

        def on_step_end(
            self, state: TrainState, metrics: Dict[str, Any], **kwargs: Any
        ) -> None:
            del metrics
            del kwargs
            self._sync(state)

        def on_eval_end(
            self, state: TrainState, eval_output: Any, **kwargs: Any
        ) -> None:
            del eval_output
            del kwargs
            self._sync(state)

        def on_train_end(self, state: TrainState, **kwargs: Any) -> None:
            del kwargs
            self._sync(state)

    callbacks.append(_ScheduleSyncCallback(schedule_manager))

    callbacks.append(
        DDPLoggingCallback(
            log_dir=run_dir, log_interval=int(args.log_interval), rank=rank
        )
    )

    return callbacks


def main() -> None:
    """Main training entry point using TrainingEngine."""

    args = parse_args()
    dist_info = setup_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    local_rank = dist_info["local_rank"]

    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    seed_everything(int(args.seed), int(rank))

    run_dir = (
        Path(args.output_dir)
        if getattr(args, "output_dir", None)
        else Path("runs") / f"run_{args.seed}"
    )
    if rank == 0:
        checkpoint_dir = run_dir / "checkpoints"
        existing_checkpoints: list[Path] = []
        if run_dir.exists() and checkpoint_dir.is_dir():
            existing_checkpoints = sorted(checkpoint_dir.glob("checkpoint*.pt"))
        if existing_checkpoints:
            checkpoint_count = len(existing_checkpoints)
            if args.force_overwrite:
                print(
                    "[Safety] --force-overwrite enabled: "
                    f"found {checkpoint_count} checkpoint file(s) in '{checkpoint_dir}'. "
                    "Proceeding to overwrite existing artifacts.",
                    flush=True,
                )
            else:
                raise RuntimeError(
                    "Refusing to overwrite existing checkpoints in "
                    f"'{checkpoint_dir}'. Found {checkpoint_count} checkpoint file(s). "
                    "Use --force-overwrite to proceed intentionally or choose a different "
                    "--output-dir."
                )
        run_dir.mkdir(parents=True, exist_ok=True)

    # Datasets/loaders
    train_dataset, val_dataset = build_datasets(args)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=int(world_size), rank=int(rank), shuffle=True
    )

    train_loader_kwargs: Dict[str, Any] = {}
    if args.train_num_workers > 0 and args.train_prefetch_factor > 0:
        train_loader_kwargs["prefetch_factor"] = int(args.train_prefetch_factor)

    val_loader_kwargs: Dict[str, Any] = {}
    if args.eval_num_workers > 0 and args.eval_prefetch_factor > 0:
        val_loader_kwargs["prefetch_factor"] = int(args.eval_prefetch_factor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        sampler=train_sampler,
        num_workers=max(0, int(args.train_num_workers)),
        pin_memory=bool(args.train_pin_memory),
        collate_fn=partial(synthetic_episode_collate, padding_literals="none"),
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=max(0, int(args.eval_num_workers)),
        pin_memory=bool(args.eval_pin_memory),
        collate_fn=partial(synthetic_episode_collate, padding_literals="none"),
        **val_loader_kwargs,
    )

    val_loaders = {"val": val_loader}

    # Total optimizer steps.
    steps_per_epoch = len(train_loader) // max(int(args.grad_accumulation), 1)
    total_steps = int(args.total_steps)
    if total_steps <= 0:
        total_steps = int(steps_per_epoch) * int(args.epochs)

    if total_steps <= 0:
        raise ValueError(
            f"Invalid total_steps={total_steps}. Check --total-steps/--epochs and dataset size."
        )

    # Model
    model = create_model(args, device)

    # Warmup pass for lazy params.
    needs_warmup = _has_uninitialized_lazy_params(model)
    if needs_warmup:
        train_sampler.set_epoch(0)
        warmup_iter = iter(train_loader)
        try:
            warmup_batch = next(warmup_iter)
        except StopIteration:
            warmup_batch = None
        if warmup_batch is not None:
            warmup_batch = move_batch_to_device(warmup_batch, device)
            with torch.no_grad():
                model(
                    warmup_batch["X_val"],
                    warmup_batch["X_mask"].float(),
                    warmup_batch["Y_val"],
                    warmup_batch["Y_mask"].float(),
                    warmup_batch["N_len"],
                    warmup_batch["M_len"],
                    warmup_batch["H_len"],
                    gumbel=False,
                )
            model.train()

    # DDP
    if world_size > 1:
        ddp_model: nn.Module = DDP(
            model, device_ids=[local_rank] if device.type == "cuda" else None
        )
    else:
        ddp_model = model

    # Optimizer / scheduler
    optimizer = create_optimizer(ddp_model, args, rank)
    scheduler = create_scheduler(optimizer, args, total_steps)

    from rule_inducer.training import (
        DDPScheduleManager,
        DDPLossComputer,
        DDPEvaluator,
        TrainingEngine,
        TrainingConfig,
        CheckpointManager,
        MetricsLogger,
    )

    schedule_manager = DDPScheduleManager(args, total_steps)
    loss_computer = DDPLossComputer(args, schedule_manager, device, ddp_model)
    evaluator = DDPEvaluator(args, device)

    callbacks = create_callbacks(args, rank, run_dir, schedule_manager, train_dataset)

    config = TrainingConfig(
        num_steps=int(total_steps),
        log_interval=int(args.log_interval),
        eval_interval=int(getattr(args, "eval_interval", 0) or 0),
        checkpoint_interval=int(getattr(args, "save_interval", 0) or 0),
        grad_clip=float(
            getattr(args, "gradient_clip", None)
            or getattr(args, "max_grad_norm", 0.0)
            or 0.0
        ),
        use_amp=bool(getattr(args, "use_amp", False)),
        grad_accumulation=int(getattr(args, "grad_accumulation", 1) or 1),
    )

    checkpoint_manager = CheckpointManager(checkpoint_dir=run_dir / "checkpoints")
    logger = MetricsLogger(
        log_path=run_dir / "engine_metrics.jsonl",
        log_interval=int(args.log_interval),
        console=True,
        console_format="step={step} loss={loss:.6f} lr={lr:.3e}",
    )

    engine = TrainingEngine(
        model=ddp_model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_computer=loss_computer,
        train_loader=train_loader,
        evaluator=evaluator,
        val_loaders=val_loaders,
        config=config,
        callbacks=callbacks,
        checkpoint_manager=checkpoint_manager,
        logger=logger,
        device=device,
        run_config={
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
    )

    resume_path = str(args.resume) if getattr(args, "resume", None) else None

    try:
        final_state = engine.fit(resume_from=resume_path)
        if rank == 0:
            print(f"Training complete! Final step: {final_state.step}", flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
