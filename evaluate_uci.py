#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from rule_inducer import (
    decode_program,
    evaluate_rules_on_examples,
)
from rule_inducer.checkpoint import load_model


def load_uci_dataset(data_dir: Path, name: str) -> Tuple[np.ndarray, np.ndarray]:
    ds_dir = data_dir / name
    X = np.load(ds_dir / "X_bool.npy")
    y = np.load(ds_dir / "y.npy")
    if y.ndim > 1:
        y = np.squeeze(y)
    return X, y


def _sanitize_X(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if np.issubdtype(X.dtype, np.floating):
        nan_mask = np.isnan(X)
        filled = np.where(nan_mask, 0.0, X).astype(np.float32)
        mask = ~nan_mask
        return filled, mask
    filled = X.astype(np.float32)
    mask = np.ones_like(X, dtype=bool)
    return filled, mask


def _create_episode(X: np.ndarray, y: np.ndarray, device: torch.device) -> dict:
    if X.ndim != 2:
        raise ValueError(f"Expected X shape [M,N], got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected y shape [M], got {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatched rows: X has {X.shape[0]}, y has {y.shape[0]}")

    n_samples, n_features = X.shape
    X_filled, X_mask_np = _sanitize_X(X)

    X_val = torch.as_tensor(X_filled, dtype=torch.float32, device=device)
    X_mask = torch.as_tensor(X_mask_np, dtype=torch.bool, device=device)
    Y_val = torch.as_tensor(y.astype(np.float32), device=device).unsqueeze(-1)
    Y_mask = torch.ones_like(Y_val, dtype=torch.bool)

    return {
        "X_val": X_val,
        "X_mask": X_mask,
        "Y_val": Y_val,
        "Y_mask": Y_mask,
        "N_len": torch.tensor([n_features], device=device, dtype=torch.long),
        "M_len": torch.tensor([n_samples], device=device, dtype=torch.long),
        "H_len": torch.tensor([1], device=device, dtype=torch.long),
    }


def _stratified_folds(y: np.ndarray, n_folds: int, seed: int = 42) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds: List[List[int]] = [[] for _ in range(n_folds)]
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        splits = np.array_split(cls_idx, n_folds)
        for fold_idx, split in enumerate(splits):
            folds[fold_idx].extend(split.tolist())
    return [np.array(fold, dtype=np.int64) for fold in folds]


def _create_ovr_tasks(y: np.ndarray) -> List[Tuple[np.ndarray, int]]:
    """Create one-vs-rest binary tasks for multi-class y.

    Returns list of (y_binary, class_idx) tuples where y_binary = (y == class_idx).
    """

    classes = np.unique(y)
    tasks: List[Tuple[np.ndarray, int]] = []
    for cls in classes:
        y_binary = (y == cls).astype(np.int64)
        tasks.append((y_binary, int(cls)))
    return tasks


def _aggregate_ovr_predictions(task_preds: List[np.ndarray]) -> np.ndarray:
    """Aggregate OvR binary predictions into multi-class predictions.

    Strategy: argmax voting. If multiple/none predict 1, use first as tiebreaker.
    """

    n_samples = task_preds[0].shape[0]
    pred_matrix = np.stack(task_preds, axis=1)  # [n_samples, n_classes]

    predictions = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        row = pred_matrix[i]
        ones = np.where(row == 1)[0]
        if len(ones) >= 1:
            predictions[i] = ones[0]
        else:
            predictions[i] = 0
    return predictions


def evaluate_on_dataset(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    n_folds: int = 5,
) -> float:
    if X.ndim != 2:
        raise ValueError(f"Expected X shape [M,N], got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected y shape [M], got {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatched rows: X has {X.shape[0]}, y has {y.shape[0]}")

    classes = np.unique(y)
    n_classes = len(classes)
    if n_classes < 2:
        raise ValueError("Dataset must contain at least 2 classes.")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[c] for c in y], dtype=np.int64)

    is_multiclass = n_classes > 2
    if is_multiclass:
        multiclass_tasks = _create_ovr_tasks(y_idx)

    counts = np.bincount(y_idx)
    min_class = int(counts.min()) if len(counts) else 0
    folds = int(min(n_folds, min_class))
    if folds < 2:
        raise ValueError(
            f"Not enough examples per class for CV: min_class_count={min_class}"
        )

    fold_indices = _stratified_folds(y_idx, folds, seed=42)
    accuracies: List[float] = []

    model.eval()

    for fold_idx in range(folds):
        support_idx = fold_indices[fold_idx]
        query_idx = np.concatenate(
            [fold_indices[i] for i in range(folds) if i != fold_idx]
        )
        if support_idx.size == 0 or query_idx.size == 0:
            raise ValueError("Empty support/query split encountered.")

        X_support = X[support_idx]
        X_query = X[query_idx]
        y_query = y_idx[query_idx]

        X_query_filled, X_query_mask = _sanitize_X(X_query)
        X_query_t = torch.as_tensor(
            X_query_filled, dtype=torch.float32, device=device
        ).unsqueeze(0)
        X_query_mask_t = torch.as_tensor(
            X_query_mask, dtype=torch.bool, device=device
        ).unsqueeze(0)
        M_query_len = torch.tensor(
            [X_query_filled.shape[0]], device=device, dtype=torch.long
        )
        H_len = torch.tensor([1], device=device, dtype=torch.long)

        if is_multiclass:
            task_preds: List[np.ndarray] = []
            for y_bin_full, _class_idx in multiclass_tasks:
                y_support_bin = y_bin_full[support_idx]
                episode = _create_episode(X_support, y_support_bin, device)

                with torch.no_grad():
                    out = model(
                        X_val=episode["X_val"].unsqueeze(0),
                        X_mask=episode["X_mask"].unsqueeze(0),
                        Y_val=episode["Y_val"].unsqueeze(0),
                        Y_mask=episode["Y_mask"].unsqueeze(0),
                        N_len=episode["N_len"],
                        M_len=episode["M_len"],
                        H_len=H_len,
                        gumbel=False,
                    )
                    program = model.export_program(
                        out,
                        N_len=episode["N_len"],
                        H_len=H_len,
                        M_len=episode["M_len"],
                        clause_threshold=0.5,
                        literal_threshold=0.5,
                    )
                    program_cpu: Dict[str, object] = {
                        k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                        for k, v in program.items()
                    }
                    rules = decode_program(
                        program_cpu,
                        episode["N_len"].detach().cpu(),
                        H_len.detach().cpu(),
                    )

                pred_Y = evaluate_rules_on_examples(
                    rules,
                    X_query_t,
                    X_query_mask_t,
                    M_query_len,
                    H_len,
                    nan_handling="fill_half",
                )
                y_pred_bin = (
                    pred_Y[0, : X_query_filled.shape[0], 0].detach().cpu().numpy() > 0.5
                ).astype(np.int64)
                task_preds.append(y_pred_bin)

            y_pred = _aggregate_ovr_predictions(task_preds)
        else:
            y_support = y_idx[support_idx]
            episode = _create_episode(X_support, y_support, device)

            with torch.no_grad():
                out = model(
                    X_val=episode["X_val"].unsqueeze(0),
                    X_mask=episode["X_mask"].unsqueeze(0),
                    Y_val=episode["Y_val"].unsqueeze(0),
                    Y_mask=episode["Y_mask"].unsqueeze(0),
                    N_len=episode["N_len"],
                    M_len=episode["M_len"],
                    H_len=H_len,
                    gumbel=False,
                )
                program = model.export_program(
                    out,
                    N_len=episode["N_len"],
                    H_len=H_len,
                    M_len=episode["M_len"],
                    clause_threshold=0.5,
                    literal_threshold=0.5,
                )
                program_cpu: Dict[str, object] = {
                    k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                    for k, v in program.items()
                }
                rules = decode_program(
                    program_cpu,
                    episode["N_len"].detach().cpu(),
                    H_len.detach().cpu(),
                )

            pred_Y = evaluate_rules_on_examples(
                rules,
                X_query_t,
                X_query_mask_t,
                M_query_len,
                H_len,
                nan_handling="fill_half",
            )
            y_pred = (
                pred_Y[0, : X_query_filled.shape[0], 0].detach().cpu().numpy() > 0.5
            ).astype(np.int64)

        fold_acc = float((y_pred == y_query).mean())
        accuracies.append(fold_acc)
        sample_k = min(5, y_query.shape[0])
        print(
            f"fold {fold_idx + 1}/{folds} acc={fold_acc:.4f} "
            f"sample y={y_query[:sample_k].tolist()} yhat={y_pred[:sample_k].tolist()}",
            flush=True,
        )

    return float(np.mean(accuracies))


def _list_datasets(data_dir: Path) -> List[str]:
    datasets = []
    for child in data_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / "X_bool.npy").exists() and (child / "y.npy").exists():
            datasets.append(child.name)
    return sorted(datasets)


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal UCI evaluation script.")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a RuleInducer checkpoint."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory containing UCI dataset subdirectories.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Evaluate all datasets.")
    group.add_argument("--dataset", type=str, help="Evaluate a single dataset.")

    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {args.data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    if args.all:
        datasets = _list_datasets(args.data_dir)
        if not datasets:
            raise FileNotFoundError(
                f"No datasets found in {args.data_dir} (expected X_bool.npy + y.npy)."
            )
    else:
        datasets = [args.dataset]

    all_accs: List[float] = []
    for name in datasets:
        ds_dir = args.data_dir / name
        if not ds_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {ds_dir}")
        X, y = load_uci_dataset(args.data_dir, name)
        acc = evaluate_on_dataset(model, X, y, device=device, n_folds=5)
        all_accs.append(acc)
        print(f"{name}: mean_accuracy={acc:.4f}", flush=True)

    if all_accs:
        mean_all = float(np.mean(all_accs))
        print(f"overall_mean_accuracy={mean_all:.4f}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
