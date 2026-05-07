#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rule_inducer._hub import materialize_rule_inducer_for_hub
from rule_inducer.checkpoint import load_model


def _run_observable_dummy_forward(model: torch.nn.Module) -> None:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    x_val = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], device=device, dtype=dtype)
    x_mask = torch.ones_like(x_val, dtype=torch.bool)
    y_val = torch.tensor([[[1.0], [0.0]]], device=device, dtype=dtype)
    y_mask = torch.ones_like(y_val, dtype=torch.bool)
    n_len = torch.tensor([2], device=device, dtype=torch.long)
    m_len = torch.tensor([2], device=device, dtype=torch.long)
    h_len = torch.tensor([1], device=device, dtype=torch.long)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(
            X_val=x_val,
            X_mask=x_mask,
            Y_val=y_val,
            Y_mask=y_mask,
            N_len=n_len,
            M_len=m_len,
            H_len=h_len,
        )
    if was_training:
        model.train()

    r_pred = output.R_pred.detach().float().cpu()
    print(
        "Dummy forward: "
        f"shape={tuple(r_pred.shape)}, "
        f"mean_prediction={r_pred.mean().item():.6f}, "
        f"min_prediction={r_pred.min().item():.6f}, "
        f"max_prediction={r_pred.max().item():.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push a public Neural Rule Inducer checkpoint to Hugging Face Hub."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the .pt checkpoint.")
    parser.add_argument(
        "--repo-id",
        default="phuayj/neural-rule-inducer",
        help="Target Hugging Face Hub model repository.",
    )
    parser.add_argument(
        "--private", action="store_true", help="Create/update the Hub repository as private."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only save the Hub-ready model locally under ./tmp; do not upload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cpu")
    model = load_model(str(checkpoint_path), device=device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Model parameters: {param_count:,}")

    _run_observable_dummy_forward(model)
    materialize_rule_inducer_for_hub(model)

    if args.dry_run:
        tmp_root = Path("tmp")
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="hf_dry_run_", dir=tmp_root) as tmp_dir:
            model.save_pretrained(tmp_dir)
            saved_files = sorted(path.name for path in Path(tmp_dir).iterdir())
            print(f"Dry run saved Hub files to temporary directory: {tmp_dir}")
            print(f"Dry run files: {saved_files}")
        print("Dry run complete; no Hugging Face upload was performed.")
        return

    hub_commit_url = model.push_to_hub(args.repo_id, private=args.private)
    api = HfApi()
    checkpoint_commit = api.upload_file(
        repo_id=args.repo_id,
        path_or_fileobj=str(checkpoint_path),
        path_in_repo="checkpoint_best.pt",
        repo_type="model",
    )
    print(f"Model weights/config pushed: {hub_commit_url}")
    print(f"Original checkpoint uploaded: {checkpoint_commit.commit_url}")
    print(f"Hub URL: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
