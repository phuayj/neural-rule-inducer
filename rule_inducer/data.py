from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, TypedDict

import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    np = None  # type: ignore

Literal = Tuple[int, bool]
Clause = List[Literal]


class Episode(TypedDict):
    """Single episode payload."""

    X_val: Tensor
    X_mask: Tensor
    Y_val: Tensor
    Y_mask: Tensor
    rules: Optional[List[List[Clause]]]


class CollatedEpisodeBatch(TypedDict):
    """Batch produced by :func:`synthetic_episode_collate`."""

    X_val: Tensor
    X_mask: Tensor
    Y_val: Tensor
    Y_mask: Tensor
    N_len: Tensor
    M_len: Tensor
    H_len: Tensor
    rules: Optional[List[Optional[List[List[Clause]]]]]


@dataclass
class SyntheticEpisodeConfig:
    """Configuration for synthetic episode generation."""

    num_atoms: Tuple[int, int] = (4, 6)
    num_examples: Tuple[int, int] = (16, 24)
    num_heads: int = 1
    max_clauses: int = 3
    max_literals: int = 3
    spurious_env_enabled: bool = False
    num_spurious_env_features: int = 0
    spurious_env_correlation: float = 0.7
    spurious_env_mode: str = "sign_flip"


def _sample_literal_pool(
    num_atoms: int, clause_size: int, rng: random.Random
) -> Clause:
    atoms = rng.sample(range(num_atoms), k=clause_size)
    clause: Clause = []
    for atom in atoms:
        clause.append((atom, rng.random() < 0.5))
    return clause


def _sample_rule(
    num_atoms: int, max_clauses: int, max_literals: int, rng: random.Random
) -> List[Clause]:
    num_clauses = rng.randint(1, max(1, max_clauses))
    rule: List[Clause] = []
    for _ in range(num_clauses):
        clause_size = rng.randint(1, max(1, max_literals))
        rule.append(_sample_literal_pool(num_atoms, clause_size, rng))
    return rule


def _evaluate_clause(clause: Clause, assignment: Tensor) -> Tensor:
    literals = []
    for atom_idx, is_positive in clause:
        lit_val = assignment[..., atom_idx]
        literals.append(lit_val if is_positive else 1.0 - lit_val)
    return torch.stack(literals, dim=-1).prod(dim=-1)


def _evaluate_rule(rule: List[Clause], assignment: Tensor) -> Tensor:
    clause_truths = [_evaluate_clause(clause, assignment) for clause in rule]
    stacked = torch.stack(clause_truths, dim=-1)
    return 1.0 - torch.prod(1.0 - stacked, dim=-1)


class SyntheticEpisodeDataset(Dataset):
    """Generates synthetic DNF episodes with optional spurious environments."""

    def __init__(
        self,
        num_episodes: int,
        config: SyntheticEpisodeConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.num_episodes = int(num_episodes)
        self.config = config or SyntheticEpisodeConfig()
        self._base_seed = int(seed) if seed is not None else 0
        self._rng = random.Random(seed)
        self.current_step: Optional[int] = None

    def __len__(self) -> int:
        return self.num_episodes

    def set_current_step(self, step: int) -> None:
        self.current_step = int(step)

    def _resolve_current_step(self, current_step: Optional[int]) -> Optional[int]:
        if current_step is not None:
            return int(current_step)
        if self.current_step is not None:
            return int(self.current_step)
        return None

    def _gen_episode(
        self, *, torch_gen: torch.Generator, current_step: Optional[int] = None
    ) -> Episode:
        cfg = self.config
        current_step = self._resolve_current_step(current_step)

        base_num_atoms = self._rng.randint(cfg.num_atoms[0], cfg.num_atoms[1])
        num_examples = self._rng.randint(cfg.num_examples[0], cfg.num_examples[1])

        rules = [
            _sample_rule(
                base_num_atoms, cfg.max_clauses, cfg.max_literals, rng=self._rng
            )
            for _ in range(cfg.num_heads)
        ]

        X_val = torch.randint(
            0,
            2,
            (num_examples, base_num_atoms),
            dtype=torch.float32,
            generator=torch_gen,
        )
        X_mask = torch.ones_like(X_val, dtype=torch.bool)

        Y_val = torch.zeros(num_examples, cfg.num_heads, dtype=torch.float32)
        for h_idx, rule in enumerate(rules):
            Y_val[:, h_idx] = _evaluate_rule(rule, X_val)

        if cfg.spurious_env_enabled and cfg.num_spurious_env_features > 0:
            y_binary = (
                Y_val[:, 0]
                if Y_val.shape[1] > 0
                else torch.zeros(num_examples, device=Y_val.device)
            )
            rho = float(cfg.spurious_env_correlation)
            split_idx = num_examples // 2
            env2_size = num_examples - split_idx
            for _ in range(int(cfg.num_spurious_env_features)):
                rand_vals = torch.rand(num_examples, generator=torch_gen)
                env1_threshold = torch.where(
                    y_binary[:split_idx] > 0.5,
                    torch.full((split_idx,), rho),
                    torch.full((split_idx,), 1.0 - rho),
                )
                spurious = torch.zeros(num_examples)
                spurious[:split_idx] = (rand_vals[:split_idx] < env1_threshold).float()

                env2_y = y_binary[split_idx:]
                if cfg.spurious_env_mode == "disappearing":
                    env2_threshold = torch.full((env2_size,), 0.5)
                elif cfg.spurious_env_mode == "strength_variation":
                    weak_rho = 0.1
                    env2_threshold = torch.where(
                        env2_y > 0.5,
                        torch.full((env2_size,), 0.5 + weak_rho),
                        torch.full((env2_size,), 0.5 - weak_rho),
                    )
                else:
                    env2_threshold = torch.where(
                        env2_y > 0.5,
                        torch.full((env2_size,), 1.0 - rho),
                        torch.full((env2_size,), rho),
                    )
                spurious[split_idx:] = (rand_vals[split_idx:] < env2_threshold).float()

                X_val = torch.cat([X_val, spurious.unsqueeze(1)], dim=1)
                X_mask = torch.cat([X_mask, X_mask.new_ones(num_examples, 1)], dim=1)

            perm = torch.randperm(num_examples, generator=torch_gen)
            X_val = X_val[perm]
            X_mask = X_mask[perm]
            Y_val = Y_val[perm]

        Y_mask = torch.ones_like(Y_val, dtype=torch.bool)

        return {
            "X_val": X_val,
            "X_mask": X_mask,
            "Y_val": Y_val,
            "Y_mask": Y_mask,
            "rules": rules,
        }

    def __getitem__(self, index: int) -> Episode:
        idx = int(index)
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"index {index} out of range for SyntheticEpisodeDataset of length {len(self)}"
            )
        episode_seed = hash((self._base_seed, idx)) & 0xFFFFFFFF
        old_state = self._rng.getstate()
        self._rng.seed(episode_seed)
        torch_gen = torch.Generator()
        torch_gen.manual_seed(episode_seed)
        try:
            return self._gen_episode(torch_gen=torch_gen)
        finally:
            self._rng.setstate(old_state)


class NPZEpisodeDataset(Dataset):
    """Loads episodes stored as NPZ files listed in a manifest."""

    def __init__(self, manifest_path: str | Path) -> None:
        if np is None:
            raise ImportError("numpy is required to use NPZEpisodeDataset")
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            self._paths = [Path(line.strip()) for line in handle if line.strip()]
        if not self._paths:
            raise ValueError(f"Manifest {manifest_path} is empty")

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> Episode:
        path = self._paths[index]
        if not path.exists():
            raise FileNotFoundError(f"Episode file not found: {path}")
        data = np.load(path, allow_pickle=True)

        def _to_tensor(key: str, dtype: torch.dtype) -> Tensor:
            if key not in data:
                raise KeyError(f"Key {key} missing from episode file {path}")
            return torch.as_tensor(data[key], dtype=dtype)

        X_val = _to_tensor("X_val", torch.float32)
        X_mask = _to_tensor("X_mask", torch.bool)
        Y_val = _to_tensor("Y_val", torch.float32)
        Y_mask = _to_tensor("Y_mask", torch.bool)
        rules = data["rules"].tolist() if "rules" in data else None

        return {
            "X_val": X_val,
            "X_mask": X_mask,
            "Y_val": Y_val,
            "Y_mask": Y_mask,
            "rules": rules,
        }


def synthetic_episode_collate(
    batch: Sequence[Episode],
    padding_literals: str,
) -> CollatedEpisodeBatch:
    """Pad a list of episodes into dense tensors."""

    if padding_literals not in ("none", "true", "both"):
        raise ValueError(
            "padding_literals must be one of {'none','true','both'}, "
            f"got {padding_literals!r}"
        )

    padding_values: List[float] = []
    if padding_literals in ("true", "both"):
        padding_values.append(1.0)
    if padding_literals == "both":
        padding_values.append(0.0)
    padding_count = len(padding_values)

    batch_size = len(batch)
    num_examples = [item["X_val"].shape[0] for item in batch]
    num_atoms = [item["X_val"].shape[1] for item in batch]
    num_heads = [item["Y_val"].shape[1] for item in batch]

    M_max = max(num_examples)
    N_max = max(num_atoms) + padding_count
    H_max = max(num_heads)

    X_val = torch.zeros(batch_size, M_max, N_max, dtype=torch.float32)
    X_mask = torch.zeros(batch_size, M_max, N_max, dtype=torch.bool)
    Y_val = torch.zeros(batch_size, M_max, H_max, dtype=torch.float32)
    Y_mask = torch.zeros(batch_size, M_max, H_max, dtype=torch.bool)

    N_len = torch.zeros(batch_size, dtype=torch.long)
    M_len = torch.zeros(batch_size, dtype=torch.long)
    H_len = torch.zeros(batch_size, dtype=torch.long)

    rules_entries: List[Optional[List[List[Clause]]]] = []
    has_rules = False

    for b_idx, item in enumerate(batch):
        m_len = int(item["X_val"].shape[0])
        n_len = int(item["X_val"].shape[1])
        h_len = int(item["Y_val"].shape[1])

        M_len[b_idx] = m_len
        N_len[b_idx] = n_len + padding_count
        H_len[b_idx] = h_len

        X_val[b_idx, :m_len, :n_len] = item["X_val"]
        X_mask[b_idx, :m_len, :n_len] = item["X_mask"]
        Y_val[b_idx, :m_len, :h_len] = item["Y_val"]
        Y_mask[b_idx, :m_len, :h_len] = item["Y_mask"]

        if padding_count:
            for offset, pad_value in enumerate(padding_values):
                col_idx = n_len + offset
                X_val[b_idx, :m_len, col_idx] = pad_value
                X_mask[b_idx, :m_len, col_idx] = True

        rule_item = item.get("rules")
        if rule_item is not None:
            has_rules = True
            rules_entries.append(rule_item)
        else:
            rules_entries.append(None)

    rules_payload = rules_entries if has_rules else None

    return {
        "X_val": X_val,
        "X_mask": X_mask,
        "Y_val": Y_val,
        "Y_mask": Y_mask,
        "N_len": N_len,
        "M_len": M_len,
        "H_len": H_len,
        "rules": rules_payload,
    }
