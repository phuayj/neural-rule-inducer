from __future__ import annotations

from dataclasses import dataclass, asdict
from numbers import Number
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import Tensor

from .data import Clause, Literal

Rule = List[Clause]


def decode_program(
    program: Dict[str, object],
    N_len: Tensor,
    H_len: Tensor,
) -> List[List[Rule]]:
    """
    Converts exported tensors into symbolic rules.

    Returns nested lists indexed as [batch][head][clause][literal].
    """

    ragged_literals = program.get("Rule_literals_ragged")  # type: ignore[assignment]
    if ragged_literals is not None:
        rules: List[List[Rule]] = []
        n_counts = N_len.tolist()
        h_counts = H_len.tolist()
        for b_idx, head_payload in enumerate(ragged_literals):  # type: ignore[enumerate]
            n_atoms = int(n_counts[b_idx])
            sample_rules: List[Rule] = []
            head_total = int(h_counts[b_idx])
            for h_idx in range(head_total):
                clause_payload: List[List[int]]
                if h_idx < len(head_payload):  # type: ignore[arg-type]
                    clause_payload = head_payload[h_idx]  # type: ignore[index]
                else:
                    clause_payload = []
                clauses: Rule = []
                for literal_indices in clause_payload:
                    literals: Set[Literal] = set()
                    for raw_idx in literal_indices:
                        idx = int(raw_idx)
                        if idx < 0:
                            continue
                        if idx < n_atoms:
                            literal: Literal = (idx, True)
                        elif idx < 2 * n_atoms:
                            literal = (idx - n_atoms, False)
                        else:
                            continue
                        literals.add(literal)
                    if literals:
                        clause_sorted = sorted(literals, key=lambda x: (x[0], not x[1]))
                        clauses.append(list(clause_sorted))
                sample_rules.append(clauses)
            rules.append(sample_rules)
        return rules

    rule_lit_index = program["Rule_lit_index"]
    clause_active = program["Rule_clause_active"]
    literal_active = program.get("Rule_lit_active")

    B, H_max, T_max, K_max = rule_lit_index.shape

    rules: List[List[Rule]] = []

    for b in range(B):
        sample_rules: List[Rule] = []
        n_atoms = int(N_len[b])
        h_count = int(H_len[b])

        for h in range(h_count):
            clauses: Rule = []
            for t in range(T_max):
                if not bool(clause_active[b, h, t]):
                    continue

                literals: Set[Literal] = set()
                for k in range(K_max):
                    idx = int(rule_lit_index[b, h, t, k])
                    if idx < 0:
                        continue
                    if literal_active is not None and not bool(
                        literal_active[b, h, t, k]
                    ):
                        continue

                    if idx < n_atoms:
                        literal: Literal = (idx, True)
                    elif idx < 2 * n_atoms:
                        literal = (idx - n_atoms, False)
                    else:
                        continue

                    literals.add(literal)

                if literals:
                    clause_sorted = sorted(literals, key=lambda x: (x[0], not x[1]))
                    clauses.append(list(clause_sorted))

            sample_rules.append(clauses)

        rules.append(sample_rules)

    return rules


def _literal_signature(literal: object) -> Optional[Tuple[int, bool]]:
    candidate = literal

    if isinstance(candidate, torch.Tensor):
        if candidate.numel() == 0:
            return None
        if candidate.ndim == 0:
            candidate = candidate.item()
        elif candidate.ndim == 1:
            candidate = candidate.tolist()
        else:
            return None

    if isinstance(candidate, tuple):
        if len(candidate) < 2:
            return None
        atom_raw, polarity_raw = candidate[0], candidate[1]
    elif isinstance(candidate, list):
        if len(candidate) < 2:
            if len(candidate) == 1:
                atom_raw, polarity_raw = candidate[0], True
            else:
                return None
        else:
            atom_raw, polarity_raw = candidate[0], candidate[1]
    elif isinstance(candidate, dict):
        if "atom" in candidate:
            atom_raw = candidate["atom"]
        elif "index" in candidate:
            atom_raw = candidate["index"]
        else:
            return None
        if "polarity" in candidate:
            polarity_raw = candidate["polarity"]
        elif "positive" in candidate:
            polarity_raw = candidate["positive"]
        elif "sign" in candidate:
            polarity_raw = candidate["sign"]
        else:
            polarity_raw = True
    elif isinstance(candidate, Number):
        atom_raw, polarity_raw = candidate, True
    else:
        return None

    try:
        atom_idx = int(atom_raw)
    except (TypeError, ValueError):
        return None

    if isinstance(polarity_raw, Number):
        polarity_bool = bool(int(polarity_raw))
    elif isinstance(polarity_raw, str):
        polarity_bool = polarity_raw not in {
            "-",
            "neg",
            "negative",
            "0",
            "false",
            "False",
        }
    else:
        polarity_bool = bool(polarity_raw)

    return atom_idx, polarity_bool


def _clause_signature(clause: Clause) -> frozenset:
    literals = []
    for lit in clause:
        parsed = _literal_signature(lit)
        if parsed is not None:
            literals.append(parsed)
    return frozenset(literals)


def compute_exact_match(
    predicted_rules: Sequence[Sequence[Rule]],
    target_rules: Sequence[Optional[Sequence[Rule]]],
) -> Tuple[int, int]:
    """Compute exact match between predicted and target DNF rules.

    Rules are compared up to permutation of clauses and literals.

    Returns:
        Tuple ``(num_matches, num_total)`` where ``num_total`` counts samples with
        non-None targets.

    Notes:
        - A sample counts as a match only if *all* heads match.
    """

    num_matches = 0
    num_total = 0

    for pred_heads, tgt_heads in zip(predicted_rules, target_rules):
        if tgt_heads is None:
            continue
        num_total += 1

        is_match = True
        for h_idx, tgt_rule in enumerate(tgt_heads):
            pred_rule = pred_heads[h_idx] if h_idx < len(pred_heads) else []

            pred_canonical = frozenset(_clause_signature(c) for c in pred_rule if c)
            tgt_canonical = frozenset(_clause_signature(c) for c in tgt_rule if c)

            if pred_canonical != tgt_canonical:
                is_match = False
                break

        if is_match:
            num_matches += 1

    return num_matches, num_total


def _recall_first_clause_matching(
    pred_clauses: List[Set[Tuple[int, bool]]],
    tgt_clauses: List[Set[Tuple[int, bool]]],
    *,
    eps: float = 1e-6,
) -> List[Tuple[int, int, Tuple[float, float, float]]]:
    """
    Greedy matching that prefers clauses maximising recall, then F1, then precision.

    Returns a list of tuples ``(pred_index, target_index, (recall, f1, precision))``.
    """

    matches: List[Tuple[int, int, Tuple[float, float, float]]] = []
    if not pred_clauses or not tgt_clauses:
        return matches

    available = set(range(len(pred_clauses)))

    for tgt_idx, tgt in enumerate(tgt_clauses):
        tgt_len = max(len(tgt), 1)
        best_choice: Optional[Tuple[int, Tuple[float, float, float]]] = None
        for pred_idx in list(available):
            pred = pred_clauses[pred_idx]
            pred_len = max(len(pred), 1)
            overlap = len(pred & tgt)
            if overlap == 0:
                continue
            recall = overlap / tgt_len
            precision = overlap / pred_len
            f1 = (2.0 * precision * recall) / (precision + recall + eps)
            score = (recall, f1, precision)
            if best_choice is None:
                best_choice = (pred_idx, score)
                continue
            _, best_score = best_choice
            if score > best_score:
                best_choice = (pred_idx, score)
        if best_choice is not None:
            pred_idx, score = best_choice
            available.remove(pred_idx)
            matches.append((pred_idx, tgt_idx, score))
            if not available:
                break

    return matches


@dataclass
class RuleMatchStats:
    clause_matched: int = 0
    clause_pred_total: int = 0
    clause_target_total: int = 0
    literal_matched: int = 0
    literal_pred_total: int = 0
    literal_target_total: int = 0
    pos_literal_matched: int = 0
    pos_literal_pred_total: int = 0
    pos_literal_target_total: int = 0
    neg_literal_matched: int = 0
    neg_literal_pred_total: int = 0
    neg_literal_target_total: int = 0

    def update(
        self, predicted: Sequence[Sequence[Rule]], target: Sequence[Sequence[Rule]]
    ) -> None:
        for pred_rules, tgt_rules in zip(predicted, target):
            for pred_rule, tgt_rule in zip(pred_rules, tgt_rules):
                pred_clause_list = [set(_clause_signature(c)) for c in pred_rule if c]
                tgt_clause_list = [set(_clause_signature(c)) for c in tgt_rule if c]

                self.clause_pred_total += len(pred_clause_list)
                self.clause_target_total += len(tgt_clause_list)

                matches = _recall_first_clause_matching(
                    pred_clause_list, tgt_clause_list
                )
                self.clause_matched += len(matches)

                pred_literals = (
                    set().union(*pred_clause_list) if pred_clause_list else set()
                )
                tgt_literals = (
                    set().union(*tgt_clause_list) if tgt_clause_list else set()
                )

                self.literal_pred_total += len(pred_literals)
                self.literal_target_total += len(tgt_literals)

                pred_pos = {lit for lit in pred_literals if lit[1]}
                pred_neg = {lit for lit in pred_literals if not lit[1]}
                tgt_pos = {lit for lit in tgt_literals if lit[1]}
                tgt_neg = {lit for lit in tgt_literals if not lit[1]}

                self.pos_literal_pred_total += len(pred_pos)
                self.pos_literal_target_total += len(tgt_pos)
                self.neg_literal_pred_total += len(pred_neg)
                self.neg_literal_target_total += len(tgt_neg)

                literal_overlap_total = 0
                pos_overlap_total = 0
                neg_overlap_total = 0
                for pred_idx, tgt_idx, _ in matches:
                    overlap = pred_clause_list[pred_idx] & tgt_clause_list[tgt_idx]
                    literal_overlap_total += len(overlap)
                    pos_overlap_total += sum(1 for lit in overlap if lit[1])
                    neg_overlap_total += sum(1 for lit in overlap if not lit[1])

                self.literal_matched += literal_overlap_total
                self.pos_literal_matched += pos_overlap_total
                self.neg_literal_matched += neg_overlap_total

    def merge(self, other: "RuleMatchStats") -> None:
        self.clause_matched += other.clause_matched
        self.clause_pred_total += other.clause_pred_total
        self.clause_target_total += other.clause_target_total
        self.literal_matched += other.literal_matched
        self.literal_pred_total += other.literal_pred_total
        self.literal_target_total += other.literal_target_total
        self.pos_literal_matched += other.pos_literal_matched
        self.pos_literal_pred_total += other.pos_literal_pred_total
        self.pos_literal_target_total += other.pos_literal_target_total
        self.neg_literal_matched += other.neg_literal_matched
        self.neg_literal_pred_total += other.neg_literal_pred_total
        self.neg_literal_target_total += other.neg_literal_target_total

    def to_metrics(self) -> Dict[str, float]:
        clause_precision = (
            self.clause_matched / self.clause_pred_total
            if self.clause_pred_total
            else 0.0
        )
        clause_recall = (
            self.clause_matched / self.clause_target_total
            if self.clause_target_total
            else 0.0
        )
        clause_f1 = (
            2 * clause_precision * clause_recall / (clause_precision + clause_recall)
            if clause_precision + clause_recall
            else 0.0
        )

        literal_precision = (
            self.literal_matched / self.literal_pred_total
            if self.literal_pred_total
            else 0.0
        )
        literal_recall = (
            self.literal_matched / self.literal_target_total
            if self.literal_target_total
            else 0.0
        )
        literal_f1 = (
            2
            * literal_precision
            * literal_recall
            / (literal_precision + literal_recall)
            if literal_precision + literal_recall
            else 0.0
        )

        pos_prec = (
            self.pos_literal_matched / self.pos_literal_pred_total
            if self.pos_literal_pred_total
            else 0.0
        )
        pos_rec = (
            self.pos_literal_matched / self.pos_literal_target_total
            if self.pos_literal_target_total
            else 0.0
        )
        pos_f1 = (
            2 * pos_prec * pos_rec / (pos_prec + pos_rec) if pos_prec + pos_rec else 0.0
        )

        neg_prec = (
            self.neg_literal_matched / self.neg_literal_pred_total
            if self.neg_literal_pred_total
            else 0.0
        )
        neg_rec = (
            self.neg_literal_matched / self.neg_literal_target_total
            if self.neg_literal_target_total
            else 0.0
        )
        neg_f1 = (
            2 * neg_prec * neg_rec / (neg_prec + neg_rec) if neg_prec + neg_rec else 0.0
        )

        return {
            "clause_precision": clause_precision,
            "clause_recall": clause_recall,
            "clause_f1": clause_f1,
            "literal_precision": literal_precision,
            "literal_recall": literal_recall,
            "literal_f1": literal_f1,
            "pos_literal_precision": pos_prec,
            "pos_literal_recall": pos_rec,
            "pos_literal_f1": pos_f1,
            "neg_literal_precision": neg_prec,
            "neg_literal_recall": neg_rec,
            "neg_literal_f1": neg_f1,
        }

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)


def rule_precision_recall(
    predicted: Sequence[Sequence[Rule]],
    target: Sequence[Sequence[Rule]],
) -> Dict[str, float]:
    stats = RuleMatchStats()
    stats.update(predicted, target)
    return stats.to_metrics()


def evaluate_rules_on_examples(
    rules: Sequence[Sequence[Rule]],
    X_val: Tensor,
    X_mask: Tensor,
    M_len: Tensor,
    H_len: Tensor,
    nan_handling: str = "fill_half",
) -> Tensor:
    """
    Evaluates symbolic rules on feature assignments.

    Args:
        rules: Nested list [batch][head][clause][literal].
        X_val: [B, M_max, N_max]
        X_mask: [B, M_max, N_max]
        M_len/H_len: lengths for examples and heads.
        nan_handling: How to handle unknowns ("fill_half" or "skip").

    Returns:
        Tensor [B, M_max, H_max] with rule truth values.
    """

    if nan_handling not in ("fill_half", "skip"):
        raise ValueError(
            "nan_handling must be 'fill_half' or 'skip', got " f"{nan_handling!r}."
        )

    B, M_max, _ = X_val.shape
    H_max = int(H_len.max().item()) if len(H_len) > 0 else 0

    outputs = torch.zeros(B, M_max, H_max, dtype=X_val.dtype, device=X_val.device)

    for b_idx, rule_heads in enumerate(rules):
        m_len = int(M_len[b_idx])
        h_len = int(H_len[b_idx])
        values = X_val[b_idx, :m_len]
        masks = X_mask[b_idx, :m_len]
        padded_values = torch.where(masks, values, torch.full_like(values, 0.5))

        ones = torch.ones(m_len, dtype=values.dtype, device=values.device)
        zeros = torch.zeros(m_len, dtype=values.dtype, device=values.device)

        for h_idx in range(h_len):
            head_rule: Rule = rule_heads[h_idx] if h_idx < len(rule_heads) else []
            if not head_rule:
                continue

            clause_scores = []
            for clause in head_rule:
                if not clause:
                    clause_scores.append(ones)
                    continue

                if nan_handling == "fill_half":
                    literal_scores = []
                    for atom_idx, is_positive in clause:
                        atom_idx = int(atom_idx)
                        if atom_idx >= values.shape[1]:
                            continue
                        literal_val = padded_values[:, atom_idx]
                        literal_scores.append(
                            literal_val if is_positive else 1.0 - literal_val
                        )

                    if literal_scores:
                        clause_scores.append(
                            torch.stack(literal_scores, dim=-1).prod(dim=-1)
                        )
                    else:
                        clause_scores.append(zeros)
                else:
                    literal_scores = []
                    literal_known = []
                    for atom_idx, is_positive in clause:
                        atom_idx = int(atom_idx)
                        if atom_idx >= values.shape[1]:
                            continue
                        literal_val = values[:, atom_idx]
                        literal_mask = masks[:, atom_idx]
                        if not is_positive:
                            literal_val = 1.0 - literal_val
                        literal_scores.append(
                            torch.where(
                                literal_mask,
                                literal_val,
                                torch.ones_like(literal_val),
                            )
                        )
                        literal_known.append(literal_mask)

                    if literal_scores:
                        literal_tensor = torch.stack(literal_scores, dim=-1)
                        known_tensor = torch.stack(literal_known, dim=-1)
                        clause_val = literal_tensor.prod(dim=-1)
                        any_known = known_tensor.any(dim=-1)
                        clause_val = torch.where(any_known, clause_val, zeros)
                        clause_scores.append(clause_val)
                    else:
                        clause_scores.append(zeros)

            if clause_scores:
                clause_tensor = torch.stack(clause_scores, dim=-1)
                truth = 1.0 - torch.prod(1.0 - clause_tensor, dim=-1)
                outputs[b_idx, :m_len, h_idx] = truth

    return outputs
