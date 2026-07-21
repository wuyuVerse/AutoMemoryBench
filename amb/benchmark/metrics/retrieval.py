"""Retrieval metrics."""

from __future__ import annotations

from amb.benchmark.metrics.classification import safe_div
import math


def recall_at_k(gold_ids: set[str], ranked_ids: list[str], k: int) -> float:
    if not gold_ids:
        return 1.0 if not ranked_ids else 0.0
    return safe_div(len(gold_ids & set(ranked_ids[:k])), len(gold_ids))


def precision_at_k(gold_ids: set[str], ranked_ids: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    if not ranked_ids[:k]:
        return 1.0 if not gold_ids else 0.0
    return safe_div(len(gold_ids & set(ranked_ids[:k])), len(ranked_ids[:k]))


def mean_reciprocal_rank(gold_ids: set[str], ranked_ids: list[str]) -> float:
    if not gold_ids:
        return 1.0 if not ranked_ids else 0.0
    for index, memory_id in enumerate(ranked_ids, start=1):
        if memory_id in gold_ids:
            return 1.0 / index
    return 0.0


def evidence_complete(gold_ids: set[str], ranked_ids: list[str]) -> float:
    if not gold_ids:
        return 1.0 if not ranked_ids else 0.0
    return 1.0 if gold_ids.issubset(set(ranked_ids)) else 0.0


def ndcg_at_k(gold_ids: set[str], ranked_ids: list[str], k: int) -> float:
    if not gold_ids:
        return 1.0 if not ranked_ids else 0.0
    gains = [1.0 if memory_id in gold_ids else 0.0 for memory_id in ranked_ids[:k]]
    dcg = _discounted_gain(gains)
    ideal_hits = min(len(gold_ids), k)
    ideal = _discounted_gain([1.0] * ideal_hits)
    return safe_div(dcg, ideal)


def _discounted_gain(gains: list[float]) -> float:
    return sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))


def importance_weighted_recall(
    gold_ids: set[str],
    activated_ids: set[str],
    importance_by_id: dict[str, float],
) -> float:
    """Recall weighted by gold-memory importance (3/4/5), not a flat count.

    Measures whether a system activated the *important* gold memories, not merely how
    many. Numerator = Σ importance over recalled gold; denominator = Σ importance over
    all gold. Missing importance defaults to 0 weight (does not inflate the denominator).
    Empty gold → 1.0 iff nothing activated (mirrors recall_at_k's empty-gold convention).
    """
    if not gold_ids:
        return 1.0 if not activated_ids else 0.0
    total = sum(importance_by_id.get(m, 0.0) for m in gold_ids)
    hit = sum(importance_by_id.get(m, 0.0) for m in (gold_ids & activated_ids))
    return safe_div(hit, total)


def salience_precision(
    gold_ids: set[str],
    activated_ids: set[str],
    importance_by_id: dict[str, float],
    high: float = 5.0,
) -> float:
    """Fraction of the activated set that is high-importance (>= `high`) gold memory.

    Measures whether the activated set is dominated by salient gold rather than diluted
    by low-value/irrelevant memories. Denominator is the full activated set (precision
    semantics). Empty activated set → 0.0 (consistent with precision_at_k's empty handling
    when gold is non-empty; there is no salient hit to credit).
    """
    if not activated_ids:
        return 0.0
    high_gold = {m for m in gold_ids if importance_by_id.get(m, 0.0) >= high}
    return safe_div(len(high_gold & activated_ids), len(activated_ids))
