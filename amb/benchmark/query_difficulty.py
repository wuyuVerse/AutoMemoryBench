"""Query-level difficulty inference and normalization."""

from __future__ import annotations

from typing import Any

from amb.benchmark.schemas.models import Query


_PROBE_BASE_COMPLEXITY = {
    "no_memory_probe": "easy",
    "answer_probe": "easy",
    "write_probe": "medium",
    "update_probe": "easy",
    "retrieval_probe": "medium",
    "forget_probe": "medium",
    "governance_probe": "medium",
    "tool_probe": "hard",
    "planning_probe": "hard",
    "compression_probe": "hard",
    "evolution_probe": "hard",
    "governed_transfer_probe": "hard",
    "scope_contrast_probe": "hard",
    "conflict_resolution_probe": "hard",
    "cross_session_synthesis_probe": "hard",
    "adversarial_state_synthesis_probe": "hard",
    "temporal_causal_reconciliation_probe": "hard",
    "policy_temporal_state_probe": "hard",
    "policy_exception_probe": "hard",
    "state_transition_audit_probe": "hard",
}

_COMPLEXITY_SCORE = {"easy": 0, "medium": 1, "hard": 2}


def resolve_query_difficulty(query: Query) -> dict[str, Any]:
    """Return a normalized difficulty object for one query.

    Existing explicit difficulty metadata is preserved when present, but the
    core structural fields are always backfilled so older artifacts remain
    scoreable without regeneration.
    """

    inferred = infer_query_difficulty(query)
    explicit = dict(query.difficulty or {})
    if not explicit:
        return inferred

    merged = dict(inferred)
    merged.update(explicit)
    level = str(merged.get("level") or inferred["level"])
    if level not in {"easy", "medium", "hard"}:
        level = inferred["level"]
    merged["level"] = level
    merged["factors"] = _normalize_factors(merged.get("factors"), fallback=inferred["factors"])
    merged["score"] = _normalize_int(merged.get("score"), default=inferred["score"])
    merged["probe_complexity"] = str(merged.get("probe_complexity") or inferred["probe_complexity"])
    merged["num_required_memories"] = _normalize_int(
        merged.get("num_required_memories"),
        default=inferred["num_required_memories"],
    )
    merged["num_forbidden_memories"] = _normalize_int(
        merged.get("num_forbidden_memories"),
        default=inferred["num_forbidden_memories"],
    )
    merged["has_counterfactual_pair"] = bool(merged.get("has_counterfactual_pair", inferred["has_counterfactual_pair"]))
    merged["requires_refusal"] = bool(merged.get("requires_refusal", inferred["requires_refusal"]))
    merged["memory_dependency"] = str(merged.get("memory_dependency") or inferred["memory_dependency"])
    return merged


def infer_query_difficulty(query: Query) -> dict[str, Any]:
    probe_type = str(query.probe_type or f"{query.task_type}_probe")
    probe_complexity = _PROBE_BASE_COMPLEXITY.get(probe_type, "medium")
    score = _COMPLEXITY_SCORE[probe_complexity]
    factors: list[str] = [probe_complexity]

    if len(query.gold_memory_ids) >= 2:
        score += 1
        factors.append("multi_memory")
    if len(query.gold_memory_ids) >= 5:
        score += 1
        factors.append("deep_evidence")
    if query.forbidden_memory_ids:
        score += 1
        factors.append("forbidden_memory")
    if query.expected_behavior.should_refuse:
        score += 1
        factors.append("refusal")
    if query.counterfactual_group_id:
        score += 1
        factors.append("counterfactual")

    level = "easy"
    if score >= 3:
        level = "hard"
    elif score >= 1:
        level = "medium"

    return {
        "level": level,
        "score": score,
        "probe_complexity": probe_complexity,
        "factors": factors,
        "num_required_memories": len(query.gold_memory_ids),
        "num_forbidden_memories": len(query.forbidden_memory_ids),
        "has_counterfactual_pair": bool(query.counterfactual_group_id),
        "requires_refusal": bool(query.expected_behavior.should_refuse),
        "memory_dependency": query.memory_dependency,
    }


def query_difficulty_level(query: Query) -> str:
    return str(resolve_query_difficulty(query).get("level", "medium"))


def _normalize_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _normalize_factors(value: Any, *, fallback: list[str]) -> list[str]:
    if isinstance(value, (list, tuple)):
        normalized = [str(item) for item in value if str(item)]
        return normalized or list(fallback)
    return list(fallback)
