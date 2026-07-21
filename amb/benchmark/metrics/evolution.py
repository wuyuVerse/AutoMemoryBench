"""Deterministic evolution/procedural-memory probe metrics."""

from __future__ import annotations


def evolution_scores(
    probe_type: str | None,
    *,
    task_scores: dict[str, float],
    retrieval_scores: dict[str, float],
) -> dict[str, float | None]:
    if probe_type != "evolution_probe":
        return {
            "feedback_reuse": None,
            "feedback_reuse_soft": None,
            "procedural_transfer": None,
            "evolution_quality": None,
            "evolution_quality_soft": None,
        }
    feedback_reuse = task_scores["must_include_satisfaction"]
    feedback_reuse_soft = task_scores.get("must_include_coverage", feedback_reuse)
    procedural_transfer = retrieval_scores["evidence_complete"]
    return {
        "feedback_reuse": feedback_reuse,
        "feedback_reuse_soft": feedback_reuse_soft,
        "procedural_transfer": procedural_transfer,
        "evolution_quality": 0.5 * feedback_reuse + 0.5 * procedural_transfer,
        "evolution_quality_soft": 0.5 * feedback_reuse_soft + 0.5 * procedural_transfer,
    }
