"""Deterministic compression-probe metrics."""

from __future__ import annotations


def compression_scores(
    probe_type: str | None,
    task_scores: dict[str, float],
    *,
    explicit_prediction_present: bool = False,
    explicit_prediction_used: bool = False,
) -> dict[str, float | None]:
    if probe_type != "compression_probe":
        return {
            "coverage": None,
            "coverage_soft": None,
            "forbidden_absence": None,
            "compression_quality": None,
            "compression_quality_soft": None,
            "explicit_prediction_present": None,
            "explicit_prediction_used": None,
        }
    coverage = task_scores["must_include_satisfaction"]
    coverage_soft = task_scores.get("must_include_coverage", coverage)
    forbidden_absence = task_scores["must_not_include_satisfaction"]
    return {
        "coverage": coverage,
        "coverage_soft": coverage_soft,
        "forbidden_absence": forbidden_absence,
        "compression_quality": 0.7 * coverage + 0.3 * forbidden_absence,
        "compression_quality_soft": 0.7 * coverage_soft + 0.3 * forbidden_absence,
        "explicit_prediction_present": float(explicit_prediction_present),
        "explicit_prediction_used": float(explicit_prediction_used),
    }
