"""Report-level statistical analysis for AutoMemoryBench."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Iterable

from amb.benchmark.analysis.statistics import (
    DEFAULT_CI_LEVEL,
    bootstrap_mean_ci,
    mean,
    numeric_or_none,
    sign_flip_p_value,
)
from amb.benchmark.evaluation.scoring import AMQ_LIFECYCLE_WEIGHTS
from amb.benchmark.schemas.io import read_json

PRIMARY_METRICS = (
    "lifecycle.amq",
    "task.task_success",
    "retrieval.recall_at_k",
    "safety.safety_pass",
)
COST_FIELDS = (
    "efficiency.latency_ms",
    "efficiency.retrieval_latency_ms",
    "efficiency.input_tokens",
    "efficiency.output_tokens",
)
WEIGHT_SENSITIVITY_EMPHASIS_FACTOR = 2.0


def analyze_report_files(
    report_paths: Iterable[str | Path],
    *,
    seed: int = 13,
    bootstrap_samples: int = 200,
) -> dict[str, Any]:
    """Load one or more evaluation reports and compute analyzer statistics."""

    loaded = [(str(path), read_json(path)) for path in report_paths]
    return analyze_reports(loaded, seed=seed, bootstrap_samples=bootstrap_samples)


def analyze_reports(
    reports: Iterable[tuple[str | None, dict[str, Any]] | dict[str, Any]],
    *,
    seed: int = 13,
    bootstrap_samples: int = 200,
) -> dict[str, Any]:
    """Compute bootstrap CIs, paired comparisons, and quality-cost frontier."""

    normalized = [_normalize_report_entry(item, index) for index, item in enumerate(reports)]
    if not normalized:
        raise ValueError("at least one report is required")

    rng = random.Random(seed)
    report_summaries = [
        _summarize_report(path, report, rng=rng, bootstrap_samples=bootstrap_samples)
        for path, report in normalized
    ]
    comparisons = _paired_comparisons(
        normalized,
        rng=rng,
        bootstrap_samples=bootstrap_samples,
    )

    return {
        "analysis_schema_version": "amst-analyzer-v1",
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "ci_level": DEFAULT_CI_LEVEL,
        "metrics": list(PRIMARY_METRICS),
        "num_reports": len(normalized),
        "reports": report_summaries,
        "comparisons": comparisons,
        "quality_cost_frontier": _quality_cost_frontier(normalized, report_summaries),
        "weight_sensitivity": build_weight_sensitivity_analysis(normalized),
    }


def build_weight_sensitivity_analysis(
    reports: Iterable[tuple[str | None, dict[str, Any]] | dict[str, Any]],
) -> dict[str, Any]:
    """Summarize how AMQ-style report ordering shifts under deterministic weight perturbations."""

    normalized = [_normalize_report_entry(item, index) for index, item in enumerate(reports)]
    return _weight_sensitivity(normalized)


def _normalize_report_entry(
    item: tuple[str | None, dict[str, Any]] | dict[str, Any],
    index: int,
) -> tuple[str | None, dict[str, Any]]:
    if isinstance(item, tuple):
        return item
    return (f"report_{index}", item)


def _summarize_report(
    path: str | None,
    report: dict[str, Any],
    *,
    rng: random.Random,
    bootstrap_samples: int,
) -> dict[str, Any]:
    per_metric_values = {metric: _per_query_values(report, metric) for metric in PRIMARY_METRICS}
    point_estimates = {
        metric: _aggregate_or_mean(report, metric, per_metric_values[metric])
        for metric in PRIMARY_METRICS
    }
    bootstrap_ci = {
        metric: bootstrap_mean_ci(per_metric_values[metric], rng=rng, samples=bootstrap_samples)
        for metric in PRIMARY_METRICS
    }
    return {
        "report_path": path,
        "system_id": str(report.get("system_id", "unknown")),
        "benchmark_id": str(report.get("benchmark_id", "unknown")),
        "num_queries": len(report.get("queries", [])),
        "num_scored_queries": numeric_or_none(report.get("aggregate", {}).get("num_scored_queries")),
        "point_estimates": point_estimates,
        "bootstrap_ci": bootstrap_ci,
        "cost": _cost_summary(report),
    }


def _paired_comparisons(
    reports: list[tuple[str | None, dict[str, Any]]],
    *,
    rng: random.Random,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    if len(reports) < 2:
        return []

    comparisons: list[dict[str, Any]] = []
    for baseline_index in range(len(reports)):
        baseline_path, baseline = reports[baseline_index]
        for candidate_index in range(baseline_index + 1, len(reports)):
            candidate_path, candidate = reports[candidate_index]
            comparisons.append(
                {
                    "baseline_report_path": baseline_path,
                    "candidate_report_path": candidate_path,
                    "baseline_system_id": str(baseline.get("system_id", "unknown")),
                    "candidate_system_id": str(candidate.get("system_id", "unknown")),
                    "direction": "candidate_minus_baseline",
                    "metrics": {
                        metric: _paired_metric_stats(
                            baseline,
                            candidate,
                            metric,
                            rng=rng,
                            bootstrap_samples=bootstrap_samples,
                        )
                        for metric in PRIMARY_METRICS
                    },
                }
            )
    return comparisons


def _paired_metric_stats(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    metric: str,
    *,
    rng: random.Random,
    bootstrap_samples: int,
) -> dict[str, Any]:
    pairs = _paired_values(baseline, candidate, metric)
    differences = [candidate_value - baseline_value for baseline_value, candidate_value in pairs]
    ci = bootstrap_mean_ci(differences, rng=rng, samples=bootstrap_samples)
    p_value = sign_flip_p_value(differences, rng=rng, samples=bootstrap_samples)
    return {
        "mean_difference": mean(differences),
        "bootstrap_ci": ci,
        "p_value": p_value,
        "p_value_method": "monte_carlo_sign_flip_two_sided",
        "num_pairs": len(differences),
    }


def _paired_values(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    metric: str,
) -> list[tuple[float, float]]:
    baseline_by_query = _query_metric_map(baseline, metric)
    candidate_by_query = _query_metric_map(candidate, metric)
    paired_ids = sorted(set(baseline_by_query) & set(candidate_by_query))
    return [(baseline_by_query[query_id], candidate_by_query[query_id]) for query_id in paired_ids]


def _query_metric_map(report: dict[str, Any], metric: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for index, query in enumerate(report.get("queries", [])):
        value = _score_at_path(query.get("scores", {}), metric)
        numeric = numeric_or_none(value)
        if numeric is not None:
            values[str(query.get("query_id", f"query_{index}"))] = numeric
    return values


def _per_query_values(report: dict[str, Any], metric: str) -> list[float]:
    return list(_query_metric_map(report, metric).values())


def _aggregate_or_mean(report: dict[str, Any], metric: str, values: list[float]) -> float | None:
    aggregate_value = numeric_or_none(report.get("aggregate", {}).get(metric))
    if aggregate_value is not None:
        return aggregate_value
    return mean(values)


def _quality_cost_frontier(
    reports: list[tuple[str | None, dict[str, Any]]],
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_costs = [_cost_summary(report) for _, report in reports]
    scales = _cost_scales(raw_costs)
    points = []
    for (path, report), summary, raw_cost in zip(reports, summaries, raw_costs):
        quality = numeric_or_none(report.get("aggregate", {}).get("lifecycle.amq"))
        if quality is None:
            quality = summary["point_estimates"].get("lifecycle.amq")
        cost_proxy = _conservative_cost_proxy(raw_cost, scales)
        point = {
            "report_path": path,
            "system_id": str(report.get("system_id", "unknown")),
            "quality_metric": "lifecycle.amq",
            "quality": quality,
            "quality_ci": summary["bootstrap_ci"].get("lifecycle.amq"),
            "latency_ms": raw_cost["latency_ms"],
            "retrieval_latency_ms": raw_cost["retrieval_latency_ms"],
            "input_tokens": raw_cost["input_tokens"],
            "output_tokens": raw_cost["output_tokens"],
            "total_tokens": raw_cost["total_tokens"],
            "cost_proxy": cost_proxy,
            "cost_proxy_note": "Missing latency/token fields are charged as twice the worst observed value, or 1.0 when no report records that field.",
            "missing_cost_fields": raw_cost["missing_cost_fields"],
            "is_frontier": False,
        }
        points.append(point)

    for point in points:
        point["is_frontier"] = _is_frontier_point(point, points)

    frontier = sorted(
        [point for point in points if point["is_frontier"]],
        key=lambda item: (float(item["cost_proxy"]), -float(item["quality"] or -math.inf)),
    )
    return {
        "quality_metric": "lifecycle.amq",
        "cost_fields": list(COST_FIELDS),
        "cost_direction": "lower_is_better",
        "quality_direction": "higher_is_better",
        "points": points,
        "frontier": frontier,
    }


def _weight_sensitivity(reports: list[tuple[str | None, dict[str, Any]]]) -> dict[str, Any]:
    default_weights = {key: float(value) for key, value in AMQ_LIFECYCLE_WEIGHTS.items()}
    profiles = _weight_profiles(default_weights)
    report_components = [_weight_sensitivity_components(path, report) for path, report in reports]
    complete_reports = [item for item in report_components if not item["missing_components"]]
    profile_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []

    if complete_reports:
        for profile in profiles:
            scored = []
            for item in complete_reports:
                weighted_amq = _weighted_amq(item["components"], profile["weights"])
                scored.append(
                    {
                        "report_path": item["report_path"],
                        "system_id": item["system_id"],
                        "weighted_amq": weighted_amq,
                    }
                )
            scored.sort(
                key=lambda entry: (
                    -float(entry["weighted_amq"]),
                    str(entry["system_id"]),
                    str(entry["report_path"] or ""),
                )
            )
            for rank, row in enumerate(scored, start=1):
                row["rank"] = rank
            profile_rows.append(
                {
                    "profile_id": profile["profile_id"],
                    "emphasized_component": profile["emphasized_component"],
                    "weights": dict(profile["weights"]),
                    "top_system_id": scored[0]["system_id"] if scored else None,
                    "scores": scored,
                }
            )
        rank_rows = _rank_stability(profile_rows)
        pairwise_rows = _pairwise_weight_order_stability(profile_rows)

    return {
        "present": bool(complete_reports),
        "amq_metric": "lifecycle.amq",
        "component_metrics": list(default_weights),
        "default_weights": default_weights,
        "profile_generation": {
            "family": "default_plus_component_emphasis",
            "emphasis_factor": WEIGHT_SENSITIVITY_EMPHASIS_FACTOR,
            "expected_num_profiles": len(profiles),
        },
        "num_reports": len(report_components),
        "num_complete_reports": len(complete_reports),
        "missing_component_reports": [
            {
                "report_path": item["report_path"],
                "system_id": item["system_id"],
                "missing_components": list(item["missing_components"]),
            }
            for item in report_components
            if item["missing_components"]
        ],
        "profiles": profile_rows,
        "rank_stability": rank_rows,
        "pairwise_order_stability": pairwise_rows,
    }


def _is_frontier_point(point: dict[str, Any], points: list[dict[str, Any]]) -> bool:
    quality = numeric_or_none(point.get("quality"))
    cost = numeric_or_none(point.get("cost_proxy"))
    if quality is None or cost is None:
        return False

    for other in points:
        if other is point:
            continue
        other_quality = numeric_or_none(other.get("quality"))
        other_cost = numeric_or_none(other.get("cost_proxy"))
        if other_quality is None or other_cost is None:
            continue
        no_worse = other_quality >= quality and other_cost <= cost
        strictly_better = other_quality > quality or other_cost < cost
        if no_worse and strictly_better:
            return False
    return True


def _cost_summary(report: dict[str, Any]) -> dict[str, Any]:
    aggregate = report.get("aggregate", {})
    latency = numeric_or_none(aggregate.get("efficiency.latency_ms"))
    retrieval_latency = numeric_or_none(aggregate.get("efficiency.retrieval_latency_ms"))
    input_tokens = numeric_or_none(aggregate.get("efficiency.input_tokens"))
    output_tokens = numeric_or_none(aggregate.get("efficiency.output_tokens"))
    total_tokens = None
    if input_tokens is not None or output_tokens is not None:
        total_tokens = (input_tokens or 0.0) + (output_tokens or 0.0)
    values = {
        "latency_ms": latency,
        "retrieval_latency_ms": retrieval_latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    missing = [
        field
        for field, value in {
            "efficiency.latency_ms": latency,
            "efficiency.retrieval_latency_ms": retrieval_latency,
            "efficiency.input_tokens": input_tokens,
            "efficiency.output_tokens": output_tokens,
        }.items()
        if value is None
    ]
    values["missing_cost_fields"] = missing
    return values


def _cost_scales(costs: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("latency_ms", "retrieval_latency_ms", "input_tokens", "output_tokens")
    scales: dict[str, float] = {}
    for key in keys:
        observed = [
            float(cost[key])
            for cost in costs
            if isinstance(cost.get(key), (int, float)) and float(cost[key]) > 0
        ]
        scales[key] = max(observed) if observed else 1.0
    return scales


def _conservative_cost_proxy(cost: dict[str, Any], scales: dict[str, float]) -> float:
    total = 0.0
    for key, scale in scales.items():
        value = numeric_or_none(cost.get(key))
        conservative_value = value if value is not None else 2.0 * scale
        total += conservative_value / scale
    return total


def _score_at_path(scores: dict[str, Any], metric: str) -> Any:
    value: Any = scores
    for part in metric.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _weight_profiles(default_weights: dict[str, float]) -> list[dict[str, Any]]:
    profiles = [
        {
            "profile_id": "default",
            "emphasized_component": None,
            "weights": dict(default_weights),
        }
    ]
    for component in default_weights:
        adjusted = dict(default_weights)
        adjusted[component] *= WEIGHT_SENSITIVITY_EMPHASIS_FACTOR
        total = sum(adjusted.values())
        profiles.append(
            {
                "profile_id": f"emphasize_{component}",
                "emphasized_component": component,
                "weights": {
                    key: value / total
                    for key, value in adjusted.items()
                },
            }
        )
    return profiles


def _weight_sensitivity_components(path: str | None, report: dict[str, Any]) -> dict[str, Any]:
    aggregate = report.get("aggregate", {})
    components = {
        component: numeric_or_none(aggregate.get(f"lifecycle.{component}"))
        for component in AMQ_LIFECYCLE_WEIGHTS
    }
    missing = [component for component, value in components.items() if value is None]
    return {
        "report_path": path,
        "system_id": str(report.get("system_id", "unknown")),
        "components": components,
        "missing_components": missing,
    }


def _weighted_amq(components: dict[str, float | None], weights: dict[str, float]) -> float:
    total_weight = 0.0
    total_score = 0.0
    for component, weight in weights.items():
        value = numeric_or_none(components.get(component))
        if value is None:
            continue
        total_score += float(value) * float(weight)
        total_weight += float(weight)
    if total_weight <= 0.0:
        return 0.0
    return total_score / total_weight


def _rank_stability(profile_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank_map: dict[str, list[int]] = {}
    for profile in profile_rows:
        for row in profile.get("scores", ()):
            rank_map.setdefault(str(row["system_id"]), []).append(int(row["rank"]))
    rows = []
    for system_id, ranks in sorted(rank_map.items()):
        rows.append(
            {
                "system_id": system_id,
                "best_rank": min(ranks),
                "worst_rank": max(ranks),
                "rank_span": max(ranks) - min(ranks),
                "num_profiles_rank_1": sum(1 for rank in ranks if rank == 1),
                "num_profiles": len(ranks),
            }
        )
    return rows


def _pairwise_weight_order_stability(profile_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_system_scores: dict[str, list[tuple[str, float]]] = {}
    for profile in profile_rows:
        profile_id = str(profile.get("profile_id"))
        for row in profile.get("scores", ()):
            per_system_scores.setdefault(str(row["system_id"]), []).append(
                (profile_id, float(row["weighted_amq"]))
            )
    system_ids = sorted(per_system_scores)
    rows = []
    for left_index, left_system_id in enumerate(system_ids):
        left_scores = dict(per_system_scores[left_system_id])
        for right_system_id in system_ids[left_index + 1 :]:
            right_scores = dict(per_system_scores[right_system_id])
            compared_profiles = sorted(set(left_scores) & set(right_scores))
            left_higher = 0
            right_higher = 0
            tied = 0
            margins: list[float] = []
            for profile_id in compared_profiles:
                margin = float(left_scores[profile_id]) - float(right_scores[profile_id])
                margins.append(margin)
                if math.isclose(margin, 0.0, abs_tol=1e-12):
                    tied += 1
                elif margin > 0.0:
                    left_higher += 1
                else:
                    right_higher += 1
            stable_winner = None
            if left_higher == len(compared_profiles) and compared_profiles:
                stable_winner = left_system_id
            elif right_higher == len(compared_profiles) and compared_profiles:
                stable_winner = right_system_id
            default_margin = margins[0] if margins else None
            rows.append(
                {
                    "system_a_id": left_system_id,
                    "system_b_id": right_system_id,
                    "num_profiles": len(compared_profiles),
                    "profiles_where_system_a_higher": left_higher,
                    "profiles_where_system_b_higher": right_higher,
                    "profiles_tied": tied,
                    "default_higher_system_id": _default_higher_system_id(
                        left_system_id,
                        right_system_id,
                        default_margin,
                    ),
                    "stable_higher_system_id": stable_winner,
                    "stable_strict_order": stable_winner is not None,
                    "min_margin_system_a_minus_system_b": min(margins) if margins else None,
                    "max_margin_system_a_minus_system_b": max(margins) if margins else None,
                }
            )
    return rows


def _default_higher_system_id(
    left_system_id: str,
    right_system_id: str,
    default_margin: float | None,
) -> str | None:
    if default_margin is None or math.isclose(default_margin, 0.0, abs_tol=1e-12):
        return None
    return left_system_id if default_margin > 0.0 else right_system_id
