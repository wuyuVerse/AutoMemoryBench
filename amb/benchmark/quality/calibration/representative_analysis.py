"""Helpers for validating representative-baseline statistical analysis artifacts."""

from __future__ import annotations

from typing import Any


REPRESENTATIVE_ANALYSIS_REQUIRED_SYSTEMS = (
    "no_memory",
    "full_history",
    "graph_memory",
    "oracle_memory",
)
REPRESENTATIVE_ANALYSIS_REQUIRED_METRICS = (
    "lifecycle.amq",
    "task.task_success",
)
REPRESENTATIVE_ANALYSIS_REQUIRED_PAIRS = (
    ("no_memory", "full_history"),
    ("no_memory", "graph_memory"),
    ("no_memory", "oracle_memory"),
    ("full_history", "graph_memory"),
    ("full_history", "oracle_memory"),
    ("graph_memory", "oracle_memory"),
)
REPRESENTATIVE_ANALYSIS_KEY_GAIN_PAIRS = (
    ("no_memory", "graph_memory"),
    ("no_memory", "oracle_memory"),
)


def summarize_representative_analysis(
    analysis: dict[str, Any],
    *,
    min_bootstrap_samples: int = 200,
    max_p_value: float = 0.05,
) -> dict[str, Any]:
    reports = {
        str(item.get("system_id")): item
        for item in analysis.get("reports", ())
        if isinstance(item, dict)
    }
    comparisons = {
        (str(item.get("baseline_system_id")), str(item.get("candidate_system_id"))): item
        for item in analysis.get("comparisons", ())
        if isinstance(item, dict)
    }
    report_checks: dict[str, dict[str, Any]] = {}
    num_report_cis_present = 0

    for system_id in REPRESENTATIVE_ANALYSIS_REQUIRED_SYSTEMS:
        summary = reports.get(system_id, {})
        ci_map = summary.get("bootstrap_ci", {}) if isinstance(summary, dict) else {}
        metric_checks: dict[str, Any] = {}
        for metric in REPRESENTATIVE_ANALYSIS_REQUIRED_METRICS:
            ci = ci_map.get(metric, {}) if isinstance(ci_map, dict) else {}
            present = _ci_present(ci)
            if present:
                num_report_cis_present += 1
            metric_checks[metric] = {
                "present": present,
                "ci": ci,
            }
        report_checks[system_id] = metric_checks

    pairwise_checks: dict[str, dict[str, Any]] = {}
    num_pairwise_metric_stats_present = 0
    num_key_memory_gains_visible = 0
    num_key_memory_gain_pairs_visible = 0
    key_gain_metric_targets = len(REPRESENTATIVE_ANALYSIS_KEY_GAIN_PAIRS) * len(REPRESENTATIVE_ANALYSIS_REQUIRED_METRICS)
    key_gain_pair_targets = len(REPRESENTATIVE_ANALYSIS_KEY_GAIN_PAIRS)
    weight_sensitivity = analysis.get("weight_sensitivity", {})
    expected_weight_profiles = (
        len(weight_sensitivity.get("default_weights", {})) + 1
        if isinstance(weight_sensitivity, dict) and weight_sensitivity.get("default_weights")
        else 0
    )

    for baseline_system_id, candidate_system_id in REPRESENTATIVE_ANALYSIS_REQUIRED_PAIRS:
        pair = comparisons.get((baseline_system_id, candidate_system_id), {})
        metrics = pair.get("metrics", {}) if isinstance(pair, dict) else {}
        key = f"{baseline_system_id}->{candidate_system_id}"
        metric_checks: dict[str, Any] = {}
        pair_has_visible_gain = False
        for metric in REPRESENTATIVE_ANALYSIS_REQUIRED_METRICS:
            stats = metrics.get(metric, {}) if isinstance(metrics, dict) else {}
            ci = stats.get("bootstrap_ci", {}) if isinstance(stats, dict) else {}
            present = (
                isinstance(stats, dict)
                and stats.get("mean_difference") is not None
                and _ci_present(ci)
                and stats.get("p_value") is not None
                and _positive_int(stats.get("num_pairs"))
            )
            if present:
                num_pairwise_metric_stats_present += 1
            visible = (
                present
                and _as_float(stats.get("mean_difference")) > 0.0
                and _as_float(ci.get("lower")) > 0.0
                and _as_float(stats.get("p_value")) <= max_p_value
            )
            if (baseline_system_id, candidate_system_id) in REPRESENTATIVE_ANALYSIS_KEY_GAIN_PAIRS and visible:
                num_key_memory_gains_visible += 1
                pair_has_visible_gain = True
            metric_checks[metric] = {
                "present": present,
                "visible_positive_gain": visible,
                "stats": stats,
            }
        if (baseline_system_id, candidate_system_id) in REPRESENTATIVE_ANALYSIS_KEY_GAIN_PAIRS and pair_has_visible_gain:
            num_key_memory_gain_pairs_visible += 1
        pairwise_checks[key] = metric_checks

    weight_rank_checks = {
        str(item.get("system_id")): item
        for item in weight_sensitivity.get("rank_stability", ())
        if isinstance(item, dict)
    }
    weight_pair_checks = {
        frozenset((str(item.get("system_a_id")), str(item.get("system_b_id")))): item
        for item in weight_sensitivity.get("pairwise_order_stability", ())
        if isinstance(item, dict)
    }
    weight_profiles_complete = (
        isinstance(weight_sensitivity, dict)
        and weight_sensitivity.get("present") is True
        and weight_sensitivity.get("num_complete_reports") == len(REPRESENTATIVE_ANALYSIS_REQUIRED_SYSTEMS)
        and len(weight_sensitivity.get("profiles", ())) == expected_weight_profiles
    )
    oracle_rank = weight_rank_checks.get("oracle_memory", {})
    oracle_top_rank_stable = weight_profiles_complete and oracle_rank.get("worst_rank") == 1
    key_weight_pair_checks: dict[str, Any] = {}
    stable_key_weight_pairs = 0
    for baseline_system_id, candidate_system_id in REPRESENTATIVE_ANALYSIS_KEY_GAIN_PAIRS:
        pair_key = f"{baseline_system_id}->{candidate_system_id}"
        pair = weight_pair_checks.get(frozenset((baseline_system_id, candidate_system_id)), {})
        stable = (
            weight_profiles_complete
            and isinstance(pair, dict)
            and pair.get("stable_strict_order") is True
            and pair.get("stable_higher_system_id") == candidate_system_id
        )
        if stable:
            stable_key_weight_pairs += 1
        key_weight_pair_checks[pair_key] = {
            "stable": stable,
            "pair": pair,
        }

    required_report_metric_count = len(REPRESENTATIVE_ANALYSIS_REQUIRED_SYSTEMS) * len(REPRESENTATIVE_ANALYSIS_REQUIRED_METRICS)
    required_pair_metric_count = len(REPRESENTATIVE_ANALYSIS_REQUIRED_PAIRS) * len(REPRESENTATIVE_ANALYSIS_REQUIRED_METRICS)
    return {
        "analysis_schema_version": analysis.get("analysis_schema_version"),
        "bootstrap_samples": _as_int(analysis.get("bootstrap_samples")),
        "ci_level": analysis.get("ci_level"),
        "bootstrap_samples_sufficient": _as_int(analysis.get("bootstrap_samples")) >= min_bootstrap_samples,
        "report_bootstrap_cis_present": num_report_cis_present == required_report_metric_count,
        "pairwise_stats_complete": num_pairwise_metric_stats_present == required_pair_metric_count,
        "key_memory_gains_statistically_visible": num_key_memory_gain_pairs_visible == key_gain_pair_targets,
        "required_report_metric_count": required_report_metric_count,
        "num_report_cis_present": num_report_cis_present,
        "required_pair_metric_count": required_pair_metric_count,
        "num_pairwise_metric_stats_present": num_pairwise_metric_stats_present,
        "required_key_gain_metric_count": key_gain_metric_targets,
        "num_key_memory_gains_visible": num_key_memory_gains_visible,
        "required_key_gain_pair_count": key_gain_pair_targets,
        "num_key_memory_gain_pairs_visible": num_key_memory_gain_pairs_visible,
        "weight_sensitivity_profiles_complete": weight_profiles_complete,
        "required_weight_profile_count": expected_weight_profiles,
        "num_weight_profiles": len(weight_sensitivity.get("profiles", ())) if isinstance(weight_sensitivity, dict) else 0,
        "oracle_top_rank_stable_under_weight_shifts": oracle_top_rank_stable,
        "key_memory_order_stable_under_weight_shifts": stable_key_weight_pairs == len(REPRESENTATIVE_ANALYSIS_KEY_GAIN_PAIRS),
        "required_weight_stable_pair_count": len(REPRESENTATIVE_ANALYSIS_KEY_GAIN_PAIRS),
        "num_weight_stable_pairs": stable_key_weight_pairs,
        "report_checks": report_checks,
        "pairwise_checks": pairwise_checks,
        "weight_rank_checks": weight_rank_checks,
        "weight_pair_checks": key_weight_pair_checks,
    }


def _ci_present(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("lower") is not None
        and value.get("mean") is not None
        and value.get("upper") is not None
        and _positive_int(value.get("num_observations"))
    )


def _positive_int(value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
