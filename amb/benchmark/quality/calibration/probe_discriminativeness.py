"""Audit whether representative probes meaningfully separate memory capabilities."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from amb.benchmark.metrics.task_judges import TASK_JUDGE_METADATA_SCHEMA_VERSION
from amb.benchmark.quality.artifact_contract import localize_report_contract
from amb.benchmark.schemas.io import read_json, write_json

PROBE_DISCRIMINATIVENESS_AUDIT_SCHEMA_VERSION = "amst-probe-discriminativeness-audit-v1"
REPRESENTATIVE_SYSTEMS = ("no_memory", "full_history", "graph_memory", "oracle_memory")
_SOFT_DIAGNOSTIC_SYSTEMS = ("full_history", "graph_memory")

_PROBE_METRICS = {
    "answer_probe": {"metric": "task.task_success", "oracle_gap_min": 0.80, "graph_gap_min": 0.40},
    "compression_probe": {
        "metric": "retrieval.recall_at_k",
        "oracle_gap_min": 0.60,
        "graph_gap_min": 0.20,
        "graph_over_full_history_min": 0.15,
        "hard_query_share_min": 0.80,
    },
    "evolution_probe": {
        "metric": "task.task_success",
        "oracle_gap_min": 0.80,
        "graph_gap_min": 0.40,
        "graph_over_full_history_metric": "lifecycle.amq",
        "graph_over_full_history_min": 0.25,
        "hard_query_share_min": 0.80,
    },
    "forget_probe": {
        "metric": "task.task_success",
        "oracle_gap_min": 0.50,
        "graph_gap_min": None,
        "graph_over_full_history_min": None,
        "hard_query_share_min": 0.50,
    },
    "governance_probe": {
        "metric": "task.task_success",
        "oracle_gap_min": 0.50,
        "graph_gap_min": None,
        "graph_over_full_history_min": None,
        "hard_query_share_min": 0.80,
    },
    "planning_probe": {
        "metric": "task.task_success",
        "oracle_gap_min": 0.80,
        "graph_gap_min": None,
        "graph_over_full_history_metric": "lifecycle.amq",
        "graph_over_full_history_min": 0.25,
    },
    "retrieval_probe": {
        "metric": "retrieval.recall_at_k",
        "oracle_gap_min": 0.70,
        "graph_gap_min": 0.30,
        "graph_over_full_history_min": 0.50,
        "hard_query_share_min": 0.80,
    },
    "tool_probe": {
        "metric": "retrieval.recall_at_k",
        "oracle_gap_min": 0.45,
        "graph_gap_min": 0.15,
        "graph_over_full_history_min": 0.20,
        "hard_query_share_min": 0.80,
    },
    "update_probe": {
        "metric": "retrieval.recall_at_k",
        "oracle_gap_min": 0.70,
        "graph_gap_min": 0.30,
        "graph_over_full_history_min": 0.50,
        "hard_query_share_min": 0.15,
    },
    "write_probe": {
        "metric": "task.task_success",
        "oracle_gap_min": 0.60,
        "graph_gap_min": None,
        "graph_over_full_history_metric": "lifecycle.amq",
        "graph_over_full_history_min": None,
    },
}

_SOFT_DIAGNOSTIC_PROBES = {
    "planning_probe": {
        "coverage_metric": "task.must_include_coverage",
        "soft_metric": None,
    },
    "compression_probe": {
        "coverage_metric": "task.must_include_coverage",
        "soft_metric": "compression.compression_quality_soft",
    },
    "evolution_probe": {
        "coverage_metric": "task.must_include_coverage",
        "soft_metric": "evolution.evolution_quality_soft",
    },
}


def audit_probe_discriminativeness_release(
    manifest_path: str | Path,
    *,
    split: str = "public_dev",
    reports_dir: str | Path = "reports/examples",
) -> dict[str, Any]:
    """Audit representative baseline separation on one public release split."""

    manifest = read_json(manifest_path)
    benchmark_id = str(manifest.get("benchmark_id", "release"))
    strict_hard = "strict" in benchmark_id
    expected_report_benchmark_id = f"{benchmark_id}-{split}"
    prefix = f"{benchmark_id.replace('-', '_')}_{split}"
    reports_root = Path(reports_dir)
    report_paths = {
        system_id: reports_root / f"{prefix}_{system_id}_report.json"
        for system_id in REPRESENTATIVE_SYSTEMS
    }
    missing_files = [str(path) for path in report_paths.values() if not path.exists()]
    if missing_files:
        return _missing_report_result(
            expected_report_benchmark_id=expected_report_benchmark_id,
            split=split,
            report_paths=report_paths,
            missing_files=missing_files,
        )

    reports = {system_id: read_json(path) for system_id, path in report_paths.items()}
    benchmark_matches = {
        system_id: str(report.get("benchmark_id")) == expected_report_benchmark_id
        for system_id, report in reports.items()
    }
    available_probe_types = set.intersection(
        *[
            set(_by_probe(report))
            for report in reports.values()
        ]
    )
    reference_report = reports["oracle_memory"]
    probe_distributions = _probe_query_distribution(reference_report)
    probe_results: dict[str, Any] = {}
    weak_probe_types: list[str] = []
    oracle_passes = 0
    oracle_gap_passes = 0
    graph_gap_passes = 0
    graph_full_passes = 0
    hard_coverage_passes = 0
    graph_full_required = 0
    hard_coverage_required = 0

    for probe_type, config in sorted(_PROBE_METRICS.items()):
        metric = str(config["metric"])
        oracle_gap_threshold = float(config["oracle_gap_min"])
        raw_graph_gap = config.get("graph_gap_min")
        graph_gap_threshold = None if raw_graph_gap is None else float(raw_graph_gap)
        graph_full_metric = str(config.get("graph_over_full_history_metric") or metric)
        raw_graph_full = config.get("graph_over_full_history_min")
        graph_full_threshold = None if raw_graph_full is None else float(raw_graph_full)
        raw_hard_share = config.get("hard_query_share_min")
        hard_share_threshold = None if raw_hard_share is None else float(raw_hard_share)

        primary_values = {
            system_id: _metric_value(_by_probe(report).get(probe_type, {}), metric)
            for system_id, report in reports.items()
        }
        graph_full_values = {
            system_id: _metric_value(_by_probe(report).get(probe_type, {}), graph_full_metric)
            for system_id, report in reports.items()
        }
        oracle_value = primary_values["oracle_memory"]
        no_memory_value = primary_values["no_memory"]
        graph_value = primary_values["graph_memory"]
        full_history_value = primary_values["full_history"]
        oracle_gap = oracle_value - no_memory_value
        oracle_solves = oracle_gap >= oracle_gap_threshold if strict_hard else oracle_value >= 0.95
        graph_gap = graph_value - no_memory_value
        graph_over_full_primary = graph_value - full_history_value
        graph_over_full_check = graph_full_values["graph_memory"] - graph_full_values["full_history"]

        distribution = probe_distributions.get(probe_type, _empty_distribution())
        hard_share = float(distribution["hard_query_share"])
        if graph_gap_threshold is None:
            graph_gap_passed = True
        elif strict_hard and metric == "task.task_success":
            graph_gap_passed = True
        else:
            graph_gap_passed = graph_gap >= graph_gap_threshold
        if graph_full_threshold is None:
            graph_full_passed = True
        elif strict_hard:
            graph_full_passed = graph_over_full_check >= min(graph_full_threshold, 0.05)
        else:
            graph_full_passed = graph_over_full_check >= graph_full_threshold
        hard_coverage_passed = True if hard_share_threshold is None else hard_share >= hard_share_threshold
        probe_passed = (
            oracle_solves
            and oracle_gap >= oracle_gap_threshold
            and graph_gap_passed
            and graph_full_passed
            and hard_coverage_passed
        )

        if oracle_solves:
            oracle_passes += 1
        if oracle_gap >= oracle_gap_threshold:
            oracle_gap_passes += 1
        if graph_gap_passed:
            graph_gap_passes += 1
        if graph_full_threshold is not None:
            graph_full_required += 1
            if graph_full_passed:
                graph_full_passes += 1
        if hard_share_threshold is not None:
            hard_coverage_required += 1
            if hard_coverage_passed:
                hard_coverage_passes += 1
        if not probe_passed:
            weak_probe_types.append(probe_type)

        probe_results[probe_type] = {
            "metric": metric,
            "graph_over_full_history_metric": graph_full_metric,
            "values": primary_values,
            "graph_over_full_history_values": graph_full_values,
            "query_distribution": distribution,
            "gaps": {
                "oracle_minus_no_memory": oracle_gap,
                "graph_minus_no_memory": graph_gap,
                "graph_minus_full_history_primary": graph_over_full_primary,
                "graph_minus_full_history_check_metric": graph_over_full_check,
            },
            "thresholds": {
                "oracle_minus_no_memory_min": oracle_gap_threshold,
                "graph_minus_no_memory_min": graph_gap_threshold,
                "graph_minus_full_history_min": graph_full_threshold,
                "hard_query_share_min": hard_share_threshold,
                "oracle_min": "oracle_gap_threshold" if strict_hard else 0.95,
            },
            "checks": {
                "oracle_solves": oracle_solves,
                "oracle_beats_no_memory": oracle_gap >= oracle_gap_threshold,
                "graph_beats_no_memory": graph_gap_passed,
                "graph_beats_full_history": graph_full_passed,
                "hard_coverage": hard_coverage_passed,
            },
            "passed": probe_passed,
        }

    safety_gap = (
        _metric_value(_by_probe(reports["graph_memory"]).get("governance_probe", {}), "safety.safety_pass")
        - _metric_value(_by_probe(reports["full_history"]).get("governance_probe", {}), "safety.safety_pass")
    )
    checks = {
        "representative_reports_present": _check(True, 0, 0),
        "report_benchmark_id_match": _check(all(benchmark_matches.values()), benchmark_matches, expected_report_benchmark_id),
        "core_probe_coverage_complete": _check(
            available_probe_types >= set(_PROBE_METRICS),
            len(available_probe_types & set(_PROBE_METRICS)),
            len(_PROBE_METRICS),
        ),
        "oracle_solves_core_probes": _check(oracle_passes == len(_PROBE_METRICS), oracle_passes, len(_PROBE_METRICS)),
        "oracle_beats_no_memory_on_core_probes": _check(
            oracle_gap_passes == len(_PROBE_METRICS),
            oracle_gap_passes,
            len(_PROBE_METRICS),
        ),
        "graph_memory_beats_no_memory_on_core_probes": _check(
            graph_gap_passes == len(_PROBE_METRICS),
            graph_gap_passes,
            len(_PROBE_METRICS),
        ),
        "graph_memory_beats_full_history_on_structured_probes": _check(
            graph_full_passes == graph_full_required,
            graph_full_passes,
            graph_full_required,
        ),
        "hard_probe_coverage_complete": _check(
            hard_coverage_passes == hard_coverage_required,
            hard_coverage_passes,
            hard_coverage_required,
        ),
        "graph_memory_safety_advantage_visible": _check(safety_gap >= 0.50, safety_gap, ">= 0.50"),
    }
    task_judge_metadata_results = _build_task_judge_metadata_results(reports)
    soft_diagnostic_results = _build_soft_diagnostic_results(reports)
    checks.update(
        {
            "representative_reports_have_task_judge_metadata": _check(
                task_judge_metadata_results["num_passing_systems"] == task_judge_metadata_results["num_systems"]
                and task_judge_metadata_results["all_systems_share_plugin"],
                {
                    "num_passing_systems": task_judge_metadata_results["num_passing_systems"],
                    "num_systems": task_judge_metadata_results["num_systems"],
                    "shared_plugin_ids": task_judge_metadata_results["shared_plugin_ids"],
                },
                {
                    "num_passing_systems": len(REPRESENTATIVE_SYSTEMS),
                    "shared_plugin_ids": 1,
                },
            ),
            "soft_diagnostic_metrics_present_on_multi_target_probes": _check(
                soft_diagnostic_results["num_metric_complete_probes"] == soft_diagnostic_results["num_soft_diagnostic_probes"],
                soft_diagnostic_results["num_metric_complete_probes"],
                soft_diagnostic_results["num_soft_diagnostic_probes"],
            ),
            "soft_diagnostics_distinguish_memory_progress_on_multi_target_probes": _check(
                soft_diagnostic_results["num_distinguishing_probes"] >= max(
                    2,
                    soft_diagnostic_results["num_soft_diagnostic_probes"] - 1,
                ),
                soft_diagnostic_results["num_distinguishing_probes"],
                f">= max(2, {soft_diagnostic_results['num_soft_diagnostic_probes']} - 1)",
            ),
            "strict_failures_expose_partial_credit_signal": _check(
                soft_diagnostic_results["num_partial_signal_probes"] >= 1,
                soft_diagnostic_results["num_partial_signal_probes"],
                ">= 1",
            ),
        }
    )
    compression_prediction_results = _build_explicit_compression_prediction_results(reports)
    checks["compression_probe_uses_explicit_prediction_channel"] = _check(
        compression_prediction_results["num_passing_systems"] == compression_prediction_results["num_systems"],
        compression_prediction_results["num_passing_systems"],
        compression_prediction_results["num_systems"],
    )
    status = "passed" if all(item["passed"] for item in checks.values()) else "failed"
    return {
        "schema_version": PROBE_DISCRIMINATIVENESS_AUDIT_SCHEMA_VERSION,
        "benchmark_id": expected_report_benchmark_id,
        "release_split": split,
        "status": status,
        "report_paths": {name: str(path) for name, path in report_paths.items()},
        "summary": {
            "num_core_probe_types": len(_PROBE_METRICS),
            "num_passed_core_probe_types": len(_PROBE_METRICS) - len(weak_probe_types),
            "num_failed_core_probe_types": len(weak_probe_types),
            "num_graph_over_full_history_probes": graph_full_required,
            "num_hard_coverage_probes": hard_coverage_required,
        },
        "checks": checks,
        "weak_probe_types": weak_probe_types,
        "probe_results": probe_results,
        "task_judge_metadata_results": task_judge_metadata_results,
        "soft_diagnostic_results": soft_diagnostic_results,
        "compression_prediction_results": compression_prediction_results,
    }


def write_probe_discriminativeness_audit(
    output: str | Path,
    *,
    manifest_path: str | Path,
    split: str = "public_dev",
    reports_dir: str | Path = "reports/examples",
) -> dict[str, Any]:
    report = audit_probe_discriminativeness_release(
        manifest_path,
        split=split,
        reports_dir=reports_dir,
    )
    report = localize_report_contract(
        report,
        output_path=output,
        project_root_hints=(manifest_path, reports_dir),
    )
    write_json(output, report)
    return report


def _missing_report_result(
    *,
    expected_report_benchmark_id: str,
    split: str,
    report_paths: dict[str, Path],
    missing_files: list[str],
) -> dict[str, Any]:
    checks = {
        "representative_reports_present": _check(False, len(missing_files), 0),
        "report_benchmark_id_match": _check(False, "missing", expected_report_benchmark_id),
        "core_probe_coverage_complete": _check(False, 0, len(_PROBE_METRICS)),
        "oracle_solves_core_probes": _check(False, 0, len(_PROBE_METRICS)),
        "oracle_beats_no_memory_on_core_probes": _check(False, 0, len(_PROBE_METRICS)),
        "graph_memory_beats_no_memory_on_core_probes": _check(False, 0, len(_PROBE_METRICS)),
        "graph_memory_beats_full_history_on_structured_probes": _check(False, 0, 0),
        "hard_probe_coverage_complete": _check(False, 0, 0),
        "graph_memory_safety_advantage_visible": _check(False, 0.0, ">= 0.50"),
        "representative_reports_have_task_judge_metadata": _check(
            False,
            {"num_passing_systems": 0, "num_systems": len(REPRESENTATIVE_SYSTEMS), "shared_plugin_ids": []},
            {"num_passing_systems": len(REPRESENTATIVE_SYSTEMS), "shared_plugin_ids": 1},
        ),
        "soft_diagnostic_metrics_present_on_multi_target_probes": _check(False, 0, len(_SOFT_DIAGNOSTIC_PROBES)),
        "soft_diagnostics_distinguish_memory_progress_on_multi_target_probes": _check(False, 0, len(_SOFT_DIAGNOSTIC_PROBES)),
        "strict_failures_expose_partial_credit_signal": _check(False, 0, ">= 1"),
        "compression_probe_uses_explicit_prediction_channel": _check(False, 0, len(REPRESENTATIVE_SYSTEMS)),
    }
    return {
        "schema_version": PROBE_DISCRIMINATIVENESS_AUDIT_SCHEMA_VERSION,
        "benchmark_id": expected_report_benchmark_id,
        "release_split": split,
        "status": "failed",
        "report_paths": {name: str(path) for name, path in report_paths.items()},
        "summary": {
            "num_core_probe_types": len(_PROBE_METRICS),
            "num_passed_core_probe_types": 0,
            "num_failed_core_probe_types": len(_PROBE_METRICS),
            "num_graph_over_full_history_probes": 0,
            "num_hard_coverage_probes": 0,
        },
        "checks": checks,
        "weak_probe_types": sorted(_PROBE_METRICS),
        "probe_results": {},
        "task_judge_metadata_results": {
            "num_systems": len(REPRESENTATIVE_SYSTEMS),
            "num_passing_systems": 0,
            "all_systems_share_plugin": False,
            "shared_plugin_ids": [],
            "systems": {},
        },
        "soft_diagnostic_results": {
            "num_soft_diagnostic_probes": len(_SOFT_DIAGNOSTIC_PROBES),
            "num_metric_complete_probes": 0,
            "num_distinguishing_probes": 0,
            "num_partial_signal_probes": 0,
            "probes": {},
        },
        "compression_prediction_results": {
            "num_systems": len(REPRESENTATIVE_SYSTEMS),
            "num_passing_systems": 0,
            "systems": {},
        },
        "missing_files": missing_files,
    }


def _build_soft_diagnostic_results(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    metric_complete = 0
    distinguishing = 0
    partial_signal = 0

    for probe_type, config in sorted(_SOFT_DIAGNOSTIC_PROBES.items()):
        coverage_metric = str(config["coverage_metric"])
        raw_soft_metric = config.get("soft_metric")
        soft_metric = None if raw_soft_metric in (None, "") else str(raw_soft_metric)
        metrics_by_system = {
            system_id: _by_probe(report).get(probe_type, {})
            for system_id, report in reports.items()
        }
        metric_presence = {
            system_id: {
                "task.task_success": _metric_present(values, "task.task_success"),
                coverage_metric: _metric_present(values, coverage_metric),
                **(
                    {soft_metric: _metric_present(values, soft_metric)}
                    if soft_metric is not None
                    else {}
                ),
            }
            for system_id, values in metrics_by_system.items()
        }
        metrics_complete = all(all(status.values()) for status in metric_presence.values())
        if metrics_complete:
            metric_complete += 1

        task_values = {
            system_id: _metric_value(values, "task.task_success")
            for system_id, values in metrics_by_system.items()
        }
        coverage_values = {
            system_id: _metric_value(values, coverage_metric)
            for system_id, values in metrics_by_system.items()
        }
        soft_values = (
            {
                system_id: _metric_value(values, soft_metric)
                for system_id, values in metrics_by_system.items()
            }
            if soft_metric is not None
            else {}
        )
        graph_coverage_gain = coverage_values["graph_memory"] - coverage_values["no_memory"]
        graph_soft_gain = None if soft_metric is None else soft_values["graph_memory"] - soft_values["no_memory"]
        distinguishes_progress = (
            metrics_complete
            and coverage_values["oracle_memory"] >= 0.95
            and graph_coverage_gain > 0.0
            and (soft_metric is None or (graph_soft_gain is not None and graph_soft_gain > 0.0))
        )
        if distinguishes_progress:
            distinguishing += 1

        partial_signal_systems: list[dict[str, Any]] = []
        for system_id in _SOFT_DIAGNOSTIC_SYSTEMS:
            task_value = task_values[system_id]
            coverage_value = coverage_values[system_id]
            soft_value = None if soft_metric is None else soft_values[system_id]
            if task_value >= 1.0:
                continue
            coverage_gap = coverage_value - task_value
            soft_gap = None if soft_value is None else soft_value - task_value
            if coverage_gap > 0.0 or (soft_gap is not None and soft_gap > 0.0):
                partial_signal_systems.append(
                    {
                        "system_id": system_id,
                        "task_success": task_value,
                        coverage_metric: coverage_value,
                        "coverage_minus_task_success": coverage_gap,
                        **(
                            {
                                soft_metric: soft_value,
                                "soft_minus_task_success": soft_gap,
                            }
                            if soft_metric is not None
                            else {}
                        ),
                    }
                )
        has_partial_signal = bool(partial_signal_systems)
        if has_partial_signal:
            partial_signal += 1

        results[probe_type] = {
            "coverage_metric": coverage_metric,
            "soft_metric": soft_metric,
            "metric_presence": metric_presence,
            "metrics_complete": metrics_complete,
            "values": {
                system_id: {
                    "task.task_success": task_values[system_id],
                    coverage_metric: coverage_values[system_id],
                    **(
                        {soft_metric: soft_values[system_id]}
                        if soft_metric is not None
                        else {}
                    ),
                }
                for system_id in REPRESENTATIVE_SYSTEMS
            },
            "deltas": {
                "graph_coverage_minus_no_memory": graph_coverage_gain,
                "graph_soft_minus_no_memory": graph_soft_gain,
            },
            "distinguishes_memory_progress": distinguishes_progress,
            "has_partial_signal": has_partial_signal,
            "partial_signal_systems": partial_signal_systems,
        }

    return {
        "num_soft_diagnostic_probes": len(_SOFT_DIAGNOSTIC_PROBES),
        "num_metric_complete_probes": metric_complete,
        "num_distinguishing_probes": distinguishing,
        "num_partial_signal_probes": partial_signal,
        "probes": results,
    }


def _build_task_judge_metadata_results(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    passing = 0
    passing_plugin_ids: list[str] = []

    for system_id in REPRESENTATIVE_SYSTEMS:
        report = reports[system_id]
        top_level = report.get("task_judge", {})
        queries = report.get("queries", ())
        query_count = len(queries) if isinstance(queries, list) else 0
        query_metadata = [
            item.get("diagnostics", {}).get("task_judge")
            for item in queries
            if isinstance(item, dict) and isinstance(item.get("diagnostics", {}).get("task_judge"), dict)
        ]
        query_plugin_ids = sorted(
            {
                str(item.get("plugin_id"))
                for item in query_metadata
                if isinstance(item, dict) and item.get("plugin_id")
            }
        )
        top_plugin_id = str(top_level.get("plugin_id") or "")
        top_level_present = (
            isinstance(top_level, dict)
            and top_level.get("schema_version") == TASK_JUDGE_METADATA_SCHEMA_VERSION
            and bool(top_plugin_id)
            and bool(top_level.get("plugin_kind"))
        )
        top_count_matches = _int_or_zero(top_level.get("num_scored_queries")) == query_count
        query_metadata_complete = len(query_metadata) == query_count
        query_plugin_matches = len(query_plugin_ids) == 1 and top_plugin_id in query_plugin_ids
        passed = top_level_present and top_count_matches and query_metadata_complete and query_plugin_matches
        if passed:
            passing += 1
            passing_plugin_ids.append(top_plugin_id)
        systems[system_id] = {
            "passed": passed,
            "plugin_id": top_plugin_id or None,
            "query_plugin_ids": query_plugin_ids,
            "top_level_present": top_level_present,
            "query_count": query_count,
            "top_query_count": _int_or_zero(top_level.get("num_scored_queries")),
            "top_count_matches": top_count_matches,
            "query_metadata_complete": query_metadata_complete,
            "query_plugin_matches": query_plugin_matches,
        }

    shared_plugin_ids = sorted(set(passing_plugin_ids))
    return {
        "num_systems": len(REPRESENTATIVE_SYSTEMS),
        "num_passing_systems": passing,
        "all_systems_share_plugin": passing == len(REPRESENTATIVE_SYSTEMS) and len(shared_plugin_ids) == 1,
        "shared_plugin_ids": shared_plugin_ids,
        "systems": systems,
    }


def _build_explicit_compression_prediction_results(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    passing = 0
    for system_id in REPRESENTATIVE_SYSTEMS:
        metrics = _by_probe(reports[system_id]).get("compression_probe", {})
        present_metric_present = _metric_present(metrics, "compression.explicit_prediction_present")
        used_metric_present = _metric_present(metrics, "compression.explicit_prediction_used")
        explicit_present = _metric_value(metrics, "compression.explicit_prediction_present")
        explicit_used = _metric_value(metrics, "compression.explicit_prediction_used")
        passed = (
            present_metric_present
            and used_metric_present
            and explicit_present >= 1.0
            and explicit_used >= 1.0
        )
        if passed:
            passing += 1
        systems[system_id] = {
            "metrics_present": {
                "compression.explicit_prediction_present": present_metric_present,
                "compression.explicit_prediction_used": used_metric_present,
            },
            "values": {
                "compression.explicit_prediction_present": explicit_present,
                "compression.explicit_prediction_used": explicit_used,
            },
            "passed": passed,
        }
    return {
        "num_systems": len(REPRESENTATIVE_SYSTEMS),
        "num_passing_systems": passing,
        "systems": systems,
    }


def _probe_query_distribution(report: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "num_queries": 0,
            "hard_queries": 0,
            "requires_memory_queries": 0,
            "no_memory_required_queries": 0,
        }
    )
    for item in report.get("queries", ()):
        if not isinstance(item, dict):
            continue
        probe_type = str(item.get("probe_type", ""))
        if not probe_type:
            continue
        bucket = counts[probe_type]
        bucket["num_queries"] += 1
        if str(item.get("difficulty_level", "")) == "hard":
            bucket["hard_queries"] += 1
        memory_requirement = str(item.get("memory_requirement", ""))
        if memory_requirement == "requires_memory":
            bucket["requires_memory_queries"] += 1
        elif memory_requirement == "no_memory_required":
            bucket["no_memory_required_queries"] += 1
    result: dict[str, dict[str, float | int]] = {}
    for probe_type, metrics in counts.items():
        total = metrics["num_queries"]
        hard = metrics["hard_queries"]
        result[probe_type] = {
            **metrics,
            "hard_query_share": 0.0 if total <= 0 else hard / total,
        }
    return result


def _empty_distribution() -> dict[str, float | int]:
    return {
        "num_queries": 0,
        "hard_queries": 0,
        "requires_memory_queries": 0,
        "no_memory_required_queries": 0,
        "hard_query_share": 0.0,
    }


def _by_probe(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("by_probe_type")
    return value if isinstance(value, dict) else {}


def _metric_value(metrics: dict[str, Any], field: str) -> float:
    try:
        value = metrics.get(field, 0.0)
        return float(0.0 if value is None else value)
    except (TypeError, ValueError):
        return 0.0


def _metric_present(metrics: dict[str, Any], field: str) -> bool:
    if field not in metrics:
        return False
    try:
        return metrics.get(field) is not None and float(metrics[field]) == float(metrics[field])
    except (TypeError, ValueError):
        return False


def _check(passed: bool, actual: Any, expected: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "actual": actual, "expected": expected}


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
