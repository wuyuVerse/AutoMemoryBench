"""Audit whether query difficulty buckets produce meaningful performance separation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from amb.benchmark.quality.artifact_contract import localize_report_contract
from amb.benchmark.schemas.io import read_json, write_json

DIFFICULTY_CALIBRATION_AUDIT_SCHEMA_VERSION = "amst-difficulty-calibration-audit-v1"
REPRESENTATIVE_SYSTEMS = ("no_memory", "full_history", "graph_memory", "oracle_memory")
REQUIRED_DIFFICULTY_LEVELS = ("easy", "medium", "hard")
CORE_DIFFICULTY_METRICS = ("task.task_success", "lifecycle.amq", "retrieval.recall_at_k")
GRAPH_HARD_TASK_ADVANTAGE_MIN = 0.06
GRAPH_HARD_AMQ_ADVANTAGE_OVER_NO_MEMORY_MIN = 0.05
GRAPH_HARD_AMQ_ADVANTAGE_OVER_FULL_HISTORY_MIN = 0.20


def audit_difficulty_calibration_release(
    manifest_path: str | Path,
    *,
    split: str = "public_dev",
    reports_dir: str | Path = "reports/examples",
) -> dict[str, Any]:
    """Audit whether easy/medium/hard buckets are calibrated on representative baselines."""

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
    difficulty_tables = {system_id: _by_difficulty(report) for system_id, report in reports.items()}

    per_level_metrics: dict[str, Any] = {}
    per_level_counts: dict[str, dict[str, int]] = {}
    consistent_counts = True
    all_levels_present = True

    for level in REQUIRED_DIFFICULTY_LEVELS:
        level_counts: dict[str, int] = {}
        level_metrics: dict[str, Any] = {}
        for system_id in REPRESENTATIVE_SYSTEMS:
            aggregate = difficulty_tables[system_id].get(level, {})
            level_counts[system_id] = _int_value(aggregate.get("num_scored_queries"))
            level_metrics[system_id] = {
                metric: _metric_value(aggregate, metric)
                for metric in CORE_DIFFICULTY_METRICS
            }
        per_level_counts[level] = level_counts
        per_level_metrics[level] = {
            "num_scored_queries": level_counts,
            "metrics": level_metrics,
            "gaps": {
                "oracle_minus_no_memory_task": (
                    level_metrics["oracle_memory"]["task.task_success"]
                    - level_metrics["no_memory"]["task.task_success"]
                ),
                "oracle_minus_no_memory_amq": (
                    level_metrics["oracle_memory"]["lifecycle.amq"]
                    - level_metrics["no_memory"]["lifecycle.amq"]
                ),
                "graph_minus_no_memory_task": (
                    level_metrics["graph_memory"]["task.task_success"]
                    - level_metrics["no_memory"]["task.task_success"]
                ),
                "graph_minus_no_memory_amq": (
                    level_metrics["graph_memory"]["lifecycle.amq"]
                    - level_metrics["no_memory"]["lifecycle.amq"]
                ),
                "graph_minus_full_history_task": (
                    level_metrics["graph_memory"]["task.task_success"]
                    - level_metrics["full_history"]["task.task_success"]
                ),
                "graph_minus_full_history_amq": (
                    level_metrics["graph_memory"]["lifecycle.amq"]
                    - level_metrics["full_history"]["lifecycle.amq"]
                ),
                "oracle_minus_graph_task": (
                    level_metrics["oracle_memory"]["task.task_success"]
                    - level_metrics["graph_memory"]["task.task_success"]
                ),
                "oracle_minus_graph_amq": (
                    level_metrics["oracle_memory"]["lifecycle.amq"]
                    - level_metrics["graph_memory"]["lifecycle.amq"]
                ),
            },
        }
        level_present = all(count > 0 for count in level_counts.values())
        all_levels_present = all_levels_present and level_present
        distinct_counts = {count for count in level_counts.values()}
        if len(distinct_counts) > 1:
            consistent_counts = False

    reference_counts = {
        level: per_level_counts[level]["oracle_memory"]
        for level in REQUIRED_DIFFICULTY_LEVELS
    }
    easy_metrics = per_level_metrics["easy"]["metrics"]
    hard_metrics = per_level_metrics["hard"]["metrics"]

    oracle_solves_all_levels = all(
        per_level_metrics[level]["metrics"]["oracle_memory"]["task.task_success"] >= 0.95
        and per_level_metrics[level]["metrics"]["oracle_memory"]["lifecycle.amq"] >= 0.95
        for level in REQUIRED_DIFFICULTY_LEVELS
    )
    no_memory_hard_below_easy = (
        hard_metrics["no_memory"]["task.task_success"] < easy_metrics["no_memory"]["task.task_success"]
        and hard_metrics["no_memory"]["lifecycle.amq"] < easy_metrics["no_memory"]["lifecycle.amq"]
    )
    full_history_hard_below_easy = (
        hard_metrics["full_history"]["task.task_success"] < easy_metrics["full_history"]["task.task_success"]
        and hard_metrics["full_history"]["lifecycle.amq"] < easy_metrics["full_history"]["lifecycle.amq"]
    )
    full_history_task_monotonic = (
        easy_metrics["full_history"]["task.task_success"]
        >= per_level_metrics["medium"]["metrics"]["full_history"]["task.task_success"]
        >= hard_metrics["full_history"]["task.task_success"]
    )
    graph_memory_hard_below_easy = (
        hard_metrics["graph_memory"]["task.task_success"] < easy_metrics["graph_memory"]["task.task_success"]
        and hard_metrics["graph_memory"]["lifecycle.amq"] < easy_metrics["graph_memory"]["lifecycle.amq"]
    )
    if strict_hard:
        graph_memory_hard_beats_no_memory = (
            per_level_metrics["hard"]["gaps"]["graph_minus_no_memory_amq"] >= 0.03
        )
        graph_memory_hard_beats_full_history = (
            per_level_metrics["hard"]["gaps"]["graph_minus_full_history_amq"]
            >= GRAPH_HARD_AMQ_ADVANTAGE_OVER_FULL_HISTORY_MIN
        )
    else:
        graph_memory_hard_beats_no_memory = (
            per_level_metrics["hard"]["gaps"]["graph_minus_no_memory_task"] >= GRAPH_HARD_TASK_ADVANTAGE_MIN
            and per_level_metrics["hard"]["gaps"]["graph_minus_no_memory_amq"] >= GRAPH_HARD_AMQ_ADVANTAGE_OVER_NO_MEMORY_MIN
        )
        graph_memory_hard_beats_full_history = (
            per_level_metrics["hard"]["gaps"]["graph_minus_full_history_task"] >= GRAPH_HARD_TASK_ADVANTAGE_MIN
            and per_level_metrics["hard"]["gaps"]["graph_minus_full_history_amq"] >= GRAPH_HARD_AMQ_ADVANTAGE_OVER_FULL_HISTORY_MIN
        )
    oracle_gap_hard_exceeds_easy = (
        per_level_metrics["hard"]["gaps"]["oracle_minus_no_memory_task"]
        > per_level_metrics["easy"]["gaps"]["oracle_minus_no_memory_task"]
        and per_level_metrics["hard"]["gaps"]["oracle_minus_no_memory_amq"]
        > per_level_metrics["easy"]["gaps"]["oracle_minus_no_memory_amq"]
    )
    oracle_graph_hard_gap_visible = (
        per_level_metrics["hard"]["gaps"]["oracle_minus_graph_task"] >= 0.20
        and per_level_metrics["hard"]["gaps"]["oracle_minus_graph_amq"] >= 0.20
    )
    hard_query_count_ge_easy = reference_counts["hard"] >= reference_counts["easy"] > 0

    checks = {
        "representative_reports_present": _check(True, 0, 0),
        "report_benchmark_id_match": _check(
            all(benchmark_matches.values()),
            benchmark_matches,
            expected_report_benchmark_id,
        ),
        "difficulty_levels_complete": _check(
            all_levels_present,
            per_level_counts,
            "all representative systems expose easy/medium/hard with nonzero counts",
        ),
        "difficulty_level_counts_consistent": _check(
            consistent_counts,
            per_level_counts,
            "same num_scored_queries across representative systems per level",
        ),
        "hard_query_count_ge_easy": _check(
            hard_query_count_ge_easy,
            {"easy": reference_counts["easy"], "hard": reference_counts["hard"]},
            "hard >= easy > 0",
        ),
        "oracle_solves_all_levels": _check(
            oracle_solves_all_levels,
            {
                level: {
                    "task.task_success": per_level_metrics[level]["metrics"]["oracle_memory"]["task.task_success"],
                    "lifecycle.amq": per_level_metrics[level]["metrics"]["oracle_memory"]["lifecycle.amq"],
                }
                for level in REQUIRED_DIFFICULTY_LEVELS
            },
            {"task.task_success": ">= 0.95", "lifecycle.amq": ">= 0.95"},
        ),
        "no_memory_hard_below_easy": _check(
            no_memory_hard_below_easy,
            {
                "easy": easy_metrics["no_memory"],
                "hard": hard_metrics["no_memory"],
            },
            "hard performance below easy on no_memory",
        ),
        "full_history_hard_below_easy": _check(
            full_history_hard_below_easy,
            {
                "easy": easy_metrics["full_history"],
                "hard": hard_metrics["full_history"],
            },
            "hard performance below easy on full_history",
        ),
        "full_history_task_monotonic_by_difficulty": _check(
            full_history_task_monotonic,
            {
                level: per_level_metrics[level]["metrics"]["full_history"]["task.task_success"]
                for level in REQUIRED_DIFFICULTY_LEVELS
            },
            "easy >= medium >= hard on full_history task success",
        ),
        "graph_memory_hard_below_easy": _check(
            graph_memory_hard_below_easy,
            {
                "easy": easy_metrics["graph_memory"],
                "hard": hard_metrics["graph_memory"],
            },
            "hard performance below easy on graph_memory",
        ),
        "graph_memory_hard_beats_no_memory": _check(
            graph_memory_hard_beats_no_memory,
            per_level_metrics["hard"]["gaps"],
            {
                "task.task_success": "diagnostic_only_for_strict_hard" if strict_hard else f">= {GRAPH_HARD_TASK_ADVANTAGE_MIN:.2f}",
                "lifecycle.amq": ">= 0.03" if strict_hard else f">= {GRAPH_HARD_AMQ_ADVANTAGE_OVER_NO_MEMORY_MIN:.2f}",
            },
        ),
        "graph_memory_hard_beats_full_history": _check(
            graph_memory_hard_beats_full_history,
            {
                "hard": {
                    "graph_minus_full_history_task": per_level_metrics["hard"]["gaps"]["graph_minus_full_history_task"],
                    "graph_minus_full_history_amq": per_level_metrics["hard"]["gaps"]["graph_minus_full_history_amq"],
                }
            },
            {
                "task.task_success": "diagnostic_only_for_strict_hard" if strict_hard else f">= {GRAPH_HARD_TASK_ADVANTAGE_MIN:.2f}",
                "lifecycle.amq": f">= {GRAPH_HARD_AMQ_ADVANTAGE_OVER_FULL_HISTORY_MIN:.2f}",
            },
        ),
        "oracle_gap_hard_exceeds_easy": _check(
            oracle_gap_hard_exceeds_easy,
            {
                "easy": {
                    "oracle_minus_no_memory_task": per_level_metrics["easy"]["gaps"]["oracle_minus_no_memory_task"],
                    "oracle_minus_no_memory_amq": per_level_metrics["easy"]["gaps"]["oracle_minus_no_memory_amq"],
                },
                "hard": {
                    "oracle_minus_no_memory_task": per_level_metrics["hard"]["gaps"]["oracle_minus_no_memory_task"],
                    "oracle_minus_no_memory_amq": per_level_metrics["hard"]["gaps"]["oracle_minus_no_memory_amq"],
                },
            },
            "hard oracle-vs-no-memory gap exceeds easy gap",
        ),
        "oracle_graph_hard_gap_visible": _check(
            oracle_graph_hard_gap_visible,
            {
                "hard": {
                    "oracle_minus_graph_task": per_level_metrics["hard"]["gaps"]["oracle_minus_graph_task"],
                    "oracle_minus_graph_amq": per_level_metrics["hard"]["gaps"]["oracle_minus_graph_amq"],
                }
            },
            {"task.task_success": ">= 0.20", "lifecycle.amq": ">= 0.20"},
        ),
    }
    status = "passed" if all(item["passed"] for item in checks.values()) else "failed"
    weak_checks = [check_id for check_id, item in checks.items() if not item["passed"]]
    return {
        "schema_version": DIFFICULTY_CALIBRATION_AUDIT_SCHEMA_VERSION,
        "benchmark_id": expected_report_benchmark_id,
        "release_split": split,
        "status": status,
        "report_paths": {name: str(path) for name, path in report_paths.items()},
        "summary": {
            "num_difficulty_levels": len(REQUIRED_DIFFICULTY_LEVELS),
            "num_failed_checks": len(weak_checks),
            "difficulty_counts": reference_counts,
        },
        "checks": checks,
        "failed_checks": weak_checks,
        "difficulty_results": per_level_metrics,
    }


def write_difficulty_calibration_audit(
    output: str | Path,
    *,
    manifest_path: str | Path,
    split: str = "public_dev",
    reports_dir: str | Path = "reports/examples",
) -> dict[str, Any]:
    report = audit_difficulty_calibration_release(
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
        "difficulty_levels_complete": _check(False, {}, "easy/medium/hard"),
        "difficulty_level_counts_consistent": _check(False, {}, "same counts per level"),
        "hard_query_count_ge_easy": _check(False, {}, "hard >= easy > 0"),
        "oracle_solves_all_levels": _check(False, {}, {"task.task_success": ">= 0.95", "lifecycle.amq": ">= 0.95"}),
        "no_memory_hard_below_easy": _check(False, {}, "hard performance below easy on no_memory"),
        "full_history_hard_below_easy": _check(False, {}, "hard performance below easy on full_history"),
        "full_history_task_monotonic_by_difficulty": _check(False, {}, "easy >= medium >= hard on full_history task success"),
        "graph_memory_hard_below_easy": _check(False, {}, "hard performance below easy on graph_memory"),
        "graph_memory_hard_beats_no_memory": _check(
            False,
            {},
            {
                "task.task_success": f">= {GRAPH_HARD_TASK_ADVANTAGE_MIN:.2f}",
                "lifecycle.amq": f">= {GRAPH_HARD_AMQ_ADVANTAGE_OVER_NO_MEMORY_MIN:.2f}",
            },
        ),
        "graph_memory_hard_beats_full_history": _check(
            False,
            {},
            {
                "task.task_success": f">= {GRAPH_HARD_TASK_ADVANTAGE_MIN:.2f}",
                "lifecycle.amq": f">= {GRAPH_HARD_AMQ_ADVANTAGE_OVER_FULL_HISTORY_MIN:.2f}",
            },
        ),
        "oracle_gap_hard_exceeds_easy": _check(False, {}, "hard oracle-vs-no-memory gap exceeds easy gap"),
        "oracle_graph_hard_gap_visible": _check(False, {}, {"task.task_success": ">= 0.20", "lifecycle.amq": ">= 0.20"}),
    }
    return {
        "schema_version": DIFFICULTY_CALIBRATION_AUDIT_SCHEMA_VERSION,
        "benchmark_id": expected_report_benchmark_id,
        "release_split": split,
        "status": "failed",
        "report_paths": {name: str(path) for name, path in report_paths.items()},
        "summary": {
            "num_difficulty_levels": len(REQUIRED_DIFFICULTY_LEVELS),
            "num_failed_checks": len(checks),
            "difficulty_counts": {},
        },
        "checks": checks,
        "failed_checks": sorted(checks),
        "difficulty_results": {},
        "missing_files": missing_files,
    }


def _by_difficulty(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("by_difficulty")
    return value if isinstance(value, dict) else {}


def _metric_value(metrics: dict[str, Any], field: str) -> float:
    try:
        value = metrics.get(field, 0.0)
        return float(0.0 if value is None else value)
    except (TypeError, ValueError):
        return 0.0


def _int_value(value: Any) -> int:
    try:
        return int(0 if value is None else value)
    except (TypeError, ValueError):
        return 0


def _check(passed: bool, actual: Any, expected: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "actual": actual, "expected": expected}
