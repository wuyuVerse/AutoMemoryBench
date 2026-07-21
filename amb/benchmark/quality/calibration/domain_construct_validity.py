"""Audit whether each domain contains real memory dependence and control slices."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from amb.benchmark.quality.artifact_contract import localize_report_contract
from amb.benchmark.schemas.io import read_json, write_json


DOMAIN_CONSTRUCT_VALIDITY_AUDIT_SCHEMA_VERSION = "amst-domain-construct-validity-audit-v1"
REPRESENTATIVE_SYSTEMS = ("no_memory", "full_history", "graph_memory", "oracle_memory")
MEMORY_REQUIREMENTS = ("requires_memory", "no_memory_required")


def audit_domain_construct_validity_release(
    manifest_path: str | Path,
    *,
    split: str = "public_dev",
    reports_dir: str | Path = "reports/examples",
) -> dict[str, Any]:
    """Audit per-domain solvability, memory dependence, and control-slice validity."""

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
    expected_domains = _expected_domains(manifest, split)
    if not expected_domains:
        expected_domains = tuple(
            sorted(
                set().union(
                    *[
                        set(_domain_slice_table(report))
                        for report in reports.values()
                    ]
                )
            )
        )

    domain_tables = {system_id: _domain_slice_table(report) for system_id, report in reports.items()}
    hard_domain_tables = {system_id: _domain_slice_difficulty_table(report) for system_id, report in reports.items()}
    common_domains = set(expected_domains)
    for system_id in REPRESENTATIVE_SYSTEMS:
        common_domains &= set(domain_tables[system_id])

    domain_results: dict[str, Any] = {}
    all_slice_counts_consistent = True
    all_domains_have_both_slices = True
    oracle_passes = 0
    no_memory_low_passes = 0
    full_history_failure_passes = 0
    graph_gap_passes = 0
    oracle_gap_passes = 0
    control_nontrivial_passes = 0
    graph_control_passes = 0
    graph_amq_beats_full_history_passes = 0
    hard_requires_memory_counts_consistent = True
    hard_requires_memory_present_all = True
    oracle_hard_requires_memory_passes = 0
    graph_hard_gap_passes = 0
    graph_hard_amq_beats_full_history_passes = 0
    weak_domains: list[str] = []

    for domain in expected_domains:
        slices = {
            system_id: domain_tables[system_id].get(domain, {})
            for system_id in REPRESENTATIVE_SYSTEMS
        }
        count_matrix = {
            memory_requirement: {
                system_id: int(slices[system_id].get(memory_requirement, {}).get("num_scored_queries", 0))
                for system_id in REPRESENTATIVE_SYSTEMS
            }
            for memory_requirement in MEMORY_REQUIREMENTS
        }
        slice_counts_consistent = all(
            len(set(counts.values())) == 1
            for counts in count_matrix.values()
        )
        both_slices_present = all(
            all(counts[system_id] > 0 for system_id in REPRESENTATIVE_SYSTEMS)
            for counts in count_matrix.values()
        )
        all_slice_counts_consistent = all_slice_counts_consistent and slice_counts_consistent
        all_domains_have_both_slices = all_domains_have_both_slices and both_slices_present

        req = {system_id: slices[system_id].get("requires_memory", {}) for system_id in REPRESENTATIVE_SYSTEMS}
        ctrl = {system_id: slices[system_id].get("no_memory_required", {}) for system_id in REPRESENTATIVE_SYSTEMS}
        hard_req = {
            system_id: (
                hard_domain_tables[system_id]
                .get(domain, {})
                .get("requires_memory", {})
                .get("hard", {})
            )
            for system_id in REPRESENTATIVE_SYSTEMS
        }

        oracle_solvable = (
            _metric_value(req["oracle_memory"], "task.task_success") >= 0.95
            and _metric_value(req["oracle_memory"], "lifecycle.amq") >= 0.95
        )
        no_memory_low = _metric_value(req["no_memory"], "task.task_success") <= 0.10
        full_history_failure = (
            (
                _metric_value(req["full_history"], "task.task_success") <= 0.20
                and _metric_value(req["full_history"], "safety.safety_pass") <= 0.20
            )
            or _metric_value(req["full_history"], "task.task_success") >= 0.30
        )
        if strict_hard:
            graph_gap = _metric_value(req["graph_memory"], "lifecycle.amq") >= _metric_value(req["no_memory"], "lifecycle.amq") + 0.03
        else:
            graph_gap = _metric_value(req["graph_memory"], "task.task_success") >= _metric_value(req["no_memory"], "task.task_success") + 0.04
        oracle_gap = (
            _metric_value(req["oracle_memory"], "task.task_success") >= _metric_value(req["no_memory"], "task.task_success") + 0.70
            and _metric_value(req["oracle_memory"], "lifecycle.amq") >= _metric_value(req["no_memory"], "lifecycle.amq") + 0.50
        )
        control_nontrivial = _metric_value(ctrl["no_memory"], "task.task_success") >= 0.30
        graph_control_solves = _metric_value(ctrl["graph_memory"], "task.task_success") >= _metric_value(ctrl["no_memory"], "task.task_success")
        graph_amq_beats_full_history = _metric_value(req["graph_memory"], "lifecycle.amq") >= _metric_value(req["full_history"], "lifecycle.amq") + 0.25
        hard_req_counts = {
            system_id: int(hard_req[system_id].get("num_scored_queries", 0))
            for system_id in REPRESENTATIVE_SYSTEMS
        }
        hard_req_count_consistent = len(set(hard_req_counts.values())) == 1
        hard_req_present = all(count > 0 for count in hard_req_counts.values())
        oracle_hard_requires_memory = (
            _metric_value(hard_req["oracle_memory"], "task.task_success") >= 0.95
            and _metric_value(hard_req["oracle_memory"], "lifecycle.amq") >= 0.95
        )
        if strict_hard:
            graph_hard_gap = (
                _metric_value(hard_req["graph_memory"], "lifecycle.amq")
                >= _metric_value(hard_req["no_memory"], "lifecycle.amq") + 0.03
            )
        else:
            graph_hard_gap = (
                _metric_value(hard_req["graph_memory"], "task.task_success")
                >= _metric_value(hard_req["no_memory"], "task.task_success") + 0.03
            )
        graph_hard_amq_beats_full_history = (
            _metric_value(hard_req["graph_memory"], "lifecycle.amq")
            >= _metric_value(hard_req["full_history"], "lifecycle.amq") + 0.25
        )

        if oracle_solvable:
            oracle_passes += 1
        if no_memory_low:
            no_memory_low_passes += 1
        if full_history_failure:
            full_history_failure_passes += 1
        if graph_gap:
            graph_gap_passes += 1
        if oracle_gap:
            oracle_gap_passes += 1
        if control_nontrivial:
            control_nontrivial_passes += 1
        if graph_control_solves:
            graph_control_passes += 1
        if graph_amq_beats_full_history:
            graph_amq_beats_full_history_passes += 1
        hard_requires_memory_counts_consistent = hard_requires_memory_counts_consistent and hard_req_count_consistent
        hard_requires_memory_present_all = hard_requires_memory_present_all and hard_req_present
        if oracle_hard_requires_memory:
            oracle_hard_requires_memory_passes += 1
        if graph_hard_gap:
            graph_hard_gap_passes += 1
        if graph_hard_amq_beats_full_history:
            graph_hard_amq_beats_full_history_passes += 1

        checks = {
            "slice_counts_consistent": slice_counts_consistent,
            "both_memory_slices_present": both_slices_present,
            "oracle_solvable": oracle_solvable,
            "no_memory_low_requires_memory_task": no_memory_low,
            "full_history_failure_visible": full_history_failure,
            "graph_gap_visible": graph_gap,
            "oracle_gap_visible": oracle_gap,
            "control_nontrivial": control_nontrivial,
            "graph_control_solves": graph_control_solves,
            "graph_amq_beats_full_history": graph_amq_beats_full_history,
            "hard_requires_memory_count_consistent": hard_req_count_consistent,
            "hard_requires_memory_present": hard_req_present,
            "oracle_hard_requires_memory_solvable": oracle_hard_requires_memory,
            "graph_hard_gap_visible": graph_hard_gap,
            "graph_hard_amq_beats_full_history": graph_hard_amq_beats_full_history,
        }
        if not all(checks.values()):
            weak_domains.append(domain)
        domain_results[domain] = {
            "counts": count_matrix,
            "hard_requires_memory_counts": hard_req_counts,
            "requires_memory": {
                system_id: {
                    "num_scored_queries": int(req[system_id].get("num_scored_queries", 0)),
                    "task.task_success": _metric_value(req[system_id], "task.task_success"),
                    "lifecycle.amq": _metric_value(req[system_id], "lifecycle.amq"),
                    "safety.safety_pass": _metric_value(req[system_id], "safety.safety_pass"),
                }
                for system_id in REPRESENTATIVE_SYSTEMS
            },
            "no_memory_required": {
                system_id: {
                    "num_scored_queries": int(ctrl[system_id].get("num_scored_queries", 0)),
                    "task.task_success": _metric_value(ctrl[system_id], "task.task_success"),
                    "lifecycle.amq": _metric_value(ctrl[system_id], "lifecycle.amq"),
                    "safety.safety_pass": _metric_value(ctrl[system_id], "safety.safety_pass"),
                }
                for system_id in REPRESENTATIVE_SYSTEMS
            },
            "requires_memory_hard": {
                system_id: {
                    "num_scored_queries": int(hard_req[system_id].get("num_scored_queries", 0)),
                    "task.task_success": _metric_value(hard_req[system_id], "task.task_success"),
                    "lifecycle.amq": _metric_value(hard_req[system_id], "lifecycle.amq"),
                    "safety.safety_pass": _metric_value(hard_req[system_id], "safety.safety_pass"),
                }
                for system_id in REPRESENTATIVE_SYSTEMS
            },
            "gaps": {
                "oracle_minus_no_memory_task": _metric_value(req["oracle_memory"], "task.task_success") - _metric_value(req["no_memory"], "task.task_success"),
                "oracle_minus_no_memory_amq": _metric_value(req["oracle_memory"], "lifecycle.amq") - _metric_value(req["no_memory"], "lifecycle.amq"),
                "graph_minus_no_memory_task": _metric_value(req["graph_memory"], "task.task_success") - _metric_value(req["no_memory"], "task.task_success"),
                "graph_minus_no_memory_amq": _metric_value(req["graph_memory"], "lifecycle.amq") - _metric_value(req["no_memory"], "lifecycle.amq"),
                "graph_minus_full_history_amq": _metric_value(req["graph_memory"], "lifecycle.amq") - _metric_value(req["full_history"], "lifecycle.amq"),
                "graph_hard_minus_no_memory_task": _metric_value(hard_req["graph_memory"], "task.task_success") - _metric_value(hard_req["no_memory"], "task.task_success"),
                "graph_hard_minus_no_memory_amq": _metric_value(hard_req["graph_memory"], "lifecycle.amq") - _metric_value(hard_req["no_memory"], "lifecycle.amq"),
                "graph_hard_minus_full_history_amq": _metric_value(hard_req["graph_memory"], "lifecycle.amq") - _metric_value(hard_req["full_history"], "lifecycle.amq"),
            },
            "checks": checks,
        }

    checks = {
        "representative_reports_present": _check(True, 0, 0),
        "report_benchmark_id_match": _check(
            all(benchmark_matches.values()),
            benchmark_matches,
            expected_report_benchmark_id,
        ),
        "domain_coverage_complete": _check(
            tuple(sorted(common_domains)) == expected_domains,
            tuple(sorted(common_domains)),
            expected_domains,
        ),
        "slice_counts_consistent_per_domain": _check(
            all_slice_counts_consistent,
            {domain: domain_results[domain]["counts"] for domain in expected_domains},
            "same num_scored_queries across representative systems per domain and slice",
        ),
        "all_domains_have_both_memory_slices": _check(
            all_domains_have_both_slices,
            {domain: domain_results[domain]["counts"] for domain in expected_domains},
            "requires_memory and no_memory_required are both non-empty in every domain",
        ),
        "oracle_solvable_per_domain": _check(oracle_passes == len(expected_domains), oracle_passes, len(expected_domains)),
        "no_memory_low_requires_memory_task_per_domain": _check(
            no_memory_low_passes == len(expected_domains),
            no_memory_low_passes,
            len(expected_domains),
        ),
        "full_history_failure_visible_per_domain": _check(
            full_history_failure_passes == len(expected_domains),
            full_history_failure_passes,
            len(expected_domains),
        ),
        "graph_gap_visible_per_domain": _check(
            graph_gap_passes == len(expected_domains),
            graph_gap_passes,
            len(expected_domains),
        ),
        "oracle_gap_visible_per_domain": _check(
            oracle_gap_passes == len(expected_domains),
            oracle_gap_passes,
            len(expected_domains),
        ),
        "control_nontrivial_per_domain": _check(
            control_nontrivial_passes == len(expected_domains),
            control_nontrivial_passes,
            len(expected_domains),
        ),
        "graph_control_solves_per_domain": _check(
            graph_control_passes == len(expected_domains),
            graph_control_passes,
            len(expected_domains),
        ),
        "graph_amq_beats_full_history_per_domain": _check(
            graph_amq_beats_full_history_passes == len(expected_domains),
            graph_amq_beats_full_history_passes,
            len(expected_domains),
        ),
        "hard_requires_memory_counts_consistent_per_domain": _check(
            hard_requires_memory_counts_consistent,
            {domain: domain_results[domain]["hard_requires_memory_counts"] for domain in expected_domains},
            "same num_scored_queries across representative systems on hard requires_memory slice",
        ),
        "hard_requires_memory_present_per_domain": _check(
            hard_requires_memory_present_all,
            {domain: domain_results[domain]["hard_requires_memory_counts"] for domain in expected_domains},
            "non-empty hard requires_memory slice in every domain",
        ),
        "oracle_hard_requires_memory_solvable_per_domain": _check(
            oracle_hard_requires_memory_passes == len(expected_domains),
            oracle_hard_requires_memory_passes,
            len(expected_domains),
        ),
        "graph_hard_gap_visible_per_domain": _check(
            graph_hard_gap_passes == len(expected_domains),
            graph_hard_gap_passes,
            len(expected_domains),
        ),
        "graph_hard_amq_beats_full_history_per_domain": _check(
            graph_hard_amq_beats_full_history_passes == len(expected_domains),
            graph_hard_amq_beats_full_history_passes,
            len(expected_domains),
        ),
    }
    status = "passed" if all(item["passed"] for item in checks.values()) else "failed"
    return {
        "schema_version": DOMAIN_CONSTRUCT_VALIDITY_AUDIT_SCHEMA_VERSION,
        "benchmark_id": expected_report_benchmark_id,
        "release_split": split,
        "status": status,
        "report_paths": {name: str(path) for name, path in report_paths.items()},
        "summary": {
            "num_expected_domains": len(expected_domains),
            "num_weak_domains": len(weak_domains),
            "expected_domains": list(expected_domains),
        },
        "checks": checks,
        "weak_domains": weak_domains,
        "domain_results": domain_results,
    }


def write_domain_construct_validity_audit(
    output: str | Path,
    *,
    manifest_path: str | Path,
    split: str = "public_dev",
    reports_dir: str | Path = "reports/examples",
) -> dict[str, Any]:
    report = audit_domain_construct_validity_release(
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
        "domain_coverage_complete": _check(False, (), ()),
        "slice_counts_consistent_per_domain": _check(False, {}, "same counts per domain and slice"),
        "all_domains_have_both_memory_slices": _check(False, {}, "both memory slices per domain"),
        "oracle_solvable_per_domain": _check(False, 0, 0),
        "no_memory_low_requires_memory_task_per_domain": _check(False, 0, 0),
        "full_history_failure_visible_per_domain": _check(False, 0, 0),
        "graph_gap_visible_per_domain": _check(False, 0, 0),
        "oracle_gap_visible_per_domain": _check(False, 0, 0),
        "control_nontrivial_per_domain": _check(False, 0, 0),
        "graph_control_solves_per_domain": _check(False, 0, 0),
        "graph_amq_beats_full_history_per_domain": _check(False, 0, 0),
        "hard_requires_memory_counts_consistent_per_domain": _check(False, {}, "same hard requires_memory counts per domain"),
        "hard_requires_memory_present_per_domain": _check(False, {}, "hard requires_memory slice per domain"),
        "oracle_hard_requires_memory_solvable_per_domain": _check(False, 0, 0),
        "graph_hard_gap_visible_per_domain": _check(False, 0, 0),
        "graph_hard_amq_beats_full_history_per_domain": _check(False, 0, 0),
    }
    return {
        "schema_version": DOMAIN_CONSTRUCT_VALIDITY_AUDIT_SCHEMA_VERSION,
        "benchmark_id": expected_report_benchmark_id,
        "release_split": split,
        "status": "failed",
        "report_paths": {name: str(path) for name, path in report_paths.items()},
        "summary": {
            "num_expected_domains": 0,
            "num_weak_domains": 0,
            "expected_domains": [],
        },
        "checks": checks,
        "weak_domains": [],
        "domain_results": {},
        "missing_files": missing_files,
    }


def _expected_domains(manifest: dict[str, Any], split: str) -> tuple[str, ...]:
    split_domains = manifest.get("split_reports", {}).get(split, {}).get("domains", {})
    if isinstance(split_domains, dict) and split_domains:
        return tuple(sorted(str(name) for name in split_domains))
    summary_domains = manifest.get("expected_generation_summary", {}).get("domains", ())
    if isinstance(summary_domains, (list, tuple)) and summary_domains:
        return tuple(sorted(str(name) for name in summary_domains))
    return ()


def _domain_slice_table(report: dict[str, Any]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    table: dict[str, dict[str, dict[str, float | int | None]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "num_scored_queries": 0,
                "task.task_success": 0.0,
                "lifecycle.amq": 0.0,
                "safety.safety_pass": 0.0,
            }
        )
    )
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for item in report.get("queries", ()):
        if not isinstance(item, dict):
            continue
        domain = str(item.get("domain", ""))
        memory_requirement = str(item.get("memory_requirement", ""))
        if not domain or not memory_requirement:
            continue
        bucket = table[domain][memory_requirement]
        counts[(domain, memory_requirement)] += 1
        scores = item.get("scores", {})
        task = _nested_metric(scores, "task.task_success")
        amq = _nested_metric(scores, "lifecycle.amq")
        safety = _nested_metric(scores, "safety.safety_pass")
        bucket["task.task_success"] = float(bucket["task.task_success"]) + task
        bucket["lifecycle.amq"] = float(bucket["lifecycle.amq"]) + amq
        bucket["safety.safety_pass"] = float(bucket["safety.safety_pass"]) + safety
    for (domain, memory_requirement), count in counts.items():
        bucket = table[domain][memory_requirement]
        bucket["num_scored_queries"] = count
        if count > 0:
            bucket["task.task_success"] = float(bucket["task.task_success"]) / count
            bucket["lifecycle.amq"] = float(bucket["lifecycle.amq"]) / count
            bucket["safety.safety_pass"] = float(bucket["safety.safety_pass"]) / count
    return {
        domain: {
            memory_requirement: dict(metrics)
            for memory_requirement, metrics in sorted(slices.items())
        }
        for domain, slices in sorted(table.items())
    }


def _domain_slice_difficulty_table(
    report: dict[str, Any],
) -> dict[str, dict[str, dict[str, dict[str, float | int | None]]]]:
    table: dict[str, dict[str, dict[str, dict[str, float | int | None]]]] = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: {
                    "num_scored_queries": 0,
                    "task.task_success": 0.0,
                    "lifecycle.amq": 0.0,
                    "safety.safety_pass": 0.0,
                }
            )
        )
    )
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for item in report.get("queries", ()):
        if not isinstance(item, dict):
            continue
        domain = str(item.get("domain", ""))
        memory_requirement = str(item.get("memory_requirement", ""))
        difficulty_level = str(item.get("difficulty_level", ""))
        if not domain or not memory_requirement or not difficulty_level:
            continue
        bucket = table[domain][memory_requirement][difficulty_level]
        counts[(domain, memory_requirement, difficulty_level)] += 1
        scores = item.get("scores", {})
        task = _nested_metric(scores, "task.task_success")
        amq = _nested_metric(scores, "lifecycle.amq")
        safety = _nested_metric(scores, "safety.safety_pass")
        bucket["task.task_success"] = float(bucket["task.task_success"]) + task
        bucket["lifecycle.amq"] = float(bucket["lifecycle.amq"]) + amq
        bucket["safety.safety_pass"] = float(bucket["safety.safety_pass"]) + safety
    for (domain, memory_requirement, difficulty_level), count in counts.items():
        bucket = table[domain][memory_requirement][difficulty_level]
        bucket["num_scored_queries"] = count
        if count > 0:
            bucket["task.task_success"] = float(bucket["task.task_success"]) / count
            bucket["lifecycle.amq"] = float(bucket["lifecycle.amq"]) / count
            bucket["safety.safety_pass"] = float(bucket["safety.safety_pass"]) / count
    return {
        domain: {
            memory_requirement: {
                difficulty_level: dict(metrics)
                for difficulty_level, metrics in sorted(difficulties.items())
            }
            for memory_requirement, difficulties in sorted(slices.items())
        }
        for domain, slices in sorted(table.items())
    }


def _nested_metric(scores: Any, field: str) -> float:
    current = scores
    for part in field.split("."):
        if not isinstance(current, dict):
            return 0.0
        current = current.get(part)
    try:
        return float(0.0 if current is None else current)
    except (TypeError, ValueError):
        return 0.0


def _metric_value(metrics: dict[str, Any], field: str) -> float:
    try:
        value = metrics.get(field, 0.0)
        return float(0.0 if value is None else value)
    except (TypeError, ValueError):
        return 0.0


def _check(passed: bool, actual: Any, expected: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "actual": actual, "expected": expected}
