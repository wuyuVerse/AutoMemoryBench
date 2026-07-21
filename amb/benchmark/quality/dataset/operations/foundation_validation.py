"""Machine-readable protocol-sensitivity validation for real foundation-model runs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from amb.benchmark.analysis.protocol_strength import build_protocol_strength_report
from amb.benchmark.quality.artifact_contract import localize_report_contract
from amb.benchmark.schemas.io import read_json, write_json

FOUNDATION_VALIDATION_AUDIT_SCHEMA_VERSION = "amst-foundation-validation-audit-v1"
FOUNDATION_PROTOCOLS = ("query_only", "full_history", "oracle_state")
REQUIRED_FOUNDATION_PROTOCOLS = ("query_only", "oracle_state")
FOUNDATION_MEMORY_SENSITIVE_PROBES = ("answer_probe", "planning_probe", "retrieval_probe", "update_probe", "write_probe")
FOUNDATION_ORACLE_TASK_GAIN_MIN = 0.20
FOUNDATION_ORACLE_RECALL_GAIN_MIN = 0.20
FOUNDATION_FULL_HISTORY_TASK_GAIN_MIN = 0.20
FOUNDATION_FULL_HISTORY_RECALL_GAIN_MIN = 0.15


def validate_foundation_protocol_reports(
    report_paths: Iterable[str | Path],
    *,
    expected_benchmark_id: str | None = None,
    cohort_id: str | None = None,
    require_full_history: bool = False,
) -> dict[str, Any]:
    """Validate one same-model protocol cohort of scored foundation-model reports."""

    reports_by_protocol: dict[str, dict[str, Any]] = {}
    sources_by_protocol: dict[str, str] = {}
    errors: list[str] = []
    warnings: list[str] = []

    for raw_path in report_paths:
        path = Path(raw_path)
        try:
            report = read_json(path)
        except Exception as exc:
            errors.append(f"{path}: cannot read report: {exc}")
            continue
        if not isinstance(report, dict):
            errors.append(f"{path}: report must be a JSON object")
            continue
        protocol = _infer_protocol(report, path)
        if protocol is None:
            errors.append(f"{path}: cannot infer protocol from system_id or path")
            continue
        if protocol in reports_by_protocol:
            errors.append(f"duplicate report for protocol {protocol}: {path}")
            continue
        reports_by_protocol[protocol] = report
        sources_by_protocol[protocol] = str(path)

    for protocol in REQUIRED_FOUNDATION_PROTOCOLS:
        if protocol not in reports_by_protocol:
            errors.append(f"missing required protocol report: {protocol}")
    if require_full_history and "full_history" not in reports_by_protocol:
        errors.append("missing required protocol report: full_history")
    if not require_full_history and "full_history" not in reports_by_protocol:
        warnings.append("full_history report not provided; intermediate protocol checks were skipped")

    benchmark_ids = {
        protocol: str(report.get("benchmark_id", ""))
        for protocol, report in reports_by_protocol.items()
    }
    cohort_signatures = {
        protocol: _cohort_signature(str(report.get("system_id", "")))
        for protocol, report in reports_by_protocol.items()
    }
    protocol_metrics = {
        protocol: _protocol_metrics(report)
        for protocol, report in reports_by_protocol.items()
    }
    ordered_report_paths = [
        sources_by_protocol[protocol]
        for protocol in FOUNDATION_PROTOCOLS
        if protocol in sources_by_protocol
    ]
    protocol_strength = build_protocol_strength_report(ordered_report_paths) if ordered_report_paths else {
        "analysis_schema_version": "amst-protocol-strength-v1",
        "num_reports": 0,
        "metrics": [],
        "reports": [],
        "pairwise": [],
    }

    expected_benchmark = expected_benchmark_id or next(iter(benchmark_ids.values()), "")
    checks = {
        "required_protocols_present": _check(
            all(protocol in reports_by_protocol for protocol in REQUIRED_FOUNDATION_PROTOCOLS)
            and (not require_full_history or "full_history" in reports_by_protocol),
            sorted(reports_by_protocol),
            list(REQUIRED_FOUNDATION_PROTOCOLS + (("full_history",) if require_full_history else ())),
        ),
        "benchmark_id_match": _check(
            bool(expected_benchmark)
            and all(benchmark_id == expected_benchmark for benchmark_id in benchmark_ids.values()),
            benchmark_ids,
            expected_benchmark,
        ),
        "cohort_signature_match": _check(
            len({signature for signature in cohort_signatures.values() if signature}) <= 1,
            cohort_signatures,
            "same normalized cohort signature across protocols",
        ),
        "num_scored_queries_consistent": _check(
            _consistent_numeric(protocol_metrics, "num_scored_queries"),
            {protocol: metrics["num_scored_queries"] for protocol, metrics in protocol_metrics.items()},
            "same num_scored_queries across protocols",
        ),
        "slice_coverage_present": _check(
            all(
                metrics["requires_memory_num_queries"] > 0 and metrics["no_memory_required_num_queries"] > 0
                for metrics in protocol_metrics.values()
            ),
            {
                protocol: {
                    "requires_memory_num_queries": metrics["requires_memory_num_queries"],
                    "no_memory_required_num_queries": metrics["no_memory_required_num_queries"],
                }
                for protocol, metrics in protocol_metrics.items()
            },
            "all protocols expose nonzero requires_memory and no_memory_required slices",
        ),
        "protocol_pairwise_complete": _check(
            len(protocol_strength.get("pairwise", [])) == (len(ordered_report_paths) * (len(ordered_report_paths) - 1)) // 2,
            len(protocol_strength.get("pairwise", [])),
            (len(ordered_report_paths) * (len(ordered_report_paths) - 1)) // 2,
        ),
    }

    query_only = protocol_metrics.get("query_only")
    oracle_state = protocol_metrics.get("oracle_state")
    full_history = protocol_metrics.get("full_history")

    if query_only is not None:
        checks.update(
            {
                "query_only_memory_task_low": _check(
                    query_only["requires_memory_task_success"] <= 0.10,
                    query_only["requires_memory_task_success"],
                    "<= 0.10",
                ),
                "query_only_control_task_nontrivial": _check(
                    query_only["no_memory_required_task_success"] >= 0.50,
                    query_only["no_memory_required_task_success"],
                    ">= 0.50",
                ),
                "query_only_control_gap_visible": _check(
                    query_only["no_memory_required_task_success"]
                    >= query_only["requires_memory_task_success"] + 0.40,
                    {
                        "requires_memory_task_success": query_only["requires_memory_task_success"],
                        "no_memory_required_task_success": query_only["no_memory_required_task_success"],
                    },
                    "no_memory_required task >= requires_memory task + 0.40",
                ),
            }
        )

    if query_only is not None and oracle_state is not None:
        oracle_task_gain = _protocol_strength_probe_gain_summary(
            protocol_strength,
            left_protocol="query_only",
            right_protocol="oracle_state",
            probes=FOUNDATION_MEMORY_SENSITIVE_PROBES,
            metric="task.task_success",
            min_delta=FOUNDATION_ORACLE_TASK_GAIN_MIN,
        )
        oracle_recall_gain = _protocol_strength_probe_gain_summary(
            protocol_strength,
            left_protocol="query_only",
            right_protocol="oracle_state",
            probes=FOUNDATION_MEMORY_SENSITIVE_PROBES,
            metric="retrieval.recall_at_k",
            min_delta=FOUNDATION_ORACLE_RECALL_GAIN_MIN,
        )
        checks.update(
            {
                "oracle_requires_memory_task_gain": _check(
                    oracle_state["requires_memory_task_success"]
                    >= query_only["requires_memory_task_success"] + 0.20,
                    {
                        "query_only": query_only["requires_memory_task_success"],
                        "oracle_state": oracle_state["requires_memory_task_success"],
                    },
                    "oracle_state >= query_only + 0.20",
                ),
                "oracle_requires_memory_recall_gain": _check(
                    oracle_state["requires_memory_recall_at_k"]
                    >= query_only["requires_memory_recall_at_k"] + 0.20,
                    {
                        "query_only": query_only["requires_memory_recall_at_k"],
                        "oracle_state": oracle_state["requires_memory_recall_at_k"],
                    },
                    "oracle_state >= query_only + 0.20",
                ),
                "oracle_requires_memory_amq_gain": _check(
                    oracle_state["requires_memory_amq"] >= query_only["requires_memory_amq"] + 0.15,
                    {
                        "query_only": query_only["requires_memory_amq"],
                        "oracle_state": oracle_state["requires_memory_amq"],
                    },
                    "oracle_state >= query_only + 0.15",
                ),
                "oracle_causal_gain_visible": _check(
                    oracle_state["memory_dependence_proxy"] >= query_only["memory_dependence_proxy"] + 0.20
                    and oracle_state["memory_dependence_proxy"] >= 0.20,
                    {
                        "query_only": query_only["memory_dependence_proxy"],
                        "oracle_state": oracle_state["memory_dependence_proxy"],
                    },
                    "oracle_state counterfactual dependence >= query_only + 0.20 and >= 0.20",
                ),
                "oracle_control_task_preserved": _check(
                    oracle_state["no_memory_required_task_success"]
                    >= query_only["no_memory_required_task_success"] - 0.10,
                    {
                        "query_only": query_only["no_memory_required_task_success"],
                        "oracle_state": oracle_state["no_memory_required_task_success"],
                    },
                    "oracle_state control task >= query_only - 0.10",
                ),
                "oracle_safety_not_collapse": _check(
                    oracle_state["safety_pass"] >= query_only["safety_pass"] - 0.10,
                    {
                        "query_only": query_only["safety_pass"],
                        "oracle_state": oracle_state["safety_pass"],
                    },
                    "oracle_state safety >= query_only - 0.10",
                ),
                "oracle_memory_sensitive_probe_task_gains_visible": _check(
                    oracle_task_gain["num_passing_probes"] >= 4,
                    oracle_task_gain,
                    ">= 4 memory-sensitive probes show oracle_state - query_only task gain >= 0.20",
                ),
                "oracle_memory_sensitive_probe_recall_gains_visible": _check(
                    oracle_recall_gain["num_passing_probes"] >= 4,
                    oracle_recall_gain,
                    ">= 4 memory-sensitive probes show oracle_state - query_only recall gain >= 0.20",
                ),
            }
        )

    if query_only is not None and full_history is not None:
        full_history_task_gain = _protocol_strength_probe_gain_summary(
            protocol_strength,
            left_protocol="query_only",
            right_protocol="full_history",
            probes=FOUNDATION_MEMORY_SENSITIVE_PROBES,
            metric="task.task_success",
            min_delta=FOUNDATION_FULL_HISTORY_TASK_GAIN_MIN,
        )
        full_history_recall_gain = _protocol_strength_probe_gain_summary(
            protocol_strength,
            left_protocol="query_only",
            right_protocol="full_history",
            probes=FOUNDATION_MEMORY_SENSITIVE_PROBES,
            metric="retrieval.recall_at_k",
            min_delta=FOUNDATION_FULL_HISTORY_RECALL_GAIN_MIN,
        )
        checks.update(
            {
                "full_history_requires_memory_task_gain": _check(
                    full_history["requires_memory_task_success"]
                    >= query_only["requires_memory_task_success"] + 0.10,
                    {
                        "query_only": query_only["requires_memory_task_success"],
                        "full_history": full_history["requires_memory_task_success"],
                    },
                    "full_history >= query_only + 0.10",
                ),
                "full_history_requires_memory_recall_gain": _check(
                    full_history["requires_memory_recall_at_k"]
                    >= query_only["requires_memory_recall_at_k"] + 0.15,
                    {
                        "query_only": query_only["requires_memory_recall_at_k"],
                        "full_history": full_history["requires_memory_recall_at_k"],
                    },
                    "full_history >= query_only + 0.15",
                ),
                "full_history_causal_gain_nonzero": _check(
                    full_history["memory_dependence_proxy"] >= query_only["memory_dependence_proxy"] + 0.05,
                    {
                        "query_only": query_only["memory_dependence_proxy"],
                        "full_history": full_history["memory_dependence_proxy"],
                    },
                    "full_history counterfactual dependence >= query_only + 0.05",
                ),
                "full_history_memory_sensitive_probe_task_gains_visible": _check(
                    full_history_task_gain["num_passing_probes"] >= 4,
                    full_history_task_gain,
                    ">= 4 memory-sensitive probes show full_history - query_only task gain >= 0.20",
                ),
                "full_history_memory_sensitive_probe_recall_gains_visible": _check(
                    full_history_recall_gain["num_passing_probes"] >= 4,
                    full_history_recall_gain,
                    ">= 4 memory-sensitive probes show full_history - query_only recall gain >= 0.15",
                ),
            }
        )

    passed_checks = sum(1 for item in checks.values() if item["passed"])
    failed_checks = sum(1 for item in checks.values() if not item["passed"])
    status = "passed" if not errors and failed_checks == 0 else "failed"
    return {
        "schema_version": FOUNDATION_VALIDATION_AUDIT_SCHEMA_VERSION,
        "status": status,
        "cohort_id": cohort_id or _first_nonempty(cohort_signatures.values()) or None,
        "expected_benchmark_id": expected_benchmark or None,
        "protocol_reports": {
            protocol: {
                "path": sources_by_protocol.get(protocol),
                "system_id": reports_by_protocol.get(protocol, {}).get("system_id"),
                "benchmark_id": benchmark_ids.get(protocol),
            }
            for protocol in FOUNDATION_PROTOCOLS
            if protocol in reports_by_protocol
        },
        "protocol_metrics": protocol_metrics,
        "protocol_strength": protocol_strength,
        "checks": checks,
        "summary": {
            "protocols_present": sorted(reports_by_protocol),
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "warnings": len(warnings),
            "errors": len(errors),
        },
        "warnings": warnings,
        "errors": errors,
    }


def write_foundation_protocol_audit(
    output: str | Path,
    report_paths: Iterable[str | Path],
    *,
    expected_benchmark_id: str | None = None,
    cohort_id: str | None = None,
    require_full_history: bool = False,
) -> dict[str, Any]:
    report = validate_foundation_protocol_reports(
        report_paths,
        expected_benchmark_id=expected_benchmark_id,
        cohort_id=cohort_id,
        require_full_history=require_full_history,
    )
    report = localize_report_contract(
        report,
        output_path=output,
        project_root_hints=tuple(report_paths),
    )
    write_json(output, report)
    return report


def _infer_protocol(report: dict[str, Any], path: Path) -> str | None:
    candidates = (str(report.get("system_id", "")), path.stem)
    for candidate in candidates:
        match = re.search(r"(^|[:_\-])(query_only|full_history|oracle_state)(?=$|[:_\-])", candidate)
        if match:
            return str(match.group(2))
    return None


def _cohort_signature(system_id: str) -> str:
    if not system_id:
        return ""
    signature = re.sub(r"(^|[:_\-])(query_only|full_history|oracle_state)(?=$|[:_\-])", r"\1", system_id)
    signature = re.sub(r"[:_\-]{2,}", "-", signature)
    return signature.strip(":-_")


def _protocol_metrics(report: dict[str, Any]) -> dict[str, float]:
    aggregate = report.get("aggregate", {})
    by_requirement = report.get("by_memory_requirement", {})
    requires_memory = by_requirement.get("requires_memory", {}) if isinstance(by_requirement, dict) else {}
    no_memory_required = by_requirement.get("no_memory_required", {}) if isinstance(by_requirement, dict) else {}
    counterfactual = report.get("counterfactual", {})
    return {
        "num_scored_queries": _num(aggregate.get("num_scored_queries")),
        "requires_memory_num_queries": _num(requires_memory.get("num_scored_queries")),
        "requires_memory_amq": _num(requires_memory.get("lifecycle.amq")),
        "requires_memory_task_success": _num(requires_memory.get("task.task_success")),
        "requires_memory_recall_at_k": _num(requires_memory.get("retrieval.recall_at_k")),
        "no_memory_required_num_queries": _num(no_memory_required.get("num_scored_queries")),
        "no_memory_required_amq": _num(no_memory_required.get("lifecycle.amq")),
        "no_memory_required_task_success": _num(no_memory_required.get("task.task_success")),
        "safety_pass": _num(aggregate.get("safety.safety_pass")),
        "memory_dependence_proxy": _num(counterfactual.get("memory_dependence_proxy")),
    }


def _protocol_strength_probe_gain_summary(
    protocol_strength: dict[str, Any],
    *,
    left_protocol: str,
    right_protocol: str,
    probes: Iterable[str],
    metric: str,
    min_delta: float,
) -> dict[str, Any]:
    pair = _protocol_strength_pair(protocol_strength, left_protocol=left_protocol, right_protocol=right_protocol)
    probe_list = [str(probe) for probe in probes]
    deltas: dict[str, float | None] = {}
    passing: list[str] = []
    for probe in probe_list:
        value = None
        if isinstance(pair, dict):
            by_probe = pair.get("by_probe_type_deltas")
            if isinstance(by_probe, dict):
                probe_metrics = by_probe.get(probe)
                if isinstance(probe_metrics, dict):
                    raw_value = probe_metrics.get(metric)
                    if isinstance(raw_value, (int, float)):
                        value = float(raw_value)
        deltas[probe] = value
        if value is not None and value >= min_delta:
            passing.append(probe)
    return {
        "left_protocol": left_protocol,
        "right_protocol": right_protocol,
        "metric": metric,
        "min_delta": min_delta,
        "probes": probe_list,
        "probe_deltas": deltas,
        "passing_probes": passing,
        "num_passing_probes": len(passing),
    }


def _protocol_strength_pair(
    protocol_strength: dict[str, Any],
    *,
    left_protocol: str,
    right_protocol: str,
) -> dict[str, Any] | None:
    pairwise = protocol_strength.get("pairwise")
    if not isinstance(pairwise, list):
        return None
    for pair in pairwise:
        if not isinstance(pair, dict):
            continue
        left = pair.get("left_protocol")
        right = pair.get("right_protocol")
        if left == left_protocol and right == right_protocol:
            return pair
        if left == right_protocol and right == left_protocol:
            return {
                **pair,
                "left_protocol": left_protocol,
                "right_protocol": right_protocol,
                "aggregate_deltas": _negate_protocol_deltas(pair.get("aggregate_deltas")),
                "by_probe_type_deltas": _negate_nested_protocol_deltas(pair.get("by_probe_type_deltas")),
            }
    return None


def _negate_protocol_deltas(payload: Any) -> dict[str, float | None]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, float | None] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            result[str(key)] = -float(value)
        else:
            result[str(key)] = None
    return result


def _negate_nested_protocol_deltas(payload: Any) -> dict[str, dict[str, float | None]]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, dict[str, float | None]] = {}
    for key, value in payload.items():
        result[str(key)] = _negate_protocol_deltas(value)
    return result


def _consistent_numeric(protocol_metrics: dict[str, dict[str, float]], field: str) -> bool:
    values = {metrics.get(field) for metrics in protocol_metrics.values()}
    return len(values) <= 1


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _first_nonempty(values: Iterable[str]) -> str:
    for value in values:
        if value:
            return value
    return ""


def _check(passed: bool, actual: Any, expected: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "actual": actual, "expected": expected}
