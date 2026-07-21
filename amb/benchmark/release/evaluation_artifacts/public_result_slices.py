"""Canonical required-slice result artifacts for AutoMemoryBench public releases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from amb.benchmark.evaluation.scoring import aggregate_reports, counterfactual_report
from amb.benchmark.artifact_contract import localize_report_contract
from amb.benchmark.schemas.io import read_json, write_json

PUBLIC_RESULT_SLICES_SCHEMA_VERSION = "amst-public-result-slices-v1"
REQUIRED_SLICE_IDS = ("requires_memory", "no_memory_required", "counterfactual")
REQUIRED_ANCHOR_SYSTEMS = ("no_memory", "oracle_memory")
REQUIRED_COMMON_METRIC_FIELDS = (
    "num_scored_queries",
    "lifecycle.amq",
    "task.task_success",
    "safety.safety_pass",
)
COUNTERFACTUAL_REQUIRED_METRIC_FIELDS = (
    *REQUIRED_COMMON_METRIC_FIELDS,
    "counterfactual.memory_dependence_proxy",
)


def build_public_result_slice_artifacts(
    report_paths: dict[str, str | Path],
    *,
    benchmark_id: str,
    release_split: str = "public_test",
    json_output_path: str | Path,
    markdown_output_path: str | Path,
) -> dict[str, Any]:
    """Build machine-readable and markdown required-slice result artifacts."""

    payload = summarize_public_result_slices(
        report_paths,
        benchmark_id=benchmark_id,
        release_split=release_split,
    )
    json_path = Path(json_output_path)
    markdown_path = Path(markdown_output_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    payload = localize_report_contract(
        payload,
        output_path=json_path,
        project_root_hints=tuple(report_paths.values()),
    )
    write_json(json_path, payload)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    payload["json_path"] = str(json_path)
    payload["markdown_path"] = str(markdown_path)
    return payload


def summarize_public_result_slices(
    report_paths: dict[str, str | Path],
    *,
    benchmark_id: str,
    release_split: str = "public_test",
) -> dict[str, Any]:
    """Summarize the canonical required slices from scored public baseline reports."""

    rows: list[dict[str, Any]] = []
    systems_seen: list[str] = []
    for system_id, raw_path in sorted(report_paths.items()):
        report_path = Path(raw_path)
        report = read_json(report_path)
        systems_seen.append(system_id)
        query_reports = list(report.get("queries", ()))
        for slice_id in REQUIRED_SLICE_IDS:
            slice_queries = _slice_query_reports(query_reports, slice_id)
            aggregate = aggregate_reports(slice_queries) if slice_queries else {}
            counterfactual = counterfactual_report(slice_queries) if slice_queries else {}
            rows.append(
                {
                    "system_id": str(report.get("system_id", system_id)),
                    "slice_id": slice_id,
                    "num_scored_queries": int(aggregate.get("num_scored_queries", 0) or 0),
                    "lifecycle.amq": _float_or_none(aggregate.get("lifecycle.amq")),
                    "task.task_success": _float_or_none(aggregate.get("task.task_success")),
                    "safety.safety_pass": _float_or_none(aggregate.get("safety.safety_pass")),
                    "counterfactual.memory_dependence_proxy": _float_or_none(counterfactual.get("memory_dependence_proxy")),
                    "source_report_path": str(report_path),
                }
            )

    checks = _validate_rows(rows, systems_seen)
    status = "passed" if all(item["passed"] for item in checks.values()) else "failed"
    return {
        "schema_version": PUBLIC_RESULT_SLICES_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "release_split": release_split,
        "required_slices": list(REQUIRED_SLICE_IDS),
        "required_anchor_systems": list(REQUIRED_ANCHOR_SYSTEMS),
        "required_metric_fields": list(REQUIRED_COMMON_METRIC_FIELDS),
        "required_metric_fields_by_slice": {
            "counterfactual": list(COUNTERFACTUAL_REQUIRED_METRIC_FIELDS),
        },
        "systems": sorted(systems_seen),
        "rows": rows,
        "checks": checks,
        "status": status,
        "summary": {
            "num_systems": len(set(systems_seen)),
            "num_rows": len(rows),
            "num_slices": len(REQUIRED_SLICE_IDS),
        },
    }


def validate_public_result_slices_payload(payload: dict[str, Any]) -> tuple[str, ...]:
    checks = payload.get("checks", {})
    errors: list[str] = []
    if payload.get("schema_version") != PUBLIC_RESULT_SLICES_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PUBLIC_RESULT_SLICES_SCHEMA_VERSION}")
    if payload.get("status") != "passed":
        errors.append("status must be passed")
    for check_id in (
        "required_slices_present",
        "required_anchor_systems_present",
        "all_rows_have_required_metrics",
        "all_rows_have_positive_num_scored_queries",
        "all_expected_rows_present",
    ):
        item = checks.get(check_id)
        if not isinstance(item, dict) or not bool(item.get("passed")):
            errors.append(f"{check_id} must be passed")
    return tuple(errors)


def _slice_query_reports(query_reports: list[dict[str, Any]], slice_id: str) -> list[dict[str, Any]]:
    if slice_id == "counterfactual":
        return [item for item in query_reports if item.get("counterfactual_group_id")]
    return [item for item in query_reports if item.get("memory_requirement") == slice_id]


def _validate_rows(rows: list[dict[str, Any]], systems_seen: list[str]) -> dict[str, dict[str, Any]]:
    slice_ids = {str(row.get("slice_id")) for row in rows}
    system_ids = {str(row.get("system_id")) for row in rows}
    missing_metrics = [
        {
            "system_id": row.get("system_id"),
            "slice_id": row.get("slice_id"),
            "missing_fields": [field for field in _required_metric_fields_for_slice(str(row.get("slice_id"))) if row.get(field) is None],
        }
        for row in rows
        if any(row.get(field) is None for field in _required_metric_fields_for_slice(str(row.get("slice_id"))))
    ]
    non_positive = [
        {
            "system_id": row.get("system_id"),
            "slice_id": row.get("slice_id"),
            "num_scored_queries": row.get("num_scored_queries"),
        }
        for row in rows
        if int(row.get("num_scored_queries", 0) or 0) <= 0
    ]
    return {
        "required_slices_present": _check(
            slice_ids >= set(REQUIRED_SLICE_IDS),
            sorted(slice_ids),
            list(REQUIRED_SLICE_IDS),
        ),
        "required_anchor_systems_present": _check(
            system_ids >= set(REQUIRED_ANCHOR_SYSTEMS),
            sorted(system_ids),
            list(REQUIRED_ANCHOR_SYSTEMS),
        ),
        "all_rows_have_required_metrics": _check(
            not missing_metrics,
            missing_metrics[:20],
            {
                "default": list(REQUIRED_COMMON_METRIC_FIELDS),
                "counterfactual": list(COUNTERFACTUAL_REQUIRED_METRIC_FIELDS),
            },
        ),
        "all_rows_have_positive_num_scored_queries": _check(
            not non_positive,
            non_positive[:20],
            "> 0 for every row",
        ),
        "all_expected_rows_present": _check(
            len(rows) == len(set(systems_seen)) * len(REQUIRED_SLICE_IDS),
            len(rows),
            len(set(systems_seen)) * len(REQUIRED_SLICE_IDS),
        ),
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    benchmark_id = str(payload.get("benchmark_id", "release"))
    release_split = str(payload.get("release_split", "public_test"))
    rows = list(payload.get("rows", ()))
    sections: list[str] = [
        f"# AutoMemoryBench {benchmark_id} Required Slice Tables",
        "",
        f"- release_split: `{release_split}`",
        f"- status: `{payload.get('status', 'unknown')}`",
        "",
    ]
    for slice_id in REQUIRED_SLICE_IDS:
        sections.append(f"## {slice_id}")
        sections.append("")
        sections.append("| System | Queries | AMQ | Task | Safety | MD Proxy |")
        sections.append("|---|---:|---:|---:|---:|---:|")
        for row in [item for item in rows if item.get("slice_id") == slice_id]:
            sections.append(
                "| "
                f"{row['system_id']} | {row['num_scored_queries']} | "
                f"{_fmt(row['lifecycle.amq'])} | {_fmt(row['task.task_success'])} | "
                f"{_fmt(row['safety.safety_pass'])} | {_fmt(row['counterfactual.memory_dependence_proxy'])} |"
            )
        sections.append("")
    return "\n".join(sections)


def _required_metric_fields_for_slice(slice_id: str) -> tuple[str, ...]:
    if slice_id == "counterfactual":
        return COUNTERFACTUAL_REQUIRED_METRIC_FIELDS
    return REQUIRED_COMMON_METRIC_FIELDS


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    number = _float_or_none(value)
    return "n/a" if number is None else f"{number:.4f}"


def _check(passed: bool, actual: Any, expected: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "actual": actual, "expected": expected}
