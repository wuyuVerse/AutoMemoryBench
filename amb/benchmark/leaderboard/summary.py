"""Build local leaderboard summaries from evaluation reports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from amb.benchmark.artifact_contract import localize_report_contract
from amb.benchmark.schemas.io import read_json, write_json


LEADERBOARD_FIELDS = (
    "rank",
    "system_id",
    "benchmark_id",
    "release_split",
    "num_scored_queries",
    "amq",
    "task_success",
    "retrieval_recall_at_k",
    "evidence_complete",
    "safety_pass",
    "memory_dependence_proxy",
    "latency_ms",
    "retrieval_latency_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "report_path",
)


def build_leaderboard_summary(report_paths: Iterable[str | Path]) -> dict[str, Any]:
    rows = [_row_from_report(Path(path), read_json(path)) for path in report_paths]
    ranked = sorted(
        rows,
        key=lambda row: (
            _sort_value(row["amq"]),
            _sort_value(row["safety_pass"]),
            _sort_value(row["task_success"]),
            -_sort_value(row["total_tokens"], lower_is_better=True),
            row["system_id"],
        ),
        reverse=True,
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return {
        "leaderboard_schema_version": "amst-leaderboard-v1",
        "ranking_metric": "amq",
        "ranking_direction": "higher_is_better",
        "num_systems": len(ranked),
        "fields": list(LEADERBOARD_FIELDS),
        "rows": ranked,
    }


def write_leaderboard_summary(
    report_paths: Iterable[str | Path],
    output_json: str | Path,
    *,
    output_csv: str | Path | None = None,
) -> dict[str, Any]:
    summary = build_leaderboard_summary(report_paths)
    summary = localize_report_contract(
        summary,
        output_path=output_json,
        project_root_hints=tuple(report_paths),
    )
    write_json(output_json, summary)
    if output_csv is not None:
        _write_csv(Path(output_csv), summary["rows"])
    return summary


def _row_from_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    aggregate = report.get("aggregate", {})
    counterfactual = report.get("counterfactual", {})
    input_tokens = _metric(aggregate, "efficiency.input_tokens")
    output_tokens = _metric(aggregate, "efficiency.output_tokens")
    return {
        "rank": None,
        "system_id": str(report.get("system_id", "unknown")),
        "benchmark_id": str(report.get("benchmark_id", "unknown")),
        "release_split": str(report.get("release_split", "unknown")),
        "num_scored_queries": _metric(aggregate, "num_scored_queries"),
        "amq": _metric(aggregate, "lifecycle.amq"),
        "task_success": _metric(aggregate, "task.task_success"),
        "retrieval_recall_at_k": _metric(aggregate, "retrieval.recall_at_k"),
        "evidence_complete": _metric(aggregate, "retrieval.evidence_complete"),
        "safety_pass": _metric(aggregate, "safety.safety_pass"),
        "memory_dependence_proxy": _metric(counterfactual, "memory_dependence_proxy"),
        "latency_ms": _metric(aggregate, "efficiency.latency_ms"),
        "retrieval_latency_ms": _metric(aggregate, "efficiency.retrieval_latency_ms"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": _sum_optional(input_tokens, output_tokens),
        "report_path": str(path),
    }


def _metric(values: dict[str, Any], key: str) -> float | int | None:
    value = values.get(key)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    return None


def _sum_optional(first: float | int | None, second: float | int | None) -> float | int | None:
    if first is None or second is None:
        return None
    return first + second


def _sort_value(value: Any, *, lower_is_better: bool = False) -> float:
    if isinstance(value, (int, float)):
        return -float(value) if lower_is_better else float(value)
    return float("-inf")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LEADERBOARD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
