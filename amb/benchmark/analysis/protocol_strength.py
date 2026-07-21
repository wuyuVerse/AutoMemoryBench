"""Protocol and model-strength comparison for AMST real-validation reports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from amb.benchmark.artifact_contract import localize_report_contract
from amb.benchmark.schemas.io import read_json, write_json

DEFAULT_METRICS = (
    "lifecycle.amq",
    "task.task_success",
    "task.must_include_coverage",
    "retrieval.recall_at_k",
    "compression.compression_quality_soft",
    "evolution.evolution_quality_soft",
    "safety.safety_pass",
)
KNOWN_PROTOCOLS = ("query_only", "full_history", "oracle_state")


def build_protocol_strength_report(
    report_paths: Iterable[str | Path],
    *,
    metrics: Iterable[str] = DEFAULT_METRICS,
) -> dict[str, Any]:
    metric_names = tuple(str(metric) for metric in metrics)
    rows = [_report_row(path, metric_names) for path in report_paths]
    return {
        "analysis_schema_version": "amst-protocol-strength-v1",
        "num_reports": len(rows),
        "metrics": list(metric_names),
        "reports": rows,
        "pairwise": _pairwise(rows, metric_names),
    }


def write_protocol_strength_report(
    report_paths: Iterable[str | Path],
    output: str | Path,
    *,
    metrics: Iterable[str] = DEFAULT_METRICS,
) -> dict[str, Any]:
    report = build_protocol_strength_report(report_paths, metrics=metrics)
    report = localize_report_contract(
        report,
        output_path=output,
        project_root_hints=tuple(report_paths),
    )
    write_json(output, report)
    return report


def _report_row(path: str | Path, metrics: tuple[str, ...]) -> dict[str, Any]:
    source = Path(path)
    report = read_json(source)
    if not isinstance(report, dict):
        raise ValueError(f"{source} is not a JSON object")
    aggregate = report.get("aggregate")
    by_probe_type = report.get("by_probe_type")
    protocol = _infer_protocol(str(report.get("system_id", "")), source)
    return {
        "report_path": str(source),
        "protocol": protocol,
        "system_id": str(report.get("system_id", source.stem)),
        "benchmark_id": str(report.get("benchmark_id", "")),
        "metrics": {
            metric: _numeric(aggregate.get(metric) if isinstance(aggregate, dict) else None)
            for metric in metrics
        },
        "by_probe_type": {
            str(probe): {metric: _numeric(values.get(metric)) for metric in metrics}
            for probe, values in sorted(by_probe_type.items())
            if isinstance(by_probe_type, dict) and isinstance(values, dict)
        }
        if isinstance(by_probe_type, dict)
        else {},
    }


def _pairwise(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            pairs.append(
                {
                    "left_protocol": left.get("protocol"),
                    "left_system_id": left["system_id"],
                    "right_protocol": right.get("protocol"),
                    "right_system_id": right["system_id"],
                    "benchmark_id": left["benchmark_id"] or right["benchmark_id"],
                    "aggregate_deltas": {
                        metric: _delta(left["metrics"].get(metric), right["metrics"].get(metric))
                        for metric in metrics
                    },
                    "by_probe_type_deltas": _by_probe_delta(left["by_probe_type"], right["by_probe_type"], metrics),
                }
            )
    return pairs


def _by_probe_delta(
    left: dict[str, dict[str, float | None]],
    right: dict[str, dict[str, float | None]],
    metrics: tuple[str, ...],
) -> dict[str, dict[str, float | None]]:
    probes = sorted(set(left) | set(right))
    return {
        probe: {
            metric: _delta(left.get(probe, {}).get(metric), right.get(probe, {}).get(metric))
            for metric in metrics
        }
        for probe in probes
    }


def _infer_protocol(system_id: str, path: Path) -> str | None:
    for candidate in (system_id, path.stem):
        match = re.search(r"(^|[:_\\-])(query_only|full_history|oracle_state)(?=$|[:_\\-])", candidate)
        if match:
            return str(match.group(2))
    return None


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return right - left


def _numeric(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
