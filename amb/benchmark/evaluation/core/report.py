"""Report helpers."""

from __future__ import annotations

from typing import Any


def print_validation(errors: tuple[str, ...], warnings: tuple[str, ...]) -> str:
    lines: list[str] = []
    if not errors and not warnings:
        return "Validation OK"
    if errors:
        lines.append("Errors:")
        lines.extend(f"- {item}" for item in errors)
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {item}" for item in warnings)
    return "\n".join(lines)


def compact_summary(report: dict[str, Any]) -> str:
    aggregate = report.get("aggregate", {})
    fields = [
        "lifecycle.amq",
        "task.task_success",
        "retrieval.recall_at_k",
        "write.write_f1",
        "update.temporal_validity",
        "safety.safety_pass",
    ]
    lines = [f"system_id: {report.get('system_id')}", f"benchmark_id: {report.get('benchmark_id')}"]
    for field in fields:
        value = aggregate.get(field)
        if isinstance(value, (int, float)):
            lines.append(f"{field}: {value:.4f}")
    lines.append(f"num_scored_queries: {aggregate.get('num_scored_queries', 0)}")
    if report.get("missing_predictions"):
        lines.append(f"missing_predictions: {len(report['missing_predictions'])}")
    if report.get("extra_predictions"):
        lines.append(f"extra_predictions: {len(report['extra_predictions'])}")
    if report.get("duplicate_predictions"):
        lines.append(f"duplicate_predictions: {len(report['duplicate_predictions'])}")
    return "\n".join(lines)
