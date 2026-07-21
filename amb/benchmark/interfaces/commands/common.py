"""Shared CLI formatting helpers."""

from __future__ import annotations


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def print_generation_summary(summary: dict) -> None:
    for key in (
        "benchmark_id",
        "num_domains",
        "base_scenarios",
        "counterfactual_scenarios",
        "total_case_variants",
        "num_cases",
        "num_queries",
        "num_memories",
        "num_events",
        "num_event_edges",
        "num_state_contracts",
        "base_queries",
        "base_memories",
        "base_events",
        "total_queries_with_counterfactuals",
        "total_memories_with_counterfactuals",
        "total_events_with_counterfactuals",
    ):
        if key in summary:
            print(f"{key}: {summary[key]}")
    checks = summary.get("final_main_scale_checks")
    if checks:
        print(f"final_main_scale_checks_passed: {all(checks.values())}")


def print_release_plan_summary(summary: dict) -> None:
    print(f"benchmark_id: {summary['benchmark_id']}")
    print(f"split_strategy: {summary['split_strategy']}")
    print(f"base_scenarios: {summary['base_scenarios']}")
    print(f"counterfactual_variants_per_base: {summary['counterfactual_variants_per_base']}")
    print(f"case_variants_per_group: {summary['case_variants_per_group']}")
    for split, report in summary["split_reports"].items():
        print(
            f"{split}: base_groups={report['base_groups']} "
            f"case_variants={report['case_variants']} queries={report['queries']}"
        )
    print(f"audit_fraction_actual: {summary['audit_fraction_actual']:.4f}")
    print(f"audit_fraction_target_met: {summary['audit_fraction_target_met']}")
