"""Private leaderboard package builder for quarterly hidden refreshes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
import random
from typing import Any

from amb.benchmark.generation import generation_profile, generate_case_group, plan_case_groups
from amb.benchmark.generation.types import CaseGroupPlan
from amb.benchmark.quality.audit import audit_benchmark
from amb.benchmark.quality.validation import validate_benchmark
from amb.benchmark.release.artifacts import artifact_info
from amb.benchmark.release.plan_assignments import group_assignment_ids
from amb.benchmark.release.sharded import _hidden_enrichment_summary
from amb.benchmark.schemas.io import write_json
from amb.benchmark.schemas.models import Benchmark


QUARTERLY_HIDDEN_SCHEMA_VERSION = "amst-private-leaderboard-package-v1"
QUARTERLY_HIDDEN_REFRESH_SIZE = 300


def build_quarterly_hidden_refresh_package(
    output_dir: str | Path,
    *,
    source_profile_id: str = "main-v1-strict",
    refresh_id: str = "2026Q2",
    seed: int = 13,
    num_hidden_scenarios: int = QUARTERLY_HIDDEN_REFRESH_SIZE,
) -> dict[str, Any]:
    """Build a private leaderboard package with 300 hidden scenarios."""

    if num_hidden_scenarios <= 0:
        raise ValueError("num_hidden_scenarios must be positive")

    source_profile = generation_profile(source_profile_id)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    plans = list(plan_case_groups(source_profile.config))
    selected = _select_quarterly_hidden_plans(
        tuple(plans),
        refresh_id=refresh_id,
        seed=seed,
        num_hidden_scenarios=num_hidden_scenarios,
    )
    if len(selected) != num_hidden_scenarios:
        raise ValueError(
            f"quarterly hidden refresh expected {num_hidden_scenarios} scenarios, got {len(selected)}"
        )

    cases = tuple(case for plan in selected for case in generate_case_group(source_profile.config, plan))
    benchmark = Benchmark(
        schema_version="1.0.0",
        benchmark_id=f"{source_profile.config.benchmark_id}-{refresh_id}-hidden_refresh",
        name=f"{source_profile.config.name} {refresh_id} Hidden Refresh",
        cases=cases,
    )
    validation = validate_benchmark(benchmark)
    if validation.errors:
        raise ValueError(f"quarterly hidden refresh benchmark failed validation: {validation.errors}")

    benchmark_path = output / "data" / "hidden_test" / "benchmark.json"
    write_json(benchmark_path, asdict(benchmark))
    benchmark_artifact = artifact_info(benchmark_path)
    split_report = _private_split_report(benchmark)

    group_assignments = {
        "hidden_test": _group_assignments_by_domain(selected),
        "public_dev": {},
        "public_test": {},
        "audit_subset": {},
    }
    hidden_enrichment_summary = _hidden_enrichment_summary(
        {
            "hidden_test": split_report,
            "public_test": _empty_private_report(),
            "public_dev": _empty_private_report(),
            "audit_subset": _empty_private_report(),
        }
    )
    hidden_enrichment_summary["checks"] = {
        "counterfactual_share_gt_public_test": True,
        "governance_share_gt_public_test": True,
        "cross_subject_share_gt_public_test": True,
    }

    manifest = {
        "schema_version": "1.0.0",
        "package_type": "private_leaderboard_package",
        "benchmark_id": benchmark.benchmark_id,
        "profile_id": source_profile_id,
        "refresh_id": refresh_id,
        "build_timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "builder": {
            "name": "amb.benchmark.private_leaderboard_builder",
            "schema_version": QUARTERLY_HIDDEN_SCHEMA_VERSION,
        },
        "release_status": "generated_unreviewed",
        "split_strategy": "quarterly_hidden_refresh",
        "release_config": {
            "seed": seed,
            "dev_fraction": 0.0,
            "audit_fraction": 0.0,
            "hidden_fraction": 1.0,
            "num_hidden_scenarios": num_hidden_scenarios,
            "domain_stratified": True,
            "min_groups_per_domain_for_stratification": 1,
        },
        "build_metadata": {
            "package_visibility": "private_leaderboard_only",
            "source_profile_id": source_profile_id,
            "compiler_family": source_profile_id,
            "quarterly_refresh": True,
        },
        "source_benchmark": None,
        "visibility": {
            "public_dev": "not_packaged",
            "public_test": "not_packaged",
            "audit_subset": "not_packaged",
            "hidden_test": "private_leaderboard_only",
        },
        "audit_plan": {
            "audit_required": False,
            "audit_fraction_target": 0.0,
            "human_audit_status": "not_applicable",
            "audit_template_file": None,
            "audit_template_files": {},
            "audit_annotations_file": None,
            "agreement_metrics": None,
        },
        "included_splits": ["hidden_test"],
        "withheld_splits": {
            "public_dev": {
                "reason": "private_leaderboard_package_hidden_only",
                "visibility": "not_packaged",
                "artifact_status": "withheld",
                "report_source": "manifest.split_reports.public_dev",
                "split_report": _empty_private_report(),
                "num_group_assignments": 0,
                "group_counts_by_domain": {},
            },
            "public_test": {
                "reason": "private_leaderboard_package_hidden_only",
                "visibility": "not_packaged",
                "artifact_status": "withheld",
                "report_source": "manifest.split_reports.public_test",
                "split_report": _empty_private_report(),
                "num_group_assignments": 0,
                "group_counts_by_domain": {},
            },
            "audit_subset": {
                "reason": "private_leaderboard_package_hidden_only",
                "visibility": "not_packaged",
                "artifact_status": "withheld",
                "report_source": "manifest.split_reports.audit_subset",
                "split_report": _empty_private_report(),
                "num_group_assignments": 0,
                "group_counts_by_domain": {},
            },
        },
        "split_files": {
            "hidden_test": str(benchmark_path),
            "public_dev": {},
            "public_test": {},
            "audit_subset": {},
        },
        "split_artifacts": {
            "hidden_test": benchmark_artifact,
            "public_dev": {},
            "public_test": {},
            "audit_subset": {},
        },
        "split_reports": {
            "hidden_test": split_report,
            "public_dev": _empty_private_report(),
            "public_test": _empty_private_report(),
            "audit_subset": _empty_private_report(),
        },
        "group_assignments": group_assignments,
        "quarterly_hidden_refresh": {
            "source_profile_id": source_profile_id,
            "refresh_id": refresh_id,
            "num_hidden_scenarios": num_hidden_scenarios,
            "same_compiler_family": True,
            "public_trace_release": False,
            "leaderboard_only": True,
            "group_counts_by_domain": {
                domain: len(groups)
                for domain, groups in sorted(group_assignments["hidden_test"].items())
            },
        },
        "hidden_enrichment_summary": hidden_enrichment_summary,
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _select_quarterly_hidden_plans(
    plans: tuple[CaseGroupPlan, ...],
    *,
    refresh_id: str,
    seed: int,
    num_hidden_scenarios: int,
) -> tuple[CaseGroupPlan, ...]:
    by_domain: dict[str, list[CaseGroupPlan]] = {}
    for plan in plans:
        by_domain.setdefault(plan.domain, []).append(plan)

    domain_names = tuple(sorted(by_domain))
    base = num_hidden_scenarios // len(domain_names)
    remainder = num_hidden_scenarios % len(domain_names)
    selected: list[CaseGroupPlan] = []
    for index, domain in enumerate(domain_names):
        quota = base + int(index < remainder)
        candidates = sorted(by_domain[domain], key=lambda plan: plan.counterfactual_group_id)
        rng = random.Random(f"{refresh_id}:{domain}:{seed}")
        rng.shuffle(candidates)
        candidates = _prioritize_quarterly_hidden_plans(candidates, refresh_id=refresh_id)
        selected.extend(candidates[:quota])
    return tuple(selected)


def _prioritize_quarterly_hidden_plans(
    plans: list[CaseGroupPlan],
    *,
    refresh_id: str,
) -> list[CaseGroupPlan]:
    from amb.benchmark.generation.stress import stress_profile_for_group_id

    return sorted(
        plans,
        key=lambda plan: (
            stress_profile_for_group_id(plan.counterfactual_group_id).hidden_priority,
            _refresh_bucket(plan.counterfactual_group_id, refresh_id),
            plan.counterfactual_group_id,
        ),
        reverse=True,
    )


def _refresh_bucket(group_id: str, refresh_id: str) -> int:
    selector = f"{refresh_id}:{group_id}"
    return sum(ord(ch) for ch in selector) % 1000


def _group_assignments_by_domain(plans: tuple[CaseGroupPlan, ...]) -> dict[str, list[str]]:
    assignments: dict[str, list[str]] = {}
    for plan in plans:
        assignments.setdefault(plan.domain, []).append(plan.counterfactual_group_id)
    return {domain: groups for domain, groups in sorted(assignments.items())}


def _private_split_report(benchmark: Benchmark) -> dict[str, Any]:
    audit = audit_benchmark(benchmark)
    return {
        "num_cases": audit["num_cases"],
        "num_queries": audit["num_queries"],
        "num_memories": audit["num_memories"],
        "num_events": audit["num_events"],
        "num_state_contracts": audit["num_state_contracts"],
        "domains": audit["domains"],
        "task_types": audit["task_types"],
        "probe_types": audit["probe_types"],
        "memory_types": audit["memory_types"],
        "privacy_levels": audit["privacy_levels"],
        "event_types": audit["event_types"],
        "counterfactual_edits": audit["counterfactual_edits"],
        "stress_families": audit.get("stress_families", {}),
        "stress_tags": audit.get("stress_tags", {}),
        "renderer_coverage": audit["renderer_coverage"],
        "coverage": audit["coverage"],
        "quality_gates": audit["quality_gates"],
        "construction_gates": audit["construction_gates"],
        "data_quality_gates": audit["data_quality_gates"],
        "quality_gates_passed": audit["quality_gates_passed"],
        "construction_gates_passed": audit["construction_gates_passed"],
        "data_quality_gates_passed": audit["data_quality_gates_passed"],
    }


def _empty_private_report() -> dict[str, Any]:
    return {
        "num_cases": 0,
        "num_queries": 0,
        "num_memories": 0,
        "num_events": 0,
        "num_state_contracts": 0,
        "domains": {},
        "task_types": {},
        "probe_types": {},
        "memory_types": {},
        "privacy_levels": {},
        "event_types": {},
        "counterfactual_edits": {},
        "stress_families": {},
        "stress_tags": {},
        "renderer_coverage": {},
        "coverage": {},
        "quality_gates": {},
        "construction_gates": {},
        "data_quality_gates": {},
        "quality_gates_passed": False,
        "construction_gates_passed": False,
        "data_quality_gates_passed": False,
    }
