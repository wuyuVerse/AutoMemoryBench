"""Sharded profile release builder for large AutoMemoryBench datasets."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from amb.benchmark.generation import generate_case_group, plan_case_groups
from amb.benchmark.generation.profiles import generation_profile
from amb.benchmark.generation.summary import expected_generation_summary
from amb.benchmark.generation.types import CaseGroupPlan, GenerationConfig
from amb.benchmark.quality.audit import audit_benchmark
from amb.benchmark.quality.validation import validate_benchmark
from amb.benchmark.release.artifacts import artifact_info, write_audit_template
from amb.benchmark.release.plan_assignments import assign_case_group_plans, group_assignment_ids
from amb.benchmark.release.splits import RELEASE_SPLITS, ReleaseConfig, planned_release_summary
from amb.benchmark.schemas.io import write_json
from amb.benchmark.schemas.models import Benchmark, Case


def build_profile_release_shards(
    profile_id: str,
    output_dir: str | Path,
    release_config: ReleaseConfig | None = None,
) -> dict[str, Any]:
    profile = generation_profile(profile_id)
    return build_sharded_release(
        profile.config,
        output_dir,
        release_config=release_config,
        profile_id=profile.profile_id,
        profile_description=profile.description,
        profile_role=profile.profile_role,
        canonical_final_main=profile.canonical_final_main,
    )


def build_sharded_release(
    generation_config: GenerationConfig,
    output_dir: str | Path,
    *,
    release_config: ReleaseConfig | None = None,
    profile_id: str | None = None,
    profile_description: str | None = None,
    profile_role: str | None = None,
    canonical_final_main: bool = False,
) -> dict[str, Any]:
    """Build split/domain benchmark shards without holding all cases in memory."""

    release_cfg = release_config or ReleaseConfig()
    output = Path(output_dir)
    expected_summary = expected_generation_summary(generation_config)
    release_plan = planned_release_summary(expected_summary, release_cfg)
    split_plan = assign_case_group_plans(
        plan_case_groups(generation_config),
        strategy=release_plan["split_strategy"],
        release_config=release_cfg,
    )

    split_reports = _empty_split_reports()
    domain_reports: dict[str, dict[str, Any]] = {}
    shard_files: dict[str, dict[str, str]] = {split: {} for split in RELEASE_SPLITS}
    shard_artifacts: dict[str, dict[str, Any]] = {split: {} for split in RELEASE_SPLITS}
    audit_templates: dict[str, str] = {}
    audit_template_artifacts: dict[str, Any] = {}
    max_materialized_shard_case_variants = 0

    for split in RELEASE_SPLITS:
        for domain, plans in sorted(split_plan[split].items()):
            cases = _generate_cases(generation_config, plans)
            if not cases:
                continue
            split_benchmark = Benchmark(
                schema_version="1.0.0",
                benchmark_id=f"{generation_config.benchmark_id}-{split}-{domain}",
                name=f"{generation_config.name} {split} {domain}",
                cases=cases,
            )
            validation = validate_benchmark(split_benchmark)
            if validation.errors:
                raise ValueError(f"generated shard {split}/{domain} failed validation: {validation.errors}")

            shard_path = output / "data" / split / "shards" / f"{domain}.json"
            write_json(shard_path, asdict(split_benchmark))
            shard_files[split][domain] = str(shard_path)
            shard_artifacts[split][domain] = artifact_info(shard_path)
            max_materialized_shard_case_variants = max(max_materialized_shard_case_variants, len(cases))

            shard_audit = audit_benchmark(split_benchmark)
            _accumulate_report(split_reports[split], shard_audit)
            _accumulate_domain_report(domain_reports, domain, split, shard_audit)

            if split == "audit_subset":
                template_path = output / "data" / "audit_subset" / "annotation_templates" / f"{domain}.jsonl"
                write_audit_template(template_path, cases)
                audit_templates[domain] = str(template_path)
                audit_template_artifacts[domain] = artifact_info(template_path)

    for report in split_reports.values():
        _finalize_report(report, require_multiple_domains=True)
    for report in domain_reports.values():
        _finalize_report(report, require_multiple_domains=False)
        for split_report in report["split_reports"].values():
            _finalize_report(split_report, require_multiple_domains=False)

    hidden_enrichment_summary = _hidden_enrichment_summary(split_reports)

    manifest = {
        "schema_version": "1.0.0",
        "benchmark_id": generation_config.benchmark_id,
        "profile_id": profile_id,
        "profile_description": profile_description,
        "profile_role": profile_role,
        "canonical_final_main": canonical_final_main,
        "generation_config": asdict(generation_config),
        "expected_generation_summary": expected_summary,
        "release_plan": release_plan,
        "build_timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "builder": {
            "name": "amb.benchmark.sharded_release_builder",
            "schema_version": "1.0.0",
        },
        "release_status": "generated_unreviewed",
        "split_strategy": release_plan["split_strategy"],
        "release_config": asdict(release_cfg),
        "build_metadata": {
            "artifact_layout": "split_domain_shards",
            "materialized_full_benchmark": False,
            "max_materialized_shard_case_variants": max_materialized_shard_case_variants,
        },
        "source_benchmark": None,
        "visibility": {
            "public_dev": "public",
            "public_test": "public",
            "audit_subset": "controlled_public",
            "hidden_test": "private_leaderboard_only",
        },
        "audit_plan": {
            "audit_required": True,
            "audit_fraction_target": release_cfg.audit_fraction,
            "human_audit_status": "template_generated",
            "audit_template_file": None,
            "audit_template_files": audit_templates,
            "audit_annotations_file": None,
            "agreement_metrics": None,
        },
        "split_files": shard_files,
        "split_artifacts": shard_artifacts,
        "audit_template_artifacts": audit_template_artifacts,
        "split_reports": split_reports,
        "domain_reports": domain_reports,
        "hidden_enrichment_summary": hidden_enrichment_summary,
        "group_assignments": group_assignment_ids(split_plan),
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _generate_cases(config: GenerationConfig, plans: tuple[CaseGroupPlan, ...]) -> tuple[Case, ...]:
    return tuple(case for plan in plans for case in generate_case_group(config, plan))


def _empty_split_reports() -> dict[str, dict[str, Any]]:
    return {split: _empty_report() for split in RELEASE_SPLITS}


def _empty_report() -> dict[str, Any]:
    return {
        "num_cases": 0,
        "num_queries": 0,
        "num_memories": 0,
        "num_events": 0,
        "num_state_contracts": 0,
        "domains": {},
        "task_types": {},
        "probe_types": {},
        "query_difficulty_levels": {},
        "query_difficulty_keys": [],
        "memory_types": {},
        "privacy_levels": {},
        "event_types": {},
        "counterfactual_edits": {},
        "stress_families": {},
        "stress_tags": {},
        "renderer_coverage": {},
        "coverage": {
            "memory_required_queries": 0,
            "no_memory_queries": 0,
            "refusal_queries": 0,
            "sensitive_memories": 0,
            "deleted_memories": 0,
            "stale_capable_memories": 0,
            "state_bound_queries": 0,
            "forbidden_probe_queries": 0,
            "governance_cases": 0,
            "counterfactual_cases": 0,
            "cross_subject_cases": 0,
        },
        "quality_gates": {},
        "construction_gates": {},
        "data_quality_gates": {},
        "construction_gates_passed": True,
        "data_quality_gates_passed": True,
        "quality_gates_passed": True,
    }


def _accumulate_domain_report(
    domain_reports: dict[str, dict[str, Any]],
    domain: str,
    split: str,
    shard_audit: dict[str, Any],
) -> None:
    report = domain_reports.setdefault(
        domain,
        {
            "domain": domain,
            **_empty_report(),
            "split_reports": {name: _empty_report() for name in RELEASE_SPLITS},
        },
    )
    _accumulate_report(report, shard_audit)
    _accumulate_report(report["split_reports"][split], shard_audit)


def _accumulate_report(report: dict[str, Any], audit: dict[str, Any]) -> None:
    report["num_cases"] += audit["num_cases"]
    report["num_queries"] += audit["num_queries"]
    report["num_memories"] += audit["num_memories"]
    report["num_events"] += audit["num_events"]
    report["num_state_contracts"] += audit["num_state_contracts"]
    _merge_counts(report["domains"], audit["domains"])
    _merge_counts(report["task_types"], audit["task_types"])
    _merge_counts(report["probe_types"], audit["probe_types"])
    _merge_counts(report["query_difficulty_levels"], audit.get("query_difficulty_levels", {}))
    report["query_difficulty_keys"] = sorted(
        set(report.get("query_difficulty_keys", ())) | set(audit.get("query_difficulty_keys", ()))
    )
    _merge_counts(report["memory_types"], audit["memory_types"])
    _merge_counts(report["privacy_levels"], audit["privacy_levels"])
    _merge_counts(report["event_types"], audit["event_types"])
    _merge_counts(report["counterfactual_edits"], audit["counterfactual_edits"])
    _merge_counts(report["stress_families"], audit.get("stress_families", {}))
    _merge_counts(report["stress_tags"], audit.get("stress_tags", {}))
    _merge_renderer_coverage(report["renderer_coverage"], audit["renderer_coverage"])
    _merge_counts(report["coverage"], audit["coverage"])
    _merge_gate_results(report["construction_gates"], audit["construction_gates"])
    _merge_gate_results(report["data_quality_gates"], audit["data_quality_gates"])


def _finalize_report(report: dict[str, Any], *, require_multiple_domains: bool) -> None:
    coverage = report["coverage"]
    quality_gates = {
        "has_cases": report["num_cases"] > 0,
        "has_queries": report["num_queries"] > 0,
        "has_memory_required_queries": coverage["memory_required_queries"] > 0,
        "has_no_memory_queries": coverage["no_memory_queries"] > 0,
        "has_refusal_queries": coverage["refusal_queries"] > 0,
        "has_sensitive_memories": coverage["sensitive_memories"] > 0,
        "has_deleted_memories": coverage["deleted_memories"] > 0,
        "has_stale_memory_candidates": coverage["stale_capable_memories"] > 0,
        "has_multiple_domains": len(report["domains"]) >= 2 if require_multiple_domains else True,
        "has_multiple_task_types": len(report["task_types"]) >= 3,
        "has_multiple_memory_types": len(report["memory_types"]) >= 3,
    }
    report["quality_gates"] = quality_gates
    report["quality_gates_passed"] = all(quality_gates.values())
    report["construction_gates_passed"] = all(report["construction_gates"].values()) if report["construction_gates"] else False
    report["data_quality_gates_passed"] = all(report["data_quality_gates"].values()) if report["data_quality_gates"] else False


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def _merge_gate_results(target: dict[str, bool], source: dict[str, bool]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, True) and bool(value)


def _merge_renderer_coverage(target: dict[str, Any], source: dict[str, Any]) -> None:
    for renderer, coverage in source.items():
        merged = target.setdefault(
            renderer,
            {
                "source_event_types": {},
                "num_source_events": 0,
                "provenance_field": coverage.get("provenance_field", "source_event_id"),
            },
        )
        _merge_counts(merged["source_event_types"], coverage.get("source_event_types", {}))
        merged["num_source_events"] = merged.get("num_source_events", 0) + int(coverage.get("num_source_events", 0))


def _hidden_enrichment_summary(split_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hidden = split_reports.get("hidden_test", {})
    public_test = split_reports.get("public_test", {})
    public_dev = split_reports.get("public_dev", {})
    return {
        "status": "enabled",
        "objective": "increase governance, counterfactual, and cross-subject stress density in hidden_test",
        "split_comparison": {
            "hidden_test": _split_enrichment_metrics(hidden),
            "public_test": _split_enrichment_metrics(public_test),
            "public_dev": _split_enrichment_metrics(public_dev),
        },
        "checks": {
            "counterfactual_share_gt_public_test": _share(hidden, "counterfactual_cases", "num_cases")
            > _share(public_test, "counterfactual_cases", "num_cases"),
            "governance_share_gt_public_test": _share(hidden, "governance_cases", "num_cases")
            > _share(public_test, "governance_cases", "num_cases"),
            "cross_subject_share_gt_public_test": _share(hidden, "cross_subject_cases", "num_cases")
            > _share(public_test, "cross_subject_cases", "num_cases"),
        },
        "hidden_stress_families": dict(sorted(hidden.get("stress_families", {}).items())),
        "hidden_stress_tags": dict(sorted(hidden.get("stress_tags", {}).items())),
    }


def _split_enrichment_metrics(report: dict[str, Any]) -> dict[str, Any]:
    coverage = report.get("coverage", {})
    return {
        "num_cases": int(report.get("num_cases", 0)),
        "governance_cases": int(coverage.get("governance_cases", 0)),
        "counterfactual_cases": int(coverage.get("counterfactual_cases", 0)),
        "cross_subject_cases": int(coverage.get("cross_subject_cases", 0)),
        "governance_share": _share(report, "governance_cases", "num_cases"),
        "counterfactual_share": _share(report, "counterfactual_cases", "num_cases"),
        "cross_subject_share": _share(report, "cross_subject_cases", "num_cases"),
    }


def _share(report: dict[str, Any], numerator_field: str, denominator_field: str) -> float:
    coverage = report.get("coverage", {})
    numerator = int(coverage.get(numerator_field, 0))
    denominator = int(report.get(denominator_field, 0))
    if denominator <= 0:
        return 0.0
    return numerator / denominator
