"""Acceptance checks for the final AutoMemoryBench main dataset construction."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from amb.benchmark.artifact_contract import localize_report_contract
from amb.benchmark.generation.domains.counterfactual import RECOMMENDED_COUNTERFACTUAL_AXES
from amb.benchmark.generation.profiles import (
    CANONICAL_FINAL_MAIN_BENCHMARK_ID,
    CANONICAL_FINAL_MAIN_PROFILE_ID,
    COMPATIBILITY_MAIN_PROFILE_IDS,
)
from amb.benchmark.quality.annotation import AUDIT_CHECK_FIELDS
from amb.benchmark.quality.difficulty_calibration import audit_difficulty_calibration_release
from amb.benchmark.quality.domain_construct_validity import audit_domain_construct_validity_release
from amb.benchmark.quality.representative_analysis import summarize_representative_analysis
from amb.benchmark.quality.release_validation import validate_release_artifacts
from amb.benchmark.quality.release_intrinsic_sanity import validate_release_intrinsic_sanity
from amb.benchmark.quality.probe_discriminativeness import audit_probe_discriminativeness_release
from amb.benchmark.quality.human_audit import verify_manifest_human_audit
from amb.benchmark.quality.human_audit_subset import audit_human_audit_subset_release
from amb.benchmark.quality.question_craftsmanship import audit_question_craftsmanship_release
from amb.benchmark.release.public_result_slices import validate_public_result_slices_payload
from amb.benchmark.release.splits import RELEASE_SPLITS
from amb.benchmark.schemas.io import load_benchmark, read_json, write_json

MAIN_DATASET_ACCEPTANCE_SCHEMA_VERSION = "amst-main-dataset-acceptance-v1"
RELEASE_FAMILY_ACCEPTANCE_SCHEMA_VERSION = "amst-release-family-acceptance-v1"

REQUIRED_MAIN_DOMAINS = (
    "personal_assistant",
    "office_collaboration",
    "coding_agent",
    "customer_support",
    "research_assistant",
    "devops_workflow",
    "education_tutoring",
    "multi_party_collaboration",
)

BASE_REQUIRED_PROBE_TYPES = (
    "answer_probe",
    "compression_probe",
    "evolution_probe",
    "forget_probe",
    "governance_probe",
    "planning_probe",
    "retrieval_probe",
    "tool_probe",
    "update_probe",
    "write_probe",
)
STRICT_REQUIRED_PROBE_TYPES = BASE_REQUIRED_PROBE_TYPES + (
    "governed_transfer_probe",
    "scope_contrast_probe",
    "conflict_resolution_probe",
    "cross_session_synthesis_probe",
    "adversarial_state_synthesis_probe",
    "temporal_causal_reconciliation_probe",
    "policy_temporal_state_probe",
    "policy_exception_probe",
    "state_transition_audit_probe",
)
REQUIRED_PROBE_TYPES = STRICT_REQUIRED_PROBE_TYPES

REQUIRED_RENDERERS = ("adversarial", "document", "platform", "tool")
REQUIRED_EVENT_TYPES = (
    "fact_introduction",
    "fact_reinforcement",
    "fact_update",
    "conflict_event",
    "expiry_event",
    "deletion_request",
    "task_result_event",
    "feedback_event",
    "procedural_event",
)
REQUIRED_EVENT_EDGE_TYPES = (
    "temporal_before",
    "supports",
    "updates",
    "contradicts",
    "invalidates",
    "authorizes",
    "forbids",
    "depends_on",
    "same_entity_as",
    "distracts",
)
REQUIRED_SPLIT_CONSTRUCTION_GATES = (
    "has_event_graph",
    "has_event_edges",
    "has_state_contracts",
    "has_state_bound_queries",
    "has_update_events",
    "has_deletion_events",
    "has_governance_events",
    "has_forbidden_probe_queries",
    "has_tool_events",
    "has_distractor_events",
)
REQUIRED_SPLIT_DATA_QUALITY_GATES = (
    "answer_query_leakage",
    "evidence_sufficiency",
    "evidence_necessity",
    "event_state_contract_closure",
    "answer_uniqueness",
    "distractor_validity",
    "governance_closure",
    "counterfactual_consistency",
    "query_construction",
    "oracle_solvability",
    "no_memory_unsolvability",
    "contamination_check",
)
REQUIRED_MEMORY_TYPES = (
    "episodic_memory",
    "semantic_memory",
    "procedural_memory",
    "reflective_memory",
    "working_state_memory",
    "governance_memory",
)
REPRESENTATIVE_BASELINES = ("no_memory", "full_history", "graph_memory", "oracle_memory")
CASE_SCALE_BOUNDS = {
    "events": (20, 80),
    "memories": (12, 40),
    "queries": (8, 21),
}
CURRENT_RELEASE_BASELINES = (
    "no_memory",
    "full_history",
    "dense_memory",
    "hybrid_memory",
    "graph_memory",
    "oracle_memory",
)
REQUIRED_PUBLIC_TEST_SANITY_CHECKS = (
    "required_and_control_slices_present",
    "oracle_high_amq",
    "oracle_high_requires_memory_task",
    "oracle_high_control_task",
    "no_memory_low_task",
    "no_memory_control_task_nontrivial",
    "no_memory_control_gap_visible",
    "no_memory_zero_recall",
    "graph_memory_beats_no_memory_amq",
    "graph_memory_beats_no_memory_requires_memory_task",
    "graph_gain_concentrates_on_requires_memory",
    "full_history_costs_more_than_graph_memory",
    "full_history_is_less_safe_than_graph",
    "full_history_answers_without_causal_dependence",
    "graph_counterfactual_advantage",
    "graph_causal_advantage_over_full_history",
    "failure_mode_diagnostics_passed",
)
REQUIRED_PUBLIC_RESULT_SLICE_CHECKS = (
    "required_slices_present",
    "required_anchor_systems_present",
    "all_rows_have_required_metrics",
    "all_rows_have_positive_num_scored_queries",
    "all_expected_rows_present",
)
REQUIRED_QUESTION_CRAFTSMANSHIP_CHECKS = (
    "all_probe_types_have_blueprints",
    "no_blueprint_violations",
    "all_core_probe_types_covered",
    "all_domains_cover_all_probe_types",
    "all_domains_cover_easy_medium_hard",
    "all_probe_types_respect_expected_difficulty_levels",
    "all_domain_probe_types_respect_expected_difficulty_levels",
    "all_lifecycle_stages_covered",
    "all_domains_cover_all_lifecycle_stages",
    "all_negative_and_safety_modes_covered",
    "all_domains_cover_negative_and_safety_modes",
    "all_failure_localization_modes_covered",
    "all_domains_cover_failure_localization_modes",
    "all_enabled_counterfactual_axes_covered",
    "all_domains_cover_enabled_counterfactual_axes",
    "no_domain_probe_skeleton_collapse",
    "no_prompt_leakage_or_shortcuts",
    "no_prompt_naturalness_issues",
    "no_evidence_contract_gaps",
    "no_gold_support_gaps",
    "no_gold_memory_state_metadata_gaps",
    "all_domains_preserve_gold_memory_state_metadata",
    "no_adversarial_competitor_gaps",
    "no_hard_query_reasoning_gaps",
    "no_temporal_protocol_gaps",
    "all_domains_follow_historical_ingestion_protocol",
    "no_behavior_contract_gaps",
    "no_state_contract_alignment_gaps",
    "counterfactual_groups_share_target_slot",
    "counterfactual_groups_change_target_slot_state",
    "no_counterfactual_comparability_gaps",
)
REQUIRED_QUERY_CONSTRUCTION_CHECKS = (
    "no_query_construction_issues",
    "no_counterfactual_comparability_issues",
    "no_gold_minimality_issues",
    "no_adversarial_competitor_issues",
    "no_state_grounded_competitor_issues",
    "no_hard_query_shortcut_issues",
    "counterfactual_groups_have_multiple_members",
    "counterfactual_groups_have_prompt_signature",
    "counterfactual_groups_single_scoring_rule",
    "counterfactual_groups_single_task_type",
    "counterfactual_groups_single_probe_type",
    "counterfactual_groups_share_target_slot",
    "counterfactual_groups_change_target_slot_state",
)
REQUIRED_PROBE_DISCRIMINATIVENESS_CHECKS = (
    "representative_reports_present",
    "report_benchmark_id_match",
    "core_probe_coverage_complete",
    "oracle_solves_core_probes",
    "oracle_beats_no_memory_on_core_probes",
    "graph_memory_beats_no_memory_on_core_probes",
    "graph_memory_beats_full_history_on_structured_probes",
    "hard_probe_coverage_complete",
    "graph_memory_safety_advantage_visible",
    "representative_reports_have_task_judge_metadata",
    "soft_diagnostic_metrics_present_on_multi_target_probes",
    "soft_diagnostics_distinguish_memory_progress_on_multi_target_probes",
    "strict_failures_expose_partial_credit_signal",
    "compression_probe_uses_explicit_prediction_channel",
)
REQUIRED_DIFFICULTY_CALIBRATION_CHECKS = (
    "representative_reports_present",
    "report_benchmark_id_match",
    "difficulty_levels_complete",
    "difficulty_level_counts_consistent",
    "hard_query_count_ge_easy",
    "oracle_solves_all_levels",
    "no_memory_hard_below_easy",
    "full_history_hard_below_easy",
    "full_history_task_monotonic_by_difficulty",
    "graph_memory_hard_below_easy",
    "graph_memory_hard_beats_no_memory",
    "graph_memory_hard_beats_full_history",
    "oracle_gap_hard_exceeds_easy",
    "oracle_graph_hard_gap_visible",
)
REQUIRED_DOMAIN_CONSTRUCT_VALIDITY_CHECKS = (
    "representative_reports_present",
    "report_benchmark_id_match",
    "domain_coverage_complete",
    "slice_counts_consistent_per_domain",
    "all_domains_have_both_memory_slices",
    "oracle_solvable_per_domain",
    "no_memory_low_requires_memory_task_per_domain",
    "full_history_failure_visible_per_domain",
    "graph_gap_visible_per_domain",
    "oracle_gap_visible_per_domain",
    "control_nontrivial_per_domain",
    "graph_control_solves_per_domain",
    "graph_amq_beats_full_history_per_domain",
    "hard_requires_memory_counts_consistent_per_domain",
    "hard_requires_memory_present_per_domain",
    "oracle_hard_requires_memory_solvable_per_domain",
    "graph_hard_gap_visible_per_domain",
    "graph_hard_amq_beats_full_history_per_domain",
)
REQUIRED_HUMAN_AUDIT_SUBSET_CHECKS = (
    "template_files_present",
    "template_summary_present",
    "template_ready_for_double_annotation",
    "all_expected_domains_covered",
    "per_domain_min_queries_at_least_100",
    "all_required_probe_types_covered",
    "all_domains_cover_required_probe_types",
    "all_domains_cover_easy_medium_hard",
    "all_audit_checks_covered",
    "all_domains_cover_all_audit_checks",
    "counterfactual_axes_covered",
    "all_domains_cover_counterfactual_axes",
    "governance_state_contracts_present",
    "all_domains_include_governance_state_contracts",
    "task_package_present",
    "task_package_progress_errors_empty",
    "task_package_template_files_match",
    "task_package_template_item_count_match",
    "task_package_expected_annotations_match",
    "task_package_declares_two_annotators",
    "task_package_task_files_match_annotators",
    "task_package_identity_digests_complete",
)
REQUIRED_FOUNDATION_VALIDATION_CHECKS = (
    "required_protocols_present",
    "benchmark_id_match",
    "cohort_signature_match",
    "num_scored_queries_consistent",
    "slice_coverage_present",
    "protocol_pairwise_complete",
    "query_only_memory_task_low",
    "query_only_control_task_nontrivial",
    "query_only_control_gap_visible",
    "oracle_requires_memory_task_gain",
    "oracle_requires_memory_recall_gain",
    "oracle_requires_memory_amq_gain",
    "oracle_causal_gain_visible",
    "oracle_control_task_preserved",
    "oracle_safety_not_collapse",
    "oracle_memory_sensitive_probe_task_gains_visible",
    "oracle_memory_sensitive_probe_recall_gains_visible",
)
OPTIONAL_FULL_HISTORY_FOUNDATION_CHECKS = (
    "full_history_requires_memory_task_gain",
    "full_history_requires_memory_recall_gain",
    "full_history_causal_gain_nonzero",
    "full_history_memory_sensitive_probe_task_gains_visible",
    "full_history_memory_sensitive_probe_recall_gains_visible",
)
REQUIRED_HIDDEN_TEST_SANITY_CHECKS = (
    "private_hidden_split_present",
    "hidden_visibility_private",
    "hidden_enrichment_summary_present",
    "hidden_counterfactual_enrichment",
    "hidden_governance_enrichment",
    "hidden_cross_subject_enrichment",
    "required_and_control_slices_present",
    "oracle_high_amq",
    "oracle_high_requires_memory_task",
    "no_memory_control_task_nontrivial",
    "no_memory_low_requires_memory_task",
    "no_memory_zero_requires_memory_recall",
    "graph_memory_high_amq",
    "graph_memory_beats_no_memory_requires_memory_task",
    "graph_memory_beats_no_memory_amq",
    "graph_memory_requires_memory_recall_high",
    "graph_memory_beats_dense_amq",
    "graph_memory_beats_hybrid_amq",
    "full_history_hidden_failure_visible",
    "graph_counterfactual_advantage",
    "graph_counterfactual_high",
    "graph_governance_high",
    "graph_update_temporal_high",
    "graph_procedural_high",
    "failure_mode_diagnostics_passed",
)


def validate_main_dataset_acceptance(
    manifest_path: str | Path,
    *,
    expected_benchmark_id: str = "amst-main-v1",
    expected_profile_id: str = "main-v1",
    run_release_validation: bool = False,
    release_validation_report_path: str | Path | None = None,
    public_release_manifest_path: str | Path | None = None,
    public_release_validation_report_path: str | Path | None = None,
    require_public_release_validation: bool = False,
    require_all_counterfactual_axes: bool = False,
    representative_reports_dir: str | Path | None = None,
    representative_split: str = "public_dev",
    require_representative_baselines: bool = False,
    intrinsic_sanity_report_path: str | Path | None = None,
    require_intrinsic_sanity: bool = False,
    human_audit_verification_report_path: str | Path | None = None,
    require_completed_human_audit: bool = False,
    strong_ai_audit_report_path: str | Path | None = None,
    require_strong_ai_audit: bool = False,
    public_test_sanity_report_path: str | Path | None = None,
    require_public_test_sanity: bool = False,
    public_result_slices_report_path: str | Path | None = None,
    require_public_result_slices: bool = False,
    hidden_test_sanity_report_path: str | Path | None = None,
    require_hidden_test_sanity: bool = False,
    human_audit_subset_report_path: str | Path | None = None,
    require_human_audit_subset: bool = False,
    question_craftsmanship_report_path: str | Path | None = None,
    require_question_craftsmanship: bool = False,
    query_construction_report_path: str | Path | None = None,
    require_query_construction: bool = False,
    probe_discriminativeness_report_path: str | Path | None = None,
    require_probe_discriminativeness: bool = False,
    difficulty_calibration_report_path: str | Path | None = None,
    require_difficulty_calibration: bool = False,
    domain_construct_validity_report_path: str | Path | None = None,
    require_domain_construct_validity: bool = False,
    foundation_validation_report_path: str | Path | None = None,
    require_foundation_validation: bool = False,
    require_query_difficulty: bool = False,
) -> dict[str, Any]:
    """Validate main-dataset scale, split, coverage, and quality-gate evidence.

    This check focuses on the generated dataset itself. It intentionally does
    not require completed human annotation, real external benchmark runs, or
    real memory-system integrations; those are separate completion blockers.
    """

    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []

    _check_equal(checks, errors, "benchmark_id", manifest.get("benchmark_id"), expected_benchmark_id)
    _check_equal(checks, errors, "profile_id", manifest.get("profile_id"), expected_profile_id)
    builder_name = manifest.get("builder", {}).get("name")
    accepted_builder_names = {
        "amb.benchmark.public_release_exporter",
        "amb.benchmark.sharded_release_builder",
        "agent_memory_benchmark.public_release_exporter",
        "agent_memory_benchmark.sharded_release_builder",
    }
    _add_check(
        checks,
        errors,
        "builder",
        builder_name in accepted_builder_names,
        f"builder: expected one of {sorted(accepted_builder_names)!r}, got {builder_name!r}",
        {"actual": builder_name, "expected": sorted(accepted_builder_names)},
    )
    _check_equal(checks, errors, "artifact_layout", manifest.get("build_metadata", {}).get("artifact_layout"), "split_domain_shards")
    _check_equal(checks, errors, "materialized_full_benchmark", manifest.get("build_metadata", {}).get("materialized_full_benchmark"), False)

    summary = manifest.get("expected_generation_summary", {})
    _check_equal(checks, errors, "base_scenarios", summary.get("base_scenarios"), 1200)
    _check_equal(checks, errors, "counterfactual_scenarios_min", _as_int(summary.get("counterfactual_scenarios")) >= 2400, True)
    expected_counterfactual_variants = _expected_counterfactual_variants(expected_profile_id)
    _check_equal(
        checks,
        errors,
        "counterfactual_variants_per_base",
        summary.get("counterfactual_variants_per_base"),
        expected_counterfactual_variants,
    )
    _check_equal(checks, errors, "num_domains", summary.get("num_domains"), 8)
    _check_contains(checks, errors, "domains", summary.get("domains", ()), REQUIRED_MAIN_DOMAINS)
    base_query_scale_check = _base_query_scale_check_id(str(expected_profile_id or manifest.get("profile_id") or ""))
    _check_equal(checks, errors, base_query_scale_check, _scale_check(summary, base_query_scale_check), True)
    _check_equal(checks, errors, "base_memories_in_18k_to_48k", _scale_check(summary, "base_memories_in_18k_to_48k"), True)
    _check_equal(checks, errors, "base_events_in_24k_to_60k", _scale_check(summary, "base_events_in_24k_to_60k"), True)
    axis_coverage = summary.get("counterfactual_axis_coverage", {})
    missing_axes = tuple(str(axis) for axis in axis_coverage.get("missing_recommended_axes", ()))
    actual_profile_id = str(manifest.get("profile_id") or "")
    canonical_profile_alignment = {
        "canonical_final_main_profile_id": CANONICAL_FINAL_MAIN_PROFILE_ID,
        "canonical_final_main_benchmark_id": CANONICAL_FINAL_MAIN_BENCHMARK_ID,
        "compatibility_main_profile_ids": list(COMPATIBILITY_MAIN_PROFILE_IDS),
        "actual_profile_id": actual_profile_id,
        "actual_benchmark_id": str(manifest.get("benchmark_id") or ""),
        "profile_role": manifest.get("profile_role"),
        "manifest_declares_canonical_final_main": bool(manifest.get("canonical_final_main")),
        "is_canonical_final_main_profile": actual_profile_id == CANONICAL_FINAL_MAIN_PROFILE_ID,
        "covers_all_recommended_axes": bool(axis_coverage.get("covers_all_recommended_axes")),
    }
    if actual_profile_id in COMPATIBILITY_MAIN_PROFILE_IDS:
        warnings.append(
            f"profile {actual_profile_id} is treated as a compatibility main release; "
            f"canonical final main profile is {CANONICAL_FINAL_MAIN_PROFILE_ID}"
        )
    if missing_axes:
        warnings.append(f"counterfactual recommended axes are not fully covered: {', '.join(missing_axes)}")
    if require_all_counterfactual_axes:
        _check_equal(
            checks,
            errors,
            "counterfactual_axes_all_recommended",
            bool(axis_coverage.get("covers_all_recommended_axes")),
            True,
        )

    release_plan = manifest.get("release_plan", {})
    _check_equal(checks, errors, "split_strategy", release_plan.get("split_strategy"), "domain_stratified_group_preserving")
    _check_equal(checks, errors, "audit_fraction_target_met", release_plan.get("audit_fraction_target_met"), True)
    _check_equal(checks, errors, "audit_fraction_actual", _as_float(release_plan.get("audit_fraction_actual"), -1.0), 0.1)
    _check_release_plan_counts(checks, errors, release_plan, expected_counterfactual_variants=expected_counterfactual_variants)

    split_reports = manifest.get("split_reports", {})
    for split in RELEASE_SPLITS:
        report = split_reports.get(split)
        if not isinstance(report, dict):
            _add_check(checks, errors, f"{split}.present", False, f"split_reports.{split} is missing")
            continue
        _check_split_report(checks, errors, split, report, profile_id=expected_profile_id)
    _check_hardness_slices(checks, errors, split_reports, expected_profile_id)
    if require_query_difficulty:
        _check_query_difficulty_coverage(checks, errors, split_reports)

    hidden_enrichment = manifest.get("hidden_enrichment_summary")
    if hidden_enrichment is not None:
        _check_equal(checks, errors, "hidden_enrichment.summary_present", True, True)
        _check_equal(checks, errors, "hidden_enrichment.status", hidden_enrichment.get("status"), "enabled")
        for check_id, expected in (
            ("counterfactual_share_gt_public_test", True),
            ("governance_share_gt_public_test", True),
            ("cross_subject_share_gt_public_test", True),
        ):
            _check_equal(
                checks,
                errors,
                f"hidden_enrichment.{check_id}",
                bool(hidden_enrichment.get("checks", {}).get(check_id)),
                expected,
            )

    case_scale_summary = _case_scale_summary(manifest, manifest_file.parent)
    _check_equal(checks, errors, "scenario_scale.summary_present", case_scale_summary is not None, True)
    if case_scale_summary is not None:
        _check_equal(
            checks,
            errors,
            "scenario_scale.all_cases_within_bounds",
            int(case_scale_summary.get("out_of_range_cases", 0)),
            0,
        )
        for field, bounds in CASE_SCALE_BOUNDS.items():
            field_summary = case_scale_summary.get("fields", {}).get(field, {})
            _check_equal(
                checks,
                errors,
                f"scenario_scale.{field}.min_in_range",
                _field_or_default(field_summary.get("min"), -1) >= bounds[0],
                True,
            )
            _check_equal(
                checks,
                errors,
                f"scenario_scale.{field}.max_in_range",
                _field_or_default(field_summary.get("max"), -1) <= bounds[1],
                True,
            )

    core_object_summary = _core_object_summary(manifest, manifest_file.parent)
    _check_equal(checks, errors, "core_objects.summary_present", core_object_summary is not None, True)
    if core_object_summary is not None:
        _check_equal(checks, errors, "core_objects.missing_canonical_form", int(core_object_summary.get("missing_canonical_form", 0)), 0)
        _check_equal(checks, errors, "core_objects.missing_memory_type_alias", int(core_object_summary.get("missing_memory_type_alias", 0)), 0)
        _check_equal(checks, errors, "core_objects.missing_source_trace_ids", int(core_object_summary.get("missing_source_trace_ids", 0)), 0)
        _check_equal(checks, errors, "core_objects.missing_required_governance_rules", int(core_object_summary.get("missing_required_governance_rules", 0)), 0)
        _check_equal(checks, errors, "core_objects.missing_scenario_metadata", int(core_object_summary.get("missing_scenario_metadata", 0)), 0)
        _check_equal(checks, errors, "core_objects.missing_scenario_time_span", int(core_object_summary.get("missing_scenario_time_span", 0)), 0)
        _check_core_event_edge_types(checks, errors, core_object_summary)
        _check_contains(checks, errors, "core_objects.transition_types", core_object_summary.get("transition_types", ()), ("update", "delete", "retain"))
        _check_domain_core_object_state_semantics(checks, errors, core_object_summary)

    audit_plan = manifest.get("audit_plan", {})
    _check_equal(checks, errors, "audit_required", audit_plan.get("audit_required"), True)
    _check_equal(checks, errors, "audit_template_files_present", bool(audit_plan.get("audit_template_files")), True)
    audit_template_summary = _audit_template_summary(audit_plan, manifest_file.parent)
    _check_equal(checks, errors, "audit_subset.template_summary_present", audit_template_summary is not None, True)
    if audit_template_summary is not None:
        _check_equal(
            checks,
            errors,
            "audit_subset.domain_probe_min_100",
            bool(audit_template_summary.get("per_domain_min_queries_at_least_100")),
            True,
        )
        _check_equal(
            checks,
            errors,
            "audit_subset.num_domains",
            int(audit_template_summary.get("num_domains", 0)),
            len(REQUIRED_MAIN_DOMAINS),
        )
        _check_contains(
            checks,
            errors,
            "audit_subset.domain_query_coverage",
            tuple(audit_template_summary.get("per_domain_query_counts", {}).keys()),
            REQUIRED_MAIN_DOMAINS,
        )
    human_audit_verification = None
    if human_audit_verification_report_path is not None or require_completed_human_audit:
        _check_equal(checks, errors, "human_audit.status", audit_plan.get("human_audit_status"), "completed")
        if audit_plan.get("human_audit_status") == "completed":
            _check_equal(checks, errors, "human_audit.audit_annotations_declared", bool(audit_plan.get("audit_annotations_file")), True)
            _check_equal(checks, errors, "human_audit.audit_task_manifest_declared", bool(audit_plan.get("audit_task_manifest_file")), True)
            _check_equal(
                checks,
                errors,
                "human_audit.annotator_attestation_declared",
                bool(audit_plan.get("annotator_attestation_file")),
                True,
            )
            if human_audit_verification_report_path is not None:
                human_audit_verification = read_json(human_audit_verification_report_path)
            elif require_completed_human_audit:
                human_audit_verification = verify_manifest_human_audit(manifest_file)
        _check_equal(
            checks,
            errors,
            "human_audit.present",
            human_audit_verification is not None,
            audit_plan.get("human_audit_status") == "completed",
        )
        if human_audit_verification is not None:
            _check_human_audit_binding(checks, errors, manifest, manifest_file, audit_plan, human_audit_verification)
            _check_equal(checks, errors, "human_audit.verification_ok", human_audit_verification.get("ok"), True)
            _check_equal(
                checks,
                errors,
                "human_audit.num_template_items_positive",
                int(human_audit_verification.get("num_template_items", 0)) > 0,
                True,
            )
            _check_equal(
                checks,
                errors,
                "human_audit.num_annotators_at_least_two",
                int(human_audit_verification.get("num_annotators", 0)) >= 2,
                True,
            )
            agreement_fields = (
                human_audit_verification.get("agreement_metrics", {}).get("fields", {})
                if isinstance(human_audit_verification.get("agreement_metrics"), dict)
                else {}
            )
            semantic_checks = (
                human_audit_verification.get("semantic_checks", {})
                if isinstance(human_audit_verification.get("semantic_checks"), dict)
                else {}
            )
            _check_contains(checks, errors, "human_audit.agreement_fields", tuple(agreement_fields.keys()), AUDIT_CHECK_FIELDS)
            for field in AUDIT_CHECK_FIELDS:
                metrics = agreement_fields.get(field, {})
                applicable_items = int(metrics.get("num_applicable_items", metrics.get("num_items", 0)))
                _check_equal(
                    checks,
                    errors,
                    f"human_audit.{field}.num_applicable_items_positive",
                    applicable_items > 0,
                    True,
                )
                _check_equal(
                    checks,
                    errors,
                    f"human_audit.{field}.num_pairs_positive",
                    int(metrics.get("num_pairs", 0)) > 0 if applicable_items > 0 else True,
                    True,
                )
                _check_equal(
                    checks,
                    errors,
                    f"human_audit.{field}.percent_agreement_present",
                    metrics.get("percent_agreement") is not None if applicable_items > 0 else True,
                    True,
                )
                _check_equal(
                    checks,
                    errors,
                    f"human_audit.{field}.cohen_kappa_present",
                    metrics.get("cohen_kappa") is not None if applicable_items > 0 else True,
                    True,
                )
            _check_contains(
                checks,
                errors,
                "human_audit.semantic_checks",
                tuple(semantic_checks.keys()),
                ("scenario_memory_required_alignment", "counterfactual_target_state_only_alignment"),
            )
            for check_id in ("scenario_memory_required_alignment", "counterfactual_target_state_only_alignment"):
                item = semantic_checks.get(check_id, {})
                _check_equal(checks, errors, f"human_audit.{check_id}.present", bool(item), True)
                if item:
                    _check_equal(checks, errors, f"human_audit.{check_id}.passed", item.get("passed"), True)
                    _check_equal(
                        checks,
                        errors,
                        f"human_audit.{check_id}.num_applicable_items_positive",
                        int(item.get("num_applicable_items", 0)) > 0,
                        True,
                    )
                    _check_equal(
                        checks,
                        errors,
                        f"human_audit.{check_id}.num_majority_mismatches_zero",
                        int(item.get("num_majority_mismatches", 0)),
                        0,
                    )
                    _check_equal(
                        checks,
                        errors,
                        f"human_audit.{check_id}.num_unresolved_items_zero",
                        int(item.get("num_unresolved_items", 0)),
                        0,
                    )
    elif audit_plan.get("human_audit_status") != "completed":
        warnings.append("human audit templates are present, but completed double annotations are still external evidence")

    strong_ai_audit = None
    if strong_ai_audit_report_path is not None or require_strong_ai_audit:
        if strong_ai_audit_report_path is not None:
            strong_ai_audit = read_json(strong_ai_audit_report_path)
        _check_equal(checks, errors, "strong_ai_audit.present", strong_ai_audit is not None, True)
        if strong_ai_audit is not None:
            _check_equal(checks, errors, "strong_ai_audit.status", strong_ai_audit.get("status"), "passed")
            _check_equal(
                checks,
                errors,
                "strong_ai_audit.num_issues",
                int(strong_ai_audit.get("summary", {}).get("num_issues", -1)),
                0,
            )
            _check_equal(
                checks,
                errors,
                "strong_ai_audit.num_failed_checks",
                int(strong_ai_audit.get("summary", {}).get("num_failed_checks", -1)),
                0,
            )
            _check_equal(
                checks,
                errors,
                "strong_ai_audit.claim_boundary_non_human",
                "not independent double-human annotation" in str(strong_ai_audit.get("claim_boundary") or ""),
                True,
            )

    representative_baseline_summary = None
    if representative_reports_dir is not None or require_representative_baselines:
        reports_dir = Path(representative_reports_dir or "reports/examples")
        representative_baseline_summary = _check_representative_baselines(
            checks,
            errors,
            manifest,
            reports_dir=reports_dir,
            split=representative_split,
            require_reports=require_representative_baselines,
        )
        if not require_representative_baselines and not representative_baseline_summary.get("present", False):
            warnings.append("representative baseline reports were not found; effectiveness sanity was not enforced")

    intrinsic_sanity = None
    if intrinsic_sanity_report_path is not None or require_intrinsic_sanity:
        if intrinsic_sanity_report_path is not None:
            intrinsic_sanity = read_json(intrinsic_sanity_report_path)
        else:
            intrinsic_sanity = validate_release_intrinsic_sanity(manifest_file)
        _check_equal(checks, errors, "intrinsic_sanity.present", intrinsic_sanity is not None, True)
        if intrinsic_sanity is not None:
            _check_equal(checks, errors, "intrinsic_sanity.status", intrinsic_sanity.get("status"), "passed")
            _check_artifact_binding(
                checks,
                errors,
                "intrinsic_sanity",
                intrinsic_sanity,
                report_path=intrinsic_sanity_report_path,
                expected_benchmark_id=str(manifest.get("benchmark_id")),
                expected_manifest_path=manifest_file,
            )
            for split in RELEASE_SPLITS:
                summary = intrinsic_sanity.get("split_intrinsic_sanity", {}).get(split, {})
                if int(summary.get("num_cases", 0)) == 0:
                    continue
                for gate in ("oracle_solvability", "no_memory_unsolvability"):
                    _check_equal(
                        checks,
                        errors,
                        f"intrinsic_sanity.{split}.{gate}",
                        bool(summary.get("gates", {}).get(gate)),
                        True,
                    )
        if not require_intrinsic_sanity and intrinsic_sanity is None:
            warnings.append("release intrinsic sanity artifact was not found; oracle/no-memory release-level sanity was not enforced")

    public_test_sanity = None
    if public_test_sanity_report_path is not None or require_public_test_sanity:
        if public_test_sanity_report_path is not None:
            public_test_sanity = read_json(public_test_sanity_report_path)
        _check_equal(checks, errors, "public_test_sanity.present", public_test_sanity is not None, True)
        if public_test_sanity is not None:
            _check_equal(checks, errors, "public_test_sanity.status", public_test_sanity.get("status"), "passed")
            _check_artifact_binding(
                checks,
                errors,
                "public_test_sanity",
                public_test_sanity,
                expected_benchmark_id=f"{manifest.get('benchmark_id')}-public_test",
                expected_release_split="public_test",
            )
            _check_contains(
                checks,
                errors,
                "public_test_sanity.required_checks",
                tuple(public_test_sanity.get("checks", {}).keys()),
                REQUIRED_PUBLIC_TEST_SANITY_CHECKS,
            )
            for check_id, item in sorted(public_test_sanity.get("checks", {}).items()):
                _check_equal(
                    checks,
                    errors,
                    f"public_test_sanity.{check_id}",
                    bool(item.get("passed")),
                    True,
                )
        if not require_public_test_sanity and public_test_sanity is None:
            warnings.append("public-test sanity artifact was not found; public-test machine-readable sanity was not enforced")

    public_result_slices = None
    if public_result_slices_report_path is not None or require_public_result_slices:
        if public_result_slices_report_path is not None:
            public_result_slices = read_json(public_result_slices_report_path)
        _check_equal(checks, errors, "public_result_slices.present", public_result_slices is not None, True)
        if public_result_slices is not None:
            _check_equal(checks, errors, "public_result_slices.status", public_result_slices.get("status"), "passed")
            _check_artifact_binding(
                checks,
                errors,
                "public_result_slices",
                public_result_slices,
                expected_benchmark_id=f"{manifest.get('benchmark_id')}-public_test",
                expected_release_split="public_test",
            )
            _check_contains(
                checks,
                errors,
                "public_result_slices.required_checks",
                tuple(public_result_slices.get("checks", {}).keys()),
                REQUIRED_PUBLIC_RESULT_SLICE_CHECKS,
            )
            for check_id in REQUIRED_PUBLIC_RESULT_SLICE_CHECKS:
                item = public_result_slices.get("checks", {}).get(check_id, {})
                _check_equal(
                    checks,
                    errors,
                    f"public_result_slices.{check_id}",
                    bool(item.get("passed")),
                    True,
                )
            _check_equal(
                checks,
                errors,
                "public_result_slices.payload_valid",
                len(validate_public_result_slices_payload(public_result_slices)) == 0,
                True,
            )
        if not require_public_result_slices and public_result_slices is None:
            warnings.append("public required-slice result artifact was not found; slice-level public reporting completeness was not enforced")

    hidden_test_sanity = None
    if hidden_test_sanity_report_path is not None or require_hidden_test_sanity:
        if hidden_test_sanity_report_path is not None:
            hidden_test_sanity = read_json(hidden_test_sanity_report_path)
        _check_equal(checks, errors, "hidden_test_sanity.present", hidden_test_sanity is not None, True)
        if hidden_test_sanity is not None:
            _check_equal(checks, errors, "hidden_test_sanity.status", hidden_test_sanity.get("status"), "passed")
            _check_artifact_binding(
                checks,
                errors,
                "hidden_test_sanity",
                hidden_test_sanity,
                expected_benchmark_id=f"{manifest.get('benchmark_id')}-hidden_test",
                expected_release_split="hidden_test",
            )
            _check_contains(
                checks,
                errors,
                "hidden_test_sanity.required_checks",
                tuple(hidden_test_sanity.get("checks", {}).keys()),
                REQUIRED_HIDDEN_TEST_SANITY_CHECKS,
            )
            for check_id, item in sorted(hidden_test_sanity.get("checks", {}).items()):
                _check_equal(
                    checks,
                    errors,
                    f"hidden_test_sanity.{check_id}",
                    bool(item.get("passed")),
                    True,
                )
        if not require_hidden_test_sanity and hidden_test_sanity is None:
            warnings.append("hidden-test sanity artifact was not found; private hidden split sanity was not enforced")

    human_audit_subset = None
    if human_audit_subset_report_path is not None or require_human_audit_subset:
        human_audit_subset_manifest_paths = [manifest_file]
        if public_release_manifest_path is not None:
            human_audit_subset_manifest_paths.append(Path(public_release_manifest_path))
        else:
            inferred_public_manifest_path = manifest_file.parent.with_name(f"{manifest_file.parent.name}_public") / manifest_file.name
            if inferred_public_manifest_path.exists():
                human_audit_subset_manifest_paths.append(inferred_public_manifest_path)
        audit_subset_manifest_path = human_audit_subset_manifest_paths[-1]
        if human_audit_subset_report_path is not None:
            human_audit_subset = read_json(human_audit_subset_report_path)
        else:
            human_audit_subset = audit_human_audit_subset_release(audit_subset_manifest_path)
        _check_equal(checks, errors, "human_audit_subset.present", human_audit_subset is not None, True)
        if human_audit_subset is not None:
            _check_equal(checks, errors, "human_audit_subset.status", human_audit_subset.get("status"), "passed")
            _check_artifact_binding(
                checks,
                errors,
                "human_audit_subset",
                human_audit_subset,
                report_path=human_audit_subset_report_path,
                expected_benchmark_id=str(manifest.get("benchmark_id")),
                expected_release_split=None,
                expected_source_type="release_manifest",
                expected_manifest_paths=tuple(human_audit_subset_manifest_paths),
            )
            _check_contains(
                checks,
                errors,
                "human_audit_subset.required_checks",
                tuple(human_audit_subset.get("checks", {}).keys()),
                REQUIRED_HUMAN_AUDIT_SUBSET_CHECKS,
            )
            for check_id in REQUIRED_HUMAN_AUDIT_SUBSET_CHECKS:
                item = human_audit_subset.get("checks", {}).get(check_id, {})
                _check_equal(
                    checks,
                    errors,
                    f"human_audit_subset.{check_id}",
                    bool(item.get("passed")),
                    True,
                )
        if not require_human_audit_subset and human_audit_subset is None:
            warnings.append("human-audit subset audit artifact was not found; pre-label audit-subset coverage was not enforced")

    question_craftsmanship = None
    if question_craftsmanship_report_path is not None or require_question_craftsmanship:
        question_craftsmanship_manifest_paths = [manifest_file]
        if public_release_manifest_path is not None:
            question_craftsmanship_manifest_paths.append(Path(public_release_manifest_path))
        else:
            inferred_public_manifest_path = manifest_file.parent.with_name(f"{manifest_file.parent.name}_public") / manifest_file.name
            if inferred_public_manifest_path.exists():
                question_craftsmanship_manifest_paths.append(inferred_public_manifest_path)
        if question_craftsmanship_report_path is not None:
            question_craftsmanship = read_json(question_craftsmanship_report_path)
        else:
            question_craftsmanship = audit_question_craftsmanship_release(manifest_file)
        _check_equal(checks, errors, "question_craftsmanship.present", question_craftsmanship is not None, True)
        if question_craftsmanship is not None:
            _check_equal(checks, errors, "question_craftsmanship.status", question_craftsmanship.get("status"), "passed")
            _check_artifact_binding(
                checks,
                errors,
                "question_craftsmanship",
                question_craftsmanship,
                report_path=question_craftsmanship_report_path,
                expected_benchmark_id=str(manifest.get("benchmark_id")),
                expected_release_split=None,
                expected_source_type="release_manifest",
                expected_manifest_paths=tuple(question_craftsmanship_manifest_paths),
            )
            _check_contains(
                checks,
                errors,
                "question_craftsmanship.required_checks",
                tuple(question_craftsmanship.get("checks", {}).keys()),
                REQUIRED_QUESTION_CRAFTSMANSHIP_CHECKS,
            )
            for check_id, item in sorted(question_craftsmanship.get("checks", {}).items()):
                _check_equal(
                    checks,
                    errors,
                    f"question_craftsmanship.{check_id}",
                    bool(item.get("passed")),
                    True,
                )
        if not require_question_craftsmanship and question_craftsmanship is None:
            warnings.append("question craftsmanship audit artifact was not found; per-question blueprint sanity was not enforced")

    query_construction = None
    if query_construction_report_path is not None or require_query_construction:
        if query_construction_report_path is not None:
            query_construction = read_json(query_construction_report_path)
        else:
            from amb.benchmark.quality.query_construction import audit_query_construction_release

            query_construction = audit_query_construction_release(manifest_file)
        _check_equal(checks, errors, "query_construction.present", query_construction is not None, True)
        if query_construction is not None:
            _check_equal(checks, errors, "query_construction.status", query_construction.get("status"), "passed")
            _check_artifact_binding(
                checks,
                errors,
                "query_construction",
                query_construction,
                report_path=query_construction_report_path,
                expected_benchmark_id=str(manifest.get("benchmark_id")),
                expected_release_split=None,
                expected_source_type="release_manifest",
                expected_manifest_path=manifest_file,
            )
            _check_contains(
                checks,
                errors,
                "query_construction.required_checks",
                tuple(query_construction.get("checks", {}).keys()),
                REQUIRED_QUERY_CONSTRUCTION_CHECKS,
            )
            for check_id, item in sorted(query_construction.get("checks", {}).items()):
                _check_equal(
                    checks,
                    errors,
                    f"query_construction.{check_id}",
                    bool(item.get("passed")),
                    True,
                )
        if not require_query_construction and query_construction is None:
            warnings.append("query construction audit artifact was not found; query-level comparability and evidence minimality were not enforced")

    probe_discriminativeness = None
    if probe_discriminativeness_report_path is not None or require_probe_discriminativeness:
        if probe_discriminativeness_report_path is not None:
            probe_discriminativeness = read_json(probe_discriminativeness_report_path)
        else:
            probe_discriminativeness = audit_probe_discriminativeness_release(
                manifest_file,
                split=representative_split,
                reports_dir=Path(representative_reports_dir or "reports/examples"),
            )
        _check_equal(checks, errors, "probe_discriminativeness.present", probe_discriminativeness is not None, True)
        if probe_discriminativeness is not None:
            _check_equal(checks, errors, "probe_discriminativeness.status", probe_discriminativeness.get("status"), "passed")
            _check_artifact_binding(
                checks,
                errors,
                "probe_discriminativeness",
                probe_discriminativeness,
                expected_benchmark_id=f"{manifest.get('benchmark_id')}-{representative_split}",
                expected_release_split=representative_split,
            )
            _check_contains(
                checks,
                errors,
                "probe_discriminativeness.required_checks",
                tuple(probe_discriminativeness.get("checks", {}).keys()),
                REQUIRED_PROBE_DISCRIMINATIVENESS_CHECKS,
            )
            for check_id, item in sorted(probe_discriminativeness.get("checks", {}).items()):
                _check_equal(
                    checks,
                    errors,
                    f"probe_discriminativeness.{check_id}",
                    bool(item.get("passed")),
                    True,
                )
        if not require_probe_discriminativeness and probe_discriminativeness is None:
            warnings.append("probe discriminativeness audit artifact was not found; representative probe separation was not enforced")

    difficulty_calibration = None
    if difficulty_calibration_report_path is not None or require_difficulty_calibration:
        if difficulty_calibration_report_path is not None:
            difficulty_calibration = read_json(difficulty_calibration_report_path)
        else:
            difficulty_calibration = audit_difficulty_calibration_release(
                manifest_file,
                split=representative_split,
                reports_dir=Path(representative_reports_dir or "reports/examples"),
            )
        _check_equal(checks, errors, "difficulty_calibration.present", difficulty_calibration is not None, True)
        if difficulty_calibration is not None:
            _check_equal(checks, errors, "difficulty_calibration.status", difficulty_calibration.get("status"), "passed")
            _check_artifact_binding(
                checks,
                errors,
                "difficulty_calibration",
                difficulty_calibration,
                expected_benchmark_id=f"{manifest.get('benchmark_id')}-{representative_split}",
                expected_release_split=representative_split,
            )
            _check_contains(
                checks,
                errors,
                "difficulty_calibration.required_checks",
                tuple(difficulty_calibration.get("checks", {}).keys()),
                REQUIRED_DIFFICULTY_CALIBRATION_CHECKS,
            )
            for check_id, item in sorted(difficulty_calibration.get("checks", {}).items()):
                _check_equal(
                    checks,
                    errors,
                    f"difficulty_calibration.{check_id}",
                    bool(item.get("passed")),
                    True,
                )
        if not require_difficulty_calibration and difficulty_calibration is None:
            warnings.append("difficulty calibration audit artifact was not found; difficulty bucket calibration was not enforced")

    domain_construct_validity = None
    if domain_construct_validity_report_path is not None or require_domain_construct_validity:
        if domain_construct_validity_report_path is not None:
            domain_construct_validity = read_json(domain_construct_validity_report_path)
        else:
            domain_construct_validity = audit_domain_construct_validity_release(
                manifest_file,
                split=representative_split,
                reports_dir=Path(representative_reports_dir or "reports/examples"),
            )
        _check_equal(checks, errors, "domain_construct_validity.present", domain_construct_validity is not None, True)
        if domain_construct_validity is not None:
            _check_equal(checks, errors, "domain_construct_validity.status", domain_construct_validity.get("status"), "passed")
            _check_artifact_binding(
                checks,
                errors,
                "domain_construct_validity",
                domain_construct_validity,
                expected_benchmark_id=f"{manifest.get('benchmark_id')}-{representative_split}",
                expected_release_split=representative_split,
            )
            _check_contains(
                checks,
                errors,
                "domain_construct_validity.required_checks",
                tuple(domain_construct_validity.get("checks", {}).keys()),
                REQUIRED_DOMAIN_CONSTRUCT_VALIDITY_CHECKS,
            )
            for check_id, item in sorted(domain_construct_validity.get("checks", {}).items()):
                _check_equal(
                    checks,
                    errors,
                    f"domain_construct_validity.{check_id}",
                    bool(item.get("passed")),
                    True,
                )
        if not require_domain_construct_validity and domain_construct_validity is None:
            warnings.append("domain construct validity audit artifact was not found; per-domain memory dependence sanity was not enforced")

    foundation_validation = None
    if foundation_validation_report_path is not None or require_foundation_validation:
        if foundation_validation_report_path is not None:
            foundation_validation = read_json(foundation_validation_report_path)
        _check_equal(checks, errors, "foundation_validation.present", foundation_validation is not None, True)
        if foundation_validation is not None:
            _check_equal(checks, errors, "foundation_validation.status", foundation_validation.get("status"), "passed")
            required_foundation_checks = list(REQUIRED_FOUNDATION_VALIDATION_CHECKS)
            protocol_reports = foundation_validation.get("protocol_reports", {})
            if isinstance(protocol_reports, dict) and "full_history" in protocol_reports:
                required_foundation_checks.extend(OPTIONAL_FULL_HISTORY_FOUNDATION_CHECKS)
            _check_contains(
                checks,
                errors,
                "foundation_validation.required_checks",
                tuple(foundation_validation.get("checks", {}).keys()),
                tuple(required_foundation_checks),
            )
            expected_foundation_scope = f"{manifest.get('benchmark_id')}-{representative_split}"
            actual_foundation_scope = str(foundation_validation.get("expected_benchmark_id") or "")
            _check_equal(
                checks,
                errors,
                "foundation_validation.expected_benchmark_scope_match",
                actual_foundation_scope.startswith(expected_foundation_scope),
                True,
            )
            for check_id, item in sorted(foundation_validation.get("checks", {}).items()):
                _check_equal(
                    checks,
                    errors,
                    f"foundation_validation.{check_id}",
                    bool(item.get("passed")),
                    True,
                )
        if not require_foundation_validation and foundation_validation is None:
            warnings.append("foundation validation artifact was not found; protocol-sensitive real-model evidence was not enforced")

    release_validation = None
    if release_validation_report_path is not None or run_release_validation:
        if release_validation_report_path is not None:
            release_validation = read_json(release_validation_report_path)
        else:
            release_validation = validate_release_artifacts(manifest_file)
        _check_equal(checks, errors, "release_validation_ok", release_validation.get("ok"), True)

    public_release_validation = None
    if public_release_validation_report_path is not None or public_release_manifest_path is not None or require_public_release_validation:
        if public_release_validation_report_path is not None:
            public_release_validation = read_json(public_release_validation_report_path)
        elif public_release_manifest_path is not None:
            public_release_validation = validate_release_artifacts(public_release_manifest_path)
        _check_equal(checks, errors, "public_release_validation.present", public_release_validation is not None, True)
        if public_release_validation is not None:
            _check_equal(checks, errors, "public_release_validation.ok", public_release_validation.get("ok"), True)

    passed = sum(1 for check in checks if check["status"] == "passed")
    failed = sum(1 for check in checks if check["status"] == "failed")
    return {
        "schema_version": MAIN_DATASET_ACCEPTANCE_SCHEMA_VERSION,
        "manifest_path": str(manifest_file),
        "benchmark_id": manifest.get("benchmark_id"),
        "status": "passed" if failed == 0 else "failed",
        "summary": {
            "passed": passed,
            "failed": failed,
            "warnings": len(warnings),
        },
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
        "canonical_profile_alignment": canonical_profile_alignment,
        "counterfactual_axis_coverage": axis_coverage,
        "case_scale_summary": case_scale_summary,
        "core_object_summary": core_object_summary,
        "audit_template_summary": audit_template_summary,
        "human_audit_verification": human_audit_verification,
        "strong_ai_audit": strong_ai_audit,
        "representative_baseline_summary": representative_baseline_summary,
        "intrinsic_sanity": intrinsic_sanity,
        "public_test_sanity": public_test_sanity,
        "public_result_slices": public_result_slices,
        "hidden_test_sanity": hidden_test_sanity,
        "human_audit_subset": human_audit_subset,
        "question_craftsmanship": question_craftsmanship,
        "query_construction": query_construction,
        "probe_discriminativeness": probe_discriminativeness,
        "difficulty_calibration": difficulty_calibration,
        "domain_construct_validity": domain_construct_validity,
        "foundation_validation": foundation_validation,
        "release_validation": release_validation,
        "public_release_validation": public_release_validation,
    }


def write_main_dataset_acceptance(
    manifest_path: str | Path,
    output: str | Path,
    *,
    expected_benchmark_id: str = "amst-main-v1",
    expected_profile_id: str = "main-v1",
    run_release_validation: bool = False,
    release_validation_report_path: str | Path | None = None,
    public_release_manifest_path: str | Path | None = None,
    public_release_validation_report_path: str | Path | None = None,
    require_public_release_validation: bool = False,
    require_all_counterfactual_axes: bool = False,
    representative_reports_dir: str | Path | None = None,
    representative_split: str = "public_dev",
    require_representative_baselines: bool = False,
    intrinsic_sanity_report_path: str | Path | None = None,
    require_intrinsic_sanity: bool = False,
    human_audit_verification_report_path: str | Path | None = None,
    require_completed_human_audit: bool = False,
    strong_ai_audit_report_path: str | Path | None = None,
    require_strong_ai_audit: bool = False,
    public_test_sanity_report_path: str | Path | None = None,
    require_public_test_sanity: bool = False,
    public_result_slices_report_path: str | Path | None = None,
    require_public_result_slices: bool = False,
    hidden_test_sanity_report_path: str | Path | None = None,
    require_hidden_test_sanity: bool = False,
    human_audit_subset_report_path: str | Path | None = None,
    require_human_audit_subset: bool = False,
    question_craftsmanship_report_path: str | Path | None = None,
    require_question_craftsmanship: bool = False,
    query_construction_report_path: str | Path | None = None,
    require_query_construction: bool = False,
    probe_discriminativeness_report_path: str | Path | None = None,
    require_probe_discriminativeness: bool = False,
    difficulty_calibration_report_path: str | Path | None = None,
    require_difficulty_calibration: bool = False,
    domain_construct_validity_report_path: str | Path | None = None,
    require_domain_construct_validity: bool = False,
    foundation_validation_report_path: str | Path | None = None,
    require_foundation_validation: bool = False,
    require_query_difficulty: bool = False,
) -> dict[str, Any]:
    report = validate_main_dataset_acceptance(
        manifest_path,
        expected_benchmark_id=expected_benchmark_id,
        expected_profile_id=expected_profile_id,
        run_release_validation=run_release_validation,
        release_validation_report_path=release_validation_report_path,
        public_release_manifest_path=public_release_manifest_path,
        public_release_validation_report_path=public_release_validation_report_path,
        require_public_release_validation=require_public_release_validation,
        require_all_counterfactual_axes=require_all_counterfactual_axes,
        representative_reports_dir=representative_reports_dir,
        representative_split=representative_split,
        require_representative_baselines=require_representative_baselines,
        intrinsic_sanity_report_path=intrinsic_sanity_report_path,
        require_intrinsic_sanity=require_intrinsic_sanity,
        human_audit_verification_report_path=human_audit_verification_report_path,
        require_completed_human_audit=require_completed_human_audit,
        strong_ai_audit_report_path=strong_ai_audit_report_path,
        require_strong_ai_audit=require_strong_ai_audit,
        public_test_sanity_report_path=public_test_sanity_report_path,
        require_public_test_sanity=require_public_test_sanity,
        public_result_slices_report_path=public_result_slices_report_path,
        require_public_result_slices=require_public_result_slices,
        hidden_test_sanity_report_path=hidden_test_sanity_report_path,
        require_hidden_test_sanity=require_hidden_test_sanity,
        human_audit_subset_report_path=human_audit_subset_report_path,
        require_human_audit_subset=require_human_audit_subset,
        question_craftsmanship_report_path=question_craftsmanship_report_path,
        require_question_craftsmanship=require_question_craftsmanship,
        query_construction_report_path=query_construction_report_path,
        require_query_construction=require_query_construction,
        probe_discriminativeness_report_path=probe_discriminativeness_report_path,
        require_probe_discriminativeness=require_probe_discriminativeness,
        difficulty_calibration_report_path=difficulty_calibration_report_path,
        require_difficulty_calibration=require_difficulty_calibration,
        domain_construct_validity_report_path=domain_construct_validity_report_path,
        require_domain_construct_validity=require_domain_construct_validity,
        foundation_validation_report_path=foundation_validation_report_path,
        require_foundation_validation=require_foundation_validation,
        require_query_difficulty=require_query_difficulty,
    )
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = output_path.resolve()
    project_root = _infer_acceptance_project_root(Path(manifest_path), output_path)
    if project_root is not None:
        report["root"] = _artifact_root_ref(output_path.parent, project_root)
        _normalize_acceptance_report_paths(report, project_root)
    write_json(output, report)
    return report


def validate_challenge_release_acceptance(
    manifest_path: str | Path,
    *,
    public_release_manifest_path: str | Path,
    release_validation_report_path: str | Path,
    public_release_validation_report_path: str | Path,
    representative_reports_dir: str | Path,
    question_craftsmanship_report_path: str | Path,
    query_construction_report_path: str | Path,
    probe_discriminativeness_report_path: str | Path,
    difficulty_calibration_report_path: str | Path,
    domain_construct_validity_report_path: str | Path,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    public_manifest_file = Path(public_release_manifest_path)
    manifest = read_json(manifest_file)
    public_manifest = read_json(public_manifest_file)
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []

    _check_equal(checks, errors, "benchmark_id", manifest.get("benchmark_id"), "amst-challenge-v1")
    _check_equal(checks, errors, "profile_id", manifest.get("profile_id"), "challenge-v1")
    expected = manifest.get("expected_generation_summary", {})
    _check_equal(checks, errors, "base_scenarios", expected.get("base_scenarios"), 240)
    _check_equal(checks, errors, "num_cases", expected.get("num_cases"), 1440)
    _check_equal(
        checks,
        errors,
        "counterfactual_variants_per_base",
        expected.get("counterfactual_variants_per_base"),
        5,
    )
    axis_coverage = expected.get("counterfactual_axis_coverage", {})
    _check_equal(
        checks,
        errors,
        "counterfactual_axes_all_recommended",
        bool(axis_coverage.get("covers_all_recommended_axes")),
        True,
    )
    _check_contains(
        checks,
        errors,
        "counterfactual_axes_covered",
        tuple(axis_coverage.get("covered_axes", [])),
        RECOMMENDED_COUNTERFACTUAL_AXES,
    )
    _check_equal(
        checks,
        errors,
        "private_hidden_split_present",
        bool(manifest.get("split_files", {}).get("hidden_test")),
        True,
    )
    _check_equal(
        checks,
        errors,
        "public_hidden_split_scrubbed",
        public_manifest.get("split_files", {}).get("hidden_test"),
        {},
    )

    split_reports = manifest.get("split_reports", {})
    for split in RELEASE_SPLITS:
        report = split_reports.get(split, {})
        if isinstance(report, dict) and report:
            _check_split_report(checks, errors, split, report, profile_id="challenge-v1")
    _check_hardness_slices(checks, errors, split_reports, "challenge-v1")
    _check_query_difficulty_coverage(checks, errors, split_reports)

    representative_baseline_summary = _check_representative_baselines(
        checks,
        errors,
        public_manifest,
        reports_dir=Path(representative_reports_dir),
        split="public_dev",
        require_reports=True,
    )

    release_validation = read_json(release_validation_report_path)
    _check_equal(checks, errors, "release_validation.ok", release_validation.get("ok"), True)
    _check_artifact_binding(
        checks,
        errors,
        "release_validation",
        release_validation,
        report_path=release_validation_report_path,
        expected_benchmark_id=str(manifest.get("benchmark_id")),
        expected_manifest_path=manifest_file,
    )

    public_release_validation = read_json(public_release_validation_report_path)
    _check_equal(checks, errors, "public_release_validation.ok", public_release_validation.get("ok"), True)
    _check_artifact_binding(
        checks,
        errors,
        "public_release_validation",
        public_release_validation,
        report_path=public_release_validation_report_path,
        expected_benchmark_id=str(public_manifest.get("benchmark_id")),
        expected_manifest_path=public_manifest_file,
    )

    question_craftsmanship = read_json(question_craftsmanship_report_path)
    _check_equal(checks, errors, "question_craftsmanship.status", question_craftsmanship.get("status"), "passed")
    _check_artifact_binding(
        checks,
        errors,
        "question_craftsmanship",
        question_craftsmanship,
        report_path=question_craftsmanship_report_path,
        expected_benchmark_id=str(manifest.get("benchmark_id")),
        expected_source_type="release_manifest",
        expected_manifest_paths=(manifest_file, public_manifest_file),
    )
    _check_contains(
        checks,
        errors,
        "question_craftsmanship.required_checks",
        tuple(question_craftsmanship.get("checks", {}).keys()),
        REQUIRED_QUESTION_CRAFTSMANSHIP_CHECKS,
    )

    query_construction = read_json(query_construction_report_path)
    _check_equal(checks, errors, "query_construction.status", query_construction.get("status"), "passed")
    _check_artifact_binding(
        checks,
        errors,
        "query_construction",
        query_construction,
        report_path=query_construction_report_path,
        expected_benchmark_id=str(manifest.get("benchmark_id")),
        expected_source_type="release_manifest",
        expected_manifest_path=manifest_file,
    )
    _check_contains(
        checks,
        errors,
        "query_construction.required_checks",
        tuple(query_construction.get("checks", {}).keys()),
        REQUIRED_QUERY_CONSTRUCTION_CHECKS,
    )

    probe_discriminativeness = read_json(probe_discriminativeness_report_path)
    _check_equal(checks, errors, "probe_discriminativeness.status", probe_discriminativeness.get("status"), "passed")
    _check_artifact_binding(
        checks,
        errors,
        "probe_discriminativeness",
        probe_discriminativeness,
        report_path=probe_discriminativeness_report_path,
        expected_benchmark_id=f"{manifest.get('benchmark_id')}-public_dev",
        expected_release_split="public_dev",
    )
    _check_contains(
        checks,
        errors,
        "probe_discriminativeness.required_checks",
        tuple(probe_discriminativeness.get("checks", {}).keys()),
        REQUIRED_PROBE_DISCRIMINATIVENESS_CHECKS,
    )

    difficulty_calibration = read_json(difficulty_calibration_report_path)
    _check_equal(checks, errors, "difficulty_calibration.status", difficulty_calibration.get("status"), "passed")
    _check_artifact_binding(
        checks,
        errors,
        "difficulty_calibration",
        difficulty_calibration,
        report_path=difficulty_calibration_report_path,
        expected_benchmark_id=f"{manifest.get('benchmark_id')}-public_dev",
        expected_release_split="public_dev",
    )
    _check_contains(
        checks,
        errors,
        "difficulty_calibration.required_checks",
        tuple(difficulty_calibration.get("checks", {}).keys()),
        REQUIRED_DIFFICULTY_CALIBRATION_CHECKS,
    )

    domain_construct_validity = read_json(domain_construct_validity_report_path)
    _check_equal(checks, errors, "domain_construct_validity.status", domain_construct_validity.get("status"), "passed")
    _check_artifact_binding(
        checks,
        errors,
        "domain_construct_validity",
        domain_construct_validity,
        report_path=domain_construct_validity_report_path,
        expected_benchmark_id=f"{manifest.get('benchmark_id')}-public_dev",
        expected_release_split="public_dev",
    )
    _check_contains(
        checks,
        errors,
        "domain_construct_validity.required_checks",
        tuple(domain_construct_validity.get("checks", {}).keys()),
        REQUIRED_DOMAIN_CONSTRUCT_VALIDITY_CHECKS,
    )

    failed_checks = [check for check in checks if check.get("status") != "passed"]
    return {
        "schema_version": RELEASE_FAMILY_ACCEPTANCE_SCHEMA_VERSION,
        "release_family": "challenge_release",
        "manifest_path": str(manifest_file),
        "public_manifest_path": str(public_manifest_file),
        "status": "passed" if not errors else "failed",
        "summary": {"passed": len(checks) - len(failed_checks), "failed": len(failed_checks), "warnings": len(warnings)},
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "representative_baseline_summary": representative_baseline_summary,
    }


def write_challenge_release_acceptance(
    manifest_path: str | Path,
    output: str | Path,
    *,
    public_release_manifest_path: str | Path,
    release_validation_report_path: str | Path,
    public_release_validation_report_path: str | Path,
    representative_reports_dir: str | Path,
    question_craftsmanship_report_path: str | Path,
    query_construction_report_path: str | Path,
    probe_discriminativeness_report_path: str | Path,
    difficulty_calibration_report_path: str | Path,
    domain_construct_validity_report_path: str | Path,
) -> dict[str, Any]:
    report = validate_challenge_release_acceptance(
        manifest_path,
        public_release_manifest_path=public_release_manifest_path,
        release_validation_report_path=release_validation_report_path,
        public_release_validation_report_path=public_release_validation_report_path,
        representative_reports_dir=representative_reports_dir,
        question_craftsmanship_report_path=question_craftsmanship_report_path,
        query_construction_report_path=query_construction_report_path,
        probe_discriminativeness_report_path=probe_discriminativeness_report_path,
        difficulty_calibration_report_path=difficulty_calibration_report_path,
        domain_construct_validity_report_path=domain_construct_validity_report_path,
    )
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = output_path.resolve()
    project_root = _infer_acceptance_project_root(Path(manifest_path), output_path)
    if project_root is not None:
        report["root"] = _artifact_root_ref(output_path.parent, project_root)
        report = localize_report_contract(
            report,
            output_path=output_path,
            project_root_hints=(
                Path(manifest_path),
                Path(public_release_manifest_path),
                Path(release_validation_report_path),
                Path(public_release_validation_report_path),
                Path(question_craftsmanship_report_path),
                Path(query_construction_report_path),
                Path(probe_discriminativeness_report_path),
                Path(difficulty_calibration_report_path),
                Path(domain_construct_validity_report_path),
            ),
        )
    write_json(output, report)
    return report


def validate_hidden_quarterly_release_acceptance(
    manifest_path: str | Path,
    *,
    release_validation_report_path: str | Path,
    intrinsic_sanity_report_path: str | Path,
    question_craftsmanship_report_path: str | Path,
    query_construction_report_path: str | Path,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []

    _check_equal(
        checks,
        errors,
        "benchmark_id",
        manifest.get("benchmark_id"),
        "amst-main-v1-strict-2026Q2-hidden_refresh",
    )
    _check_equal(checks, errors, "profile_id", manifest.get("profile_id"), "main-v1-strict")
    _check_equal(
        checks,
        errors,
        "package_type",
        manifest.get("package_type"),
        "private_leaderboard_package",
    )
    refresh = manifest.get("quarterly_hidden_refresh", {})
    _check_equal(checks, errors, "hidden_refresh.num_hidden_scenarios", refresh.get("num_hidden_scenarios"), 300)
    _check_equal(
        checks,
        errors,
        "hidden_refresh.same_compiler_family",
        bool(refresh.get("same_compiler_family")),
        True,
    )
    _check_equal(
        checks,
        errors,
        "hidden_refresh.source_profile_id",
        refresh.get("source_profile_id"),
        "main-v1-strict",
    )

    split_reports = manifest.get("split_reports", {})
    hidden_test = split_reports.get("hidden_test", {})
    _check_equal(checks, errors, "hidden_test.num_cases_positive", _as_int(hidden_test.get("num_cases")) > 0, True)
    if isinstance(hidden_test, dict) and hidden_test:
        _check_split_report(checks, errors, "hidden_test", hidden_test, profile_id="hidden-quarterly-v1")

    release_validation = read_json(release_validation_report_path)
    _check_equal(checks, errors, "release_validation.ok", release_validation.get("ok"), True)
    _check_artifact_binding(
        checks,
        errors,
        "release_validation",
        release_validation,
        report_path=release_validation_report_path,
        expected_benchmark_id=str(manifest.get("benchmark_id")),
        expected_manifest_path=manifest_file,
    )

    intrinsic_sanity = read_json(intrinsic_sanity_report_path)
    _check_equal(checks, errors, "intrinsic_sanity.status", intrinsic_sanity.get("status"), "passed")
    _check_artifact_binding(
        checks,
        errors,
        "intrinsic_sanity",
        intrinsic_sanity,
        report_path=intrinsic_sanity_report_path,
        expected_benchmark_id=str(manifest.get("benchmark_id")),
        expected_manifest_path=manifest_file,
    )

    question_craftsmanship = read_json(question_craftsmanship_report_path)
    _check_equal(checks, errors, "question_craftsmanship.status", question_craftsmanship.get("status"), "passed")
    _check_artifact_binding(
        checks,
        errors,
        "question_craftsmanship",
        question_craftsmanship,
        report_path=question_craftsmanship_report_path,
        expected_benchmark_id=str(manifest.get("benchmark_id")),
        expected_source_type="release_manifest",
        expected_manifest_path=manifest_file,
    )
    _check_contains(
        checks,
        errors,
        "question_craftsmanship.required_checks",
        tuple(question_craftsmanship.get("checks", {}).keys()),
        REQUIRED_QUESTION_CRAFTSMANSHIP_CHECKS,
    )

    query_construction = read_json(query_construction_report_path)
    _check_equal(checks, errors, "query_construction.status", query_construction.get("status"), "passed")
    _check_artifact_binding(
        checks,
        errors,
        "query_construction",
        query_construction,
        report_path=query_construction_report_path,
        expected_benchmark_id=str(manifest.get("benchmark_id")),
        expected_source_type="release_manifest",
        expected_manifest_path=manifest_file,
    )
    _check_contains(
        checks,
        errors,
        "query_construction.required_checks",
        tuple(query_construction.get("checks", {}).keys()),
        REQUIRED_QUERY_CONSTRUCTION_CHECKS,
    )

    failed_checks = [check for check in checks if check.get("status") != "passed"]
    return {
        "schema_version": RELEASE_FAMILY_ACCEPTANCE_SCHEMA_VERSION,
        "release_family": "quarterly_hidden_refresh",
        "manifest_path": str(manifest_file),
        "status": "passed" if not errors else "failed",
        "summary": {"passed": len(checks) - len(failed_checks), "failed": len(failed_checks), "warnings": len(warnings)},
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def write_hidden_quarterly_release_acceptance(
    manifest_path: str | Path,
    output: str | Path,
    *,
    release_validation_report_path: str | Path,
    intrinsic_sanity_report_path: str | Path,
    question_craftsmanship_report_path: str | Path,
    query_construction_report_path: str | Path,
) -> dict[str, Any]:
    report = validate_hidden_quarterly_release_acceptance(
        manifest_path,
        release_validation_report_path=release_validation_report_path,
        intrinsic_sanity_report_path=intrinsic_sanity_report_path,
        question_craftsmanship_report_path=question_craftsmanship_report_path,
        query_construction_report_path=query_construction_report_path,
    )
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = output_path.resolve()
    project_root = _infer_acceptance_project_root(Path(manifest_path), output_path)
    if project_root is not None:
        report["root"] = _artifact_root_ref(output_path.parent, project_root)
        report = localize_report_contract(
            report,
            output_path=output_path,
            project_root_hints=(
                Path(manifest_path),
                Path(release_validation_report_path),
                Path(intrinsic_sanity_report_path),
                Path(question_craftsmanship_report_path),
                Path(query_construction_report_path),
            ),
        )
    write_json(output, report)
    return report


def _current_example_artifact_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_current{path.suffix}")


def _prefer_current_example_artifact(path: Path) -> Path:
    current_path = _current_example_artifact_path(path)
    return current_path if current_path.exists() else path


def _resolve_foundation_protocol_report_paths(
    foundation_validation_path: Path,
    project_root: Path,
) -> tuple[Path, ...]:
    if not foundation_validation_path.exists():
        return ()
    foundation_validation = read_json(foundation_validation_path)
    protocol_reports = foundation_validation.get("protocol_reports", {})
    if not isinstance(protocol_reports, dict):
        return ()
    paths: list[Path] = []
    for protocol in ("query_only", "full_history", "oracle_state"):
        protocol_report = protocol_reports.get(protocol)
        if not isinstance(protocol_report, dict):
            continue
        raw_path = protocol_report.get("path")
        if not raw_path:
            continue
        resolved = Path(str(raw_path))
        paths.append(resolved if resolved.is_absolute() else project_root / resolved)
    return tuple(paths)


def _artifact_path_ref(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()).as_posix())
    except ValueError:
        return str(path)


def _report_prefix(benchmark_id: str, split: str | None = None) -> str:
    prefix = benchmark_id.replace("-", "_")
    return f"{prefix}_{split}" if split else prefix


def _canonical_baseline_report_paths(
    reports_root: Path,
    *,
    benchmark_id: str,
    split: str,
) -> dict[str, Path]:
    prefix = _report_prefix(benchmark_id, split)
    return {
        kind: reports_root / f"{prefix}_{kind}_report.json"
        for kind in CURRENT_RELEASE_BASELINES
    }


def _summary_metrics_from_report(report: dict[str, Any]) -> dict[str, float | None]:
    aggregate = report.get("aggregate", {})
    counterfactual = report.get("counterfactual", {})
    by_requirement = report.get("by_memory_requirement", {})
    requires_memory = by_requirement.get("requires_memory", {}) if isinstance(by_requirement, dict) else {}
    no_memory_required = by_requirement.get("no_memory_required", {}) if isinstance(by_requirement, dict) else {}
    return {
        "amq": _float_or_none(aggregate.get("lifecycle.amq")),
        "task_success": _float_or_none(aggregate.get("task.task_success")),
        "recall_at_k": _float_or_none(aggregate.get("retrieval.recall_at_k")),
        "requires_memory_num_queries": _float_or_none(requires_memory.get("num_scored_queries")),
        "requires_memory_amq": _float_or_none(requires_memory.get("lifecycle.amq")),
        "requires_memory_task_success": _float_or_none(requires_memory.get("task.task_success")),
        "requires_memory_recall_at_k": _float_or_none(requires_memory.get("retrieval.recall_at_k")),
        "no_memory_required_num_queries": _float_or_none(no_memory_required.get("num_scored_queries")),
        "no_memory_required_amq": _float_or_none(no_memory_required.get("lifecycle.amq")),
        "no_memory_required_task_success": _float_or_none(no_memory_required.get("task.task_success")),
        "no_memory_required_recall_at_k": _float_or_none(no_memory_required.get("retrieval.recall_at_k")),
        "safety_pass": _float_or_none(aggregate.get("safety.safety_pass")),
        "input_tokens": _float_or_none(aggregate.get("efficiency.input_tokens")),
        "memory_dependence_proxy": _float_or_none(counterfactual.get("memory_dependence_proxy")),
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_current_failure_mode_diagnostics(
    *,
    manifest_path: Path,
    output_path: Path,
    split: str,
    report_paths: dict[str, Path],
) -> dict[str, Any]:
    from amb.benchmark.release.failure_modes import summarize_failure_mode_reports

    manifest = read_json(manifest_path)
    diagnostics = summarize_failure_mode_reports(
        report_paths,
        benchmark_id=f"{manifest.get('benchmark_id', 'release')}-{split}",
        release_split=split,
    )
    diagnostics = localize_report_contract(
        diagnostics,
        output_path=output_path,
        project_root_hints=(manifest_path, *report_paths.values()),
    )
    write_json(output_path, diagnostics)
    diagnostics["diagnostics_path"] = str(output_path)
    return diagnostics


def _artifact_is_fresh(output_path: Path, source_paths: tuple[Path, ...]) -> bool:
    if not output_path.exists():
        return False
    try:
        output_mtime = output_path.stat().st_mtime
    except FileNotFoundError:
        return False
    for source_path in source_paths:
        if not source_path.exists():
            return False
        if source_path.stat().st_mtime > output_mtime:
            return False
    return True


def _read_or_build_json_artifact(
    output_path: Path,
    *,
    source_paths: tuple[Path, ...],
    build: callable,
) -> dict[str, Any]:
    if _artifact_is_fresh(output_path, source_paths):
        return read_json(output_path)
    canonical_current = Path.cwd() / "reports" / "examples" / output_path.name
    if canonical_current.resolve() != output_path.resolve() and _artifact_is_fresh(canonical_current, source_paths):
        return _write_current_json_copy(
            source_path=canonical_current,
            output_path=output_path,
            project_root_hints=source_paths,
        )
    return build()


def _write_current_json_copy(
    *,
    source_path: Path,
    output_path: Path,
    project_root_hints: tuple[Path, ...],
) -> dict[str, Any]:
    artifact = localize_report_contract(
        read_json(source_path),
        output_path=output_path,
        project_root_hints=project_root_hints,
    )
    write_json(output_path, artifact)
    return artifact


def _write_current_representative_analysis(
    *,
    report_paths: tuple[Path, ...],
    output_path: Path,
    bootstrap_samples: int = 200,
    seed: int = 13,
) -> dict[str, Any]:
    from amb.benchmark.analysis import analyze_report_files

    analysis = analyze_report_files(
        [str(path) for path in report_paths],
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )
    analysis = localize_report_contract(
        analysis,
        output_path=output_path,
        project_root_hints=report_paths,
    )
    write_json(output_path, analysis)
    return analysis


def _representative_analysis_source_paths(
    project_root: Path,
    report_paths: tuple[Path, ...],
) -> tuple[Path, ...]:
    return (
        *report_paths,
        project_root / "amb/benchmark/analysis/report_analysis.py",
        project_root / "amb/benchmark/evaluation/scoring.py",
    )


def _representative_report_source_paths(
    reports_root: Path,
    public_manifest_path: Path,
    split: str,
) -> tuple[Path, ...]:
    manifest = read_json(public_manifest_path)
    return tuple(
        _canonical_baseline_report_paths(
            reports_root,
            benchmark_id=str(manifest.get("benchmark_id", "release")),
            split=split,
        ).values()
    )


def _write_current_public_dev_effectiveness_artifacts(
    project_root: Path,
    *,
    public_manifest_path: Path,
    output_root: Path,
) -> dict[str, str]:
    from amb.benchmark.leaderboard import write_leaderboard_summary

    output_root.mkdir(parents=True, exist_ok=True)
    source_reports_root = project_root / "reports/examples"
    public_manifest = read_json(public_manifest_path)
    benchmark_id = str(public_manifest.get("benchmark_id", "release"))
    split = "public_dev"
    prefix = _report_prefix(benchmark_id, split)
    source_report_paths = _canonical_baseline_report_paths(
        source_reports_root,
        benchmark_id=benchmark_id,
        split=split,
    )
    current_report_paths: dict[str, Path] = {}
    artifacts: dict[str, str] = {}

    for kind, source_path in source_report_paths.items():
        current_path = output_root / f"{source_path.stem}_current{source_path.suffix}"
        _read_or_build_json_artifact(
            current_path,
            source_paths=(source_path,),
            build=lambda source_path=source_path, current_path=current_path: _write_current_json_copy(
                source_path=source_path,
                output_path=current_path,
                project_root_hints=(source_path, public_manifest_path),
            ),
        )
        current_report_paths[kind] = current_path
        artifacts[f"{split}_{kind}_report"] = _artifact_path_ref(current_path, project_root)

    analysis_path = output_root / f"{prefix}_representative_baselines_analysis_current.json"
    analysis_report_paths = tuple(current_report_paths[kind] for kind in REPRESENTATIVE_BASELINES)
    _read_or_build_json_artifact(
        analysis_path,
        source_paths=_representative_analysis_source_paths(project_root, analysis_report_paths),
        build=lambda: _write_current_representative_analysis(
            report_paths=analysis_report_paths,
            output_path=analysis_path,
        ),
    )
    artifacts[f"{split}_representative_analysis"] = _artifact_path_ref(analysis_path, project_root)

    leaderboard_path = output_root / f"{prefix}_leaderboard_current.json"
    _read_or_build_json_artifact(
        leaderboard_path,
        source_paths=tuple(current_report_paths.values()),
        build=lambda: write_leaderboard_summary(tuple(current_report_paths.values()), leaderboard_path),
    )
    artifacts[f"{split}_leaderboard"] = _artifact_path_ref(leaderboard_path, project_root)

    failure_mode_path = output_root / f"{prefix}_failure_mode_diagnostics_current.json"
    _read_or_build_json_artifact(
        failure_mode_path,
        source_paths=(public_manifest_path, *tuple(current_report_paths.values())),
        build=lambda: _write_current_failure_mode_diagnostics(
            manifest_path=public_manifest_path,
            output_path=failure_mode_path,
            split=split,
            report_paths=current_report_paths,
        ),
    )
    artifacts[f"{split}_failure_mode_diagnostics"] = _artifact_path_ref(failure_mode_path, project_root)

    return artifacts


def _write_current_public_dev_representative_artifacts(
    project_root: Path,
    *,
    public_manifest_path: Path,
    output_root: Path,
) -> dict[str, str]:
    from amb.benchmark.leaderboard import write_leaderboard_summary

    output_root.mkdir(parents=True, exist_ok=True)
    source_reports_root = project_root / "reports/examples"
    public_manifest = read_json(public_manifest_path)
    benchmark_id = str(public_manifest.get("benchmark_id", "release"))
    split = "public_dev"
    prefix = _report_prefix(benchmark_id, split)
    artifacts: dict[str, str] = {}
    current_report_paths: dict[str, Path] = {}

    for kind in REPRESENTATIVE_BASELINES:
        source_path = source_reports_root / f"{prefix}_{kind}_report.json"
        current_path = output_root / f"{source_path.stem}_current{source_path.suffix}"
        _read_or_build_json_artifact(
            current_path,
            source_paths=(source_path,),
            build=lambda source_path=source_path, current_path=current_path: _write_current_json_copy(
                source_path=source_path,
                output_path=current_path,
                project_root_hints=(source_path, public_manifest_path),
            ),
        )
        current_report_paths[kind] = current_path
        artifacts[f"{split}_{kind}_report"] = _artifact_path_ref(current_path, project_root)

    analysis_path = output_root / f"{prefix}_representative_baselines_analysis_current.json"
    analysis_report_paths = tuple(current_report_paths[kind] for kind in REPRESENTATIVE_BASELINES)
    _read_or_build_json_artifact(
        analysis_path,
        source_paths=_representative_analysis_source_paths(project_root, analysis_report_paths),
        build=lambda: _write_current_representative_analysis(
            report_paths=analysis_report_paths,
            output_path=analysis_path,
        ),
    )
    artifacts[f"{split}_representative_analysis"] = _artifact_path_ref(analysis_path, project_root)

    leaderboard_path = output_root / f"{prefix}_leaderboard_current.json"
    _read_or_build_json_artifact(
        leaderboard_path,
        source_paths=tuple(current_report_paths.values()),
        build=lambda: write_leaderboard_summary(tuple(current_report_paths.values()), leaderboard_path),
    )
    artifacts[f"{split}_leaderboard"] = _artifact_path_ref(leaderboard_path, project_root)
    return artifacts


def _write_canonical_main_release_support_currents(
    project_root: Path,
    *,
    release_manifest_path: Path,
    public_manifest_path: Path,
    output_prefix: str,
    output_root: Path,
) -> dict[str, Any]:
    from amb.benchmark.quality.lineage import write_lineage_audit
    from amb.benchmark.quality.release_intrinsic_sanity import write_release_intrinsic_sanity
    from amb.benchmark.quality.release_validation import write_release_validation
    from amb.benchmark.release.hidden_test_sanity import build_hidden_test_sanity_summary
    from amb.benchmark.release.public_result_slices import build_public_result_slice_artifacts
    from amb.benchmark.release.public_test_sanity import build_public_test_sanity_summary

    output_root.mkdir(parents=True, exist_ok=True)
    source_reports_root = project_root / "reports/examples"
    release_manifest = read_json(release_manifest_path)
    public_manifest = read_json(public_manifest_path)
    artifacts: dict[str, str] = {}

    release_validation_path = output_root / f"{output_prefix}_release_validation_current.json"
    _read_or_build_json_artifact(
        release_validation_path,
        source_paths=(release_manifest_path,),
        build=lambda: write_release_validation(release_manifest_path, release_validation_path),
    )
    artifacts["release_validation"] = _artifact_path_ref(release_validation_path, project_root)

    public_release_validation_path = output_root / f"{output_prefix}_public_release_validation_current.json"
    _read_or_build_json_artifact(
        public_release_validation_path,
        source_paths=(public_manifest_path,),
        build=lambda: write_release_validation(public_manifest_path, public_release_validation_path),
    )
    artifacts["public_release_validation"] = _artifact_path_ref(public_release_validation_path, project_root)

    intrinsic_sanity_path = output_root / f"{output_prefix}_intrinsic_sanity_current.json"
    _read_or_build_json_artifact(
        intrinsic_sanity_path,
        source_paths=(release_manifest_path,),
        build=lambda: write_release_intrinsic_sanity(release_manifest_path, intrinsic_sanity_path),
    )
    artifacts["intrinsic_sanity"] = _artifact_path_ref(intrinsic_sanity_path, project_root)

    lineage_path = output_root / f"{output_prefix}_lineage_audit_current.json"
    _read_or_build_json_artifact(
        lineage_path,
        source_paths=(release_manifest_path,),
        build=lambda: write_lineage_audit(lineage_path, manifest_path=release_manifest_path),
    )
    artifacts["lineage"] = _artifact_path_ref(lineage_path, project_root)

    public_test_report_paths = _canonical_baseline_report_paths(
        source_reports_root,
        benchmark_id=str(public_manifest.get("benchmark_id", "release")),
        split="public_test",
    )
    public_test_diagnostics_path = output_root / f"{output_prefix}_public_test_failure_mode_diagnostics_current.json"
    public_test_diagnostics = _read_or_build_json_artifact(
        public_test_diagnostics_path,
        source_paths=(public_manifest_path, *tuple(public_test_report_paths.values())),
        build=lambda: _write_current_failure_mode_diagnostics(
            manifest_path=public_manifest_path,
            output_path=public_test_diagnostics_path,
            split="public_test",
            report_paths=public_test_report_paths,
        ),
    )
    artifacts["public_test_failure_mode_diagnostics"] = _artifact_path_ref(public_test_diagnostics_path, project_root)

    public_test_baselines = {
        "benchmark_id": f"{public_manifest.get('benchmark_id', 'release')}-public_test",
        "release_split": "public_test",
        "baseline_kinds": list(CURRENT_RELEASE_BASELINES),
        "report_paths": {kind: str(path) for kind, path in public_test_report_paths.items()},
        "summary_metrics": {
            kind: _summary_metrics_from_report(read_json(path))
            for kind, path in public_test_report_paths.items()
        },
    }
    public_test_sanity_path = output_root / f"{output_prefix}_public_test_sanity_current.json"
    _read_or_build_json_artifact(
        public_test_sanity_path,
        source_paths=(public_test_diagnostics_path, *tuple(public_test_report_paths.values())),
        build=lambda: build_public_test_sanity_summary(
            public_test_baselines,
            public_test_diagnostics,
            output_path=public_test_sanity_path,
        ),
    )
    artifacts["public_test_sanity"] = _artifact_path_ref(public_test_sanity_path, project_root)

    public_test_required_slices_path = output_root / f"{output_prefix}_public_test_required_slices_current.json"
    _read_or_build_json_artifact(
        public_test_required_slices_path,
        source_paths=tuple(public_test_report_paths.values()),
        build=lambda: build_public_result_slice_artifacts(
            public_test_report_paths,
            benchmark_id=f"{public_manifest.get('benchmark_id', 'release')}-public_test",
            release_split="public_test",
            json_output_path=public_test_required_slices_path,
            markdown_output_path=output_root / f"{output_prefix}_public_test_required_slices_current.md",
        ),
    )
    artifacts["public_result_slices"] = _artifact_path_ref(public_test_required_slices_path, project_root)

    hidden_test_report_paths = _canonical_baseline_report_paths(
        source_reports_root,
        benchmark_id=str(release_manifest.get("benchmark_id", "release")),
        split="hidden_test",
    )
    hidden_test_diagnostics_path = output_root / f"{output_prefix}_hidden_test_failure_mode_diagnostics_current.json"
    hidden_test_diagnostics = _read_or_build_json_artifact(
        hidden_test_diagnostics_path,
        source_paths=(release_manifest_path, *tuple(hidden_test_report_paths.values())),
        build=lambda: _write_current_failure_mode_diagnostics(
            manifest_path=release_manifest_path,
            output_path=hidden_test_diagnostics_path,
            split="hidden_test",
            report_paths=hidden_test_report_paths,
        ),
    )
    artifacts["hidden_test_failure_mode_diagnostics"] = _artifact_path_ref(hidden_test_diagnostics_path, project_root)

    hidden_test_baselines = {
        "benchmark_id": f"{release_manifest.get('benchmark_id', 'release')}-hidden_test",
        "release_split": "hidden_test",
        "baseline_kinds": list(CURRENT_RELEASE_BASELINES),
        "report_paths": {kind: str(path) for kind, path in hidden_test_report_paths.items()},
        "summary_metrics": {
            kind: _summary_metrics_from_report(read_json(path))
            for kind, path in hidden_test_report_paths.items()
        },
    }
    hidden_test_sanity_path = output_root / f"{output_prefix}_hidden_test_sanity_current.json"
    _read_or_build_json_artifact(
        hidden_test_sanity_path,
        source_paths=(release_manifest_path, hidden_test_diagnostics_path, *tuple(hidden_test_report_paths.values())),
        build=lambda: build_hidden_test_sanity_summary(
            release_manifest,
            hidden_test_baselines,
            hidden_test_diagnostics,
            output_path=hidden_test_sanity_path,
        ),
    )
    artifacts["hidden_test_sanity"] = _artifact_path_ref(hidden_test_sanity_path, project_root)
    artifacts.update(
        _write_current_public_dev_effectiveness_artifacts(
            project_root,
            public_manifest_path=public_manifest_path,
            output_root=output_root,
        )
    )

    return {
        "profile_prefix": output_prefix,
        "artifacts": artifacts,
    }


def write_canonical_main_release_support_currents(
    root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Refresh watcher-owned support evidence for the compatibility main release."""

    project_root = Path(root)
    release_manifest_path = project_root / "data/releases/amst_main_v1/manifest.json"
    public_manifest_path = project_root / "data/releases/amst_main_v1_public/manifest.json"
    if not release_manifest_path.exists() or not public_manifest_path.exists():
        return None
    return _write_canonical_main_release_support_currents(
        project_root,
        release_manifest_path=release_manifest_path,
        public_manifest_path=public_manifest_path,
        output_prefix="amst_main_v1",
        output_root=Path(output_dir) if output_dir is not None else project_root / "reports/examples",
    )


def write_canonical_strict_main_release_support_currents(
    root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Refresh watcher-owned support evidence for the canonical strict main release."""

    project_root = Path(root)
    release_manifest_path = project_root / "data/releases/amst_main_v1_strict/manifest.json"
    public_manifest_path = project_root / "data/releases/amst_main_v1_strict_public/manifest.json"
    if not release_manifest_path.exists() or not public_manifest_path.exists():
        return None
    return _write_canonical_main_release_support_currents(
        project_root,
        release_manifest_path=release_manifest_path,
        public_manifest_path=public_manifest_path,
        output_prefix="amst_main_v1_strict",
        output_root=Path(output_dir) if output_dir is not None else project_root / "reports/examples",
    )


def write_canonical_challenge_release_currents(
    root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Refresh watcher-owned validation and quality evidence for the challenge release."""

    from amb.benchmark.quality.release_validation import write_release_validation

    project_root = Path(root)
    release_manifest_path = project_root / "data/releases/amst_challenge_v1/manifest.json"
    public_manifest_path = project_root / "data/releases/amst_challenge_v1_public/manifest.json"
    if not release_manifest_path.exists() or not public_manifest_path.exists():
        return None

    output_root = Path(output_dir) if output_dir is not None else project_root / "reports/examples"
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    release_validation_path = output_root / "amst_challenge_v1_release_validation_current.json"
    _read_or_build_json_artifact(
        release_validation_path,
        source_paths=(release_manifest_path,),
        build=lambda: write_release_validation(release_manifest_path, release_validation_path),
    )
    artifacts["release_validation"] = _artifact_path_ref(release_validation_path, project_root)

    public_release_validation_path = output_root / "amst_challenge_v1_public_release_validation_current.json"
    _read_or_build_json_artifact(
        public_release_validation_path,
        source_paths=(public_manifest_path,),
        build=lambda: write_release_validation(public_manifest_path, public_release_validation_path),
    )
    artifacts["public_release_validation"] = _artifact_path_ref(public_release_validation_path, project_root)

    quality = _write_canonical_main_quality_audits_current(
        project_root,
        release_manifest_path=release_manifest_path,
        public_manifest_path=public_manifest_path,
        output_prefix="amst_challenge_v1",
        output_root=output_root,
    )
    artifacts.update(dict(quality.get("artifacts", {})))
    artifacts.update(
        _write_current_public_dev_representative_artifacts(
            project_root,
            public_manifest_path=public_manifest_path,
            output_root=output_root,
        )
    )
    return {
        "profile_prefix": "amst_challenge_v1",
        "artifacts": artifacts,
    }


def write_canonical_hidden_quarterly_release_currents(
    root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Refresh watcher-owned validation and quality evidence for the quarterly hidden refresh."""

    from amb.benchmark.quality.question_craftsmanship import write_question_craftsmanship_audit
    from amb.benchmark.quality.query_construction import write_query_construction_audit
    from amb.benchmark.quality.release_intrinsic_sanity import write_release_intrinsic_sanity
    from amb.benchmark.quality.release_validation import write_release_validation

    project_root = Path(root)
    manifest_path = project_root / "data/releases/amst_hidden_quarterly_v1/manifest.json"
    if not manifest_path.exists():
        return None

    output_root = Path(output_dir) if output_dir is not None else project_root / "reports/examples"
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    release_validation_path = output_root / "amst_hidden_quarterly_v1_release_validation_current.json"
    _read_or_build_json_artifact(
        release_validation_path,
        source_paths=(manifest_path,),
        build=lambda: write_release_validation(manifest_path, release_validation_path),
    )
    artifacts["release_validation"] = _artifact_path_ref(release_validation_path, project_root)

    intrinsic_sanity_path = output_root / "amst_hidden_quarterly_v1_release_intrinsic_sanity_current.json"
    _read_or_build_json_artifact(
        intrinsic_sanity_path,
        source_paths=(manifest_path,),
        build=lambda: write_release_intrinsic_sanity(manifest_path, intrinsic_sanity_path),
    )
    artifacts["intrinsic_sanity"] = _artifact_path_ref(intrinsic_sanity_path, project_root)

    question_craftsmanship_path = output_root / "amst_hidden_quarterly_v1_question_craftsmanship_audit_current.json"
    _read_or_build_json_artifact(
        question_craftsmanship_path,
        source_paths=(manifest_path,),
        build=lambda: write_question_craftsmanship_audit(question_craftsmanship_path, manifest_path=manifest_path),
    )
    artifacts["question_craftsmanship"] = _artifact_path_ref(question_craftsmanship_path, project_root)

    query_construction_path = output_root / "amst_hidden_quarterly_v1_query_construction_audit_current.json"
    _read_or_build_json_artifact(
        query_construction_path,
        source_paths=(manifest_path,),
        build=lambda: write_query_construction_audit(query_construction_path, manifest_path=manifest_path),
    )
    artifacts["query_construction"] = _artifact_path_ref(query_construction_path, project_root)

    return {
        "profile_prefix": "amst_hidden_quarterly_v1",
        "artifacts": artifacts,
    }


def write_canonical_challenge_acceptance_current(
    root: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any] | None:
    """Refresh the canonical challenge acceptance current artifact when inputs exist."""

    project_root = Path(root)
    manifest_path = project_root / "data/releases/amst_challenge_v1/manifest.json"
    public_manifest_path = project_root / "data/releases/amst_challenge_v1_public/manifest.json"
    if not manifest_path.exists() or not public_manifest_path.exists():
        return None
    output_path = (
        Path(output)
        if output is not None
        else project_root / "reports/examples/amst_challenge_v1_acceptance_current.json"
    )
    output_root = project_root / "reports/examples"
    try:
        write_canonical_challenge_release_currents(project_root, output_dir=output_root)
    except FileNotFoundError:
        pass
    return write_challenge_release_acceptance(
        manifest_path,
        output_path,
        release_validation_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_challenge_v1_release_validation.json"
        ),
        public_release_manifest_path=public_manifest_path,
        public_release_validation_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_challenge_v1_public_release_validation.json"
        ),
        representative_reports_dir=project_root / "reports/examples",
        question_craftsmanship_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_challenge_v1_question_craftsmanship_audit.json"
        ),
        query_construction_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_challenge_v1_query_construction_audit.json"
        ),
        probe_discriminativeness_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_challenge_v1_probe_discriminativeness_audit.json"
        ),
        difficulty_calibration_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_challenge_v1_difficulty_calibration_audit.json"
        ),
        domain_construct_validity_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_challenge_v1_domain_construct_validity_audit.json"
        ),
    )


def write_canonical_hidden_quarterly_acceptance_current(
    root: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any] | None:
    """Refresh the canonical quarterly hidden acceptance current artifact when inputs exist."""

    project_root = Path(root)
    manifest_path = project_root / "data/releases/amst_hidden_quarterly_v1/manifest.json"
    if not manifest_path.exists():
        return None
    output_path = (
        Path(output)
        if output is not None
        else project_root / "reports/examples/amst_hidden_quarterly_v1_acceptance_current.json"
    )
    output_root = project_root / "reports/examples"
    try:
        write_canonical_hidden_quarterly_release_currents(project_root, output_dir=output_root)
    except FileNotFoundError:
        pass
    return write_hidden_quarterly_release_acceptance(
        manifest_path,
        output_path,
        release_validation_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_hidden_quarterly_v1_release_validation.json"
        ),
        intrinsic_sanity_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_hidden_quarterly_v1_release_intrinsic_sanity.json"
        ),
        question_craftsmanship_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_hidden_quarterly_v1_question_craftsmanship_audit.json"
        ),
        query_construction_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_hidden_quarterly_v1_query_construction_audit.json"
        ),
    )


def _write_canonical_main_quality_audits_current(
    project_root: Path,
    *,
    release_manifest_path: Path,
    public_manifest_path: Path,
    output_prefix: str,
    output_root: Path,
    representative_split: str = "public_dev",
    foundation_validation_source_path: Path | None = None,
) -> dict[str, Any]:
    from amb.benchmark.quality.difficulty_calibration import write_difficulty_calibration_audit
    from amb.benchmark.quality.domain_construct_validity import write_domain_construct_validity_audit
    from amb.benchmark.quality.foundation_validation import write_foundation_protocol_audit
    from amb.benchmark.quality.human_audit_subset import write_human_audit_subset_audit
    from amb.benchmark.quality.probe_discriminativeness import write_probe_discriminativeness_audit
    from amb.benchmark.quality.question_craftsmanship import write_question_craftsmanship_audit
    from amb.benchmark.quality.query_construction import write_query_construction_audit

    output_root.mkdir(parents=True, exist_ok=True)
    source_reports_root = project_root / "reports/examples"
    artifacts: dict[str, str] = {}
    public_human_audit_task_manifest = (
        project_root
        / "reports/examples/human_audit_tasks"
        / public_manifest_path.parent.name.removeprefix("amst_")
        / "task_manifest.json"
    )

    human_audit_subset_path = output_root / f"{output_prefix}_human_audit_subset_audit_current.json"
    if public_human_audit_task_manifest.exists():
        _read_or_build_json_artifact(
            human_audit_subset_path,
            source_paths=(public_manifest_path, public_human_audit_task_manifest),
            build=lambda: write_human_audit_subset_audit(
                human_audit_subset_path,
                manifest_path=public_manifest_path,
                task_manifest_path=public_human_audit_task_manifest,
            ),
        )
        artifacts["human_audit_subset"] = _artifact_path_ref(human_audit_subset_path, project_root)
    elif human_audit_subset_path.exists():
        human_audit_subset_path.unlink()

    question_craftsmanship_path = output_root / f"{output_prefix}_question_craftsmanship_audit_current.json"
    _read_or_build_json_artifact(
        question_craftsmanship_path,
        source_paths=(public_manifest_path,),
        build=lambda: write_question_craftsmanship_audit(question_craftsmanship_path, manifest_path=public_manifest_path),
    )
    artifacts["question_craftsmanship"] = _artifact_path_ref(question_craftsmanship_path, project_root)

    query_construction_path = output_root / f"{output_prefix}_query_construction_audit_current.json"
    _read_or_build_json_artifact(
        query_construction_path,
        source_paths=(release_manifest_path,),
        build=lambda: write_query_construction_audit(query_construction_path, manifest_path=release_manifest_path),
    )
    artifacts["query_construction"] = _artifact_path_ref(query_construction_path, project_root)

    probe_discriminativeness_path = output_root / f"{output_prefix}_probe_discriminativeness_audit_current.json"
    _read_or_build_json_artifact(
        probe_discriminativeness_path,
        source_paths=(public_manifest_path, *tuple(_representative_report_source_paths(source_reports_root, public_manifest_path, representative_split))),
        build=lambda: write_probe_discriminativeness_audit(
            probe_discriminativeness_path,
            manifest_path=public_manifest_path,
            split=representative_split,
            reports_dir=source_reports_root,
        ),
    )
    artifacts["probe_discriminativeness"] = _artifact_path_ref(probe_discriminativeness_path, project_root)

    difficulty_calibration_path = output_root / f"{output_prefix}_difficulty_calibration_audit_current.json"
    _read_or_build_json_artifact(
        difficulty_calibration_path,
        source_paths=(public_manifest_path, *tuple(_representative_report_source_paths(source_reports_root, public_manifest_path, representative_split))),
        build=lambda: write_difficulty_calibration_audit(
            difficulty_calibration_path,
            manifest_path=public_manifest_path,
            split=representative_split,
            reports_dir=source_reports_root,
        ),
    )
    artifacts["difficulty_calibration"] = _artifact_path_ref(difficulty_calibration_path, project_root)

    domain_construct_validity_path = output_root / f"{output_prefix}_domain_construct_validity_audit_current.json"
    _read_or_build_json_artifact(
        domain_construct_validity_path,
        source_paths=(public_manifest_path, *tuple(_representative_report_source_paths(source_reports_root, public_manifest_path, representative_split))),
        build=lambda: write_domain_construct_validity_audit(
            domain_construct_validity_path,
            manifest_path=public_manifest_path,
            split=representative_split,
            reports_dir=source_reports_root,
        ),
    )
    artifacts["domain_construct_validity"] = _artifact_path_ref(domain_construct_validity_path, project_root)

    if foundation_validation_source_path is not None:
        report_paths = _resolve_foundation_protocol_report_paths(foundation_validation_source_path, project_root)
        if report_paths:
            foundation_source = read_json(foundation_validation_source_path)
            foundation_validation_path = output_root / f"{output_prefix}_foundation_validation_audit_current.json"
            _read_or_build_json_artifact(
                foundation_validation_path,
                source_paths=(foundation_validation_source_path, *report_paths),
                build=lambda: write_foundation_protocol_audit(
                    foundation_validation_path,
                    report_paths,
                    expected_benchmark_id=foundation_source.get("expected_benchmark_id"),
                    cohort_id=foundation_source.get("cohort_id"),
                    require_full_history="full_history" in dict(foundation_source.get("protocol_reports", {})),
                ),
            )
            artifacts["foundation_validation"] = _artifact_path_ref(foundation_validation_path, project_root)

    return {
        "profile_prefix": output_prefix,
        "artifacts": artifacts,
    }


def write_canonical_main_quality_audits_current(
    root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Refresh current question-quality audits for the compatibility main release."""

    project_root = Path(root)
    release_manifest_path = project_root / "data/releases/amst_main_v1/manifest.json"
    public_manifest_path = project_root / "data/releases/amst_main_v1_public/manifest.json"
    if not release_manifest_path.exists() or not public_manifest_path.exists():
        return None
    return _write_canonical_main_quality_audits_current(
        project_root,
        release_manifest_path=release_manifest_path,
        public_manifest_path=public_manifest_path,
        output_prefix="amst_main_v1",
        output_root=Path(output_dir) if output_dir is not None else project_root / "reports/examples",
    )


def write_canonical_strict_main_quality_audits_current(
    root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Refresh current question-quality audits for the canonical strict main release."""

    project_root = Path(root)
    release_manifest_path = project_root / "data/releases/amst_main_v1_strict/manifest.json"
    public_manifest_path = project_root / "data/releases/amst_main_v1_strict_public/manifest.json"
    if not release_manifest_path.exists() or not public_manifest_path.exists():
        return None
    foundation_validation_source_path = project_root / "reports/examples/amst_main_v1_strict_foundation_validation_audit.json"
    return _write_canonical_main_quality_audits_current(
        project_root,
        release_manifest_path=release_manifest_path,
        public_manifest_path=public_manifest_path,
        output_prefix="amst_main_v1_strict",
        output_root=Path(output_dir) if output_dir is not None else project_root / "reports/examples",
        foundation_validation_source_path=foundation_validation_source_path,
    )


def write_canonical_main_acceptance_current(
    root: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any] | None:
    """Refresh the canonical compatibility-main acceptance current artifact when inputs exist.

    Temporary test roots that only exercise the real-system watcher do not carry the
    canonical main-v1 release tree. In that case this helper returns ``None`` so the
    watcher can keep running without fabricating repo-specific fixtures.
    """

    project_root = Path(root)
    manifest_path = project_root / "data/releases/amst_main_v1/manifest.json"
    if not manifest_path.exists():
        return None
    output_path = Path(output) if output is not None else project_root / "reports/examples/amst_main_v1_acceptance_current.json"
    output_root = project_root / "reports/examples"
    try:
        write_canonical_main_release_support_currents(project_root, output_dir=output_root)
        write_canonical_main_quality_audits_current(project_root, output_dir=output_root)
    except FileNotFoundError:
        pass
    return write_main_dataset_acceptance(
        manifest_path,
        output_path,
        expected_benchmark_id="amst-main-v1",
        expected_profile_id="main-v1",
        run_release_validation=True,
        release_validation_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_release_validation.json"
        ),
        public_release_manifest_path=project_root / "data/releases/amst_main_v1_public/manifest.json",
        public_release_validation_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_public_release_validation.json"
        ),
        require_public_release_validation=True,
        representative_reports_dir=project_root / "reports/examples",
        representative_split="public_dev",
        require_representative_baselines=True,
        intrinsic_sanity_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_intrinsic_sanity.json"
        ),
        require_intrinsic_sanity=True,
        public_test_sanity_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_public_test_sanity.json"
        ),
        require_public_test_sanity=True,
        public_result_slices_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_public_test_required_slices.json"
        ),
        require_public_result_slices=True,
        hidden_test_sanity_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_hidden_test_sanity.json"
        ),
        require_hidden_test_sanity=True,
        human_audit_subset_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_human_audit_subset_audit.json"
        ),
        require_human_audit_subset=True,
        question_craftsmanship_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_question_craftsmanship_audit.json"
        ),
        require_question_craftsmanship=True,
        query_construction_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_query_construction_audit.json"
        ),
        require_query_construction=True,
        probe_discriminativeness_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_probe_discriminativeness_audit.json"
        ),
        require_probe_discriminativeness=True,
        difficulty_calibration_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_difficulty_calibration_audit.json"
        ),
        require_difficulty_calibration=True,
        domain_construct_validity_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_domain_construct_validity_audit.json"
        ),
        require_domain_construct_validity=True,
        require_query_difficulty=True,
    )


def write_canonical_strict_main_acceptance_current(
    root: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any] | None:
    """Refresh the canonical strict-main acceptance current artifact when inputs exist."""

    project_root = Path(root)
    manifest_path = project_root / "data/releases/amst_main_v1_strict/manifest.json"
    if not manifest_path.exists():
        return None
    output_path = (
        Path(output)
        if output is not None
        else project_root / "reports/examples/amst_main_v1_strict_acceptance_current.json"
    )
    output_root = project_root / "reports/examples"
    try:
        write_canonical_strict_main_release_support_currents(project_root, output_dir=output_root)
        write_canonical_strict_main_quality_audits_current(project_root, output_dir=output_root)
    except FileNotFoundError:
        pass
    return write_main_dataset_acceptance(
        manifest_path,
        output_path,
        expected_benchmark_id="amst-main-v1-strict",
        expected_profile_id="main-v1-strict",
        run_release_validation=True,
        release_validation_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_strict_release_validation.json"
        ),
        public_release_manifest_path=project_root / "data/releases/amst_main_v1_strict_public/manifest.json",
        public_release_validation_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_strict_public_release_validation.json"
        ),
        require_public_release_validation=True,
        require_all_counterfactual_axes=True,
        representative_reports_dir=project_root / "reports/examples",
        representative_split="public_dev",
        require_representative_baselines=True,
        intrinsic_sanity_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_strict_intrinsic_sanity.json"
        ),
        require_intrinsic_sanity=True,
        public_test_sanity_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_strict_public_test_sanity.json"
        ),
        require_public_test_sanity=True,
        public_result_slices_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_strict_public_test_required_slices.json"
        ),
        require_public_result_slices=True,
        hidden_test_sanity_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_strict_hidden_test_sanity.json"
        ),
        require_hidden_test_sanity=True,
        human_audit_subset_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_strict_human_audit_subset_audit.json"
        ),
        require_human_audit_subset=True,
        question_craftsmanship_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_strict_question_craftsmanship_audit.json"
        ),
        require_question_craftsmanship=True,
        query_construction_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_strict_query_construction_audit.json"
        ),
        require_query_construction=True,
        probe_discriminativeness_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_strict_probe_discriminativeness_audit.json"
        ),
        require_probe_discriminativeness=True,
        difficulty_calibration_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_strict_difficulty_calibration_audit.json"
        ),
        require_difficulty_calibration=True,
        domain_construct_validity_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_strict_domain_construct_validity_audit.json"
        ),
        require_domain_construct_validity=True,
        foundation_validation_report_path=_prefer_current_example_artifact(
            project_root / "reports/examples/amst_main_v1_strict_foundation_validation_audit.json"
        ),
        require_foundation_validation=True,
        require_query_difficulty=True,
    )


def _check_split_report(
    checks: list[dict[str, Any]],
    errors: list[str],
    split: str,
    report: dict[str, Any],
    *,
    profile_id: str | None,
) -> None:
    _check_equal(checks, errors, f"{split}.quality_gates_passed", report.get("quality_gates_passed"), True)
    _check_equal(checks, errors, f"{split}.construction_gates_passed", report.get("construction_gates_passed"), True)
    _check_equal(checks, errors, f"{split}.data_quality_gates_passed", report.get("data_quality_gates_passed"), True)
    _check_contains(
        checks,
        errors,
        f"{split}.construction_gates",
        report.get("construction_gates", {}).keys(),
        REQUIRED_SPLIT_CONSTRUCTION_GATES,
    )
    for gate in REQUIRED_SPLIT_CONSTRUCTION_GATES:
        _check_equal(
            checks,
            errors,
            f"{split}.construction_gates.{gate}",
            bool(report.get("construction_gates", {}).get(gate)),
            True,
        )
    _check_contains(
        checks,
        errors,
        f"{split}.data_quality_gates",
        report.get("data_quality_gates", {}).keys(),
        REQUIRED_SPLIT_DATA_QUALITY_GATES,
    )
    for gate in REQUIRED_SPLIT_DATA_QUALITY_GATES:
        _check_equal(
            checks,
            errors,
            f"{split}.data_quality_gates.{gate}",
            bool(report.get("data_quality_gates", {}).get(gate)),
            True,
        )
    _check_contains(checks, errors, f"{split}.domains", report.get("domains", {}).keys(), REQUIRED_MAIN_DOMAINS)
    _check_contains(
        checks,
        errors,
        f"{split}.probe_types",
        report.get("probe_types", {}).keys(),
        _required_probe_types_for_profile(profile_id),
    )
    _check_contains(checks, errors, f"{split}.memory_types", _normalized_memory_types(report.get("memory_types", {}).keys()), REQUIRED_MEMORY_TYPES)
    _check_contains(checks, errors, f"{split}.event_types", report.get("event_types", {}).keys(), REQUIRED_EVENT_TYPES)
    _check_contains(checks, errors, f"{split}.renderer_coverage", report.get("renderer_coverage", {}).keys(), REQUIRED_RENDERERS)
    coverage = report.get("coverage", {})
    for field in (
        "memory_required_queries",
        "no_memory_queries",
        "refusal_queries",
        "sensitive_memories",
        "deleted_memories",
        "stale_capable_memories",
        "state_bound_queries",
        "forbidden_probe_queries",
    ):
        _check_equal(checks, errors, f"{split}.coverage.{field}", _as_int(coverage.get(field)) > 0, True)
    for field in ("governance_cases", "counterfactual_cases", "cross_subject_cases"):
        if field not in coverage:
            continue
        _check_equal(
            checks,
            errors,
            f"{split}.coverage.{field}.present",
            isinstance(coverage.get(field), int),
            True,
        )


def _case_scale_summary(manifest: dict[str, Any], manifest_dir: Path) -> dict[str, Any] | None:
    split_files = manifest.get("split_files", {})
    field_values = {field: [] for field in CASE_SCALE_BOUNDS}
    out_of_range_cases: list[dict[str, Any]] = []
    split_case_counts: dict[str, int] = {}
    found_any = False

    for split in RELEASE_SPLITS:
        entries = _split_entries(split_files.get(split))
        if not entries:
            continue
        for _, raw_path in entries:
            path = _resolve_audit_template_path(str(raw_path), manifest_dir)
            if not path.exists():
                continue
            benchmark = load_benchmark(path)
            found_any = True
            split_case_counts[split] = split_case_counts.get(split, 0) + len(benchmark.cases)
            for case in benchmark.cases:
                counts = {
                    "events": len(case.events),
                    "memories": len(case.gold_memory_units),
                    "queries": len(case.queries),
                }
                for field, value in counts.items():
                    field_values[field].append(value)
                violations = {
                    field: value
                    for field, value in counts.items()
                    if not _within_case_scale_bounds(field, value)
                }
                if violations:
                    out_of_range_cases.append(
                        {
                            "case_id": case.case_id,
                            "domain": case.domain,
                            "counts": counts,
                            "violations": violations,
                        }
                    )

    if not found_any:
        return None

    return {
        "num_cases": sum(split_case_counts.values()),
        "split_case_counts": split_case_counts,
        "fields": {
            field: {
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "mean": (sum(values) / len(values)) if values else None,
                "bounds": list(CASE_SCALE_BOUNDS[field]),
            }
            for field, values in field_values.items()
        },
        "out_of_range_cases": len(out_of_range_cases),
        "sample_out_of_range_cases": out_of_range_cases[:20],
    }


def _core_object_summary(manifest: dict[str, Any], manifest_dir: Path) -> dict[str, Any] | None:
    split_files = manifest.get("split_files", {})
    found_any = False
    event_edge_types: set[str] = set()
    transition_types: set[str] = set()
    required_governance_rules: set[str] = set()
    per_domain_state_semantics: dict[str, dict[str, Any]] = {}
    missing_canonical_form = 0
    missing_memory_type_alias = 0
    missing_source_trace_ids = 0
    missing_required_governance_rules = 0
    missing_scenario_metadata = 0
    missing_scenario_time_span = 0
    num_memories = 0
    num_state_contracts = 0

    for split in RELEASE_SPLITS:
        entries = _split_entries(split_files.get(split))
        if not entries:
            continue
        for _, raw_path in entries:
            path = _resolve_audit_template_path(str(raw_path), manifest_dir)
            if not path.exists():
                continue
            benchmark = load_benchmark(path)
            found_any = True
            for case in benchmark.cases:
                domain_summary = per_domain_state_semantics.setdefault(
                    str(case.domain),
                    {
                        "num_memories": 0,
                        "memories_with_validity_window": 0,
                        "non_storable_memories": 0,
                        "deletable_memories": 0,
                        "non_normal_privacy_memories": 0,
                        "non_default_authorization_scope_memories": 0,
                        "governance_state_contracts": 0,
                        "transition_types": set(),
                    },
                )
                if case.scenario is None:
                    missing_scenario_metadata += 1
                elif case.scenario.time_span is None:
                    missing_scenario_time_span += 1
                event_edge_types.update(edge.edge_type for edge in case.event_edges)
                for memory in case.gold_memory_units:
                    num_memories += 1
                    domain_summary["num_memories"] += 1
                    if not memory.canonical_form:
                        missing_canonical_form += 1
                    if not memory.memory_type:
                        missing_memory_type_alias += 1
                    if not memory.source_trace_ids:
                        missing_source_trace_ids += 1
                    domain_summary["memories_with_validity_window"] += int(memory.valid_until is not None)
                    domain_summary["non_storable_memories"] += int(not memory.should_store)
                    domain_summary["deletable_memories"] += int(bool(memory.should_delete))
                    domain_summary["non_normal_privacy_memories"] += int(str(memory.privacy_level).lower() != "normal")
                    domain_summary["non_default_authorization_scope_memories"] += int(
                        str(memory.authorization_scope).lower() != "same_user"
                    )
                for contract in case.state_contracts:
                    num_state_contracts += 1
                    domain_summary["governance_state_contracts"] += int(bool(contract.required_governance_rules))
                    transition_types.update(transition.transition_type for transition in contract.transitions)
                    domain_summary["transition_types"].update(transition.transition_type for transition in contract.transitions)
                    required_governance_rules.update(contract.required_governance_rules)
                    if not contract.required_governance_rules:
                        missing_required_governance_rules += 1

    if not found_any:
        return None

    return {
        "num_memories": num_memories,
        "num_state_contracts": num_state_contracts,
        "event_edge_types": sorted(event_edge_types),
        "transition_types": sorted(transition_types),
        "required_governance_rules": sorted(required_governance_rules),
        "per_domain_state_semantics": {
            domain: {
                **summary,
                "transition_types": sorted(summary["transition_types"]),
            }
            for domain, summary in sorted(per_domain_state_semantics.items())
        },
        "missing_canonical_form": missing_canonical_form,
        "missing_memory_type_alias": missing_memory_type_alias,
        "missing_source_trace_ids": missing_source_trace_ids,
        "missing_required_governance_rules": missing_required_governance_rules,
        "missing_scenario_metadata": missing_scenario_metadata,
        "missing_scenario_time_span": missing_scenario_time_span,
    }


def _normalized_memory_types(values: Any) -> tuple[str, ...]:
    canonical = {
        "preference": "semantic_memory",
        "user_fact": "semantic_memory",
        "planning_constraint": "working_state_memory",
        "tool_observation": "episodic_memory",
        "task_outcome": "episodic_memory",
        "feedback_memory": "reflective_memory",
        "governance_rule": "governance_memory",
        "security_boundary": "governance_memory",
        "ephemeral_context": "working_state_memory",
    }
    normalized = {canonical.get(str(value), str(value)) for value in values or ()}
    return tuple(sorted(normalized))


def _check_core_event_edge_types(
    checks: list[dict[str, Any]],
    errors: list[str],
    core_object_summary: dict[str, Any],
) -> None:
    present = {str(value) for value in core_object_summary.get("event_edge_types", ())}
    required = {
        "temporal_before",
        "supports",
        "updates",
        "contradicts",
        "invalidates",
        "depends_on",
        "same_entity_as",
        "distracts",
    }
    _check_contains(checks, errors, "core_objects.event_edge_types", present, tuple(sorted(required)))
    authorizes_or_forbids = "authorizes" in present or "forbids" in present
    _add_check(
        checks,
        errors,
        "core_objects.event_edge_types.governance_edges_present",
        authorizes_or_forbids,
        "core_objects.event_edge_types.governance_edges_present: expected authorizes or forbids edge coverage",
        {"present": sorted(present), "expected_any_of": ["authorizes", "forbids"]},
    )


def _check_domain_core_object_state_semantics(
    checks: list[dict[str, Any]],
    errors: list[str],
    core_object_summary: dict[str, Any],
) -> None:
    per_domain = core_object_summary.get("per_domain_state_semantics", {})
    _check_equal(checks, errors, "core_objects.per_domain_state_semantics_present", bool(per_domain), True)
    if not isinstance(per_domain, dict) or not per_domain:
        return

    missing_validity_windows = [
        domain for domain, summary in sorted(per_domain.items()) if int(summary.get("memories_with_validity_window", 0)) <= 0
    ]
    _add_check(
        checks,
        errors,
        "core_objects.all_domains_cover_validity_windows",
        not missing_validity_windows,
        f"core_objects.all_domains_cover_validity_windows: missing {missing_validity_windows}",
        {"missing_domains": missing_validity_windows},
    )

    missing_storage_or_deletion = {
        domain: {
            "non_storable_memories": int(summary.get("non_storable_memories", 0)),
            "deletable_memories": int(summary.get("deletable_memories", 0)),
        }
        for domain, summary in sorted(per_domain.items())
        if int(summary.get("non_storable_memories", 0)) <= 0 or int(summary.get("deletable_memories", 0)) <= 0
    }
    _add_check(
        checks,
        errors,
        "core_objects.all_domains_cover_storage_and_deletion_semantics",
        not missing_storage_or_deletion,
        f"core_objects.all_domains_cover_storage_and_deletion_semantics: missing {sorted(missing_storage_or_deletion)}",
        {"missing_domains": missing_storage_or_deletion},
    )

    missing_privacy_or_authorization = {
        domain: {
            "non_normal_privacy_memories": int(summary.get("non_normal_privacy_memories", 0)),
            "non_default_authorization_scope_memories": int(summary.get("non_default_authorization_scope_memories", 0)),
        }
        for domain, summary in sorted(per_domain.items())
        if int(summary.get("non_normal_privacy_memories", 0)) <= 0
        or int(summary.get("non_default_authorization_scope_memories", 0)) <= 0
    }
    _add_check(
        checks,
        errors,
        "core_objects.all_domains_cover_privacy_and_authorization_semantics",
        not missing_privacy_or_authorization,
        f"core_objects.all_domains_cover_privacy_and_authorization_semantics: missing {sorted(missing_privacy_or_authorization)}",
        {"missing_domains": missing_privacy_or_authorization},
    )

    required_transitions = {"update", "delete", "retain"}
    missing_governance_or_transitions = {
        domain: {
            "governance_state_contracts": int(summary.get("governance_state_contracts", 0)),
            "transition_types": sorted(summary.get("transition_types", ())),
        }
        for domain, summary in sorted(per_domain.items())
        if int(summary.get("governance_state_contracts", 0)) <= 0
        or not required_transitions <= {str(value) for value in summary.get("transition_types", ())}
    }
    _add_check(
        checks,
        errors,
        "core_objects.all_domains_cover_governance_and_transition_semantics",
        not missing_governance_or_transitions,
        f"core_objects.all_domains_cover_governance_and_transition_semantics: missing {sorted(missing_governance_or_transitions)}",
        {"missing_domains": missing_governance_or_transitions},
    )


def _within_case_scale_bounds(field: str, value: int) -> bool:
    low, high = CASE_SCALE_BOUNDS[field]
    return low <= value <= high


def _check_release_plan_counts(
    checks: list[dict[str, Any]],
    errors: list[str],
    release_plan: dict[str, Any],
    *,
    expected_counterfactual_variants: int,
) -> None:
    split_plan = release_plan.get("split_reports", {})
    total_base_groups = sum(_as_int(split_plan.get(split, {}).get("base_groups")) for split in RELEASE_SPLITS)
    total_case_variants = sum(_as_int(split_plan.get(split, {}).get("case_variants")) for split in RELEASE_SPLITS)
    _check_equal(checks, errors, "release_plan.total_base_groups", total_base_groups, 1200)
    _check_equal(
        checks,
        errors,
        "release_plan.total_case_variants",
        total_case_variants,
        1200 * (1 + expected_counterfactual_variants),
    )
    _check_equal(checks, errors, "release_plan.audit_subset.base_groups", split_plan.get("audit_subset", {}).get("base_groups"), 120)
    _check_equal(checks, errors, "release_plan.hidden_test.base_groups", split_plan.get("hidden_test", {}).get("base_groups"), 240)
    _check_equal(checks, errors, "release_plan.public_dev.base_groups", split_plan.get("public_dev", {}).get("base_groups"), 240)
    _check_equal(checks, errors, "release_plan.public_test.base_groups", split_plan.get("public_test", {}).get("base_groups"), 600)


def _check_hardness_slices(
    checks: list[dict[str, Any]],
    errors: list[str],
    split_reports: dict[str, Any],
    profile_id: str,
) -> None:
    public_dev = split_reports.get("public_dev", {})
    audit_subset = split_reports.get("audit_subset", {})
    expected_edits = _expected_counterfactual_edits(profile_id)

    _check_equal(
        checks,
        errors,
        "public_dev.coverage.counterfactual_cases",
        _as_int(public_dev.get("coverage", {}).get("counterfactual_cases")) > 0,
        True,
    )
    _check_equal(
        checks,
        errors,
        "public_dev.coverage.governance_cases",
        _as_int(public_dev.get("coverage", {}).get("governance_cases")) > 0,
        True,
    )
    _check_equal(
        checks,
        errors,
        "public_dev.coverage.cross_subject_cases",
        _as_int(public_dev.get("coverage", {}).get("cross_subject_cases")) > 0,
        True,
    )
    _check_equal(
        checks,
        errors,
        "audit_subset.coverage.counterfactual_cases",
        _as_int(audit_subset.get("coverage", {}).get("counterfactual_cases")) > 0,
        True,
    )
    _check_equal(
        checks,
        errors,
        "audit_subset.coverage.governance_cases",
        _as_int(audit_subset.get("coverage", {}).get("governance_cases")) > 0,
        True,
    )
    _check_equal(
        checks,
        errors,
        "audit_subset.coverage.cross_subject_cases",
        _as_int(audit_subset.get("coverage", {}).get("cross_subject_cases")) > 0,
        True,
    )
    _check_contains(
        checks,
        errors,
        "public_dev.counterfactual_edits",
        tuple(public_dev.get("counterfactual_edits", {}).keys()),
        expected_edits,
    )
    _check_contains(
        checks,
        errors,
        "audit_subset.counterfactual_edits",
        tuple(audit_subset.get("counterfactual_edits", {}).keys()),
        expected_edits,
    )


def _check_query_difficulty_coverage(
    checks: list[dict[str, Any]],
    errors: list[str],
    split_reports: dict[str, Any],
) -> None:
    for split in RELEASE_SPLITS:
        report = split_reports.get(split, {})
        levels = report.get("query_difficulty_levels", {})
        _check_equal(
            checks,
            errors,
            f"{split}.query_difficulty.present",
            bool(levels),
            True,
        )
        if not levels:
            continue
        _check_contains(
            checks,
            errors,
            f"{split}.query_difficulty.levels",
            tuple(levels.keys()),
            ("easy", "medium", "hard"),
        )
        _check_equal(
            checks,
            errors,
            f"{split}.query_difficulty.missing",
            _as_int(levels.get("missing")),
            0,
        )


class _Unset:
    pass


_UNSET = _Unset()


def _check_equal(
    checks: list[dict[str, Any]],
    errors: list[str],
    check_id: str,
    actual: Any,
    expected: Any,
) -> None:
    passed = actual == expected
    detail = {"actual": actual, "expected": expected}
    _add_check(checks, errors, check_id, passed, f"{check_id}: expected {expected!r}, got {actual!r}", detail)


def _artifact_root_ref(base_dir: Path, project_root: Path) -> str:
    try:
        return Path(os.path.relpath(project_root.resolve(), base_dir.resolve())).as_posix()
    except ValueError:
        return str(project_root.resolve())


def _infer_acceptance_project_root(manifest_path: Path, output_path: Path) -> Path | None:
    manifest_resolved = manifest_path.resolve()
    output_resolved = output_path.resolve()
    for candidate in (
        _project_root_from_named_ancestor(manifest_resolved, "data"),
        _project_root_from_named_ancestor(output_resolved, "reports"),
    ):
        if candidate is not None:
            return candidate
    return None


def _project_root_from_named_ancestor(path: Path, anchor: str) -> Path | None:
    current = path if path.is_dir() else path.parent
    for parent in (current, *current.parents):
        if parent.name == anchor:
            return parent.parent
    return None


def _normalize_acceptance_report_paths(value: Any, project_root: Path) -> Any:
    if isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _normalize_acceptance_report_paths(item, project_root)
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _normalize_acceptance_report_paths(item, project_root)
        return value
    if isinstance(value, str):
        return _project_relative_or_original(value, project_root)
    return value


def _project_relative_or_original(raw_value: str, project_root: Path) -> str:
    try:
        path = Path(raw_value)
    except TypeError:
        return raw_value
    if not path.is_absolute():
        return raw_value
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return raw_value


def _check_artifact_binding(
    checks: list[dict[str, Any]],
    errors: list[str],
    prefix: str,
    report: dict[str, Any],
    *,
    report_path: str | Path | None = None,
    expected_benchmark_id: str | None = None,
    expected_release_split: str | None | object = _UNSET,
    expected_source_type: str | None = None,
    expected_manifest_path: Path | None = None,
    expected_manifest_paths: tuple[Path, ...] | None = None,
) -> None:
    if expected_benchmark_id is not None:
        _check_equal(checks, errors, f"{prefix}.benchmark_id", report.get("benchmark_id"), expected_benchmark_id)
    if expected_release_split is not _UNSET:
        _check_equal(checks, errors, f"{prefix}.release_split", report.get("release_split"), expected_release_split)
    if expected_source_type is not None:
        _check_equal(checks, errors, f"{prefix}.source_type", report.get("source_type"), expected_source_type)
    allowed_manifest_paths = tuple(path for path in (expected_manifest_path,) if path is not None)
    if expected_manifest_paths is not None:
        allowed_manifest_paths = tuple(path for path in expected_manifest_paths if path is not None)
    if allowed_manifest_paths:
        manifest_path = report.get("manifest_path")
        _check_equal(checks, errors, f"{prefix}.manifest_path_present", bool(manifest_path), True)
        if manifest_path:
            resolved_manifest_path = _resolve_report_path(report, manifest_path, report_path)
            expected_manifest_path_set = {_resolved_path(path) for path in allowed_manifest_paths}
            _add_check(
                checks,
                errors,
                f"{prefix}.manifest_path",
                resolved_manifest_path in expected_manifest_path_set,
                f"{prefix}.manifest_path: expected one of {sorted(expected_manifest_path_set)}, got {resolved_manifest_path!r}",
                {"actual": resolved_manifest_path, "expected": sorted(expected_manifest_path_set)},
            )


def _check_human_audit_binding(
    checks: list[dict[str, Any]],
    errors: list[str],
    manifest: dict[str, Any],
    manifest_file: Path,
    audit_plan: dict[str, Any],
    verification: dict[str, Any],
) -> None:
    manifest_dir = manifest_file.parent
    _check_equal(checks, errors, "human_audit.benchmark_id", verification.get("benchmark_id"), str(manifest.get("benchmark_id") or ""))
    _check_equal(
        checks,
        errors,
        "human_audit.manifest_path",
        _resolve_path_against(manifest_dir, verification.get("manifest_path")),
        _resolve_path_against(manifest_dir, manifest_file),
    )
    _check_equal(
        checks,
        errors,
        "human_audit.task_manifest_file",
        _resolve_path_against(manifest_dir, verification.get("task_manifest_file")),
        _resolve_path_against(manifest_dir, audit_plan.get("audit_task_manifest_file")),
    )
    _check_equal(
        checks,
        errors,
        "human_audit.annotations_file",
        _resolve_path_against(manifest_dir, verification.get("annotations_file")),
        _resolve_path_against(manifest_dir, audit_plan.get("audit_annotations_file")),
    )
    _check_equal(
        checks,
        errors,
        "human_audit.annotator_attestation_file",
        _resolve_path_against(manifest_dir, verification.get("annotator_attestation_file")),
        _resolve_path_against(manifest_dir, audit_plan.get("annotator_attestation_file")),
    )
    expected_templates = sorted(_expected_human_audit_template_paths(audit_plan, manifest_dir))
    actual_templates = sorted(
        path
        for path in (
            _resolve_path_against(manifest_dir, value)
            for value in verification.get("template_files", ())
        )
        if path is not None
    )
    _check_equal(checks, errors, "human_audit.template_files", actual_templates, expected_templates)
    digest_payload = verification.get("file_digests", {}) if isinstance(verification.get("file_digests"), dict) else {}
    _check_equal(
        checks,
        errors,
        "human_audit.manifest_file_sha256",
        digest_payload.get("manifest_file"),
        _file_sha256(manifest_file),
    )
    _check_equal(
        checks,
        errors,
        "human_audit.template_files_sha256",
        _normalize_digest_map(digest_payload.get("template_files")),
        {path: _file_sha256(Path(path)) for path in expected_templates},
    )
    _check_equal(
        checks,
        errors,
        "human_audit.annotations_file_sha256",
        digest_payload.get("annotations_file"),
        _file_sha256(Path(_resolve_path_against(manifest_dir, audit_plan.get("audit_annotations_file")) or "")),
    )
    _check_equal(
        checks,
        errors,
        "human_audit.task_manifest_file_sha256",
        digest_payload.get("task_manifest_file"),
        _file_sha256(Path(_resolve_path_against(manifest_dir, audit_plan.get("audit_task_manifest_file")) or "")),
    )
    _check_equal(
        checks,
        errors,
        "human_audit.annotator_attestation_file_sha256",
        digest_payload.get("annotator_attestation_file"),
        _file_sha256(Path(_resolve_path_against(manifest_dir, audit_plan.get("annotator_attestation_file")) or "")),
    )
    adjudication_required = bool((verification.get("adjudication_summary") or {}).get("required"))
    if adjudication_required or audit_plan.get("audit_adjudication_file") or verification.get("adjudication_file"):
        _check_equal(
            checks,
            errors,
            "human_audit.adjudication_file_declared",
            bool(audit_plan.get("audit_adjudication_file")),
            True,
        )
        _check_equal(
            checks,
            errors,
            "human_audit.adjudication_file",
            _resolve_path_against(manifest_dir, verification.get("adjudication_file")),
            _resolve_path_against(manifest_dir, audit_plan.get("audit_adjudication_file")),
        )
        _check_equal(
            checks,
            errors,
            "human_audit.adjudication_file_sha256",
            digest_payload.get("adjudication_file"),
            _file_sha256(Path(_resolve_path_against(manifest_dir, audit_plan.get("audit_adjudication_file")) or "")),
        )


def _check_contains(
    checks: list[dict[str, Any]],
    errors: list[str],
    check_id: str,
    actual_values: Any,
    required_values: tuple[str, ...],
) -> None:
    actual = {str(value) for value in actual_values or ()}
    missing = [value for value in required_values if value not in actual]
    detail = {"missing": missing, "required": list(required_values), "actual_count": len(actual)}
    _add_check(checks, errors, check_id, not missing, f"{check_id}: missing {missing}", detail)


def _add_check(
    checks: list[dict[str, Any]],
    errors: list[str],
    check_id: str,
    passed: bool,
    message: str,
    detail: dict[str, Any] | None = None,
) -> None:
    checks.append({"check_id": check_id, "status": "passed" if passed else "failed", "detail": detail or {}})
    if not passed:
        errors.append(message)


def _check_flag(
    checks: list[dict[str, Any]],
    errors: list[str],
    check_id: str,
    passed: bool,
    *,
    detail: dict[str, Any] | None = None,
) -> None:
    _add_check(checks, errors, check_id, passed, f"{check_id}: expected representative sanity check to pass", detail)


def _required_probe_types_for_profile(profile_id: str | None) -> tuple[str, ...]:
    return STRICT_REQUIRED_PROBE_TYPES if profile_id == CANONICAL_FINAL_MAIN_PROFILE_ID else BASE_REQUIRED_PROBE_TYPES


def _base_query_scale_check_id(profile_id: str | None) -> str:
    return "base_queries_in_9600_to_30000" if profile_id == CANONICAL_FINAL_MAIN_PROFILE_ID else "base_queries_in_9600_to_18000"


def _scale_check(summary: dict[str, Any], key: str) -> bool | None:
    value = summary.get("final_main_scale_checks", {}).get(key)
    return bool(value) if value is not None else None


def _resolved_path(value: str | Path) -> str:
    return str(Path(value).resolve())


def _resolve_report_path(report: dict[str, Any], raw_value: str | Path, report_path: str | Path | None) -> str:
    path = Path(raw_value)
    if path.is_absolute():
        return str(path.resolve())
    if report_path is not None:
        report_file = Path(report_path).resolve()
        root = report.get("root")
        if isinstance(root, str) and root:
            return str((report_file.parent / root / path).resolve())
        return str((report_file.parent / path).resolve())
    return str(path.resolve())


def _resolve_path_against(base_dir: Path, raw_value: Any) -> str | None:
    if raw_value in (None, ""):
        return None
    path = Path(str(raw_value))
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _expected_human_audit_template_paths(audit_plan: dict[str, Any], manifest_dir: Path) -> list[str]:
    paths: list[str] = []
    template_files = audit_plan.get("audit_template_files")
    if isinstance(template_files, dict):
        for raw_path in template_files.values():
            resolved = _resolve_path_against(manifest_dir, raw_path)
            if resolved is not None:
                paths.append(resolved)
    raw_template = audit_plan.get("audit_template_file")
    if raw_template:
        resolved = _resolve_path_against(manifest_dir, raw_template)
        if resolved is not None and resolved not in paths:
            paths.append(resolved)
    return paths


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_digest_map(value: Any) -> dict[str, str | None]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, str | None] = {}
    for raw_path, digest in value.items():
        try:
            normalized[str(Path(str(raw_path)).resolve())] = str(digest) if digest is not None else None
        except OSError:
            normalized[str(raw_path)] = str(digest) if digest is not None else None
    return normalized


def _audit_template_summary(audit_plan: dict[str, Any], manifest_dir: Path) -> dict[str, Any] | None:
    template_files = audit_plan.get("audit_template_files")
    if not isinstance(template_files, dict) or not template_files:
        return None
    per_domain_queries: dict[str, int] = {}
    per_domain_cases: dict[str, set[str]] = {}
    total_queries = 0
    for domain, raw_path in sorted(template_files.items()):
        path = _resolve_audit_template_path(str(raw_path), manifest_dir)
        if not path.exists():
            return None
        data = read_jsonl(path)
        per_domain_queries[str(domain)] = len(data)
        per_domain_cases[str(domain)] = {str(item.get("case_id")) for item in data if item.get("case_id") is not None}
        total_queries += len(data)
    min_queries = min(per_domain_queries.values()) if per_domain_queries else 0
    return {
        "num_domains": len(per_domain_queries),
        "num_queries": total_queries,
        "per_domain_query_counts": {key: per_domain_queries[key] for key in sorted(per_domain_queries)},
        "per_domain_case_counts": {key: len(per_domain_cases[key]) for key in sorted(per_domain_cases)},
        "per_domain_min_queries": min_queries,
        "per_domain_min_queries_at_least_100": min_queries >= 100,
    }


def _split_entries(value: Any) -> tuple[tuple[str, str], ...]:
    if isinstance(value, dict):
        return tuple((str(label), str(path)) for label, path in sorted(value.items()))
    if isinstance(value, str):
        return (("benchmark", value),)
    return ()


def _check_representative_baselines(
    checks: list[dict[str, Any]],
    errors: list[str],
    manifest: dict[str, Any],
    *,
    reports_dir: Path,
    split: str,
    require_reports: bool,
) -> dict[str, Any]:
    summary = _representative_baseline_summary(manifest, reports_dir=reports_dir, split=split)
    if require_reports or summary.get("present"):
        _check_flag(
            checks,
            errors,
            "representative_baselines.present",
            bool(summary.get("present")),
            detail={"reports_dir": str(reports_dir), "missing_files": summary.get("missing_files", [])},
        )
    if not summary.get("present"):
        return summary

    _check_flag(
        checks,
        errors,
        "representative_baselines.report_benchmark_id_match",
        bool(summary.get("report_benchmark_id_match")),
        detail=summary.get("benchmark_ids"),
    )
    _check_flag(
        checks,
        errors,
        "representative_baselines.leaderboard_top_is_oracle",
        bool(summary.get("leaderboard_top_is_oracle")),
        detail={"leaderboard_top_system_id": summary.get("leaderboard_top_system_id")},
    )
    for check_id, detail in summary.get("sanity_checks", {}).items():
        _check_flag(
            checks,
            errors,
            f"representative_baselines.{check_id}",
            bool(detail.get("passed")),
            detail=detail,
        )
    return summary


def _representative_baseline_summary(manifest: dict[str, Any], *, reports_dir: Path, split: str) -> dict[str, Any]:
    benchmark_id = str(manifest.get("benchmark_id", "release"))
    expected_report_benchmark_id = f"{benchmark_id}-{split}"
    prefix = f"{benchmark_id.replace('-', '_')}_{split}"
    report_paths = {
        kind: _prefer_current_example_artifact(reports_dir / f"{prefix}_{kind}_report.json")
        for kind in REPRESENTATIVE_BASELINES
    }
    analysis_path = _prefer_current_example_artifact(reports_dir / f"{prefix}_representative_baselines_analysis.json")
    leaderboard_path = _prefer_current_example_artifact(reports_dir / f"{prefix}_leaderboard.json")
    all_paths = {**report_paths, "analysis": analysis_path, "leaderboard": leaderboard_path}
    missing_files = [str(path) for path in all_paths.values() if not path.exists()]
    if missing_files:
        return {
            "present": False,
            "reports_dir": str(reports_dir),
            "release_split": split,
            "expected_report_benchmark_id": expected_report_benchmark_id,
            "report_paths": {key: str(path) for key, path in all_paths.items()},
            "missing_files": missing_files,
        }

    reports = {kind: read_json(path) for kind, path in report_paths.items()}
    analysis = read_json(analysis_path)
    if not isinstance(analysis.get("weight_sensitivity"), dict) or not analysis.get("weight_sensitivity", {}).get("profiles"):
        from amb.benchmark.analysis import build_weight_sensitivity_analysis

        analysis = dict(analysis)
        analysis["weight_sensitivity"] = build_weight_sensitivity_analysis(
            [(str(report_paths[kind]), reports[kind]) for kind in REPRESENTATIVE_BASELINES]
        )
    leaderboard = read_json(leaderboard_path)
    leaderboard_rows = leaderboard.get("rows", []) if isinstance(leaderboard, dict) else []
    leaderboard_top_system_id = leaderboard_rows[0].get("system_id") if leaderboard_rows else None
    report_benchmark_ids = {kind: str(report.get("benchmark_id")) for kind, report in reports.items()}
    metrics = {kind: _representative_metrics(report) for kind, report in reports.items()}
    analysis_summary = summarize_representative_analysis(analysis)

    no_memory = metrics["no_memory"]
    full_history = metrics["full_history"]
    graph_memory = metrics["graph_memory"]
    oracle_memory = metrics["oracle_memory"]
    sanity_checks = {
        "oracle_high_amq": {
            "passed": oracle_memory["amq"] >= 0.85,
            "actual": oracle_memory["amq"],
            "expected": ">= 0.85",
        },
        "oracle_requires_memory_task_high": {
            "passed": oracle_memory["requires_memory_task_success"] >= 0.85,
            "actual": oracle_memory["requires_memory_task_success"],
            "expected": ">= 0.85",
        },
        "no_memory_task_low": {
            "passed": no_memory["task_success"] < 0.35,
            "actual": no_memory["task_success"],
            "expected": "< 0.35",
        },
        "no_memory_requires_memory_task_near_zero": {
            "passed": no_memory["requires_memory_task_success"] <= 0.05,
            "actual": no_memory["requires_memory_task_success"],
            "expected": "<= 0.05",
        },
        "graph_memory_beats_no_memory_amq": {
            "passed": graph_memory["amq"] > no_memory["amq"],
            "actual": {"graph_memory": graph_memory["amq"], "no_memory": no_memory["amq"]},
            "expected": "graph_memory > no_memory",
        },
        "full_history_not_worse_than_no_memory_task": {
            "passed": full_history["task_success"] >= no_memory["task_success"],
            "actual": {"full_history": full_history["task_success"], "no_memory": no_memory["task_success"]},
            "expected": "full_history >= no_memory",
        },
        "graph_memory_beats_full_history_requires_memory_recall": {
            "passed": graph_memory["requires_memory_recall_at_k"] > full_history["requires_memory_recall_at_k"],
            "actual": {
                "graph_memory": graph_memory["requires_memory_recall_at_k"],
                "full_history": full_history["requires_memory_recall_at_k"],
            },
            "expected": "graph_memory > full_history",
        },
        "full_history_costs_more_than_graph_memory": {
            "passed": full_history["input_tokens"] > graph_memory["input_tokens"],
            "actual": {"full_history": full_history["input_tokens"], "graph_memory": graph_memory["input_tokens"]},
            "expected": "full_history > graph_memory",
        },
        "full_history_is_safety_weak": {
            "passed": full_history["safety_pass"] < 0.5,
            "actual": full_history["safety_pass"],
            "expected": "< 0.5",
        },
        "graph_memory_is_safety_strong": {
            "passed": graph_memory["safety_pass"] >= 0.95,
            "actual": graph_memory["safety_pass"],
            "expected": ">= 0.95",
        },
        "representative_analysis_bootstrap_samples_sufficient": {
            "passed": analysis_summary["bootstrap_samples_sufficient"],
            "actual": analysis_summary["bootstrap_samples"],
            "expected": ">= 200",
        },
        "representative_analysis_report_bootstrap_cis_present": {
            "passed": analysis_summary["report_bootstrap_cis_present"],
            "actual": analysis_summary["num_report_cis_present"],
            "expected": analysis_summary["required_report_metric_count"],
            "detail": analysis_summary["report_checks"],
        },
        "representative_analysis_pairwise_stats_complete": {
            "passed": analysis_summary["pairwise_stats_complete"],
            "actual": analysis_summary["num_pairwise_metric_stats_present"],
            "expected": analysis_summary["required_pair_metric_count"],
            "detail": analysis_summary["pairwise_checks"],
        },
        "representative_analysis_memory_gains_statistically_visible": {
            "passed": analysis_summary["key_memory_gains_statistically_visible"],
            "actual": analysis_summary["num_key_memory_gain_pairs_visible"],
            "expected": analysis_summary["required_key_gain_pair_count"],
            "metric_visible_count": analysis_summary["num_key_memory_gains_visible"],
            "metric_visible_expected": analysis_summary["required_key_gain_metric_count"],
            "detail": {
                key: analysis_summary["pairwise_checks"][key]
                for key in ("no_memory->graph_memory", "no_memory->oracle_memory")
                if key in analysis_summary["pairwise_checks"]
            },
        },
        "representative_analysis_weight_sensitivity_profiles_complete": {
            "passed": analysis_summary["weight_sensitivity_profiles_complete"],
            "actual": analysis_summary["num_weight_profiles"],
            "expected": analysis_summary["required_weight_profile_count"],
            "detail": analysis_summary["weight_rank_checks"],
        },
        "representative_analysis_oracle_top_rank_stable_under_weight_shifts": {
            "passed": analysis_summary["oracle_top_rank_stable_under_weight_shifts"],
            "actual": analysis_summary["weight_rank_checks"].get("oracle_memory"),
            "expected": "oracle_memory worst_rank == 1",
        },
        "representative_analysis_key_memory_order_stable_under_weight_shifts": {
            "passed": analysis_summary["key_memory_order_stable_under_weight_shifts"],
            "actual": analysis_summary["num_weight_stable_pairs"],
            "expected": analysis_summary["required_weight_stable_pair_count"],
            "detail": analysis_summary["weight_pair_checks"],
        },
    }
    return {
        "present": True,
        "reports_dir": str(reports_dir),
        "release_split": split,
        "expected_report_benchmark_id": expected_report_benchmark_id,
        "report_paths": {key: str(path) for key, path in all_paths.items()},
        "benchmark_ids": report_benchmark_ids,
        "report_benchmark_id_match": all(value == expected_report_benchmark_id for value in report_benchmark_ids.values()),
        "leaderboard_top_system_id": leaderboard_top_system_id,
        "leaderboard_top_is_oracle": leaderboard_top_system_id == "oracle_memory",
        "metrics": metrics,
        "analysis_summary": analysis_summary,
        "sanity_checks": sanity_checks,
    }


def _representative_metrics(report: dict[str, Any]) -> dict[str, Any]:
    aggregate = report.get("aggregate", {})
    requires_memory = report.get("by_memory_requirement", {}).get("requires_memory", {})
    return {
        "amq": _as_float(aggregate.get("lifecycle.amq"), 0.0),
        "task_success": _as_float(aggregate.get("task.task_success"), 0.0),
        "recall_at_k": _as_float(aggregate.get("retrieval.recall_at_k"), 0.0),
        "safety_pass": _as_float(aggregate.get("safety.safety_pass"), 0.0),
        "input_tokens": _as_float(aggregate.get("efficiency.input_tokens"), 0.0),
        "requires_memory_amq": _as_float(requires_memory.get("lifecycle.amq"), 0.0),
        "requires_memory_task_success": _as_float(requires_memory.get("task.task_success"), 0.0),
        "requires_memory_recall_at_k": _as_float(requires_memory.get("retrieval.recall_at_k"), 0.0),
    }


def _resolve_audit_template_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return manifest_dir / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _field_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _expected_counterfactual_variants(profile_id: str) -> int:
    if profile_id == "main-v1-strict":
        return 5
    return 2


def _expected_counterfactual_edits(profile_id: str) -> tuple[str, ...]:
    if profile_id == "main-v1-strict":
        return (
            "base",
            "update_value",
            "retain_deleted_memory",
            "authorize_sensitive_memory",
            "tool_result",
            "role_project_boundary",
        )
    return ("base", "update_value", "retain_deleted_memory")
