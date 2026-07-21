"""Validation, audit, annotation, and data-quality gates."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


_BASE = Path(__file__).resolve().parent
__path__ = [
    str(_BASE),
    str(_BASE / "core"),
    str(_BASE / "human"),
    str(_BASE / "dataset"),
    str(_BASE / "dataset" / "acceptance"),
    str(_BASE / "dataset" / "construction"),
    str(_BASE / "dataset" / "operations"),
    str(_BASE / "calibration"),
    str(_BASE / "systems"),
]

_EXPORTS = {
    "AUDIT_CHECK_FIELDS": ("amb.benchmark.quality.annotation", "AUDIT_CHECK_FIELDS"),
    "DIFFICULTY_CALIBRATION_AUDIT_SCHEMA_VERSION": (
        "amb.benchmark.quality.difficulty_calibration",
        "DIFFICULTY_CALIBRATION_AUDIT_SCHEMA_VERSION",
    ),
    "DOMAIN_CONSTRUCT_VALIDITY_AUDIT_SCHEMA_VERSION": (
        "amb.benchmark.quality.domain_construct_validity",
        "DOMAIN_CONSTRUCT_VALIDITY_AUDIT_SCHEMA_VERSION",
    ),
    "FOUNDATION_VALIDATION_AUDIT_SCHEMA_VERSION": (
        "amb.benchmark.quality.foundation_validation",
        "FOUNDATION_VALIDATION_AUDIT_SCHEMA_VERSION",
    ),
    "LINEAGE_AUDIT_SCHEMA_VERSION": ("amb.benchmark.quality.lineage", "LINEAGE_AUDIT_SCHEMA_VERSION"),
    "MAIN_DATASET_ACCEPTANCE_SCHEMA_VERSION": (
        "amb.benchmark.quality.main_dataset_acceptance",
        "MAIN_DATASET_ACCEPTANCE_SCHEMA_VERSION",
    ),
    "OBJECTIVE_CHECKLIST_SCHEMA_VERSION": (
        "amb.benchmark.quality.objective_checklist",
        "OBJECTIVE_CHECKLIST_SCHEMA_VERSION",
    ),
    "PROBE_DISCRIMINATIVENESS_AUDIT_SCHEMA_VERSION": (
        "amb.benchmark.quality.probe_discriminativeness",
        "PROBE_DISCRIMINATIVENESS_AUDIT_SCHEMA_VERSION",
    ),
    "QUALITY_GATES": ("amb.benchmark.quality.gates", "QUALITY_GATES"),
    "QUERY_CONSTRUCTION_AUDIT_SCHEMA_VERSION": (
        "amb.benchmark.quality.query_construction",
        "QUERY_CONSTRUCTION_AUDIT_SCHEMA_VERSION",
    ),
    "QUESTION_CRAFTSMANSHIP_AUDIT_SCHEMA_VERSION": (
        "amb.benchmark.quality.question_craftsmanship",
        "QUESTION_CRAFTSMANSHIP_AUDIT_SCHEMA_VERSION",
    ),
    "REAL_SYSTEM_EVIDENCE_SCHEMA_VERSION": (
        "amb.benchmark.quality.real_system",
        "REAL_SYSTEM_EVIDENCE_SCHEMA_VERSION",
    ),
    "RELEASE_INTRINSIC_SANITY_SCHEMA_VERSION": (
        "amb.benchmark.quality.release_intrinsic_sanity",
        "RELEASE_INTRINSIC_SANITY_SCHEMA_VERSION",
    ),
    "REQUIRED_REAL_SYSTEM_PROVIDERS": (
        "amb.benchmark.quality.real_system",
        "REQUIRED_REAL_SYSTEM_PROVIDERS",
    ),
    "AuditAnnotation": ("amb.benchmark.quality.annotation", "AuditAnnotation"),
    "ValidationResult": ("amb.benchmark.quality.validation", "ValidationResult"),
    "apply_human_audit_adjudication_packet": (
        "amb.benchmark.quality.human_audit",
        "apply_human_audit_adjudication_packet",
    ),
    "apply_human_audit_annotator_packet": (
        "amb.benchmark.quality.annotation",
        "apply_human_audit_annotator_packet",
    ),
    "audit_benchmark": ("amb.benchmark.quality.audit", "audit_benchmark"),
    "audit_benchmark_lineage": ("amb.benchmark.quality.lineage", "audit_benchmark_lineage"),
    "audit_difficulty_calibration_release": (
        "amb.benchmark.quality.difficulty_calibration",
        "audit_difficulty_calibration_release",
    ),
    "audit_domain_construct_validity_release": (
        "amb.benchmark.quality.domain_construct_validity",
        "audit_domain_construct_validity_release",
    ),
    "audit_probe_discriminativeness_release": (
        "amb.benchmark.quality.probe_discriminativeness",
        "audit_probe_discriminativeness_release",
    ),
    "audit_query_construction_benchmark": (
        "amb.benchmark.quality.query_construction",
        "audit_query_construction_benchmark",
    ),
    "audit_query_construction_release": (
        "amb.benchmark.quality.query_construction",
        "audit_query_construction_release",
    ),
    "audit_question_craftsmanship_benchmark": (
        "amb.benchmark.quality.question_craftsmanship",
        "audit_question_craftsmanship_benchmark",
    ),
    "audit_question_craftsmanship_release": (
        "amb.benchmark.quality.question_craftsmanship",
        "audit_question_craftsmanship_release",
    ),
    "audit_release_lineage": ("amb.benchmark.quality.lineage", "audit_release_lineage"),
    "backfill_real_system_run_metadata": (
        "amb.benchmark.quality.real_system",
        "backfill_real_system_run_metadata",
    ),
    "build_objective_checklist": ("amb.benchmark.quality.objective_checklist", "build_objective_checklist"),
    "compute_agreement": ("amb.benchmark.quality.annotation", "compute_agreement"),
    "contamination_report": ("amb.benchmark.quality.contamination", "contamination_report"),
    "extract_benchmark_text_records": (
        "amb.benchmark.quality.contamination",
        "extract_benchmark_text_records",
    ),
    "finalize_real_system_run": ("amb.benchmark.quality.real_system", "finalize_real_system_run"),
    "ingest_human_audit_return_packets": (
        "amb.benchmark.quality.human_audit",
        "ingest_human_audit_return_packets",
    ),
    "load_audit_annotations": ("amb.benchmark.quality.annotation", "load_audit_annotations"),
    "merge_completed_human_audit_tasks": (
        "amb.benchmark.quality.annotation",
        "merge_completed_human_audit_tasks",
    ),
    "merge_real_system_matrix_summaries": (
        "amb.benchmark.quality.real_system",
        "merge_real_system_matrix_summaries",
    ),
    "normalize_reference_records": ("amb.benchmark.quality.contamination", "normalize_reference_records"),
    "normalize_reference_texts": ("amb.benchmark.quality.contamination", "normalize_reference_texts"),
    "quality_checks": ("amb.benchmark.quality.gates", "quality_checks"),
    "reconcile_human_audit_evidence_bundle": (
        "amb.benchmark.quality.human_audit",
        "reconcile_human_audit_evidence_bundle",
    ),
    "refresh_real_system_canonical_matrix": (
        "amb.benchmark.quality.real_system",
        "refresh_real_system_canonical_matrix",
    ),
    "shingle_jaccard": ("amb.benchmark.quality.contamination", "shingle_jaccard"),
    "summarize_human_audit_progress": (
        "amb.benchmark.quality.annotation",
        "summarize_human_audit_progress",
    ),
    "summarize_real_system_run_progress": (
        "amb.benchmark.quality.real_system",
        "summarize_real_system_run_progress",
    ),
    "sync_human_audit_return_inbox": (
        "amb.benchmark.quality.human_audit",
        "sync_human_audit_return_inbox",
    ),
    "validate_benchmark": ("amb.benchmark.quality.validation", "validate_benchmark"),
    "validate_foundation_protocol_reports": (
        "amb.benchmark.quality.foundation_validation",
        "validate_foundation_protocol_reports",
    ),
    "validate_main_dataset_acceptance": (
        "amb.benchmark.quality.main_dataset_acceptance",
        "validate_main_dataset_acceptance",
    ),
    "validate_predictions": ("amb.benchmark.quality.validation", "validate_predictions"),
    "validate_real_system_matrix_summary": (
        "amb.benchmark.quality.real_system",
        "validate_real_system_matrix_summary",
    ),
    "validate_real_system_report": ("amb.benchmark.quality.real_system", "validate_real_system_report"),
    "validate_release_intrinsic_sanity": (
        "amb.benchmark.quality.release_intrinsic_sanity",
        "validate_release_intrinsic_sanity",
    ),
    "write_difficulty_calibration_audit": (
        "amb.benchmark.quality.difficulty_calibration",
        "write_difficulty_calibration_audit",
    ),
    "write_domain_construct_validity_audit": (
        "amb.benchmark.quality.domain_construct_validity",
        "write_domain_construct_validity_audit",
    ),
    "write_foundation_protocol_audit": (
        "amb.benchmark.quality.foundation_validation",
        "write_foundation_protocol_audit",
    ),
    "write_lineage_audit": ("amb.benchmark.quality.lineage", "write_lineage_audit"),
    "write_main_dataset_acceptance": (
        "amb.benchmark.quality.main_dataset_acceptance",
        "write_main_dataset_acceptance",
    ),
    "write_merged_real_system_matrix_summary": (
        "amb.benchmark.quality.real_system",
        "write_merged_real_system_matrix_summary",
    ),
    "write_objective_checklist": ("amb.benchmark.quality.objective_checklist", "write_objective_checklist"),
    "write_probe_discriminativeness_audit": (
        "amb.benchmark.quality.probe_discriminativeness",
        "write_probe_discriminativeness_audit",
    ),
    "write_query_construction_audit": (
        "amb.benchmark.quality.query_construction",
        "write_query_construction_audit",
    ),
    "write_question_craftsmanship_audit": (
        "amb.benchmark.quality.question_craftsmanship",
        "write_question_craftsmanship_audit",
    ),
    "write_real_system_canonical_refresh": (
        "amb.benchmark.quality.real_system",
        "write_real_system_canonical_refresh",
    ),
    "write_real_system_run_progress": (
        "amb.benchmark.quality.real_system",
        "write_real_system_run_progress",
    ),
    "write_release_intrinsic_sanity": (
        "amb.benchmark.quality.release_intrinsic_sanity",
        "write_release_intrinsic_sanity",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value
