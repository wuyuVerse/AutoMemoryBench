"""Machine-readable audit for pre-label human-audit subset coverage and task packaging."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from amb.benchmark.quality.annotation import (
    AUDIT_CHECK_FIELDS,
    summarize_audit_templates,
    summarize_human_audit_progress,
)
from amb.benchmark.quality.artifact_contract import localize_report_contract
from amb.benchmark.schemas.io import read_json, write_json

HUMAN_AUDIT_SUBSET_AUDIT_SCHEMA_VERSION = "amst-human-audit-subset-audit-v1"
REQUIRED_DIFFICULTY_LEVELS = ("easy", "medium", "hard")
REQUIRED_COUNTERFACTUAL_AXES = ("current_value", "deletion_state")
REQUIRED_AUDIT_PROBE_TYPES = (
    "answer_probe",
    "compression_probe",
    "evolution_probe",
    "forget_probe",
    "governance_probe",
    "no_memory_probe",
    "planning_probe",
    "retrieval_probe",
    "tool_probe",
    "update_probe",
    "write_probe",
)
STRICT_REQUIRED_AUDIT_PROBE_TYPES = tuple(
    sorted(
        (
            *REQUIRED_AUDIT_PROBE_TYPES,
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
    )
)


def audit_human_audit_subset_release(
    manifest_path: str | Path,
    *,
    task_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit audit-subset templates and pending double-annotation task packaging."""

    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    audit_plan = manifest.get("audit_plan", {})

    template_paths = _template_paths_from_audit_plan(audit_plan, manifest_file.parent)
    expected_domains = tuple(sorted(template_paths))
    missing_template_files = [
        str(path)
        for _, path in sorted(template_paths.items())
        if not path.exists()
    ]

    template_rows: list[dict[str, Any]] = []
    template_summary = None
    if template_paths and not missing_template_files:
        template_summary = summarize_audit_templates(path for _, path in sorted(template_paths.items()))
        template_rows = _load_template_rows(template_paths)

    required_probe_types = _required_audit_probe_types(manifest.get("profile_id"))
    coverage = _build_template_coverage(
        template_rows,
        expected_domains=expected_domains,
        required_probe_types=required_probe_types,
    )
    task_package = _task_package_summary(
        task_manifest_path=task_manifest_path,
        audit_plan=audit_plan,
        manifest_dir=manifest_file.parent,
        expected_template_paths=tuple(path.resolve() for _, path in sorted(template_paths.items())),
        expected_num_template_items=len(template_rows),
    )

    checks = {
        "template_files_present": _check(
            bool(template_paths) and not missing_template_files,
            {
                "num_template_files": len(template_paths),
                "missing_template_files": missing_template_files,
            },
            "all declared audit template files exist",
        ),
        "template_summary_present": _check(
            template_summary is not None,
            template_summary is not None,
            True,
        ),
        "template_ready_for_double_annotation": _check(
            bool(template_summary and template_summary.get("ready_for_double_annotation")),
            template_summary.get("ready_for_double_annotation") if template_summary is not None else None,
            True,
        ),
        "all_expected_domains_covered": _check(
            tuple(coverage["domains"]["observed"]) == expected_domains,
            coverage["domains"]["observed"],
            expected_domains,
        ),
        "per_domain_min_queries_at_least_100": _check(
            coverage["domains"]["min_query_count"] >= 100,
            coverage["domains"]["min_query_count"],
            ">= 100",
        ),
        "all_required_probe_types_covered": _check(
            set(coverage["probe_types"]["observed"]) == set(required_probe_types),
            coverage["probe_types"]["observed"],
            required_probe_types,
        ),
        "all_domains_cover_required_probe_types": _check(
            coverage["domains"]["all_domains_cover_required_probe_types"],
            coverage["domains"]["per_domain_probe_type_counts"],
            "every domain covers all required audit probe types",
        ),
        "all_domains_cover_easy_medium_hard": _check(
            coverage["domains"]["all_domains_cover_required_difficulty_levels"],
            coverage["domains"]["per_domain_difficulty_levels"],
            REQUIRED_DIFFICULTY_LEVELS,
        ),
        "all_audit_checks_covered": _check(
            set(coverage["applicable_checks"]["observed"]) == set(AUDIT_CHECK_FIELDS),
            coverage["applicable_checks"]["observed"],
            AUDIT_CHECK_FIELDS,
        ),
        "all_domains_cover_all_audit_checks": _check(
            coverage["domains"]["all_domains_cover_required_checks"],
            coverage["domains"]["per_domain_applicable_checks"],
            AUDIT_CHECK_FIELDS,
        ),
        "counterfactual_axes_covered": _check(
            set(REQUIRED_COUNTERFACTUAL_AXES).issubset(set(coverage["counterfactual_axes"]["observed"])),
            coverage["counterfactual_axes"]["observed"],
            REQUIRED_COUNTERFACTUAL_AXES,
        ),
        "all_domains_cover_counterfactual_axes": _check(
            coverage["domains"]["all_domains_cover_required_counterfactual_axes"],
            coverage["domains"]["per_domain_counterfactual_axes"],
            REQUIRED_COUNTERFACTUAL_AXES,
        ),
        "governance_state_contracts_present": _check(
            coverage["state_contracts"]["num_rows_with_governance_rules"] > 0,
            coverage["state_contracts"]["num_rows_with_governance_rules"],
            "> 0",
        ),
        "all_domains_include_governance_state_contracts": _check(
            coverage["domains"]["all_domains_include_governance_rules"],
            coverage["domains"]["per_domain_governance_rule_rows"],
            "every domain has rows with governance rule summaries",
        ),
        "task_package_present": _check(
            task_package["present"],
            task_package["task_manifest_path"],
            "task manifest exists",
        ),
        "task_package_progress_errors_empty": _check(
            not task_package["progress_errors"],
            task_package["progress_errors"],
            [],
        ),
        "task_package_template_files_match": _check(
            task_package["template_files_match"],
            task_package["resolved_template_files"],
            [path.as_posix() for path in tuple(path.resolve() for _, path in sorted(template_paths.items()))],
        ),
        "task_package_template_item_count_match": _check(
            task_package["num_template_items_match"],
            {
                "task_manifest": task_package["num_template_items"],
                "template_rows": len(template_rows),
            },
            "equal",
        ),
        "task_package_expected_annotations_match": _check(
            task_package["expected_annotations_match"],
            {
                "task_manifest": task_package["expected_annotations"],
                "template_rows_x_num_annotators": len(template_rows) * task_package["num_annotators"],
            },
            "equal",
        ),
        "task_package_declares_two_annotators": _check(
            task_package["num_annotators"] >= 2,
            task_package["num_annotators"],
            ">= 2",
        ),
        "task_package_task_files_match_annotators": _check(
            task_package["task_files_match_annotators"],
            {
                "annotator_ids": task_package["annotator_ids"],
                "task_file_annotators": task_package["task_file_annotators"],
            },
            "equal",
        ),
        "task_package_identity_digests_complete": _check(
            task_package["identity_digests_complete"],
            {
                "task_identity_digest_keys": task_package["task_identity_digest_keys"],
                "annotator_ids": task_package["annotator_ids"],
            },
            "one digest per annotator",
        ),
    }

    return {
        "schema_version": HUMAN_AUDIT_SUBSET_AUDIT_SCHEMA_VERSION,
        "benchmark_id": manifest.get("benchmark_id"),
        "source_type": "release_manifest",
        "manifest_path": str(manifest_file),
        "task_manifest_path": task_package["task_manifest_path"],
        "status": "passed" if all(item["passed"] for item in checks.values()) else "failed",
        "summary": {
            "num_domains": len(coverage["domains"]["observed"]),
            "num_template_rows": len(template_rows),
            "num_cases": coverage["domains"]["num_cases"],
            "num_queries": coverage["domains"]["num_queries"],
            "num_counterfactual_rows": coverage["counterfactual_axes"]["num_rows"],
            "num_rows_with_governance_rules": coverage["state_contracts"]["num_rows_with_governance_rules"],
            "num_expected_annotations": task_package["expected_annotations"],
            "num_task_package_annotators": task_package["num_annotators"],
            "num_progress_errors": len(task_package["progress_errors"]),
        },
        "checks": checks,
        "template_summary": template_summary,
        "coverage": coverage,
        "task_package": task_package,
        "missing_template_files": missing_template_files,
    }


def write_human_audit_subset_audit(
    output: str | Path,
    *,
    manifest_path: str | Path,
    task_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    report = audit_human_audit_subset_release(
        manifest_path,
        task_manifest_path=task_manifest_path,
    )
    report = localize_report_contract(
        report,
        output_path=output,
        project_root_hints=(manifest_path, task_manifest_path),
    )
    write_json(output, report)
    return report


def _required_audit_probe_types(profile_id: Any) -> tuple[str, ...]:
    if str(profile_id or "") == "main-v1-strict":
        return STRICT_REQUIRED_AUDIT_PROBE_TYPES
    return REQUIRED_AUDIT_PROBE_TYPES


def _template_paths_from_audit_plan(audit_plan: Any, manifest_dir: Path) -> dict[str, Path]:
    if not isinstance(audit_plan, dict):
        return {}
    template_files = audit_plan.get("audit_template_files")
    if not isinstance(template_files, dict):
        return {}
    resolved: dict[str, Path] = {}
    for domain, raw_path in sorted(template_files.items()):
        if not isinstance(raw_path, str) or not raw_path:
            continue
        resolved[str(domain)] = _resolve_path(raw_path, manifest_dir)
    return resolved


def _load_template_rows(template_paths: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, path in sorted(template_paths.items()):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
    return rows


def _build_template_coverage(
    rows: list[dict[str, Any]],
    *,
    expected_domains: tuple[str, ...],
    required_probe_types: tuple[str, ...],
) -> dict[str, Any]:
    domain_cases: dict[str, set[str]] = defaultdict(set)
    domain_queries: dict[str, int] = defaultdict(int)
    domain_probe_types: dict[str, set[str]] = defaultdict(set)
    domain_difficulties: dict[str, set[str]] = defaultdict(set)
    domain_checks: dict[str, set[str]] = defaultdict(set)
    domain_axes: dict[str, set[str]] = defaultdict(set)
    domain_governance_rows: dict[str, int] = defaultdict(int)
    probe_counter: Counter[str] = Counter()
    check_counter: Counter[str] = Counter()
    axis_counter: Counter[str] = Counter()
    num_rows_with_governance_rules = 0

    for row in rows:
        domain = str(row.get("domain", "unknown"))
        case_id = str(row.get("case_id", ""))
        probe_type = str(row.get("probe_type", "unknown"))
        difficulty_level = str(row.get("difficulty_level", "unknown"))
        domain_cases[domain].add(case_id)
        domain_queries[domain] += 1
        domain_probe_types[domain].add(probe_type)
        domain_difficulties[domain].add(difficulty_level)
        probe_counter[probe_type] += 1

        applicable_checks = _string_tuple(row.get("applicable_checks"))
        domain_checks[domain].update(applicable_checks)
        for field in applicable_checks:
            check_counter[field] += 1

        raw_axis = row.get("counterfactual_axis")
        if raw_axis:
            axis = str(raw_axis)
            domain_axes[domain].add(axis)
            axis_counter[axis] += 1

        state_contract_summary = row.get("state_contract_summary")
        governance_rules = ()
        if isinstance(state_contract_summary, dict):
            governance_rules = _string_tuple(state_contract_summary.get("required_governance_rules"))
        if governance_rules:
            num_rows_with_governance_rules += 1
            domain_governance_rows[domain] += 1

    observed_domains = tuple(sorted(domain_queries))
    per_domain_probe_type_counts = {
        domain: len(domain_probe_types.get(domain, set()))
        for domain in expected_domains
    }
    per_domain_difficulty_levels = {
        domain: tuple(sorted(domain_difficulties.get(domain, set())))
        for domain in expected_domains
    }
    per_domain_applicable_checks = {
        domain: tuple(sorted(domain_checks.get(domain, set())))
        for domain in expected_domains
    }
    per_domain_counterfactual_axes = {
        domain: tuple(sorted(domain_axes.get(domain, set())))
        for domain in expected_domains
    }
    per_domain_governance_rule_rows = {
        domain: domain_governance_rows.get(domain, 0)
        for domain in expected_domains
    }

    return {
        "domains": {
            "observed": observed_domains,
            "num_cases": sum(len(case_ids) for case_ids in domain_cases.values()),
            "num_queries": sum(domain_queries.values()),
            "min_query_count": min(domain_queries.values()) if domain_queries else 0,
            "per_domain_query_counts": {domain: domain_queries.get(domain, 0) for domain in expected_domains},
            "per_domain_case_counts": {domain: len(domain_cases.get(domain, set())) for domain in expected_domains},
            "per_domain_probe_type_counts": per_domain_probe_type_counts,
            "per_domain_difficulty_levels": per_domain_difficulty_levels,
            "per_domain_applicable_checks": per_domain_applicable_checks,
            "per_domain_counterfactual_axes": per_domain_counterfactual_axes,
            "per_domain_governance_rule_rows": per_domain_governance_rule_rows,
            "all_domains_cover_required_probe_types": all(
                set(required_probe_types).issubset(domain_probe_types.get(domain, set()))
                for domain in expected_domains
            ),
            "all_domains_cover_required_difficulty_levels": all(
                set(REQUIRED_DIFFICULTY_LEVELS) == domain_difficulties.get(domain, set())
                for domain in expected_domains
            ),
            "all_domains_cover_required_checks": all(
                set(AUDIT_CHECK_FIELDS) == domain_checks.get(domain, set())
                for domain in expected_domains
            ),
            "all_domains_cover_required_counterfactual_axes": all(
                set(REQUIRED_COUNTERFACTUAL_AXES).issubset(domain_axes.get(domain, set()))
                for domain in expected_domains
            ),
            "all_domains_include_governance_rules": all(
                domain_governance_rows.get(domain, 0) > 0
                for domain in expected_domains
            ),
        },
        "probe_types": {
            "observed": tuple(sorted(probe_counter)),
            "counts": {name: probe_counter[name] for name in sorted(probe_counter)},
        },
        "applicable_checks": {
            "observed": tuple(sorted(check_counter)),
            "counts": {name: check_counter[name] for name in sorted(check_counter)},
        },
        "counterfactual_axes": {
            "observed": tuple(sorted(axis_counter)),
            "counts": {name: axis_counter[name] for name in sorted(axis_counter)},
            "num_rows": sum(axis_counter.values()),
        },
        "state_contracts": {
            "num_rows_with_governance_rules": num_rows_with_governance_rules,
        },
    }


def _task_package_summary(
    *,
    task_manifest_path: str | Path | None,
    audit_plan: Any,
    manifest_dir: Path,
    expected_template_paths: tuple[Path, ...],
    expected_num_template_items: int,
) -> dict[str, Any]:
    resolved_path = _resolve_task_manifest_path(task_manifest_path, audit_plan, manifest_dir)
    summary = {
        "present": False,
        "task_manifest_path": str(resolved_path) if resolved_path is not None else None,
        "progress_errors": ["task manifest not provided"],
        "num_template_items": 0,
        "expected_annotations": 0,
        "num_annotators": 0,
        "annotator_ids": [],
        "task_file_annotators": [],
        "task_identity_digest_keys": [],
        "resolved_template_files": [],
        "template_files_match": False,
        "num_template_items_match": False,
        "expected_annotations_match": False,
        "task_files_match_annotators": False,
        "identity_digests_complete": False,
    }
    if resolved_path is None:
        return summary
    if not resolved_path.exists():
        summary["progress_errors"] = [f"task manifest does not exist: {resolved_path}"]
        return summary

    progress = summarize_human_audit_progress(resolved_path)
    payload = read_json(resolved_path)
    task_files = payload.get("task_files", {}) if isinstance(payload, dict) else {}
    task_identity_digests = payload.get("task_identity_digests", {}) if isinstance(payload, dict) else {}
    annotator_ids = tuple(sorted(_string_tuple(payload.get("annotator_ids"))))
    task_file_annotators = tuple(sorted(str(key) for key in task_files)) if isinstance(task_files, dict) else ()
    digest_keys = tuple(sorted(str(key) for key in task_identity_digests)) if isinstance(task_identity_digests, dict) else ()
    raw_template_files = payload.get("template_files", []) if isinstance(payload, dict) else []
    resolved_template_files = tuple(
        sorted(
            _resolve_path(str(raw_path), resolved_path.parent).resolve().as_posix()
            for raw_path in raw_template_files
            if isinstance(raw_path, str) and raw_path
        )
    )
    expected_template_refs = tuple(sorted(path.resolve().as_posix() for path in expected_template_paths))
    num_annotators = _as_int(payload.get("num_annotators")) if isinstance(payload, dict) else 0
    num_template_items = _as_int(payload.get("num_template_items")) if isinstance(payload, dict) else 0
    expected_annotations = _as_int(payload.get("expected_annotations")) if isinstance(payload, dict) else 0

    return {
        "present": True,
        "task_manifest_path": str(resolved_path),
        "progress": progress,
        "progress_errors": list(progress.get("errors", [])),
        "num_template_items": num_template_items,
        "expected_annotations": expected_annotations,
        "num_annotators": num_annotators,
        "annotator_ids": list(annotator_ids),
        "task_file_annotators": list(task_file_annotators),
        "task_identity_digest_keys": list(digest_keys),
        "resolved_template_files": list(resolved_template_files),
        "template_files_match": resolved_template_files == expected_template_refs,
        "num_template_items_match": num_template_items == expected_num_template_items,
        "expected_annotations_match": expected_annotations == expected_num_template_items * num_annotators,
        "task_files_match_annotators": annotator_ids == task_file_annotators,
        "identity_digests_complete": digest_keys == annotator_ids,
    }


def _resolve_task_manifest_path(
    task_manifest_path: str | Path | None,
    audit_plan: Any,
    manifest_dir: Path,
) -> Path | None:
    if task_manifest_path is not None:
        return _resolve_path(str(task_manifest_path), manifest_dir)
    if not isinstance(audit_plan, dict):
        return None
    raw_path = audit_plan.get("audit_task_manifest_file")
    if isinstance(raw_path, str) and raw_path:
        return _resolve_path(raw_path, manifest_dir)
    return None


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    return base_dir / path


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if item is not None)
    return ()


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _check(passed: bool, actual: Any, expected: Any) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }
