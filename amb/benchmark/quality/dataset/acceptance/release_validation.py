"""Validation for AutoMemoryBench release manifests and shard artifacts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib
import json
from typing import Any, Iterable

from amb.benchmark.release.splits import RELEASE_SPLITS
from amb.benchmark.quality.artifact_contract import localize_report_contract
from amb.benchmark.quality.audit import audit_benchmark
from amb.benchmark.quality.human_audit import verify_manifest_human_audit
from amb.benchmark.quality.validation import ValidationResult, validate_benchmark
from amb.benchmark.schemas.io import load_benchmark, read_json, write_json
from amb.benchmark.schemas.models import Benchmark, Case


def validate_release_artifacts(manifest_path: str | Path) -> dict[str, Any]:
    """Validate a release manifest and the benchmark artifacts it references."""

    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    if manifest.get("package_type") == "private_leaderboard_package":
        return _validate_private_leaderboard_package(manifest_file, manifest)
    manifest_dir = manifest_file.parent
    errors: list[str] = []
    warnings: list[str] = []
    split_reports = _empty_reports()
    seen_case_ids: set[str] = set()
    group_to_split: dict[str, str] = {}
    group_case_counts: dict[str, int] = defaultdict(int)
    audit_template_expected: dict[str, dict[str, dict[str, Any]]] = {}
    withheld_splits = set(manifest.get("withheld_splits", {}))
    artifact_checks = _validate_artifacts(manifest, manifest_dir, errors)
    _check_public_package_hidden_policy(manifest, errors)

    split_files = manifest.get("split_files", {})
    for split in RELEASE_SPLITS:
        entries = _split_entries(split_files.get(split))
        if not entries:
            if split not in withheld_splits:
                warnings.append(f"{split} has no benchmark artifacts")
            continue
        for label, raw_path in entries:
            path = _resolve_path(raw_path, manifest_dir)
            if not path.exists():
                errors.append(f"{split}/{label} artifact does not exist: {raw_path}")
                continue
            try:
                benchmark = load_benchmark(path)
            except Exception as exc:  # noqa: BLE001 - include parse errors in validation report
                errors.append(f"{split}/{label} could not be loaded: {exc}")
                continue
            if label != "benchmark":
                mismatched_domains = sorted({case.domain for case in benchmark.cases if case.domain != label})
                if mismatched_domains:
                    errors.append(f"{split}/{label} shard contains mismatched domains: {mismatched_domains}")
            _validate_benchmark_artifact(split, label, path, benchmark, errors, warnings)
            _accumulate_report(split_reports[split], audit_benchmark(benchmark))
            _check_case_and_group_integrity(
                split,
                benchmark.cases,
                seen_case_ids=seen_case_ids,
                group_to_split=group_to_split,
                group_case_counts=group_case_counts,
                errors=errors,
            )
            if split == "audit_subset":
                audit_template_expected.setdefault(label, {}).update(_audit_template_records(benchmark.cases))

    for split, report in split_reports.items():
        _finalize_report(report, require_multiple_domains=bool(report["num_cases"]))
        if split in withheld_splits:
            _check_withheld_split(split, manifest, errors)
            continue
        _check_manifest_split_report(split, report, manifest.get("split_reports", {}).get(split), errors)
        _check_release_plan(split, report, manifest.get("release_plan", {}).get("split_reports", {}).get(split), errors)

    _check_expected_generation_summary(manifest.get("expected_generation_summary"), split_reports, errors, manifest, withheld_splits)
    _check_group_assignments(manifest.get("group_assignments"), group_to_split, errors, withheld_splits)
    _check_hidden_enrichment_summary(manifest.get("hidden_enrichment_summary"), manifest.get("split_reports", {}), errors)
    _check_audit_templates(manifest, manifest_dir, audit_template_expected, errors)
    _check_completed_human_audit(manifest, manifest_file, errors)

    return {
        "manifest_path": str(manifest_file),
        "benchmark_id": manifest.get("benchmark_id"),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "artifact_checks": artifact_checks,
        "split_reports": split_reports,
        "withheld_splits": {
            split: manifest.get("withheld_splits", {}).get(split, {})
            for split in sorted(withheld_splits)
        },
        "num_case_ids": len(seen_case_ids),
        "num_case_groups": len(group_to_split),
        "group_case_count_distribution": dict(sorted(_count_values(group_case_counts.values()).items())),
    }


def write_release_validation(manifest_path: str | Path, output: str | Path) -> dict[str, Any]:
    report = validate_release_artifacts(manifest_path)
    report = localize_report_contract(
        report,
        output_path=output,
        project_root_hints=(manifest_path,),
    )
    write_json(output, report)
    return report


def _validate_private_leaderboard_package(manifest_file: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest_dir = manifest_file.parent
    errors: list[str] = []
    warnings: list[str] = []
    artifact_checks = _validate_artifacts(manifest, manifest_dir, errors)
    split_files = manifest.get("split_files", {})
    hidden_path_value = split_files.get("hidden_test")
    if not hidden_path_value:
        errors.append("private leaderboard package must include split_files.hidden_test")
        hidden_report = {}
    else:
        hidden_path = _resolve_path(str(hidden_path_value), manifest_dir)
        benchmark = load_benchmark(hidden_path)
        audit = audit_benchmark(benchmark)
        hidden_report = audit
        _validate_benchmark_artifact("hidden_test", "benchmark", hidden_path, benchmark, errors, warnings)
        manifest_report = manifest.get("split_reports", {}).get("hidden_test", {})
        _check_manifest_split_report("hidden_test", _private_like_split_report(audit), manifest_report, errors)
        hidden_cases = len({str(case.difficulty.values.get("counterfactual_group_id", case.case_id)) for case in benchmark.cases})
        refresh = manifest.get("quarterly_hidden_refresh", {})
        if hidden_cases != int(refresh.get("num_hidden_scenarios", -1)):
            errors.append(
                "private leaderboard package hidden scenario count mismatch: "
                f"manifest={refresh.get('num_hidden_scenarios')} actual={hidden_cases}"
            )
    return {
        "manifest_path": str(manifest_file),
        "benchmark_id": manifest.get("benchmark_id"),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "artifact_checks": artifact_checks,
        "split_reports": {"hidden_test": _private_like_split_report(hidden_report) if hidden_report else {}},
        "withheld_splits": {},
        "num_case_ids": int(manifest.get("split_reports", {}).get("hidden_test", {}).get("num_cases", 0)),
        "num_case_groups": int(manifest.get("quarterly_hidden_refresh", {}).get("num_hidden_scenarios", 0)),
        "group_case_count_distribution": {},
    }


def _validate_benchmark_artifact(
    split: str,
    label: str,
    path: Path,
    benchmark: Benchmark,
    errors: list[str],
    warnings: list[str],
) -> None:
    result = validate_benchmark(benchmark)
    for error in result.errors:
        errors.append(f"{split}/{label}/{path.name}: {error}")
    for warning in result.warnings:
        warnings.append(f"{split}/{label}/{path.name}: {warning}")


def _check_case_and_group_integrity(
    split: str,
    cases: tuple[Case, ...],
    *,
    seen_case_ids: set[str],
    group_to_split: dict[str, str],
    group_case_counts: dict[str, int],
    errors: list[str],
) -> None:
    for case in cases:
        if case.case_id in seen_case_ids:
            errors.append(f"duplicate case_id across release artifacts: {case.case_id}")
        seen_case_ids.add(case.case_id)
        group_id = str(case.difficulty.values.get("counterfactual_group_id") or case.case_id)
        previous_split = group_to_split.setdefault(group_id, split)
        if previous_split != split:
            errors.append(f"case group {group_id} appears in both {previous_split} and {split}")
        group_case_counts[group_id] += 1


def _validate_artifacts(manifest: dict[str, Any], manifest_dir: Path, errors: list[str]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for key, artifact in _flatten_artifacts(manifest.get("split_artifacts", {})):
        checks[key] = _artifact_check(key, artifact, manifest_dir, errors)
    for key, artifact in _flatten_artifacts(manifest.get("audit_template_artifacts", {})):
        checks[f"audit_template/{key}"] = _artifact_check(f"audit_template/{key}", artifact, manifest_dir, errors)
    return checks


def _check_public_package_hidden_policy(manifest: dict[str, Any], errors: list[str]) -> None:
    if manifest.get("package_type") != "public_release_export":
        return
    hidden_files = manifest.get("split_files", {}).get("hidden_test")
    hidden_artifacts = manifest.get("split_artifacts", {}).get("hidden_test")
    if hidden_files:
        errors.append("public release export must not include hidden_test split_files")
    if hidden_artifacts:
        errors.append("public release export must not include hidden_test split_artifacts")
    hidden = manifest.get("withheld_splits", {}).get("hidden_test")
    if not hidden:
        errors.append("public release export must declare withheld_splits.hidden_test")
        return
    if hidden.get("artifact_status") != "withheld":
        errors.append("withheld_splits.hidden_test.artifact_status must be 'withheld'")
    if hidden.get("visibility") != "private_leaderboard_only":
        errors.append("withheld_splits.hidden_test.visibility must be private_leaderboard_only")


def _artifact_check(name: str, artifact: dict[str, Any], manifest_dir: Path, errors: list[str]) -> dict[str, Any]:
    raw_path = artifact.get("path")
    if not raw_path:
        errors.append(f"{name} artifact is missing path")
        return {"exists": False, "path": raw_path}
    path = _resolve_path(str(raw_path), manifest_dir)
    if not path.exists():
        errors.append(f"{name} artifact does not exist: {raw_path}")
        return {"exists": False, "path": str(path)}
    size_bytes = path.stat().st_size
    digest = _sha256(path)
    if "size_bytes" in artifact and int(artifact["size_bytes"]) != size_bytes:
        errors.append(f"{name} size mismatch: manifest={artifact['size_bytes']} actual={size_bytes}")
    if "sha256" in artifact and str(artifact["sha256"]) != digest:
        errors.append(f"{name} sha256 mismatch")
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": size_bytes,
        "sha256": digest,
    }


def _check_manifest_split_report(
    split: str,
    actual: dict[str, Any],
    expected: dict[str, Any] | None,
    errors: list[str],
) -> None:
    if expected is None:
        errors.append(f"manifest is missing split_reports.{split}")
        return
    for field in ("num_cases", "num_queries", "num_memories", "num_events", "num_state_contracts"):
        if int(expected.get(field, -1)) != int(actual[field]):
            errors.append(f"{split} {field} mismatch: manifest={expected.get(field)} actual={actual[field]}")
    for field in ("domains", "probe_types", "counterfactual_edits", "renderer_coverage"):
        if field not in expected and field in {"counterfactual_edits", "renderer_coverage"}:
            continue
        if dict(expected.get(field, {})) != dict(actual[field]):
            errors.append(f"{split} {field} mismatch")
    for field in ("quality_gates_passed", "construction_gates_passed", "data_quality_gates_passed"):
        if field in expected and bool(expected[field]) != bool(actual[field]):
            errors.append(f"{split} {field} mismatch: manifest={expected[field]} actual={actual[field]}")


def _check_release_plan(
    split: str,
    actual: dict[str, Any],
    planned: dict[str, Any] | None,
    errors: list[str],
) -> None:
    if not planned:
        return
    planned_fields = {
        "case_variants": "num_cases",
        "queries": "num_queries",
        "memories": "num_memories",
        "events": "num_events",
    }
    for plan_field, actual_field in planned_fields.items():
        if plan_field in planned and int(planned[plan_field]) != int(actual[actual_field]):
            errors.append(
                f"{split} does not match release_plan.{plan_field}: "
                f"planned={planned[plan_field]} actual={actual[actual_field]}"
            )


def _check_expected_generation_summary(
    summary: dict[str, Any] | None,
    split_reports: dict[str, dict[str, Any]],
    errors: list[str],
    manifest: dict[str, Any],
    withheld_splits: set[str],
) -> None:
    if not summary:
        return
    withheld_reports = {
        split: manifest.get("split_reports", {}).get(split, {})
        for split in withheld_splits
    }

    def total(field: str) -> int:
        visible = sum(report[field] for split, report in split_reports.items() if split not in withheld_splits)
        withheld = sum(int(report.get(field, 0)) for report in withheld_reports.values())
        return visible + withheld

    totals = {
        "num_cases": total("num_cases"),
        "total_queries_with_counterfactuals": total("num_queries"),
        "total_memories_with_counterfactuals": total("num_memories"),
        "total_events_with_counterfactuals": total("num_events"),
        "total_state_contracts_with_counterfactuals": total("num_state_contracts"),
    }
    for key, actual in totals.items():
        if key in summary and int(summary[key]) != int(actual):
            errors.append(f"expected_generation_summary.{key} mismatch: expected={summary[key]} actual={actual}")


def _check_group_assignments(
    group_assignments: dict[str, Any] | None,
    group_to_split: dict[str, str],
    errors: list[str],
    withheld_splits: set[str],
) -> None:
    if not group_assignments:
        return
    assigned: dict[str, str] = {}
    for split, value in group_assignments.items():
        if split in withheld_splits:
            continue
        if isinstance(value, dict):
            group_ids = [group_id for groups in value.values() for group_id in groups]
        elif isinstance(value, list):
            group_ids = value
        else:
            errors.append(f"group_assignments.{split} must be a list or domain mapping")
            continue
        for group_id in group_ids:
            group_id = str(group_id)
            previous = assigned.setdefault(group_id, str(split))
            if previous != str(split):
                errors.append(f"group_assignments places {group_id} in both {previous} and {split}")
    if set(assigned) != set(group_to_split):
        missing = sorted(set(group_to_split) - set(assigned))
        extra = sorted(set(assigned) - set(group_to_split))
        errors.append(f"group_assignments mismatch: missing={missing[:10]} extra={extra[:10]}")
    for group_id, split in assigned.items():
        actual_split = group_to_split.get(group_id)
        if actual_split and actual_split != split:
            errors.append(f"group_assignments places {group_id} in {split}, artifact has {actual_split}")


def _check_withheld_split(split: str, manifest: dict[str, Any], errors: list[str]) -> None:
    split_report = manifest.get("split_reports", {}).get(split)
    if not isinstance(split_report, dict):
        errors.append(f"withheld split {split} is missing split_reports entry")
        return
    if int(split_report.get("num_cases", 0)) <= 0:
        errors.append(f"withheld split {split} must retain non-zero split report counts")
    withheld_report = manifest.get("withheld_splits", {}).get(split, {}).get("split_report")
    if withheld_report and withheld_report != split_report:
        errors.append(f"withheld_splits.{split}.split_report does not match split_reports.{split}")
    _check_release_plan(split, split_report, manifest.get("release_plan", {}).get("split_reports", {}).get(split), errors)


def _check_hidden_enrichment_summary(
    summary: dict[str, Any] | None,
    split_reports: dict[str, Any],
    errors: list[str],
) -> None:
    if not summary:
        return
    split_comparison = summary.get("split_comparison", {})
    for split in ("hidden_test", "public_test", "public_dev"):
        actual = _split_enrichment_metrics(split_reports.get(split, {}))
        expected = split_comparison.get(split)
        if expected is None:
            errors.append(f"hidden_enrichment_summary.split_comparison.{split} is missing")
            continue
        if expected != actual:
            errors.append(f"hidden_enrichment_summary.split_comparison.{split} mismatch")
    checks = summary.get("checks", {})
    hidden = split_reports.get("hidden_test", {})
    public_test = split_reports.get("public_test", {})
    expected_checks = {
        "counterfactual_share_gt_public_test": _share(hidden, "counterfactual_cases", "num_cases")
        > _share(public_test, "counterfactual_cases", "num_cases"),
        "governance_share_gt_public_test": _share(hidden, "governance_cases", "num_cases")
        > _share(public_test, "governance_cases", "num_cases"),
        "cross_subject_share_gt_public_test": _share(hidden, "cross_subject_cases", "num_cases")
        > _share(public_test, "cross_subject_cases", "num_cases"),
    }
    if checks != expected_checks:
        errors.append("hidden_enrichment_summary.checks mismatch")


def _check_audit_templates(
    manifest: dict[str, Any],
    manifest_dir: Path,
    expected_by_label: dict[str, dict[str, dict[str, Any]]],
    errors: list[str],
) -> None:
    audit_plan = manifest.get("audit_plan", {})
    template_files = audit_plan.get("audit_template_files")
    if isinstance(template_files, dict):
        actual_labels = set(template_files)
        expected_labels = set(expected_by_label)
        if actual_labels != expected_labels:
            errors.append(
                "audit_template_files labels mismatch: "
                f"missing={sorted(expected_labels - actual_labels)} extra={sorted(actual_labels - expected_labels)}"
            )
        for label, expected_records in expected_by_label.items():
            raw_path = template_files.get(label)
            if raw_path is None:
                continue
            _check_audit_template_file(label, _resolve_path(str(raw_path), manifest_dir), expected_records, errors)
        return

    raw_path = audit_plan.get("audit_template_file")
    if raw_path:
        merged: dict[str, dict[str, Any]] = {}
        for records in expected_by_label.values():
            merged.update(records)
        _check_audit_template_file("audit_template_file", _resolve_path(str(raw_path), manifest_dir), merged, errors)


def _check_completed_human_audit(manifest: dict[str, Any], manifest_file: Path, errors: list[str]) -> None:
    if manifest.get("audit_plan", {}).get("human_audit_status") != "completed":
        return
    report = verify_manifest_human_audit(manifest_file)
    for error in report["errors"]:
        errors.append(f"completed human audit: {error}")


def _audit_template_records(cases: tuple[Case, ...]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for case in cases:
        for query in case.queries:
            records[query.query_id] = {
                "case_id": case.case_id,
                "query_id": query.query_id,
                "domain": case.domain,
                "probe_type": query.probe_type,
                "state_contract_id": query.state_contract_id,
                "gold_memory_ids": list(query.gold_memory_ids),
                "forbidden_memory_ids": list(query.forbidden_memory_ids),
            }
    return records


def _check_audit_template_file(
    label: str,
    path: Path,
    expected_records: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    if not path.exists():
        errors.append(f"audit template {label} does not exist: {path}")
        return
    actual_records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors.append(f"audit template {label}:{line_number} is not valid JSONL: {exc}")
                continue
            query_id = str(record.get("query_id", ""))
            if not query_id:
                errors.append(f"audit template {label}:{line_number} is missing query_id")
                continue
            if query_id in actual_records:
                errors.append(f"audit template {label} duplicates query_id {query_id}")
            actual_records[query_id] = {
                "case_id": record.get("case_id"),
                "query_id": query_id,
                "domain": record.get("domain"),
                "probe_type": record.get("probe_type"),
                "state_contract_id": record.get("state_contract_id"),
                "gold_memory_ids": list(record.get("gold_memory_ids", [])),
                "forbidden_memory_ids": list(record.get("forbidden_memory_ids", [])),
            }

    if set(actual_records) != set(expected_records):
        missing = sorted(set(expected_records) - set(actual_records))
        extra = sorted(set(actual_records) - set(expected_records))
        errors.append(f"audit template {label} query set mismatch: missing={missing[:10]} extra={extra[:10]}")
    for query_id, expected in expected_records.items():
        actual = actual_records.get(query_id)
        if actual is not None and actual != expected:
            errors.append(f"audit template {label} record mismatch for {query_id}")


def _split_entries(value: Any) -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [("benchmark", value)]
    if isinstance(value, dict):
        return [(str(label), str(path)) for label, path in sorted(value.items())]
    return []


def _flatten_artifacts(value: Any, prefix: str = "") -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict) and {"path", "sha256", "size_bytes"} <= set(value):
        yield prefix.strip("/") or "artifact", value
        return
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            yield from _flatten_artifacts(child, child_prefix)


def _empty_reports() -> dict[str, dict[str, Any]]:
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
        "memory_types": {},
        "privacy_levels": {},
        "event_types": {},
        "counterfactual_edits": {},
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
        },
        "quality_gates": {},
        "construction_gates": {},
        "data_quality_gates": {},
        "quality_gates_passed": False,
        "construction_gates_passed": False,
        "data_quality_gates_passed": False,
    }


def _accumulate_report(report: dict[str, Any], audit: dict[str, Any]) -> None:
    report["num_cases"] += audit["num_cases"]
    report["num_queries"] += audit["num_queries"]
    report["num_memories"] += audit["num_memories"]
    report["num_events"] += audit["num_events"]
    report["num_state_contracts"] += audit["num_state_contracts"]
    for field in ("domains", "task_types", "probe_types", "memory_types", "privacy_levels", "event_types", "counterfactual_edits", "coverage"):
        _merge_counts(report[field], audit[field])
    _merge_renderer_coverage(report["renderer_coverage"], audit["renderer_coverage"])
    _merge_gate_results(report["construction_gates"], audit["construction_gates"])
    _merge_gate_results(report["data_quality_gates"], audit["data_quality_gates"])


def _finalize_report(report: dict[str, Any], *, require_multiple_domains: bool) -> None:
    coverage = report["coverage"]
    report["quality_gates"] = {
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
    report["quality_gates_passed"] = all(report["quality_gates"].values())
    report["construction_gates_passed"] = all(report["construction_gates"].values()) if report["construction_gates"] else False
    report["data_quality_gates_passed"] = all(report["data_quality_gates"].values()) if report["data_quality_gates"] else False


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


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


def _private_like_split_report(audit: dict[str, Any]) -> dict[str, Any]:
    if not audit:
        return {}
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


def _count_values(values: Iterable[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for value in values:
        counts[int(value)] = counts.get(int(value), 0) + 1
    return counts


def _resolve_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return manifest_dir / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
