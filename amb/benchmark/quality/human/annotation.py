"""Human annotation agreement utilities for AutoMemoryBench audit subsets."""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable
import zipfile


AUDIT_CHECK_FIELDS = (
    "evidence_sufficient",
    "answer_unique",
    "governance_boundary_clear",
    "trace_natural",
    "scenario_memory_required",
    "counterfactual_target_state_only",
)

AUDIT_CHECK_DEFINITIONS = {
    "evidence_sufficient": "Whether the record contains enough evidence metadata for a qualified annotator to judge the item.",
    "answer_unique": "Whether the query has a unique answer, tool action, or refusal outcome under the stated state contract.",
    "governance_boundary_clear": "Whether authorization, privacy, deletion, and scope boundaries are clear enough to judge compliance.",
    "trace_natural": "Whether the scenario trace and query phrasing look natural rather than label-leaking or templated.",
    "scenario_memory_required": "Whether solving the query truly requires cross-turn or cross-session memory rather than prompt-local reasoning only.",
    "counterfactual_target_state_only": "Whether the counterfactual pair changes only the intended target state while keeping query intent comparable.",
}

COUNTERFACTUAL_ONLY_AUDIT_CHECKS = frozenset({"counterfactual_target_state_only"})
HUMAN_AUDIT_CLI_PREFIX = "PYTHONPATH=. python -m agent_memory_benchmark"

ANNOTATION_SHEET_COLUMNS = (
    "case_id",
    "query_id",
    "annotator_id",
    "annotation_status",
    "domain",
    "probe_type",
    "task_type",
    "difficulty_level",
    "memory_requirement",
    "memory_dependency",
    "counterfactual_group_id",
    "counterfactual_axis",
    "counterfactual_edit",
    "scoring_rule",
    "applicable_checks",
    "prompt",
    "expected_behavior_json",
    "gold_memory_ids_json",
    "forbidden_memory_ids_json",
    "gold_memory_evidence_json",
    "forbidden_memory_evidence_json",
    "relevant_events_json",
    "state_contract_id",
    "state_contract_summary_json",
    "source_template",
    "source_template_line",
    "evidence_sufficient",
    "answer_unique",
    "governance_boundary_clear",
    "trace_natural",
    "scenario_memory_required",
    "counterfactual_target_state_only",
    "notes",
)


@dataclass(frozen=True)
class AuditAnnotation:
    case_id: str
    query_id: str
    annotator_id: str
    checks: dict[str, bool | None]
    applicable_checks: frozenset[str]
    line_number: int

    @property
    def item_id(self) -> tuple[str, str]:
        return (self.case_id, self.query_id)


def summarize_human_audit_progress(
    task_manifest_path: str | Path,
    annotations_path: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize double-annotation task coverage without marking audit completion."""

    manifest_file = Path(task_manifest_path)
    with manifest_file.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        raise ValueError("human-audit task manifest must be a JSON object")

    errors: list[str] = []
    _verify_task_manifest_fingerprints(manifest, manifest_file.parent, errors)
    task_records = _task_records_from_task_manifest(manifest, manifest_file.parent, errors)
    expected_tasks = {(record["case_id"], record["query_id"], record["annotator_id"]) for record in task_records}
    expected_items = {(case_id, query_id) for case_id, query_id, _ in expected_tasks}
    expected_annotators = {annotator_id for _, _, annotator_id in expected_tasks}
    item_dimensions = _task_item_dimensions(task_records, errors)
    task_file_progress = _task_file_annotation_progress(manifest, manifest_file.parent, errors)
    task_file_started_keys = task_file_progress["started_annotation_keys"]
    task_file_completed_keys = task_file_progress["completed_annotation_keys"]
    manifest_expected = manifest.get("expected_annotations")
    parsed_expected = _optional_int(manifest_expected)
    if manifest_expected is not None and parsed_expected is None:
        errors.append("task manifest expected_annotations must be an integer")
    elif parsed_expected is not None and parsed_expected != len(expected_tasks):
        errors.append(
            f"task manifest expected_annotations mismatch: manifest={manifest_expected} actual={len(expected_tasks)}"
        )
    manifest_items = manifest.get("num_template_items")
    parsed_items = _optional_int(manifest_items)
    if manifest_items is not None and parsed_items is None:
        errors.append("task manifest num_template_items must be an integer")
    elif parsed_items is not None and parsed_items != len(expected_items):
        errors.append(f"task manifest num_template_items mismatch: manifest={manifest_items} actual={len(expected_items)}")

    annotation_errors: list[str] = []
    annotation_keys: set[tuple[str, str, str]] = set()
    duplicate_annotations: list[str] = []
    complete_check_annotations = 0
    annotations: tuple[AuditAnnotation, ...] = ()
    if annotations_path is not None:
        try:
            annotations = load_audit_annotations(annotations_path)
        except Exception as exc:  # noqa: BLE001 - surface user annotation errors
            annotation_errors.append(f"could not load annotations: {exc}")
        else:
            seen: set[tuple[str, str, str]] = set()
            for annotation in annotations:
                key = (*annotation.item_id, annotation.annotator_id)
                if key in seen:
                    duplicate_annotations.append(_format_task_key(key))
                seen.add(key)
                annotation_keys.add(key)
                if all(annotation.checks[field] is not None for field in annotation.applicable_checks):
                    complete_check_annotations += 1

    missing_annotations = sorted(expected_tasks - annotation_keys)
    extra_annotations = sorted(annotation_keys - expected_tasks)
    errors.extend(annotation_errors)
    if duplicate_annotations:
        errors.append(f"duplicate annotations: {duplicate_annotations[:10]}")
    if extra_annotations:
        errors.append(f"annotations contain non-task items: {[_format_task_key(key) for key in extra_annotations[:10]]}")

    annotation_source = "merged_annotations" if annotations_path is not None else "task_files"
    matched_annotation_keys = (expected_tasks & annotation_keys) if annotations_path is not None else task_file_completed_keys
    complete_annotation_keys = {
        (*annotation.item_id, annotation.annotator_id)
        for annotation in annotations
        if (*annotation.item_id, annotation.annotator_id) in expected_tasks
        and all(annotation.checks[field] is not None for field in annotation.applicable_checks)
    } if annotations_path is not None else task_file_completed_keys
    current_complete_check_annotations = complete_check_annotations if annotations_path is not None else len(task_file_completed_keys)
    current_missing_annotations = sorted(expected_tasks - matched_annotation_keys)
    annotator_progress = _annotator_progress(
        manifest,
        task_records,
        task_file_started_keys,
        matched_annotation_keys,
        complete_annotation_keys,
    )
    domain_progress = _dimension_progress(
        expected_tasks,
        task_file_started_keys,
        matched_annotation_keys,
        complete_annotation_keys,
        item_dimensions,
        dimension_key="domain",
    )
    probe_type_progress = _dimension_progress(
        expected_tasks,
        task_file_started_keys,
        matched_annotation_keys,
        complete_annotation_keys,
        item_dimensions,
        dimension_key="probe_type",
    )
    completion_fraction = len(matched_annotation_keys) / len(expected_tasks) if expected_tasks else 0.0
    ready_for_merge = not errors and task_file_completed_keys == expected_tasks
    ready_for_finalize = (
        annotations_path is not None
        and not errors
        and not missing_annotations
        and complete_check_annotations == len(expected_tasks)
    )
    recommended_next_commands = _human_audit_progress_next_commands(
        manifest_file,
        Path(annotations_path) if annotations_path is not None else None,
        ready_for_merge=ready_for_merge,
        ready_for_finalize=ready_for_finalize,
    )
    return {
        "human_audit_progress_schema_version": "amst-human-audit-progress-v1",
        "task_manifest_file": str(manifest_file),
        "annotations_file": str(annotations_path) if annotations_path is not None else None,
        "annotation_source": annotation_source,
        "status": "ready_for_finalize" if ready_for_finalize else "ready_for_merge" if ready_for_merge else "in_progress",
        "ready_for_merge": ready_for_merge,
        "ready_for_finalize": ready_for_finalize,
        "num_template_items": len(expected_items),
        "num_expected_annotations": len(expected_tasks),
        "num_expected_annotators": len(expected_annotators),
        "expected_annotator_ids": sorted(expected_annotators),
        "num_loaded_annotations": len(annotations),
        "num_matched_annotations": len(matched_annotation_keys),
        "num_complete_check_annotations": current_complete_check_annotations,
        "num_task_file_started_annotations": len(task_file_started_keys),
        "num_task_file_completed_annotations": len(task_file_completed_keys),
        "task_file_completion_fraction": len(task_file_completed_keys) / len(expected_tasks) if expected_tasks else 0.0,
        "completion_fraction": completion_fraction,
        "annotator_progress": annotator_progress,
        "domain_progress": domain_progress,
        "probe_type_progress": probe_type_progress,
        "missing_annotations": [_format_task_key(key) for key in current_missing_annotations[:50]],
        "num_missing_annotations": max(len(expected_tasks) - len(matched_annotation_keys), 0),
        "extra_annotations": [_format_task_key(key) for key in extra_annotations[:50]],
        "num_extra_annotations": len(extra_annotations),
        "duplicate_annotations": duplicate_annotations[:50],
        "num_duplicate_annotations": len(duplicate_annotations),
        "recommended_next_commands": recommended_next_commands,
        "errors": errors,
    }


def summarize_audit_templates(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Summarize generated audit annotation templates before human labeling."""

    summary = {
        "num_templates": 0,
        "num_records": 0,
        "num_cases": 0,
        "num_queries": 0,
        "domains": {},
        "probe_types": {},
        "checks": {field: 0 for field in AUDIT_CHECK_FIELDS},
        "applicable_checks": {field: 0 for field in AUDIT_CHECK_FIELDS},
        "missing_check_fields": [],
    }
    case_ids: set[str] = set()
    query_ids: set[str] = set()
    missing_check_fields: set[str] = set()

    for path in paths:
        source = Path(path)
        summary["num_templates"] += 1
        with source.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source}:{line_number}: invalid JSONL record") from exc
                summary["num_records"] += 1
                case_id = str(record.get("case_id", ""))
                query_id = str(record.get("query_id", ""))
                if case_id:
                    case_ids.add(case_id)
                if query_id:
                    query_ids.add(query_id)
                _increment(summary["domains"], str(record.get("domain", "unknown")))
                _increment(summary["probe_types"], str(record.get("probe_type", "unknown")))
                checks = record.get("checks", {})
                if not isinstance(checks, dict):
                    checks = {}
                applicable_checks = set(_resolve_applicable_checks(record, source, line_number))
                for field in AUDIT_CHECK_FIELDS:
                    if field in checks:
                        summary["checks"][field] += 1
                    else:
                        missing_check_fields.add(field)
                    if field in applicable_checks:
                        summary["applicable_checks"][field] += 1

    summary["num_cases"] = len(case_ids)
    summary["num_queries"] = len(query_ids)
    summary["missing_check_fields"] = sorted(missing_check_fields)
    summary["ready_for_double_annotation"] = (
        summary["num_records"] > 0
        and summary["num_records"] == summary["num_queries"]
        and not summary["missing_check_fields"]
    )
    return summary


def write_double_annotation_tasks(
    template_paths: Iterable[str | Path],
    annotator_ids: Iterable[str],
    output_path: str | Path,
) -> dict[str, Any]:
    """Expand audit templates into pending double-annotation JSONL tasks."""

    annotators = _validated_annotators(annotator_ids)
    templates = _load_template_records(template_paths)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for template in templates:
            for annotator_id in annotators:
                task = _pending_task_record(template, annotator_id)
                fh.write(json.dumps(task, ensure_ascii=False, sort_keys=True))
                fh.write("\n")

    return {
        "annotation_schema_version": "amst-human-audit-tasks-v1",
        "output_file": str(target),
        "num_template_records": len(templates),
        "num_annotation_tasks": len(templates) * len(annotators),
        "num_annotators": len(annotators),
        "annotator_ids": annotators,
        "checks": list(AUDIT_CHECK_FIELDS),
        "check_definitions": dict(AUDIT_CHECK_DEFINITIONS),
        "ready_for_human_annotation": bool(templates and annotators),
        "completion_status": "pending_human_labels",
    }


def write_double_annotation_task_package(
    template_paths: Iterable[str | Path],
    annotator_ids: Iterable[str],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write one pending JSONL task file per annotator plus a task manifest."""

    annotators = _validated_annotators(annotator_ids)
    templates = _load_template_records(template_paths)
    output = Path(output_dir)
    task_dir = output / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    template_files = sorted({template["source"] for template in templates})
    task_files: dict[str, str] = {}
    task_identity_digests: dict[str, str] = {}
    for annotator_id in annotators:
        path = task_dir / f"{_safe_filename(annotator_id)}.jsonl"
        pending_rows: list[dict[str, Any]] = []
        with path.open("w", encoding="utf-8") as fh:
            for template in templates:
                task = _pending_task_record(template, annotator_id)
                pending_rows.append(task)
                fh.write(json.dumps(task, ensure_ascii=False, sort_keys=True))
                fh.write("\n")
        task_files[annotator_id] = str(path)
        task_identity_digests[annotator_id] = _task_identity_digest_from_records(pending_rows)

    manifest = {
        "annotation_task_schema_version": "amst-human-audit-task-package-v1",
        "status": "assigned",
        "template_files": template_files,
        "template_file_digests": _template_file_digests(template_files),
        "task_files": task_files,
        "task_identity_digests": task_identity_digests,
        "num_template_items": len(templates),
        "num_annotators": len(annotators),
        "expected_annotations": len(templates) * len(annotators),
        "annotator_ids": annotators,
        "checks": list(AUDIT_CHECK_FIELDS),
        "check_definitions": dict(AUDIT_CHECK_DEFINITIONS),
        "completion_status": "pending_human_labels",
    }
    manifest_path = output / "task_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    manifest["task_manifest_file"] = str(manifest_path)
    return manifest


def write_human_audit_annotation_sheets(
    task_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Export per-annotator spreadsheet-friendly CSV sheets from a task package."""

    manifest_file = Path(task_manifest_path)
    with manifest_file.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        raise ValueError("human-audit task manifest must be a JSON object")

    errors: list[str] = []
    _verify_task_manifest_fingerprints(manifest, manifest_file.parent, errors)
    task_files = manifest.get("task_files")
    if not isinstance(task_files, dict) or not task_files:
        errors.append("task manifest task_files must be a non-empty object")
    if errors:
        raise ValueError("cannot export human-audit annotation sheets: " + "; ".join(errors[:8]))

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    annotation_sheet_files: dict[str, str] = {}
    per_annotator_rows: dict[str, int] = {}
    total_rows = 0
    for annotator_id, raw_path in sorted(task_files.items()):
        annotator = str(annotator_id)
        source_path = _resolve_task_path(manifest_file.parent, str(raw_path))
        records = _load_task_records_for_annotator(source_path, annotator)
        output_path = target_dir / f"{_safe_filename(annotator)}.csv"
        with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=ANNOTATION_SHEET_COLUMNS)
            writer.writeheader()
            for record in records:
                writer.writerow(_annotation_sheet_row(record))
        annotation_sheet_files[annotator] = str(output_path)
        per_annotator_rows[annotator] = len(records)
        total_rows += len(records)

    return {
        "annotation_sheet_schema_version": "amst-human-audit-annotation-sheets-v1",
        "task_manifest_file": str(manifest_file),
        "output_dir": str(target_dir),
        "annotation_sheet_files": annotation_sheet_files,
        "per_annotator_rows": per_annotator_rows,
        "num_sheets": len(annotation_sheet_files),
        "num_rows": total_rows,
    }


def apply_human_audit_annotation_sheet(
    task_manifest_path: str | Path,
    annotator_id: str,
    sheet_path: str | Path,
) -> dict[str, Any]:
    """Apply an edited annotation CSV sheet back into its canonical JSONL task file."""

    manifest_file = Path(task_manifest_path)
    with manifest_file.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        raise ValueError("human-audit task manifest must be a JSON object")

    errors: list[str] = []
    _verify_task_manifest_fingerprints(manifest, manifest_file.parent, errors)
    task_files = manifest.get("task_files")
    annotator = str(annotator_id).strip()
    if not annotator:
        errors.append("annotator_id is required")
    if not isinstance(task_files, dict) or annotator not in task_files:
        errors.append(f"annotator {annotator!r} is not declared in task_manifest task_files")
    if errors:
        raise ValueError("cannot apply human-audit annotation sheet: " + "; ".join(errors[:8]))

    task_file = _resolve_task_path(manifest_file.parent, str(task_files[annotator]))
    records = _load_task_records_for_annotator(task_file, annotator)
    record_map = {(str(record["case_id"]), str(record["query_id"]), str(record["annotator_id"])): record for record in records}
    expected_keys = set(record_map)

    sheet_file = Path(sheet_path)
    with sheet_file.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("annotation sheet CSV must include a header row")
        missing_columns = [column for column in ("case_id", "query_id", "annotator_id", *AUDIT_CHECK_FIELDS, "notes") if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"annotation sheet CSV is missing required columns: {missing_columns}")
        seen_keys: set[tuple[str, str, str]] = set()
        started_annotations = 0
        completed_annotations = 0
        updated_records: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row_number, row in enumerate(reader, start=2):
            case_id = str(row.get("case_id", "")).strip()
            query_id = str(row.get("query_id", "")).strip()
            row_annotator = str(row.get("annotator_id", "")).strip()
            if not case_id or not query_id or not row_annotator:
                raise ValueError(f"{sheet_file}:{row_number}: case_id, query_id, and annotator_id are required")
            if row_annotator != annotator:
                raise ValueError(
                    f"{sheet_file}:{row_number}: annotator_id {row_annotator!r} does not match requested annotator {annotator!r}"
                )
            key = (case_id, query_id, row_annotator)
            if key in seen_keys:
                raise ValueError(f"{sheet_file}:{row_number}: duplicate sheet row for {_format_task_key(key)}")
            seen_keys.add(key)
            if key not in record_map:
                raise ValueError(f"{sheet_file}:{row_number}: sheet row does not match any task: {_format_task_key(key)}")

            record = dict(record_map[key])
            applicable_checks = set(_resolve_applicable_checks(record, task_file, 1))
            updated_checks: dict[str, bool | None] = {}
            any_started = False
            all_complete = True
            for field in AUDIT_CHECK_FIELDS:
                value = _sheet_binary_value(row.get(field), sheet_file, row_number, field)
                if field not in applicable_checks and value is not None:
                    raise ValueError(f"{sheet_file}:{row_number}: non-applicable check {field!r} must be blank")
                if field in applicable_checks and value is None:
                    all_complete = False
                if field in applicable_checks and value is not None:
                    any_started = True
                updated_checks[field] = value
            notes = _sheet_notes_value(row.get("notes"))
            if notes is not None:
                any_started = True

            record["checks"] = updated_checks
            record["notes"] = notes
            record["annotation_status"] = "completed" if all_complete else "pending"
            updated_records[key] = record
            if any_started:
                started_annotations += 1
            if all_complete:
                completed_annotations += 1

    missing_keys = sorted(expected_keys - set(updated_records))
    if missing_keys:
        raise ValueError(
            "annotation sheet is missing task rows: "
            + ", ".join(_format_task_key(key) for key in missing_keys[:10])
        )

    ordered_records = []
    for record in records:
        key = (str(record["case_id"]), str(record["query_id"]), str(record["annotator_id"]))
        ordered_records.append(updated_records[key])
    task_file.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in ordered_records) + "\n",
        encoding="utf-8",
    )
    return {
        "annotation_sheet_apply_schema_version": "amst-human-audit-annotation-sheet-apply-v1",
        "task_manifest_file": str(manifest_file),
        "task_file": str(task_file),
        "sheet_file": str(sheet_file),
        "annotator_id": annotator,
        "num_rows": len(ordered_records),
        "num_started_annotations": started_annotations,
        "num_completed_annotations": completed_annotations,
        "status": "completed" if completed_annotations == len(ordered_records) else "updated",
    }


def apply_human_audit_annotator_packet(
    task_manifest_path: str | Path,
    packet_path: str | Path,
    annotator_id: str | None = None,
) -> dict[str, Any]:
    """Apply a returned annotator packet directory or zip archive back into the canonical task file."""

    manifest_file = Path(task_manifest_path)
    with manifest_file.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        raise ValueError("human-audit task manifest must be a JSON object")

    errors: list[str] = []
    _verify_task_manifest_fingerprints(manifest, manifest_file.parent, errors)
    task_files = manifest.get("task_files")
    task_identity_digests = manifest.get("task_identity_digests")
    if not isinstance(task_files, dict) or not task_files:
        errors.append("task manifest task_files must be a non-empty object")
    if not isinstance(task_identity_digests, dict) or not task_identity_digests:
        errors.append("task manifest task_identity_digests must be a non-empty object")
    if errors:
        raise ValueError("cannot apply human-audit annotator packet: " + "; ".join(errors[:8]))

    source = Path(packet_path)
    if not source.exists():
        raise ValueError(f"annotator packet path does not exist: {source}")

    packet_type = "directory"
    packet_root: Path
    packet_sheet_file: Path
    packet_task_file: Path
    packet_manifest: dict[str, Any] | None = None
    resolved_annotator: str
    expected_digest: str
    actual_digest: str
    with TemporaryDirectory() as temp_dir:
        if source.is_dir():
            packet_root = _resolve_annotator_packet_root(source)
        elif source.is_file() and source.suffix.lower() == ".zip":
            packet_type = "zip"
            extract_root = Path(temp_dir)
            try:
                with zipfile.ZipFile(source) as archive:
                    archive.extractall(extract_root)
            except zipfile.BadZipFile as exc:
                raise ValueError(f"annotator packet archive is invalid: {source}") from exc
            packet_root = _resolve_annotator_packet_root(extract_root)
        else:
            raise ValueError("annotator packet must be a directory or .zip archive")

        packet_manifest = _load_optional_annotator_packet_manifest(packet_root)
        packet_sheet_file = _single_annotator_packet_file(packet_root, "*.csv", "annotation sheet")
        packet_task_file = _single_annotator_packet_file(packet_root, "*.jsonl", "task file")
        resolved_annotator = _resolve_annotator_packet_annotator_id(
            packet_sheet_file,
            packet_task_file,
            annotator_id,
        )
        if resolved_annotator not in task_files:
            raise ValueError(f"annotator {resolved_annotator!r} is not declared in task_manifest task_files")
        expected_digest = task_identity_digests.get(resolved_annotator, "")
        if not isinstance(expected_digest, str) or not expected_digest:
            raise ValueError(f"task manifest is missing task identity digest for annotator {resolved_annotator!r}")
        actual_digest = _task_identity_digest_from_path(packet_task_file)
        if actual_digest != expected_digest:
            raise ValueError(
                "annotator packet task identity digest mismatch for "
                f"{resolved_annotator!r}: expected={expected_digest} actual={actual_digest}"
            )
        _validate_annotator_packet_manifest(
            packet_manifest,
            packet_root,
            packet_sheet_file,
            packet_task_file,
            resolved_annotator,
            actual_digest,
        )

        apply_summary = apply_human_audit_annotation_sheet(
            manifest_file,
            resolved_annotator,
            packet_sheet_file,
        )

    return {
        "annotation_packet_apply_schema_version": "amst-human-audit-annotator-packet-apply-v1",
        "task_manifest_file": str(manifest_file),
        "packet_path": str(source),
        "packet_type": packet_type,
        "packet_root": str(packet_root),
        "annotator_id": resolved_annotator,
        "packet_manifest_file": str(packet_root / "packet_manifest.json") if packet_manifest is not None else None,
        "sheet_file": str(packet_sheet_file),
        "packet_task_file": str(packet_task_file),
        "expected_task_identity_digest": expected_digest,
        "verified_task_identity_digest": actual_digest,
        **apply_summary,
    }


def merge_completed_human_audit_tasks(task_manifest_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Merge fully filled per-annotator task files into a completed annotation JSONL."""

    manifest_file = Path(task_manifest_path)
    with manifest_file.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        raise ValueError("human-audit task manifest must be a JSON object")

    errors: list[str] = []
    _verify_task_manifest_fingerprints(manifest, manifest_file.parent, errors)
    records = _completed_annotation_records_from_task_manifest(manifest, manifest_file.parent, errors)
    expected_tasks = _expected_tasks_from_task_manifest(manifest, manifest_file.parent, errors)
    record_keys = {(record["case_id"], record["query_id"], record["annotator_id"]) for record in records}
    missing = sorted(expected_tasks - record_keys)
    extra = sorted(record_keys - expected_tasks)
    if missing:
        errors.append(f"completed task files are missing annotations: {[_format_task_key(key) for key in missing[:10]]}")
    if extra:
        errors.append(f"completed task files contain non-task annotations: {[_format_task_key(key) for key in extra[:10]]}")
    if errors:
        raise ValueError("completed human audit task merge failed: " + "; ".join(errors[:8]))

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for record in sorted(records, key=lambda item: (item["case_id"], item["query_id"], item["annotator_id"])):
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            fh.write("\n")

    return {
        "annotation_merge_schema_version": "amst-human-audit-annotation-merge-v1",
        "task_manifest_file": str(manifest_file),
        "annotations_file": str(target),
        "num_annotations": len(records),
        "num_template_items": len({(record["case_id"], record["query_id"]) for record in records}),
        "num_annotators": len({record["annotator_id"] for record in records}),
        "annotator_ids": sorted({record["annotator_id"] for record in records}),
        "ready_for_finalize": True,
        "completion_status": "completed_annotations_merged",
    }


def load_audit_annotations(path: str | Path) -> tuple[AuditAnnotation, ...]:
    """Read audit subset annotation JSONL records."""

    annotations: list[AuditAnnotation] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSONL record") from exc
            annotations.append(_annotation_from_record(record, source, line_number))
    return tuple(annotations)


def compute_agreement(annotations: Iterable[AuditAnnotation]) -> dict[str, Any]:
    """Compute pairwise percent agreement and Cohen's kappa for audit checks."""

    by_item: dict[tuple[str, str], list[AuditAnnotation]] = defaultdict(list)
    annotator_ids: set[str] = set()
    total_records = 0
    for annotation in annotations:
        by_item[annotation.item_id].append(annotation)
        annotator_ids.add(annotation.annotator_id)
        total_records += 1

    fields = {field: _field_agreement(field, by_item) for field in AUDIT_CHECK_FIELDS}
    return {
        "num_annotations": total_records,
        "num_items": len(by_item),
        "num_annotators": len(annotator_ids),
        "annotator_ids": sorted(annotator_ids),
        "fields": fields,
    }


def _annotation_from_record(record: dict[str, Any], source: Path, line_number: int) -> AuditAnnotation:
    if not isinstance(record, dict):
        raise ValueError(f"{source}:{line_number}: annotation record must be an object")
    try:
        case_id = str(record["case_id"])
        query_id = str(record["query_id"])
    except KeyError as exc:
        raise ValueError(f"{source}:{line_number}: missing required field {exc.args[0]!r}") from exc

    annotator = record.get("annotator_id")
    if annotator is None or str(annotator).strip() == "":
        raise ValueError(f"{source}:{line_number}: missing required field 'annotator_id'")

    raw_checks = record.get("checks", {})
    if raw_checks is None:
        raw_checks = {}
    if not isinstance(raw_checks, dict):
        raise ValueError(f"{source}:{line_number}: 'checks' must be an object")

    checks = {
        field: _binary_value(raw_checks.get(field, record.get(field)), source, line_number, field)
        for field in AUDIT_CHECK_FIELDS
    }
    applicable_checks = frozenset(_resolve_applicable_checks(record, source, line_number))
    return AuditAnnotation(
        case_id=case_id,
        query_id=query_id,
        annotator_id=str(annotator),
        checks=checks,
        applicable_checks=applicable_checks,
        line_number=line_number,
    )


def _binary_value(value: Any, source: Path, line_number: int, field: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1", "pass", "passed"}:
            return True
        if normalized in {"false", "f", "no", "n", "0", "fail", "failed"}:
            return False
    raise ValueError(f"{source}:{line_number}: check {field!r} must be binary or null")


def _field_agreement(field: str, by_item: dict[tuple[str, str], list[AuditAnnotation]]) -> dict[str, Any]:
    pairs: list[tuple[bool, bool]] = []
    compared_items = 0
    skipped_items = 0
    applicable_items = 0
    not_applicable_items = 0

    for item_annotations in by_item.values():
        applicability = {
            annotation.annotator_id: field in annotation.applicable_checks
            for annotation in item_annotations
        }
        if applicability and not any(applicability.values()):
            not_applicable_items += 1
            continue
        if len(set(applicability.values())) > 1:
            raise ValueError(
                f"inconsistent applicability for case_id={item_annotations[0].case_id!r}, "
                f"query_id={item_annotations[0].query_id!r}, field={field!r}"
            )
        applicable_items += 1
        ratings: dict[str, bool] = {}
        for annotation in item_annotations:
            if field not in annotation.applicable_checks:
                continue
            value = annotation.checks[field]
            if value is None:
                continue
            if annotation.annotator_id in ratings:
                raise ValueError(
                    f"duplicate annotation for case_id={annotation.case_id!r}, "
                    f"query_id={annotation.query_id!r}, annotator_id={annotation.annotator_id!r}, "
                    f"field={field!r}"
                )
            ratings[annotation.annotator_id] = value
        if len(ratings) < 2:
            skipped_items += 1
            continue
        compared_items += 1
        for left, right in combinations(sorted(ratings), 2):
            pairs.append((ratings[left], ratings[right]))

    agreements = sum(1 for left, right in pairs if left == right)
    percent_agreement = agreements / len(pairs) if pairs else None
    return {
        "num_applicable_items": applicable_items,
        "num_not_applicable_items": not_applicable_items,
        "num_items": compared_items,
        "num_skipped_items": skipped_items,
        "num_pairs": len(pairs),
        "num_agreements": agreements,
        "percent_agreement": percent_agreement,
        "cohen_kappa": _cohen_kappa(pairs),
    }


def _cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    if not pairs:
        return None

    total = len(pairs)
    observed = sum(1 for left, right in pairs if left == right) / total
    left_true = sum(1 for left, _ in pairs if left) / total
    right_true = sum(1 for _, right in pairs if right) / total
    left_false = 1.0 - left_true
    right_false = 1.0 - right_true
    expected = left_true * right_true + left_false * right_false
    denominator = 1.0 - expected
    if denominator == 0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / denominator


def _increment(values: dict[str, int], key: str) -> None:
    values[key] = values.get(key, 0) + 1


def _validated_annotators(annotator_ids: Iterable[str]) -> list[str]:
    annotators = [str(annotator).strip() for annotator in annotator_ids if str(annotator).strip()]
    if len(annotators) < 2:
        raise ValueError("at least two annotator ids are required")
    if len(set(annotators)) != len(annotators):
        raise ValueError("annotator ids must be unique")
    return annotators


def _human_audit_progress_next_commands(
    task_manifest_path: Path,
    annotations_path: Path | None,
    *,
    ready_for_merge: bool,
    ready_for_finalize: bool,
) -> list[str]:
    progress_output = task_manifest_path.parent / "progress.json"
    commands: list[str] = []
    if annotations_path is None:
        commands.append(
            f"{HUMAN_AUDIT_CLI_PREFIX} export-human-audit-annotation-sheets "
            f"--task-manifest {task_manifest_path} "
            f"--output-dir {task_manifest_path.parent / 'sheets'}"
        )
        if ready_for_merge:
            merged_annotations = task_manifest_path.parent / "completed_annotations.jsonl"
            commands.append(
                f"{HUMAN_AUDIT_CLI_PREFIX} merge-completed-human-audit-tasks "
                f"--task-manifest {task_manifest_path} "
                f"--output {merged_annotations}"
            )
            commands.append(
                f"{HUMAN_AUDIT_CLI_PREFIX} human-audit-progress "
                f"--task-manifest {task_manifest_path} "
                f"--annotations {merged_annotations} "
                f"--output {progress_output}"
            )
            return commands
        commands.append(
            f"{HUMAN_AUDIT_CLI_PREFIX} human-audit-progress "
            f"--task-manifest {task_manifest_path} "
            f"--output {progress_output}"
        )
        return commands

    commands.append(
        f"{HUMAN_AUDIT_CLI_PREFIX} human-audit-progress "
        f"--task-manifest {task_manifest_path} "
        f"--annotations {annotations_path} "
        f"--output {progress_output}"
    )
    if ready_for_finalize:
        commands.insert(
            0,
            f"{HUMAN_AUDIT_CLI_PREFIX} agreement "
            f"--annotations {annotations_path} "
            f"--output {task_manifest_path.parent / 'agreement_metrics.json'}"
        )
    return commands


def _load_template_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    seen_items: set[tuple[str, str]] = set()
    for path in paths:
        source = Path(path)
        with source.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source}:{line_number}: invalid JSONL record") from exc
                case_id = str(record.get("case_id", ""))
                query_id = str(record.get("query_id", ""))
                if not case_id or not query_id:
                    raise ValueError(f"{source}:{line_number}: template record must include case_id and query_id")
                key = (case_id, query_id)
                if key in seen_items:
                    raise ValueError(f"{source}:{line_number}: duplicate audit template item {case_id}/{query_id}")
                seen_items.add(key)
                checks = record.get("checks")
                if not isinstance(checks, dict) or any(field not in checks for field in AUDIT_CHECK_FIELDS):
                    raise ValueError(f"{source}:{line_number}: template record is missing required check fields")
                _resolve_applicable_checks(record, source, line_number)
                templates.append({"record": record, "source": str(source), "line_number": line_number})
    if not templates:
        raise ValueError("no audit template records found")
    return templates


def _load_task_records_for_annotator(path: Path, annotator_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL task record: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: task record must be an object")
            if str(record.get("annotator_id", "")) != annotator_id:
                raise ValueError(
                    f"{path}:{line_number}: task annotator_id {record.get('annotator_id')!r} does not match {annotator_id!r}"
                )
            records.append(record)
    return records


def _annotation_sheet_row(record: dict[str, Any]) -> dict[str, str]:
    checks = record.get("checks")
    if not isinstance(checks, dict):
        checks = {}
    applicable_checks = {str(item) for item in record.get("applicable_checks", [])}
    return {
        "case_id": str(record.get("case_id", "")),
        "query_id": str(record.get("query_id", "")),
        "annotator_id": str(record.get("annotator_id", "")),
        "annotation_status": str(record.get("annotation_status", "pending")),
        "domain": str(record.get("domain", "")),
        "probe_type": str(record.get("probe_type", "")),
        "task_type": str(record.get("task_type", "")),
        "difficulty_level": str(record.get("difficulty_level", "")),
        "memory_requirement": str(record.get("memory_requirement", "")),
        "memory_dependency": str(record.get("memory_dependency", "")),
        "counterfactual_group_id": str(record.get("counterfactual_group_id", "")),
        "counterfactual_axis": str(record.get("counterfactual_axis", "")),
        "counterfactual_edit": str(record.get("counterfactual_edit", "")),
        "scoring_rule": str(record.get("scoring_rule", "")),
        "applicable_checks": ";".join(str(item) for item in record.get("applicable_checks", [])),
        "prompt": str(record.get("prompt", "")),
        "expected_behavior_json": json.dumps(record.get("expected_behavior", {}), ensure_ascii=False, sort_keys=True),
        "gold_memory_ids_json": json.dumps(record.get("gold_memory_ids", []), ensure_ascii=False, sort_keys=True),
        "forbidden_memory_ids_json": json.dumps(record.get("forbidden_memory_ids", []), ensure_ascii=False, sort_keys=True),
        "gold_memory_evidence_json": json.dumps(record.get("gold_memory_evidence", []), ensure_ascii=False, sort_keys=True),
        "forbidden_memory_evidence_json": json.dumps(record.get("forbidden_memory_evidence", []), ensure_ascii=False, sort_keys=True),
        "relevant_events_json": json.dumps(record.get("relevant_events", []), ensure_ascii=False, sort_keys=True),
        "state_contract_id": str(record.get("state_contract_id", "")),
        "state_contract_summary_json": json.dumps(
            record.get("state_contract_summary", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "source_template": str(record.get("source_template", "")),
        "source_template_line": str(record.get("source_template_line", "")),
        "evidence_sufficient": _sheet_check_text("evidence_sufficient", applicable_checks, checks),
        "answer_unique": _sheet_check_text("answer_unique", applicable_checks, checks),
        "governance_boundary_clear": _sheet_check_text("governance_boundary_clear", applicable_checks, checks),
        "trace_natural": _sheet_check_text("trace_natural", applicable_checks, checks),
        "scenario_memory_required": _sheet_check_text("scenario_memory_required", applicable_checks, checks),
        "counterfactual_target_state_only": _sheet_check_text("counterfactual_target_state_only", applicable_checks, checks),
        "notes": str(record.get("notes") or ""),
    }


def _sheet_bool_text(value: bool | None) -> str:
    if value is None:
        return ""
    return "TRUE" if value else "FALSE"


def _sheet_check_text(field: str, applicable_checks: set[str], checks: dict[str, Any]) -> str:
    if field not in applicable_checks:
        return "N/A"
    return _sheet_bool_text(checks.get(field))


def _sheet_binary_value(value: Any, source: Path, row_number: int, field: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().upper() in {"", "N/A", "NA"}:
        return None
    return _binary_value(value, source, row_number, field)


def _sheet_notes_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _resolve_annotator_packet_root(path: Path) -> Path:
    if _looks_like_annotator_packet_root(path):
        return path
    child_dirs = [child for child in sorted(path.iterdir()) if child.is_dir()]
    packet_roots = [child for child in child_dirs if _looks_like_annotator_packet_root(child)]
    if len(packet_roots) == 1:
        return packet_roots[0]
    raise ValueError(f"could not locate annotator packet root under {path}")


def _looks_like_annotator_packet_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(path.glob("*.csv")) and any(path.glob("*.jsonl"))


def _single_annotator_packet_file(packet_root: Path, pattern: str, label: str) -> Path:
    matches = sorted(packet_root.glob(pattern))
    if not matches:
        raise ValueError(f"annotator packet is missing {label}: {packet_root}")
    if len(matches) > 1:
        raise ValueError(f"annotator packet has multiple {label} candidates: {[path.name for path in matches]}")
    return matches[0]


def _resolve_annotator_packet_annotator_id(
    sheet_path: Path,
    task_path: Path,
    annotator_id: str | None,
) -> str:
    sheet_annotators = _sheet_annotator_ids(sheet_path)
    task_annotators = _task_record_annotator_ids(task_path)
    candidates = sheet_annotators | task_annotators
    if annotator_id is not None and str(annotator_id).strip():
        candidate = str(annotator_id).strip()
        if sheet_annotators and candidate not in sheet_annotators:
            raise ValueError(
                f"annotator packet sheet annotator ids {sorted(sheet_annotators)} do not match requested annotator {candidate!r}"
            )
        if task_annotators and candidate not in task_annotators:
            raise ValueError(
                f"annotator packet task annotator ids {sorted(task_annotators)} do not match requested annotator {candidate!r}"
            )
        return candidate
    if len(candidates) != 1:
        raise ValueError(f"could not infer a unique annotator_id from annotator packet: {sorted(candidates)}")
    return next(iter(candidates))


def _load_optional_annotator_packet_manifest(packet_root: Path) -> dict[str, Any] | None:
    manifest_path = packet_root / "packet_manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        raise ValueError(f"annotator packet manifest must be a JSON object: {manifest_path}")
    return manifest


def _validate_annotator_packet_manifest(
    packet_manifest: dict[str, Any] | None,
    packet_root: Path,
    sheet_path: Path,
    task_path: Path,
    annotator_id: str,
    task_identity_digest: str,
) -> None:
    if packet_manifest is None:
        return
    if packet_manifest.get("schema_version") != "amst-human-audit-annotator-packet-v1":
        raise ValueError("annotator packet manifest schema_version is invalid")
    if packet_manifest.get("packet_type") != "annotator":
        raise ValueError("annotator packet manifest packet_type must be 'annotator'")
    if str(packet_manifest.get("annotator_id") or "") != annotator_id:
        raise ValueError("annotator packet manifest annotator_id does not match packet contents")
    for field, actual_path in (
        ("sheet_file", sheet_path),
        ("task_file", task_path),
    ):
        raw_path = packet_manifest.get(field)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"annotator packet manifest is missing {field}")
        resolved_path = raw_path if Path(raw_path).is_absolute() else packet_root / raw_path
        if resolved_path.resolve() != actual_path.resolve():
            raise ValueError(f"annotator packet manifest {field} does not match packet contents")
    if packet_manifest.get("task_identity_digest") != task_identity_digest:
        raise ValueError("annotator packet manifest task_identity_digest does not match packet task file")


def _sheet_annotator_ids(path: Path) -> set[str]:
    annotators: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "annotator_id" not in reader.fieldnames:
            raise ValueError(f"annotation sheet CSV must include annotator_id column: {path}")
        for row_number, row in enumerate(reader, start=2):
            annotator = str(row.get("annotator_id", "")).strip()
            if not annotator:
                raise ValueError(f"{path}:{row_number}: annotator_id is required")
            annotators.add(annotator)
    return annotators


def _task_record_annotator_ids(path: Path) -> set[str]:
    annotators: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL task record: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: task record must be an object")
            annotator = str(record.get("annotator_id", "")).strip()
            if not annotator:
                raise ValueError(f"{path}:{line_number}: annotator_id is required")
            annotators.add(annotator)
    return annotators


def _pending_task_record(template: dict[str, Any], annotator_id: str) -> dict[str, Any]:
    task = dict(template["record"])
    task.pop("audit_reference", None)
    task["annotator_id"] = annotator_id
    task["annotation_status"] = "pending"
    task["checks"] = {field: None for field in AUDIT_CHECK_FIELDS}
    task["source_template"] = template["source"]
    task["source_template_line"] = template["line_number"]
    task.setdefault("notes", None)
    return task


def _task_records_from_task_manifest(
    manifest: dict[str, Any],
    manifest_dir: Path,
    errors: list[str],
) -> list[dict[str, str]]:
    task_files = manifest.get("task_files")
    if not isinstance(task_files, dict) or not task_files:
        errors.append("task manifest task_files must be a non-empty object")
        return []

    records: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for annotator_id, raw_path in sorted(task_files.items()):
        annotator = str(annotator_id)
        path = _resolve_task_path(manifest_dir, str(raw_path))
        if not path.exists():
            errors.append(f"task file does not exist for annotator {annotator!r}: {path}")
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path}:{line_number}: invalid JSONL task record: {exc}")
                    continue
                if not isinstance(record, dict):
                    errors.append(f"{path}:{line_number}: task record must be an object")
                    continue
                case_id = str(record.get("case_id", ""))
                query_id = str(record.get("query_id", ""))
                task_annotator = str(record.get("annotator_id", ""))
                if not case_id or not query_id or not task_annotator:
                    errors.append(f"{path}:{line_number}: task record must include case_id, query_id, and annotator_id")
                    continue
                if task_annotator != annotator:
                    errors.append(
                        f"{path}:{line_number}: task annotator_id {task_annotator!r} does not match manifest key {annotator!r}"
                    )
                key = (case_id, query_id, task_annotator)
                if key in seen:
                    errors.append(f"{path}:{line_number}: duplicate task {_format_task_key(key)}")
                seen.add(key)
                records.append(
                    {
                        "case_id": case_id,
                        "query_id": query_id,
                        "annotator_id": task_annotator,
                        "domain": str(record.get("domain", "unknown") or "unknown"),
                        "probe_type": str(record.get("probe_type", "unknown") or "unknown"),
                        "task_file": str(path),
                    }
                )
    return records


def _expected_tasks_from_task_manifest(
    manifest: dict[str, Any],
    manifest_dir: Path,
    errors: list[str],
) -> set[tuple[str, str, str]]:
    return {
        (record["case_id"], record["query_id"], record["annotator_id"])
        for record in _task_records_from_task_manifest(manifest, manifest_dir, errors)
    }


def _completed_annotation_records_from_task_manifest(
    manifest: dict[str, Any],
    manifest_dir: Path,
    errors: list[str],
) -> list[dict[str, Any]]:
    task_files = manifest.get("task_files")
    if not isinstance(task_files, dict) or not task_files:
        errors.append("task manifest task_files must be a non-empty object")
        return []
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for annotator_id, raw_path in sorted(task_files.items()):
        annotator = str(annotator_id)
        path = _resolve_task_path(manifest_dir, str(raw_path))
        if not path.exists():
            errors.append(f"task file does not exist for annotator {annotator!r}: {path}")
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path}:{line_number}: invalid JSONL task record: {exc}")
                    continue
                if not isinstance(record, dict):
                    errors.append(f"{path}:{line_number}: task record must be an object")
                    continue
                annotation = _completed_annotation_record(record, annotator, path, line_number, errors)
                if annotation is None:
                    continue
                key = (annotation["case_id"], annotation["query_id"], annotation["annotator_id"])
                if key in seen:
                    errors.append(f"{path}:{line_number}: duplicate completed annotation {_format_task_key(key)}")
                    continue
                seen.add(key)
                records.append(annotation)
    return records


def _completed_annotation_record(
    record: dict[str, Any],
    manifest_annotator_id: str,
    source: Path,
    line_number: int,
    errors: list[str],
) -> dict[str, Any] | None:
    case_id = str(record.get("case_id", ""))
    query_id = str(record.get("query_id", ""))
    annotator_id = str(record.get("annotator_id", ""))
    if not case_id or not query_id or not annotator_id:
        errors.append(f"{source}:{line_number}: task record must include case_id, query_id, and annotator_id")
        return None
    if annotator_id != manifest_annotator_id:
        errors.append(
            f"{source}:{line_number}: task annotator_id {annotator_id!r} does not match manifest key {manifest_annotator_id!r}"
        )
        return None
    raw_checks = record.get("checks")
    if not isinstance(raw_checks, dict):
        errors.append(f"{source}:{line_number}: checks must be an object")
        return None
    try:
        applicable_checks = tuple(_resolve_applicable_checks(record, source, line_number))
    except ValueError as exc:
        errors.append(str(exc))
        return None
    applicable_set = set(applicable_checks)
    checks: dict[str, bool | None] = {}
    for field in AUDIT_CHECK_FIELDS:
        try:
            value = _binary_value(raw_checks.get(field, record.get(field)), source, line_number, field)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if field in applicable_set and value is None:
            errors.append(f"{source}:{line_number}: completed task check {field!r} is missing")
        elif field not in applicable_set and value is not None:
            errors.append(f"{source}:{line_number}: non-applicable task check {field!r} must be null")
        else:
            checks[field] = value
    if len(checks) != len(AUDIT_CHECK_FIELDS):
        return None
    return {
        "case_id": case_id,
        "query_id": query_id,
        "annotator_id": annotator_id,
        "checks": checks,
        "applicable_checks": list(applicable_checks),
        "annotation_status": "completed",
        "source_task_file": str(source),
        "source_task_line": line_number,
        "source_template": record.get("source_template"),
        "source_template_line": record.get("source_template_line"),
        "notes": record.get("notes"),
    }


def _resolve_applicable_checks(record: dict[str, Any], source: Path, line_number: int) -> tuple[str, ...]:
    raw = record.get("applicable_checks")
    if raw is None:
        return tuple(field for field in AUDIT_CHECK_FIELDS if _default_check_applicability(field, record))
    if not isinstance(raw, list):
        raise ValueError(f"{source}:{line_number}: applicable_checks must be a list")
    applicable: list[str] = []
    seen: set[str] = set()
    for value in raw:
        field = str(value)
        if field not in AUDIT_CHECK_FIELDS:
            raise ValueError(f"{source}:{line_number}: unknown applicable check field {field!r}")
        if field in seen:
            raise ValueError(f"{source}:{line_number}: duplicate applicable check field {field!r}")
        applicable.append(field)
        seen.add(field)
    required_defaults = {field for field in AUDIT_CHECK_FIELDS if _default_check_applicability(field, record)}
    if not required_defaults.issubset(seen):
        missing = sorted(required_defaults - seen)
        raise ValueError(f"{source}:{line_number}: applicable_checks are missing required fields: {missing}")
    return tuple(applicable)


def _default_check_applicability(field: str, record: dict[str, Any]) -> bool:
    if field in COUNTERFACTUAL_ONLY_AUDIT_CHECKS:
        return bool(record.get("counterfactual_group_id") or record.get("counterfactual_context"))
    return True


def _resolve_task_path(manifest_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidate = manifest_dir / path
    if candidate.exists():
        return candidate
    if path.exists():
        return path
    return candidate


def _task_item_dimensions(
    task_records: Iterable[dict[str, str]],
    errors: list[str],
) -> dict[tuple[str, str], dict[str, str]]:
    dimensions: dict[tuple[str, str], dict[str, str]] = {}
    for record in task_records:
        item_key = (record["case_id"], record["query_id"])
        current = {
            "domain": record["domain"],
            "probe_type": record["probe_type"],
        }
        previous = dimensions.get(item_key)
        if previous is not None and previous != current:
            errors.append(
                "task item metadata mismatch across annotators for "
                f"{record['case_id']}/{record['query_id']}: {previous} vs {current}"
            )
            continue
        dimensions[item_key] = current
    return dimensions


def _annotator_progress(
    manifest: dict[str, Any],
    task_records: Iterable[dict[str, str]],
    started_annotation_keys: set[tuple[str, str, str]],
    matched_annotation_keys: set[tuple[str, str, str]],
    complete_annotation_keys: set[tuple[str, str, str]],
) -> dict[str, dict[str, Any]]:
    task_files = manifest.get("task_files")
    task_file_map = {
        str(annotator_id): str(raw_path)
        for annotator_id, raw_path in sorted(task_files.items())
    } if isinstance(task_files, dict) else {}

    expected_by_annotator: dict[str, int] = defaultdict(int)
    for record in task_records:
        expected_by_annotator[record["annotator_id"]] += 1

    started_by_annotator: dict[str, int] = defaultdict(int)
    for _, _, annotator_id in started_annotation_keys:
        started_by_annotator[annotator_id] += 1

    matched_by_annotator: dict[str, int] = defaultdict(int)
    for _, _, annotator_id in matched_annotation_keys:
        matched_by_annotator[annotator_id] += 1

    complete_by_annotator: dict[str, int] = defaultdict(int)
    for _, _, annotator_id in complete_annotation_keys:
        complete_by_annotator[annotator_id] += 1

    progress: dict[str, dict[str, Any]] = {}
    for annotator_id in sorted(expected_by_annotator):
        expected = expected_by_annotator[annotator_id]
        matched = matched_by_annotator.get(annotator_id, 0)
        complete = complete_by_annotator.get(annotator_id, 0)
        progress[annotator_id] = {
            "task_file": task_file_map.get(annotator_id),
            "num_expected_annotations": expected,
            "num_started_annotations": started_by_annotator.get(annotator_id, 0),
            "num_matched_annotations": matched,
            "num_complete_check_annotations": complete,
            "num_missing_annotations": max(expected - matched, 0),
            "completion_fraction": matched / expected if expected else 0.0,
        }
    return progress


def _dimension_progress(
    expected_tasks: Iterable[tuple[str, str, str]],
    started_annotation_keys: set[tuple[str, str, str]],
    matched_annotation_keys: set[tuple[str, str, str]],
    complete_annotation_keys: set[tuple[str, str, str]],
    item_dimensions: dict[tuple[str, str], dict[str, str]],
    *,
    dimension_key: str,
) -> dict[str, dict[str, Any]]:
    expected_annotations: dict[str, int] = defaultdict(int)
    started_annotations: dict[str, int] = defaultdict(int)
    matched_annotations: dict[str, int] = defaultdict(int)
    complete_annotations: dict[str, int] = defaultdict(int)
    item_sets: dict[str, set[tuple[str, str]]] = defaultdict(set)

    for case_id, query_id, _ in expected_tasks:
        item_key = (case_id, query_id)
        dimension_value = item_dimensions.get(item_key, {}).get(dimension_key, "unknown")
        expected_annotations[dimension_value] += 1
        item_sets[dimension_value].add(item_key)

    for case_id, query_id, _ in started_annotation_keys:
        item_key = (case_id, query_id)
        dimension_value = item_dimensions.get(item_key, {}).get(dimension_key, "unknown")
        started_annotations[dimension_value] += 1

    for case_id, query_id, _ in matched_annotation_keys:
        item_key = (case_id, query_id)
        dimension_value = item_dimensions.get(item_key, {}).get(dimension_key, "unknown")
        matched_annotations[dimension_value] += 1

    for case_id, query_id, _ in complete_annotation_keys:
        item_key = (case_id, query_id)
        dimension_value = item_dimensions.get(item_key, {}).get(dimension_key, "unknown")
        complete_annotations[dimension_value] += 1

    progress: dict[str, dict[str, Any]] = {}
    for dimension_value in sorted(expected_annotations):
        expected = expected_annotations[dimension_value]
        matched = matched_annotations.get(dimension_value, 0)
        complete = complete_annotations.get(dimension_value, 0)
        progress[dimension_value] = {
            "num_template_items": len(item_sets.get(dimension_value, set())),
            "num_expected_annotations": expected,
            "num_started_annotations": started_annotations.get(dimension_value, 0),
            "num_matched_annotations": matched,
            "num_complete_check_annotations": complete,
            "num_missing_annotations": max(expected - matched, 0),
            "completion_fraction": matched / expected if expected else 0.0,
        }
    return progress


def _task_file_annotation_progress(
    manifest: dict[str, Any],
    manifest_dir: Path,
    errors: list[str],
) -> dict[str, set[tuple[str, str, str]]]:
    task_files = manifest.get("task_files")
    if not isinstance(task_files, dict) or not task_files:
        errors.append("task manifest task_files must be a non-empty object")
        return {
            "started_annotation_keys": set(),
            "completed_annotation_keys": set(),
        }

    started_annotation_keys: set[tuple[str, str, str]] = set()
    completed_annotation_keys: set[tuple[str, str, str]] = set()
    seen: set[tuple[str, str, str]] = set()

    for annotator_id, raw_path in sorted(task_files.items()):
        annotator = str(annotator_id)
        path = _resolve_task_path(manifest_dir, str(raw_path))
        if not path.exists():
            errors.append(f"task file does not exist for annotator {annotator!r}: {path}")
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path}:{line_number}: invalid JSONL task record: {exc}")
                    continue
                if not isinstance(record, dict):
                    errors.append(f"{path}:{line_number}: task record must be an object")
                    continue
                case_id = str(record.get("case_id", ""))
                query_id = str(record.get("query_id", ""))
                task_annotator = str(record.get("annotator_id", ""))
                if not case_id or not query_id or not task_annotator:
                    errors.append(f"{path}:{line_number}: task record must include case_id, query_id, and annotator_id")
                    continue
                if task_annotator != annotator:
                    errors.append(
                        f"{path}:{line_number}: task annotator_id {task_annotator!r} does not match manifest key {annotator!r}"
                    )
                key = (case_id, query_id, task_annotator)
                if key in seen:
                    continue
                seen.add(key)

                raw_checks = record.get("checks")
                if raw_checks is None:
                    raw_checks = {}
                if not isinstance(raw_checks, dict):
                    errors.append(f"{path}:{line_number}: checks must be an object")
                    continue
                try:
                    applicable_checks = tuple(_resolve_applicable_checks(record, path, line_number))
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                applicable_set = set(applicable_checks)

                has_started = False
                is_completed = True
                for field in AUDIT_CHECK_FIELDS:
                    try:
                        value = _binary_value(raw_checks.get(field, record.get(field)), path, line_number, field)
                    except ValueError as exc:
                        errors.append(str(exc))
                        is_completed = False
                        continue
                    if field in applicable_set:
                        if value is not None:
                            has_started = True
                        else:
                            is_completed = False
                    elif value is not None:
                        errors.append(f"{path}:{line_number}: non-applicable task check {field!r} must be null")
                        is_completed = False
                if record.get("notes") not in (None, ""):
                    has_started = True

                annotation_status = str(record.get("annotation_status", "") or "")
                if annotation_status == "completed" and not is_completed:
                    errors.append(f"{path}:{line_number}: annotation_status is completed but checks are incomplete")
                if has_started:
                    started_annotation_keys.add(key)
                if is_completed:
                    completed_annotation_keys.add(key)

    return {
        "started_annotation_keys": started_annotation_keys,
        "completed_annotation_keys": completed_annotation_keys,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _template_file_digests(raw_template_paths: Iterable[str | Path]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for raw_path in raw_template_paths:
        key = str(raw_path)
        path = Path(raw_path)
        if not path.exists():
            raise ValueError(f"template file does not exist: {path}")
        digests[key] = _file_sha256(path)
    return digests


def _task_identity_digest_from_records(records: Iterable[dict[str, Any]]) -> str:
    normalized_lines = [
        json.dumps(_task_identity_record(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    payload = ("\n".join(normalized_lines) + ("\n" if normalized_lines else "")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _task_identity_digest_from_path(path: Path) -> str:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL task record: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: task record must be an object")
            records.append(record)
    return _task_identity_digest_from_records(records)


def _task_identity_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in record.items():
        if key in {"checks", "annotation_status", "notes"}:
            continue
        if key == "applicable_checks" and isinstance(value, list):
            normalized[key] = sorted(str(item) for item in value)
            continue
        normalized[key] = value
    return normalized


def _verify_task_manifest_fingerprints(
    manifest: dict[str, Any],
    manifest_dir: Path,
    errors: list[str],
) -> None:
    raw_template_files = manifest.get("template_files")
    template_file_digests = manifest.get("template_file_digests")
    if isinstance(raw_template_files, list) and raw_template_files:
        expected_template_keys = sorted(str(path) for path in raw_template_files)
        if not isinstance(template_file_digests, dict) or not template_file_digests:
            errors.append("task manifest template_file_digests must be a non-empty object")
        else:
            actual_template_keys = sorted(str(key) for key in template_file_digests)
            if actual_template_keys != expected_template_keys:
                errors.append("task manifest template_file_digests keys do not match template_files")
            for raw_path in expected_template_keys:
                resolved_path = _resolve_task_path(manifest_dir, raw_path)
                if not resolved_path.exists():
                    errors.append(f"template file does not exist: {resolved_path}")
                    continue
                expected_digest = template_file_digests.get(raw_path)
                if not isinstance(expected_digest, str) or not expected_digest.strip():
                    errors.append(f"task manifest template_file_digests is missing digest for template {raw_path!r}")
                    continue
                if _file_sha256(resolved_path) != expected_digest:
                    errors.append(f"template file sha256 mismatch: {resolved_path}")

    task_files = manifest.get("task_files")
    task_identity_digests = manifest.get("task_identity_digests")
    if isinstance(task_files, dict) and task_files:
        expected_annotators = sorted(str(annotator_id) for annotator_id in task_files)
        if not isinstance(task_identity_digests, dict) or not task_identity_digests:
            errors.append("task manifest task_identity_digests must be a non-empty object")
        else:
            actual_annotators = sorted(str(key) for key in task_identity_digests)
            if actual_annotators != expected_annotators:
                errors.append("task manifest task_identity_digests keys do not match task_files annotators")
            for annotator_id, raw_path in sorted(task_files.items()):
                resolved_path = _resolve_task_path(manifest_dir, str(raw_path))
                if not resolved_path.exists():
                    continue
                expected_digest = task_identity_digests.get(str(annotator_id))
                if not isinstance(expected_digest, str) or not expected_digest.strip():
                    errors.append(
                        f"task manifest task_identity_digests is missing digest for annotator {annotator_id!r}"
                    )
                    continue
                try:
                    actual_digest = _task_identity_digest_from_path(resolved_path)
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                if actual_digest != expected_digest:
                    errors.append(f"task identity sha256 mismatch for annotator {annotator_id!r}: {resolved_path}")


def _format_task_key(key: tuple[str, str, str]) -> str:
    return f"{key[0]}/{key[1]}/{key[2]}"


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value).strip("._") or "annotator"
