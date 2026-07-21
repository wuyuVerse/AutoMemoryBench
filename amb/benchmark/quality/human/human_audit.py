"""Completed human-audit verification for release manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable
import zipfile

from amb.benchmark.quality.annotation import (
    AUDIT_CHECK_DEFINITIONS,
    AUDIT_CHECK_FIELDS,
    AuditAnnotation,
    apply_human_audit_annotator_packet,
    _safe_filename,
    _sheet_binary_value,
    _sheet_notes_value,
    _expected_tasks_from_task_manifest,
    _task_identity_digest_from_records,
    _task_identity_digest_from_path,
    _template_file_digests,
    _resolve_task_path,
    _resolve_annotator_packet_annotator_id,
    _resolve_annotator_packet_root,
    _single_annotator_packet_file,
    compute_agreement,
    load_audit_annotations,
    merge_completed_human_audit_tasks,
    summarize_human_audit_progress,
    write_human_audit_annotation_sheets,
)
from amb.benchmark.schemas.io import read_json, write_json

ItemKey = tuple[str, str]
HUMAN_AUDIT_ATTESTATION_SCHEMA_VERSION = "amst-human-audit-attestation-v1"
HUMAN_AUDIT_VERIFICATION_SCHEMA_VERSION = "amst-human-audit-verification-v1"
HUMAN_AUDIT_ADJUDICATION_PACKAGE_SCHEMA_VERSION = "amst-human-audit-adjudication-package-v1"
HUMAN_AUDIT_ADJUDICATION_DECISION_SCHEMA_VERSION = "amst-human-audit-adjudication-decision-v1"
HUMAN_AUDIT_ANNOTATOR_PACKET_SCHEMA_VERSION = "amst-human-audit-annotator-packet-v1"
HUMAN_AUDIT_ADJUDICATION_PACKET_SCHEMA_VERSION = "amst-human-audit-adjudication-packet-v1"
HUMAN_AUDIT_RETURN_INBOX_STATE_SCHEMA_VERSION = "amst-human-audit-return-inbox-state-v1"
HUMAN_AUDIT_BUNDLE_SANDBOX_SIGNED_AT = "2026-05-13T00:00:00Z"
HUMAN_AUDIT_BUNDLE_SANDBOX_ANNOTATION_GUIDELINE = "docs/annotation_guideline.md"
HUMAN_AUDIT_BUNDLE_SANDBOX_ADJUDICATION_POLICY = "Disagreements are adjudicated after double annotation."
HUMAN_AUDIT_CLI_PREFIX = "PYTHONPATH=. python -m agent_memory_benchmark"
HUMAN_AUDIT_OPERATOR_WATCH_INTERVAL_S = "120"
HUMAN_AUDIT_WATCH_STOP_EXIT_CODES = {
    "max_iterations": 0,
    "rejected_returns": 2,
    "ready": 3,
}
HUMAN_AUDIT_ADJUDICATION_SHEET_COLUMNS = (
    "case_id",
    "query_id",
    "adjudicator_id",
    "adjudication_status",
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
    "consensus_checks_json",
    "disagreement_fields_json",
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


def verify_manifest_human_audit(manifest_path: str | Path) -> dict[str, Any]:
    """Verify that a release manifest's completed human audit is reproducible."""

    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    manifest_dir = manifest_file.parent
    audit_plan = manifest.get("audit_plan", {})
    errors: list[str] = []

    if audit_plan.get("human_audit_status") != "completed":
        errors.append("human_audit_status is not completed")

    template_paths = _manifest_template_paths(audit_plan, manifest_dir)
    if not template_paths:
        errors.append("audit template file is missing")

    raw_annotations = audit_plan.get("audit_annotations_file")
    annotations_path = _resolve_path(str(raw_annotations), manifest_dir) if raw_annotations else None
    if annotations_path is None:
        errors.append("audit_annotations_file is missing")
        return _verification_report(template_paths, None, None, errors)
    raw_task_manifest = audit_plan.get("audit_task_manifest_file")
    task_manifest_path = _resolve_path(str(raw_task_manifest), manifest_dir) if raw_task_manifest else None
    if task_manifest_path is None:
        errors.append("audit_task_manifest_file is missing")
    raw_attestation = audit_plan.get("annotator_attestation_file")
    attestation_path = _resolve_path(str(raw_attestation), manifest_dir) if raw_attestation else None
    if attestation_path is None:
        errors.append("annotator_attestation_file is missing")
    raw_adjudication = audit_plan.get("audit_adjudication_file")
    adjudication_path = _resolve_path(str(raw_adjudication), manifest_dir) if raw_adjudication else None

    report = verify_completed_human_audit(
        template_paths,
        annotations_path,
        task_manifest_path=task_manifest_path,
        annotator_attestation_path=attestation_path,
        agreement_metrics=audit_plan.get("agreement_metrics"),
        adjudication_path=adjudication_path,
        require_adjudication=True,
        benchmark_id=str(manifest.get("benchmark_id") or ""),
        manifest_path=manifest_file,
    )
    report["errors"] = errors + report["errors"]
    report["ok"] = not report["errors"]
    return report


def verify_completed_human_audit(
    template_paths: Iterable[str | Path],
    annotations_path: str | Path,
    *,
    task_manifest_path: str | Path | None = None,
    annotator_attestation_path: str | Path | None = None,
    agreement_metrics: dict[str, Any] | None = None,
    adjudication_path: str | Path | None = None,
    require_adjudication: bool = False,
    benchmark_id: str | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify double annotations against generated audit templates."""

    template_files = tuple(Path(path) for path in template_paths)
    annotation_file = Path(annotations_path)
    task_manifest_file = Path(task_manifest_path) if task_manifest_path is not None else None
    attestation_file = Path(annotator_attestation_path) if annotator_attestation_path is not None else None
    adjudication_file = Path(adjudication_path) if adjudication_path is not None else None
    errors: list[str] = []
    expected_items = _load_template_items(template_files, errors)
    expected_applicable_counts = _expected_applicable_counts(expected_items)

    try:
        annotations = load_audit_annotations(annotation_file)
    except Exception as exc:  # noqa: BLE001 - report user-facing annotation parsing errors
        errors.append(f"could not load audit annotations: {exc}")
        return _verification_report(
            template_files,
            annotation_file,
            None,
            errors,
            expected_items=expected_items,
            task_manifest_path=task_manifest_file,
            annotator_attestation_path=attestation_file,
            adjudication_path=adjudication_file,
        )

    grouped = _group_annotations(annotations)
    _check_annotation_coverage(expected_items, grouped, errors)
    if task_manifest_file is not None:
        errors.extend(_verify_task_manifest_template_files(task_manifest_file, template_files))
        errors.extend(_verify_task_manifest_progress(task_manifest_file, annotation_file))
    if attestation_file is not None:
        errors.extend(_verify_annotator_attestation(task_manifest_file, annotation_file, attestation_file, annotations))
    agreement = _compute_agreement_or_error(annotations, errors)
    semantic_checks = _semantic_checks(expected_items, grouped)
    if agreement is not None:
        _check_agreement_completeness(agreement, expected_applicable_counts, errors)
        if agreement_metrics is not None and not _agreement_metrics_match(agreement, agreement_metrics):
            errors.append("agreement_metrics do not match recomputed annotation agreement")
    elif agreement_metrics is not None:
        errors.append("agreement_metrics are present but annotations could not be recomputed")
    disagreement_summary = summarize_disagreement_summary(expected_items, grouped)
    adjudication_summary = _verify_adjudication_artifact(
        _disagreement_items(expected_items, grouped),
        adjudication_file,
        require_adjudication=require_adjudication,
        errors=errors,
    )

    return _verification_report(
        template_files,
        annotation_file,
        agreement,
        errors,
        expected_items=expected_items,
        annotations=annotations,
        task_manifest_path=task_manifest_file,
        annotator_attestation_path=attestation_file,
        semantic_checks=semantic_checks,
        disagreement_summary=disagreement_summary,
        adjudication_path=adjudication_file,
        adjudication_summary=adjudication_summary,
        benchmark_id=benchmark_id,
        manifest_path=Path(manifest_path) if manifest_path is not None else None,
    )


def finalize_human_audit_manifest(
    manifest_path: str | Path,
    annotations_path: str | Path,
    *,
    task_manifest_path: str | Path,
    annotator_attestation_path: str | Path,
    agreement_output: str | Path | None = None,
    adjudication_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute agreement, verify coverage, and mark a manifest human audit completed."""

    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    manifest_dir = manifest_file.parent
    audit_plan = manifest.setdefault("audit_plan", {})
    template_paths = _manifest_template_paths(audit_plan, manifest_dir)
    annotation_file = Path(annotations_path)
    task_manifest_file = Path(task_manifest_path)
    attestation_file = Path(annotator_attestation_path)

    verification = verify_completed_human_audit(
        template_paths,
        annotation_file,
        task_manifest_path=task_manifest_file,
        annotator_attestation_path=attestation_file,
        adjudication_path=adjudication_path,
        require_adjudication=True,
    )
    if verification["errors"]:
        raise ValueError("completed human audit verification failed: " + "; ".join(verification["errors"][:5]))

    agreement_metrics = verification["agreement_metrics"]
    audit_plan["human_audit_status"] = "completed"
    audit_plan["audit_annotations_file"] = _relative_or_absolute(annotation_file, manifest_dir)
    audit_plan["audit_task_manifest_file"] = _relative_or_absolute(task_manifest_file, manifest_dir)
    audit_plan["annotator_attestation_file"] = _relative_or_absolute(attestation_file, manifest_dir)
    if adjudication_path is not None:
        audit_plan["audit_adjudication_file"] = _relative_or_absolute(Path(adjudication_path), manifest_dir)
    audit_plan["agreement_metrics"] = agreement_metrics
    if agreement_output is not None:
        output = Path(agreement_output)
        write_json(output, agreement_metrics)
        audit_plan["agreement_metrics_file"] = _relative_or_absolute(output, manifest_dir)

    write_json(manifest_file, manifest)
    return verify_manifest_human_audit(manifest_file)


def build_human_audit_attestation(
    task_manifest_path: str | Path,
    annotations_path: str | Path,
    output_path: str | Path,
    *,
    annotation_guideline: str,
    adjudication_policy: str,
    signed_at: str,
) -> dict[str, Any]:
    """Build a reproducible annotator attestation from completed audit artifacts."""

    task_manifest_file = Path(task_manifest_path)
    annotation_file = Path(annotations_path)
    output_file = Path(output_path)

    progress = summarize_human_audit_progress(task_manifest_file, annotations_path=annotation_file)
    if progress.get("ready_for_finalize") is not True:
        errors = "; ".join(str(error) for error in progress.get("errors", [])[:5])
        raise ValueError(
            "cannot build human-audit attestation before annotations are ready for finalize"
            + (f": {errors}" if errors else "")
        )

    annotations = load_audit_annotations(annotation_file)
    annotator_ids = sorted({annotation.annotator_id for annotation in annotations})
    if not annotator_ids:
        raise ValueError("cannot build human-audit attestation without completed annotations")
    signed_at_value = str(signed_at).strip()
    if not signed_at_value:
        raise ValueError("signed_at is required")
    guideline_value = str(annotation_guideline).strip()
    if not guideline_value:
        raise ValueError("annotation_guideline is required")
    adjudication_value = str(adjudication_policy).strip()
    if not adjudication_value:
        raise ValueError("adjudication_policy is required")

    attestation = {
        "attestation_schema_version": HUMAN_AUDIT_ATTESTATION_SCHEMA_VERSION,
        "task_manifest_file": str(task_manifest_file),
        "annotations_file": str(annotation_file),
        "annotators": [
            {
                "annotator_id": annotator_id,
                "signed_at": signed_at_value,
                "independent_annotation": True,
                "conflict_of_interest": False,
            }
            for annotator_id in annotator_ids
        ],
        "protocol": {
            "annotation_guideline": guideline_value,
            "adjudication_policy": adjudication_value,
        },
    }
    write_json(output_file, attestation)
    errors = _verify_annotator_attestation(task_manifest_file, annotation_file, output_file, annotations)
    if errors:
        raise ValueError("generated human-audit attestation failed verification: " + "; ".join(errors[:5]))
    return attestation


def build_human_audit_sandbox_sample(
    task_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    num_items: int = 2,
    annotation_guideline: str,
    adjudication_policy: str,
    signed_at: str,
) -> dict[str, Any]:
    """Create a minimal finalize-ready human-audit sandbox from a real task package.

    This helper is for pipeline validation only. It never mutates a release manifest.
    """

    task_manifest_file = Path(task_manifest_path)
    sandbox_dir = Path(output_dir)
    manifest = read_json(task_manifest_file)
    if not isinstance(manifest, dict):
        raise ValueError("human-audit task manifest must be a JSON object")
    if num_items <= 0:
        raise ValueError("num_items must be positive")

    errors: list[str] = []
    expected_tasks = sorted(_expected_tasks_from_task_manifest(manifest, task_manifest_file.parent, errors))
    if errors:
        raise ValueError("cannot build human-audit sandbox sample: " + "; ".join(errors[:5]))
    if not expected_tasks:
        raise ValueError("cannot build human-audit sandbox sample from an empty task manifest")

    selected_items = sorted({(case_id, query_id) for case_id, query_id, _ in expected_tasks})[:num_items]
    if not selected_items:
        raise ValueError("no audit items were selected for the sandbox sample")
    selected_item_keys = set(selected_items)

    task_files = manifest.get("task_files")
    if not isinstance(task_files, dict) or not task_files:
        raise ValueError("task manifest task_files must be a non-empty object")

    sandbox_dir.mkdir(parents=True, exist_ok=True)
    completed_task_dir = sandbox_dir / "tasks"
    completed_task_dir.mkdir(parents=True, exist_ok=True)
    completed_task_files: dict[str, str] = {}
    completed_task_records: dict[str, list[dict[str, Any]]] = {}
    for annotator_id, raw_path in sorted(task_files.items()):
        source_path = _resolve_task_path(task_manifest_file.parent, str(raw_path))
        if not source_path.exists():
            raise ValueError(f"task file does not exist for annotator {annotator_id!r}: {source_path}")
        selected_records: list[dict[str, Any]] = []
        with source_path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                item_key = (str(record.get("case_id", "")), str(record.get("query_id", "")))
                if item_key not in selected_item_keys:
                    continue
                task = dict(record)
                task["annotation_status"] = "completed"
                applicable = set(_template_applicable_checks(task))
                task["checks"] = {field: (True if field in applicable else None) for field in AUDIT_CHECK_FIELDS}
                task.setdefault("notes", None)
                selected_records.append(task)
        if len(selected_records) != len(selected_items):
            raise ValueError(f"annotator {annotator_id!r} task file does not cover all selected sandbox items")
        target_path = completed_task_dir / f"{annotator_id}.jsonl"
        target_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in selected_records) + "\n",
            encoding="utf-8",
        )
        completed_task_files[str(annotator_id)] = str(target_path)
        completed_task_records[str(annotator_id)] = selected_records

    selected_template_paths = _materialize_selected_template_files(
        manifest.get("template_files", []),
        task_manifest_file.parent,
        selected_item_keys,
        sandbox_dir / "templates",
    )
    sandbox_template_files = [str(path) for path in selected_template_paths]

    sandbox_manifest = {
        "annotation_task_schema_version": manifest.get("annotation_task_schema_version", "amst-human-audit-task-package-v1"),
        "status": "sandbox_sample_completed",
        "template_files": sandbox_template_files,
        "template_file_digests": _template_file_digests(sandbox_template_files),
        "task_files": completed_task_files,
        "task_identity_digests": {
            annotator_id: _task_identity_digest_from_records(rows)
            for annotator_id, rows in sorted(completed_task_records.items())
        },
        "num_template_items": len(selected_items),
        "num_annotators": len(completed_task_files),
        "expected_annotations": len(selected_items) * len(completed_task_files),
        "annotator_ids": sorted(completed_task_files),
        "checks": list(AUDIT_CHECK_FIELDS),
        "completion_status": "completed_annotations_merged",
        "sandbox_source_task_manifest_file": str(task_manifest_file),
    }
    sandbox_manifest = _localize_human_audit_task_manifest_payload(
        sandbox_manifest,
        source_bases=(sandbox_dir,),
        target_base=sandbox_dir,
    )
    sandbox_task_manifest_path = sandbox_dir / "task_manifest.json"
    write_json(sandbox_task_manifest_path, sandbox_manifest)

    annotations_path = sandbox_dir / "completed_annotations.jsonl"
    merge_summary = merge_completed_human_audit_tasks(sandbox_task_manifest_path, annotations_path)
    merge_summary = _localize_human_audit_merge_summary_payload(
        merge_summary,
        source_bases=(sandbox_dir,),
        target_base=sandbox_dir,
    )
    attestation_path = sandbox_dir / "attestation.json"
    attestation = build_human_audit_attestation(
        sandbox_task_manifest_path,
        annotations_path,
        attestation_path,
        annotation_guideline=annotation_guideline,
        adjudication_policy=adjudication_policy,
        signed_at=signed_at,
    )
    attestation = _localize_human_audit_attestation_payload(
        attestation,
        source_bases=(sandbox_dir, task_manifest_file.parent),
        target_base=sandbox_dir,
    )
    write_json(attestation_path, attestation)
    agreement_path = sandbox_dir / "agreement.json"
    agreement = compute_agreement(load_audit_annotations(annotations_path))
    write_json(agreement_path, agreement)

    verification = verify_completed_human_audit(
        selected_template_paths,
        annotations_path,
        task_manifest_path=sandbox_task_manifest_path,
        annotator_attestation_path=attestation_path,
        agreement_metrics=agreement,
    )
    verification_path = sandbox_dir / "verification_report.json"
    verification = _localize_human_audit_verification_payload(
        verification,
        source_bases=(sandbox_dir, task_manifest_file.parent),
        target_base=sandbox_dir,
    )
    write_json(verification_path, verification)
    if verification["ok"] is not True:
        raise ValueError("generated human-audit sandbox sample failed verification")

    return {
        "schema_version": "amst-human-audit-sandbox-sample-v1",
        "source_task_manifest_file": str(task_manifest_file),
        "sandbox_task_manifest_file": str(sandbox_task_manifest_path),
        "completed_annotations_file": str(annotations_path),
        "attestation_file": str(attestation_path),
        "agreement_file": str(agreement_path),
        "verification_report_file": str(verification_path),
        "merge_summary": merge_summary,
        "attestation": attestation,
        "agreement_metrics": agreement,
        "verification": verification,
        "selected_items": [f"{case_id}/{query_id}" for case_id, query_id in selected_items],
        "num_items": len(selected_items),
        "num_annotators": len(completed_task_files),
    }


def summarize_human_audit_disagreements(
    task_manifest_path: str | Path,
    annotations_path: str | Path,
) -> dict[str, Any]:
    """Summarize item-level human-audit disagreements after double annotation."""

    task_manifest_file = Path(task_manifest_path)
    annotation_file = Path(annotations_path)
    manifest = read_json(task_manifest_file)
    if not isinstance(manifest, dict):
        raise ValueError("human-audit task manifest must be a JSON object")

    template_files = manifest.get("template_files")
    if not isinstance(template_files, list) or not template_files:
        raise ValueError("human-audit task manifest must include non-empty template_files")
    template_paths = tuple(_resolve_path(str(path), task_manifest_file.parent) for path in template_files)

    verification = verify_completed_human_audit(
        template_paths,
        annotation_file,
        task_manifest_path=task_manifest_file,
    )
    if verification["ok"] is not True:
        raise ValueError("cannot summarize human-audit disagreements before completed annotations verify cleanly")

    errors: list[str] = []
    expected_items = _load_template_items(template_paths, errors)
    if errors:
        raise ValueError("cannot summarize human-audit disagreements: " + "; ".join(errors[:5]))
    annotations = load_audit_annotations(annotation_file)
    grouped = _group_annotations(annotations)
    disagreement_summary = summarize_disagreement_summary(expected_items, grouped)
    disagreement_items = _disagreement_items(expected_items, grouped)
    return {
        "schema_version": HUMAN_AUDIT_ADJUDICATION_PACKAGE_SCHEMA_VERSION,
        "task_manifest_file": str(task_manifest_file),
        "annotations_file": str(annotation_file),
        "template_files": [str(path) for path in template_paths],
        "adjudication_required": bool(disagreement_items),
        **disagreement_summary,
        "items_with_disagreements": disagreement_items,
    }


def build_human_audit_adjudication_package(
    task_manifest_path: str | Path,
    annotations_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create a machine-readable adjudication package for disagreement items only."""

    summary = summarize_human_audit_disagreements(task_manifest_path, annotations_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tasks_path = output / "adjudication_tasks.jsonl"
    disagreement_items = list(summary["items_with_disagreements"])
    tasks_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in disagreement_items) + ("\n" if disagreement_items else ""),
        encoding="utf-8",
    )

    package_manifest = {
        "schema_version": HUMAN_AUDIT_ADJUDICATION_PACKAGE_SCHEMA_VERSION,
        "status": "pending_adjudication" if disagreement_items else "not_required",
        "task_manifest_file": summary["task_manifest_file"],
        "annotations_file": summary["annotations_file"],
        "template_files": summary["template_files"],
        "num_items": summary["num_items"],
        "num_disagreement_items": summary["num_disagreement_items"],
        "num_disagreement_fields": summary["num_disagreement_fields"],
        "num_tied_fields": summary["num_tied_fields"],
        "fields_with_disagreements": dict(summary["fields_with_disagreements"]),
        "adjudication_required": summary["adjudication_required"],
        "adjudication_tasks_file": str(tasks_path),
    }
    manifest_path = output / "adjudication_manifest.json"
    summary_path = output / "adjudication_summary.json"
    write_json(manifest_path, package_manifest)
    write_json(summary_path, summary)
    package_manifest["adjudication_manifest_file"] = str(manifest_path)
    package_manifest["adjudication_summary_file"] = str(summary_path)
    package_manifest["task_file_sha256"] = _file_sha256(tasks_path)
    return package_manifest


def write_human_audit_adjudication_sheet(
    adjudication_tasks_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Export a spreadsheet-friendly CSV sheet for a pending adjudication package."""

    tasks_file = Path(adjudication_tasks_path)
    records = _load_adjudication_task_records(tasks_file)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HUMAN_AUDIT_ADJUDICATION_SHEET_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(_adjudication_sheet_row(record))
    return {
        "adjudication_sheet_schema_version": "amst-human-audit-adjudication-sheet-v1",
        "adjudication_tasks_file": str(tasks_file),
        "sheet_file": str(output_file),
        "num_rows": len(records),
        "task_file_sha256": _file_sha256(tasks_file),
    }


def apply_human_audit_adjudication_sheet(
    adjudication_tasks_path: str | Path,
    sheet_path: str | Path,
    *,
    adjudicator_id: str | None = None,
) -> dict[str, Any]:
    """Apply an adjudication CSV sheet back into its canonical JSONL task file."""

    tasks_file = Path(adjudication_tasks_path)
    records = _load_adjudication_task_records(tasks_file)
    record_map = {(str(record["case_id"]), str(record["query_id"])): record for record in records}
    expected_keys = set(record_map)
    sheet_file = Path(sheet_path)

    with sheet_file.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("adjudication sheet CSV must include a header row")
        required_columns = ("case_id", "query_id", "adjudicator_id", *AUDIT_CHECK_FIELDS, "notes")
        missing_columns = [column for column in required_columns if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"adjudication sheet CSV is missing required columns: {missing_columns}")

        resolved_adjudicator: str | None = None
        seen_keys: set[ItemKey] = set()
        started_rows = 0
        completed_rows = 0
        updated_records: dict[ItemKey, dict[str, Any]] = {}
        for row_number, row in enumerate(reader, start=2):
            case_id = str(row.get("case_id", "")).strip()
            query_id = str(row.get("query_id", "")).strip()
            if not case_id or not query_id:
                raise ValueError(f"{sheet_file}:{row_number}: case_id and query_id are required")
            key = (case_id, query_id)
            if key in seen_keys:
                raise ValueError(f"{sheet_file}:{row_number}: duplicate adjudication row for {_format_item(key)}")
            seen_keys.add(key)
            if key not in record_map:
                raise ValueError(f"{sheet_file}:{row_number}: adjudication row does not match any disagreement item: {_format_item(key)}")

            row_adjudicator = str(row.get("adjudicator_id", "")).strip()
            if adjudicator_id is not None and str(adjudicator_id).strip():
                requested = str(adjudicator_id).strip()
                if row_adjudicator and row_adjudicator != requested:
                    raise ValueError(
                        f"{sheet_file}:{row_number}: adjudicator_id {row_adjudicator!r} does not match requested adjudicator {requested!r}"
                    )
                row_adjudicator = requested
            if not row_adjudicator:
                raise ValueError(f"{sheet_file}:{row_number}: adjudicator_id is required")
            if resolved_adjudicator is None:
                resolved_adjudicator = row_adjudicator
            elif row_adjudicator != resolved_adjudicator:
                raise ValueError(
                    f"{sheet_file}:{row_number}: adjudicator_id {row_adjudicator!r} does not match prior adjudicator {resolved_adjudicator!r}"
                )

            record = dict(record_map[key])
            target_fields = set(_record_adjudication_target_fields(record))
            updated_checks: dict[str, bool | None] = {}
            any_started = False
            all_complete = True
            for field in AUDIT_CHECK_FIELDS:
                value = _sheet_binary_value(row.get(field), sheet_file, row_number, field)
                if field not in target_fields and value is not None:
                    raise ValueError(f"{sheet_file}:{row_number}: non-target adjudication field {field!r} must be blank")
                if field in target_fields and value is None:
                    all_complete = False
                if field in target_fields and value is not None:
                    any_started = True
                    updated_checks[field] = value
            notes_value = _sheet_notes_value(row.get("notes"))
            if notes_value is not None:
                any_started = True

            record["adjudicator_id"] = row_adjudicator
            record["adjudication_checks"] = updated_checks
            record["adjudication_status"] = "completed" if all_complete else "pending"
            record["notes"] = notes_value
            updated_records[key] = record
            if any_started:
                started_rows += 1
            if all_complete:
                completed_rows += 1

    missing_keys = sorted(expected_keys - set(updated_records))
    if missing_keys:
        raise ValueError(
            "adjudication sheet is missing disagreement rows: "
            + ", ".join(_format_item(key) for key in missing_keys[:10])
        )

    ordered_records = []
    for record in records:
        key = (str(record["case_id"]), str(record["query_id"]))
        ordered_records.append(updated_records[key])
    tasks_file.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in ordered_records) + "\n",
        encoding="utf-8",
    )
    return {
        "adjudication_sheet_apply_schema_version": "amst-human-audit-adjudication-sheet-apply-v1",
        "adjudication_tasks_file": str(tasks_file),
        "sheet_file": str(sheet_file),
        "adjudicator_id": resolved_adjudicator,
        "num_rows": len(ordered_records),
        "num_started_rows": started_rows,
        "num_completed_rows": completed_rows,
        "status": "completed" if completed_rows == len(ordered_records) else "updated",
    }


def apply_human_audit_adjudication_packet(
    bundle_dir: str | Path,
    packet_path: str | Path,
    *,
    adjudicator_id: str | None = None,
    reconcile_when_ready: bool = False,
    signed_at: str | None = None,
    annotation_guideline: str | None = None,
    adjudication_policy: str | None = None,
) -> dict[str, Any]:
    """Apply a returned adjudication packet directory or zip archive back into the canonical bundle."""

    bundle_root = Path(bundle_dir)
    bundle_manifest_path = bundle_root / "bundle_manifest.json"
    bundle_manifest = read_json(bundle_manifest_path)
    if not isinstance(bundle_manifest, dict):
        raise ValueError("bundle_manifest.json must be a JSON object")

    adjudication_package = bundle_manifest.get("adjudication_package")
    if not isinstance(adjudication_package, dict) or not adjudication_package:
        raise ValueError("bundle manifest does not include a pending adjudication_package")
    tasks_value = adjudication_package.get("adjudication_tasks_file")
    if not isinstance(tasks_value, str) or not tasks_value:
        raise ValueError("adjudication_package is missing adjudication_tasks_file")
    adjudication_tasks_path = _resolve_path(tasks_value, bundle_root)
    if not adjudication_tasks_path.exists():
        raise ValueError(f"adjudication tasks file does not exist: {adjudication_tasks_path}")

    expected_digest = str(adjudication_package.get("task_file_sha256") or _file_sha256(adjudication_tasks_path))
    source = Path(packet_path)
    if not source.exists():
        raise ValueError(f"adjudication packet path does not exist: {source}")

    packet_type = "directory"
    packet_root: Path
    packet_sheet_file: Path
    packet_task_file: Path
    packet_manifest: dict[str, Any] | None = None
    with TemporaryDirectory() as temp_dir:
        if source.is_dir():
            packet_root = _resolve_adjudication_packet_root(source)
        elif source.is_file() and source.suffix.lower() == ".zip":
            packet_type = "zip"
            extract_root = Path(temp_dir)
            try:
                with zipfile.ZipFile(source) as archive:
                    archive.extractall(extract_root)
            except zipfile.BadZipFile as exc:
                raise ValueError(f"adjudication packet archive is invalid: {source}") from exc
            packet_root = _resolve_adjudication_packet_root(extract_root)
        else:
            raise ValueError("adjudication packet must be a directory or .zip archive")

        packet_manifest = _load_optional_adjudication_packet_manifest(packet_root)
        packet_sheet_file = _single_adjudication_packet_file(packet_root, "*.csv", "adjudication sheet")
        packet_task_file = _single_adjudication_packet_file(packet_root, "*.jsonl", "adjudication task file")
        actual_digest = _file_sha256(packet_task_file)
        if actual_digest != expected_digest:
            raise ValueError(
                "adjudication packet task digest mismatch: "
                f"expected={expected_digest} actual={actual_digest}"
            )
        _validate_adjudication_packet_manifest(
            packet_manifest,
            packet_root,
            packet_sheet_file,
            packet_task_file,
            actual_digest,
        )
        apply_summary = apply_human_audit_adjudication_sheet(
            adjudication_tasks_path,
            packet_sheet_file,
            adjudicator_id=adjudicator_id,
        )

    bundle_manifest["adjudication_file"] = _relative_or_absolute(adjudication_tasks_path, bundle_root)
    refreshed_bundle = _refresh_human_audit_bundle_local_state(bundle_root, bundle_manifest)
    progress = refreshed_bundle.get("progress", {})
    reconcile_report: dict[str, Any] | None = None
    reconcile_status = "not_requested"
    if reconcile_when_ready:
        if progress.get("ready_for_finalize"):
            reconcile_report = reconcile_human_audit_evidence_bundle(
                bundle_root,
                signed_at=signed_at,
                annotation_guideline=annotation_guideline,
                adjudication_policy=adjudication_policy,
            )
            reconcile_status = "completed"
        else:
            reconcile_status = "skipped_not_ready"

    return {
        "schema_version": "amst-human-audit-adjudication-packet-apply-v1",
        "bundle_dir": str(bundle_root),
        "packet_path": str(source),
        "packet_type": packet_type,
        "packet_root": str(packet_root),
        "adjudication_tasks_file": str(adjudication_tasks_path),
        "packet_manifest_file": str(packet_root / "packet_manifest.json") if packet_manifest is not None else None,
        "expected_task_file_sha256": expected_digest,
        "verified_task_file_sha256": actual_digest,
        "ready_for_finalize": progress.get("ready_for_finalize"),
        "reconcile_requested": reconcile_when_ready,
        "reconcile_status": reconcile_status,
        "reconcile_report": reconcile_report,
        **apply_summary,
    }


def sync_human_audit_return_inbox(
    bundle_dir: str | Path,
    *,
    annotator_inboxes: Iterable[str | Path] = (),
    adjudication_inboxes: Iterable[str | Path] = (),
    reconcile_when_ready: bool = False,
    signed_at: str | None = None,
    annotation_guideline: str | None = None,
    adjudication_policy: str | None = None,
) -> dict[str, Any]:
    """Discover returned packet files from inbox directories and ingest them into a bundle."""

    bundle_root = Path(bundle_dir)
    bundle_manifest = read_json(bundle_root / "bundle_manifest.json")
    if not isinstance(bundle_manifest, dict):
        raise ValueError("bundle_manifest.json must be a JSON object")
    state_path = _bundle_return_inbox_state_path(bundle_root)
    processed_state = _load_human_audit_return_inbox_state(bundle_root)
    bundle_state_changed = False

    resolved_annotator_inboxes = [Path(path) for path in annotator_inboxes]
    resolved_adjudication_inboxes = [Path(path) for path in adjudication_inboxes]
    return_inbox = bundle_manifest.get("return_inbox")
    if isinstance(return_inbox, dict):
        if not resolved_annotator_inboxes:
            raw_annotator_inbox = return_inbox.get("annotator_inbox")
            if isinstance(raw_annotator_inbox, str) and raw_annotator_inbox:
                resolved_annotator_inboxes = [_resolve_path(raw_annotator_inbox, bundle_root)]
        if not resolved_adjudication_inboxes:
            raw_adjudication_inbox = return_inbox.get("adjudication_inbox")
            if isinstance(raw_adjudication_inbox, str) and raw_adjudication_inbox:
                resolved_adjudication_inboxes = [_resolve_path(raw_adjudication_inbox, bundle_root)]

    (
        annotator_candidates,
        skipped_annotator_candidates,
        invalid_annotator_candidates,
        skipped_invalid_annotator_candidates,
    ) = _discover_annotator_return_packets(
        resolved_annotator_inboxes,
        processed_state=processed_state,
    )
    if invalid_annotator_candidates:
        _record_rejected_annotator_packets(bundle_root, processed_state, invalid_annotator_candidates)
        _write_human_audit_return_inbox_state(bundle_root, processed_state)
        bundle_state_changed = True
    annotator_summary = None
    if annotator_candidates:
        annotator_summary = ingest_human_audit_return_packets(
            bundle_root,
            [entry["packet_path"] for entry in annotator_candidates],
            reconcile_when_ready=reconcile_when_ready,
            signed_at=signed_at,
            annotation_guideline=annotation_guideline,
            adjudication_policy=adjudication_policy,
        )
        _record_processed_annotator_packets(bundle_root, processed_state, annotator_candidates)
        _write_human_audit_return_inbox_state(bundle_root, processed_state)
        bundle_state_changed = True

    bundle_manifest = read_json(bundle_root / "bundle_manifest.json")
    if not isinstance(bundle_manifest, dict):
        raise ValueError("bundle_manifest.json must be a JSON object after annotator ingest")

    adjudication_candidate = None
    adjudication_summary = None
    skipped_adjudication_candidates: list[dict[str, Any]] = []
    invalid_adjudication_candidates: list[dict[str, Any]] = []
    skipped_invalid_adjudication_candidates: list[dict[str, Any]] = []
    if bundle_manifest.get("adjudication_packet") and not bundle_manifest.get("adjudication_file"):
        (
            adjudication_candidate,
            skipped_adjudication_candidates,
            invalid_adjudication_candidates,
            skipped_invalid_adjudication_candidates,
        ) = _discover_adjudication_return_packet(
            resolved_adjudication_inboxes,
            processed_state=processed_state,
        )
        if invalid_adjudication_candidates:
            _record_rejected_adjudication_packets(bundle_root, processed_state, invalid_adjudication_candidates)
            _write_human_audit_return_inbox_state(bundle_root, processed_state)
            bundle_state_changed = True
        if adjudication_candidate is not None:
            adjudication_summary = apply_human_audit_adjudication_packet(
                bundle_root,
                adjudication_candidate["packet_path"],
                reconcile_when_ready=reconcile_when_ready,
                signed_at=signed_at,
                annotation_guideline=annotation_guideline,
                adjudication_policy=adjudication_policy,
            )
            _record_processed_adjudication_packet(bundle_root, processed_state, adjudication_candidate)
            _write_human_audit_return_inbox_state(bundle_root, processed_state)
            bundle_state_changed = True

    if bundle_state_changed:
        current_bundle_manifest = read_json(bundle_root / "bundle_manifest.json")
        if not isinstance(current_bundle_manifest, dict):
            raise ValueError("bundle_manifest.json must be a JSON object before local state refresh")
        _refresh_human_audit_bundle_local_state(bundle_root, current_bundle_manifest)

    final_bundle_manifest = read_json(bundle_root / "bundle_manifest.json")
    if not isinstance(final_bundle_manifest, dict):
        raise ValueError("bundle_manifest.json must be a JSON object after inbox sync")

    progress = final_bundle_manifest.get("progress", {}) if isinstance(final_bundle_manifest.get("progress"), dict) else {}
    next_action = _bundle_recommended_next_refs(final_bundle_manifest)
    return_archive_summary = _return_archive_summary(processed_state)
    return_reject_archive_summary = _return_reject_archive_summary(processed_state)
    return {
        "schema_version": "amst-human-audit-inbox-sync-v1",
        "bundle_dir": str(bundle_root),
        "annotator_inboxes": [str(path) for path in resolved_annotator_inboxes],
        "adjudication_inboxes": [str(path) for path in resolved_adjudication_inboxes],
        "annotator_candidates": [
            {
                "annotator_id": entry["annotator_id"],
                "packet_path": str(entry["packet_path"]),
                "packet_fingerprint": entry["packet_fingerprint"],
            }
            for entry in annotator_candidates
        ],
        "annotator_skipped_processed_candidates": [
            {
                "annotator_id": entry["annotator_id"],
                "packet_path": str(entry["packet_path"]),
                "packet_fingerprint": entry["packet_fingerprint"],
            }
            for entry in skipped_annotator_candidates
        ],
        "annotator_invalid_candidates": [
            {
                "packet_path": str(entry["packet_path"]),
                "packet_fingerprint": entry["packet_fingerprint"],
                "error": entry["error"],
            }
            for entry in invalid_annotator_candidates
        ],
        "annotator_skipped_rejected_candidates": [
            {
                "packet_path": str(entry["packet_path"]),
                "packet_fingerprint": entry["packet_fingerprint"],
                "error": entry["error"],
            }
            for entry in skipped_invalid_annotator_candidates
        ],
        "adjudication_candidate": (
            {
                "packet_path": str(adjudication_candidate["packet_path"]),
                "packet_fingerprint": adjudication_candidate["packet_fingerprint"],
            }
            if adjudication_candidate is not None
            else None
        ),
        "adjudication_skipped_processed_candidates": [
            {
                "packet_path": str(entry["packet_path"]),
                "packet_fingerprint": entry["packet_fingerprint"],
            }
            for entry in skipped_adjudication_candidates
        ],
        "adjudication_invalid_candidates": [
            {
                "packet_path": str(entry["packet_path"]),
                "packet_fingerprint": entry["packet_fingerprint"],
                "error": entry["error"],
            }
            for entry in invalid_adjudication_candidates
        ],
        "adjudication_skipped_rejected_candidates": [
            {
                "packet_path": str(entry["packet_path"]),
                "packet_fingerprint": entry["packet_fingerprint"],
                "error": entry["error"],
            }
            for entry in skipped_invalid_adjudication_candidates
        ],
        "annotator_ingest_summary": annotator_summary,
        "adjudication_apply_summary": adjudication_summary,
        "return_inbox_state_file": str(state_path),
        "processed_return_archives": return_archive_summary,
        "rejected_return_archives": return_reject_archive_summary,
        "rejected_return_summary": _rejected_return_summary(processed_state),
        "ready_for_merge": progress.get("ready_for_merge"),
        "ready_for_finalize": progress.get("ready_for_finalize"),
        "bundle_status": progress.get("status"),
        "recommended_next_commands": final_bundle_manifest.get("recommended_next_commands", []),
        "next_command_id": next_action.get("next_command_id"),
        "next_command": next_action.get("next_command"),
        "next_script_id": next_action.get("next_script_id"),
        "next_script": next_action.get("next_script"),
        "next_script_file": next_action.get("next_script_file"),
    }


def watch_human_audit_return_inbox(
    bundle_dir: str | Path,
    *,
    annotator_inboxes: Iterable[str | Path] = (),
    adjudication_inboxes: Iterable[str | Path] = (),
    reconcile_when_ready: bool = False,
    signed_at: str | None = None,
    annotation_guideline: str | None = None,
    adjudication_policy: str | None = None,
    interval_s: float = 60.0,
    max_iterations: int = 1,
    stop_when_ready: bool = False,
    stop_when_rejected: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Repeatedly sync returned human-audit packets from inboxes into a bundle."""

    if interval_s < 0:
        raise ValueError("interval_s must be non-negative")
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")

    bundle_root = Path(bundle_dir)
    output_file = Path(output_path) if output_path is not None else None
    iteration_count = 0
    last_sync_summary: dict[str, Any] | None = None

    while True:
        iteration_count += 1
        last_sync_summary = sync_human_audit_return_inbox(
            bundle_root,
            annotator_inboxes=annotator_inboxes,
            adjudication_inboxes=adjudication_inboxes,
            reconcile_when_ready=reconcile_when_ready,
            signed_at=signed_at,
            annotation_guideline=annotation_guideline,
            adjudication_policy=adjudication_policy,
        )
        bundle_manifest = read_json(bundle_root / "bundle_manifest.json")
        if not isinstance(bundle_manifest, dict):
            raise ValueError("bundle_manifest.json must be a JSON object after inbox watch iteration")
        progress = bundle_manifest.get("progress", {}) if isinstance(bundle_manifest.get("progress"), dict) else {}
        ready = bool(progress.get("ready_for_merge") or progress.get("ready_for_finalize"))
        rejected_return_summary = (
            last_sync_summary.get("rejected_return_summary", {})
            if isinstance(last_sync_summary, dict) and isinstance(last_sync_summary.get("rejected_return_summary"), dict)
            else {}
        )
        has_rejected_returns = bool(
            int(rejected_return_summary.get("num_rejected_annotator_packets", 0) or 0)
            or int(rejected_return_summary.get("num_rejected_adjudication_packets", 0) or 0)
        )

        stop_reason: str | None = None
        if stop_when_ready and ready:
            stop_reason = "ready"
        elif stop_when_rejected and has_rejected_returns:
            stop_reason = "rejected_returns"
        elif max_iterations > 0 and iteration_count >= max_iterations:
            stop_reason = "max_iterations"

        watch_stop_actions = _bundle_watch_stop_actions(
            bundle_root,
            bundle_manifest.get("operator_scripts", {})
            if isinstance(bundle_manifest.get("operator_scripts"), dict)
            else {},
        )
        next_action = _bundle_watch_next_action(
            {
                "recommended_next_command_id": bundle_manifest.get("recommended_next_command_id"),
                "recommended_next_command": bundle_manifest.get("recommended_next_command"),
                "recommended_next_script_id": bundle_manifest.get("recommended_next_script_id"),
                "recommended_next_script": bundle_manifest.get("recommended_next_script"),
                "recommended_next_script_file": bundle_manifest.get("recommended_next_script_file"),
                "watch_stop_actions": watch_stop_actions,
            },
            stop_reason=stop_reason,
        )

        report = {
            "schema_version": "amst-human-audit-inbox-watch-v1",
            "bundle_dir": str(bundle_root),
            "bundle_manifest_file": str(bundle_root / "bundle_manifest.json"),
            "handoff_manifest_file": bundle_manifest.get("handoff_manifest_file"),
            "return_inbox_state_file": bundle_manifest.get("return_inbox_state_file"),
            "iteration_count": iteration_count,
            "interval_s": interval_s,
            "max_iterations": max_iterations,
            "stop_when_ready": stop_when_ready,
            "stop_when_rejected": stop_when_rejected,
            "reconcile_when_ready": reconcile_when_ready,
            "bundle_status": progress.get("status"),
            "ready_for_merge": progress.get("ready_for_merge"),
            "ready_for_finalize": progress.get("ready_for_finalize"),
            "has_rejected_returns": has_rejected_returns,
            "rejected_return_summary": rejected_return_summary,
            "rejected_returns_report_file": bundle_manifest.get("rejected_returns_report_file"),
            "return_inbox": bundle_manifest.get("return_inbox"),
            "return_archive": bundle_manifest.get("return_archive"),
            "return_reject_archive": bundle_manifest.get("return_reject_archive"),
            "operator_scripts": bundle_manifest.get("operator_scripts", {}),
            "recommended_next_commands": bundle_manifest.get("recommended_next_commands", []),
            "next_command_id": next_action.get("next_command_id"),
            "next_command": next_action.get("next_command"),
            "next_script_id": next_action.get("next_script_id"),
            "next_script": next_action.get("next_script"),
            "next_script_file": next_action.get("next_script_file"),
            "watch_stop_exit_codes": dict(HUMAN_AUDIT_WATCH_STOP_EXIT_CODES),
            "watch_stop_actions": watch_stop_actions,
            "annotator_inboxes": last_sync_summary.get("annotator_inboxes", []) if isinstance(last_sync_summary, dict) else [],
            "adjudication_inboxes": last_sync_summary.get("adjudication_inboxes", []) if isinstance(last_sync_summary, dict) else [],
            "last_sync_summary": last_sync_summary,
            "stop_reason": stop_reason,
            "stop_exit_code": _watch_stop_exit_code(stop_reason),
            "stop_action": watch_stop_actions.get(stop_reason) if isinstance(stop_reason, str) else None,
        }
        if output_file is not None:
            write_json(output_file, report)
        if stop_reason is not None:
            return report
        if interval_s > 0:
            time.sleep(interval_s)


def build_human_audit_evidence_bundle(
    output_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    task_manifest_path: str | Path | None = None,
    annotations_path: str | Path | None = None,
    annotator_attestation_path: str | Path | None = None,
    adjudication_path: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize a self-contained human-audit handoff / evidence bundle."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = Path(manifest_path) if manifest_path is not None else None
    manifest: dict[str, Any] | None = None
    manifest_dir = manifest_file.parent if manifest_file is not None else None
    audit_plan: dict[str, Any] = {}
    benchmark_id: str | None = None
    if manifest_file is not None:
        manifest = read_json(manifest_file)
        if not isinstance(manifest, dict):
            raise ValueError("release manifest must be a JSON object")
        audit_plan = manifest.get("audit_plan", {}) if isinstance(manifest.get("audit_plan"), dict) else {}
        benchmark_id = str(manifest.get("benchmark_id")) if manifest.get("benchmark_id") is not None else None

    source_task_manifest = _resolve_bundle_artifact_path(task_manifest_path, audit_plan.get("audit_task_manifest_file"), manifest_dir)
    if source_task_manifest is None or not source_task_manifest.exists():
        raise ValueError("human-audit evidence bundle requires a task_manifest_path or manifest audit_task_manifest_file")

    task_manifest = read_json(source_task_manifest)
    if not isinstance(task_manifest, dict):
        raise ValueError("human-audit task manifest must be a JSON object")
    source_task_manifest_snapshot = _task_manifest_snapshot(source_task_manifest, task_manifest)

    raw_template_files = task_manifest.get("template_files")
    if not isinstance(raw_template_files, list) or not raw_template_files:
        raise ValueError("human-audit task manifest must include non-empty template_files")
    source_template_paths = tuple(_resolve_path(str(path), source_task_manifest.parent) for path in raw_template_files)
    if not source_template_paths:
        raise ValueError("human-audit evidence bundle requires task-manifest template_files")

    bundle_template_dir = output / "templates"
    bundle_task_dir = output / "tasks"
    bundle_template_dir.mkdir(parents=True, exist_ok=True)
    bundle_task_dir.mkdir(parents=True, exist_ok=True)

    copied_template_paths: list[Path] = []
    template_target_map: dict[Path, Path] = {}
    for source_path in source_template_paths:
        if not source_path.exists():
            raise ValueError(f"human-audit template file does not exist: {source_path}")
        target_path = bundle_template_dir / source_path.name
        shutil.copy2(source_path, target_path)
        copied_template_paths.append(target_path)
        template_target_map[source_path.resolve()] = target_path

    task_files = task_manifest.get("task_files")
    if not isinstance(task_files, dict) or not task_files:
        raise ValueError("human-audit task manifest must include non-empty task_files")
    copied_task_files: dict[str, str] = {}
    for annotator_id, raw_path in sorted(task_files.items()):
        source_path = _resolve_task_path(source_task_manifest.parent, str(raw_path))
        if not source_path.exists():
            raise ValueError(f"human-audit task file does not exist for annotator {annotator_id!r}: {source_path}")
        target_path = bundle_task_dir / source_path.name
        shutil.copy2(source_path, target_path)
        copied_task_files[str(annotator_id)] = _relative_or_absolute(target_path, output)

    bundle_task_manifest = dict(task_manifest)
    bundle_task_manifest["template_files"] = [_relative_or_absolute(path, output) for path in copied_template_paths]
    bundle_task_manifest["template_file_digests"] = {
        relative_path: _file_sha256(path)
        for relative_path, path in zip(bundle_task_manifest["template_files"], copied_template_paths, strict=True)
    }
    bundle_task_manifest["task_files"] = copied_task_files
    bundle_task_manifest["task_identity_digests"] = {
        annotator_id: _task_identity_digest_from_path(output / relative_path)
        for annotator_id, relative_path in sorted(copied_task_files.items())
    }
    bundle_task_manifest_root = _bundle_root_ref(output)
    if bundle_task_manifest_root is not None:
        bundle_task_manifest["root"] = bundle_task_manifest_root
    copied_task_manifest_path = output / "task_manifest.json"
    write_json(copied_task_manifest_path, bundle_task_manifest)
    bundle_task_manifest_snapshot = _task_manifest_snapshot(copied_task_manifest_path, bundle_task_manifest)

    source_annotations = _resolve_bundle_artifact_path(annotations_path, audit_plan.get("audit_annotations_file"), manifest_dir)
    source_attestation = _resolve_bundle_artifact_path(
        annotator_attestation_path,
        audit_plan.get("annotator_attestation_file"),
        manifest_dir,
    )
    source_adjudication = _resolve_bundle_artifact_path(
        adjudication_path,
        audit_plan.get("audit_adjudication_file"),
        manifest_dir,
    )

    copied_annotations_path = _copy_optional_bundle_artifact(source_annotations, output / "completed_annotations.jsonl")
    copied_attestation_path = _copy_optional_bundle_artifact(source_attestation, output / "attestation.json")
    copied_adjudication_path = _copy_optional_bundle_artifact(source_adjudication, output / "adjudication_tasks.jsonl")
    copied_documentation_files, copied_documentation_digests = _copy_bundle_documentation(manifest_file, output)
    if copied_attestation_path is not None:
        copied_attestation = read_json(copied_attestation_path)
        if not isinstance(copied_attestation, dict):
            raise ValueError("annotator attestation must be a JSON object")
        copied_attestation["task_manifest_file"] = _relative_or_absolute(copied_task_manifest_path, output)
        if copied_annotations_path is not None:
            copied_attestation["annotations_file"] = _relative_or_absolute(copied_annotations_path, output)
        write_json(copied_attestation_path, copied_attestation)

    copied_manifest_path: Path | None = None
    if manifest is not None and manifest_file is not None:
        copied_manifest = json.loads(json.dumps(manifest))
        copied_audit_plan = copied_manifest.setdefault("audit_plan", {})
        if isinstance(copied_audit_plan.get("audit_template_files"), dict):
            copied_audit_plan["audit_template_files"] = {
                domain: _relative_or_absolute(
                    template_target_map.get(_resolve_path(str(raw_path), manifest_file.parent).resolve(), Path(str(raw_path))),
                    output,
                )
                for domain, raw_path in sorted(copied_audit_plan["audit_template_files"].items())
            }
        raw_template = copied_audit_plan.get("audit_template_file")
        if raw_template:
            resolved = _resolve_path(str(raw_template), manifest_file.parent).resolve()
            copied_target = template_target_map.get(resolved)
            if copied_target is not None:
                copied_audit_plan["audit_template_file"] = _relative_or_absolute(copied_target, output)
        copied_audit_plan["audit_task_manifest_file"] = _relative_or_absolute(copied_task_manifest_path, output)
        if copied_annotations_path is not None:
            copied_audit_plan["audit_annotations_file"] = _relative_or_absolute(copied_annotations_path, output)
        if copied_attestation_path is not None:
            copied_audit_plan["annotator_attestation_file"] = _relative_or_absolute(copied_attestation_path, output)
        if copied_adjudication_path is not None:
            copied_audit_plan["audit_adjudication_file"] = _relative_or_absolute(copied_adjudication_path, output)
        copied_manifest_root = _bundle_root_ref(output)
        if copied_manifest_root is not None:
            copied_manifest["root"] = copied_manifest_root
        copied_manifest_path = output / "release_manifest.json"
        write_json(copied_manifest_path, copied_manifest)

    progress = summarize_human_audit_progress(copied_task_manifest_path, annotations_path=copied_annotations_path)
    progress_path = output / "progress.json"
    annotation_sheet_summary = write_human_audit_annotation_sheets(
        copied_task_manifest_path,
        output / "sheets",
    )
    bundle_annotation_sheet_files = {
        annotator_id: _relative_or_absolute(Path(path), output)
        for annotator_id, path in sorted(annotation_sheet_summary["annotation_sheet_files"].items())
    }
    bundle_annotation_sheet_summary = dict(annotation_sheet_summary)
    bundle_annotation_sheet_summary["output_dir"] = _relative_or_absolute(Path(annotation_sheet_summary["output_dir"]), output)
    bundle_annotation_sheet_summary["annotation_sheet_files"] = bundle_annotation_sheet_files
    return_inbox = _ensure_bundle_return_inbox(output)
    return_archive = _ensure_bundle_return_archive(output)
    return_reject_archive = _ensure_bundle_return_reject_archive(output)
    return_inbox_state_file = _bundle_return_inbox_state_file(output)
    return_inbox_state = _load_human_audit_return_inbox_state(output)
    rejected_return_summary = _rejected_return_summary(return_inbox_state)
    rejected_returns_report_file = _bundle_rejected_returns_report_file(output)
    write_json(output / "rejected_returns_report.json", summarize_human_audit_rejected_returns(output))
    sandbox_sample = _bundle_relative_sandbox_sample(
        build_human_audit_sandbox_sample(
            copied_task_manifest_path,
            output / "sandbox",
            num_items=1,
            annotation_guideline=HUMAN_AUDIT_BUNDLE_SANDBOX_ANNOTATION_GUIDELINE,
            adjudication_policy=HUMAN_AUDIT_BUNDLE_SANDBOX_ADJUDICATION_POLICY,
            signed_at=HUMAN_AUDIT_BUNDLE_SANDBOX_SIGNED_AT,
        ),
        output,
    )
    annotator_packets = _build_annotator_packets(
        output,
        copied_task_files,
        bundle_annotation_sheet_files,
        copied_documentation_files,
        sandbox_sample,
        progress.get("annotator_progress"),
    )
    verification: dict[str, Any] | None = None
    verification_path: Path | None = None
    agreement_path: Path | None = None
    adjudication_package: dict[str, Any] | None = None
    if copied_annotations_path is not None:
        if copied_manifest_path is not None and copied_manifest_path.exists():
            copied_manifest_payload = read_json(copied_manifest_path)
            copied_audit_plan = copied_manifest_payload.get("audit_plan", {}) if isinstance(copied_manifest_payload, dict) else {}
            completed = copied_audit_plan.get("human_audit_status") == "completed"
            has_metrics = bool(copied_audit_plan.get("agreement_metrics"))
            if completed and has_metrics:
                verification = verify_manifest_human_audit(copied_manifest_path)
            else:
                verification = verify_completed_human_audit(
                    tuple(copied_template_paths),
                    copied_annotations_path,
                    task_manifest_path=copied_task_manifest_path,
                    annotator_attestation_path=copied_attestation_path,
                    adjudication_path=copied_adjudication_path,
                    benchmark_id=benchmark_id,
                    manifest_path=copied_manifest_path,
                )
        else:
            verification = verify_completed_human_audit(
                tuple(copied_template_paths),
                copied_annotations_path,
                task_manifest_path=copied_task_manifest_path,
                annotator_attestation_path=copied_attestation_path,
                adjudication_path=copied_adjudication_path,
                benchmark_id=benchmark_id,
            )
        verification_path = output / "verification_report.json"
        write_json(verification_path, verification)
        if verification.get("agreement_metrics") is not None:
            agreement_path = output / "agreement_metrics.json"
            write_json(agreement_path, verification["agreement_metrics"])
        if verification.get("adjudication_recommended") and copied_adjudication_path is None:
            adjudication_package = _bundle_relative_adjudication_package(
                build_human_audit_adjudication_package(
                    copied_task_manifest_path,
                    copied_annotations_path,
                    output / "adjudication",
                ),
                output,
            )
    adjudication_packet = _build_adjudication_packet(output, adjudication_package, copied_documentation_files)
    pending_return_packets, pending_return_packet_paths = _bundle_pending_return_packets(output, return_inbox)
    recommended_next_commands = _bundle_next_commands(
        output,
        copied_manifest_path,
        copied_task_manifest_path,
        copied_annotations_path,
        copied_attestation_path,
        copied_adjudication_path,
        progress,
        verification,
        rejected_return_summary=rejected_return_summary,
        pending_return_packets=pending_return_packets,
    )
    operator_scripts = _write_bundle_operator_scripts(output)
    operator_commands = _bundle_operator_commands(output, str(copied_task_manifest_path))
    watch_stop_actions = _bundle_watch_stop_actions(output, operator_scripts)
    return_inbox_watch_file = _bundle_return_inbox_watch_file(output)
    return_inbox_sync_report_file = _bundle_return_inbox_sync_report_file(output)
    recommended_next_action = _bundle_recommended_next_action(output, operator_scripts, recommended_next_commands)
    bundle_progress = _bundle_progress_payload(
        output,
        progress,
        annotation_sheet_files=bundle_annotation_sheet_files,
        annotator_packets=annotator_packets,
        adjudication_packet=adjudication_packet,
        sandbox_sample=sandbox_sample,
        return_inbox=return_inbox,
        return_archive=return_archive,
        return_reject_archive=return_reject_archive,
        rejected_return_summary=rejected_return_summary,
        rejected_returns_report_file=rejected_returns_report_file,
        return_inbox_watch_file=return_inbox_watch_file,
        return_inbox_sync_report_file=return_inbox_sync_report_file,
        return_inbox_state_file=return_inbox_state_file,
        pending_return_packets=pending_return_packets,
        pending_return_packet_paths=pending_return_packet_paths,
        watch_stop_exit_codes=HUMAN_AUDIT_WATCH_STOP_EXIT_CODES,
        watch_stop_actions=watch_stop_actions,
        operator_scripts=operator_scripts,
        operator_commands=operator_commands,
        recommended_next_commands=recommended_next_commands,
        recommended_next_command_id=recommended_next_action["recommended_next_command_id"],
        recommended_next_command=recommended_next_action["recommended_next_command"],
        recommended_next_script_id=recommended_next_action["recommended_next_script_id"],
        recommended_next_script=recommended_next_action["recommended_next_script"],
        recommended_next_script_file=recommended_next_action["recommended_next_script_file"],
    )
    write_json(progress_path, bundle_progress)

    bundle_report = {
        "schema_version": "amst-human-audit-evidence-bundle-v1",
        "root": _bundle_root_ref(output),
        "bundle_dir": str(output),
        "benchmark_id": benchmark_id,
        "source_manifest_file": str(manifest_file) if manifest_file is not None else None,
        "source_task_manifest_file": str(source_task_manifest),
        "source_task_manifest_snapshot": source_task_manifest_snapshot,
        "source_annotations_file": str(source_annotations) if source_annotations is not None else None,
        "source_annotator_attestation_file": str(source_attestation) if source_attestation is not None else None,
        "source_adjudication_file": str(source_adjudication) if source_adjudication is not None else None,
        "release_manifest_file": str(copied_manifest_path) if copied_manifest_path is not None else None,
        "task_manifest_file": str(copied_task_manifest_path),
        "bundle_task_manifest_snapshot": bundle_task_manifest_snapshot,
        "template_files": [str(path) for path in copied_template_paths],
        "task_files": dict(copied_task_files),
        "annotations_file": str(copied_annotations_path) if copied_annotations_path is not None else None,
        "annotator_attestation_file": str(copied_attestation_path) if copied_attestation_path is not None else None,
        "adjudication_file": str(copied_adjudication_path) if copied_adjudication_path is not None else None,
        "verification_report_file": str(verification_path) if verification_path is not None else None,
        "agreement_metrics_file": str(agreement_path) if agreement_path is not None else None,
        "documentation_files": copied_documentation_files,
        "documentation_file_digests": copied_documentation_digests,
        "annotation_sheet_files": bundle_annotation_sheet_files,
        "annotation_sheet_summary": bundle_annotation_sheet_summary,
        "sandbox_sample": sandbox_sample,
        "annotator_packets": annotator_packets,
        "adjudication_package": adjudication_package,
        "adjudication_packet": adjudication_packet,
        "return_inbox": return_inbox,
        "return_archive": return_archive,
        "return_reject_archive": return_reject_archive,
        "rejected_return_summary": rejected_return_summary,
        "rejected_returns_report_file": rejected_returns_report_file,
        "return_inbox_watch_file": return_inbox_watch_file,
        "return_inbox_sync_report_file": return_inbox_sync_report_file,
        "return_inbox_state_file": return_inbox_state_file,
        "pending_return_packets": pending_return_packets,
        "pending_return_packet_paths": pending_return_packet_paths,
        "watch_stop_exit_codes": dict(HUMAN_AUDIT_WATCH_STOP_EXIT_CODES),
        "watch_stop_actions": watch_stop_actions,
        "operator_scripts": operator_scripts,
        "operator_commands": operator_commands,
        "checks": list(bundle_task_manifest.get("checks", [])) if isinstance(bundle_task_manifest.get("checks"), list) else list(AUDIT_CHECK_DEFINITIONS),
        "check_definitions": (
            {str(key): str(value) for key, value in sorted(bundle_task_manifest.get("check_definitions", {}).items())}
            if isinstance(bundle_task_manifest.get("check_definitions"), dict)
            else dict(AUDIT_CHECK_DEFINITIONS)
        ),
        "status": bundle_progress.get("status"),
        "ready_for_merge": bundle_progress.get("ready_for_merge"),
        "ready_for_finalize": bundle_progress.get("ready_for_finalize"),
        "annotator_ids": bundle_progress.get("expected_annotator_ids", []),
        "num_annotators": bundle_progress.get("num_expected_annotators"),
        "num_expected_annotations": bundle_progress.get("num_expected_annotations"),
        "num_matched_annotations": bundle_progress.get("num_matched_annotations"),
        "num_missing_annotations": bundle_progress.get("num_missing_annotations"),
        "progress_file": str(progress_path),
        "progress": bundle_progress,
        "verification": verification,
        "recommended_next_commands": recommended_next_commands,
        "recommended_next_command_id": recommended_next_action["recommended_next_command_id"],
        "recommended_next_command": recommended_next_action["recommended_next_command"],
        "recommended_next_script_id": recommended_next_action["recommended_next_script_id"],
        "recommended_next_script": recommended_next_action["recommended_next_script"],
        "recommended_next_script_file": recommended_next_action["recommended_next_script_file"],
        "handoff_manifest_file": str(output / "handoff_manifest.json"),
    }
    readme_path = output / "README.md"
    bundle_report["readme_file"] = str(readme_path)
    write_json(output / "bundle_manifest.json", bundle_report)
    _write_initial_human_audit_return_inbox_sidecars(output, bundle_report=bundle_report, operator_scripts=operator_scripts)
    write_json(output / "handoff_manifest.json", _build_handoff_manifest(bundle_report))
    readme_path.write_text(_bundle_readme(bundle_report), encoding="utf-8")
    return bundle_report


def verify_human_audit_evidence_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Verify that a human-audit evidence bundle is self-consistent and reproducible."""

    bundle_root = Path(bundle_dir)
    bundle_manifest_path = bundle_root / "bundle_manifest.json"
    errors: list[str] = []
    try:
        bundle_manifest = read_json(bundle_manifest_path)
    except Exception as exc:  # noqa: BLE001 - report user-facing bundle parse errors
        return {
            "schema_version": "amst-human-audit-evidence-bundle-verification-v1",
            "root": _bundle_root_ref(bundle_root),
            "bundle_dir": str(bundle_root),
            "bundle_manifest_file": str(bundle_manifest_path),
            "ok": False,
            "status": "failed",
            "errors": [f"could not load bundle_manifest.json: {exc}"],
            "checks": {},
        }
    if not isinstance(bundle_manifest, dict):
        return {
            "schema_version": "amst-human-audit-evidence-bundle-verification-v1",
            "root": _bundle_root_ref(bundle_root),
            "bundle_dir": str(bundle_root),
            "bundle_manifest_file": str(bundle_manifest_path),
            "ok": False,
            "status": "failed",
            "errors": ["bundle_manifest.json must be a JSON object"],
            "checks": {},
        }

    checks: dict[str, Any] = {}
    _record_bundle_check(
        checks,
        errors,
        "bundle_manifest.schema_version",
        bundle_manifest.get("schema_version"),
        "amst-human-audit-evidence-bundle-v1",
    )
    _record_bundle_check(
        checks,
        errors,
        "bundle_manifest.bundle_dir",
        str(bundle_root),
        str(_resolve_path(str(bundle_manifest.get("bundle_dir", "")), bundle_manifest_path.parent)),
    )
    _record_bundle_check(
        checks,
        errors,
        "bundle_documentation.file_digests",
        _bundle_documentation_digests(bundle_manifest_path.parent, bundle_manifest),
        bundle_manifest.get("documentation_file_digests", {}),
    )
    _record_bundle_check(
        checks,
        errors,
        "bundle_annotation_sheets.missing_files",
        _bundle_missing_annotation_sheets(bundle_manifest_path.parent, bundle_manifest),
        [],
    )
    _record_bundle_check(
        checks,
        errors,
        "bundle_return_inbox.missing_dirs",
        _bundle_missing_return_inbox_dirs(bundle_manifest_path.parent, bundle_manifest),
        [],
    )
    _record_bundle_check(
        checks,
        errors,
        "bundle_return_inbox_state.missing_files",
        _bundle_missing_return_inbox_state_files(bundle_manifest_path.parent, bundle_manifest),
        [],
    )
    _record_bundle_check(
        checks,
        errors,
        "bundle_return_inbox_watch.missing_files",
        _bundle_missing_return_inbox_watch_files(bundle_manifest_path.parent, bundle_manifest),
        [],
    )
    _record_bundle_check(
        checks,
        errors,
        "bundle_return_inbox_watch.summary",
        _bundle_return_inbox_watch_signature(bundle_manifest_path.parent, bundle_manifest),
        _expected_bundle_return_inbox_watch_signature(bundle_manifest_path.parent, bundle_manifest),
    )
    _record_bundle_check(
        checks,
        errors,
        "bundle_return_inbox_sync.missing_files",
        _bundle_missing_return_inbox_sync_report_files(bundle_manifest_path.parent, bundle_manifest),
        [],
    )
    _record_bundle_check(
        checks,
        errors,
        "bundle_return_inbox_sync.summary",
        _bundle_return_inbox_sync_signature(bundle_manifest_path.parent, bundle_manifest),
        _expected_bundle_return_inbox_sync_signature(bundle_manifest_path.parent, bundle_manifest),
    )
    _record_bundle_check(
        checks,
        errors,
        "bundle_rejected_returns_report.missing_files",
        _bundle_missing_rejected_returns_report_files(bundle_manifest_path.parent, bundle_manifest),
        [],
    )
    _record_bundle_check(
        checks,
        errors,
        "bundle_return_archive.missing_dirs",
        _bundle_missing_return_archive_dirs(bundle_manifest_path.parent, bundle_manifest),
        [],
    )
    _record_bundle_check(
        checks,
        errors,
        "bundle_return_reject_archive.missing_dirs",
        _bundle_missing_return_reject_archive_dirs(bundle_manifest_path.parent, bundle_manifest),
        [],
    )
    _record_bundle_check(
        checks,
        errors,
        "bundle_operator_scripts.missing_files",
        _bundle_missing_operator_scripts(bundle_manifest_path.parent, bundle_manifest),
        [],
    )
    _record_bundle_check(
        checks,
        errors,
        "bundle_annotator_packets.missing_files",
        _bundle_missing_annotator_packet_files(bundle_manifest_path.parent, bundle_manifest),
        [],
    )
    _record_bundle_check(
        checks,
        errors,
        "bundle_handoff_manifest.summary",
        _bundle_handoff_manifest_signature(bundle_manifest_path.parent, bundle_manifest),
        _expected_bundle_handoff_manifest_signature(bundle_manifest_path.parent, bundle_manifest),
    )
    _record_bundle_check(
        checks,
        errors,
        "bundle_adjudication_packet.missing_files",
        _bundle_missing_adjudication_packet_files(bundle_manifest_path.parent, bundle_manifest),
        [],
    )

    task_manifest_path = _resolve_path(str(bundle_manifest.get("task_manifest_file", "")), bundle_manifest_path.parent)
    actual_bundle_snapshot = _task_manifest_snapshot(task_manifest_path)
    _record_bundle_check(
        checks,
        errors,
        "bundle_task_manifest.snapshot",
        _snapshot_signature(actual_bundle_snapshot),
        _snapshot_signature(bundle_manifest.get("bundle_task_manifest_snapshot")),
    )

    annotations_value = bundle_manifest.get("annotations_file")
    annotations_path = (
        _resolve_path(str(annotations_value), bundle_manifest_path.parent)
        if annotations_value
        else None
    )
    progress = summarize_human_audit_progress(task_manifest_path, annotations_path=annotations_path)
    _record_bundle_check(checks, errors, "bundle_task_manifest.progress_errors", progress.get("errors", []), [])

    source_task_manifest_value = bundle_manifest.get("source_task_manifest_file")
    source_snapshot = None
    if source_task_manifest_value:
        source_task_manifest_path = _resolve_path(str(source_task_manifest_value), bundle_manifest_path.parent)
        if source_task_manifest_path.exists():
            source_snapshot = _task_manifest_snapshot(source_task_manifest_path)
            _record_bundle_check(
                checks,
                errors,
                "source_task_manifest.snapshot",
                _snapshot_signature(source_snapshot),
                _snapshot_signature(bundle_manifest.get("source_task_manifest_snapshot")),
            )
            _record_bundle_check(
                checks,
                errors,
                "source_bundle_alignment",
                _snapshot_signature(actual_bundle_snapshot),
                _snapshot_signature(source_snapshot),
            )
        else:
            errors.append(f"source task manifest does not exist: {source_task_manifest_path}")
            checks["source_task_manifest.snapshot"] = {
                "passed": False,
                "actual": None,
                "expected": _snapshot_signature(bundle_manifest.get("source_task_manifest_snapshot")),
            }

    verification_path = None
    recomputed_verification: dict[str, Any] | None = None
    verification_value = bundle_manifest.get("verification_report_file")
    if verification_value:
        verification_path = _resolve_path(str(verification_value), bundle_manifest_path.parent)
    if verification_path is not None and verification_path.exists():
        stored_verification = read_json(verification_path)
        recomputed_verification = _recompute_bundle_verification(bundle_manifest_path.parent, bundle_manifest)
        _record_bundle_check(
            checks,
            errors,
            "bundle_verification.ok",
            recomputed_verification.get("ok"),
            stored_verification.get("ok") if isinstance(stored_verification, dict) else None,
        )
        if isinstance(stored_verification, dict):
            _record_bundle_check(
                checks,
                errors,
                "bundle_verification.file_digests",
                recomputed_verification.get("file_digests"),
                stored_verification.get("file_digests"),
            )
            _record_bundle_check(
                checks,
                errors,
                "bundle_verification.semantic_checks",
                recomputed_verification.get("semantic_checks"),
                stored_verification.get("semantic_checks"),
            )
    elif bundle_manifest.get("verification") is not None:
        errors.append("bundle manifest includes verification payload but verification_report_file is missing")

    return {
        "schema_version": "amst-human-audit-evidence-bundle-verification-v1",
        "root": _bundle_root_ref(bundle_root),
        "bundle_dir": str(bundle_root),
        "bundle_manifest_file": str(bundle_manifest_path),
        "task_manifest_file": str(task_manifest_path),
        "annotations_file": str(annotations_path) if annotations_path is not None else None,
        "ok": not errors,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "checks": checks,
        "progress": progress,
        "recomputed_verification": recomputed_verification,
    }


def _bundle_progress_payload(
    bundle_root: Path,
    progress: dict[str, Any],
    *,
    annotation_sheet_files: dict[str, str],
    annotator_packets: dict[str, Any],
    adjudication_packet: dict[str, Any] | None,
    sandbox_sample: dict[str, Any] | None,
    return_inbox: dict[str, str],
    return_archive: dict[str, str],
    return_reject_archive: dict[str, str],
    rejected_return_summary: dict[str, Any],
    rejected_returns_report_file: str,
    return_inbox_watch_file: str,
    return_inbox_sync_report_file: str,
    return_inbox_state_file: str,
    pending_return_packets: list[str],
    pending_return_packet_paths: list[str],
    watch_stop_exit_codes: dict[str, int],
    watch_stop_actions: dict[str, Any],
    operator_scripts: dict[str, str],
    operator_commands: dict[str, str],
    recommended_next_commands: list[str],
    recommended_next_command_id: str | None,
    recommended_next_command: str | None,
    recommended_next_script_id: str | None,
    recommended_next_script: str | None,
    recommended_next_script_file: str | None,
) -> dict[str, Any]:
    payload = dict(progress)
    payload["root"] = _bundle_root_ref(bundle_root)
    payload["bundle_dir"] = str(bundle_root)
    payload["bundle_manifest_file"] = str(bundle_root / "bundle_manifest.json")
    payload["handoff_manifest_file"] = str(bundle_root / "handoff_manifest.json")
    payload["annotation_sheet_files"] = dict(annotation_sheet_files)
    payload["annotator_packets"] = dict(annotator_packets)
    payload["adjudication_packet"] = adjudication_packet
    payload["sandbox_sample"] = sandbox_sample
    payload["return_inbox"] = dict(return_inbox)
    payload["return_archive"] = dict(return_archive)
    payload["return_reject_archive"] = dict(return_reject_archive)
    payload["rejected_return_summary"] = dict(rejected_return_summary)
    payload["rejected_returns_report_file"] = rejected_returns_report_file
    payload["return_inbox_watch_file"] = return_inbox_watch_file
    payload["return_inbox_sync_report_file"] = return_inbox_sync_report_file
    payload["return_inbox_state_file"] = return_inbox_state_file
    payload["pending_return_packets"] = list(pending_return_packets)
    payload["pending_return_packet_paths"] = list(pending_return_packet_paths)
    payload["watch_stop_exit_codes"] = dict(watch_stop_exit_codes)
    payload["watch_stop_actions"] = dict(watch_stop_actions)
    payload["operator_scripts"] = dict(operator_scripts)
    payload["operator_commands"] = dict(operator_commands)
    payload["recommended_next_commands"] = list(recommended_next_commands)
    payload["recommended_next_command_id"] = recommended_next_command_id
    payload["recommended_next_command"] = recommended_next_command
    payload["recommended_next_script_id"] = recommended_next_script_id
    payload["recommended_next_script"] = recommended_next_script
    payload["recommended_next_script_file"] = recommended_next_script_file
    return payload


def _first_recommended_bundle_command(bundle_manifest: dict[str, Any]) -> str | None:
    commands = bundle_manifest.get("recommended_next_commands")
    if not isinstance(commands, list):
        return None
    for command in commands:
        if isinstance(command, str) and command.strip():
            return command
    return None


def _bundle_operator_script_id_for_command(command: str | None) -> str | None:
    if not isinstance(command, str) or not command.strip():
        return None
    if "watch-human-audit-return-inbox" in command:
        return "watch_return_inbox"
    if "reconcile-human-audit-evidence-bundle" in command:
        return "reconcile_when_ready"
    if "human-audit-progress" in command:
        return "progress"
    if "sync-human-audit-return-inbox" in command:
        return "sync_return_inbox"
    if "summarize-human-audit-rejected-returns" in command:
        return "review_rejected_returns"
    if "verify-human-audit-evidence-bundle" in command:
        return "verify_bundle"
    return None


def _bundle_next_operator_script_refs(
    bundle_root: Path,
    bundle_manifest: dict[str, Any],
    command: str | None,
) -> tuple[str | None, str | None]:
    script_id = _bundle_operator_script_id_for_command(command)
    if script_id is None:
        return None, None
    operator_scripts = bundle_manifest.get("operator_scripts")
    if not isinstance(operator_scripts, dict):
        return None, None
    raw_script = operator_scripts.get(script_id)
    if not isinstance(raw_script, str) or not raw_script.strip():
        return None, None
    script_path = Path(raw_script)
    script_file = script_path if script_path.is_absolute() else bundle_root / script_path
    return raw_script, str(script_file)


def _bundle_recommended_next_action(
    bundle_root: Path,
    operator_scripts: dict[str, str],
    recommended_next_commands: list[str],
) -> dict[str, Any]:
    command = None
    for candidate in recommended_next_commands:
        if isinstance(candidate, str) and candidate.strip():
            command = candidate
            break
    script_id = _bundle_operator_script_id_for_command(command)
    next_script = None
    next_script_file = None
    if script_id is not None:
        next_script, next_script_file = _bundle_operator_script_ref(bundle_root, operator_scripts, script_id)
    return {
        "recommended_next_command_id": script_id,
        "recommended_next_command": command,
        "recommended_next_script_id": script_id,
        "recommended_next_script": next_script,
        "recommended_next_script_file": next_script_file,
    }


def _bundle_operator_script_ref(
    bundle_root: Path,
    operator_scripts: dict[str, str],
    script_id: str,
) -> tuple[str | None, str | None]:
    raw_script = operator_scripts.get(script_id)
    if not isinstance(raw_script, str) or not raw_script.strip():
        return None, None
    script_path = Path(raw_script)
    script_file = script_path if script_path.is_absolute() else bundle_root / script_path
    return raw_script, str(script_file)


def _watch_stop_exit_code(stop_reason: str | None) -> int:
    if not isinstance(stop_reason, str) or not stop_reason.strip():
        return 0
    return int(HUMAN_AUDIT_WATCH_STOP_EXIT_CODES.get(stop_reason, 0) or 0)


def _bundle_progress_command(bundle_root: Path, task_manifest_file: str | None) -> str | None:
    if not isinstance(task_manifest_file, str) or not task_manifest_file:
        return None
    return (
        f"{HUMAN_AUDIT_CLI_PREFIX} human-audit-progress "
        f"--task-manifest {task_manifest_file} "
        f"--output {bundle_root / 'progress.json'}"
    )


def _bundle_annotation_guideline_cli_path(bundle_root: Path) -> str:
    return str(bundle_root / HUMAN_AUDIT_BUNDLE_SANDBOX_ANNOTATION_GUIDELINE)


def _bundle_watch_return_inbox_command(bundle_root: Path) -> str:
    return (
        f"{HUMAN_AUDIT_CLI_PREFIX} watch-human-audit-return-inbox "
        f"--bundle-dir {bundle_root} --interval-s {HUMAN_AUDIT_OPERATOR_WATCH_INTERVAL_S} --max-iterations 0 "
        "--stop-when-ready "
        "--stop-when-rejected "
        f"--output {bundle_root / 'return_inbox_watch.json'}"
    )


def _bundle_sync_return_inbox_command(bundle_root: Path) -> str:
    return (
        f"{HUMAN_AUDIT_CLI_PREFIX} sync-human-audit-return-inbox "
        f"--bundle-dir {bundle_root} "
        "--reconcile-when-ready "
        f"--signed-at {HUMAN_AUDIT_BUNDLE_SANDBOX_SIGNED_AT} "
        f"--annotation-guideline {_bundle_annotation_guideline_cli_path(bundle_root)} "
        f"--adjudication-policy '{HUMAN_AUDIT_BUNDLE_SANDBOX_ADJUDICATION_POLICY}' "
        f"--output {bundle_root / 'return_inbox_sync.json'}"
    )


def _bundle_reconcile_when_ready_command(bundle_root: Path) -> str:
    return (
        f"{HUMAN_AUDIT_CLI_PREFIX} reconcile-human-audit-evidence-bundle "
        f"--bundle-dir {bundle_root} "
        f"--signed-at {HUMAN_AUDIT_BUNDLE_SANDBOX_SIGNED_AT} "
        f"--annotation-guideline {_bundle_annotation_guideline_cli_path(bundle_root)} "
        f"--adjudication-policy '{HUMAN_AUDIT_BUNDLE_SANDBOX_ADJUDICATION_POLICY}'"
    )


def _bundle_review_rejected_returns_command(bundle_root: Path) -> str:
    return (
        f"{HUMAN_AUDIT_CLI_PREFIX} summarize-human-audit-rejected-returns "
        f"--bundle-dir {bundle_root} --output {bundle_root / 'rejected_returns_report.json'}"
    )


def _bundle_verify_bundle_command(bundle_root: Path) -> str:
    return (
        f"{HUMAN_AUDIT_CLI_PREFIX} verify-human-audit-evidence-bundle "
        f"--bundle-dir {bundle_root} --output {bundle_root / 'bundle_verification.json'}"
    )


def _bundle_operator_commands(bundle_root: Path, task_manifest_file: str | None) -> dict[str, str]:
    commands: dict[str, str] = {}
    progress_command = _bundle_progress_command(bundle_root, task_manifest_file)
    if progress_command is not None:
        commands["progress"] = progress_command
    commands["watch_return_inbox"] = _bundle_watch_return_inbox_command(bundle_root)
    commands["sync_return_inbox"] = _bundle_sync_return_inbox_command(bundle_root)
    commands["review_rejected_returns"] = _bundle_review_rejected_returns_command(bundle_root)
    commands["reconcile_when_ready"] = _bundle_reconcile_when_ready_command(bundle_root)
    commands["verify_bundle"] = _bundle_verify_bundle_command(bundle_root)
    return commands


def _bundle_watch_stop_actions(
    bundle_root: Path,
    operator_scripts: dict[str, str],
) -> dict[str, dict[str, Any]]:
    watch_script, watch_script_file = _bundle_operator_script_ref(bundle_root, operator_scripts, "watch_return_inbox")
    review_script, review_script_file = _bundle_operator_script_ref(
        bundle_root,
        operator_scripts,
        "review_rejected_returns",
    )
    reconcile_script, reconcile_script_file = _bundle_operator_script_ref(
        bundle_root,
        operator_scripts,
        "reconcile_when_ready",
    )
    return {
        "max_iterations": {
            "exit_code": HUMAN_AUDIT_WATCH_STOP_EXIT_CODES["max_iterations"],
            "kind": "continue_waiting",
            "next_command": _bundle_watch_return_inbox_command(bundle_root),
            "next_command_id": "watch_return_inbox",
            "next_script": watch_script,
            "next_script_file": watch_script_file,
            "next_script_id": "watch_return_inbox",
        },
        "rejected_returns": {
            "exit_code": HUMAN_AUDIT_WATCH_STOP_EXIT_CODES["rejected_returns"],
            "kind": "triage_rejected_returns",
            "next_command": _bundle_review_rejected_returns_command(bundle_root),
            "next_command_id": "review_rejected_returns",
            "next_script": review_script,
            "next_script_file": review_script_file,
            "next_script_id": "review_rejected_returns",
        },
        "ready": {
            "exit_code": HUMAN_AUDIT_WATCH_STOP_EXIT_CODES["ready"],
            "kind": "reconcile_ready_bundle",
            "next_command": _bundle_reconcile_when_ready_command(bundle_root),
            "next_command_id": "reconcile_when_ready",
            "next_script": reconcile_script,
            "next_script_file": reconcile_script_file,
            "next_script_id": "reconcile_when_ready",
        },
    }


def _bundle_recommended_next_refs(bundle_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "next_command_id": bundle_manifest.get("recommended_next_command_id"),
        "next_command": bundle_manifest.get("recommended_next_command"),
        "next_script_id": bundle_manifest.get("recommended_next_script_id"),
        "next_script": bundle_manifest.get("recommended_next_script"),
        "next_script_file": bundle_manifest.get("recommended_next_script_file"),
    }


def _bundle_watch_next_action(
    bundle_manifest: dict[str, Any],
    *,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    next_action = _bundle_recommended_next_refs(bundle_manifest)
    if isinstance(stop_reason, str) and stop_reason.strip():
        watch_stop_actions = bundle_manifest.get("watch_stop_actions")
        stop_action = watch_stop_actions.get(stop_reason) if isinstance(watch_stop_actions, dict) else None
        if isinstance(stop_action, dict):
            for key in ("next_command_id", "next_command", "next_script_id", "next_script", "next_script_file"):
                value = stop_action.get(key)
                if isinstance(value, str) and value.strip():
                    next_action[key] = value
    return next_action


def _ensure_bundle_return_inbox(bundle_root: Path) -> dict[str, str]:
    annotator_inbox = bundle_root / "returns" / "annotators"
    adjudication_inbox = bundle_root / "returns" / "adjudication"
    annotator_inbox.mkdir(parents=True, exist_ok=True)
    adjudication_inbox.mkdir(parents=True, exist_ok=True)
    return {
        "annotator_inbox": _relative_or_absolute(annotator_inbox, bundle_root),
        "adjudication_inbox": _relative_or_absolute(adjudication_inbox, bundle_root),
    }


def _ensure_bundle_return_archive(bundle_root: Path) -> dict[str, str]:
    annotator_archive = bundle_root / "returns" / "processed" / "annotators"
    adjudication_archive = bundle_root / "returns" / "processed" / "adjudication"
    annotator_archive.mkdir(parents=True, exist_ok=True)
    adjudication_archive.mkdir(parents=True, exist_ok=True)
    return {
        "annotator_archive": _relative_or_absolute(annotator_archive, bundle_root),
        "adjudication_archive": _relative_or_absolute(adjudication_archive, bundle_root),
    }


def _ensure_bundle_return_reject_archive(bundle_root: Path) -> dict[str, str]:
    annotator_archive = bundle_root / "returns" / "rejected" / "annotators"
    adjudication_archive = bundle_root / "returns" / "rejected" / "adjudication"
    annotator_archive.mkdir(parents=True, exist_ok=True)
    adjudication_archive.mkdir(parents=True, exist_ok=True)
    return {
        "annotator_archive": _relative_or_absolute(annotator_archive, bundle_root),
        "adjudication_archive": _relative_or_absolute(adjudication_archive, bundle_root),
    }


def _bundle_return_inbox_watch_file(bundle_root: Path) -> str:
    return _relative_or_absolute(bundle_root / "return_inbox_watch.json", bundle_root)


def _bundle_return_inbox_sync_report_file(bundle_root: Path) -> str:
    return _relative_or_absolute(bundle_root / "return_inbox_sync.json", bundle_root)


def _write_initial_human_audit_return_inbox_sidecars(
    bundle_root: Path,
    *,
    bundle_report: dict[str, Any],
    operator_scripts: dict[str, str],
) -> None:
    sync_path = bundle_root / "return_inbox_sync.json"
    existing_sync = read_json(sync_path) if sync_path.exists() else None
    sync_payload = dict(existing_sync) if isinstance(existing_sync, dict) else {}
    sync_next_action = _bundle_recommended_next_refs(bundle_report)
    sync_payload.update(
        {
            "schema_version": "amst-human-audit-inbox-sync-v1",
            "root": (
                bundle_report.get("root")
                if isinstance(bundle_report.get("root"), str)
                else _bundle_root_ref(bundle_root)
            ),
            "bundle_dir": str(bundle_root),
            "return_inbox_state_file": str(bundle_root / "return_inbox_state.json"),
            "ready_for_merge": bundle_report.get("ready_for_merge"),
            "ready_for_finalize": bundle_report.get("ready_for_finalize"),
            "bundle_status": bundle_report.get("status"),
            "recommended_next_commands": list(bundle_report.get("recommended_next_commands", []))
            if isinstance(bundle_report.get("recommended_next_commands"), list)
            else [],
            "next_command_id": sync_next_action.get("next_command_id"),
            "next_command": sync_next_action.get("next_command"),
            "next_script_id": sync_next_action.get("next_script_id"),
            "next_script": sync_next_action.get("next_script"),
            "next_script_file": sync_next_action.get("next_script_file"),
        }
    )
    sync_payload.setdefault("status", "not_started")
    sync_payload.setdefault("annotator_inboxes", [])
    sync_payload.setdefault("adjudication_inboxes", [])
    sync_payload.setdefault("annotator_candidates", [])
    sync_payload.setdefault("annotator_skipped_processed_candidates", [])
    sync_payload.setdefault("annotator_invalid_candidates", [])
    sync_payload.setdefault("annotator_skipped_rejected_candidates", [])
    sync_payload.setdefault("adjudication_candidate", None)
    sync_payload.setdefault("adjudication_skipped_processed_candidates", [])
    sync_payload.setdefault("adjudication_invalid_candidates", [])
    sync_payload.setdefault("adjudication_skipped_rejected_candidates", [])
    sync_payload.setdefault("annotator_ingest_summary", None)
    sync_payload.setdefault("adjudication_apply_summary", None)
    sync_payload.setdefault(
        "processed_return_archives",
        {"annotator_archive_files": [], "adjudication_archive_file": None},
    )
    sync_payload.setdefault(
        "rejected_return_archives",
        {"annotator_archive_files": [], "adjudication_archive_files": []},
    )
    sync_payload.setdefault(
        "rejected_return_summary",
        dict(bundle_report.get("rejected_return_summary"))
        if isinstance(bundle_report.get("rejected_return_summary"), dict)
        else {},
    )
    write_json(sync_path, sync_payload)
    watch_path = bundle_root / "return_inbox_watch.json"
    existing_watch = read_json(watch_path) if watch_path.exists() else None
    watch_payload = dict(existing_watch) if isinstance(existing_watch, dict) else {}
    next_action = _bundle_watch_next_action(bundle_report)
    watch_payload.update(
        {
            "schema_version": "amst-human-audit-inbox-watch-v1",
            "root": (
                bundle_report.get("root")
                if isinstance(bundle_report.get("root"), str)
                else _bundle_root_ref(bundle_root)
            ),
            "bundle_dir": str(bundle_root),
            "bundle_manifest_file": str(bundle_root / "bundle_manifest.json"),
            "handoff_manifest_file": bundle_report.get("handoff_manifest_file"),
            "return_inbox_state_file": bundle_report.get("return_inbox_state_file"),
            "bundle_status": bundle_report.get("status"),
            "ready_for_merge": bundle_report.get("ready_for_merge"),
            "ready_for_finalize": bundle_report.get("ready_for_finalize"),
            "rejected_returns_report_file": bundle_report.get("rejected_returns_report_file"),
            "return_inbox": dict(bundle_report.get("return_inbox", {}))
            if isinstance(bundle_report.get("return_inbox"), dict)
            else {},
            "return_archive": dict(bundle_report.get("return_archive", {}))
            if isinstance(bundle_report.get("return_archive"), dict)
            else {},
            "return_reject_archive": dict(bundle_report.get("return_reject_archive", {}))
            if isinstance(bundle_report.get("return_reject_archive"), dict)
            else {},
            "operator_scripts": dict(operator_scripts),
            "recommended_next_commands": list(bundle_report.get("recommended_next_commands", []))
            if isinstance(bundle_report.get("recommended_next_commands"), list)
            else [],
            "watch_stop_exit_codes": dict(HUMAN_AUDIT_WATCH_STOP_EXIT_CODES),
            "watch_stop_actions": (
                dict(bundle_report.get("watch_stop_actions"))
                if isinstance(bundle_report.get("watch_stop_actions"), dict)
                else {}
            ),
        }
    )
    watch_payload.setdefault("status", "not_started")
    watch_payload.setdefault("iteration_count", 0)
    watch_payload.setdefault("interval_s", HUMAN_AUDIT_OPERATOR_WATCH_INTERVAL_S)
    watch_payload.setdefault("max_iterations", 0)
    watch_payload.setdefault("stop_when_ready", True)
    watch_payload.setdefault("stop_when_rejected", True)
    watch_payload.setdefault("reconcile_when_ready", False)
    watch_payload.setdefault(
        "has_rejected_returns",
        bool(
            isinstance(bundle_report.get("rejected_return_summary"), dict)
            and (
                int(bundle_report["rejected_return_summary"].get("num_rejected_annotator_packets", 0) or 0)
                or int(bundle_report["rejected_return_summary"].get("num_rejected_adjudication_packets", 0) or 0)
            )
        ),
    )
    watch_payload.setdefault(
        "rejected_return_summary",
        dict(bundle_report.get("rejected_return_summary"))
        if isinstance(bundle_report.get("rejected_return_summary"), dict)
        else {},
    )
    watch_payload["next_command_id"] = next_action.get("next_command_id")
    watch_payload["next_command"] = next_action.get("next_command")
    watch_payload["next_script_id"] = next_action.get("next_script_id")
    watch_payload["next_script"] = next_action.get("next_script")
    watch_payload["next_script_file"] = next_action.get("next_script_file")
    watch_payload.setdefault("annotator_inboxes", [])
    watch_payload.setdefault("adjudication_inboxes", [])
    watch_payload.setdefault("last_sync_summary", None)
    watch_payload.setdefault("stop_reason", None)
    watch_payload.setdefault("stop_exit_code", None)
    watch_payload.setdefault("stop_action", None)
    write_json(watch_path, watch_payload)


def _bundle_rejected_returns_report_file(bundle_root: Path) -> str:
    return _relative_or_absolute(bundle_root / "rejected_returns_report.json", bundle_root)


def _bundle_pending_return_packets(
    bundle_root: Path,
    return_inbox: dict[str, str],
) -> tuple[list[str], list[str]]:
    pending_files: list[str] = []
    pending_paths: list[str] = []
    for raw_path in sorted(return_inbox.values()):
        inbox_path = _resolve_path(raw_path, bundle_root)
        if not inbox_path.exists():
            continue
        for packet_path in sorted(inbox_path.glob("*.zip")):
            pending_files.append(_relative_or_absolute(packet_path, bundle_root))
            pending_paths.append(str(packet_path.resolve()))
    return pending_files, pending_paths


def _bundle_return_inbox_state_path(bundle_root: Path) -> Path:
    return bundle_root / "return_inbox_state.json"


def _bundle_return_inbox_state_file(bundle_root: Path) -> str:
    return _relative_or_absolute(_bundle_return_inbox_state_path(bundle_root), bundle_root)


def _default_human_audit_return_inbox_state(bundle_root: Path) -> dict[str, Any]:
    return {
        "schema_version": HUMAN_AUDIT_RETURN_INBOX_STATE_SCHEMA_VERSION,
        "root": _bundle_root_ref(bundle_root),
        "bundle_dir": str(bundle_root),
        "annotator_packets": {},
        "adjudication_packet": None,
        "rejected_annotator_packets": [],
        "rejected_adjudication_packets": [],
    }


def _normalize_return_inbox_state_entry(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    packet_path = value.get("packet_path")
    packet_fingerprint = value.get("packet_fingerprint")
    if not isinstance(packet_path, str) or not packet_path:
        return None
    if not isinstance(packet_fingerprint, str) or not packet_fingerprint:
        return None
    normalized = {
        "packet_path": packet_path,
        "packet_fingerprint": packet_fingerprint,
    }
    source_kind = value.get("source_kind")
    if isinstance(source_kind, str) and source_kind:
        normalized["source_kind"] = source_kind
    processed_archive_file = value.get("processed_archive_file")
    if isinstance(processed_archive_file, str) and processed_archive_file:
        normalized["processed_archive_file"] = processed_archive_file
    processed_at = value.get("processed_at")
    if isinstance(processed_at, str) and processed_at:
        normalized["processed_at"] = processed_at
    return normalized


def _normalize_return_inbox_rejection_entry(value: Any) -> dict[str, Any] | None:
    normalized = _normalize_return_inbox_state_entry(value)
    if normalized is None:
        return None
    rejection_error = value.get("rejection_error") if isinstance(value, dict) else None
    if isinstance(rejection_error, str) and rejection_error:
        normalized["rejection_error"] = rejection_error
    rejected_archive_file = value.get("rejected_archive_file") if isinstance(value, dict) else None
    if isinstance(rejected_archive_file, str) and rejected_archive_file:
        normalized["rejected_archive_file"] = rejected_archive_file
    return normalized


def _load_human_audit_return_inbox_state(bundle_root: Path) -> dict[str, Any]:
    state_path = _bundle_return_inbox_state_path(bundle_root)
    if not state_path.exists():
        state = _default_human_audit_return_inbox_state(bundle_root)
        write_json(state_path, state)
        return state
    state = read_json(state_path)
    if not isinstance(state, dict):
        raise ValueError("return_inbox_state.json must be a JSON object")
    if state.get("schema_version") != HUMAN_AUDIT_RETURN_INBOX_STATE_SCHEMA_VERSION:
        raise ValueError("return_inbox_state.json schema_version is invalid")
    raw_annotator_packets = state.get("annotator_packets")
    annotator_packets: dict[str, dict[str, Any]] = {}
    if isinstance(raw_annotator_packets, dict):
        for annotator_id, value in sorted(raw_annotator_packets.items()):
            normalized = _normalize_return_inbox_state_entry(value)
            if normalized is not None:
                annotator_packets[str(annotator_id)] = normalized
    adjudication_packet = _normalize_return_inbox_state_entry(state.get("adjudication_packet"))
    raw_rejected_annotator_packets = state.get("rejected_annotator_packets")
    rejected_annotator_packets = []
    if isinstance(raw_rejected_annotator_packets, list):
        for entry in raw_rejected_annotator_packets:
            normalized = _normalize_return_inbox_rejection_entry(entry)
            if normalized is not None:
                rejected_annotator_packets.append(normalized)
    raw_rejected_adjudication_packets = state.get("rejected_adjudication_packets")
    rejected_adjudication_packets = []
    if isinstance(raw_rejected_adjudication_packets, list):
        for entry in raw_rejected_adjudication_packets:
            normalized = _normalize_return_inbox_rejection_entry(entry)
            if normalized is not None:
                rejected_adjudication_packets.append(normalized)
    normalized_state = {
        "schema_version": HUMAN_AUDIT_RETURN_INBOX_STATE_SCHEMA_VERSION,
        "root": _bundle_root_ref(bundle_root),
        "bundle_dir": str(bundle_root),
        "annotator_packets": annotator_packets,
        "adjudication_packet": adjudication_packet,
        "rejected_annotator_packets": rejected_annotator_packets,
        "rejected_adjudication_packets": rejected_adjudication_packets,
    }
    if normalized_state != state:
        write_json(state_path, normalized_state)
    return normalized_state


def _write_human_audit_return_inbox_state(bundle_root: Path, state: dict[str, Any]) -> Path:
    raw_annotator_packets = state.get("annotator_packets")
    annotator_packets: dict[str, dict[str, Any]] = {}
    if isinstance(raw_annotator_packets, dict):
        for annotator_id, entry in sorted(raw_annotator_packets.items()):
            normalized = _normalize_return_inbox_state_entry(entry)
            if normalized is not None:
                annotator_packets[str(annotator_id)] = normalized
    raw_rejected_annotator_packets = state.get("rejected_annotator_packets")
    rejected_annotator_packets = []
    if isinstance(raw_rejected_annotator_packets, list):
        for entry in raw_rejected_annotator_packets:
            normalized = _normalize_return_inbox_rejection_entry(entry)
            if normalized is not None:
                rejected_annotator_packets.append(normalized)
    raw_rejected_adjudication_packets = state.get("rejected_adjudication_packets")
    rejected_adjudication_packets = []
    if isinstance(raw_rejected_adjudication_packets, list):
        for entry in raw_rejected_adjudication_packets:
            normalized = _normalize_return_inbox_rejection_entry(entry)
            if normalized is not None:
                rejected_adjudication_packets.append(normalized)
    normalized_state = {
        "schema_version": HUMAN_AUDIT_RETURN_INBOX_STATE_SCHEMA_VERSION,
        "root": _bundle_root_ref(bundle_root),
        "bundle_dir": str(bundle_root),
        "annotator_packets": annotator_packets,
        "adjudication_packet": _normalize_return_inbox_state_entry(state.get("adjudication_packet")),
        "rejected_annotator_packets": rejected_annotator_packets,
        "rejected_adjudication_packets": rejected_adjudication_packets,
    }
    state_path = _bundle_return_inbox_state_path(bundle_root)
    write_json(state_path, normalized_state)
    return state_path


def _bundle_operator_script_preamble() -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'BUNDLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"\n'
        'REPO_ROOT_DEFAULT="."\n'
        'if [ -n "$REPO_ROOT_DEFAULT" ] && [ -d "$REPO_ROOT_DEFAULT/agent_memory_benchmark" ] && [ -f "$REPO_ROOT_DEFAULT/pyproject.toml" ]; then\n'
        '  REPO_ROOT="$REPO_ROOT_DEFAULT"\n'
        "else\n"
        '  REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"\n'
        "fi\n"
        'cd "$REPO_ROOT"\n\n'
    )


def _write_bundle_operator_scripts(bundle_root: Path) -> dict[str, str]:
    scripts_dir = bundle_root / "bin"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    preamble = _bundle_operator_script_preamble()
    watch_output = _bundle_return_inbox_watch_file(bundle_root)
    rejected_returns_output = _bundle_rejected_returns_report_file(bundle_root)
    scripts = {
        "progress": (
            "progress.sh",
            preamble
            + f'{HUMAN_AUDIT_CLI_PREFIX} human-audit-progress '
            + '--task-manifest "$BUNDLE_DIR/task_manifest.json" '
            + '--output "$BUNDLE_DIR/progress.json"\n',
        ),
        "watch_return_inbox": (
            "watch_return_inbox.sh",
            preamble
            + f'INTERVAL_S="${{AMST_HUMAN_AUDIT_WATCH_INTERVAL_S:-{HUMAN_AUDIT_OPERATOR_WATCH_INTERVAL_S}}}"\n'
            + 'MAX_ITERATIONS="${AMST_HUMAN_AUDIT_WATCH_MAX_ITERATIONS:-0}"\n'
            + 'STOP_WHEN_READY="${AMST_HUMAN_AUDIT_STOP_WHEN_READY:-1}"\n'
            + 'STOP_WHEN_REJECTED="${AMST_HUMAN_AUDIT_STOP_WHEN_REJECTED:-1}"\n'
            + 'STOP_WHEN_READY_FLAG=""\n'
            + 'if [ "$STOP_WHEN_READY" != "0" ] && [ "$STOP_WHEN_READY" != "false" ] && [ "$STOP_WHEN_READY" != "False" ]; then\n'
            + '  STOP_WHEN_READY_FLAG="--stop-when-ready"\n'
            + "fi\n"
            + 'STOP_WHEN_REJECTED_FLAG=""\n'
            + 'if [ "$STOP_WHEN_REJECTED" != "0" ] && [ "$STOP_WHEN_REJECTED" != "false" ] && [ "$STOP_WHEN_REJECTED" != "False" ]; then\n'
            + '  STOP_WHEN_REJECTED_FLAG="--stop-when-rejected"\n'
            + "fi\n"
            + f'{HUMAN_AUDIT_CLI_PREFIX} watch-human-audit-return-inbox '
            + '--bundle-dir "$BUNDLE_DIR" '
            + '--interval-s "$INTERVAL_S" '
            + '--max-iterations "$MAX_ITERATIONS" '
            + '$STOP_WHEN_READY_FLAG '
            + '$STOP_WHEN_REJECTED_FLAG '
            + f'--output "$BUNDLE_DIR/{watch_output}"\n',
        ),
        "sync_return_inbox": (
            "sync_return_inbox.sh",
            preamble
            + f'{HUMAN_AUDIT_CLI_PREFIX} sync-human-audit-return-inbox '
            + '--bundle-dir "$BUNDLE_DIR" '
            + '--reconcile-when-ready '
            + f'--signed-at {HUMAN_AUDIT_BUNDLE_SANDBOX_SIGNED_AT} '
            + f'--annotation-guideline "$BUNDLE_DIR/{HUMAN_AUDIT_BUNDLE_SANDBOX_ANNOTATION_GUIDELINE}" '
            + f"--adjudication-policy '{HUMAN_AUDIT_BUNDLE_SANDBOX_ADJUDICATION_POLICY}'\n",
        ),
        "review_rejected_returns": (
            "review_rejected_returns.sh",
            preamble
            + f'{HUMAN_AUDIT_CLI_PREFIX} summarize-human-audit-rejected-returns '
            + '--bundle-dir "$BUNDLE_DIR" '
            + f'--output "$BUNDLE_DIR/{rejected_returns_output}"\n',
        ),
        "reconcile_when_ready": (
            "reconcile_when_ready.sh",
            preamble
            + f'{HUMAN_AUDIT_CLI_PREFIX} reconcile-human-audit-evidence-bundle '
            + '--bundle-dir "$BUNDLE_DIR" '
            + f'--signed-at {HUMAN_AUDIT_BUNDLE_SANDBOX_SIGNED_AT} '
            + f'--annotation-guideline "$BUNDLE_DIR/{HUMAN_AUDIT_BUNDLE_SANDBOX_ANNOTATION_GUIDELINE}" '
            + f"--adjudication-policy '{HUMAN_AUDIT_BUNDLE_SANDBOX_ADJUDICATION_POLICY}'\n",
        ),
        "verify_bundle": (
            "verify_bundle.sh",
            preamble
            + f'{HUMAN_AUDIT_CLI_PREFIX} verify-human-audit-evidence-bundle '
            + '--bundle-dir "$BUNDLE_DIR" '
            + '--output "$BUNDLE_DIR/bundle_verification.json"\n',
        ),
    }
    script_refs: dict[str, str] = {}
    for script_id, (filename, content) in scripts.items():
        script_path = scripts_dir / filename
        script_path.write_text(content, encoding="utf-8")
        script_path.chmod(script_path.stat().st_mode | 0o111)
        script_refs[script_id] = _relative_or_absolute(script_path, bundle_root)
    return script_refs


def _build_handoff_manifest(bundle_report: dict[str, Any]) -> dict[str, Any]:
    bundle_dir = str(bundle_report.get("bundle_dir", ""))
    bundle_root = Path(bundle_dir) if bundle_dir else None
    operator_commands = (
        {str(key): str(value) for key, value in bundle_report.get("operator_commands", {}).items() if isinstance(value, str)}
        if isinstance(bundle_report.get("operator_commands"), dict)
        else {}
    )
    runtime_snapshot = _bundle_handoff_runtime_snapshot(bundle_root, bundle_report) if bundle_root is not None else {}
    return {
        "schema_version": "amst-human-audit-handoff-manifest-v1",
        "root": bundle_report.get("root"),
        "bundle_dir": bundle_dir,
        "bundle_manifest_file": f"{bundle_dir}/bundle_manifest.json" if bundle_dir else None,
        "status": bundle_report.get("status"),
        "ready_for_merge": bundle_report.get("ready_for_merge"),
        "ready_for_finalize": bundle_report.get("ready_for_finalize"),
        "annotator_packets": bundle_report.get("annotator_packets", {}),
        "adjudication_packet": bundle_report.get("adjudication_packet"),
        "annotation_sheet_files": bundle_report.get("annotation_sheet_files", {}),
        "sandbox_sample": bundle_report.get("sandbox_sample"),
        "return_inbox": bundle_report.get("return_inbox", {}),
        "return_archive": bundle_report.get("return_archive", {}),
        "return_reject_archive": bundle_report.get("return_reject_archive", {}),
        "rejected_return_summary": bundle_report.get("rejected_return_summary", {}),
        "rejected_returns_report_file": bundle_report.get("rejected_returns_report_file"),
        "return_inbox_watch_file": bundle_report.get("return_inbox_watch_file"),
        "return_inbox_sync_report_file": bundle_report.get("return_inbox_sync_report_file"),
        "return_inbox_state_file": bundle_report.get("return_inbox_state_file"),
        "pending_return_packets": bundle_report.get("pending_return_packets", []),
        "pending_return_packet_paths": bundle_report.get("pending_return_packet_paths", []),
        "watch_stop_exit_codes": bundle_report.get("watch_stop_exit_codes", {}),
        "watch_stop_actions": bundle_report.get("watch_stop_actions", {}),
        "recommended_next_commands": bundle_report.get("recommended_next_commands", []),
        "recommended_next_command_id": bundle_report.get("recommended_next_command_id"),
        "recommended_next_command": bundle_report.get("recommended_next_command"),
        "recommended_next_script_id": bundle_report.get("recommended_next_script_id"),
        "recommended_next_script": bundle_report.get("recommended_next_script"),
        "recommended_next_script_file": bundle_report.get("recommended_next_script_file"),
        "operator_scripts": bundle_report.get("operator_scripts", {}),
        "operator_commands": operator_commands,
        **runtime_snapshot,
    }


_HANDOFF_SIDECAR_NEXT_ACTION_FIELDS = (
    "next_command_id",
    "next_command",
    "next_script_id",
    "next_script",
    "next_script_file",
)


def _bundle_nonempty_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return str(value)
    return None


def _update_bundle_handoff_sidecar_snapshot(
    snapshot: dict[str, Any],
    prefix: str,
    payload: dict[str, Any] | None,
    *,
    fallback_status: str | None,
    fallback_next_action: dict[str, Any],
    include_stop_reason: bool = False,
) -> None:
    status = _bundle_nonempty_string(payload.get("status")) if isinstance(payload, dict) else None
    status = status or fallback_status
    if status is not None:
        snapshot[f"{prefix}_status"] = status
    if include_stop_reason and isinstance(payload, dict):
        stop_reason = _bundle_nonempty_string(payload.get("stop_reason"))
        if stop_reason is not None:
            snapshot[f"{prefix}_stop_reason"] = stop_reason
    for field in _HANDOFF_SIDECAR_NEXT_ACTION_FIELDS:
        value = _bundle_nonempty_string(payload.get(field)) if isinstance(payload, dict) else None
        if value is None:
            value = _bundle_nonempty_string(fallback_next_action.get(field))
        if value is not None:
            snapshot[f"{prefix}_{field}"] = value


def _bundle_handoff_runtime_snapshot(bundle_dir: Path, bundle_report: dict[str, Any]) -> dict[str, Any]:
    sync_payload = None
    raw_sync_path = bundle_report.get("return_inbox_sync_report_file")
    sync_path = (
        _resolve_path(raw_sync_path, bundle_dir)
        if isinstance(raw_sync_path, str) and raw_sync_path
        else bundle_dir / "return_inbox_sync.json"
    )
    if sync_path.exists():
        try:
            candidate = read_json(sync_path)
        except Exception:  # noqa: BLE001 - malformed sidecars should degrade to fallback fields
            candidate = None
        sync_payload = candidate if isinstance(candidate, dict) else None

    watch_payload = None
    raw_watch_path = bundle_report.get("return_inbox_watch_file")
    watch_path = (
        _resolve_path(raw_watch_path, bundle_dir)
        if isinstance(raw_watch_path, str) and raw_watch_path
        else bundle_dir / "return_inbox_watch.json"
    )
    if watch_path.exists():
        try:
            candidate = read_json(watch_path)
        except Exception:  # noqa: BLE001 - malformed sidecars should degrade to fallback fields
            candidate = None
        watch_payload = candidate if isinstance(candidate, dict) else None

    stop_reason = _bundle_nonempty_string(watch_payload.get("stop_reason")) if isinstance(watch_payload, dict) else None
    snapshot: dict[str, Any] = {}
    _update_bundle_handoff_sidecar_snapshot(
        snapshot,
        "return_inbox_sync",
        sync_payload,
        fallback_status="not_started",
        fallback_next_action=_bundle_recommended_next_refs(bundle_report),
    )
    _update_bundle_handoff_sidecar_snapshot(
        snapshot,
        "return_inbox_watch",
        watch_payload,
        fallback_status="not_started",
        fallback_next_action=_bundle_watch_next_action(bundle_report, stop_reason=stop_reason),
        include_stop_reason=True,
    )
    return snapshot


def reconcile_human_audit_evidence_bundle(
    bundle_dir: str | Path,
    *,
    source_manifest_path: str | Path | None = None,
    source_task_manifest_path: str | Path | None = None,
    source_annotations_output: str | Path | None = None,
    source_attestation_output: str | Path | None = None,
    source_adjudication_output: str | Path | None = None,
    source_agreement_output: str | Path | None = None,
    annotation_guideline: str | None = None,
    adjudication_policy: str | None = None,
    signed_at: str | None = None,
) -> dict[str, Any]:
    """Reconcile a completed human-audit bundle back into its canonical source release artifacts."""

    bundle_root = Path(bundle_dir)
    bundle_manifest_path = bundle_root / "bundle_manifest.json"
    bundle_manifest = read_json(bundle_manifest_path)
    if not isinstance(bundle_manifest, dict):
        raise ValueError("bundle_manifest.json must be a JSON object")

    bundle_verification = verify_human_audit_evidence_bundle(bundle_root)
    if not bundle_verification.get("ok"):
        raise ValueError("cannot reconcile invalid human-audit bundle: " + "; ".join(bundle_verification.get("errors", [])[:5]))

    bundle_task_manifest_path = _resolve_path(str(bundle_manifest.get("task_manifest_file", "")), bundle_root)
    bundle_annotations_value = bundle_manifest.get("annotations_file")
    bundle_annotations_path = _resolve_path(str(bundle_annotations_value), bundle_root) if bundle_annotations_value else None
    bundle_attestation_value = bundle_manifest.get("annotator_attestation_file")
    bundle_attestation_path = _resolve_path(str(bundle_attestation_value), bundle_root) if bundle_attestation_value else None
    bundle_adjudication_value = bundle_manifest.get("adjudication_file")
    bundle_adjudication_path = _resolve_path(str(bundle_adjudication_value), bundle_root) if bundle_adjudication_value else None

    if bundle_annotations_path is None:
        progress = summarize_human_audit_progress(bundle_task_manifest_path)
        if progress.get("ready_for_merge") is not True:
            raise ValueError(
                "bundle is not ready to merge completed annotations: "
                f"status={progress.get('status')} errors={progress.get('errors', [])[:5]}"
            )
        merge_completed_human_audit_tasks(bundle_task_manifest_path, bundle_root / "completed_annotations.jsonl")
        bundle_annotations_path = bundle_root / "completed_annotations.jsonl"

    progress_with_annotations = summarize_human_audit_progress(
        bundle_task_manifest_path,
        annotations_path=bundle_annotations_path,
    )
    if progress_with_annotations.get("ready_for_finalize") is not True:
        raise ValueError(
            "bundle annotations are not ready for finalize: "
            f"status={progress_with_annotations.get('status')} errors={progress_with_annotations.get('errors', [])[:5]}"
        )

    if bundle_attestation_path is None:
        guideline_value = str(annotation_guideline or "").strip()
        adjudication_value = str(adjudication_policy or "").strip()
        signed_at_value = str(signed_at or "").strip()
        if not guideline_value or not adjudication_value or not signed_at_value:
            raise ValueError(
                "annotation_guideline, adjudication_policy, and signed_at are required "
                "when reconciling a bundle without an attestation"
            )
        build_human_audit_attestation(
            bundle_task_manifest_path,
            bundle_annotations_path,
            bundle_root / "attestation.json",
            annotation_guideline=guideline_value,
            adjudication_policy=adjudication_value,
            signed_at=signed_at_value,
        )
        bundle_attestation_path = bundle_root / "attestation.json"

    resolved_source_manifest = (
        Path(source_manifest_path)
        if source_manifest_path is not None
        else _bundle_required_source_path(bundle_manifest, "source_manifest_file")
    )
    resolved_source_task_manifest = (
        Path(source_task_manifest_path)
        if source_task_manifest_path is not None
        else _bundle_required_source_path(bundle_manifest, "source_task_manifest_file")
    )
    if not resolved_source_manifest.exists():
        raise ValueError(f"source manifest does not exist: {resolved_source_manifest}")
    if not resolved_source_task_manifest.exists():
        raise ValueError(f"source task manifest does not exist: {resolved_source_task_manifest}")

    source_manifest_payload = read_json(resolved_source_manifest)
    source_audit_plan = (
        source_manifest_payload.get("audit_plan", {})
        if isinstance(source_manifest_payload, dict) and isinstance(source_manifest_payload.get("audit_plan"), dict)
        else {}
    )
    source_task_manifest_dir = resolved_source_task_manifest.parent
    resolved_source_annotations = _reconcile_output_path(
        explicit_path=source_annotations_output,
        manifest_path=resolved_source_manifest,
        manifest_value=source_audit_plan.get("audit_annotations_file"),
        fallback_path=source_task_manifest_dir / "completed_annotations.jsonl",
    )
    resolved_source_attestation = _reconcile_output_path(
        explicit_path=source_attestation_output,
        manifest_path=resolved_source_manifest,
        manifest_value=source_audit_plan.get("annotator_attestation_file"),
        fallback_path=source_task_manifest_dir / "attestation.json",
    )
    resolved_source_adjudication = None
    if bundle_adjudication_path is not None:
        resolved_source_adjudication = _reconcile_output_path(
            explicit_path=source_adjudication_output,
            manifest_path=resolved_source_manifest,
            manifest_value=source_audit_plan.get("audit_adjudication_file"),
            fallback_path=source_task_manifest_dir / "adjudication_tasks.jsonl",
        )
    resolved_source_agreement = None
    if source_agreement_output is not None or source_audit_plan.get("agreement_metrics_file"):
        resolved_source_agreement = _reconcile_output_path(
            explicit_path=source_agreement_output,
            manifest_path=resolved_source_manifest,
            manifest_value=source_audit_plan.get("agreement_metrics_file"),
            fallback_path=source_task_manifest_dir / "agreement_metrics.json",
        )

    _sync_bundle_task_files_to_source(bundle_task_manifest_path, resolved_source_task_manifest)
    _copy_optional_bundle_artifact(bundle_annotations_path, resolved_source_annotations)
    _copy_optional_bundle_artifact(bundle_attestation_path, resolved_source_attestation)
    _rewrite_attestation_paths(
        resolved_source_attestation,
        task_manifest_path=resolved_source_task_manifest,
        annotations_path=resolved_source_annotations,
    )
    if bundle_adjudication_path is not None and resolved_source_adjudication is not None:
        _copy_optional_bundle_artifact(bundle_adjudication_path, resolved_source_adjudication)

    source_finalize_report = finalize_human_audit_manifest(
        resolved_source_manifest,
        resolved_source_annotations,
        task_manifest_path=resolved_source_task_manifest,
        annotator_attestation_path=resolved_source_attestation,
        agreement_output=resolved_source_agreement,
        adjudication_path=resolved_source_adjudication,
    )

    rebuilt_bundle = build_human_audit_evidence_bundle(
        bundle_root,
        manifest_path=resolved_source_manifest,
        task_manifest_path=resolved_source_task_manifest,
        annotations_path=resolved_source_annotations,
        annotator_attestation_path=resolved_source_attestation,
        adjudication_path=resolved_source_adjudication,
    )
    rebuilt_bundle_verification = verify_human_audit_evidence_bundle(bundle_root)
    write_json(bundle_root / "bundle_verification.json", rebuilt_bundle_verification)

    return {
        "schema_version": "amst-human-audit-bundle-reconcile-v1",
        "bundle_dir": str(bundle_root),
        "source_manifest_file": str(resolved_source_manifest),
        "source_task_manifest_file": str(resolved_source_task_manifest),
        "source_annotations_file": str(resolved_source_annotations),
        "source_annotator_attestation_file": str(resolved_source_attestation),
        "source_adjudication_file": str(resolved_source_adjudication) if resolved_source_adjudication is not None else None,
        "source_agreement_metrics_file": str(resolved_source_agreement) if resolved_source_agreement is not None else None,
        "bundle_annotations_file": str(bundle_annotations_path),
        "bundle_attestation_file": str(bundle_attestation_path),
        "bundle_adjudication_file": str(bundle_adjudication_path) if bundle_adjudication_path is not None else None,
        "source_finalize_ok": source_finalize_report.get("ok"),
        "bundle_verification_ok": rebuilt_bundle_verification.get("ok"),
        "bundle_status": rebuilt_bundle.get("progress", {}).get("status"),
        "source_verification_report": source_finalize_report,
    }


def ingest_human_audit_return_packets(
    bundle_dir: str | Path,
    packet_paths: Iterable[str | Path],
    *,
    reconcile_when_ready: bool = False,
    signed_at: str | None = None,
    annotation_guideline: str | None = None,
    adjudication_policy: str | None = None,
) -> dict[str, Any]:
    """Apply one or more returned annotator packets into a bundle-local task package."""

    bundle_root = Path(bundle_dir)
    bundle_manifest_path = bundle_root / "bundle_manifest.json"
    bundle_manifest = read_json(bundle_manifest_path)
    if not isinstance(bundle_manifest, dict):
        raise ValueError("bundle_manifest.json must be a JSON object")

    task_manifest_value = bundle_manifest.get("task_manifest_file")
    if not isinstance(task_manifest_value, str) or not task_manifest_value:
        raise ValueError("bundle manifest does not include task_manifest_file")
    task_manifest_path = _resolve_path(task_manifest_value, bundle_root)

    packets = [Path(path) for path in packet_paths]
    if not packets:
        raise ValueError("at least one returned annotator packet is required")

    applied_annotators: set[str] = set()
    apply_summaries: list[dict[str, Any]] = []
    for packet_path in packets:
        summary = apply_human_audit_annotator_packet(task_manifest_path, packet_path)
        annotator = str(summary.get("annotator_id"))
        if annotator in applied_annotators:
            raise ValueError(f"duplicate returned packet for annotator {annotator!r}")
        applied_annotators.add(annotator)
        apply_summaries.append(summary)

    refreshed_bundle = _refresh_human_audit_bundle_local_state(bundle_root, bundle_manifest)
    progress = refreshed_bundle.get("progress", {})
    reconcile_report: dict[str, Any] | None = None
    reconcile_status = "not_requested"
    if reconcile_when_ready:
        verification = refreshed_bundle.get("verification")
        if isinstance(verification, dict) and verification.get("adjudication_recommended") and not refreshed_bundle.get("adjudication_file"):
            reconcile_status = "skipped_adjudication_required"
        elif progress.get("ready_for_merge") or progress.get("ready_for_finalize"):
            reconcile_report = reconcile_human_audit_evidence_bundle(
                bundle_root,
                signed_at=signed_at,
                annotation_guideline=annotation_guideline,
                adjudication_policy=adjudication_policy,
            )
            reconcile_status = "completed"
        else:
            reconcile_status = "skipped_not_ready"

    return {
        "schema_version": "amst-human-audit-packet-ingest-v1",
        "bundle_dir": str(bundle_root),
        "task_manifest_file": str(task_manifest_path),
        "num_packets": len(apply_summaries),
        "applied_annotators": sorted(applied_annotators),
        "packet_summaries": apply_summaries,
        "progress": progress,
        "ready_for_merge": progress.get("ready_for_merge"),
        "ready_for_finalize": progress.get("ready_for_finalize"),
        "reconcile_requested": reconcile_when_ready,
        "reconcile_status": reconcile_status,
        "reconcile_report": reconcile_report,
    }


def _task_manifest_snapshot(
    task_manifest_path: Path,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = manifest if manifest is not None else read_json(task_manifest_path)
    if not isinstance(payload, dict):
        raise ValueError("human-audit task manifest must be a JSON object")
    raw_template_files = payload.get("template_files")
    if not isinstance(raw_template_files, list) or not raw_template_files:
        raise ValueError("human-audit task manifest must include non-empty template_files")
    template_digest_values: list[str] = []
    for raw_path in raw_template_files:
        resolved = _resolve_path(str(raw_path), task_manifest_path.parent)
        digest = _file_sha256(resolved)
        if digest is not None:
            template_digest_values.append(digest)
    raw_task_identity_digests = payload.get("task_identity_digests")
    task_identity_digests = (
        {str(key): str(value) for key, value in sorted(raw_task_identity_digests.items())}
        if isinstance(raw_task_identity_digests, dict)
        else {}
    )
    raw_annotator_ids = payload.get("annotator_ids")
    annotator_ids = sorted(str(value) for value in raw_annotator_ids) if isinstance(raw_annotator_ids, list) else []
    return {
        "num_template_items": int(payload.get("num_template_items", 0)),
        "expected_annotations": int(payload.get("expected_annotations", 0)),
        "num_annotators": int(payload.get("num_annotators", 0)),
        "annotator_ids": annotator_ids,
        "template_digest_values": sorted(template_digest_values),
        "task_identity_digests": task_identity_digests,
    }


def _snapshot_signature(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    raw_task_identity = value.get("task_identity_digests")
    raw_annotators = value.get("annotator_ids")
    raw_template_digests = value.get("template_digest_values")
    return {
        "num_template_items": int(value.get("num_template_items", 0)),
        "expected_annotations": int(value.get("expected_annotations", 0)),
        "num_annotators": int(value.get("num_annotators", 0)),
        "annotator_ids": sorted(str(item) for item in raw_annotators) if isinstance(raw_annotators, list) else [],
        "template_digest_values": sorted(str(item) for item in raw_template_digests) if isinstance(raw_template_digests, list) else [],
        "task_identity_digests": (
            {str(key): str(raw_task_identity[key]) for key in sorted(raw_task_identity)}
            if isinstance(raw_task_identity, dict)
            else {}
        ),
    }


def _record_bundle_check(
    checks: dict[str, Any],
    errors: list[str],
    check_id: str,
    actual: Any,
    expected: Any,
) -> None:
    passed = actual == expected
    checks[check_id] = {
        "passed": passed,
        "actual": actual,
        "expected": expected,
    }
    if not passed:
        errors.append(f"{check_id} mismatch")


def _recompute_bundle_verification(bundle_dir: Path, bundle_manifest: dict[str, Any]) -> dict[str, Any]:
    raw_template_files = bundle_manifest.get("template_files")
    if not isinstance(raw_template_files, list) or not raw_template_files:
        raise ValueError("bundle manifest must include non-empty template_files")
    template_paths = tuple(_resolve_path(str(path), bundle_dir) for path in raw_template_files)
    annotations_value = bundle_manifest.get("annotations_file")
    if not annotations_value:
        raise ValueError("bundle manifest does not include annotations_file for verification recomputation")
    annotations_path = _resolve_path(str(annotations_value), bundle_dir)
    task_manifest_value = bundle_manifest.get("task_manifest_file")
    if not task_manifest_value:
        raise ValueError("bundle manifest does not include task_manifest_file")
    task_manifest_path = _resolve_path(str(task_manifest_value), bundle_dir)
    attestation_value = bundle_manifest.get("annotator_attestation_file")
    attestation_path = _resolve_path(str(attestation_value), bundle_dir) if attestation_value else None
    adjudication_value = bundle_manifest.get("adjudication_file")
    adjudication_path = _resolve_path(str(adjudication_value), bundle_dir) if adjudication_value else None
    release_manifest_value = bundle_manifest.get("release_manifest_file")
    release_manifest_path = _resolve_path(str(release_manifest_value), bundle_dir) if release_manifest_value else None
    benchmark_id = bundle_manifest.get("benchmark_id")

    if release_manifest_path is not None and release_manifest_path.exists():
        payload = read_json(release_manifest_path)
        audit_plan = payload.get("audit_plan", {}) if isinstance(payload, dict) else {}
        if audit_plan.get("human_audit_status") == "completed" and audit_plan.get("agreement_metrics"):
            return verify_manifest_human_audit(release_manifest_path)
    return verify_completed_human_audit(
        template_paths,
        annotations_path,
        task_manifest_path=task_manifest_path,
        annotator_attestation_path=attestation_path,
        adjudication_path=adjudication_path,
        benchmark_id=str(benchmark_id) if benchmark_id is not None else None,
        manifest_path=release_manifest_path,
    )


def _refresh_human_audit_bundle_local_state(bundle_root: Path, bundle_manifest: dict[str, Any]) -> dict[str, Any]:
    task_manifest_value = bundle_manifest.get("task_manifest_file")
    if not isinstance(task_manifest_value, str) or not task_manifest_value:
        raise ValueError("bundle manifest does not include task_manifest_file")
    task_manifest_path = _resolve_path(task_manifest_value, bundle_root)
    annotations_value = bundle_manifest.get("annotations_file")
    annotations_path = _resolve_path(str(annotations_value), bundle_root) if annotations_value else None
    attestation_value = bundle_manifest.get("annotator_attestation_file")
    attestation_path = _resolve_path(str(attestation_value), bundle_root) if attestation_value else None
    adjudication_value = bundle_manifest.get("adjudication_file")
    adjudication_path = _resolve_path(str(adjudication_value), bundle_root) if adjudication_value else None
    release_manifest_value = bundle_manifest.get("release_manifest_file")
    release_manifest_path = _resolve_path(str(release_manifest_value), bundle_root) if release_manifest_value else None
    benchmark_id = bundle_manifest.get("benchmark_id")

    progress = summarize_human_audit_progress(task_manifest_path, annotations_path=annotations_path)
    if annotations_path is None and progress.get("ready_for_merge") is True:
        annotations_path = bundle_root / "completed_annotations.jsonl"
        merge_summary = merge_completed_human_audit_tasks(task_manifest_path, annotations_path)
        bundle_manifest["annotations_file"] = str(annotations_path)
        bundle_manifest["merge_summary"] = merge_summary
        progress = summarize_human_audit_progress(task_manifest_path, annotations_path=annotations_path)
    progress_path = bundle_root / "progress.json"

    annotation_sheet_summary = write_human_audit_annotation_sheets(task_manifest_path, bundle_root / "sheets")
    bundle_annotation_sheet_files = {
        annotator_id: _relative_or_absolute(Path(path), bundle_root)
        for annotator_id, path in sorted(annotation_sheet_summary["annotation_sheet_files"].items())
    }
    bundle_annotation_sheet_summary = dict(annotation_sheet_summary)
    bundle_annotation_sheet_summary["output_dir"] = _relative_or_absolute(Path(annotation_sheet_summary["output_dir"]), bundle_root)
    bundle_annotation_sheet_summary["annotation_sheet_files"] = bundle_annotation_sheet_files
    return_inbox = _ensure_bundle_return_inbox(bundle_root)
    return_archive = _ensure_bundle_return_archive(bundle_root)
    return_reject_archive = _ensure_bundle_return_reject_archive(bundle_root)
    return_inbox_state_file = _bundle_return_inbox_state_file(bundle_root)
    return_inbox_state = _load_human_audit_return_inbox_state(bundle_root)
    rejected_return_summary = _rejected_return_summary(return_inbox_state)
    rejected_returns_report_file = _bundle_rejected_returns_report_file(bundle_root)
    write_json(bundle_root / "rejected_returns_report.json", summarize_human_audit_rejected_returns(bundle_root))
    sandbox_sample = _bundle_relative_sandbox_sample(
        build_human_audit_sandbox_sample(
            task_manifest_path,
            bundle_root / "sandbox",
            num_items=1,
            annotation_guideline=HUMAN_AUDIT_BUNDLE_SANDBOX_ANNOTATION_GUIDELINE,
            adjudication_policy=HUMAN_AUDIT_BUNDLE_SANDBOX_ADJUDICATION_POLICY,
            signed_at=HUMAN_AUDIT_BUNDLE_SANDBOX_SIGNED_AT,
        ),
        bundle_root,
    )

    task_manifest_payload = read_json(task_manifest_path)
    task_files = task_manifest_payload.get("task_files", {}) if isinstance(task_manifest_payload, dict) else {}
    documentation_files = (
        {str(key): str(value) for key, value in sorted(bundle_manifest.get("documentation_files", {}).items())}
        if isinstance(bundle_manifest.get("documentation_files"), dict)
        else {}
    )
    annotator_packets = _build_annotator_packets(
        bundle_root,
        {str(key): str(value) for key, value in sorted(task_files.items())},
        bundle_annotation_sheet_files,
        documentation_files,
        sandbox_sample,
        progress.get("annotator_progress"),
    )

    verification = None
    verification_path = None
    adjudication_package = None
    if annotations_path is not None and annotations_path.exists():
        raw_template_files = bundle_manifest.get("template_files")
        if not isinstance(raw_template_files, list) or not raw_template_files:
            raise ValueError("bundle manifest must include non-empty template_files")
        template_paths = tuple(_resolve_path(str(path), bundle_root) for path in raw_template_files)
        verification = verify_completed_human_audit(
            template_paths,
            annotations_path,
            task_manifest_path=task_manifest_path,
            annotator_attestation_path=attestation_path,
            adjudication_path=adjudication_path,
            benchmark_id=str(benchmark_id) if benchmark_id is not None else None,
            manifest_path=release_manifest_path,
        )
        verification_path = bundle_root / "verification_report.json"
        write_json(verification_path, verification)
        bundle_manifest["verification_report_file"] = str(verification_path)
        bundle_manifest["verification"] = verification
        if verification.get("agreement_metrics") is not None:
            agreement_path = bundle_root / "agreement_metrics.json"
            write_json(agreement_path, verification["agreement_metrics"])
            bundle_manifest["agreement_metrics_file"] = str(agreement_path)
        if verification.get("adjudication_recommended") and adjudication_path is None:
            adjudication_package = _bundle_relative_adjudication_package(
                build_human_audit_adjudication_package(
                    task_manifest_path,
                    annotations_path,
                    bundle_root / "adjudication",
                ),
                bundle_root,
            )
    else:
        bundle_manifest["verification"] = None
        bundle_manifest["verification_report_file"] = None
    bundle_manifest["adjudication_package"] = adjudication_package
    adjudication_packet = _build_adjudication_packet(
        bundle_root,
        adjudication_package,
        documentation_files,
    )
    bundle_manifest["adjudication_packet"] = adjudication_packet
    pending_return_packets, pending_return_packet_paths = _bundle_pending_return_packets(bundle_root, return_inbox)
    recommended_next_commands = _bundle_next_commands(
        bundle_root,
        release_manifest_path,
        task_manifest_path,
        annotations_path,
        attestation_path,
        adjudication_path,
        progress,
        verification if isinstance(verification, dict) else None,
        rejected_return_summary=rejected_return_summary,
        pending_return_packets=pending_return_packets,
    )
    operator_scripts = _write_bundle_operator_scripts(bundle_root)
    operator_commands = _bundle_operator_commands(bundle_root, str(task_manifest_path))
    watch_stop_actions = _bundle_watch_stop_actions(bundle_root, operator_scripts)
    return_inbox_watch_file = _bundle_return_inbox_watch_file(bundle_root)
    return_inbox_sync_report_file = _bundle_return_inbox_sync_report_file(bundle_root)
    bundle_progress = _bundle_progress_payload(
        bundle_root,
        progress,
        annotation_sheet_files=bundle_annotation_sheet_files,
        annotator_packets=annotator_packets,
        adjudication_packet=adjudication_packet,
        sandbox_sample=sandbox_sample,
        return_inbox=return_inbox,
        return_archive=return_archive,
        return_reject_archive=return_reject_archive,
        rejected_return_summary=rejected_return_summary,
        rejected_returns_report_file=rejected_returns_report_file,
        return_inbox_watch_file=return_inbox_watch_file,
        return_inbox_sync_report_file=return_inbox_sync_report_file,
        return_inbox_state_file=return_inbox_state_file,
        pending_return_packets=pending_return_packets,
        pending_return_packet_paths=pending_return_packet_paths,
        watch_stop_exit_codes=HUMAN_AUDIT_WATCH_STOP_EXIT_CODES,
        watch_stop_actions=watch_stop_actions,
        operator_scripts=operator_scripts,
        operator_commands=operator_commands,
        recommended_next_commands=recommended_next_commands,
        recommended_next_command_id=None,
        recommended_next_command=None,
        recommended_next_script_id=None,
        recommended_next_script=None,
        recommended_next_script_file=None,
    )
    recommended_next_action = _bundle_recommended_next_action(bundle_root, operator_scripts, recommended_next_commands)
    bundle_progress["recommended_next_command_id"] = recommended_next_action["recommended_next_command_id"]
    bundle_progress["recommended_next_command"] = recommended_next_action["recommended_next_command"]
    bundle_progress["recommended_next_script_id"] = recommended_next_action["recommended_next_script_id"]
    bundle_progress["recommended_next_script"] = recommended_next_action["recommended_next_script"]
    bundle_progress["recommended_next_script_file"] = recommended_next_action["recommended_next_script_file"]
    write_json(progress_path, bundle_progress)
    bundle_manifest["status"] = bundle_progress.get("status")
    bundle_manifest["ready_for_merge"] = bundle_progress.get("ready_for_merge")
    bundle_manifest["ready_for_finalize"] = bundle_progress.get("ready_for_finalize")
    bundle_manifest["annotator_ids"] = bundle_progress.get("expected_annotator_ids", [])
    bundle_manifest["num_annotators"] = bundle_progress.get("num_expected_annotators")
    bundle_manifest["num_expected_annotations"] = bundle_progress.get("num_expected_annotations")
    bundle_manifest["num_matched_annotations"] = bundle_progress.get("num_matched_annotations")
    bundle_manifest["num_missing_annotations"] = bundle_progress.get("num_missing_annotations")
    bundle_manifest["progress_file"] = str(progress_path)
    bundle_manifest["progress"] = bundle_progress
    bundle_manifest["annotation_sheet_files"] = bundle_annotation_sheet_files
    bundle_manifest["annotation_sheet_summary"] = bundle_annotation_sheet_summary
    bundle_manifest["sandbox_sample"] = sandbox_sample
    bundle_manifest["annotator_packets"] = annotator_packets
    bundle_manifest["return_inbox"] = return_inbox
    bundle_manifest["return_archive"] = return_archive
    bundle_manifest["return_reject_archive"] = return_reject_archive
    bundle_manifest["rejected_return_summary"] = rejected_return_summary
    bundle_manifest["rejected_returns_report_file"] = rejected_returns_report_file
    bundle_manifest["return_inbox_watch_file"] = return_inbox_watch_file
    bundle_manifest["return_inbox_sync_report_file"] = return_inbox_sync_report_file
    bundle_manifest["return_inbox_state_file"] = return_inbox_state_file
    bundle_manifest["pending_return_packets"] = pending_return_packets
    bundle_manifest["pending_return_packet_paths"] = pending_return_packet_paths
    bundle_manifest["watch_stop_exit_codes"] = dict(HUMAN_AUDIT_WATCH_STOP_EXIT_CODES)
    bundle_manifest["watch_stop_actions"] = watch_stop_actions
    bundle_manifest["operator_scripts"] = operator_scripts
    bundle_manifest["operator_commands"] = operator_commands
    bundle_manifest["recommended_next_commands"] = recommended_next_commands
    bundle_manifest["recommended_next_command_id"] = recommended_next_action["recommended_next_command_id"]
    bundle_manifest["recommended_next_command"] = recommended_next_action["recommended_next_command"]
    bundle_manifest["recommended_next_script_id"] = recommended_next_action["recommended_next_script_id"]
    bundle_manifest["recommended_next_script"] = recommended_next_action["recommended_next_script"]
    bundle_manifest["recommended_next_script_file"] = recommended_next_action["recommended_next_script_file"]
    bundle_manifest["root"] = _bundle_root_ref(bundle_root)
    bundle_manifest["handoff_manifest_file"] = str(bundle_root / "handoff_manifest.json")
    write_json(bundle_root / "bundle_manifest.json", bundle_manifest)
    _write_initial_human_audit_return_inbox_sidecars(
        bundle_root,
        bundle_report=bundle_manifest,
        operator_scripts=operator_scripts,
    )
    write_json(bundle_root / "handoff_manifest.json", _build_handoff_manifest(bundle_manifest))
    readme_path = bundle_root / "README.md"
    readme_path.write_text(_bundle_readme(bundle_manifest), encoding="utf-8")
    return bundle_manifest


def _bundle_required_source_path(bundle_manifest: dict[str, Any], field: str) -> Path:
    raw_value = bundle_manifest.get(field)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"bundle manifest is missing required {field}")
    return Path(raw_value)


def _bundle_relative_adjudication_package(package: dict[str, Any], bundle_root: Path) -> dict[str, Any]:
    normalized = json.loads(json.dumps(package))
    for key in (
        "task_manifest_file",
        "annotations_file",
        "adjudication_tasks_file",
        "adjudication_manifest_file",
        "adjudication_summary_file",
    ):
        value = normalized.get(key)
        if isinstance(value, str) and value:
            normalized[key] = _relative_or_absolute(Path(value), bundle_root)
    template_files = normalized.get("template_files")
    if isinstance(template_files, list):
        normalized["template_files"] = [_relative_or_absolute(Path(path), bundle_root) for path in template_files]
    return normalized


def _rebase_human_audit_path(
    raw_path: str,
    *,
    source_bases: tuple[Path, ...],
    target_base: Path,
    require_exists: bool = True,
) -> str:
    if not raw_path:
        return raw_path
    resolved_path: Path | None = None
    for base_dir in source_bases:
        candidate = _resolve_path(raw_path, base_dir)
        if resolved_path is None:
            resolved_path = candidate
        if candidate.exists():
            resolved_path = candidate
            break
    if resolved_path is None:
        return raw_path
    if require_exists and not resolved_path.exists():
        return raw_path
    try:
        return Path(os.path.relpath(resolved_path.resolve(), target_base.resolve())).as_posix()
    except ValueError:
        return str(resolved_path)


def _localize_human_audit_task_manifest_payload(
    payload: dict[str, Any],
    *,
    source_bases: tuple[Path, ...],
    target_base: Path,
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload))
    template_files = normalized.get("template_files")
    if isinstance(template_files, list):
        normalized["template_files"] = [
            _rebase_human_audit_path(str(path), source_bases=source_bases, target_base=target_base)
            for path in template_files
        ]
    template_digests = normalized.get("template_file_digests")
    if isinstance(template_digests, dict):
        normalized["template_file_digests"] = {
            _rebase_human_audit_path(str(path), source_bases=source_bases, target_base=target_base): digest
            for path, digest in sorted(template_digests.items())
        }
    task_files = normalized.get("task_files")
    if isinstance(task_files, dict):
        normalized["task_files"] = {
            str(annotator_id): _rebase_human_audit_path(str(path), source_bases=source_bases, target_base=target_base)
            for annotator_id, path in sorted(task_files.items())
        }
    source_task_manifest = normalized.get("sandbox_source_task_manifest_file")
    if isinstance(source_task_manifest, str) and source_task_manifest:
        normalized["sandbox_source_task_manifest_file"] = _rebase_human_audit_path(
            source_task_manifest,
            source_bases=source_bases,
            target_base=target_base,
        )
    root_ref = _bundle_root_ref(target_base)
    if root_ref is not None:
        normalized["root"] = root_ref
    return normalized


def _localize_human_audit_merge_summary_payload(
    payload: dict[str, Any],
    *,
    source_bases: tuple[Path, ...],
    target_base: Path,
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload))
    for field in ("task_manifest_file", "annotations_file"):
        raw_path = normalized.get(field)
        if isinstance(raw_path, str) and raw_path:
            normalized[field] = _rebase_human_audit_path(raw_path, source_bases=source_bases, target_base=target_base)
    return normalized


def _localize_human_audit_attestation_payload(
    payload: dict[str, Any],
    *,
    source_bases: tuple[Path, ...],
    target_base: Path,
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload))
    for field in ("task_manifest_file", "annotations_file"):
        raw_path = normalized.get(field)
        if isinstance(raw_path, str) and raw_path:
            normalized[field] = _rebase_human_audit_path(raw_path, source_bases=source_bases, target_base=target_base)
    protocol = normalized.get("protocol")
    if isinstance(protocol, dict):
        raw_guideline = protocol.get("annotation_guideline")
        if isinstance(raw_guideline, str) and raw_guideline:
            protocol["annotation_guideline"] = _rebase_human_audit_path(
                raw_guideline,
                source_bases=source_bases,
                target_base=target_base,
            )
    root_ref = _bundle_root_ref(target_base)
    if root_ref is not None:
        normalized["root"] = root_ref
    return normalized


def _localize_human_audit_verification_payload(
    payload: dict[str, Any],
    *,
    source_bases: tuple[Path, ...],
    target_base: Path,
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload))
    template_files = normalized.get("template_files")
    if isinstance(template_files, list):
        normalized["template_files"] = [
            _rebase_human_audit_path(str(path), source_bases=source_bases, target_base=target_base)
            for path in template_files
        ]
    for field in (
        "manifest_path",
        "annotations_file",
        "task_manifest_file",
        "annotator_attestation_file",
        "adjudication_file",
    ):
        raw_path = normalized.get(field)
        if isinstance(raw_path, str) and raw_path:
            normalized[field] = _rebase_human_audit_path(raw_path, source_bases=source_bases, target_base=target_base)
    adjudication_summary = normalized.get("adjudication_summary")
    if isinstance(adjudication_summary, dict):
        adjudication_file = adjudication_summary.get("adjudication_file")
        if isinstance(adjudication_file, str) and adjudication_file:
            adjudication_summary["adjudication_file"] = _rebase_human_audit_path(
                adjudication_file,
                source_bases=source_bases,
                target_base=target_base,
            )
    file_digests = normalized.get("file_digests")
    if isinstance(file_digests, dict):
        template_digests = file_digests.get("template_files")
        if isinstance(template_digests, dict):
            file_digests["template_files"] = {
                _rebase_human_audit_path(str(path), source_bases=source_bases, target_base=target_base): digest
                for path, digest in sorted(template_digests.items())
            }
    root_ref = _bundle_root_ref(target_base)
    if root_ref is not None:
        normalized["root"] = root_ref
    return normalized


def _localize_human_audit_sandbox_sample_payload(
    sample: dict[str, Any],
    *,
    source_bases: tuple[Path, ...],
    target_base: Path,
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(sample))
    for field in (
        "source_task_manifest_file",
        "sandbox_task_manifest_file",
        "completed_annotations_file",
        "attestation_file",
        "agreement_file",
        "verification_report_file",
    ):
        raw_path = normalized.get(field)
        if isinstance(raw_path, str) and raw_path:
            normalized[field] = _rebase_human_audit_path(raw_path, source_bases=source_bases, target_base=target_base)
    merge_summary = normalized.get("merge_summary")
    if isinstance(merge_summary, dict):
        normalized["merge_summary"] = _localize_human_audit_merge_summary_payload(
            merge_summary,
            source_bases=source_bases,
            target_base=target_base,
        )
    attestation = normalized.get("attestation")
    if isinstance(attestation, dict):
        normalized["attestation"] = _localize_human_audit_attestation_payload(
            attestation,
            source_bases=source_bases,
            target_base=target_base,
        )
    verification = normalized.get("verification")
    if isinstance(verification, dict):
        normalized["verification"] = _localize_human_audit_verification_payload(
            verification,
            source_bases=source_bases,
            target_base=target_base,
        )
    root_ref = _bundle_root_ref(target_base)
    if root_ref is not None:
        normalized["root"] = root_ref
    return normalized


def _bundle_relative_sandbox_sample(sample: dict[str, Any], bundle_root: Path) -> dict[str, Any]:
    sandbox_dir = bundle_root / "sandbox"
    return _localize_human_audit_sandbox_sample_payload(
        sample,
        source_bases=(sandbox_dir, bundle_root),
        target_base=bundle_root,
    )


def _reconcile_output_path(
    *,
    explicit_path: str | Path | None,
    manifest_path: Path,
    manifest_value: str | Path | None,
    fallback_path: Path,
) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)
    if manifest_value is not None:
        return _resolve_path(str(manifest_value), manifest_path.parent)
    return fallback_path


def _sync_bundle_task_files_to_source(bundle_task_manifest_path: Path, source_task_manifest_path: Path) -> None:
    bundle_manifest = read_json(bundle_task_manifest_path)
    source_manifest = read_json(source_task_manifest_path)
    if not isinstance(bundle_manifest, dict) or not isinstance(source_manifest, dict):
        raise ValueError("human-audit task manifest must be a JSON object")
    bundle_task_files = bundle_manifest.get("task_files")
    source_task_files = source_manifest.get("task_files")
    if not isinstance(bundle_task_files, dict) or not isinstance(source_task_files, dict):
        raise ValueError("human-audit task manifest task_files must be a non-empty object")
    for annotator_id, source_raw_path in sorted(source_task_files.items()):
        annotator = str(annotator_id)
        bundle_raw_path = bundle_task_files.get(annotator)
        if not isinstance(bundle_raw_path, str) or not bundle_raw_path:
            raise ValueError(f"bundle task manifest is missing task file for annotator {annotator!r}")
        bundle_task_file = _resolve_task_path(bundle_task_manifest_path.parent, bundle_raw_path)
        source_task_file = _resolve_task_path(source_task_manifest_path.parent, str(source_raw_path))
        source_task_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle_task_file, source_task_file)


def _rewrite_attestation_paths(
    attestation_path: Path,
    *,
    task_manifest_path: Path,
    annotations_path: Path,
) -> None:
    attestation = read_json(attestation_path)
    if not isinstance(attestation, dict):
        raise ValueError("annotator attestation must be a JSON object")
    attestation["task_manifest_file"] = str(task_manifest_path)
    attestation["annotations_file"] = str(annotations_path)
    write_json(attestation_path, attestation)


def _resolve_bundle_artifact_path(
    explicit_path: str | Path | None,
    manifest_value: str | Path | None,
    manifest_dir: Path | None,
) -> Path | None:
    if explicit_path is not None:
        return Path(explicit_path)
    if manifest_value is None or manifest_dir is None:
        return None
    return _resolve_path(str(manifest_value), manifest_dir)


def _copy_optional_bundle_artifact(source: Path | None, target: Path) -> Path | None:
    if source is None:
        return None
    if not source.exists():
        raise ValueError(f"human-audit artifact does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _bundle_next_commands(
    bundle_dir: Path,
    manifest_path: Path | None,
    task_manifest_path: Path,
    annotations_path: Path | None,
    attestation_path: Path | None,
    adjudication_path: Path | None,
    progress: dict[str, Any],
    verification: dict[str, Any] | None,
    rejected_return_summary: dict[str, Any] | None = None,
    pending_return_packets: list[str] | None = None,
) -> list[str]:
    commands: list[str] = []
    task_manifest_arg = str(task_manifest_path)
    progress_output = str(bundle_dir / "progress.json")
    bundle_annotation_guideline = _bundle_annotation_guideline_cli_path(bundle_dir)
    reconcile_command = (
        f"{HUMAN_AUDIT_CLI_PREFIX} reconcile-human-audit-evidence-bundle "
        f"--bundle-dir {bundle_dir} "
        "--signed-at 2026-05-13T00:00:00Z "
        f"--annotation-guideline {bundle_annotation_guideline} "
        "--adjudication-policy 'Disagreements are adjudicated after double annotation.'"
    )
    manifest_completed = False
    if manifest_path is not None and manifest_path.exists():
        manifest_payload = read_json(manifest_path)
        audit_plan = manifest_payload.get("audit_plan", {}) if isinstance(manifest_payload, dict) else {}
        manifest_completed = audit_plan.get("human_audit_status") == "completed" and bool(audit_plan.get("agreement_metrics"))
    num_rejected_annotator_packets = 0
    num_rejected_adjudication_packets = 0
    if isinstance(rejected_return_summary, dict):
        num_rejected_annotator_packets = int(rejected_return_summary.get("num_rejected_annotator_packets", 0) or 0)
        num_rejected_adjudication_packets = int(rejected_return_summary.get("num_rejected_adjudication_packets", 0) or 0)
    if num_rejected_annotator_packets or num_rejected_adjudication_packets:
        commands.append(
            f"{HUMAN_AUDIT_CLI_PREFIX} summarize-human-audit-rejected-returns "
            f"--bundle-dir {bundle_dir} --output {bundle_dir / 'rejected_returns_report.json'}"
        )
        commands.append(
            f"{HUMAN_AUDIT_CLI_PREFIX} sync-human-audit-return-inbox "
            f"--bundle-dir {bundle_dir} "
            "--reconcile-when-ready "
            f"--signed-at {HUMAN_AUDIT_BUNDLE_SANDBOX_SIGNED_AT} "
            f"--annotation-guideline {bundle_annotation_guideline} "
            f"--adjudication-policy '{HUMAN_AUDIT_BUNDLE_SANDBOX_ADJUDICATION_POLICY}'"
        )
        return commands
    pending_packets = [str(item) for item in pending_return_packets or [] if isinstance(item, str) and item.strip()]
    if pending_packets:
        commands.append(_bundle_sync_return_inbox_command(bundle_dir))
        if annotations_path is not None:
            commands.append(
                f"{HUMAN_AUDIT_CLI_PREFIX} human-audit-progress "
                f"--task-manifest {task_manifest_arg} "
                f"--annotations {annotations_path} "
                f"--output {progress_output}"
            )
        else:
            commands.append(
                f"{HUMAN_AUDIT_CLI_PREFIX} human-audit-progress "
                f"--task-manifest {task_manifest_arg} "
                f"--output {progress_output}"
            )
        return commands
    if annotations_path is None:
        if progress.get("ready_for_merge") is True:
            commands.append(reconcile_command)
            return commands
        commands.append(
            f"{HUMAN_AUDIT_CLI_PREFIX} watch-human-audit-return-inbox "
            f"--bundle-dir {bundle_dir} --interval-s {HUMAN_AUDIT_OPERATOR_WATCH_INTERVAL_S} --max-iterations 0 "
            "--stop-when-ready "
            "--stop-when-rejected "
            f"--output {bundle_dir / 'return_inbox_watch.json'}"
        )
        commands.append(
            f"{HUMAN_AUDIT_CLI_PREFIX} human-audit-progress "
            f"--task-manifest {task_manifest_arg} "
            f"--output {progress_output}"
        )
        return commands

    annotations_arg = str(annotations_path)
    if verification and verification.get("adjudication_recommended") and adjudication_path is None:
        commands.append(
            f"{HUMAN_AUDIT_CLI_PREFIX} apply-human-audit-adjudication-packet "
            f"--bundle-dir {bundle_dir} --packet RETURNED_ADJUDICATION_PACKET.zip "
            "--reconcile-when-ready --signed-at 2026-05-13T00:00:00Z "
            f"--annotation-guideline {bundle_annotation_guideline} "
            "--adjudication-policy 'Disagreements are adjudicated after double annotation.'"
        )
        return commands
    if progress.get("ready_for_finalize") is True and attestation_path is None:
        commands.append(reconcile_command)
        return commands

    if manifest_completed and manifest_path is not None:
        commands.append(
            f"{HUMAN_AUDIT_CLI_PREFIX} verify-manifest-human-audit "
            f"--manifest {manifest_path} "
            f"--output {bundle_dir / 'verification_report.json'}"
        )
        return commands

    if progress.get("ready_for_finalize") is True and attestation_path is not None and manifest_path is not None:
        finalize = (
            f"{HUMAN_AUDIT_CLI_PREFIX} finalize-human-audit "
            f"--manifest {manifest_path} "
            f"--annotations {annotations_arg} "
            f"--task-manifest {task_manifest_arg} "
            f"--annotator-attestation {attestation_path} "
            f"--agreement-output {bundle_dir / 'agreement_metrics.json'}"
        )
        if adjudication_path is not None:
            finalize += f" --adjudication {adjudication_path}"
        commands.append(finalize)
        if manifest_path is not None:
            commands.append(
                f"{HUMAN_AUDIT_CLI_PREFIX} verify-manifest-human-audit "
                f"--manifest {manifest_path} "
                f"--output {bundle_dir / 'verification_report.json'}"
            )
        return commands

    commands.append(
        f"{HUMAN_AUDIT_CLI_PREFIX} human-audit-progress "
        f"--task-manifest {task_manifest_arg} "
        f"--annotations {annotations_arg} "
        f"--output {progress_output}"
    )
    if manifest_path is not None:
        commands.append(
            f"{HUMAN_AUDIT_CLI_PREFIX} verify-manifest-human-audit "
            f"--manifest {manifest_path} "
            f"--output {bundle_dir / 'verification_report.json'}"
        )
    return commands


def _bundle_readme(bundle_report: dict[str, Any]) -> str:
    commands = bundle_report.get("recommended_next_commands", [])
    command_text = "\n".join(f"- `{command}`" for command in commands) if commands else "- No further command is currently suggested."
    progress = bundle_report.get("progress", {}) if isinstance(bundle_report.get("progress"), dict) else {}
    verification = bundle_report.get("verification", {}) if isinstance(bundle_report.get("verification"), dict) else {}
    bundle_dir_value = bundle_report.get("bundle_dir")
    bundle_guideline_command_path = (
        str(Path(bundle_dir_value) / HUMAN_AUDIT_BUNDLE_SANDBOX_ANNOTATION_GUIDELINE)
        if isinstance(bundle_dir_value, str) and bundle_dir_value
        else HUMAN_AUDIT_BUNDLE_SANDBOX_ANNOTATION_GUIDELINE
    )
    annotator_progress = _bundle_progress_lines(progress.get("annotator_progress"), "annotator")
    domain_progress = _bundle_progress_lines(progress.get("domain_progress"), "domain")
    probe_progress = _bundle_progress_lines(progress.get("probe_type_progress"), "probe")
    documentation_text = _bundle_documentation_lines(bundle_report.get("documentation_files"))
    sandbox_sample_text = _bundle_sandbox_sample_lines(bundle_report.get("sandbox_sample"))
    handoff_manifest_text = _bundle_handoff_manifest_lines(bundle_report.get("handoff_manifest_file"))
    rejected_returns_text = _bundle_rejected_returns_lines(
        bundle_report.get("rejected_return_summary"),
        bundle_report.get("rejected_returns_report_file"),
    )
    return_inbox_text = _bundle_return_inbox_lines(
        bundle_report.get("return_inbox"),
        bundle_report.get("return_archive"),
        bundle_report.get("return_reject_archive"),
        bundle_report.get("return_inbox_state_file"),
    )
    operator_scripts_text = _bundle_operator_scripts_lines(bundle_report.get("operator_scripts"))
    annotation_sheet_text = _bundle_annotation_sheet_lines(bundle_report.get("annotation_sheet_files"))
    annotation_sheet_apply_text = _bundle_annotation_sheet_apply_lines(
        bundle_report.get("task_manifest_file"),
        bundle_report.get("annotation_sheet_files"),
    )
    annotator_packet_text = _bundle_annotator_packet_lines(bundle_report.get("annotator_packets"))
    annotator_packet_apply_text = _bundle_annotator_packet_apply_lines(
        bundle_report.get("task_manifest_file"),
        bundle_report.get("annotator_packets"),
    )
    annotator_packet_batch_ingest_text = _bundle_annotator_packet_batch_ingest_lines(
        bundle_report.get("bundle_dir"),
        bundle_report.get("return_inbox"),
    )
    return_inbox_watch_text = _bundle_return_inbox_watch_lines(bundle_report.get("bundle_dir"))
    watch_stop_actions_text = _bundle_watch_stop_actions_lines(bundle_report.get("watch_stop_actions"))
    adjudication_package_text = _bundle_adjudication_package_lines(bundle_report.get("adjudication_package"))
    adjudication_packet_text = _bundle_adjudication_packet_lines(bundle_report.get("adjudication_packet"))
    adjudication_packet_apply_text = _bundle_adjudication_packet_apply_lines(bundle_report.get("bundle_dir"))
    return_inbox_sync_text = _bundle_return_inbox_sync_lines(
        bundle_report.get("bundle_dir"),
        bundle_report.get("return_inbox"),
    )
    rejected_returns_review_text = _bundle_rejected_returns_review_lines(bundle_report.get("bundle_dir"))
    check_definition_text = _bundle_check_definition_lines(
        bundle_report.get("checks"),
        bundle_report.get("check_definitions"),
    )
    return (
        "# Human Audit Evidence Bundle\n\n"
        "This directory materializes a self-contained AutoMemoryBench human-audit package.\n\n"
        "## Summary\n\n"
        f"- `task_manifest`: `{bundle_report.get('task_manifest_file')}`\n"
        f"- `release_manifest`: `{bundle_report.get('release_manifest_file')}`\n"
        f"- `annotations`: `{bundle_report.get('annotations_file')}`\n"
        f"- `attestation`: `{bundle_report.get('annotator_attestation_file')}`\n"
        f"- `adjudication`: `{bundle_report.get('adjudication_file')}`\n"
        f"- `status`: `{progress.get('status')}`\n"
        f"- `expected_annotations`: `{progress.get('num_expected_annotations')}`\n"
        f"- `matched_annotations`: `{progress.get('num_matched_annotations')}`\n"
        f"- `missing_annotations`: `{progress.get('num_missing_annotations')}`\n"
        f"- `ready_for_merge`: `{progress.get('ready_for_merge')}`\n"
        f"- `ready_for_finalize`: `{progress.get('ready_for_finalize')}`\n"
        f"- `verification_ok`: `{verification.get('ok')}`\n\n"
        "## Bundle Documentation\n\n"
        f"{documentation_text}\n\n"
        "## Handoff Manifest\n\n"
        f"{handoff_manifest_text}\n\n"
        "## Rejected Returns\n\n"
        f"{rejected_returns_text}\n\n"
        "## Return Inbox\n\n"
        f"{return_inbox_text}\n\n"
        "## Operator Scripts\n\n"
        f"{operator_scripts_text}\n\n"
        "## Sandbox Sample\n\n"
        f"{sandbox_sample_text}\n\n"
        "## Annotation Sheets\n\n"
        f"{annotation_sheet_text}\n\n"
        "## Annotator Packets\n\n"
        f"{annotator_packet_text}\n\n"
        "## Annotator Packet Apply Commands\n\n"
        f"{annotator_packet_apply_text}\n\n"
        "## Annotator Packet Batch Ingest\n\n"
        f"{annotator_packet_batch_ingest_text}\n\n"
        "## Return Inbox Watch Command\n\n"
        f"{return_inbox_watch_text}\n\n"
        "## Watch Stop Actions\n\n"
        f"{watch_stop_actions_text}\n\n"
        "## Spreadsheet Apply Commands\n\n"
        f"{annotation_sheet_apply_text}\n\n"
        "## Adjudication Package\n\n"
        f"{adjudication_package_text}\n\n"
        "## Adjudication Packet\n\n"
        f"{adjudication_packet_text}\n\n"
        "## Adjudication Packet Apply Command\n\n"
        f"{adjudication_packet_apply_text}\n\n"
        "## Return Inbox Sync Command\n\n"
        f"{return_inbox_sync_text}\n\n"
        "## Rejected Return Review Command\n\n"
        f"{rejected_returns_review_text}\n\n"
        "## Audit Checks\n\n"
        f"{check_definition_text}\n\n"
        "## Execution Workflow\n\n"
        "1. Read `docs/annotation_guideline.md` first when that file is present in this bundle.\n"
        "2. Review the finalize-ready sandbox sample under `sandbox/` before distributing the full packet to annotators.\n"
        "3. For external handoff, prefer sending each annotator only their packet archive under `annotators/<id>.zip`.\n"
        "4. Each annotator may either edit their own CSV sheet under `sheets/` or edit the canonical JSONL task file under `tasks/` directly.\n"
        "5. If multiple packet archives are returned together, prefer batch ingesting them back into this bundle before running progress checks.\n"
        "6. Fill `checks` and optional `notes`, but do not modify immutable scenario fields.\n"
        "7. Run the progress command to confirm there are no missing, duplicate, or extra annotations.\n"
        "8. If disagreement items are materialized, hand off the adjudication packet archive under `adjudication/packet.zip` and ingest the returned packet before final reconcile.\n"
        "9. Once the bundle reaches `ready_for_merge` or `ready_for_finalize`, reconcile it back into the canonical source artifacts with "
        f"`{HUMAN_AUDIT_CLI_PREFIX} reconcile-human-audit-evidence-bundle --bundle-dir {bundle_report.get('bundle_dir')} "
        f"--signed-at 2026-05-13T00:00:00Z --annotation-guideline {bundle_guideline_command_path} "
        "--adjudication-policy 'Disagreements are adjudicated after double annotation.'`.\n"
        "10. After reconcile succeeds, verify the completed manifest and agreement metrics.\n\n"
        "## Annotator Workload\n\n"
        f"{annotator_progress}\n\n"
        "## Domain Coverage\n\n"
        f"{domain_progress}\n\n"
        "## Probe Coverage\n\n"
        f"{probe_progress}\n\n"
        "## Suggested Commands\n\n"
        f"{command_text}\n"
    )


def _bundle_progress_lines(progress: Any, label: str) -> str:
    if not isinstance(progress, dict) or not progress:
        return "- No progress summary is currently available."

    lines: list[str] = []
    for key, payload in sorted(progress.items()):
        if not isinstance(payload, dict):
            continue
        expected = payload.get("num_expected_annotations")
        matched = payload.get("num_matched_annotations")
        missing = payload.get("num_missing_annotations")
        fraction = payload.get("completion_fraction")
        fraction_text = f"{fraction:.3f}" if isinstance(fraction, (int, float)) else "n/a"
        line = (
            f"- `{label}={key}`: matched={matched}/{expected}, "
            f"missing={missing}, completion_fraction={fraction_text}"
        )
        task_file = payload.get("task_file")
        if task_file:
            line += f", task_file=`{task_file}`"
        lines.append(line)
    return "\n".join(lines) if lines else "- No progress summary is currently available."


def _bundle_documentation_lines(documentation_files: Any) -> str:
    if not isinstance(documentation_files, dict) or not documentation_files:
        return "- No release documentation files were copied into this bundle."
    return "\n".join(
        f"- `{name}`: `{path}`"
        for name, path in sorted((str(key), str(value)) for key, value in documentation_files.items())
    )


def _bundle_handoff_manifest_lines(handoff_manifest_file: Any) -> str:
    if not isinstance(handoff_manifest_file, str) or not handoff_manifest_file:
        return "- No handoff manifest is currently materialized."
    return f"- `handoff_manifest`: `{handoff_manifest_file}`"


def _bundle_rejected_returns_lines(
    rejected_return_summary: Any,
    rejected_returns_report_file: Any,
) -> str:
    if not isinstance(rejected_return_summary, dict):
        return "- No rejected return summary is currently materialized."
    annotator_count = int(rejected_return_summary.get("num_rejected_annotator_packets", 0) or 0)
    adjudication_count = int(rejected_return_summary.get("num_rejected_adjudication_packets", 0) or 0)
    lines = [
        f"- `num_rejected_annotator_packets`: `{annotator_count}`",
        f"- `num_rejected_adjudication_packets`: `{adjudication_count}`",
    ]
    if isinstance(rejected_returns_report_file, str) and rejected_returns_report_file:
        lines.append(f"- `report_file`: `{rejected_returns_report_file}`")
    for label, entries in (
        ("annotator", rejected_return_summary.get("rejected_annotator_packets")),
        ("adjudication", rejected_return_summary.get("rejected_adjudication_packets")),
    ):
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            packet_path = entry.get("packet_path")
            archive_file = entry.get("rejected_archive_file")
            rejection_error = entry.get("rejection_error")
            lines.append(
                f"- `{label}` rejected packet: packet_path=`{packet_path}` archive=`{archive_file}` error=`{rejection_error}`"
            )
    return "\n".join(lines)


def _bundle_return_inbox_lines(
    return_inbox: Any,
    return_archive: Any,
    return_reject_archive: Any,
    return_inbox_state_file: Any,
) -> str:
    if not isinstance(return_inbox, dict) or not return_inbox:
        return "- No bundle-local return inbox is currently materialized."
    annotator_inbox = return_inbox.get("annotator_inbox")
    adjudication_inbox = return_inbox.get("adjudication_inbox")
    lines = [
        f"- `annotator_inbox`: `{annotator_inbox}`",
        f"- `adjudication_inbox`: `{adjudication_inbox}`",
    ]
    if isinstance(return_archive, dict) and return_archive:
        annotator_archive = return_archive.get("annotator_archive")
        adjudication_archive = return_archive.get("adjudication_archive")
        lines.append(f"- `annotator_archive`: `{annotator_archive}`")
        lines.append(f"- `adjudication_archive`: `{adjudication_archive}`")
    if isinstance(return_reject_archive, dict) and return_reject_archive:
        annotator_archive = return_reject_archive.get("annotator_archive")
        adjudication_archive = return_reject_archive.get("adjudication_archive")
        lines.append(f"- `annotator_reject_archive`: `{annotator_archive}`")
        lines.append(f"- `adjudication_reject_archive`: `{adjudication_archive}`")
    if isinstance(return_inbox_state_file, str) and return_inbox_state_file:
        lines.append(f"- `state_file`: `{return_inbox_state_file}`")
    return "\n".join(lines)


def _bundle_operator_scripts_lines(operator_scripts: Any) -> str:
    if not isinstance(operator_scripts, dict) or not operator_scripts:
        return "- No bundle-local operator scripts are currently materialized."
    return "\n".join(
        f"- `{script_id}`: `{path}`"
        for script_id, path in sorted((str(key), str(value)) for key, value in operator_scripts.items())
    )


def _bundle_rejected_returns_review_lines(bundle_dir: Any) -> str:
    if not isinstance(bundle_dir, str) or not bundle_dir:
        return "- No rejected-return review command is currently materialized."
    return (
        f"- `{HUMAN_AUDIT_CLI_PREFIX} summarize-human-audit-rejected-returns "
        f"--bundle-dir {bundle_dir} --output {bundle_dir}/rejected_returns_report.json`"
    )


def _bundle_sandbox_sample_lines(sandbox_sample: Any) -> str:
    if not isinstance(sandbox_sample, dict) or not sandbox_sample:
        return "- No finalize-ready sandbox sample is currently materialized."
    selected_items = sandbox_sample.get("selected_items")
    selected_items_text = (
        ", ".join(f"`{item}`" for item in selected_items)
        if isinstance(selected_items, list) and selected_items
        else "n/a"
    )
    lines = [
        f"- `task_manifest`: `{sandbox_sample.get('sandbox_task_manifest_file')}`",
        f"- `completed_annotations`: `{sandbox_sample.get('completed_annotations_file')}`",
        f"- `attestation`: `{sandbox_sample.get('attestation_file')}`",
        f"- `agreement`: `{sandbox_sample.get('agreement_file')}`",
        f"- `verification_report`: `{sandbox_sample.get('verification_report_file')}`",
        f"- `selected_items`: {selected_items_text}",
        f"- `num_items`: `{sandbox_sample.get('num_items')}`",
        f"- `num_annotators`: `{sandbox_sample.get('num_annotators')}`",
    ]
    verification = sandbox_sample.get("verification")
    if isinstance(verification, dict):
        lines.append(f"- `verification_ok`: `{verification.get('ok')}`")
    return "\n".join(lines)


def _annotator_packet_sandbox_lines(sandbox_sample: Any) -> str:
    if not isinstance(sandbox_sample, dict) or not sandbox_sample:
        return "- No packet-local sandbox sample is currently available."
    lines = [
        f"- `task_manifest`: `{sandbox_sample.get('sandbox_task_manifest_file')}`",
        f"- `completed_annotations`: `{sandbox_sample.get('completed_annotations_file')}`",
        f"- `verification_report`: `{sandbox_sample.get('verification_report_file')}`",
    ]
    selected_items = sandbox_sample.get("selected_items")
    if isinstance(selected_items, list) and selected_items:
        lines.append("- `selected_items`: " + ", ".join(f"`{item}`" for item in selected_items))
    verification = sandbox_sample.get("verification")
    if isinstance(verification, dict):
        lines.append(f"- `verification_ok`: `{verification.get('ok')}`")
    return "\n".join(lines)


def _bundle_annotation_sheet_lines(annotation_sheet_files: Any) -> str:
    if not isinstance(annotation_sheet_files, dict) or not annotation_sheet_files:
        return "- No spreadsheet annotation sheets are currently available."
    return "\n".join(
        f"- `annotator={annotator}`: `{path}`"
        for annotator, path in sorted((str(key), str(value)) for key, value in annotation_sheet_files.items())
    )


def _bundle_annotator_packet_lines(annotator_packets: Any) -> str:
    if not isinstance(annotator_packets, dict) or not annotator_packets:
        return "- No per-annotator packets are currently available."
    lines: list[str] = []
    for annotator, packet in sorted(annotator_packets.items()):
        if not isinstance(packet, dict):
            continue
        packet_dir = packet.get("packet_dir")
        sheet_file = packet.get("sheet_file")
        readme_file = packet.get("readme_file")
        packet_manifest_file = packet.get("packet_manifest_file")
        sandbox_sample = packet.get("sandbox_sample") if isinstance(packet.get("sandbox_sample"), dict) else {}
        sandbox_task_manifest = sandbox_sample.get("sandbox_task_manifest_file")
        lines.append(
            f"- `annotator={annotator}`: packet_dir=`{packet_dir}`, sheet=`{sheet_file}`, readme=`{readme_file}`, "
            f"manifest=`{packet_manifest_file}`, sandbox=`{sandbox_task_manifest}`, archive=`{packet.get('archive_file')}`"
        )
    return "\n".join(lines) if lines else "- No per-annotator packets are currently available."


def _bundle_annotation_sheet_apply_lines(task_manifest_file: Any, annotation_sheet_files: Any) -> str:
    if not isinstance(task_manifest_file, str) or not isinstance(annotation_sheet_files, dict) or not annotation_sheet_files:
        return "- No spreadsheet apply commands are currently available."
    return "\n".join(
        f"- `{HUMAN_AUDIT_CLI_PREFIX} apply-human-audit-annotation-sheet "
        f"--task-manifest {task_manifest_file} --annotator-id {annotator} --sheet {path}`"
        for annotator, path in sorted((str(key), str(value)) for key, value in annotation_sheet_files.items())
    )


def _bundle_annotator_packet_apply_lines(task_manifest_file: Any, annotator_packets: Any) -> str:
    if not isinstance(task_manifest_file, str) or not isinstance(annotator_packets, dict) or not annotator_packets:
        return "- No annotator-packet apply commands are currently available."
    lines: list[str] = []
    for annotator, packet in sorted(annotator_packets.items()):
        if not isinstance(packet, dict):
            continue
        packet_path = packet.get("archive_file") or packet.get("packet_dir")
        if not isinstance(packet_path, str) or not packet_path:
            continue
        lines.append(
            f"- `{HUMAN_AUDIT_CLI_PREFIX} apply-human-audit-annotator-packet "
            f"--task-manifest {task_manifest_file} --annotator-id {annotator} --packet {packet_path}`"
        )
    return "\n".join(lines) if lines else "- No annotator-packet apply commands are currently available."


def _bundle_annotator_packet_batch_ingest_lines(bundle_dir: Any, return_inbox: Any) -> str:
    if not isinstance(bundle_dir, str) or not bundle_dir:
        return "- No packet-batch ingest command is currently available."
    annotator_inbox = "RETURNS"
    if isinstance(return_inbox, dict):
        annotator_inbox = str(return_inbox.get("annotator_inbox") or annotator_inbox)
    return (
        f"- `{HUMAN_AUDIT_CLI_PREFIX} ingest-human-audit-return-packets "
        f"--bundle-dir {bundle_dir} --packets {annotator_inbox}/*.zip`"
    )


def _bundle_return_inbox_watch_lines(bundle_dir: Any) -> str:
    if not isinstance(bundle_dir, str) or not bundle_dir:
        return "- No return-inbox watch command is currently available."
    return (
        f"- `{HUMAN_AUDIT_CLI_PREFIX} watch-human-audit-return-inbox "
        f"--bundle-dir {bundle_dir} --interval-s {HUMAN_AUDIT_OPERATOR_WATCH_INTERVAL_S} --max-iterations 0 "
        "--stop-when-ready "
        "--stop-when-rejected "
        f"--output {Path(bundle_dir) / 'return_inbox_watch.json'}`\n"
        "- `exit_codes`: `0=max_iterations`, `2=rejected_returns`, `3=ready`"
    )


def _bundle_watch_stop_actions_lines(watch_stop_actions: Any) -> str:
    if not isinstance(watch_stop_actions, dict) or not watch_stop_actions:
        return "- No watch stop-action contract is currently available."
    lines: list[str] = []
    for stop_reason, action in sorted(watch_stop_actions.items()):
        if not isinstance(action, dict):
            continue
        lines.append(
            f"- `stop_reason={stop_reason}`: "
            f"`exit_code={action.get('exit_code')}` "
            f"`kind={action.get('kind')}` "
            f"`next_script={action.get('next_script')}`"
        )
        next_command = action.get("next_command")
        if isinstance(next_command, str) and next_command:
            lines.append(f"  command: `{next_command}`")
    return "\n".join(lines) if lines else "- No watch stop-action contract is currently available."


def _bundle_adjudication_package_lines(adjudication_package: Any) -> str:
    if not isinstance(adjudication_package, dict) or not adjudication_package:
        return "- No adjudication package is currently materialized."
    tasks = adjudication_package.get("adjudication_tasks_file")
    manifest = adjudication_package.get("adjudication_manifest_file")
    summary = adjudication_package.get("adjudication_summary_file")
    status = adjudication_package.get("status")
    return (
        f"- `status`: `{status}`\n"
        f"- `tasks`: `{tasks}`\n"
        f"- `manifest`: `{manifest}`\n"
        f"- `summary`: `{summary}`"
    )


def _bundle_check_definition_lines(checks: Any, check_definitions: Any) -> str:
    checks_list = [str(item) for item in checks] if isinstance(checks, list) else list(AUDIT_CHECK_DEFINITIONS)
    definitions = (
        {str(key): str(value) for key, value in check_definitions.items()}
        if isinstance(check_definitions, dict)
        else dict(AUDIT_CHECK_DEFINITIONS)
    )
    lines: list[str] = []
    for check_id in checks_list:
        description = definitions.get(check_id)
        if description:
            lines.append(f"- `{check_id}`: {description}")
        else:
            lines.append(f"- `{check_id}`")
    return "\n".join(lines) if lines else "- No audit check definitions are currently available."


def _load_adjudication_task_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid adjudication JSONL record") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: adjudication record must be a JSON object")
            records.append(record)
    return records


def _record_adjudication_target_fields(record: dict[str, Any]) -> tuple[str, ...]:
    raw_checks = record.get("adjudication_checks")
    if not isinstance(raw_checks, dict):
        return ()
    return tuple(field for field in AUDIT_CHECK_FIELDS if field in raw_checks)


def _adjudication_sheet_row(record: dict[str, Any]) -> dict[str, str]:
    target_fields = set(_record_adjudication_target_fields(record))
    adjudication_checks = record.get("adjudication_checks") if isinstance(record.get("adjudication_checks"), dict) else {}
    applicable_checks = record.get("applicable_checks")
    row = {
        "case_id": str(record.get("case_id", "")),
        "query_id": str(record.get("query_id", "")),
        "adjudicator_id": str(record.get("adjudicator_id", "") or ""),
        "adjudication_status": str(record.get("adjudication_status", "pending")),
        "domain": str(record.get("domain", "")),
        "probe_type": str(record.get("probe_type", "")),
        "task_type": str(record.get("task_type", "")),
        "difficulty_level": str(record.get("difficulty_level", "")),
        "memory_requirement": str(record.get("memory_requirement", "")),
        "memory_dependency": str(record.get("memory_dependency", "")),
        "counterfactual_group_id": str(record.get("counterfactual_group_id", "") or ""),
        "counterfactual_axis": str(record.get("counterfactual_axis", "") or ""),
        "counterfactual_edit": str(record.get("counterfactual_edit", "") or ""),
        "scoring_rule": str(record.get("scoring_rule", "")),
        "applicable_checks": ";".join(str(value) for value in applicable_checks) if isinstance(applicable_checks, list) else "",
        "prompt": str(record.get("prompt", "")),
        "consensus_checks_json": json.dumps(record.get("consensus_checks", {}), ensure_ascii=False, sort_keys=True),
        "disagreement_fields_json": json.dumps(record.get("disagreement_fields", {}), ensure_ascii=False, sort_keys=True),
        "source_template": str(record.get("source_template", "") or ""),
        "source_template_line": str(record.get("source_template_line", "") or ""),
        "notes": str(record.get("notes", "") or ""),
    }
    for field in AUDIT_CHECK_FIELDS:
        if field not in target_fields:
            row[field] = "N/A"
            continue
        value = adjudication_checks.get(field)
        if value is True:
            row[field] = "TRUE"
        elif value is False:
            row[field] = "FALSE"
        else:
            row[field] = ""
    return row


def _bundle_adjudication_packet_lines(adjudication_packet: Any) -> str:
    if not isinstance(adjudication_packet, dict) or not adjudication_packet:
        return "- No adjudication packet is currently available."
    return (
        f"- `packet_dir`: `{adjudication_packet.get('packet_dir')}`\n"
        f"- `sheet`: `{adjudication_packet.get('sheet_file')}`\n"
        f"- `task`: `{adjudication_packet.get('task_file')}`\n"
        f"- `readme`: `{adjudication_packet.get('readme_file')}`\n"
        f"- `manifest`: `{adjudication_packet.get('packet_manifest_file')}`\n"
        f"- `archive`: `{adjudication_packet.get('archive_file')}`"
    )


def _bundle_adjudication_packet_apply_lines(bundle_dir: Any) -> str:
    if not isinstance(bundle_dir, str) or not bundle_dir:
        return "- No adjudication packet apply command is currently available."
    guideline = str(Path(bundle_dir) / HUMAN_AUDIT_BUNDLE_SANDBOX_ANNOTATION_GUIDELINE)
    return (
        f"- `{HUMAN_AUDIT_CLI_PREFIX} apply-human-audit-adjudication-packet "
        f"--bundle-dir {bundle_dir} --packet RETURNED_ADJUDICATION_PACKET.zip "
        "--reconcile-when-ready --signed-at 2026-05-13T00:00:00Z "
        f"--annotation-guideline {guideline} "
        "--adjudication-policy 'Disagreements are adjudicated after double annotation.'`"
    )


def _bundle_return_inbox_sync_lines(bundle_dir: Any, return_inbox: Any) -> str:
    if not isinstance(bundle_dir, str) or not bundle_dir:
        return "- No return-inbox sync command is currently available."
    guideline = str(Path(bundle_dir) / HUMAN_AUDIT_BUNDLE_SANDBOX_ANNOTATION_GUIDELINE)
    annotator_inbox = "RETURNS/annotators"
    adjudication_inbox = "RETURNS/adjudication"
    if isinstance(return_inbox, dict):
        annotator_inbox = str(return_inbox.get("annotator_inbox") or annotator_inbox)
        adjudication_inbox = str(return_inbox.get("adjudication_inbox") or adjudication_inbox)
    return (
        f"- `{HUMAN_AUDIT_CLI_PREFIX} sync-human-audit-return-inbox "
        f"--bundle-dir {bundle_dir} "
        f"--annotator-inbox {annotator_inbox} --adjudication-inbox {adjudication_inbox} "
        "--reconcile-when-ready --signed-at 2026-05-13T00:00:00Z "
        f"--annotation-guideline {guideline} "
        "--adjudication-policy 'Disagreements are adjudicated after double annotation.'`"
    )


def _load_optional_adjudication_packet_manifest(packet_root: Path) -> dict[str, Any] | None:
    manifest_path = packet_root / "packet_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"adjudication packet manifest must be a JSON object: {manifest_path}")
    return manifest


def _validate_adjudication_packet_manifest(
    packet_manifest: dict[str, Any] | None,
    packet_root: Path,
    sheet_path: Path,
    task_path: Path,
    task_file_sha256: str,
) -> None:
    if packet_manifest is None:
        return
    if packet_manifest.get("schema_version") != HUMAN_AUDIT_ADJUDICATION_PACKET_SCHEMA_VERSION:
        raise ValueError("adjudication packet manifest schema_version is invalid")
    if packet_manifest.get("packet_type") != "adjudication":
        raise ValueError("adjudication packet manifest packet_type must be 'adjudication'")
    for field, actual_path in (
        ("sheet_file", sheet_path),
        ("task_file", task_path),
    ):
        raw_path = packet_manifest.get(field)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"adjudication packet manifest is missing {field}")
        resolved_path = Path(raw_path) if Path(raw_path).is_absolute() else packet_root / raw_path
        if resolved_path.resolve() != actual_path.resolve():
            raise ValueError(f"adjudication packet manifest {field} does not match packet contents")
    if packet_manifest.get("task_file_sha256") != task_file_sha256:
        raise ValueError("adjudication packet manifest task_file_sha256 does not match packet task file")


def _resolve_adjudication_packet_root(path: Path) -> Path:
    if _looks_like_adjudication_packet_root(path):
        return path
    child_dirs = [child for child in sorted(path.iterdir()) if child.is_dir()]
    packet_roots = [child for child in child_dirs if _looks_like_adjudication_packet_root(child)]
    if len(packet_roots) == 1:
        return packet_roots[0]
    raise ValueError(f"could not locate adjudication packet root under {path}")


def _looks_like_adjudication_packet_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(path.glob("*.csv")) and any(path.glob("*.jsonl"))


def _single_adjudication_packet_file(packet_root: Path, pattern: str, label: str) -> Path:
    matches = sorted(packet_root.glob(pattern))
    if not matches:
        raise ValueError(f"adjudication packet is missing {label}: {packet_root}")
    if len(matches) > 1:
        raise ValueError(f"adjudication packet has multiple {label} candidates: {[path.name for path in matches]}")
    return matches[0]


def _discover_return_packet_candidates(inboxes: Iterable[str | Path]) -> list[Path]:
    candidates: list[Path] = []
    for raw_inbox in inboxes:
        inbox = Path(raw_inbox)
        if not inbox.exists():
            raise ValueError(f"return inbox does not exist: {inbox}")
        if inbox.is_file():
            candidates.append(inbox)
            continue
        for child in sorted(inbox.iterdir()):
            if child.name.startswith("."):
                continue
            if child.is_dir() or child.suffix.lower() == ".zip":
                candidates.append(child)
    return candidates


def _packet_content_fingerprint(packet_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in packet_root.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(packet_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _raw_return_packet_fingerprint(packet_path: Path) -> str:
    if packet_path.is_file():
        digest = _file_sha256(packet_path)
        if digest is None:
            raise ValueError(f"cannot fingerprint return packet file: {packet_path}")
        return digest
    if packet_path.is_dir():
        return _packet_content_fingerprint(packet_path)
    raise ValueError(f"return packet path must be a file or directory: {packet_path}")


def _processed_return_inbox_state_entry(
    packet_path: Path,
    *,
    packet_fingerprint: str,
    processed_archive_file: str | None = None,
) -> dict[str, Any]:
    entry = {
        "packet_path": str(packet_path),
        "packet_fingerprint": packet_fingerprint,
        "source_kind": "directory" if packet_path.is_dir() else "archive",
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if processed_archive_file:
        entry["processed_archive_file"] = processed_archive_file
    return entry


def _copy_processed_return_packet(
    bundle_root: Path,
    packet_path: Path,
    *,
    archive_dir: Path,
    packet_label: str,
    packet_fingerprint: str,
) -> str:
    safe_label = _safe_filename(packet_label) or "packet"
    fingerprint_prefix = packet_fingerprint[:12]
    if packet_path.is_file():
        suffix = packet_path.suffix if packet_path.suffix else ".bin"
        target = archive_dir / f"{safe_label}_{fingerprint_prefix}{suffix}"
        if not target.exists():
            shutil.copy2(packet_path, target)
    else:
        target = archive_dir / f"{safe_label}_{fingerprint_prefix}"
        if not target.exists():
            shutil.copytree(packet_path, target)
    return _relative_or_absolute(target, bundle_root)


def _processed_return_archive_entry(
    bundle_root: Path,
    return_archive: dict[str, str],
    packet_path: Path,
    *,
    packet_label: str,
    packet_fingerprint: str,
    archive_field: str,
) -> str | None:
    raw_archive_dir = return_archive.get(archive_field)
    if not isinstance(raw_archive_dir, str) or not raw_archive_dir:
        return None
    archive_dir = _resolve_path(raw_archive_dir, bundle_root)
    archive_dir.mkdir(parents=True, exist_ok=True)
    return _copy_processed_return_packet(
        bundle_root,
        packet_path,
        archive_dir=archive_dir,
        packet_label=packet_label,
        packet_fingerprint=packet_fingerprint,
    )


def _processed_return_archive_from_bundle(bundle_root: Path) -> dict[str, str]:
    bundle_manifest_path = bundle_root / "bundle_manifest.json"
    if bundle_manifest_path.exists():
        bundle_manifest = read_json(bundle_manifest_path)
        if isinstance(bundle_manifest, dict) and isinstance(bundle_manifest.get("return_archive"), dict):
            return {
                str(key): str(value)
                for key, value in sorted(bundle_manifest["return_archive"].items())
                if isinstance(value, str) and value
            }
    return _ensure_bundle_return_archive(bundle_root)


def _rejected_return_archive_from_bundle(bundle_root: Path) -> dict[str, str]:
    bundle_manifest_path = bundle_root / "bundle_manifest.json"
    if bundle_manifest_path.exists():
        bundle_manifest = read_json(bundle_manifest_path)
        if isinstance(bundle_manifest, dict) and isinstance(bundle_manifest.get("return_reject_archive"), dict):
            return {
                str(key): str(value)
                for key, value in sorted(bundle_manifest["return_reject_archive"].items())
                if isinstance(value, str) and value
            }
    return _ensure_bundle_return_reject_archive(bundle_root)


def _return_archive_summary(processed_state: dict[str, Any]) -> dict[str, Any]:
    annotator_packets = processed_state.get("annotator_packets")
    annotator_archives = {}
    if isinstance(annotator_packets, dict):
        for annotator_id, entry in sorted(annotator_packets.items()):
            if isinstance(entry, dict):
                archive_file = entry.get("processed_archive_file")
                if isinstance(archive_file, str) and archive_file:
                    annotator_archives[str(annotator_id)] = archive_file
    adjudication_archive = None
    adjudication_entry = processed_state.get("adjudication_packet")
    if isinstance(adjudication_entry, dict):
        archive_file = adjudication_entry.get("processed_archive_file")
        if isinstance(archive_file, str) and archive_file:
            adjudication_archive = archive_file
    return {
        "annotator_archives": annotator_archives,
        "adjudication_archive": adjudication_archive,
    }


def _return_reject_archive_summary(processed_state: dict[str, Any]) -> dict[str, Any]:
    annotator_archives: list[dict[str, Any]] = []
    raw_rejected_annotator_packets = processed_state.get("rejected_annotator_packets")
    if isinstance(raw_rejected_annotator_packets, list):
        for entry in raw_rejected_annotator_packets:
            if isinstance(entry, dict):
                archive_file = entry.get("rejected_archive_file")
                if isinstance(archive_file, str) and archive_file:
                    annotator_archives.append(
                        {
                            "packet_path": entry.get("packet_path"),
                            "packet_fingerprint": entry.get("packet_fingerprint"),
                            "rejection_error": entry.get("rejection_error"),
                            "rejected_archive_file": archive_file,
                        }
                    )
    adjudication_archives: list[dict[str, Any]] = []
    raw_rejected_adjudication_packets = processed_state.get("rejected_adjudication_packets")
    if isinstance(raw_rejected_adjudication_packets, list):
        for entry in raw_rejected_adjudication_packets:
            if isinstance(entry, dict):
                archive_file = entry.get("rejected_archive_file")
                if isinstance(archive_file, str) and archive_file:
                    adjudication_archives.append(
                        {
                            "packet_path": entry.get("packet_path"),
                            "packet_fingerprint": entry.get("packet_fingerprint"),
                            "rejection_error": entry.get("rejection_error"),
                            "rejected_archive_file": archive_file,
                        }
                    )
    return {
        "annotator_archives": annotator_archives,
        "adjudication_archives": adjudication_archives,
    }


def _rejected_return_summary(processed_state: dict[str, Any]) -> dict[str, Any]:
    rejected_annotator_packets: list[dict[str, Any]] = []
    raw_rejected_annotator_packets = processed_state.get("rejected_annotator_packets")
    if isinstance(raw_rejected_annotator_packets, list):
        for entry in raw_rejected_annotator_packets:
            if not isinstance(entry, dict):
                continue
            summary_entry = {
                "packet_path": entry.get("packet_path"),
                "packet_fingerprint": entry.get("packet_fingerprint"),
                "rejection_error": entry.get("rejection_error"),
                "rejected_archive_file": entry.get("rejected_archive_file"),
            }
            rejected_annotator_packets.append(summary_entry)
    rejected_adjudication_packets: list[dict[str, Any]] = []
    raw_rejected_adjudication_packets = processed_state.get("rejected_adjudication_packets")
    if isinstance(raw_rejected_adjudication_packets, list):
        for entry in raw_rejected_adjudication_packets:
            if not isinstance(entry, dict):
                continue
            summary_entry = {
                "packet_path": entry.get("packet_path"),
                "packet_fingerprint": entry.get("packet_fingerprint"),
                "rejection_error": entry.get("rejection_error"),
                "rejected_archive_file": entry.get("rejected_archive_file"),
            }
            rejected_adjudication_packets.append(summary_entry)
    return {
        "num_rejected_annotator_packets": len(rejected_annotator_packets),
        "num_rejected_adjudication_packets": len(rejected_adjudication_packets),
        "rejected_annotator_packets": rejected_annotator_packets,
        "rejected_adjudication_packets": rejected_adjudication_packets,
    }


def summarize_human_audit_rejected_returns(bundle_dir: str | Path) -> dict[str, Any]:
    """Summarize currently quarantined invalid returned packets for operator triage."""

    bundle_root = Path(bundle_dir)
    bundle_manifest_path = bundle_root / "bundle_manifest.json"
    if bundle_manifest_path.exists():
        bundle_manifest = read_json(bundle_manifest_path)
        if not isinstance(bundle_manifest, dict):
            raise ValueError("bundle_manifest.json must be a JSON object")
    processed_state = _load_human_audit_return_inbox_state(bundle_root)
    rejected_summary = _rejected_return_summary(processed_state)
    return_reject_archive = _rejected_return_archive_from_bundle(bundle_root)
    report = {
        "schema_version": "amst-human-audit-rejected-returns-v1",
        "root": _bundle_root_ref(bundle_root),
        "bundle_dir": str(bundle_root),
        "bundle_manifest_file": str(bundle_manifest_path),
        "return_inbox_state_file": str(_bundle_return_inbox_state_path(bundle_root)),
        "return_reject_archive": return_reject_archive,
        "rejected_return_summary": rejected_summary,
        "has_rejected_returns": bool(
            rejected_summary["num_rejected_annotator_packets"] or rejected_summary["num_rejected_adjudication_packets"]
        ),
        "recommended_actions": [
            "Review the rejected packet archive copy and the original returned packet path for each entry.",
            "Replace or remove invalid returned packets from the inbox before rerunning inbox sync/watch.",
            "Rerun sync/watch after triage to confirm the rejected summary returns to zero.",
        ],
    }
    return report


def _record_processed_annotator_packets(
    bundle_root: Path,
    processed_state: dict[str, Any],
    packets: Iterable[dict[str, Any]],
) -> None:
    annotator_packets = processed_state.setdefault("annotator_packets", {})
    if not isinstance(annotator_packets, dict):
        annotator_packets = {}
        processed_state["annotator_packets"] = annotator_packets
    return_archive = _processed_return_archive_from_bundle(bundle_root)
    for entry in packets:
        annotator_id = str(entry["annotator_id"])
        packet_path = Path(entry["packet_path"])
        packet_fingerprint = str(entry["packet_fingerprint"])
        processed_archive_file = _processed_return_archive_entry(
            bundle_root,
            return_archive,
            packet_path,
            packet_label=annotator_id,
            packet_fingerprint=packet_fingerprint,
            archive_field="annotator_archive",
        )
        annotator_packets[annotator_id] = _processed_return_inbox_state_entry(
            packet_path,
            packet_fingerprint=packet_fingerprint,
            processed_archive_file=processed_archive_file,
        )


def _record_processed_adjudication_packet(
    bundle_root: Path,
    processed_state: dict[str, Any],
    packet: dict[str, Any],
) -> None:
    packet_path = Path(packet["packet_path"])
    packet_fingerprint = str(packet["packet_fingerprint"])
    return_archive = _processed_return_archive_from_bundle(bundle_root)
    processed_archive_file = _processed_return_archive_entry(
        bundle_root,
        return_archive,
        packet_path,
        packet_label="adjudication",
        packet_fingerprint=packet_fingerprint,
        archive_field="adjudication_archive",
    )
    processed_state["adjudication_packet"] = _processed_return_inbox_state_entry(
        packet_path,
        packet_fingerprint=packet_fingerprint,
        processed_archive_file=processed_archive_file,
    )


def _rejected_return_inbox_state_entry(
    packet_path: Path,
    *,
    packet_fingerprint: str,
    rejection_error: str,
    rejected_archive_file: str | None = None,
) -> dict[str, Any]:
    entry = _processed_return_inbox_state_entry(
        packet_path,
        packet_fingerprint=packet_fingerprint,
    )
    entry["rejection_error"] = rejection_error
    if rejected_archive_file:
        entry["rejected_archive_file"] = rejected_archive_file
    return entry


def _record_rejected_annotator_packets(
    bundle_root: Path,
    processed_state: dict[str, Any],
    packets: Iterable[dict[str, Any]],
) -> None:
    raw_entries = processed_state.get("rejected_annotator_packets")
    fingerprint_index: dict[str, dict[str, Any]] = {}
    if isinstance(raw_entries, list):
        for entry in raw_entries:
            normalized = _normalize_return_inbox_rejection_entry(entry)
            if normalized is not None:
                fingerprint_index[str(normalized["packet_fingerprint"])] = normalized
    return_reject_archive = _rejected_return_archive_from_bundle(bundle_root)
    for packet in packets:
        packet_path = Path(packet["packet_path"])
        packet_fingerprint = str(packet["packet_fingerprint"])
        rejected_archive_file = _processed_return_archive_entry(
            bundle_root,
            return_reject_archive,
            packet_path,
            packet_label="annotator_reject",
            packet_fingerprint=packet_fingerprint,
            archive_field="annotator_archive",
        )
        fingerprint_index[packet_fingerprint] = _rejected_return_inbox_state_entry(
            packet_path,
            packet_fingerprint=packet_fingerprint,
            rejection_error=str(packet["error"]),
            rejected_archive_file=rejected_archive_file,
        )
    processed_state["rejected_annotator_packets"] = [
        fingerprint_index[fingerprint]
        for fingerprint in sorted(fingerprint_index)
    ]


def _record_rejected_adjudication_packets(
    bundle_root: Path,
    processed_state: dict[str, Any],
    packets: Iterable[dict[str, Any]],
) -> None:
    raw_entries = processed_state.get("rejected_adjudication_packets")
    fingerprint_index: dict[str, dict[str, Any]] = {}
    if isinstance(raw_entries, list):
        for entry in raw_entries:
            normalized = _normalize_return_inbox_rejection_entry(entry)
            if normalized is not None:
                fingerprint_index[str(normalized["packet_fingerprint"])] = normalized
    return_reject_archive = _rejected_return_archive_from_bundle(bundle_root)
    for packet in packets:
        packet_path = Path(packet["packet_path"])
        packet_fingerprint = str(packet["packet_fingerprint"])
        rejected_archive_file = _processed_return_archive_entry(
            bundle_root,
            return_reject_archive,
            packet_path,
            packet_label="adjudication_reject",
            packet_fingerprint=packet_fingerprint,
            archive_field="adjudication_archive",
        )
        fingerprint_index[packet_fingerprint] = _rejected_return_inbox_state_entry(
            packet_path,
            packet_fingerprint=packet_fingerprint,
            rejection_error=str(packet["error"]),
            rejected_archive_file=rejected_archive_file,
        )
    processed_state["rejected_adjudication_packets"] = [
        fingerprint_index[fingerprint]
        for fingerprint in sorted(fingerprint_index)
    ]


def _inspect_annotator_return_packet(packet_path: Path) -> dict[str, Any]:
    with TemporaryDirectory() as temp_dir:
        if packet_path.is_dir():
            packet_root = _resolve_annotator_packet_root(packet_path)
        elif packet_path.is_file() and packet_path.suffix.lower() == ".zip":
            extract_root = Path(temp_dir)
            try:
                with zipfile.ZipFile(packet_path) as archive:
                    archive.extractall(extract_root)
            except zipfile.BadZipFile as exc:
                raise ValueError(f"annotator packet archive is invalid: {packet_path}") from exc
            packet_root = _resolve_annotator_packet_root(extract_root)
        else:
            raise ValueError("annotator return packet must be a directory or .zip archive")
        sheet_file = _single_annotator_packet_file(packet_root, "*.csv", "annotation sheet")
        task_file = _single_annotator_packet_file(packet_root, "*.jsonl", "task file")
        annotator_id = _resolve_annotator_packet_annotator_id(sheet_file, task_file, None)
        packet_fingerprint = _packet_content_fingerprint(packet_root)
    return {
        "annotator_id": annotator_id,
        "packet_path": packet_path,
        "packet_fingerprint": packet_fingerprint,
        "sort_key": (packet_path.stat().st_mtime, str(packet_path)),
    }


def _rejected_return_fingerprints(processed_state: dict[str, Any] | None, field: str) -> dict[str, dict[str, Any]]:
    if not isinstance(processed_state, dict):
        return {}
    raw_entries = processed_state.get(field)
    if not isinstance(raw_entries, list):
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for entry in raw_entries:
        normalized = _normalize_return_inbox_rejection_entry(entry)
        if normalized is not None:
            entries[str(normalized["packet_fingerprint"])] = normalized
    return entries


def _discover_annotator_return_packets(
    inboxes: Iterable[str | Path],
    *,
    processed_state: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    skipped_invalid: list[dict[str, Any]] = []
    processed_annotators = (
        processed_state.get("annotator_packets")
        if isinstance(processed_state, dict) and isinstance(processed_state.get("annotator_packets"), dict)
        else {}
    )
    rejected_packets = _rejected_return_fingerprints(processed_state, "rejected_annotator_packets")
    for candidate_path in _discover_return_packet_candidates(inboxes):
        raw_packet_fingerprint = _raw_return_packet_fingerprint(candidate_path)
        rejected_entry = rejected_packets.get(raw_packet_fingerprint)
        if rejected_entry is not None:
            skipped_invalid.append(
                {
                    "packet_path": candidate_path,
                    "packet_fingerprint": raw_packet_fingerprint,
                    "error": str(rejected_entry.get("rejection_error") or "previously rejected"),
                    "sort_key": (candidate_path.stat().st_mtime, str(candidate_path)),
                }
            )
            continue
        try:
            inspected = _inspect_annotator_return_packet(candidate_path)
        except Exception as exc:  # noqa: BLE001 - invalid returned packets should be surfaced, not crash the inbox loop
            invalid.append(
                {
                    "packet_path": candidate_path,
                    "packet_fingerprint": raw_packet_fingerprint,
                    "error": str(exc),
                    "sort_key": (candidate_path.stat().st_mtime, str(candidate_path)),
                }
            )
            continue
        annotator_id = str(inspected["annotator_id"])
        processed_entry = processed_annotators.get(annotator_id) if isinstance(processed_annotators, dict) else None
        if (
            isinstance(processed_entry, dict)
            and processed_entry.get("packet_fingerprint") == inspected["packet_fingerprint"]
        ):
            skipped.append(inspected)
            continue
        prior = selected.get(annotator_id)
        if prior is None or inspected["sort_key"] > prior["sort_key"]:
            selected[annotator_id] = inspected
    return [selected[key] for key in sorted(selected)], sorted(
        skipped,
        key=lambda item: (str(item["annotator_id"]), item["sort_key"]),
    ), sorted(
        invalid,
        key=lambda item: item["sort_key"],
    ), sorted(
        skipped_invalid,
        key=lambda item: item["sort_key"],
    )


def _inspect_adjudication_return_packet(packet_path: Path) -> dict[str, Any]:
    with TemporaryDirectory() as temp_dir:
        if packet_path.is_dir():
            packet_root = _resolve_adjudication_packet_root(packet_path)
        elif packet_path.is_file() and packet_path.suffix.lower() == ".zip":
            extract_root = Path(temp_dir)
            try:
                with zipfile.ZipFile(packet_path) as archive:
                    archive.extractall(extract_root)
            except zipfile.BadZipFile as exc:
                raise ValueError(f"adjudication packet archive is invalid: {packet_path}") from exc
            packet_root = _resolve_adjudication_packet_root(extract_root)
        else:
            raise ValueError("adjudication return packet must be a directory or .zip archive")
        _single_adjudication_packet_file(packet_root, "*.csv", "adjudication sheet")
        _single_adjudication_packet_file(packet_root, "*.jsonl", "adjudication task file")
        packet_fingerprint = _packet_content_fingerprint(packet_root)
    return {
        "packet_path": packet_path,
        "packet_fingerprint": packet_fingerprint,
        "sort_key": (packet_path.stat().st_mtime, str(packet_path)),
    }


def _discover_adjudication_return_packet(
    inboxes: Iterable[str | Path],
    *,
    processed_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected: dict[str, Any] | None = None
    skipped: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    skipped_invalid: list[dict[str, Any]] = []
    processed_entry = (
        processed_state.get("adjudication_packet")
        if isinstance(processed_state, dict) and isinstance(processed_state.get("adjudication_packet"), dict)
        else None
    )
    rejected_packets = _rejected_return_fingerprints(processed_state, "rejected_adjudication_packets")
    for candidate_path in _discover_return_packet_candidates(inboxes):
        raw_packet_fingerprint = _raw_return_packet_fingerprint(candidate_path)
        rejected_entry = rejected_packets.get(raw_packet_fingerprint)
        if rejected_entry is not None:
            skipped_invalid.append(
                {
                    "packet_path": candidate_path,
                    "packet_fingerprint": raw_packet_fingerprint,
                    "error": str(rejected_entry.get("rejection_error") or "previously rejected"),
                    "sort_key": (candidate_path.stat().st_mtime, str(candidate_path)),
                }
            )
            continue
        try:
            inspected = _inspect_adjudication_return_packet(candidate_path)
        except Exception as exc:  # noqa: BLE001 - invalid returned packets should be surfaced, not crash the inbox loop
            invalid.append(
                {
                    "packet_path": candidate_path,
                    "packet_fingerprint": raw_packet_fingerprint,
                    "error": str(exc),
                    "sort_key": (candidate_path.stat().st_mtime, str(candidate_path)),
                }
            )
            continue
        if (
            isinstance(processed_entry, dict)
            and processed_entry.get("packet_fingerprint") == inspected["packet_fingerprint"]
        ):
            skipped.append(inspected)
            continue
        if selected is None or inspected["sort_key"] > selected["sort_key"]:
            selected = inspected
    return selected, sorted(skipped, key=lambda item: item["sort_key"]), sorted(
        invalid,
        key=lambda item: item["sort_key"],
    ), sorted(
        skipped_invalid,
        key=lambda item: item["sort_key"],
    )


def _copy_bundle_documentation(manifest_file: Path | None, output_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    if manifest_file is None:
        return {}, {}
    source_docs_dir = manifest_file.parent / "docs"
    if not source_docs_dir.exists():
        return {}, {}

    selected_names = (
        "README.md",
        "annotation_guideline.md",
        "governance_privacy_statement.md",
    )
    target_docs_dir = output_dir / "docs"
    copied_files: dict[str, str] = {}
    copied_digests: dict[str, str] = {}
    for name in selected_names:
        source_path = source_docs_dir / name
        if not source_path.exists():
            continue
        target_docs_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_docs_dir / name
        shutil.copy2(source_path, target_path)
        copied_files[name] = _relative_or_absolute(target_path, output_dir)
        copied_digests[name] = _file_sha256(target_path)
    return copied_files, copied_digests


def _bundle_documentation_digests(bundle_dir: Path, bundle_manifest: dict[str, Any]) -> dict[str, str]:
    documentation_files = bundle_manifest.get("documentation_files")
    if not isinstance(documentation_files, dict):
        return {}
    digests: dict[str, str] = {}
    for name, raw_path in sorted(documentation_files.items()):
        path = _resolve_path(str(raw_path), bundle_dir)
        if not path.exists():
            digests[str(name)] = "<missing>"
            continue
        digests[str(name)] = _file_sha256(path)
    return digests


def _bundle_missing_annotation_sheets(bundle_dir: Path, bundle_manifest: dict[str, Any]) -> list[str]:
    annotation_sheet_files = bundle_manifest.get("annotation_sheet_files")
    if not isinstance(annotation_sheet_files, dict):
        return []
    missing: list[str] = []
    for annotator, raw_path in sorted(annotation_sheet_files.items()):
        path = _resolve_path(str(raw_path), bundle_dir)
        if not path.exists():
            missing.append(f"{annotator}:{raw_path}")
    return missing


def _bundle_missing_return_inbox_dirs(bundle_dir: Path, bundle_manifest: dict[str, Any]) -> list[str]:
    return_inbox = bundle_manifest.get("return_inbox")
    if not isinstance(return_inbox, dict):
        return []
    missing: list[str] = []
    for field in ("annotator_inbox", "adjudication_inbox"):
        raw_path = return_inbox.get(field)
        if not isinstance(raw_path, str) or not raw_path:
            missing.append(f"{field}:<missing>")
            continue
        path = _resolve_path(raw_path, bundle_dir)
        if not path.exists() or not path.is_dir():
            missing.append(f"{field}:{raw_path}")
    return missing


def _bundle_missing_return_inbox_state_files(bundle_dir: Path, bundle_manifest: dict[str, Any]) -> list[str]:
    raw_path = bundle_manifest.get("return_inbox_state_file")
    if not isinstance(raw_path, str) or not raw_path:
        return ["return_inbox_state_file:<missing>"]
    path = _resolve_path(raw_path, bundle_dir)
    if not path.exists():
        return [f"return_inbox_state_file:{raw_path}"]
    if not path.is_file():
        return [f"return_inbox_state_file:{raw_path}:not_file"]
    return []


def _bundle_missing_return_inbox_watch_files(bundle_dir: Path, bundle_manifest: dict[str, Any]) -> list[str]:
    raw_path = bundle_manifest.get("return_inbox_watch_file")
    if not isinstance(raw_path, str) or not raw_path:
        return ["return_inbox_watch_file:<missing>"]
    path = _resolve_path(raw_path, bundle_dir)
    if not path.exists():
        return [f"return_inbox_watch_file:{raw_path}"]
    if not path.is_file():
        return [f"return_inbox_watch_file:{raw_path}:not_file"]
    return []


def _bundle_missing_return_inbox_sync_report_files(bundle_dir: Path, bundle_manifest: dict[str, Any]) -> list[str]:
    raw_path = bundle_manifest.get("return_inbox_sync_report_file")
    if not isinstance(raw_path, str) or not raw_path:
        return ["return_inbox_sync_report_file:<missing>"]
    path = _resolve_path(raw_path, bundle_dir)
    if not path.exists():
        return [f"return_inbox_sync_report_file:{raw_path}"]
    if not path.is_file():
        return [f"return_inbox_sync_report_file:{raw_path}:not_file"]
    return []


def _bundle_missing_rejected_returns_report_files(bundle_dir: Path, bundle_manifest: dict[str, Any]) -> list[str]:
    raw_path = bundle_manifest.get("rejected_returns_report_file")
    if not isinstance(raw_path, str) or not raw_path:
        return ["rejected_returns_report_file:<missing>"]
    path = _resolve_path(raw_path, bundle_dir)
    if not path.exists():
        return [f"rejected_returns_report_file:{raw_path}"]
    if not path.is_file():
        return [f"rejected_returns_report_file:{raw_path}:not_file"]
    return []


def _bundle_missing_return_archive_dirs(bundle_dir: Path, bundle_manifest: dict[str, Any]) -> list[str]:
    return_archive = bundle_manifest.get("return_archive")
    if not isinstance(return_archive, dict):
        return []
    missing: list[str] = []
    for field in ("annotator_archive", "adjudication_archive"):
        raw_path = return_archive.get(field)
        if not isinstance(raw_path, str) or not raw_path:
            missing.append(f"{field}:<missing>")
            continue
        path = _resolve_path(raw_path, bundle_dir)
        if not path.exists() or not path.is_dir():
            missing.append(f"{field}:{raw_path}")
    return missing


def _bundle_missing_return_reject_archive_dirs(bundle_dir: Path, bundle_manifest: dict[str, Any]) -> list[str]:
    return_archive = bundle_manifest.get("return_reject_archive")
    if not isinstance(return_archive, dict):
        return []
    missing: list[str] = []
    for field in ("annotator_archive", "adjudication_archive"):
        raw_path = return_archive.get(field)
        if not isinstance(raw_path, str) or not raw_path:
            missing.append(f"{field}:<missing>")
            continue
        path = _resolve_path(raw_path, bundle_dir)
        if not path.exists() or not path.is_dir():
            missing.append(f"{field}:{raw_path}")
    return missing


def _bundle_missing_operator_scripts(bundle_dir: Path, bundle_manifest: dict[str, Any]) -> list[str]:
    operator_scripts = bundle_manifest.get("operator_scripts")
    if not isinstance(operator_scripts, dict):
        return []
    missing: list[str] = []
    for script_id, raw_path in sorted(operator_scripts.items()):
        if not isinstance(raw_path, str) or not raw_path:
            missing.append(f"{script_id}:<missing>")
            continue
        path = _resolve_path(raw_path, bundle_dir)
        if not path.exists():
            missing.append(f"{script_id}:{raw_path}")
            continue
        if not path.is_file():
            missing.append(f"{script_id}:{raw_path}:not_file")
            continue
        if path.stat().st_mode & 0o111 == 0:
            missing.append(f"{script_id}:{raw_path}:not_executable")
    return missing


def _bundle_handoff_manifest_signature(bundle_dir: Path, bundle_manifest: dict[str, Any]) -> dict[str, Any]:
    raw_path = bundle_manifest.get("handoff_manifest_file")
    if not isinstance(raw_path, str) or not raw_path:
        return {}
    handoff_path = _resolve_path(raw_path, bundle_dir)
    if not handoff_path.exists():
        return {}
    try:
        handoff_manifest = read_json(handoff_path)
    except Exception:  # noqa: BLE001 - verifier should report malformed handoff manifests as mismatch
        return {}
    if not isinstance(handoff_manifest, dict):
        return {}
    return {
        "schema_version": handoff_manifest.get("schema_version"),
        "bundle_dir": handoff_manifest.get("bundle_dir"),
        "bundle_manifest_file": handoff_manifest.get("bundle_manifest_file"),
        "status": handoff_manifest.get("status"),
        "ready_for_merge": handoff_manifest.get("ready_for_merge"),
        "ready_for_finalize": handoff_manifest.get("ready_for_finalize"),
        "return_inbox": handoff_manifest.get("return_inbox"),
        "return_archive": handoff_manifest.get("return_archive"),
        "return_reject_archive": handoff_manifest.get("return_reject_archive"),
        "rejected_return_summary": handoff_manifest.get("rejected_return_summary"),
        "rejected_returns_report_file": handoff_manifest.get("rejected_returns_report_file"),
        "return_inbox_watch_file": handoff_manifest.get("return_inbox_watch_file"),
        "return_inbox_sync_report_file": handoff_manifest.get("return_inbox_sync_report_file"),
        "return_inbox_state_file": handoff_manifest.get("return_inbox_state_file"),
        "pending_return_packets": handoff_manifest.get("pending_return_packets"),
        "recommended_next_command_id": handoff_manifest.get("recommended_next_command_id"),
        "recommended_next_command": handoff_manifest.get("recommended_next_command"),
        "recommended_next_script_id": handoff_manifest.get("recommended_next_script_id"),
        "recommended_next_script": handoff_manifest.get("recommended_next_script"),
        "recommended_next_script_file": handoff_manifest.get("recommended_next_script_file"),
        "return_inbox_sync_status": handoff_manifest.get("return_inbox_sync_status"),
        "return_inbox_sync_next_command_id": handoff_manifest.get("return_inbox_sync_next_command_id"),
        "return_inbox_sync_next_command": handoff_manifest.get("return_inbox_sync_next_command"),
        "return_inbox_sync_next_script_id": handoff_manifest.get("return_inbox_sync_next_script_id"),
        "return_inbox_sync_next_script": handoff_manifest.get("return_inbox_sync_next_script"),
        "return_inbox_sync_next_script_file": handoff_manifest.get("return_inbox_sync_next_script_file"),
        "return_inbox_watch_status": handoff_manifest.get("return_inbox_watch_status"),
        "return_inbox_watch_stop_reason": handoff_manifest.get("return_inbox_watch_stop_reason"),
        "return_inbox_watch_next_command_id": handoff_manifest.get("return_inbox_watch_next_command_id"),
        "return_inbox_watch_next_command": handoff_manifest.get("return_inbox_watch_next_command"),
        "return_inbox_watch_next_script_id": handoff_manifest.get("return_inbox_watch_next_script_id"),
        "return_inbox_watch_next_script": handoff_manifest.get("return_inbox_watch_next_script"),
        "return_inbox_watch_next_script_file": handoff_manifest.get("return_inbox_watch_next_script_file"),
        "annotator_packet_ids": sorted(handoff_manifest.get("annotator_packets", {}).keys())
        if isinstance(handoff_manifest.get("annotator_packets"), dict)
        else [],
        "operator_script_ids": sorted(handoff_manifest.get("operator_scripts", {}).keys())
        if isinstance(handoff_manifest.get("operator_scripts"), dict)
        else [],
    }


def _bundle_return_inbox_watch_signature(bundle_dir: Path, bundle_manifest: dict[str, Any]) -> dict[str, Any]:
    raw_path = bundle_manifest.get("return_inbox_watch_file")
    if not isinstance(raw_path, str) or not raw_path:
        return {}
    watch_path = _resolve_path(raw_path, bundle_dir)
    if not watch_path.exists():
        return {}
    try:
        watch_report = read_json(watch_path)
    except Exception:
        return {}
    if not isinstance(watch_report, dict):
        return {}
    return {
        "schema_version": watch_report.get("schema_version"),
        "bundle_dir": watch_report.get("bundle_dir"),
        "bundle_manifest_file": watch_report.get("bundle_manifest_file"),
        "handoff_manifest_file": watch_report.get("handoff_manifest_file"),
        "return_inbox_state_file": watch_report.get("return_inbox_state_file"),
        "rejected_returns_report_file": watch_report.get("rejected_returns_report_file"),
        "return_inbox": watch_report.get("return_inbox"),
        "return_archive": watch_report.get("return_archive"),
        "return_reject_archive": watch_report.get("return_reject_archive"),
        "next_command_id": watch_report.get("next_command_id"),
        "next_command": watch_report.get("next_command"),
        "next_script_id": watch_report.get("next_script_id"),
        "next_script": watch_report.get("next_script"),
        "next_script_file": watch_report.get("next_script_file"),
        "watch_stop_exit_codes": watch_report.get("watch_stop_exit_codes"),
    }


def _expected_bundle_return_inbox_watch_signature(bundle_dir: Path, bundle_manifest: dict[str, Any]) -> dict[str, Any]:
    stop_reason = None
    raw_watch_path = bundle_manifest.get("return_inbox_watch_file")
    if isinstance(raw_watch_path, str) and raw_watch_path:
        watch_path = _resolve_path(raw_watch_path, bundle_dir)
        if watch_path.exists():
            try:
                watch_payload = read_json(watch_path)
            except Exception:  # noqa: BLE001 - malformed sidecar should still fail summary matching
                watch_payload = None
            if isinstance(watch_payload, dict) and isinstance(watch_payload.get("stop_reason"), str):
                stop_reason = watch_payload.get("stop_reason")
    next_action = _bundle_watch_next_action(bundle_manifest, stop_reason=stop_reason)
    return {
        "schema_version": "amst-human-audit-inbox-watch-v1",
        "bundle_dir": str(bundle_dir),
        "bundle_manifest_file": str(bundle_dir / "bundle_manifest.json"),
        "handoff_manifest_file": bundle_manifest.get("handoff_manifest_file"),
        "return_inbox_state_file": bundle_manifest.get("return_inbox_state_file"),
        "rejected_returns_report_file": bundle_manifest.get("rejected_returns_report_file"),
        "return_inbox": bundle_manifest.get("return_inbox"),
        "return_archive": bundle_manifest.get("return_archive"),
        "return_reject_archive": bundle_manifest.get("return_reject_archive"),
        "next_command_id": next_action.get("next_command_id"),
        "next_command": next_action.get("next_command"),
        "next_script_id": next_action.get("next_script_id"),
        "next_script": next_action.get("next_script"),
        "next_script_file": next_action.get("next_script_file"),
        "watch_stop_exit_codes": bundle_manifest.get("watch_stop_exit_codes"),
    }


def _bundle_return_inbox_sync_signature(bundle_dir: Path, bundle_manifest: dict[str, Any]) -> dict[str, Any]:
    raw_path = bundle_manifest.get("return_inbox_sync_report_file")
    if not isinstance(raw_path, str) or not raw_path:
        return {}
    sync_path = _resolve_path(raw_path, bundle_dir)
    if not sync_path.exists():
        return {}
    try:
        sync_report = read_json(sync_path)
    except Exception:
        return {}
    if not isinstance(sync_report, dict):
        return {}
    return {
        "schema_version": sync_report.get("schema_version"),
        "bundle_dir": sync_report.get("bundle_dir"),
        "return_inbox_state_file": sync_report.get("return_inbox_state_file"),
        "next_command_id": sync_report.get("next_command_id"),
        "next_command": sync_report.get("next_command"),
        "next_script_id": sync_report.get("next_script_id"),
        "next_script": sync_report.get("next_script"),
        "next_script_file": sync_report.get("next_script_file"),
    }


def _expected_bundle_return_inbox_sync_signature(bundle_dir: Path, bundle_manifest: dict[str, Any]) -> dict[str, Any]:
    next_action = _bundle_recommended_next_refs(bundle_manifest)
    return {
        "schema_version": "amst-human-audit-inbox-sync-v1",
        "bundle_dir": str(bundle_dir),
        "return_inbox_state_file": str(bundle_dir / "return_inbox_state.json"),
        "next_command_id": next_action.get("next_command_id"),
        "next_command": next_action.get("next_command"),
        "next_script_id": next_action.get("next_script_id"),
        "next_script": next_action.get("next_script"),
        "next_script_file": next_action.get("next_script_file"),
    }


def _expected_bundle_handoff_manifest_signature(bundle_dir: Path, bundle_manifest: dict[str, Any]) -> dict[str, Any]:
    runtime_snapshot = _bundle_handoff_runtime_snapshot(bundle_dir, bundle_manifest)
    return {
        "schema_version": "amst-human-audit-handoff-manifest-v1",
        "bundle_dir": str(bundle_dir),
        "bundle_manifest_file": str(bundle_dir / "bundle_manifest.json"),
        "status": bundle_manifest.get("status"),
        "ready_for_merge": bundle_manifest.get("ready_for_merge"),
        "ready_for_finalize": bundle_manifest.get("ready_for_finalize"),
        "return_inbox": bundle_manifest.get("return_inbox"),
        "return_archive": bundle_manifest.get("return_archive"),
        "return_reject_archive": bundle_manifest.get("return_reject_archive"),
        "rejected_return_summary": bundle_manifest.get("rejected_return_summary"),
        "rejected_returns_report_file": bundle_manifest.get("rejected_returns_report_file"),
        "return_inbox_watch_file": bundle_manifest.get("return_inbox_watch_file"),
        "return_inbox_sync_report_file": bundle_manifest.get("return_inbox_sync_report_file"),
        "return_inbox_state_file": bundle_manifest.get("return_inbox_state_file"),
        "pending_return_packets": bundle_manifest.get("pending_return_packets"),
        "recommended_next_command_id": bundle_manifest.get("recommended_next_command_id"),
        "recommended_next_command": bundle_manifest.get("recommended_next_command"),
        "recommended_next_script_id": bundle_manifest.get("recommended_next_script_id"),
        "recommended_next_script": bundle_manifest.get("recommended_next_script"),
        "recommended_next_script_file": bundle_manifest.get("recommended_next_script_file"),
        "annotator_packet_ids": sorted(bundle_manifest.get("annotator_packets", {}).keys())
        if isinstance(bundle_manifest.get("annotator_packets"), dict)
        else [],
        "operator_script_ids": sorted(bundle_manifest.get("operator_scripts", {}).keys())
        if isinstance(bundle_manifest.get("operator_scripts"), dict)
        else [],
        "return_inbox_sync_status": runtime_snapshot.get("return_inbox_sync_status"),
        "return_inbox_sync_next_command_id": runtime_snapshot.get("return_inbox_sync_next_command_id"),
        "return_inbox_sync_next_command": runtime_snapshot.get("return_inbox_sync_next_command"),
        "return_inbox_sync_next_script_id": runtime_snapshot.get("return_inbox_sync_next_script_id"),
        "return_inbox_sync_next_script": runtime_snapshot.get("return_inbox_sync_next_script"),
        "return_inbox_sync_next_script_file": runtime_snapshot.get("return_inbox_sync_next_script_file"),
        "return_inbox_watch_status": runtime_snapshot.get("return_inbox_watch_status"),
        "return_inbox_watch_stop_reason": runtime_snapshot.get("return_inbox_watch_stop_reason"),
        "return_inbox_watch_next_command_id": runtime_snapshot.get("return_inbox_watch_next_command_id"),
        "return_inbox_watch_next_command": runtime_snapshot.get("return_inbox_watch_next_command"),
        "return_inbox_watch_next_script_id": runtime_snapshot.get("return_inbox_watch_next_script_id"),
        "return_inbox_watch_next_script": runtime_snapshot.get("return_inbox_watch_next_script"),
        "return_inbox_watch_next_script_file": runtime_snapshot.get("return_inbox_watch_next_script_file"),
    }


def _bundle_missing_annotator_packet_files(bundle_dir: Path, bundle_manifest: dict[str, Any]) -> list[str]:
    annotator_packets = bundle_manifest.get("annotator_packets")
    if not isinstance(annotator_packets, dict):
        return []
    missing: list[str] = []
    for annotator, packet in sorted(annotator_packets.items()):
        if not isinstance(packet, dict):
            missing.append(f"{annotator}:<invalid-packet>")
            continue
        for field in ("packet_dir", "sheet_file", "task_file", "readme_file", "packet_manifest_file", "archive_file"):
            raw_path = packet.get(field)
            if not isinstance(raw_path, str) or not raw_path:
                missing.append(f"{annotator}:{field}:<missing>")
                continue
            path = _resolve_path(raw_path, bundle_dir)
            if not path.exists():
                missing.append(f"{annotator}:{field}:{raw_path}")
        documentation_files = packet.get("documentation_files")
        if isinstance(documentation_files, dict):
            for name, raw_path in sorted(documentation_files.items()):
                path = _resolve_path(str(raw_path), bundle_dir)
                if not path.exists():
                    missing.append(f"{annotator}:doc:{name}:{raw_path}")
        sandbox_sample = packet.get("sandbox_sample")
        if isinstance(sandbox_sample, dict):
            for field in (
                "sandbox_task_manifest_file",
                "completed_annotations_file",
                "attestation_file",
                "agreement_file",
                "verification_report_file",
            ):
                raw_path = sandbox_sample.get(field)
                if not isinstance(raw_path, str) or not raw_path:
                    missing.append(f"{annotator}:sandbox:{field}:<missing>")
                    continue
                path = _resolve_path(raw_path, bundle_dir)
                if not path.exists():
                    missing.append(f"{annotator}:sandbox:{field}:{raw_path}")
    return missing


def _bundle_missing_adjudication_packet_files(bundle_dir: Path, bundle_manifest: dict[str, Any]) -> list[str]:
    adjudication_packet = bundle_manifest.get("adjudication_packet")
    if not isinstance(adjudication_packet, dict) or not adjudication_packet:
        return []
    missing: list[str] = []
    for field in ("packet_dir", "sheet_file", "task_file", "readme_file", "packet_manifest_file", "archive_file"):
        raw_path = adjudication_packet.get(field)
        if not isinstance(raw_path, str) or not raw_path:
            missing.append(f"{field}:<missing>")
            continue
        path = _resolve_path(raw_path, bundle_dir)
        if not path.exists():
            missing.append(f"{field}:{raw_path}")
    documentation_files = adjudication_packet.get("documentation_files")
    if isinstance(documentation_files, dict):
        for name, raw_path in sorted(documentation_files.items()):
            path = _resolve_path(str(raw_path), bundle_dir)
            if not path.exists():
                missing.append(f"doc:{name}:{raw_path}")
    return missing


def _copy_annotator_packet_sandbox(
    bundle_dir: Path,
    packet_dir: Path,
    sandbox_sample: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(sandbox_sample, dict) or not sandbox_sample:
        return None
    sandbox_task_manifest_value = sandbox_sample.get("sandbox_task_manifest_file")
    if not isinstance(sandbox_task_manifest_value, str) or not sandbox_task_manifest_value:
        return None
    sandbox_task_manifest_path = _resolve_path(sandbox_task_manifest_value, bundle_dir)
    if not sandbox_task_manifest_path.exists():
        return None
    source_sandbox_dir = sandbox_task_manifest_path.parent
    target_sandbox_dir = packet_dir / source_sandbox_dir.name
    shutil.copytree(source_sandbox_dir, target_sandbox_dir, dirs_exist_ok=True)
    target_task_manifest_path = target_sandbox_dir / "task_manifest.json"
    if target_task_manifest_path.exists():
        target_task_manifest = read_json(target_task_manifest_path)
        if isinstance(target_task_manifest, dict):
            write_json(
                target_task_manifest_path,
                _localize_human_audit_task_manifest_payload(
                    target_task_manifest,
                    source_bases=(target_sandbox_dir,),
                    target_base=target_sandbox_dir,
                ),
            )
    target_attestation_path = target_sandbox_dir / "attestation.json"
    if target_attestation_path.exists():
        target_attestation = read_json(target_attestation_path)
        if isinstance(target_attestation, dict):
            write_json(
                target_attestation_path,
                _localize_human_audit_attestation_payload(
                    target_attestation,
                    source_bases=(target_sandbox_dir,),
                    target_base=target_sandbox_dir,
                ),
            )
    target_verification_path = target_sandbox_dir / "verification_report.json"
    if target_verification_path.exists():
        target_verification = read_json(target_verification_path)
        if isinstance(target_verification, dict):
            write_json(
                target_verification_path,
                _localize_human_audit_verification_payload(
                    target_verification,
                    source_bases=(target_sandbox_dir,),
                    target_base=target_sandbox_dir,
                ),
            )
    packet_sample: dict[str, Any] = {
        "schema_version": sandbox_sample.get("schema_version"),
        "source_task_manifest_file": sandbox_sample.get("source_task_manifest_file"),
        "sandbox_task_manifest_file": _relative_or_absolute(target_sandbox_dir / "task_manifest.json", bundle_dir),
        "completed_annotations_file": _relative_or_absolute(target_sandbox_dir / "completed_annotations.jsonl", bundle_dir),
        "attestation_file": _relative_or_absolute(target_sandbox_dir / "attestation.json", bundle_dir),
        "agreement_file": _relative_or_absolute(target_sandbox_dir / "agreement.json", bundle_dir),
        "verification_report_file": _relative_or_absolute(target_sandbox_dir / "verification_report.json", bundle_dir),
        "selected_items": json.loads(json.dumps(sandbox_sample.get("selected_items", []))),
        "num_items": sandbox_sample.get("num_items"),
        "num_annotators": sandbox_sample.get("num_annotators"),
        "agreement_metrics": json.loads(json.dumps(sandbox_sample.get("agreement_metrics"))),
    }
    merge_summary = sandbox_sample.get("merge_summary")
    if isinstance(merge_summary, dict):
        packet_sample["merge_summary"] = dict(merge_summary)
        packet_sample["merge_summary"]["task_manifest_file"] = packet_sample["sandbox_task_manifest_file"]
        packet_sample["merge_summary"]["annotations_file"] = packet_sample["completed_annotations_file"]
    attestation_path = target_sandbox_dir / "attestation.json"
    if attestation_path.exists():
        attestation = read_json(attestation_path)
        if isinstance(attestation, dict):
            packet_sample["attestation"] = _localize_human_audit_attestation_payload(
                attestation,
                source_bases=(target_sandbox_dir,),
                target_base=bundle_dir,
            )
    verification_path = target_sandbox_dir / "verification_report.json"
    if verification_path.exists():
        verification = read_json(verification_path)
        if isinstance(verification, dict):
            packet_sample["verification"] = _localize_human_audit_verification_payload(
                verification,
                source_bases=(target_sandbox_dir,),
                target_base=bundle_dir,
            )
    root_ref = _bundle_root_ref(bundle_dir)
    if root_ref is not None:
        packet_sample["root"] = root_ref
    return packet_sample


def _packet_localize_sandbox_sample(
    sandbox_sample: dict[str, Any] | None,
    bundle_dir: Path,
    packet_dir: Path,
) -> dict[str, Any] | None:
    if not isinstance(sandbox_sample, dict) or not sandbox_sample:
        return None
    sandbox_dir = packet_dir / "sandbox"
    return _localize_human_audit_sandbox_sample_payload(
        sandbox_sample,
        source_bases=(bundle_dir, sandbox_dir),
        target_base=packet_dir,
    )


def _build_annotator_packets(
    bundle_dir: Path,
    task_files: dict[str, str],
    annotation_sheet_files: dict[str, str],
    documentation_files: dict[str, str],
    sandbox_sample: dict[str, Any] | None,
    annotator_progress: Any,
) -> dict[str, dict[str, Any]]:
    packets: dict[str, dict[str, Any]] = {}
    packet_root = bundle_dir / "annotators"
    for annotator_id in sorted(task_files):
        task_file = _resolve_path(task_files[annotator_id], bundle_dir)
        sheet_file = _resolve_path(annotation_sheet_files[annotator_id], bundle_dir)
        packet_dir = packet_root / _safe_filename(annotator_id)
        packet_dir.mkdir(parents=True, exist_ok=True)
        packet_task_path = packet_dir / task_file.name
        packet_sheet_path = packet_dir / sheet_file.name
        shutil.copy2(task_file, packet_task_path)
        shutil.copy2(sheet_file, packet_sheet_path)

        packet_docs_dir = packet_dir / "docs"
        packet_doc_map: dict[str, str] = {}
        packet_local_doc_map: dict[str, str] = {}
        for name, raw_path in sorted(documentation_files.items()):
            source_path = _resolve_path(raw_path, bundle_dir)
            packet_docs_dir.mkdir(parents=True, exist_ok=True)
            target_path = packet_docs_dir / Path(source_path).name
            shutil.copy2(source_path, target_path)
            packet_doc_map[name] = _relative_or_absolute(target_path, bundle_dir)
            packet_local_doc_map[name] = _relative_or_absolute(target_path, packet_dir)

        packet_sandbox = _copy_annotator_packet_sandbox(bundle_dir, packet_dir, sandbox_sample)
        packet_sandbox_readme = _packet_localize_sandbox_sample(packet_sandbox, bundle_dir, packet_dir)

        packet_manifest_path = packet_dir / "packet_manifest.json"
        write_json(
            packet_manifest_path,
            _build_annotator_packet_manifest(
                annotator_id,
                packet_dir,
                packet_sheet_path,
                packet_task_path,
                packet_local_doc_map,
                packet_sandbox_readme,
                packet_manifest_path.name,
            ),
        )
        packet_readme_path = packet_dir / "README.md"
        packet_readme_path.write_text(
            _annotator_packet_readme(
                annotator_id,
                packet_sheet_path,
                packet_task_path,
                packet_local_doc_map,
                packet_sandbox_readme,
                annotator_progress.get(annotator_id) if isinstance(annotator_progress, dict) else None,
                packet_manifest_path.name,
            ),
            encoding="utf-8",
        )
        archive_path = packet_root / f"{packet_dir.name}.zip"
        _write_deterministic_zip(
            output_zip=archive_path,
            source_dir=packet_dir,
            root_dir=packet_root,
        )
        packets[annotator_id] = {
            "packet_dir": _relative_or_absolute(packet_dir, bundle_dir),
            "sheet_file": _relative_or_absolute(packet_sheet_path, bundle_dir),
            "task_file": _relative_or_absolute(packet_task_path, bundle_dir),
            "readme_file": _relative_or_absolute(packet_readme_path, bundle_dir),
            "packet_manifest_file": _relative_or_absolute(packet_manifest_path, bundle_dir),
            "archive_file": _relative_or_absolute(archive_path, bundle_dir),
            "documentation_files": packet_doc_map,
            "sandbox_sample": packet_sandbox,
        }
    return packets


def _build_adjudication_packet(
    bundle_dir: Path,
    adjudication_package: dict[str, Any] | None,
    documentation_files: dict[str, str],
) -> dict[str, Any] | None:
    if not isinstance(adjudication_package, dict) or not adjudication_package:
        return None
    tasks_value = adjudication_package.get("adjudication_tasks_file")
    if not isinstance(tasks_value, str) or not tasks_value:
        return None
    tasks_path = _resolve_path(tasks_value, bundle_dir)
    if not tasks_path.exists():
        return None

    sheet_summary = write_human_audit_adjudication_sheet(tasks_path, bundle_dir / "adjudication" / "adjudication.csv")
    sheet_path = Path(sheet_summary["sheet_file"])
    packet_dir = bundle_dir / "adjudication" / "packet"
    packet_dir.mkdir(parents=True, exist_ok=True)
    packet_task_path = packet_dir / tasks_path.name
    packet_sheet_path = packet_dir / sheet_path.name
    shutil.copy2(tasks_path, packet_task_path)
    shutil.copy2(sheet_path, packet_sheet_path)

    packet_docs_dir = packet_dir / "docs"
    packet_doc_map: dict[str, str] = {}
    packet_local_doc_map: dict[str, str] = {}
    for name, raw_path in sorted(documentation_files.items()):
        source_path = _resolve_path(raw_path, bundle_dir)
        packet_docs_dir.mkdir(parents=True, exist_ok=True)
        target_path = packet_docs_dir / Path(source_path).name
        shutil.copy2(source_path, target_path)
        packet_doc_map[name] = _relative_or_absolute(target_path, bundle_dir)
        packet_local_doc_map[name] = _relative_or_absolute(target_path, packet_dir)

    packet_manifest_path = packet_dir / "packet_manifest.json"
    write_json(
        packet_manifest_path,
        _build_adjudication_packet_manifest(
            packet_dir,
            packet_sheet_path,
            packet_task_path,
            packet_local_doc_map,
            adjudication_package,
            packet_manifest_path.name,
        ),
    )
    packet_readme_path = packet_dir / "README.md"
    packet_readme_path.write_text(
        _adjudication_packet_readme(
            packet_sheet_path,
            packet_task_path,
            packet_local_doc_map,
            adjudication_package,
            packet_manifest_path.name,
        ),
        encoding="utf-8",
    )
    archive_path = packet_dir.parent / f"{packet_dir.name}.zip"
    _write_deterministic_zip(
        output_zip=archive_path,
        source_dir=packet_dir,
        root_dir=packet_dir.parent,
    )
    return {
        "packet_dir": _relative_or_absolute(packet_dir, bundle_dir),
        "sheet_file": _relative_or_absolute(packet_sheet_path, bundle_dir),
        "task_file": _relative_or_absolute(packet_task_path, bundle_dir),
        "readme_file": _relative_or_absolute(packet_readme_path, bundle_dir),
        "packet_manifest_file": _relative_or_absolute(packet_manifest_path, bundle_dir),
        "archive_file": _relative_or_absolute(archive_path, bundle_dir),
        "documentation_files": packet_doc_map,
        "task_file_sha256": _file_sha256(tasks_path),
        "num_items": adjudication_package.get("num_disagreement_items"),
        "status": adjudication_package.get("status"),
    }


def _annotator_packet_readme(
    annotator_id: str,
    sheet_path: Path,
    task_path: Path,
    documentation_files: dict[str, str],
    sandbox_sample: dict[str, str] | None,
    progress: Any,
    packet_manifest_file: str,
) -> str:
    workload_lines = []
    if isinstance(progress, dict):
        workload_lines.append(f"- expected_annotations: `{progress.get('num_expected_annotations')}`")
        workload_lines.append(f"- started_annotations: `{progress.get('num_started_annotations')}`")
        workload_lines.append(f"- completed_annotations: `{progress.get('num_matched_annotations')}`")
        workload_lines.append(f"- missing_annotations: `{progress.get('num_missing_annotations')}`")
    documentation_text = "\n".join(
        f"- `{name}`: `{path}`" for name, path in sorted(documentation_files.items())
    ) if documentation_files else "- No packet-local docs are available."
    sandbox_text = _annotator_packet_sandbox_lines(sandbox_sample)
    workload_text = "\n".join(workload_lines) if workload_lines else "- No workload summary is available."
    return (
        f"# Human Audit Packet: {annotator_id}\n\n"
        "This packet is intended for a single annotator.\n\n"
        "## Files\n\n"
        f"- `sheet.csv`: `{sheet_path.name}`\n"
        f"- `task.jsonl`: `{task_path.name}`\n"
        f"- `packet_manifest.json`: `{packet_manifest_file}`\n\n"
        "## Documentation\n\n"
        f"{documentation_text}\n\n"
        "## Sandbox Sample\n\n"
        f"{sandbox_text}\n\n"
        "## Workload\n\n"
        f"{workload_text}\n\n"
        "## Instructions\n\n"
        "1. Read `docs/annotation_guideline.md` first when present.\n"
        "2. Review the worked example under `sandbox/` before starting the live task.\n"
        "3. Fill only `sheet.csv` unless you have a strong reason to work in JSONL.\n"
        "4. Do not edit immutable scenario fields.\n"
        "5. Return the edited `sheet.csv` or a re-zipped copy of this packet to the benchmark operator.\n"
    )


def _adjudication_packet_readme(
    sheet_path: Path,
    task_path: Path,
    documentation_files: dict[str, str],
    adjudication_package: dict[str, Any],
    packet_manifest_file: str,
) -> str:
    documentation_text = "\n".join(
        f"- `{name}`: `{path}`" for name, path in sorted(documentation_files.items())
    ) if documentation_files else "- No packet-local docs are available."
    return (
        "# Human Audit Adjudication Packet\n\n"
        "This packet is intended for a single adjudicator to resolve disagreement items only.\n\n"
        "## Files\n\n"
        f"- `sheet.csv`: `{sheet_path.name}`\n"
        f"- `task.jsonl`: `{task_path.name}`\n"
        f"- `packet_manifest.json`: `{packet_manifest_file}`\n\n"
        "## Documentation\n\n"
        f"{documentation_text}\n\n"
        "## Disagreement Scope\n\n"
        f"- `status`: `{adjudication_package.get('status')}`\n"
        f"- `num_disagreement_items`: `{adjudication_package.get('num_disagreement_items')}`\n"
        f"- `num_disagreement_fields`: `{adjudication_package.get('num_disagreement_fields')}`\n\n"
        "## Instructions\n\n"
        "1. Read `docs/annotation_guideline.md` first when present.\n"
        "2. Fill `sheet.csv` with your adjudication decisions and your `adjudicator_id`.\n"
        "3. Only disagreement target fields should contain TRUE/FALSE; non-target fields must remain blank or `N/A`.\n"
        "4. Do not edit immutable scenario fields in `task.jsonl`.\n"
        "5. Return the edited `sheet.csv` or a re-zipped copy of this packet to the benchmark operator.\n"
    )


def _write_deterministic_zip(*, output_zip: Path, source_dir: Path, root_dir: Path) -> None:
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            arcname = path.relative_to(root_dir).as_posix()
            info = zipfile.ZipInfo(filename=arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            archive.writestr(info, path.read_bytes())


def _build_annotator_packet_manifest(
    annotator_id: str,
    packet_dir: Path,
    sheet_path: Path,
    task_path: Path,
    documentation_files: dict[str, str],
    sandbox_sample: dict[str, str] | None,
    packet_manifest_file: str,
) -> dict[str, Any]:
    manifest = {
        "schema_version": HUMAN_AUDIT_ANNOTATOR_PACKET_SCHEMA_VERSION,
        "packet_type": "annotator",
        "annotator_id": str(annotator_id),
        "sheet_file": sheet_path.name,
        "task_file": task_path.name,
        "readme_file": "README.md",
        "packet_manifest_file": packet_manifest_file,
        "documentation_files": dict(documentation_files),
        "sandbox_sample": sandbox_sample,
        "task_identity_digest": _task_identity_digest_from_path(task_path),
    }
    root_ref = _bundle_root_ref(packet_dir)
    if root_ref is not None:
        manifest["root"] = root_ref
    return manifest


def _build_adjudication_packet_manifest(
    packet_dir: Path,
    sheet_path: Path,
    task_path: Path,
    documentation_files: dict[str, str],
    adjudication_package: dict[str, Any],
    packet_manifest_file: str,
) -> dict[str, Any]:
    manifest = {
        "schema_version": HUMAN_AUDIT_ADJUDICATION_PACKET_SCHEMA_VERSION,
        "packet_type": "adjudication",
        "sheet_file": sheet_path.name,
        "task_file": task_path.name,
        "readme_file": "README.md",
        "packet_manifest_file": packet_manifest_file,
        "documentation_files": dict(documentation_files),
        "task_file_sha256": _file_sha256(task_path),
        "num_items": adjudication_package.get("num_disagreement_items"),
        "status": adjudication_package.get("status"),
    }
    root_ref = _bundle_root_ref(packet_dir)
    if root_ref is not None:
        manifest["root"] = root_ref
    return manifest


def _manifest_template_paths(audit_plan: dict[str, Any], manifest_dir: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    template_files = audit_plan.get("audit_template_files")
    if isinstance(template_files, dict):
        paths.extend(_resolve_path(str(path), manifest_dir) for path in template_files.values())
    raw_template = audit_plan.get("audit_template_file")
    if raw_template:
        paths.append(_resolve_path(str(raw_template), manifest_dir))
    return tuple(paths)


def _materialize_selected_template_files(
    raw_paths: Iterable[str | Path],
    manifest_dir: Path,
    selected_item_keys: set[ItemKey],
    output_dir: Path,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    template_paths: list[Path] = []
    for raw_path in raw_paths:
        source_path = _resolve_path(str(raw_path), manifest_dir)
        if not source_path.exists():
            continue
        selected_records: list[dict[str, Any]] = []
        with source_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = _record_item_key(record)
                if key in selected_item_keys:
                    selected_records.append(record)
        if not selected_records:
            continue
        target_path = output_dir / source_path.name
        target_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in selected_records) + "\n",
            encoding="utf-8",
        )
        template_paths.append(target_path)
    return tuple(template_paths)


def _load_template_items(template_paths: tuple[Path, ...], errors: list[str]) -> dict[ItemKey, dict[str, Any]]:
    items: dict[ItemKey, dict[str, Any]] = {}
    for path in template_paths:
        if not path.exists():
            errors.append(f"audit template does not exist: {path}")
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"audit template {path}:{line_number} is not valid JSONL: {exc}")
                    continue
                key = _record_item_key(record)
                if key is None:
                    errors.append(f"audit template {path}:{line_number} is missing case_id or query_id")
                    continue
                if key in items:
                    errors.append(f"audit template duplicates item {_format_item(key)}")
                items[key] = record
    return items


def _group_annotations(annotations: tuple[AuditAnnotation, ...]) -> dict[ItemKey, list[AuditAnnotation]]:
    grouped: dict[ItemKey, list[AuditAnnotation]] = defaultdict(list)
    for annotation in annotations:
        grouped[annotation.item_id].append(annotation)
    return grouped


def _expected_applicable_counts(expected_items: dict[ItemKey, dict[str, Any]]) -> dict[str, int]:
    counts = {field: 0 for field in AUDIT_CHECK_FIELDS}
    for record in expected_items.values():
        for field in _template_applicable_checks(record):
            counts[field] += 1
    return counts


def _template_applicable_checks(record: dict[str, Any]) -> frozenset[str]:
    raw = record.get("applicable_checks")
    if isinstance(raw, list):
        return frozenset(str(value) for value in raw if str(value) in AUDIT_CHECK_FIELDS)
    checks = set(AUDIT_CHECK_FIELDS)
    if not (record.get("counterfactual_group_id") or record.get("counterfactual_context")):
        checks.discard("counterfactual_target_state_only")
    return frozenset(checks)


def _semantic_checks(
    expected_items: dict[ItemKey, dict[str, Any]],
    grouped: dict[ItemKey, list[AuditAnnotation]],
) -> dict[str, Any]:
    return {
        "scenario_memory_required_alignment": _field_alignment_against_reference(
            "scenario_memory_required",
            "requires_memory",
            expected_items,
            grouped,
        ),
        "counterfactual_target_state_only_alignment": _field_alignment_against_reference(
            "counterfactual_target_state_only",
            "counterfactual_target_state_only",
            expected_items,
            grouped,
        ),
    }


def _disagreement_items(
    expected_items: dict[ItemKey, dict[str, Any]],
    grouped: dict[ItemKey, list[AuditAnnotation]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key, record in sorted(expected_items.items()):
        annotations = grouped.get(key, [])
        if len(annotations) < 2:
            continue
        applicable_checks = tuple(_template_applicable_checks(record))
        disagreement_fields: dict[str, dict[str, Any]] = {}
        consensus_checks: dict[str, bool] = {}
        for field in applicable_checks:
            votes = {
                annotation.annotator_id: annotation.checks[field]
                for annotation in annotations
                if field in annotation.applicable_checks and annotation.checks[field] is not None
            }
            unique_votes = {value for value in votes.values() if value is not None}
            if len(unique_votes) > 1:
                disagreement_fields[field] = {
                    "votes": dict(sorted(votes.items())),
                    "majority_vote": _majority_vote(annotations, field),
                }
            elif len(unique_votes) == 1:
                consensus_checks[field] = next(iter(unique_votes))
        if not disagreement_fields:
            continue
        item = dict(record)
        item["applicable_checks"] = list(applicable_checks)
        item["consensus_checks"] = consensus_checks
        item["disagreement_fields"] = disagreement_fields
        item["adjudication_checks"] = {field: None for field in sorted(disagreement_fields)}
        item["adjudication_status"] = "pending"
        item["adjudicator_id"] = None
        item.setdefault("notes", None)
        items.append(item)
    return items


def summarize_disagreement_summary(
    expected_items: dict[ItemKey, dict[str, Any]],
    grouped: dict[ItemKey, list[AuditAnnotation]],
) -> dict[str, Any]:
    items = _disagreement_items(expected_items, grouped)
    field_counts: dict[str, int] = defaultdict(int)
    tied_fields = 0
    for item in items:
        for field, payload in item["disagreement_fields"].items():
            field_counts[field] += 1
            if payload.get("majority_vote") is None:
                tied_fields += 1
    return {
        "num_items": len(expected_items),
        "num_disagreement_items": len(items),
        "num_disagreement_fields": sum(len(item["disagreement_fields"]) for item in items),
        "num_tied_fields": tied_fields,
        "fields_with_disagreements": {field: field_counts[field] for field in sorted(field_counts)},
        "disagreement_items": [
            {
                "case_id": item["case_id"],
                "query_id": item["query_id"],
                "probe_type": item.get("probe_type"),
                "domain": item.get("domain"),
                "disagreement_fields": sorted(item["disagreement_fields"]),
            }
            for item in items
        ],
    }


def _field_alignment_against_reference(
    field: str,
    reference_key: str,
    expected_items: dict[ItemKey, dict[str, Any]],
    grouped: dict[ItemKey, list[AuditAnnotation]],
) -> dict[str, Any]:
    applicable_items = 0
    majority_matches = 0
    majority_mismatches: list[str] = []
    unresolved_items: list[str] = []
    missing_reference_items: list[str] = []
    for key, record in sorted(expected_items.items()):
        if field not in _template_applicable_checks(record):
            continue
        applicable_items += 1
        reference = record.get("audit_reference", {}).get(reference_key)
        if reference is None:
            missing_reference_items.append(_format_item(key))
            continue
        majority_vote = _majority_vote(grouped.get(key, ()), field)
        if majority_vote is None:
            unresolved_items.append(_format_item(key))
            continue
        if bool(reference) == majority_vote:
            majority_matches += 1
        else:
            majority_mismatches.append(_format_item(key))
    return {
        "field": field,
        "reference_key": reference_key,
        "description": AUDIT_CHECK_DEFINITIONS.get(field),
        "num_applicable_items": applicable_items,
        "num_majority_matches": majority_matches,
        "num_majority_mismatches": len(majority_mismatches),
        "num_unresolved_items": len(unresolved_items),
        "num_missing_reference_items": len(missing_reference_items),
        "majority_mismatch_items": majority_mismatches[:50],
        "unresolved_items": unresolved_items[:50],
        "missing_reference_items": missing_reference_items[:50],
        "passed": not majority_mismatches and not unresolved_items and not missing_reference_items and applicable_items > 0,
    }


def _majority_vote(item_annotations: Iterable[AuditAnnotation], field: str) -> bool | None:
    votes = [annotation.checks[field] for annotation in item_annotations if field in annotation.applicable_checks]
    values = [vote for vote in votes if vote is not None]
    if not values:
        return None
    true_votes = sum(1 for vote in values if vote is True)
    false_votes = sum(1 for vote in values if vote is False)
    if true_votes == false_votes:
        return None
    return true_votes > false_votes


def _check_annotation_coverage(
    expected_items: dict[ItemKey, dict[str, Any]],
    grouped: dict[ItemKey, list[AuditAnnotation]],
    errors: list[str],
) -> None:
    expected_keys = set(expected_items)
    actual_keys = set(grouped)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing:
        errors.append(f"audit annotations missing template items: {_format_items(missing[:10])}")
    if extra:
        errors.append(f"audit annotations contain non-template items: {_format_items(extra[:10])}")

    for key in sorted(expected_keys & actual_keys):
        item_annotations = grouped[key]
        expected_applicable = _template_applicable_checks(expected_items[key])
        annotator_ids = [annotation.annotator_id for annotation in item_annotations]
        distinct_annotators = set(annotator_ids)
        if len(distinct_annotators) < 2:
            errors.append(f"audit item {_format_item(key)} has fewer than two annotators")
        if len(distinct_annotators) != len(annotator_ids):
            errors.append(f"audit item {_format_item(key)} duplicates an annotator_id")
        for annotation in item_annotations:
            if annotation.applicable_checks != expected_applicable:
                errors.append(
                    f"audit item {_format_item(key)} annotator {annotation.annotator_id} "
                    "has applicability fields inconsistent with the template"
                )
            missing_fields = [field for field in expected_applicable if annotation.checks[field] is None]
            if missing_fields:
                errors.append(
                    f"audit item {_format_item(key)} annotator {annotation.annotator_id} "
                    f"has null checks: {missing_fields}"
                )
            non_applicable_non_null = [
                field
                for field in AUDIT_CHECK_FIELDS
                if field not in expected_applicable and annotation.checks[field] is not None
            ]
            if non_applicable_non_null:
                errors.append(
                    f"audit item {_format_item(key)} annotator {annotation.annotator_id} "
                    f"has non-null non-applicable checks: {non_applicable_non_null}"
                )


def _compute_agreement_or_error(annotations: tuple[AuditAnnotation, ...], errors: list[str]) -> dict[str, Any] | None:
    try:
        return compute_agreement(annotations)
    except Exception as exc:  # noqa: BLE001 - surface malformed double-label data
        errors.append(f"could not compute agreement: {exc}")
        return None


def _check_agreement_completeness(
    agreement: dict[str, Any],
    expected_applicable_counts: dict[str, int],
    errors: list[str],
) -> None:
    for field in AUDIT_CHECK_FIELDS:
        metrics = agreement.get("fields", {}).get(field)
        if not isinstance(metrics, dict):
            errors.append(f"agreement field {field} is missing")
            continue
        expected_items = int(expected_applicable_counts.get(field, 0))
        if int(metrics.get("num_applicable_items", 0)) != expected_items:
            errors.append(
                f"agreement field {field} applicable item count mismatch: "
                f"expected={expected_items} actual={metrics.get('num_applicable_items')}"
            )
        if int(metrics.get("num_not_applicable_items", 0)) != max(int(agreement.get("num_items", 0)) - expected_items, 0):
            errors.append(f"agreement field {field} not-applicable item count mismatch")
        if int(metrics.get("num_items", 0)) != expected_items:
            errors.append(
                f"agreement field {field} compared item count mismatch: "
                f"expected={expected_items} actual={metrics.get('num_items')}"
            )
        if expected_items > 0 and int(metrics.get("num_skipped_items", 0)) != 0:
            errors.append(f"agreement field {field} has skipped items: {metrics.get('num_skipped_items')}")
        if expected_items > 0 and int(metrics.get("num_pairs", 0)) < expected_items:
            errors.append(f"agreement field {field} has too few annotation pairs")
        if expected_items > 0 and metrics.get("cohen_kappa") is None:
            errors.append(f"agreement field {field} has null cohen_kappa")


def _agreement_metrics_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    if int(actual.get("num_annotations", -1)) != int(expected.get("num_annotations", -2)):
        return False
    if int(actual.get("num_items", -1)) != int(expected.get("num_items", -2)):
        return False
    if int(actual.get("num_annotators", -1)) != int(expected.get("num_annotators", -2)):
        return False
    if list(actual.get("annotator_ids", [])) != list(expected.get("annotator_ids", [])):
        return False
    actual_fields = actual.get("fields", {})
    expected_fields = expected.get("fields", {})
    if not isinstance(actual_fields, dict) or not isinstance(expected_fields, dict):
        return False
    for field in AUDIT_CHECK_FIELDS:
        if not _field_metrics_match(actual_fields.get(field), expected_fields.get(field)):
            return False
    return True


def _field_metrics_match(actual: Any, expected: Any) -> bool:
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    keys = (
        "num_applicable_items",
        "num_not_applicable_items",
        "num_items",
        "num_skipped_items",
        "num_pairs",
        "num_agreements",
    )
    for key in keys:
        if int(actual.get(key, -1)) != int(expected.get(key, -2)):
            return False
    return _number_match(actual.get("percent_agreement"), expected.get("percent_agreement")) and _number_match(
        actual.get("cohen_kappa"),
        expected.get("cohen_kappa"),
    )


def _number_match(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return abs(float(left) - float(right)) <= 1e-12
    except (TypeError, ValueError):
        return False


def _verify_task_manifest_progress(task_manifest_path: Path, annotations_path: Path) -> tuple[str, ...]:
    try:
        progress = summarize_human_audit_progress(task_manifest_path, annotations_path=annotations_path)
    except Exception as exc:  # noqa: BLE001 - expose task manifest parse/coverage failures
        return (f"could not verify audit task manifest: {exc}",)
    if progress.get("ready_for_finalize") is True:
        return ()
    errors = [f"audit task manifest is not ready for finalize: status={progress.get('status')}"]
    errors.extend(str(error) for error in progress.get("errors", []))
    if progress.get("num_missing_annotations"):
        errors.append(f"audit task manifest missing annotations: {progress.get('num_missing_annotations')}")
    if progress.get("num_extra_annotations"):
        errors.append(f"audit task manifest extra annotations: {progress.get('num_extra_annotations')}")
    return tuple(errors)


def _verify_annotator_attestation(
    task_manifest_path: Path | None,
    annotations_path: Path,
    attestation_path: Path,
    annotations: tuple[AuditAnnotation, ...],
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        attestation = read_json(attestation_path)
    except Exception as exc:
        return (f"could not load annotator attestation: {exc}",)
    if not isinstance(attestation, dict):
        return ("annotator attestation must be a JSON object",)
    if attestation.get("attestation_schema_version") != HUMAN_AUDIT_ATTESTATION_SCHEMA_VERSION:
        errors.append(f"annotator attestation schema must be {HUMAN_AUDIT_ATTESTATION_SCHEMA_VERSION}")
    if task_manifest_path is None:
        errors.append("audit_task_manifest_file is required when annotator attestation is provided")
    else:
        errors.extend(
            _check_attestation_path(
                attestation,
                "task_manifest_file",
                task_manifest_path,
                attestation_path.parent,
            )
        )
    errors.extend(_check_attestation_path(attestation, "annotations_file", annotations_path, attestation_path.parent))
    expected_annotators = sorted({annotation.annotator_id for annotation in annotations})
    raw_annotators = attestation.get("annotators")
    if not isinstance(raw_annotators, list) or not raw_annotators:
        errors.append("annotator attestation annotators must be a non-empty list")
        return tuple(errors)
    seen: set[str] = set()
    for index, item in enumerate(raw_annotators, start=1):
        if not isinstance(item, dict):
            errors.append(f"annotators[{index}] must be an object")
            continue
        annotator_id = str(item.get("annotator_id", "")).strip()
        if not annotator_id:
            errors.append(f"annotators[{index}].annotator_id is required")
        elif annotator_id in seen:
            errors.append(f"annotators[{index}].annotator_id is duplicated")
        else:
            seen.add(annotator_id)
        if not item.get("signed_at"):
            errors.append(f"annotators[{index}].signed_at is required")
        if item.get("independent_annotation") is not True:
            errors.append(f"annotators[{index}].independent_annotation must be true")
        if item.get("conflict_of_interest") is not False:
            errors.append(f"annotators[{index}].conflict_of_interest must be false")
    if sorted(seen) != expected_annotators:
        errors.append("annotator attestation annotator ids must match completed annotations")
    protocol = attestation.get("protocol")
    if not isinstance(protocol, dict):
        errors.append("annotator attestation protocol must be an object")
    else:
        if not protocol.get("annotation_guideline"):
            errors.append("annotator attestation protocol.annotation_guideline is required")
        if not protocol.get("adjudication_policy"):
            errors.append("annotator attestation protocol.adjudication_policy is required")
    return tuple(errors)


def _verify_adjudication_artifact(
    disagreement_items: list[dict[str, Any]],
    adjudication_path: Path | None,
    *,
    require_adjudication: bool,
    errors: list[str],
) -> dict[str, Any]:
    expected_items = {
        (str(item["case_id"]), str(item["query_id"])): item
        for item in disagreement_items
    }
    required = bool(expected_items)
    summary = {
        "schema_version": HUMAN_AUDIT_ADJUDICATION_DECISION_SCHEMA_VERSION,
        "adjudication_file": str(adjudication_path) if adjudication_path is not None else None,
        "required": required,
        "provided": adjudication_path is not None,
        "status": "not_required" if not required else "missing",
        "num_expected_items": len(expected_items),
        "num_completed_items": 0,
        "num_resolved_fields": 0,
        "adjudicator_ids": [],
    }
    if not required and adjudication_path is None:
        return summary
    if adjudication_path is None:
        if require_adjudication:
            errors.append("human-audit disagreements require audit_adjudication_file")
        return summary
    if not adjudication_path.exists():
        errors.append(f"audit adjudication file does not exist: {adjudication_path}")
        summary["status"] = "invalid"
        return summary

    seen: set[ItemKey] = set()
    adjudicator_ids: set[str] = set()
    resolved_fields = 0
    completed_items = 0
    extra_items: list[str] = []
    with adjudication_path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"audit adjudication {adjudication_path}:{line_number} is not valid JSONL: {exc}")
                continue
            if not isinstance(record, dict):
                errors.append(f"audit adjudication {adjudication_path}:{line_number} must be a JSON object")
                continue
            key = _record_item_key(record)
            if key is None:
                errors.append(f"audit adjudication {adjudication_path}:{line_number} is missing case_id or query_id")
                continue
            if key in seen:
                errors.append(f"audit adjudication duplicates item {_format_item(key)}")
                continue
            seen.add(key)
            expected = expected_items.get(key)
            if expected is None:
                extra_items.append(_format_item(key))
                continue
            adjudicator_id = str(record.get("adjudicator_id", "")).strip()
            if not adjudicator_id:
                errors.append(f"audit adjudication {_format_item(key)} is missing adjudicator_id")
            else:
                adjudicator_ids.add(adjudicator_id)
            if record.get("adjudication_status") != "completed":
                errors.append(f"audit adjudication {_format_item(key)} must have adjudication_status=completed")
            raw_checks = record.get("adjudication_checks")
            if not isinstance(raw_checks, dict):
                errors.append(f"audit adjudication {_format_item(key)} must include adjudication_checks")
                continue
            target_fields = sorted(expected["adjudication_checks"])
            invalid = False
            for field in target_fields:
                value = raw_checks.get(field)
                if value not in (True, False):
                    errors.append(
                        f"audit adjudication {_format_item(key)} must resolve field {field} with a boolean decision"
                    )
                    invalid = True
            unexpected_non_null = [
                field for field, value in raw_checks.items() if field not in target_fields and value is not None
            ]
            if unexpected_non_null:
                errors.append(
                    f"audit adjudication {_format_item(key)} has non-null non-target fields: {unexpected_non_null}"
                )
                invalid = True
            if not invalid:
                resolved_fields += len(target_fields)
                completed_items += 1

    missing = sorted(set(expected_items) - seen)
    if missing:
        errors.append(f"audit adjudication missing disagreement items: {_format_items(missing[:10])}")
    if extra_items:
        errors.append(f"audit adjudication contains non-disagreement items: {extra_items[:10]}")

    summary["status"] = "completed" if not missing and not extra_items else "invalid"
    if errors and required and adjudication_path is not None:
        summary["status"] = "invalid"
    summary["num_completed_items"] = completed_items
    summary["num_resolved_fields"] = resolved_fields
    summary["adjudicator_ids"] = sorted(adjudicator_ids)
    return summary


def _check_attestation_path(attestation: dict[str, Any], key: str, expected: Path, base_dir: Path) -> tuple[str, ...]:
    raw = attestation.get(key)
    if not raw:
        return (f"annotator attestation {key} is required",)
    actual = _resolve_path(str(raw), base_dir)
    try:
        if actual.resolve() != expected.resolve():
            return (f"annotator attestation {key} does not match verified artifact",)
    except FileNotFoundError:
        return (f"annotator attestation {key} does not exist",)
    return ()


def _verify_task_manifest_template_files(
    task_manifest_path: Path,
    template_paths: tuple[Path, ...],
) -> tuple[str, ...]:
    try:
        manifest = read_json(task_manifest_path)
    except Exception as exc:  # noqa: BLE001 - surface task-manifest parsing errors
        return (f"could not load task manifest for template verification: {exc}",)
    raw_template_files = manifest.get("template_files")
    if not isinstance(raw_template_files, list) or not raw_template_files:
        return ("human-audit task manifest must include non-empty template_files",)
    expected = sorted(path.resolve() for path in template_paths)
    actual = sorted(_resolve_path(str(path), task_manifest_path.parent).resolve() for path in raw_template_files)
    if actual != expected:
        return ("task_manifest.template_files do not match verified template_files",)
    return ()


def _verification_report(
    template_paths: tuple[Path, ...],
    annotations_path: Path | None,
    agreement: dict[str, Any] | None,
    errors: list[str],
    *,
    expected_items: dict[ItemKey, dict[str, Any]] | None = None,
    annotations: tuple[AuditAnnotation, ...] = (),
    task_manifest_path: Path | None = None,
    annotator_attestation_path: Path | None = None,
    semantic_checks: dict[str, Any] | None = None,
    disagreement_summary: dict[str, Any] | None = None,
    adjudication_path: Path | None = None,
    adjudication_summary: dict[str, Any] | None = None,
    benchmark_id: str | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    annotators = sorted({annotation.annotator_id for annotation in annotations})
    template_digest_map = {
        str(path.resolve()): _file_sha256(path)
        for path in template_paths
    }
    return {
        "schema_version": HUMAN_AUDIT_VERIFICATION_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "ok": not errors,
        "errors": errors,
        "template_files": [str(path) for path in template_paths],
        "annotations_file": str(annotations_path) if annotations_path is not None else None,
        "task_manifest_file": str(task_manifest_path) if task_manifest_path is not None else None,
        "annotator_attestation_file": str(annotator_attestation_path) if annotator_attestation_path is not None else None,
        "adjudication_file": str(adjudication_path) if adjudication_path is not None else None,
        "file_digests": {
            "manifest_file": _file_sha256(manifest_path) if manifest_path is not None else None,
            "template_files": template_digest_map,
            "annotations_file": _file_sha256(annotations_path) if annotations_path is not None else None,
            "task_manifest_file": _file_sha256(task_manifest_path) if task_manifest_path is not None else None,
            "annotator_attestation_file": _file_sha256(annotator_attestation_path) if annotator_attestation_path is not None else None,
            "adjudication_file": _file_sha256(adjudication_path) if adjudication_path is not None else None,
        },
        "num_template_items": len(expected_items or {}),
        "num_annotations": len(annotations),
        "num_annotators": len(annotators),
        "annotator_ids": annotators,
        "agreement_metrics": agreement,
        "semantic_checks": semantic_checks or {},
        "disagreement_summary": disagreement_summary or {},
        "adjudication_summary": adjudication_summary or {},
        "adjudication_recommended": bool((disagreement_summary or {}).get("num_disagreement_items", 0)),
    }


def _record_item_key(record: dict[str, Any]) -> ItemKey | None:
    case_id = record.get("case_id")
    query_id = record.get("query_id")
    if case_id is None or query_id is None:
        return None
    return (str(case_id), str(query_id))


def _format_item(item: ItemKey) -> str:
    return f"{item[0]}/{item[1]}"


def _format_items(items: Iterable[ItemKey]) -> list[str]:
    return [_format_item(item) for item in items]


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    if path.exists():
        return path
    return candidate


def _relative_or_absolute(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def _bundle_contract_project_root(bundle_root: Path) -> Path | None:
    resolved_bundle_root = bundle_root.resolve()
    for candidate in (resolved_bundle_root, *resolved_bundle_root.parents):
        if candidate.name != "current":
            continue
        if candidate.parent.name != "human_audit_bundle":
            continue
        if candidate.parent.parent.name != "reports":
            continue
        return candidate.parent.parent.parent
    return None


def _bundle_root_ref(bundle_root: Path) -> str | None:
    project_root = _bundle_contract_project_root(bundle_root)
    if project_root is None:
        return None
    try:
        return Path(os.path.relpath(project_root.resolve(), bundle_root.resolve())).as_posix()
    except ValueError:
        return str(project_root.resolve())


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
