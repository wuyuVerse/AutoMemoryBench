"""Unified evidence-readiness report for AutoMemoryBench completion work."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from amb.benchmark.analysis.external_protocol import (
    _external_operator_commands,
    _external_recommended_next_refs,
    build_external_cohort_expansion_plan,
    build_external_cohort_expansion_validation,
    build_external_evidence_gap_report,
    build_external_evidence_plan,
    summarize_external_cohort_rejected_returns,
    validate_external_evidence_set,
)
from amb.benchmark.integrations.config_validation import validate_integration_config_files
from amb.benchmark.quality.annotation import summarize_human_audit_progress
from amb.benchmark.quality.completion_audit import _artifact_root_ref
from amb.benchmark.quality.completion_audit import build_completion_audit
from amb.benchmark.quality.completion_audit import build_git_hygiene_report
from amb.benchmark.quality.human_audit import HUMAN_AUDIT_BUNDLE_SANDBOX_ADJUDICATION_POLICY
from amb.benchmark.quality.human_audit import HUMAN_AUDIT_BUNDLE_SANDBOX_ANNOTATION_GUIDELINE
from amb.benchmark.quality.human_audit import HUMAN_AUDIT_BUNDLE_SANDBOX_SIGNED_AT
from amb.benchmark.quality.human_audit import HUMAN_AUDIT_CLI_PREFIX
from amb.benchmark.quality.human_audit import HUMAN_AUDIT_OPERATOR_WATCH_INTERVAL_S
from amb.benchmark.quality.human_audit import HUMAN_AUDIT_WATCH_STOP_EXIT_CODES
from amb.benchmark.quality.real_system import (
    ordered_real_system_analysis_candidates,
    ordered_real_system_matrix_summary_candidates,
    summarize_real_system_analysis,
    validate_real_system_matrix_summary,
)
from amb.benchmark.schemas.io import read_json, write_json

EVIDENCE_READINESS_SCHEMA_VERSION = "amst-evidence-readiness-v1"
_DEFAULT_REAL_SYSTEM_CONFIG_PATHS = (
    "configs/real_system/mem0_siliconflow_real.json",
    "configs/real_system/letta_memgpt_local_real.json",
    "configs/real_system/langmem_real.json",
    "configs/real_system/zep_graphiti_graphiti_core_local.json",
)
_DEFAULT_REAL_SYSTEM_CONFIG_VALIDATION_OUTPUT = "reports/real_system_runs/canonical_public_dev_refresh_config_validation.json"


def build_evidence_readiness_report(
    root: str | Path = ".",
    *,
    external_output_dir: str | Path = "reports/external",
    human_task_manifest: str | Path | None = None,
    human_annotations: str | Path | None = None,
    real_system_summary: str | Path | None = None,
    real_system_sample_validation: str | Path | None = None,
    real_system_sample_summary: str | Path | None = None,
    integration_configs: Iterable[str | Path] | None = None,
    integration_config_validation_report: str | Path | None = None,
) -> dict[str, Any]:
    """Build a non-mutating readiness report across all completion evidence gates."""

    project = Path(root)
    completion = build_completion_audit(project)
    completion_checks = [asdict(check) for check in completion.checks]
    blockers = [check for check in completion_checks if check["status"] != "passed"]
    external_plan = build_external_evidence_plan(project, output_dir=external_output_dir)
    external_validation = _external_validation_section(project, external_plan)
    external_gap = _external_gap_section(project, external_plan)
    external_expansion = _external_expansion_section(project, external_plan)
    human_progress = _human_progress_section(project, human_task_manifest, human_annotations)
    real_system = _real_system_section(project, real_system_summary)
    real_system_sample = _real_system_sample_section(
        project,
        validation_path=real_system_sample_validation,
        summary_path=real_system_sample_summary,
    )
    git_hygiene = build_git_hygiene_report(project)
    config_validation = _config_validation_section(
        project,
        integration_configs,
        validation_report_path=integration_config_validation_report,
    )
    next_actions = _next_actions(
        project,
        blockers,
        human_progress,
        real_system,
        real_system_sample,
        git_hygiene,
        config_validation,
        external_plan,
        external_validation,
        external_gap,
        external_expansion,
    )

    status = "ready" if completion.status == "complete" else "blocked"
    return {
        "schema_version": EVIDENCE_READINESS_SCHEMA_VERSION,
        "root": str(project.resolve()),
        "status": status,
        "completion_audit": {
            "status": completion.status,
            "summary": completion.summary,
            "blockers": blockers,
        },
        "human_audit_progress": human_progress,
        "external_benchmark_evidence": external_plan,
        "external_benchmark_validation": external_validation,
        "external_benchmark_gap": external_gap,
        "external_benchmark_expansion": external_expansion,
        "real_system_evidence": real_system,
        "real_system_sample_evidence": real_system_sample,
        "git_hygiene_evidence": git_hygiene,
        "integration_config_validation": config_validation,
        "next_actions": next_actions,
    }


def write_evidence_readiness_report(
    output: str | Path,
    *,
    root: str | Path = ".",
    external_output_dir: str | Path = "reports/external",
    human_task_manifest: str | Path | None = None,
    human_annotations: str | Path | None = None,
    real_system_summary: str | Path | None = None,
    real_system_sample_validation: str | Path | None = None,
    real_system_sample_summary: str | Path | None = None,
    integration_configs: Iterable[str | Path] | None = None,
    integration_config_validation_report: str | Path | None = None,
) -> dict[str, Any]:
    project = Path(root)
    report = build_evidence_readiness_report(
        root,
        external_output_dir=external_output_dir,
        human_task_manifest=human_task_manifest,
        human_annotations=human_annotations,
        real_system_summary=real_system_summary,
        real_system_sample_validation=real_system_sample_validation,
        real_system_sample_summary=real_system_sample_summary,
        integration_configs=integration_configs,
        integration_config_validation_report=integration_config_validation_report,
    )
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = project / output_path
    root_ref = _artifact_root_ref(output_path.parent, project)
    report["root"] = root_ref
    git_hygiene = report.get("git_hygiene_evidence")
    if isinstance(git_hygiene, dict):
        git_hygiene["root"] = root_ref
    real_system = report.get("real_system_evidence")
    if isinstance(real_system, dict):
        _normalize_real_system_section_paths(real_system, project)
    real_system_sample = report.get("real_system_sample_evidence")
    if isinstance(real_system_sample, dict):
        _normalize_real_system_section_paths(real_system_sample, project)
    write_json(output, report)
    return report


def _normalize_real_system_section_paths(section: dict[str, Any], project: Path) -> None:
    for key in ("matrix_summary", "validation_artifact"):
        raw_value = section.get(key)
        if isinstance(raw_value, str) and raw_value.strip():
            section[key] = _project_relative_or_absolute(project, _resolve_project_relative_path(project, raw_value))
    systems = section.get("systems")
    if not isinstance(systems, list):
        return
    for row in systems:
        if not isinstance(row, dict):
            continue
        raw_report_path = row.get("report_path")
        if isinstance(raw_report_path, str) and raw_report_path.strip():
            row["report_path"] = _project_relative_or_absolute(
                project,
                _resolve_project_relative_path(project, raw_report_path),
            )


def _human_progress_section(
    project: Path,
    task_manifest: str | Path | None,
    annotations: str | Path | None,
) -> dict[str, Any]:
    resolved_task_manifest = (
        _resolve_optional_path(project, task_manifest)
        if task_manifest is not None
        else _default_human_task_manifest(project)
    )
    resolved_annotations = (
        _resolve_optional_path(project, annotations)
        if annotations is not None
        else _default_human_annotations(project, resolved_task_manifest)
    )
    if resolved_task_manifest is None:
        return {
            "status": "not_provided",
            "ready_for_finalize": False,
            "notes": "Provide --human-task-manifest and --human-annotations to check human-audit collection progress.",
        }
    progress = summarize_human_audit_progress(resolved_task_manifest, annotations_path=resolved_annotations)
    progress["auto_discovered"] = task_manifest is None
    progress["annotations_auto_discovered"] = annotations is None and resolved_annotations is not None
    bundle_manifest = _bundle_manifest_for_task_manifest(resolved_task_manifest)
    if bundle_manifest is not None:
        progress["bundle_manifest_file"] = str(bundle_manifest)
        progress["bundle_dir"] = str(bundle_manifest.parent)
        bundle_dir = bundle_manifest.parent
        try:
            bundle_manifest_payload = read_json(bundle_manifest)
        except Exception:  # noqa: BLE001 - readiness should degrade rather than fail hard on bundle parse errors
            bundle_manifest_payload = None
        if isinstance(bundle_manifest_payload, dict):
            if isinstance(bundle_manifest_payload.get("handoff_manifest_file"), str):
                progress["handoff_manifest_file"] = str(bundle_manifest_payload["handoff_manifest_file"])
            if isinstance(bundle_manifest_payload.get("return_inbox"), dict):
                progress["return_inbox"] = {
                    str(key): str(value)
                    for key, value in sorted(bundle_manifest_payload["return_inbox"].items())
                }
                progress["return_inbox_paths"] = {
                    str(key): str(_resolve_bundle_relative_path(bundle_dir, value))
                    for key, value in sorted(bundle_manifest_payload["return_inbox"].items())
                    if isinstance(value, str) and value.strip()
                }
            elif (bundle_dir / "returns" / "annotators").exists() and (bundle_dir / "returns" / "adjudication").exists():
                progress["return_inbox"] = {
                    "annotator_inbox": "returns/annotators",
                    "adjudication_inbox": "returns/adjudication",
                }
                progress["return_inbox_paths"] = {
                    "annotator_inbox": str(bundle_dir / "returns" / "annotators"),
                    "adjudication_inbox": str(bundle_dir / "returns" / "adjudication"),
                }
            if isinstance(bundle_manifest_payload.get("return_archive"), dict):
                progress["return_archive"] = {
                    str(key): str(value)
                    for key, value in sorted(bundle_manifest_payload["return_archive"].items())
                }
                progress["return_archive_paths"] = {
                    str(key): str(_resolve_bundle_relative_path(bundle_dir, value))
                    for key, value in sorted(bundle_manifest_payload["return_archive"].items())
                    if isinstance(value, str) and value.strip()
                }
            elif (bundle_dir / "returns" / "processed" / "annotators").exists() and (
                bundle_dir / "returns" / "processed" / "adjudication"
            ).exists():
                progress["return_archive"] = {
                    "annotator_archive": "returns/processed/annotators",
                    "adjudication_archive": "returns/processed/adjudication",
                }
                progress["return_archive_paths"] = {
                    "annotator_archive": str(bundle_dir / "returns" / "processed" / "annotators"),
                    "adjudication_archive": str(bundle_dir / "returns" / "processed" / "adjudication"),
                }
            if isinstance(bundle_manifest_payload.get("return_reject_archive"), dict):
                progress["return_reject_archive"] = {
                    str(key): str(value)
                    for key, value in sorted(bundle_manifest_payload["return_reject_archive"].items())
                }
                progress["return_reject_archive_paths"] = {
                    str(key): str(_resolve_bundle_relative_path(bundle_dir, value))
                    for key, value in sorted(bundle_manifest_payload["return_reject_archive"].items())
                    if isinstance(value, str) and value.strip()
                }
            elif (bundle_dir / "returns" / "rejected" / "annotators").exists() and (
                bundle_dir / "returns" / "rejected" / "adjudication"
            ).exists():
                progress["return_reject_archive"] = {
                    "annotator_archive": "returns/rejected/annotators",
                    "adjudication_archive": "returns/rejected/adjudication",
                }
                progress["return_reject_archive_paths"] = {
                    "annotator_archive": str(bundle_dir / "returns" / "rejected" / "annotators"),
                    "adjudication_archive": str(bundle_dir / "returns" / "rejected" / "adjudication"),
                }
            if isinstance(bundle_manifest_payload.get("return_inbox_watch_file"), str):
                progress["return_inbox_watch_file"] = str(bundle_manifest_payload["return_inbox_watch_file"])
                progress["return_inbox_watch_output_file"] = str(
                    _resolve_bundle_relative_path(bundle_dir, bundle_manifest_payload["return_inbox_watch_file"])
                )
            elif (bundle_dir / "return_inbox_watch.json").exists():
                progress["return_inbox_watch_file"] = "return_inbox_watch.json"
                progress["return_inbox_watch_output_file"] = str(bundle_dir / "return_inbox_watch.json")
            if isinstance(bundle_manifest_payload.get("return_inbox_state_file"), str):
                progress["return_inbox_state_file"] = str(bundle_manifest_payload["return_inbox_state_file"])
                progress["return_inbox_state_path"] = str(
                    _resolve_bundle_relative_path(bundle_dir, bundle_manifest_payload["return_inbox_state_file"])
                )
            elif (bundle_dir / "return_inbox_state.json").exists():
                progress["return_inbox_state_file"] = "return_inbox_state.json"
                progress["return_inbox_state_path"] = str(bundle_dir / "return_inbox_state.json")
            if isinstance(bundle_manifest_payload.get("watch_stop_exit_codes"), dict):
                progress["watch_stop_exit_codes"] = {
                    str(key): int(value)
                    for key, value in sorted(bundle_manifest_payload["watch_stop_exit_codes"].items())
                    if isinstance(value, int)
                }
            elif "return_inbox_watch_file" in progress or "operator_scripts" in progress:
                progress["watch_stop_exit_codes"] = dict(HUMAN_AUDIT_WATCH_STOP_EXIT_CODES)
            if isinstance(bundle_manifest_payload.get("watch_stop_actions"), dict):
                progress["watch_stop_actions"] = {
                    str(key): dict(value)
                    for key, value in sorted(bundle_manifest_payload["watch_stop_actions"].items())
                    if isinstance(value, dict)
                }
            rejected_return_summary = _bundle_rejected_return_summary(bundle_dir, bundle_manifest_payload)
            if rejected_return_summary is not None:
                progress["rejected_return_summary"] = rejected_return_summary
            if isinstance(bundle_manifest_payload.get("rejected_returns_report_file"), str):
                progress["rejected_returns_report_file"] = str(bundle_manifest_payload["rejected_returns_report_file"])
                progress["rejected_returns_report_path"] = str(
                    _resolve_bundle_relative_path(bundle_dir, bundle_manifest_payload["rejected_returns_report_file"])
                )
            elif (bundle_dir / "rejected_returns_report.json").exists():
                progress["rejected_returns_report_file"] = "rejected_returns_report.json"
                progress["rejected_returns_report_path"] = str(bundle_dir / "rejected_returns_report.json")
            if isinstance(bundle_manifest_payload.get("return_inbox_sync_report_file"), str):
                progress["return_inbox_sync_report_file"] = str(bundle_manifest_payload["return_inbox_sync_report_file"])
                progress["return_inbox_sync_report_path"] = str(
                    _resolve_bundle_relative_path(bundle_dir, bundle_manifest_payload["return_inbox_sync_report_file"])
                )
            elif bundle_manifest is not None or (bundle_dir / "return_inbox_sync.json").exists():
                progress["return_inbox_sync_report_file"] = "return_inbox_sync.json"
                progress["return_inbox_sync_report_path"] = str(bundle_dir / "return_inbox_sync.json")
            sync_payload = _read_optional_json(progress.get("return_inbox_sync_report_path"))
            watch_payload = _read_optional_json(progress.get("return_inbox_watch_output_file"))
            _annotate_sidecar_runtime_fields(progress, "return_inbox_sync", sync_payload)
            _annotate_sidecar_runtime_fields(progress, "return_inbox_watch", watch_payload)
            if isinstance(bundle_manifest_payload.get("operator_scripts"), dict):
                progress["operator_scripts"] = {
                    str(key): str(value)
                    for key, value in sorted(bundle_manifest_payload["operator_scripts"].items())
                }
                progress["operator_script_files"] = {
                    str(key): str(_resolve_bundle_relative_path(bundle_dir, value))
                    for key, value in sorted(bundle_manifest_payload["operator_scripts"].items())
                    if isinstance(value, str) and value.strip()
                }
            if isinstance(bundle_manifest_payload.get("operator_commands"), dict):
                progress["operator_commands"] = {
                    str(key): str(value)
                    for key, value in sorted(bundle_manifest_payload["operator_commands"].items())
                    if isinstance(value, str) and value.strip()
                }
                default_operator_commands = _default_operator_commands(
                    bundle_dir,
                    progress.get("task_manifest_file") if isinstance(progress.get("task_manifest_file"), str) else None,
                )
                sync_command = progress["operator_commands"].get("sync_return_inbox")
                default_sync_command = default_operator_commands.get("sync_return_inbox")
                if (
                    isinstance(sync_command, str)
                    and isinstance(default_sync_command, str)
                    and "--output" not in sync_command
                ):
                    progress["operator_commands"]["sync_return_inbox"] = default_sync_command
            elif "operator_scripts" in progress or "return_inbox_watch_file" in progress:
                progress["operator_commands"] = _default_operator_commands(
                    bundle_dir,
                    progress.get("task_manifest_file") if isinstance(progress.get("task_manifest_file"), str) else None,
                )
            if isinstance(bundle_manifest_payload.get("pending_return_packets"), list):
                progress["pending_return_packets"] = [
                    str(value) for value in bundle_manifest_payload["pending_return_packets"] if isinstance(value, str) and value.strip()
                ]
            if isinstance(bundle_manifest_payload.get("pending_return_packet_paths"), list):
                progress["pending_return_packet_paths"] = [
                    str(value) for value in bundle_manifest_payload["pending_return_packet_paths"] if isinstance(value, str) and value.strip()
                ]
            if isinstance(bundle_manifest_payload.get("recommended_next_command_id"), str):
                progress["recommended_next_command_id"] = str(bundle_manifest_payload["recommended_next_command_id"])
            if isinstance(bundle_manifest_payload.get("recommended_next_command"), str):
                progress["recommended_next_command"] = str(bundle_manifest_payload["recommended_next_command"])
            if isinstance(bundle_manifest_payload.get("recommended_next_script_id"), str):
                progress["recommended_next_script_id"] = str(bundle_manifest_payload["recommended_next_script_id"])
            if isinstance(bundle_manifest_payload.get("recommended_next_script"), str):
                progress["recommended_next_script"] = str(bundle_manifest_payload["recommended_next_script"])
            if isinstance(bundle_manifest_payload.get("recommended_next_script_file"), str):
                progress["recommended_next_script_file"] = str(bundle_manifest_payload["recommended_next_script_file"])
            pending_packets: list[str] = []
            pending_packet_paths: list[str] = []
            for raw_path in progress.get("return_inbox_paths", {}).values():
                inbox_path = Path(raw_path)
                if not inbox_path.exists():
                    continue
                for packet_path in sorted(inbox_path.glob("*.zip")):
                    pending_packets.append(
                        packet_path.relative_to(bundle_dir).as_posix()
                        if packet_path.is_relative_to(bundle_dir)
                        else str(packet_path)
                    )
                    pending_packet_paths.append(str(packet_path))
            progress["pending_return_packets"] = pending_packets
            progress["pending_return_packet_paths"] = pending_packet_paths
            recommended_commands = progress.get("recommended_next_commands")
            recommended_command = None
            if isinstance(recommended_commands, list):
                for value in recommended_commands:
                    if isinstance(value, str) and value.strip():
                        recommended_command = str(value)
                        break
            if pending_packets and isinstance(recommended_command, str):
                if "watch-human-audit-return-inbox" in recommended_command or "human-audit-progress" in recommended_command:
                    sync_command = progress.get("operator_commands", {}).get("sync_return_inbox")
                    if isinstance(sync_command, str) and sync_command.strip():
                        recommended_command = sync_command
            recommended_command_id = None
            recommended_script_id = None
            recommended_script = None
            recommended_script_file = None
            if isinstance(recommended_command, str):
                if "watch-human-audit-return-inbox" in recommended_command:
                    recommended_command_id = "watch_return_inbox"
                elif "reconcile-human-audit-evidence-bundle" in recommended_command:
                    recommended_command_id = "reconcile_when_ready"
                elif "human-audit-progress" in recommended_command:
                    recommended_command_id = "progress"
                elif "sync-human-audit-return-inbox" in recommended_command:
                    recommended_command_id = "sync_return_inbox"
                elif "summarize-human-audit-rejected-returns" in recommended_command:
                    recommended_command_id = "review_rejected_returns"
                elif "verify-human-audit-evidence-bundle" in recommended_command:
                    recommended_command_id = "verify_bundle"
            if isinstance(recommended_command_id, str):
                recommended_script_id = recommended_command_id
                script_value = progress.get("operator_scripts", {}).get(recommended_command_id)
                if isinstance(script_value, str) and script_value.strip():
                    recommended_script = script_value
                script_file_value = progress.get("operator_script_files", {}).get(recommended_command_id)
                if isinstance(script_file_value, str) and script_file_value.strip():
                    recommended_script_file = script_file_value
            progress["recommended_next_command_id"] = recommended_command_id
            progress["recommended_next_command"] = recommended_command
            progress["recommended_next_script_id"] = recommended_script_id
            progress["recommended_next_script"] = recommended_script
            progress["recommended_next_script_file"] = recommended_script_file
            if "operator_scripts" in progress or "return_inbox_watch_file" in progress:
                progress["watch_stop_actions"] = _normalize_watch_stop_actions(
                    bundle_dir,
                    progress.get("operator_scripts") if isinstance(progress.get("operator_scripts"), dict) else {},
                    progress.get("watch_stop_actions"),
                )
            if isinstance(bundle_manifest_payload.get("annotation_sheet_files"), dict):
                progress["annotation_sheet_files"] = {
                    str(key): str(value)
                    for key, value in sorted(bundle_manifest_payload["annotation_sheet_files"].items())
                }
            if isinstance(bundle_manifest_payload.get("sandbox_sample"), dict):
                progress["sandbox_sample"] = dict(bundle_manifest_payload["sandbox_sample"])
            if isinstance(bundle_manifest_payload.get("annotator_packets"), dict):
                progress["annotator_packets"] = {
                    str(key): value
                    for key, value in sorted(bundle_manifest_payload["annotator_packets"].items())
                    if isinstance(value, dict)
                }
            if isinstance(bundle_manifest_payload.get("adjudication_packet"), dict):
                progress["adjudication_packet"] = dict(bundle_manifest_payload["adjudication_packet"])
            if isinstance(bundle_manifest_payload.get("recommended_next_commands"), list):
                progress["recommended_next_commands"] = [
                    str(value) for value in bundle_manifest_payload["recommended_next_commands"] if str(value).strip()
                ]
                recommended_commands = progress.get("recommended_next_commands")
                recommended_command = None
                if isinstance(recommended_commands, list):
                    for value in recommended_commands:
                        if isinstance(value, str) and value.strip():
                            recommended_command = str(value)
                            break
                if progress.get("pending_return_packets") and isinstance(recommended_command, str):
                    if "watch-human-audit-return-inbox" in recommended_command or "human-audit-progress" in recommended_command:
                        sync_command = progress.get("operator_commands", {}).get("sync_return_inbox")
                        if isinstance(sync_command, str) and sync_command.strip():
                            recommended_command = sync_command
                recommended_command_id = None
                recommended_script_id = None
                recommended_script = None
                recommended_script_file = None
                if isinstance(recommended_command, str):
                    if "watch-human-audit-return-inbox" in recommended_command:
                        recommended_command_id = "watch_return_inbox"
                    elif "reconcile-human-audit-evidence-bundle" in recommended_command:
                        recommended_command_id = "reconcile_when_ready"
                    elif "human-audit-progress" in recommended_command:
                        recommended_command_id = "progress"
                    elif "sync-human-audit-return-inbox" in recommended_command:
                        recommended_command_id = "sync_return_inbox"
                    elif "summarize-human-audit-rejected-returns" in recommended_command:
                        recommended_command_id = "review_rejected_returns"
                    elif "verify-human-audit-evidence-bundle" in recommended_command:
                        recommended_command_id = "verify_bundle"
                if isinstance(recommended_command_id, str):
                    recommended_script_id = recommended_command_id
                    script_value = progress.get("operator_scripts", {}).get(recommended_command_id)
                    if isinstance(script_value, str) and script_value.strip():
                        recommended_script = script_value
                    script_file_value = progress.get("operator_script_files", {}).get(recommended_command_id)
                    if isinstance(script_file_value, str) and script_file_value.strip():
                        recommended_script_file = script_file_value
                progress["recommended_next_command_id"] = recommended_command_id
                progress["recommended_next_command"] = recommended_command
                progress["recommended_next_script_id"] = recommended_script_id
                progress["recommended_next_script"] = recommended_script
                progress["recommended_next_script_file"] = recommended_script_file
            _fill_recommended_next_from_sidecars(progress, watch_payload, sync_payload)
    return progress


def _real_system_section(project: Path, summary_path: str | Path | None) -> dict[str, Any]:
    resolved = _resolve_optional_path(project, summary_path) if summary_path is not None else _default_real_system_summary(project)
    if resolved is None:
        return {
            "schema_version": "amst-real-system-evidence-v1",
            "scope": "canonical_completion_gate",
            "completion_gate": True,
            "expected_benchmark_id": "amst-main-v1-public_dev",
            "expected_release_split": "public_dev",
            "status": "missing",
            "matrix_summary": None,
            "required_paths": [
                "reports/real_system_runs/matrix_summary.json",
                "reports/examples/amst_main_v1_real_system_matrix_summary.json",
            ],
            "errors": ["real-system matrix summary was not found"],
        }
    report = validate_real_system_matrix_summary(
        resolved,
        expected_benchmark_id="amst-main-v1-public_dev",
        expected_release_split="public_dev",
    )
    report["scope"] = "canonical_completion_gate"
    report["completion_gate"] = True
    analysis_candidates = ordered_real_system_analysis_candidates(project, resolved)
    analysis_path = next((path for path in analysis_candidates if path.exists()), None)
    if analysis_path is None:
        if report.get("status") == "passed":
            report["status"] = "incomplete"
        report["analysis_artifact"] = (
            _project_relative_or_absolute(project, analysis_candidates[0]) if analysis_candidates else None
        )
        report["analysis_summary"] = None
        report["errors"] = list(report.get("errors", [])) + ["real-system analysis artifact was not found"]
        return report
    analysis = _read_optional_json(analysis_path)
    if not isinstance(analysis, dict):
        if report.get("status") == "passed":
            report["status"] = "invalid"
        report["analysis_artifact"] = _project_relative_or_absolute(project, analysis_path)
        report["analysis_summary"] = None
        report["errors"] = list(report.get("errors", [])) + [
            f"{analysis_path}: real-system analysis artifact must be a JSON object"
        ]
        return report
    analysis_summary = summarize_real_system_analysis(analysis)
    report["analysis_artifact"] = _project_relative_or_absolute(project, analysis_path)
    report["analysis_summary"] = analysis_summary
    if not (
        analysis_summary.get("bootstrap_samples_sufficient") is True
        and analysis_summary.get("report_bootstrap_cis_present") is True
        and analysis_summary.get("pairwise_stats_complete") is True
        and analysis_summary.get("quality_cost_frontier_complete") is True
        and analysis_summary.get("weight_sensitivity_profiles_complete") is True
    ):
        if report.get("status") == "passed":
            report["status"] = "incomplete"
        report["errors"] = list(report.get("errors", [])) + ["real-system analysis artifact is incomplete"]
    return report


def _real_system_sample_section(
    project: Path,
    *,
    validation_path: str | Path | None,
    summary_path: str | Path | None,
) -> dict[str, Any]:
    resolved_validation = (
        _resolve_optional_path(project, validation_path)
        if validation_path is not None
        else _default_real_system_sample_validation(project)
    )
    if resolved_validation is not None and resolved_validation.exists():
        try:
            report = read_json(resolved_validation)
        except Exception as exc:
            return {
                "schema_version": "amst-real-system-evidence-v1",
                "scope": "sample_progress_only",
                "completion_gate": False,
                "status": "invalid",
                "matrix_summary": None,
                "validation_artifact": str(resolved_validation),
                "errors": [f"{resolved_validation}: cannot read sample validation artifact: {exc}"],
            }
        if not isinstance(report, dict):
            return {
                "schema_version": "amst-real-system-evidence-v1",
                "scope": "sample_progress_only",
                "completion_gate": False,
                "status": "invalid",
                "matrix_summary": None,
                "validation_artifact": str(resolved_validation),
                "errors": [f"{resolved_validation}: sample validation artifact must be a JSON object"],
            }
        report = dict(report)
        report["scope"] = "sample_progress_only"
        report["completion_gate"] = False
        report["validation_artifact"] = str(resolved_validation)
        report["derived_from_summary"] = False
        return report

    resolved_summary = (
        _resolve_optional_path(project, summary_path)
        if summary_path is not None
        else _default_real_system_sample_summary(project)
    )
    if resolved_summary is None:
        return {
            "schema_version": "amst-real-system-evidence-v1",
            "scope": "sample_progress_only",
            "completion_gate": False,
            "status": "not_provided",
            "matrix_summary": None,
            "errors": [],
            "notes": "Optional sample-level real-system progress can be surfaced with --real-system-sample-validation or --real-system-sample-summary.",
        }
    summary = _read_real_system_summary(resolved_summary)
    expected_benchmark_id = str(summary.get("benchmark_id")) if isinstance(summary.get("benchmark_id"), str) else None
    expected_release_split = str(summary.get("release_split")) if isinstance(summary.get("release_split"), str) else None
    report = validate_real_system_matrix_summary(
        resolved_summary,
        expected_benchmark_id=expected_benchmark_id,
        expected_release_split=expected_release_split,
    )
    report["scope"] = "sample_progress_only"
    report["completion_gate"] = False
    report["derived_from_summary"] = True
    report["notes"] = "This sample-level evidence is informative progress only and does not satisfy the canonical completion gate."
    return report


def _config_validation_section(
    project: Path,
    config_paths: Iterable[str | Path] | None,
    *,
    validation_report_path: str | Path | None = None,
) -> dict[str, Any]:
    configs = tuple(config_paths or _default_real_system_config_paths(project))
    resolved_validation_report = (
        _resolve_optional_path(project, validation_report_path)
        if validation_report_path is not None
        else None
    )
    if resolved_validation_report is not None:
        persisted = _read_optional_json(resolved_validation_report)
        if isinstance(persisted, dict) and persisted:
            report = dict(persisted)
            report["validation_artifact"] = _project_relative_or_absolute(project, resolved_validation_report)
            if configs:
                report["config_files"] = [_project_relative_or_absolute(project, Path(path)) for path in configs]
            else:
                config_entries = report.get("configs")
                if isinstance(config_entries, list):
                    derived_config_files = []
                    for item in config_entries:
                        if not isinstance(item, dict):
                            continue
                        raw_path = item.get("path")
                        if not isinstance(raw_path, str) or not raw_path.strip():
                            continue
                        derived_config_files.append(
                            _project_relative_or_absolute(project, _resolve_project_relative_path(project, raw_path))
                        )
                    if derived_config_files:
                        report["config_files"] = derived_config_files
            return report
    if not configs:
        return {
            "schema_version": "amst-integration-config-validation-v1",
            "status": "not_provided",
            "num_configs": 0,
            "configs": [],
            "errors": [],
            "notes": "Provide --integration-configs to preflight real-system configs before running the matrix.",
        }
    report = validate_integration_config_files(configs)
    report["validation_artifact"] = _DEFAULT_REAL_SYSTEM_CONFIG_VALIDATION_OUTPUT
    report["config_files"] = [_project_relative_or_absolute(project, Path(path)) for path in configs]
    return report


def _external_validation_section(project: Path, external_plan: dict[str, Any]) -> dict[str, Any]:
    correlation_paths = []
    for requirement in external_plan.get("requirements", []):
        if not isinstance(requirement, dict):
            continue
        raw_path = requirement.get("correlation_report")
        if not raw_path:
            continue
        correlation_paths.append(_resolve_project_relative_path(project, raw_path))
    validation = validate_external_evidence_set(
        correlation_paths,
        min_shared_systems=3,
        min_shared_control_systems=1,
        min_shared_real_memory_systems=1,
    )
    validation["validation_artifact"] = "reports/external/evidence_validation.json"
    validation["completion_gate_min_shared_systems"] = 3
    validation["completion_gate_min_shared_control_systems"] = 1
    validation["completion_gate_min_shared_real_memory_systems"] = 1
    return validation


def _external_gap_section(project: Path, external_plan: dict[str, Any]) -> dict[str, Any]:
    correlation_paths = []
    for requirement in external_plan.get("requirements", []):
        if not isinstance(requirement, dict):
            continue
        raw_path = requirement.get("correlation_report")
        if not raw_path:
            continue
        correlation_paths.append(_resolve_project_relative_path(project, raw_path))
    validation_path = _default_real_system_validation(project)
    gap = build_external_evidence_gap_report(
        correlation_paths,
        real_system_validation_path=validation_path,
        min_shared_systems=3,
        min_shared_control_systems=1,
        min_shared_real_memory_systems=1,
    )
    gap["gap_report_file"] = "reports/external/evidence_gap_report.json"
    gap["gap_report_path"] = _project_relative_or_absolute(project, project / "reports/external/evidence_gap_report.json")
    return gap


def _external_expansion_section(project: Path, external_plan: dict[str, Any]) -> dict[str, Any]:
    expansion_path = project / "reports/external/cohort_expansion_plan.json"
    validation_path = project / "reports/external/cohort_expansion_validation.json"
    persisted = _read_optional_json(expansion_path)
    if isinstance(persisted, dict) and persisted:
        _annotate_external_expansion_scripts(persisted, expansion_path.parent, project)
        _annotate_external_expansion_packets(persisted, expansion_path.parent, project)
        _normalize_external_expansion_paths(persisted, project)
        persisted["operator_commands"] = _external_operator_commands(expansion_path.parent, persisted, repo_root=project)
        _annotate_external_expansion_handoff(persisted, expansion_path.parent, project)
        _normalize_external_expansion_paths(persisted, project)
        persisted["expansion_plan_file"] = "reports/external/cohort_expansion_plan.json"
        persisted["expansion_plan_path"] = _project_relative_or_absolute(project, expansion_path)
        persisted.setdefault("readme_file", "README.md")
        persisted.setdefault("readme_path", _project_relative_or_absolute(project, expansion_path.parent / "README.md"))
        validation = _read_optional_json(validation_path)
        if not isinstance(validation, dict):
            validation = build_external_cohort_expansion_validation(
                persisted,
                root=expansion_path.parent,
                require_artifacts=False,
            )
        persisted["validation_file"] = "cohort_expansion_validation.json"
        persisted["validation_path"] = _project_relative_or_absolute(project, validation_path)
        persisted["validation_status"] = validation.get("status")
        persisted["validation_errors"] = list(validation.get("errors", [])) if isinstance(validation.get("errors"), list) else []
        return persisted
    correlation_paths = []
    for requirement in external_plan.get("requirements", []):
        if not isinstance(requirement, dict):
            continue
        raw_path = requirement.get("correlation_report")
        if not raw_path:
            continue
        correlation_paths.append(_resolve_project_relative_path(project, raw_path))
    real_system_validation_path = _default_real_system_validation(project)
    report = build_external_cohort_expansion_plan(
        correlation_paths,
        project_root=project,
        real_system_validation_path=real_system_validation_path,
        min_shared_systems=3,
        min_shared_control_systems=1,
        min_shared_real_memory_systems=1,
    )
    _annotate_external_expansion_scripts(report, expansion_path.parent, project)
    _annotate_external_expansion_packets(report, expansion_path.parent, project)
    _normalize_external_expansion_paths(report, project)
    report["operator_commands"] = _external_operator_commands(expansion_path.parent, report, repo_root=project)
    _annotate_external_expansion_handoff(report, expansion_path.parent, project)
    _normalize_external_expansion_paths(report, project)
    report["expansion_plan_file"] = "reports/external/cohort_expansion_plan.json"
    report["expansion_plan_path"] = _project_relative_or_absolute(project, project / "reports/external/cohort_expansion_plan.json")
    report["readme_file"] = "README.md"
    report["readme_path"] = _project_relative_or_absolute(project, expansion_path.parent / "README.md")
    validation = _read_optional_json(validation_path)
    if not isinstance(validation, dict):
        validation = build_external_cohort_expansion_validation(
            report,
            root=expansion_path.parent,
            require_artifacts=False,
        )
    report["validation_file"] = "cohort_expansion_validation.json"
    report["validation_path"] = _project_relative_or_absolute(project, validation_path)
    report["validation_status"] = validation.get("status")
    report["validation_errors"] = list(validation.get("errors", [])) if isinstance(validation.get("errors"), list) else []
    return report


def _default_real_system_summary(project: Path) -> Path | None:
    candidates = ordered_real_system_matrix_summary_candidates(project)
    return candidates[0] if candidates else None


def _default_real_system_validation(project: Path) -> Path | None:
    candidates = (
        project / "reports/real_system_runs/canonical_public_dev_refresh_current_matrix_validation.json",
        project / "reports/examples/canonical_public_dev_refresh_current_matrix_validation.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    globbed = sorted((project / "reports/real_system_runs").glob("*current_matrix_validation.json"))
    if globbed:
        return globbed[0]
    return None


def _default_real_system_refresh_spec(project: Path) -> Path | None:
    candidate = project / "configs/real_system/canonical_public_dev_refresh.json"
    if candidate.exists():
        return candidate
    return None


def _default_real_system_config_paths(project: Path) -> tuple[Path, ...]:
    return tuple(path for path in (project / rel for rel in _DEFAULT_REAL_SYSTEM_CONFIG_PATHS) if path.exists())


def _default_real_system_release_manifest(project: Path) -> Path | None:
    candidates = (
        project / "data/releases/amst_main_v1_public/manifest.json",
        project / "data/releases/amst_main_v1_strict_public/manifest.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _default_human_release_manifest(project: Path) -> Path | None:
    candidates = (
        project / "data/releases/amst_main_v1_strict_public/manifest.json",
        project / "data/releases/amst_main_v1_public/manifest.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _default_git_hygiene_plan_manifest(project: Path) -> Path | None:
    candidates = (
        project / "reports/git_hygiene/current/manifest.json",
        project / "reports/examples/git_hygiene/current/manifest.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _default_human_task_manifest(project: Path) -> Path | None:
    candidates = (
        project / "reports/human_audit_bundle/current/task_manifest.json",
        project / "reports/human_audit_bundle/main_v1_strict_public/task_manifest.json",
        project / "reports/examples/human_audit_bundle/main_v1_strict_public/task_manifest.json",
        project / "reports/human_audit_bundle/main_v1_public/task_manifest.json",
        project / "reports/examples/human_audit_bundle/main_v1_public/task_manifest.json",
        project / "reports/human_audit_tasks/main_v1_strict_public/task_manifest.json",
        project / "reports/examples/human_audit_tasks/main_v1_strict_public/task_manifest.json",
        project / "reports/human_audit_tasks/main_v1_public/task_manifest.json",
        project / "reports/examples/human_audit_tasks/main_v1_public/task_manifest.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _default_human_annotations(project: Path, task_manifest_path: Path | None) -> Path | None:
    if task_manifest_path is None:
        return None
    candidates = (
        task_manifest_path.parent / "completed_annotations.jsonl",
        task_manifest_path.parent / "annotations.jsonl",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _bundle_manifest_for_task_manifest(task_manifest_path: Path | None) -> Path | None:
    if task_manifest_path is None:
        return None
    candidate = task_manifest_path.parent / "bundle_manifest.json"
    if candidate.exists():
        return candidate
    return None


def _default_real_system_sample_validation(project: Path) -> Path | None:
    candidates = sorted((project / "reports/real_system_runs").glob("*sample*_matrix_validation.json"))
    if candidates:
        return candidates[-1]
    candidates = sorted((project / "reports/examples").glob("*sample*_matrix_validation.json"))
    if candidates:
        return candidates[-1]
    return None


def _default_real_system_sample_summary(project: Path) -> Path | None:
    candidates = sorted((project / "reports/real_system_runs").glob("*sample*_matrix_summary.json"))
    if candidates:
        return candidates[-1]
    candidates = sorted((project / "reports/examples").glob("*sample*_matrix_summary.json"))
    if candidates:
        return candidates[-1]
    return None


def _resolve_optional_path(project: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    project_relative = project / path
    if project_relative.exists():
        return project_relative
    if path.exists():
        return path
    return project_relative


def _normalize_external_expansion_paths(report: dict[str, Any], project: Path) -> None:
    def normalize_value(value: Any) -> Any:
        if not isinstance(value, str) or not value:
            return value
        try:
            path = Path(value)
        except TypeError:
            return value
        if path.is_absolute():
            return _project_relative_or_absolute(project, path)
        return value

    for key in (
        "expansion_plan_path",
        "readme_path",
        "validation_path",
        "handoff_manifest_path",
        "return_inbox_sync_report_path",
        "return_inbox_state_path",
        "return_inbox_watch_output_file",
        "rejected_returns_report_path",
        "recommended_next_script_file",
        "recommended_next_command",
    ):
        if key in report:
            report[key] = normalize_value(report.get(key))

    operator_script_files = report.get("operator_script_files")
    if isinstance(operator_script_files, dict):
        for key, value in list(operator_script_files.items()):
            operator_script_files[key] = normalize_value(value)

    operator_commands = report.get("operator_commands")
    if isinstance(operator_commands, dict):
        for key, value in list(operator_commands.items()):
            operator_commands[key] = normalize_value(value)

    return_reject_archive_paths = report.get("return_reject_archive_paths")
    if isinstance(return_reject_archive_paths, dict):
        for key, value in list(return_reject_archive_paths.items()):
            return_reject_archive_paths[key] = normalize_value(value)

    pending_return_packet_paths = report.get("pending_return_packet_paths")
    if isinstance(pending_return_packet_paths, list):
        report["pending_return_packet_paths"] = [normalize_value(item) for item in pending_return_packet_paths]

    recommended_candidate = report.get("recommended_completion_candidate")
    if isinstance(recommended_candidate, dict):
        for key in (
            "script_file",
            "packet_path",
            "archive_path",
            "packet_manifest_path",
            "run_script_path",
            "package_return_path",
            "env_template_path",
            "readme_path",
        ):
            if key in recommended_candidate:
                recommended_candidate[key] = normalize_value(recommended_candidate.get(key))

    candidate_packets = report.get("candidate_packets")
    if isinstance(candidate_packets, dict):
        for packet in candidate_packets.values():
            if not isinstance(packet, dict):
                continue
            for key in (
                "packet_path",
                "archive_path",
                "packet_manifest_path",
                "run_script_path",
                "package_return_path",
                "env_template_path",
                "readme_path",
            ):
                if key in packet:
                    packet[key] = normalize_value(packet.get(key))


def _annotate_external_expansion_scripts(report: dict[str, Any], expansion_dir: Path, project: Path) -> None:
    operator_scripts = report.get("operator_scripts")
    if not isinstance(operator_scripts, dict):
        operator_scripts = {}
        report["operator_scripts"] = operator_scripts
    operator_script_files = report.get("operator_script_files")
    if not isinstance(operator_script_files, dict):
        operator_script_files = {}
        report["operator_script_files"] = operator_script_files
    for script_id in (
        "build_cohort_expansion_plan",
        "refresh_canonical",
        "apply_return_packet",
        "sync_return_inbox",
        "watch_return_inbox",
        "review_rejected_returns",
    ):
        script_rel = f"bin/{script_id}.sh"
        operator_scripts.setdefault(script_id, script_rel)
        operator_script_files.setdefault(script_id, _project_relative_or_absolute(project, expansion_dir / script_rel))
    recommended_candidate = (
        report.get("recommended_completion_candidate")
        if isinstance(report.get("recommended_completion_candidate"), dict)
        else None
    )
    if recommended_candidate is None:
        return
    provider = recommended_candidate.get("provider")
    if not isinstance(provider, str) or not provider:
        return
    script_id = f"expand_with_{provider}"
    script_rel = f"bin/{script_id}.sh"
    operator_scripts.setdefault(script_id, script_rel)
    operator_script_files.setdefault(script_id, _project_relative_or_absolute(project, expansion_dir / script_rel))
    recommended_candidate.setdefault("script_id", script_id)
    recommended_candidate.setdefault("script", script_rel)
    recommended_candidate.setdefault("script_file", _project_relative_or_absolute(project, expansion_dir / script_rel))


def _annotate_external_expansion_packets(report: dict[str, Any], expansion_dir: Path, project: Path) -> None:
    candidate_packets = report.get("candidate_packets")
    if not isinstance(candidate_packets, dict):
        candidate_packets = {}
        report["candidate_packets"] = candidate_packets
    candidates = report.get("available_real_memory_candidates")
    if isinstance(candidates, list):
        for item in candidates:
            if not isinstance(item, dict):
                continue
            provider = item.get("provider")
            system_id = item.get("system_id")
            if not isinstance(provider, str) or not provider:
                continue
            packet = candidate_packets.get(provider)
            if not isinstance(packet, dict):
                packet = {}
                candidate_packets[provider] = packet
            packet["provider"] = provider
            if isinstance(system_id, str) and system_id:
                packet["system_id"] = system_id
            packet_dir_rel = f"candidates/{provider}"
            packet["packet_dir"] = packet_dir_rel
            packet["packet_path"] = _project_relative_or_absolute(project, expansion_dir / packet_dir_rel)
            packet["archive_file"] = f"candidates/{provider}.zip"
            packet["archive_path"] = _project_relative_or_absolute(project, expansion_dir / "candidates" / f"{provider}.zip")
            packet["packet_manifest_file"] = f"candidates/{provider}/packet_manifest.json"
            packet["packet_manifest_path"] = _project_relative_or_absolute(
                project,
                expansion_dir / "candidates" / provider / "packet_manifest.json",
            )
            packet["run_script_file"] = f"candidates/{provider}/run.sh"
            packet["run_script_path"] = _project_relative_or_absolute(project, expansion_dir / "candidates" / provider / "run.sh")
            packet["package_return_file"] = f"candidates/{provider}/package_return.sh"
            packet["package_return_path"] = _project_relative_or_absolute(
                project,
                expansion_dir / "candidates" / provider / "package_return.sh",
            )
            packet["env_template_file"] = f"candidates/{provider}/env.template"
            packet["env_template_path"] = _project_relative_or_absolute(
                project,
                expansion_dir / "candidates" / provider / "env.template",
            )
            packet["readme_file"] = f"candidates/{provider}/README.md"
            packet["readme_path"] = _project_relative_or_absolute(project, expansion_dir / "candidates" / provider / "README.md")
    recommended_candidate = (
        report.get("recommended_completion_candidate")
        if isinstance(report.get("recommended_completion_candidate"), dict)
        else None
    )
    if recommended_candidate is None:
        return
    provider = recommended_candidate.get("provider")
    if not isinstance(provider, str) or not provider:
        return
    packet = candidate_packets.get(provider)
    if not isinstance(packet, dict):
        return
    for key in (
        "packet_dir",
        "packet_path",
        "archive_file",
        "archive_path",
        "packet_manifest_file",
        "packet_manifest_path",
        "run_script_file",
        "run_script_path",
        "package_return_file",
        "package_return_path",
        "env_template_file",
        "env_template_path",
        "readme_file",
        "readme_path",
    ):
        if key in packet:
            recommended_candidate[key] = packet[key]


def _annotate_external_expansion_handoff(report: dict[str, Any], expansion_dir: Path, project: Path) -> None:
    report["handoff_manifest_file"] = "handoff_manifest.json"
    report["handoff_manifest_path"] = _project_relative_or_absolute(project, expansion_dir / "handoff_manifest.json")
    report.setdefault("return_inbox", {"candidate_inbox": "returns/inbox"})
    report.setdefault("return_archive", {"candidate_archive": "returns/processed"})
    report.setdefault("return_reject_archive", {"candidate_archive": "returns/rejected"})
    report.setdefault("return_inbox_sync_report_file", "return_inbox_sync.json")
    report.setdefault("return_inbox_sync_report_path", _project_relative_or_absolute(project, expansion_dir / "return_inbox_sync.json"))
    report.setdefault("return_inbox_state_file", "return_inbox_state.json")
    report.setdefault("return_inbox_state_path", _project_relative_or_absolute(project, expansion_dir / "return_inbox_state.json"))
    report.setdefault("return_inbox_watch_file", "return_inbox_watch.json")
    report.setdefault("return_inbox_watch_output_file", _project_relative_or_absolute(project, expansion_dir / "return_inbox_watch.json"))
    report.setdefault("rejected_returns_report_file", "rejected_returns_report.json")
    report.setdefault("rejected_returns_report_path", _project_relative_or_absolute(project, expansion_dir / "rejected_returns_report.json"))
    report["return_reject_archive_paths"] = {
        key: _project_relative_or_absolute(project, expansion_dir / value)
        for key, value in sorted(report["return_reject_archive"].items())
        if isinstance(value, str) and value
    }
    pending_packets = sorted((expansion_dir / "returns" / "inbox").glob("*.zip")) if (expansion_dir / "returns" / "inbox").exists() else []
    report["pending_return_packets"] = [
        path.relative_to(expansion_dir).as_posix() if path.is_relative_to(expansion_dir) else str(path)
        for path in pending_packets
    ]
    report["pending_return_packet_paths"] = [_project_relative_or_absolute(project, path) for path in pending_packets]
    report["rejected_return_summary"] = summarize_external_cohort_rejected_returns(expansion_dir)
    report.setdefault("watch_stop_exit_codes", {"max_iterations": 0, "rejected_returns": 2, "ready": 3})
    operator_commands = report.get("operator_commands")
    if not isinstance(operator_commands, dict):
        operator_commands = {}
        report["operator_commands"] = operator_commands
    operator_scripts = report.get("operator_scripts") if isinstance(report.get("operator_scripts"), dict) else {}
    operator_script_files = (
        report.get("operator_script_files") if isinstance(report.get("operator_script_files"), dict) else {}
    )
    report.update(
        _external_recommended_next_refs(
            report,
            operator_commands={str(key): str(value) for key, value in operator_commands.items() if isinstance(value, str)},
            operator_scripts={str(key): str(value) for key, value in operator_scripts.items() if isinstance(value, str)},
            operator_script_files={
                str(key): str(value) for key, value in operator_script_files.items() if isinstance(value, str)
            },
        )
    )
    sync_payload = _read_optional_json(expansion_dir / "return_inbox_sync.json")
    watch_payload = _read_optional_json(expansion_dir / "return_inbox_watch.json")
    _annotate_sidecar_runtime_fields(report, "return_inbox_sync", sync_payload)
    _annotate_sidecar_runtime_fields(report, "return_inbox_watch", watch_payload)
    _fill_recommended_next_from_sidecars(report, watch_payload, sync_payload)


def _read_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    try:
        payload = read_json(candidate)
    except Exception:  # noqa: BLE001 - readiness should degrade instead of failing hard on stale artifacts
        return None
    return payload if isinstance(payload, dict) else None


_SIDECAR_NEXT_ACTION_FIELDS = (
    "next_command_id",
    "next_command",
    "next_script_id",
    "next_script",
    "next_script_file",
)


def _nonempty_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return str(value)
    return None


def _annotate_sidecar_runtime_fields(report: dict[str, Any], prefix: str, payload: dict[str, Any] | None) -> None:
    if not isinstance(payload, dict):
        return
    status = _nonempty_string(payload.get("status"))
    if status is not None:
        report[f"{prefix}_status"] = status
    stop_reason = _nonempty_string(payload.get("stop_reason"))
    if stop_reason is not None:
        report[f"{prefix}_stop_reason"] = stop_reason
    for field in _SIDECAR_NEXT_ACTION_FIELDS:
        value = _nonempty_string(payload.get(field))
        if value is not None:
            report[f"{prefix}_{field}"] = value


def _fill_recommended_next_from_sidecars(
    report: dict[str, Any],
    *payloads: dict[str, Any] | None,
) -> None:
    fallback: dict[str, str] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for field in _SIDECAR_NEXT_ACTION_FIELDS:
            value = _nonempty_string(payload.get(field))
            if value is not None:
                fallback[field] = value
        if fallback:
            break
    if not fallback:
        return
    mapping = {
        "recommended_next_command_id": "next_command_id",
        "recommended_next_command": "next_command",
        "recommended_next_script_id": "next_script_id",
        "recommended_next_script": "next_script",
        "recommended_next_script_file": "next_script_file",
    }
    for target_key, source_key in mapping.items():
        if _nonempty_string(report.get(target_key)) is None and source_key in fallback:
            report[target_key] = fallback[source_key]


def _resolve_project_relative_path(project: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project / path


def _project_relative_or_absolute(project: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_bundle_relative_path(bundle_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return bundle_dir / path


def _default_watch_stop_actions(
    bundle_dir: Path,
    operator_scripts: dict[str, str],
) -> dict[str, dict[str, Any]]:
    def _script_refs(script_id: str) -> tuple[str | None, str | None]:
        raw_script = operator_scripts.get(script_id)
        if not isinstance(raw_script, str) or not raw_script.strip():
            return None, None
        return raw_script, str(_resolve_bundle_relative_path(bundle_dir, raw_script))

    watch_script, watch_script_file = _script_refs("watch_return_inbox")
    review_script, review_script_file = _script_refs("review_rejected_returns")
    reconcile_script, reconcile_script_file = _script_refs("reconcile_when_ready")
    return {
        "max_iterations": {
            "exit_code": HUMAN_AUDIT_WATCH_STOP_EXIT_CODES["max_iterations"],
            "kind": "continue_waiting",
            "next_command": (
                f"{HUMAN_AUDIT_CLI_PREFIX} watch-human-audit-return-inbox "
                f"--bundle-dir {bundle_dir} --interval-s {HUMAN_AUDIT_OPERATOR_WATCH_INTERVAL_S} --max-iterations 0 "
                "--stop-when-ready --stop-when-rejected "
                f"--output {bundle_dir / 'return_inbox_watch.json'}"
            ),
            "next_command_id": "watch_return_inbox",
            "next_script": watch_script,
            "next_script_file": watch_script_file,
            "next_script_id": "watch_return_inbox",
        },
        "rejected_returns": {
            "exit_code": HUMAN_AUDIT_WATCH_STOP_EXIT_CODES["rejected_returns"],
            "kind": "triage_rejected_returns",
            "next_command": (
                f"{HUMAN_AUDIT_CLI_PREFIX} summarize-human-audit-rejected-returns "
                f"--bundle-dir {bundle_dir} --output {bundle_dir / 'rejected_returns_report.json'}"
            ),
            "next_command_id": "review_rejected_returns",
            "next_script": review_script,
            "next_script_file": review_script_file,
            "next_script_id": "review_rejected_returns",
        },
        "ready": {
            "exit_code": HUMAN_AUDIT_WATCH_STOP_EXIT_CODES["ready"],
            "kind": "reconcile_ready_bundle",
            "next_command": (
                f"{HUMAN_AUDIT_CLI_PREFIX} reconcile-human-audit-evidence-bundle "
                f"--bundle-dir {bundle_dir} "
                f"--signed-at {HUMAN_AUDIT_BUNDLE_SANDBOX_SIGNED_AT} "
                f"--annotation-guideline {HUMAN_AUDIT_BUNDLE_SANDBOX_ANNOTATION_GUIDELINE} "
                f"--adjudication-policy '{HUMAN_AUDIT_BUNDLE_SANDBOX_ADJUDICATION_POLICY}'"
            ),
            "next_command_id": "reconcile_when_ready",
            "next_script": reconcile_script,
            "next_script_file": reconcile_script_file,
            "next_script_id": "reconcile_when_ready",
        },
    }


def _default_operator_commands(bundle_dir: Path, task_manifest_file: str | None) -> dict[str, str]:
    commands: dict[str, str] = {}
    if isinstance(task_manifest_file, str) and task_manifest_file:
        commands["progress"] = (
            f"{HUMAN_AUDIT_CLI_PREFIX} human-audit-progress "
            f"--task-manifest {task_manifest_file} "
            f"--output {bundle_dir / 'progress.json'}"
        )
    commands["watch_return_inbox"] = (
        f"{HUMAN_AUDIT_CLI_PREFIX} watch-human-audit-return-inbox "
        f"--bundle-dir {bundle_dir} --interval-s {HUMAN_AUDIT_OPERATOR_WATCH_INTERVAL_S} --max-iterations 0 "
        "--stop-when-ready --stop-when-rejected "
        f"--output {bundle_dir / 'return_inbox_watch.json'}"
    )
    commands["sync_return_inbox"] = (
        f"{HUMAN_AUDIT_CLI_PREFIX} sync-human-audit-return-inbox "
        f"--bundle-dir {bundle_dir} "
        "--reconcile-when-ready "
        f"--signed-at {HUMAN_AUDIT_BUNDLE_SANDBOX_SIGNED_AT} "
        f"--annotation-guideline {HUMAN_AUDIT_BUNDLE_SANDBOX_ANNOTATION_GUIDELINE} "
        f"--adjudication-policy '{HUMAN_AUDIT_BUNDLE_SANDBOX_ADJUDICATION_POLICY}' "
        f"--output {bundle_dir / 'return_inbox_sync.json'}"
    )
    commands["review_rejected_returns"] = (
        f"{HUMAN_AUDIT_CLI_PREFIX} summarize-human-audit-rejected-returns "
        f"--bundle-dir {bundle_dir} --output {bundle_dir / 'rejected_returns_report.json'}"
    )
    commands["reconcile_when_ready"] = (
        f"{HUMAN_AUDIT_CLI_PREFIX} reconcile-human-audit-evidence-bundle "
        f"--bundle-dir {bundle_dir} "
        f"--signed-at {HUMAN_AUDIT_BUNDLE_SANDBOX_SIGNED_AT} "
        f"--annotation-guideline {HUMAN_AUDIT_BUNDLE_SANDBOX_ANNOTATION_GUIDELINE} "
        f"--adjudication-policy '{HUMAN_AUDIT_BUNDLE_SANDBOX_ADJUDICATION_POLICY}'"
    )
    commands["verify_bundle"] = (
        f"{HUMAN_AUDIT_CLI_PREFIX} verify-human-audit-evidence-bundle "
        f"--bundle-dir {bundle_dir} --output {bundle_dir / 'bundle_verification.json'}"
    )
    return commands


def _normalize_watch_stop_actions(
    bundle_dir: Path,
    operator_scripts: dict[str, str],
    raw_actions: Any,
) -> dict[str, dict[str, Any]]:
    default_actions = _default_watch_stop_actions(bundle_dir, operator_scripts)
    if not isinstance(raw_actions, dict):
        return default_actions
    normalized: dict[str, dict[str, Any]] = {}
    for stop_reason, default_action in sorted(default_actions.items()):
        raw_action = raw_actions.get(stop_reason)
        if isinstance(raw_action, dict):
            merged = dict(default_action)
            merged.update({str(key): value for key, value in raw_action.items()})
            normalized[stop_reason] = merged
        else:
            normalized[stop_reason] = dict(default_action)
    for stop_reason, raw_action in sorted(raw_actions.items()):
        if stop_reason not in normalized and isinstance(raw_action, dict):
            normalized[str(stop_reason)] = dict(raw_action)
    return normalized


def _bundle_rejected_return_summary(
    bundle_dir: Path,
    bundle_manifest_payload: dict[str, Any],
) -> dict[str, Any] | None:
    raw_summary = bundle_manifest_payload.get("rejected_return_summary")
    if isinstance(raw_summary, dict):
        return _normalize_rejected_return_summary(bundle_dir, raw_summary)
    raw_state_file = bundle_manifest_payload.get("return_inbox_state_file")
    if isinstance(raw_state_file, str) and raw_state_file.strip():
        state_path = _resolve_bundle_relative_path(bundle_dir, raw_state_file)
    else:
        state_path = bundle_dir / "return_inbox_state.json"
    if not state_path.exists():
        return None
    try:
        state_payload = read_json(state_path)
    except Exception:  # noqa: BLE001 - readiness should degrade rather than fail hard on state parse errors
        return None
    if not isinstance(state_payload, dict):
        return None
    return _normalize_rejected_return_summary(
        bundle_dir,
        {
            "rejected_annotator_packets": state_payload.get("rejected_annotator_packets"),
            "rejected_adjudication_packets": state_payload.get("rejected_adjudication_packets"),
        },
    )


def _normalize_rejected_return_summary(bundle_dir: Path, raw_summary: dict[str, Any]) -> dict[str, Any]:
    rejected_annotator_packets = _normalize_rejected_return_entries(
        bundle_dir,
        raw_summary.get("rejected_annotator_packets"),
    )
    rejected_adjudication_packets = _normalize_rejected_return_entries(
        bundle_dir,
        raw_summary.get("rejected_adjudication_packets"),
    )
    return {
        "num_rejected_annotator_packets": len(rejected_annotator_packets),
        "num_rejected_adjudication_packets": len(rejected_adjudication_packets),
        "rejected_annotator_packets": rejected_annotator_packets,
        "rejected_adjudication_packets": rejected_adjudication_packets,
    }


def _normalize_rejected_return_entries(bundle_dir: Path, raw_entries: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not isinstance(raw_entries, list):
        return entries
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        normalized_entry: dict[str, Any] = {}
        packet_path = entry.get("packet_path")
        if isinstance(packet_path, str) and packet_path.strip():
            normalized_entry["packet_path"] = packet_path
        packet_fingerprint = entry.get("packet_fingerprint")
        if isinstance(packet_fingerprint, str) and packet_fingerprint.strip():
            normalized_entry["packet_fingerprint"] = packet_fingerprint
        rejection_error = entry.get("rejection_error")
        if isinstance(rejection_error, str) and rejection_error.strip():
            normalized_entry["rejection_error"] = rejection_error
        rejected_archive_file = entry.get("rejected_archive_file")
        if isinstance(rejected_archive_file, str) and rejected_archive_file.strip():
            normalized_entry["rejected_archive_file"] = rejected_archive_file
            normalized_entry["rejected_archive_path"] = str(
                _resolve_bundle_relative_path(bundle_dir, rejected_archive_file)
            )
        if normalized_entry:
            entries.append(normalized_entry)
    return entries


def _read_real_system_summary(path: Path) -> dict[str, Any]:
    try:
        summary = read_json(path)
    except Exception:
        return {}
    return summary if isinstance(summary, dict) else {}


def _next_actions(
    project: Path,
    blockers: list[dict[str, Any]],
    human_progress: dict[str, Any],
    real_system: dict[str, Any],
    real_system_sample: dict[str, Any],
    git_hygiene: dict[str, Any],
    config_validation: dict[str, Any],
    external_plan: dict[str, Any],
    external_validation: dict[str, Any],
    external_gap: dict[str, Any],
    external_expansion: dict[str, Any],
) -> list[dict[str, Any]]:
    blocker_ids = {str(blocker["check_id"]) for blocker in blockers}
    blocker_statuses = {
        str(blocker["check_id"]): str(blocker["status"])
        for blocker in blockers
        if isinstance(blocker.get("check_id"), str) and isinstance(blocker.get("status"), str)
    }
    actions: list[dict[str, Any]] = []
    if "human_audit" in blocker_ids:
        human_cli_prefix = "PYTHONPATH=. python -m agent_memory_benchmark"
        task_manifest_file = human_progress.get("task_manifest_file")
        annotations_file = human_progress.get("annotations_file")
        bundle_manifest_file = human_progress.get("bundle_manifest_file")
        bundle_dir_value = human_progress.get("bundle_dir")
        operator_scripts = human_progress.get("operator_scripts") if isinstance(human_progress.get("operator_scripts"), dict) else {}
        operator_script_files = (
            human_progress.get("operator_script_files")
            if isinstance(human_progress.get("operator_script_files"), dict)
            else {}
        )
        operator_commands = (
            human_progress.get("operator_commands")
            if isinstance(human_progress.get("operator_commands"), dict)
            else {}
        )
        recommended_next_command_id = (
            str(human_progress.get("recommended_next_command_id"))
            if isinstance(human_progress.get("recommended_next_command_id"), str)
            and str(human_progress.get("recommended_next_command_id")).strip()
            else None
        )
        recommended_next_command = (
            str(human_progress.get("recommended_next_command"))
            if isinstance(human_progress.get("recommended_next_command"), str)
            and str(human_progress.get("recommended_next_command")).strip()
            else None
        )
        recommended_next_script_id = (
            str(human_progress.get("recommended_next_script_id"))
            if isinstance(human_progress.get("recommended_next_script_id"), str)
            and str(human_progress.get("recommended_next_script_id")).strip()
            else None
        )
        recommended_next_script = (
            str(human_progress.get("recommended_next_script"))
            if isinstance(human_progress.get("recommended_next_script"), str)
            and str(human_progress.get("recommended_next_script")).strip()
            else None
        )
        recommended_next_script_file = (
            str(human_progress.get("recommended_next_script_file"))
            if isinstance(human_progress.get("recommended_next_script_file"), str)
            and str(human_progress.get("recommended_next_script_file")).strip()
            else None
        )
        next_command_id: str | None = None
        next_script: str | None = None
        next_script_id: str | None = None
        next_script_file: str | None = None
        next_command_expected_exit_codes: dict[str, int] | None = None
        next_command_stop_actions: dict[str, Any] | None = None
        pending_return_packets = (
            list(human_progress.get("pending_return_packets", []))
            if isinstance(human_progress.get("pending_return_packets"), list)
            else []
        )
        release_manifest = _default_human_release_manifest(project)
        bundle_dir = Path(bundle_dir_value) if isinstance(bundle_dir_value, str) and bundle_dir_value else project / "reports/human_audit_bundle/current"
        bundle_annotation_guideline = bundle_dir / "docs/annotation_guideline.md"
        command = f"{human_cli_prefix} human-audit-progress --task-manifest TASK_MANIFEST --annotations COMPLETED.jsonl --output progress.json"
        recommended_commands = human_progress.get("recommended_next_commands")
        watch_stop_exit_codes = (
            human_progress.get("watch_stop_exit_codes")
            if isinstance(human_progress.get("watch_stop_exit_codes"), dict)
            else {}
        )
        if not watch_stop_exit_codes and bundle_manifest_file:
            watch_stop_exit_codes = dict(HUMAN_AUDIT_WATCH_STOP_EXIT_CODES)
        watch_stop_actions = (
            human_progress.get("watch_stop_actions")
            if isinstance(human_progress.get("watch_stop_actions"), dict)
            else {}
        )
        if not watch_stop_actions and bundle_manifest_file:
            watch_stop_actions = _default_watch_stop_actions(bundle_dir, operator_scripts)
        if bundle_manifest_file and task_manifest_file:
            progress_output = bundle_dir / "progress.json"
            command = (
                f"{human_cli_prefix} watch-human-audit-return-inbox "
                f"--bundle-dir {bundle_dir} --interval-s 120 --max-iterations 0"
            )
            if annotations_file:
                command = (
                    f"{human_cli_prefix} human-audit-progress "
                    f"--task-manifest {task_manifest_file} "
                    f"--annotations {annotations_file} "
                    f"--output {progress_output}"
                )
            if human_progress.get("ready_for_merge") or human_progress.get("ready_for_finalize"):
                command = (
                    f"{human_cli_prefix} reconcile-human-audit-evidence-bundle "
                    f"--bundle-dir {bundle_dir} "
                    "--signed-at 2026-05-13T00:00:00Z "
                    f"--annotation-guideline {bundle_annotation_guideline} "
                    "--adjudication-policy 'Disagreements are adjudicated after double annotation.'"
                )
        elif task_manifest_file:
            command = (
                f"{human_cli_prefix} build-human-audit-evidence-bundle "
                f"--task-manifest {task_manifest_file} "
                f"--output-dir {bundle_dir}"
            )
            if release_manifest is not None:
                command = (
                    f"{human_cli_prefix} build-human-audit-evidence-bundle "
                    f"--manifest {release_manifest} "
                    f"--task-manifest {task_manifest_file} "
                    f"--output-dir {bundle_dir}"
                )
            if annotations_file:
                command += f" --annotations {annotations_file}"
        if human_progress.get("ready_for_finalize") and task_manifest_file and annotations_file:
            command = (
                f"{human_cli_prefix} generate-human-audit-attestation "
                f"--task-manifest {task_manifest_file} "
                f"--annotations {annotations_file} "
                f"--output {bundle_dir / 'attestation.json'} "
                "--signed-at 2026-05-13T00:00:00Z "
                f"--annotation-guideline {bundle_annotation_guideline} "
                "--adjudication-policy 'Disagreements are adjudicated after double annotation.'"
            )
        if bundle_manifest_file and isinstance(recommended_commands, list) and recommended_commands:
            command = str(recommended_commands[0])
        if pending_return_packets and isinstance(command, str):
            if "watch-human-audit-return-inbox" in command or "human-audit-progress" in command:
                sync_command = operator_commands.get("sync_return_inbox")
                if isinstance(sync_command, str) and sync_command.strip():
                    command = sync_command
        if (
            isinstance(command, str)
            and recommended_next_command
            and command == recommended_next_command
        ):
            next_command_id = recommended_next_command_id
            next_script_id = recommended_next_script_id
            next_script = recommended_next_script
            next_script_file = recommended_next_script_file
        if isinstance(command, str):
            if "watch-human-audit-return-inbox" in command:
                next_command_id = next_command_id or "watch_return_inbox"
                script_path = operator_scripts.get("watch_return_inbox")
                if isinstance(script_path, str) and script_path.strip():
                    next_script = script_path
                    next_script_id = next_script_id or "watch_return_inbox"
                script_file = operator_script_files.get("watch_return_inbox")
                if isinstance(script_file, str) and script_file.strip():
                    next_script_file = script_file
                next_command_expected_exit_codes = {
                    str(key): int(value) for key, value in sorted(watch_stop_exit_codes.items()) if isinstance(value, int)
                }
                next_command_stop_actions = {
                    str(key): dict(value) for key, value in sorted(watch_stop_actions.items()) if isinstance(value, dict)
                }
            elif "reconcile-human-audit-evidence-bundle" in command:
                next_command_id = next_command_id or "reconcile_when_ready"
                script_path = operator_scripts.get("reconcile_when_ready")
                if isinstance(script_path, str) and script_path.strip():
                    next_script = script_path
                    next_script_id = next_script_id or "reconcile_when_ready"
                script_file = operator_script_files.get("reconcile_when_ready")
                if isinstance(script_file, str) and script_file.strip():
                    next_script_file = script_file
            elif "human-audit-progress" in command:
                next_command_id = next_command_id or "progress"
                script_path = operator_scripts.get("progress")
                if isinstance(script_path, str) and script_path.strip():
                    next_script = script_path
                    next_script_id = next_script_id or "progress"
                script_file = operator_script_files.get("progress")
                if isinstance(script_file, str) and script_file.strip():
                    next_script_file = script_file
            elif "sync-human-audit-return-inbox" in command:
                next_command_id = next_command_id or "sync_return_inbox"
                script_path = operator_scripts.get("sync_return_inbox")
                if isinstance(script_path, str) and script_path.strip():
                    next_script = script_path
                    next_script_id = next_script_id or "sync_return_inbox"
                script_file = operator_script_files.get("sync_return_inbox")
                if isinstance(script_file, str) and script_file.strip():
                    next_script_file = script_file
            elif "summarize-human-audit-rejected-returns" in command:
                next_command_id = next_command_id or "review_rejected_returns"
                script_path = operator_scripts.get("review_rejected_returns")
                if isinstance(script_path, str) and script_path.strip():
                    next_script = script_path
                    next_script_id = next_script_id or "review_rejected_returns"
                script_file = operator_script_files.get("review_rejected_returns")
                if isinstance(script_file, str) and script_file.strip():
                    next_script_file = script_file
        actions.append(
            _next_action_record(
                "human_audit",
                blocker_status=blocker_statuses.get("human_audit"),
                kind="human_audit_operator",
                next_command_id=next_command_id,
                next_script_id=next_script_id,
                next_command=command,
                next_script=next_script,
                next_script_file=next_script_file,
                next_command_expected_exit_codes=next_command_expected_exit_codes,
                next_command_stop_actions=next_command_stop_actions,
                reason="Human audit still requires a concrete annotator handoff package, verified attestation, and finalize-generated agreement metrics.",
                handoff_packets=_human_handoff_packets(human_progress),
                handoff_packet_files=_human_handoff_packet_files(human_progress, bundle_dir),
                handoff_manifest_file=human_progress.get("handoff_manifest_file"),
                return_inbox=human_progress.get("return_inbox"),
                return_inbox_paths=human_progress.get("return_inbox_paths"),
                return_archive=human_progress.get("return_archive"),
                return_archive_paths=human_progress.get("return_archive_paths"),
                return_reject_archive=human_progress.get("return_reject_archive"),
                return_reject_archive_paths=human_progress.get("return_reject_archive_paths"),
                rejected_return_summary=human_progress.get("rejected_return_summary"),
                rejected_returns_report_file=human_progress.get("rejected_returns_report_file"),
                rejected_returns_report_path=human_progress.get("rejected_returns_report_path"),
                return_inbox_sync_report_file=human_progress.get("return_inbox_sync_report_file"),
                return_inbox_sync_report_path=human_progress.get("return_inbox_sync_report_path"),
                return_inbox_state_file=human_progress.get("return_inbox_state_file"),
                return_inbox_state_path=human_progress.get("return_inbox_state_path"),
                pending_return_packets=pending_return_packets,
                pending_return_packet_paths=human_progress.get("pending_return_packet_paths"),
                return_inbox_watch_file=human_progress.get("return_inbox_watch_file"),
                return_inbox_watch_output_file=human_progress.get("return_inbox_watch_output_file"),
                watch_stop_exit_codes=watch_stop_exit_codes,
                watch_stop_actions=watch_stop_actions,
                operator_scripts=operator_scripts,
                operator_script_files=operator_script_files,
                operator_commands=operator_commands,
            )
        )
    if "real_memory_system_integrations" in blocker_ids:
        refresh_spec = _default_real_system_refresh_spec(project)
        config_paths = tuple(_project_relative_or_absolute(project, path) for path in _default_real_system_config_paths(project))
        config_arg = " ".join(config_paths) if config_paths else "mem0.json letta.json langmem.json zep_graphiti.json"
        validation_output = _DEFAULT_REAL_SYSTEM_CONFIG_VALIDATION_OUTPUT
        release_manifest = _default_real_system_release_manifest(project)
        release_manifest_arg = (
            _project_relative_or_absolute(project, release_manifest) if release_manifest is not None else "MANIFEST"
        )
        command = (
            "python -m agent_memory_benchmark validate-integration-configs "
            f"--configs {config_arg} --output {validation_output}"
        )
        if refresh_spec is not None and real_system.get("status") != "passed":
            command = (
                "python -m agent_memory_benchmark watch-real-system-canonical "
                f"--spec {refresh_spec} --root {project} --interval-s 120 --max-iterations 0"
            )
        elif real_system_sample.get("status") == "passed" and real_system.get("status") != "passed":
            command = (
                "python -m agent_memory_benchmark run-release-agent-matrix "
                f"--manifest {release_manifest_arg} --split public_dev --configs {config_arg} "
                "--output-dir reports/real_system_runs"
            )
        elif config_validation.get("status") == "passed" and real_system.get("status") != "passed":
            command = (
                "python -m agent_memory_benchmark run-release-agent-matrix "
                f"--manifest {release_manifest_arg} --split public_dev --configs {config_arg} "
                "--output-dir reports/real_system_runs"
            )
        actions.append(
            _next_action_record(
                "real_memory_system_integrations",
                blocker_status=blocker_statuses.get("real_memory_system_integrations"),
                kind="real_system_operator",
                next_command=command,
                reason="Four real provider runs and a passing matrix validation are still required.",
            )
        )
    if "git_hygiene" in blocker_ids:
        report_path = project / "reports/examples/amst_git_hygiene_current.json"
        recommended_batches = git_hygiene.get("recommended_batches", [])
        rebuild_plan_command = (
            "python -m agent_memory_benchmark build-git-hygiene-plan "
            f"--root {project} --output-dir {project / 'reports/git_hygiene/current'}"
        )
        command = f"python -m agent_memory_benchmark git-hygiene-report --root {project} --output {report_path}"
        plan_manifest = _default_git_hygiene_plan_manifest(project)
        if plan_manifest is not None:
            try:
                plan = read_json(plan_manifest)
            except Exception:
                plan = {}
            batches = plan.get("batches", []) if isinstance(plan, dict) else []
            if batches:
                current_batch_id = batches[0].get("batch_id")
                live_batch_id = recommended_batches[0].get("batch_id") if recommended_batches else None
                if live_batch_id and current_batch_id != live_batch_id:
                    command = rebuild_plan_command
                else:
                    safe_stage_command = batches[0].get("safe_stage_command")
                    if isinstance(safe_stage_command, str) and safe_stage_command:
                        command = safe_stage_command
                    else:
                        command = f"python -m agent_memory_benchmark git-hygiene-report --root {project} --output {report_path}"
            elif recommended_batches:
                command = rebuild_plan_command
        elif recommended_batches:
            command = rebuild_plan_command
        actions.append(
            _next_action_record(
                "git_hygiene",
                blocker_status=blocker_statuses.get("git_hygiene"),
                kind="git_hygiene_operator",
                next_command=command,
                reason="Critical tracked or untracked implementation, client, test, or config assets still need staged reconciliation.",
            )
        )
    if "external_benchmark_correlation" in blocker_ids:
        ready = external_plan.get("summary", {}).get("ready", 0)
        requirements = [
            requirement
            for requirement in external_plan.get("requirements", [])
            if isinstance(requirement, dict)
        ]
        missing_requirements = [item for item in requirements if item.get("status") == "missing"]
        next_missing = missing_requirements[0] if missing_requirements else None
        next_missing_benchmark = str(next_missing.get("benchmark_id")) if isinstance(next_missing, dict) else "BENCH"
        command = (
            "python -m agent_memory_benchmark normalize-external-scores "
            f"--input RAW --benchmark-id {next_missing_benchmark} "
            f"--output reports/external/{next_missing_benchmark}_scores.json"
        )
        if ready:
            command = "python -m agent_memory_benchmark external-correlation-batch --amst-reports AMST_REPORTS --external-scores reports/external/*_scores.json --output-dir reports/external"
        if external_plan.get("status") == "passed" and external_validation.get("status") != "passed":
            command = (
                "python -m agent_memory_benchmark summarize-external-evidence-gaps "
                "--correlations reports/external/*_correlation.json "
                "--real-system-validation reports/real_system_runs/canonical_public_dev_refresh_current_matrix_validation.json "
                "--output reports/external/evidence_gap_report.json "
                "--min-shared-systems 3 --min-control-shared-systems 1 --min-real-memory-shared-systems 1"
            )
            if external_expansion.get("status") == "ready":
                command = (
                    "python -m agent_memory_benchmark build-external-cohort-expansion-plan "
                    "--correlations reports/external/*_correlation.json "
                    "--real-system-validation reports/real_system_runs/canonical_public_dev_refresh_current_matrix_validation.json "
                    "--output reports/external/cohort_expansion_plan.json "
                    "--min-shared-systems 3 --min-control-shared-systems 1 --min-real-memory-shared-systems 1"
                )
        recommended_candidate = (
            external_expansion.get("recommended_completion_candidate")
            if isinstance(external_expansion.get("recommended_completion_candidate"), dict)
            else {}
        )
        pending_return_packets = (
            list(external_expansion.get("pending_return_packets", []))
            if isinstance(external_expansion.get("pending_return_packets"), list)
            else []
        )
        rejected_return_summary = (
            external_expansion.get("rejected_return_summary")
            if isinstance(external_expansion.get("rejected_return_summary"), dict)
            else {}
        )
        has_rejected_returns = bool(rejected_return_summary.get("num_rejected_candidate_packets"))
        next_command_id: str | None = None
        next_script_id: str | None = None
        next_script = external_expansion.get("recommended_next_script")
        next_script_file = external_expansion.get("recommended_next_script_file")
        if not (isinstance(next_script, str) and next_script.strip()):
            next_script = recommended_candidate.get("script")
        if not (isinstance(next_script_file, str) and next_script_file.strip()):
            next_script_file = recommended_candidate.get("script_file")
        recommended_next_command = external_expansion.get("recommended_next_command")
        use_recommended_next = bool(has_rejected_returns or pending_return_packets) or external_expansion.get("status") in {
            "ready",
            "not_needed",
        }
        if use_recommended_next and isinstance(recommended_next_command, str) and recommended_next_command.strip():
            command = recommended_next_command
            if isinstance(external_expansion.get("recommended_next_command_id"), str):
                next_command_id = str(external_expansion.get("recommended_next_command_id"))
            if isinstance(external_expansion.get("recommended_next_script_id"), str):
                next_script_id = str(external_expansion.get("recommended_next_script_id"))
        if external_expansion.get("status") == "ready":
            if isinstance(next_script_file, str) and next_script_file.strip():
                command = next_script_file
            elif isinstance(next_script, str) and next_script.strip():
                command = str(_resolve_project_relative_path(project, next_script))
        if has_rejected_returns:
            next_script = external_expansion.get("operator_scripts", {}).get("review_rejected_returns")
            next_script_file = external_expansion.get("operator_script_files", {}).get("review_rejected_returns")
            if isinstance(next_script_file, str) and next_script_file.strip():
                command = next_script_file
            elif isinstance(next_script, str) and next_script.strip():
                command = str(_resolve_project_relative_path(project, next_script))
        elif pending_return_packets:
            next_script = external_expansion.get("operator_scripts", {}).get("sync_return_inbox")
            next_script_file = external_expansion.get("operator_script_files", {}).get("sync_return_inbox")
            if isinstance(next_script_file, str) and next_script_file.strip():
                command = next_script_file
        actions.append(
            _next_action_record(
                "external_benchmark_correlation",
                blocker_status=blocker_statuses.get("external_benchmark_correlation"),
                kind="external_benchmark_operator",
                next_command_id=next_command_id,
                next_script_id=next_script_id,
                next_command=command,
                next_script=next_script,
                next_script_file=next_script_file,
                gap_status=external_gap.get("status"),
                expansion_status=external_expansion.get("status"),
                control_only_smoke_cohort=bool(external_gap.get("control_only_smoke_cohort")),
                providers_missing_from_all_external_benchmarks=list(
                    external_gap.get("providers_missing_from_all_external_benchmarks", [])
                )
                if isinstance(external_gap.get("providers_missing_from_all_external_benchmarks"), list)
                else [],
                minimum_completion_candidate_providers=[
                    item.get("provider")
                    for item in external_expansion.get("minimum_completion_candidates", [])
                    if isinstance(item, dict) and item.get("provider")
                ]
                if isinstance(external_expansion.get("minimum_completion_candidates"), list)
                else [],
                recommended_next_priority=external_gap.get("recommended_next_priority"),
                gap_report_file=external_gap.get("gap_report_file"),
                gap_report_path=external_gap.get("gap_report_path"),
                expansion_plan_file=external_expansion.get("expansion_plan_file"),
                expansion_plan_path=external_expansion.get("expansion_plan_path"),
                recommended_candidate_provider=recommended_candidate.get("provider"),
                recommended_candidate_system_id=recommended_candidate.get("system_id"),
                recommended_candidate_packet_dir=recommended_candidate.get("packet_dir"),
                recommended_candidate_packet_path=recommended_candidate.get("packet_path"),
                recommended_candidate_archive_file=recommended_candidate.get("archive_file"),
                recommended_candidate_archive_path=recommended_candidate.get("archive_path"),
                recommended_candidate_packet_manifest_file=recommended_candidate.get("packet_manifest_file"),
                recommended_candidate_packet_manifest_path=recommended_candidate.get("packet_manifest_path"),
                recommended_candidate_run_script_file=recommended_candidate.get("run_script_file"),
                recommended_candidate_run_script_path=recommended_candidate.get("run_script_path"),
                recommended_candidate_package_return_file=recommended_candidate.get("package_return_file"),
                recommended_candidate_package_return_path=recommended_candidate.get("package_return_path"),
                recommended_candidate_env_template_file=recommended_candidate.get("env_template_file"),
                recommended_candidate_env_template_path=recommended_candidate.get("env_template_path"),
                recommended_candidate_readme_file=recommended_candidate.get("readme_file"),
                recommended_candidate_readme_path=recommended_candidate.get("readme_path"),
                expansion_validation_file=external_expansion.get("validation_file"),
                expansion_validation_path=external_expansion.get("validation_path"),
                expansion_validation_status=external_expansion.get("validation_status"),
                expansion_validation_errors=external_expansion.get("validation_errors"),
                expansion_readme_file=external_expansion.get("readme_file"),
                expansion_readme_path=external_expansion.get("readme_path"),
                handoff_manifest_file=external_expansion.get("handoff_manifest_file"),
                handoff_manifest_path=external_expansion.get("handoff_manifest_path"),
                return_inbox=external_expansion.get("return_inbox"),
                return_archive=external_expansion.get("return_archive"),
                return_reject_archive=external_expansion.get("return_reject_archive"),
                return_reject_archive_paths=external_expansion.get("return_reject_archive_paths"),
                rejected_return_summary=rejected_return_summary,
                rejected_returns_report_file=external_expansion.get("rejected_returns_report_file"),
                rejected_returns_report_path=external_expansion.get("rejected_returns_report_path"),
                return_inbox_sync_report_file=external_expansion.get("return_inbox_sync_report_file"),
                return_inbox_sync_report_path=external_expansion.get("return_inbox_sync_report_path"),
                return_inbox_state_file=external_expansion.get("return_inbox_state_file"),
                return_inbox_state_path=external_expansion.get("return_inbox_state_path"),
                return_inbox_watch_file=external_expansion.get("return_inbox_watch_file"),
                return_inbox_watch_output_file=external_expansion.get("return_inbox_watch_output_file"),
                watch_stop_exit_codes=external_expansion.get("watch_stop_exit_codes"),
                pending_return_packets=pending_return_packets,
                pending_return_packet_paths=external_expansion.get("pending_return_packet_paths"),
                operator_scripts=external_expansion.get("operator_scripts"),
                operator_script_files=external_expansion.get("operator_script_files"),
                operator_commands=external_expansion.get("operator_commands"),
                reason=(
                    "External benchmark correlation requires normalized real scores, correlation reports, and a shared "
                    "3+ same-system cohort that includes both control anchors and real memory systems."
                ),
            )
        )
    return actions


def _next_action_record(
    item_id: str,
    *,
    blocker_status: str | None,
    kind: str,
    **payload: Any,
) -> dict[str, Any]:
    action = {
        "item_id": item_id,
        "blocker": item_id,
        "kind": kind,
        "status": blocker_status or "incomplete",
    }
    action.update(payload)
    return action


def _human_handoff_packets(human_progress: dict[str, Any]) -> list[str]:
    packets: list[str] = []
    annotator_packets = human_progress.get("annotator_packets")
    if isinstance(annotator_packets, dict):
        for annotator_id, packet in sorted(annotator_packets.items()):
            if not isinstance(packet, dict):
                continue
            archive_file = packet.get("archive_file")
            packet_dir = packet.get("packet_dir")
            if isinstance(archive_file, str) and archive_file:
                packets.append(f"{annotator_id}:{archive_file}")
            elif isinstance(packet_dir, str) and packet_dir:
                packets.append(f"{annotator_id}:{packet_dir}")
    adjudication_packet = human_progress.get("adjudication_packet")
    if isinstance(adjudication_packet, dict):
        archive_file = adjudication_packet.get("archive_file")
        packet_dir = adjudication_packet.get("packet_dir")
        if isinstance(archive_file, str) and archive_file:
            packets.append(f"adjudication:{archive_file}")
        elif isinstance(packet_dir, str) and packet_dir:
            packets.append(f"adjudication:{packet_dir}")
    return packets


def _human_handoff_packet_files(human_progress: dict[str, Any], bundle_dir: Path) -> dict[str, str]:
    packet_files: dict[str, str] = {}
    annotator_packets = human_progress.get("annotator_packets")
    if isinstance(annotator_packets, dict):
        for annotator_id, packet in sorted(annotator_packets.items()):
            if not isinstance(packet, dict):
                continue
            archive_file = packet.get("archive_file")
            packet_dir = packet.get("packet_dir")
            if isinstance(archive_file, str) and archive_file:
                packet_files[str(annotator_id)] = str(_resolve_bundle_relative_path(bundle_dir, archive_file))
            elif isinstance(packet_dir, str) and packet_dir:
                packet_files[str(annotator_id)] = str(_resolve_bundle_relative_path(bundle_dir, packet_dir))
    adjudication_packet = human_progress.get("adjudication_packet")
    if isinstance(adjudication_packet, dict):
        archive_file = adjudication_packet.get("archive_file")
        packet_dir = adjudication_packet.get("packet_dir")
        if isinstance(archive_file, str) and archive_file:
            packet_files["adjudication"] = str(_resolve_bundle_relative_path(bundle_dir, archive_file))
        elif isinstance(packet_dir, str) and packet_dir:
            packet_files["adjudication"] = str(_resolve_bundle_relative_path(bundle_dir, packet_dir))
    return packet_files
