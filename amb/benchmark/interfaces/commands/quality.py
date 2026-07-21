"""Quality, validation, and annotation CLI commands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from amb.benchmark.analysis import write_external_canonical_refresh
from amb.benchmark.artifact_contract import localize_report_contract
from amb.benchmark.evaluation.report import print_validation
from amb.benchmark.evaluation.scoring import DEFAULT_RETRIEVAL_K
from amb.benchmark.interfaces.commands.common import format_metric
from amb.benchmark.quality.annotation import (
    apply_human_audit_annotator_packet,
    apply_human_audit_annotation_sheet,
    compute_agreement,
    load_audit_annotations,
    merge_completed_human_audit_tasks,
    summarize_human_audit_progress,
    summarize_audit_templates,
    write_human_audit_annotation_sheets,
    write_double_annotation_task_package,
)
from amb.benchmark.quality.audit import audit_benchmark
from amb.benchmark.quality.completion_audit import write_completion_audit
from amb.benchmark.quality.completion_audit import write_git_hygiene_report
from amb.benchmark.quality.completion_audit import write_git_hygiene_plan
from amb.benchmark.quality.difficulty_calibration import write_difficulty_calibration_audit
from amb.benchmark.quality.domain_construct_validity import write_domain_construct_validity_audit
from amb.benchmark.quality.evidence_readiness import write_evidence_readiness_report
from amb.benchmark.quality.foundation_validation import write_foundation_protocol_audit
from amb.benchmark.quality.human_audit import finalize_human_audit_manifest
from amb.benchmark.quality.human_audit import apply_human_audit_adjudication_packet
from amb.benchmark.quality.human_audit import build_human_audit_adjudication_package
from amb.benchmark.quality.human_audit import build_human_audit_attestation
from amb.benchmark.quality.human_audit import build_human_audit_evidence_bundle
from amb.benchmark.quality.human_audit import build_human_audit_sandbox_sample
from amb.benchmark.quality.human_audit import ingest_human_audit_return_packets
from amb.benchmark.quality.human_audit import summarize_human_audit_rejected_returns
from amb.benchmark.quality.human_audit import sync_human_audit_return_inbox
from amb.benchmark.quality.human_audit import watch_human_audit_return_inbox
from amb.benchmark.quality.human_audit import reconcile_human_audit_evidence_bundle
from amb.benchmark.quality.human_audit import summarize_human_audit_disagreements
from amb.benchmark.quality.human_audit import verify_human_audit_evidence_bundle
from amb.benchmark.quality.human_audit import verify_manifest_human_audit
from amb.benchmark.quality.lineage import write_lineage_audit
from amb.benchmark.quality.main_dataset_acceptance import write_canonical_main_acceptance_current
from amb.benchmark.quality.main_dataset_acceptance import write_canonical_challenge_release_currents
from amb.benchmark.quality.main_dataset_acceptance import write_canonical_challenge_acceptance_current
from amb.benchmark.quality.main_dataset_acceptance import write_canonical_hidden_quarterly_release_currents
from amb.benchmark.quality.main_dataset_acceptance import write_canonical_hidden_quarterly_acceptance_current
from amb.benchmark.quality.main_dataset_acceptance import write_canonical_main_quality_audits_current
from amb.benchmark.quality.main_dataset_acceptance import write_canonical_strict_main_acceptance_current
from amb.benchmark.quality.main_dataset_acceptance import write_canonical_strict_main_quality_audits_current
from amb.benchmark.quality.main_dataset_acceptance import write_canonical_strict_main_release_support_currents
from amb.benchmark.quality.main_dataset_acceptance import write_challenge_release_acceptance
from amb.benchmark.quality.main_dataset_acceptance import write_hidden_quarterly_release_acceptance
from amb.benchmark.quality.main_dataset_acceptance import write_main_dataset_acceptance
from amb.benchmark.quality.objective_checklist import write_objective_checklist
from amb.benchmark.quality.probe_discriminativeness import write_probe_discriminativeness_audit
from amb.benchmark.quality.question_craftsmanship import write_question_craftsmanship_audit
from amb.benchmark.quality.query_construction import write_query_construction_audit
from amb.benchmark.quality.release_intrinsic_sanity import write_release_intrinsic_sanity
from amb.benchmark.quality.release_validation import validate_release_artifacts, write_release_validation
from amb.benchmark.quality.real_system import (
    backfill_real_system_run_metadata,
    default_real_system_config_validation_output,
    default_real_system_analysis_output,
    finalize_real_system_run,
    write_real_system_canonical_refresh,
    write_merged_real_system_matrix_summary,
    write_real_system_matrix_validation,
    write_real_system_run_progress,
)
from amb.benchmark.quality.strong_ai_audit import write_strong_ai_audit
from amb.benchmark.quality.strong_ai_audit import write_strong_ai_audit_ledger
from amb.benchmark.quality.validation import validate_benchmark, validate_predictions
from amb.benchmark.schemas.io import load_benchmark, load_predictions, read_json, write_json


WATCH_REAL_SYSTEM_CANONICAL_LOCK_SCHEMA = "amst-watch-real-system-canonical-lock-v1"


def _default_watch_real_system_canonical_lock_file(project_root: Path) -> Path:
    return project_root / "reports" / "real_system_runs" / "canonical_watch_current.lock"


def _watch_process_cmdline(pid: int) -> str | None:
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        raw = proc_cmdline.read_bytes()
    except OSError:
        return None
    if not raw:
        return None
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _watch_process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_watch_real_system_canonical_lock(lock_file: Path, *, project_root: Path) -> dict[str, object]:
    payload = {
        "schema_version": WATCH_REAL_SYSTEM_CANONICAL_LOCK_SCHEMA,
        "pid": os.getpid(),
        "root": str(project_root.resolve()),
        "command": "watch-real-system-canonical",
        "cmdline": _watch_process_cmdline(os.getpid()),
    }
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(lock_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            try:
                existing = read_json(lock_file)
            except Exception:
                existing = {}
            owner_pid = int(existing.get("pid", 0) or 0)
            owner_cmdline = _watch_process_cmdline(owner_pid) if owner_pid > 0 else None
            if owner_pid > 0 and _watch_process_is_running(owner_pid) and owner_pid != os.getpid():
                owner_desc = owner_cmdline or str(existing.get("cmdline") or f"pid {owner_pid}")
                raise RuntimeError(
                    f"watch-real-system-canonical already running under pid {owner_pid}; "
                    f"lock file: {lock_file}; owner: {owner_desc}"
                )
            try:
                lock_file.unlink()
            except FileNotFoundError:
                pass
            continue
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
                fh.write("\n")
        except Exception:
            try:
                lock_file.unlink()
            except FileNotFoundError:
                pass
            raise
        return payload


def _release_watch_real_system_canonical_lock(lock_file: Path) -> None:
    try:
        existing = read_json(lock_file)
    except Exception:
        existing = None
    owner_pid = int(existing.get("pid", 0) or 0) if isinstance(existing, dict) else 0
    if owner_pid and owner_pid != os.getpid():
        return
    try:
        lock_file.unlink()
    except FileNotFoundError:
        pass


def register_quality_commands(subparsers: argparse._SubParsersAction) -> None:
    validate = subparsers.add_parser("validate", help="Validate a benchmark and optional prediction file")
    validate.add_argument("--benchmark", required=True)
    validate.add_argument("--predictions")
    validate.set_defaults(handler=cmd_validate)

    audit = subparsers.add_parser("audit", help="Audit benchmark coverage for lifecycle and safety gates")
    audit.add_argument("--benchmark", required=True)
    audit.add_argument("--output")
    audit.set_defaults(handler=cmd_audit)

    lineage = subparsers.add_parser("lineage-audit", help="Audit query-to-state-memory-event-turn lineage")
    lineage.add_argument("--benchmark")
    lineage.add_argument("--manifest")
    lineage.add_argument("--split", action="append")
    lineage.add_argument("--output", required=True)
    lineage.set_defaults(handler=cmd_lineage_audit)

    query_construction = subparsers.add_parser(
        "query-construction-audit",
        help="Audit query construction quality, counterfactual comparability, and evidence minimality",
    )
    query_construction.add_argument("--benchmark")
    query_construction.add_argument("--manifest")
    query_construction.add_argument("--split")
    query_construction.add_argument("--output", required=True)
    query_construction.set_defaults(handler=cmd_query_construction_audit)

    question_craftsmanship = subparsers.add_parser(
        "question-craftsmanship-audit",
        help="Audit per-question probe blueprints, evidence roles, and standard-answer tightness",
    )
    question_craftsmanship.add_argument("--benchmark")
    question_craftsmanship.add_argument("--manifest")
    question_craftsmanship.add_argument("--split")
    question_craftsmanship.add_argument("--output", required=True)
    question_craftsmanship.set_defaults(handler=cmd_question_craftsmanship_audit)

    probe_discriminativeness = subparsers.add_parser(
        "probe-discriminativeness-audit",
        help="Audit whether representative probes separate no-memory, full-history, graph-memory, and oracle baselines",
    )
    probe_discriminativeness.add_argument("--manifest", required=True)
    probe_discriminativeness.add_argument("--split", default="public_dev")
    probe_discriminativeness.add_argument("--reports-dir", default="reports/examples")
    probe_discriminativeness.add_argument("--output", required=True)
    probe_discriminativeness.set_defaults(handler=cmd_probe_discriminativeness_audit)

    difficulty_calibration = subparsers.add_parser(
        "difficulty-calibration-audit",
        help="Audit whether easy/medium/hard query buckets produce meaningful representative-baseline separation",
    )
    difficulty_calibration.add_argument("--manifest", required=True)
    difficulty_calibration.add_argument("--split", default="public_dev")
    difficulty_calibration.add_argument("--reports-dir", default="reports/examples")
    difficulty_calibration.add_argument("--output", required=True)
    difficulty_calibration.set_defaults(handler=cmd_difficulty_calibration_audit)

    domain_construct_validity = subparsers.add_parser(
        "domain-construct-validity-audit",
        help="Audit whether each domain contains solvable memory-required slices, nontrivial control slices, and stable baseline gaps",
    )
    domain_construct_validity.add_argument("--manifest", required=True)
    domain_construct_validity.add_argument("--split", default="public_dev")
    domain_construct_validity.add_argument("--reports-dir", default="reports/examples")
    domain_construct_validity.add_argument("--output", required=True)
    domain_construct_validity.set_defaults(handler=cmd_domain_construct_validity_audit)

    foundation_validation = subparsers.add_parser(
        "foundation-validation-audit",
        help="Validate same-model query_only/full_history/oracle_state protocol sensitivity from scored real-model reports",
    )
    foundation_validation.add_argument("--reports", nargs="+", required=True)
    foundation_validation.add_argument("--output", required=True)
    foundation_validation.add_argument("--expected-benchmark-id")
    foundation_validation.add_argument("--cohort-id")
    foundation_validation.add_argument("--require-full-history", action="store_true")
    foundation_validation.set_defaults(handler=cmd_foundation_validation_audit)

    validate_release = subparsers.add_parser("validate-release", help="Validate a release manifest and referenced artifacts")
    validate_release.add_argument("--manifest", required=True)
    validate_release.add_argument("--output")
    validate_release.set_defaults(handler=cmd_validate_release)

    main_acceptance = subparsers.add_parser(
        "main-dataset-acceptance",
        help="Validate main-v1 dataset construction scale, split, coverage, and quality evidence",
    )
    main_acceptance.add_argument("--manifest", required=True)
    main_acceptance.add_argument("--output", required=True)
    main_acceptance.add_argument("--expected-benchmark-id", default="amst-main-v1")
    main_acceptance.add_argument("--expected-profile-id", default="main-v1")
    main_acceptance.add_argument("--run-release-validation", action="store_true")
    main_acceptance.add_argument("--release-validation-report")
    main_acceptance.add_argument("--public-release-manifest")
    main_acceptance.add_argument("--public-release-validation-report")
    main_acceptance.add_argument("--require-public-release-validation", action="store_true")
    main_acceptance.add_argument("--require-all-counterfactual-axes", action="store_true")
    main_acceptance.add_argument("--representative-reports-dir")
    main_acceptance.add_argument("--representative-split", default="public_dev")
    main_acceptance.add_argument("--require-representative-baselines", action="store_true")
    main_acceptance.add_argument("--intrinsic-sanity-report")
    main_acceptance.add_argument("--require-intrinsic-sanity", action="store_true")
    main_acceptance.add_argument("--human-audit-verification-report")
    main_acceptance.add_argument("--require-completed-human-audit", action="store_true")
    main_acceptance.add_argument("--strong-ai-audit-report")
    main_acceptance.add_argument("--require-strong-ai-audit", action="store_true")
    main_acceptance.add_argument("--public-test-sanity-report")
    main_acceptance.add_argument("--require-public-test-sanity", action="store_true")
    main_acceptance.add_argument("--public-result-slices-report")
    main_acceptance.add_argument("--require-public-result-slices", action="store_true")
    main_acceptance.add_argument("--hidden-test-sanity-report")
    main_acceptance.add_argument("--require-hidden-test-sanity", action="store_true")
    main_acceptance.add_argument("--question-craftsmanship-report")
    main_acceptance.add_argument("--require-question-craftsmanship", action="store_true")
    main_acceptance.add_argument("--query-construction-report")
    main_acceptance.add_argument("--require-query-construction", action="store_true")
    main_acceptance.add_argument("--probe-discriminativeness-report")
    main_acceptance.add_argument("--require-probe-discriminativeness", action="store_true")
    main_acceptance.add_argument("--difficulty-calibration-report")
    main_acceptance.add_argument("--require-difficulty-calibration", action="store_true")
    main_acceptance.add_argument("--domain-construct-validity-report")
    main_acceptance.add_argument("--require-domain-construct-validity", action="store_true")
    main_acceptance.add_argument("--foundation-validation-report")
    main_acceptance.add_argument("--require-foundation-validation", action="store_true")
    main_acceptance.add_argument("--require-query-difficulty", action="store_true")
    main_acceptance.set_defaults(handler=cmd_main_dataset_acceptance)

    challenge_acceptance = subparsers.add_parser(
        "challenge-release-acceptance",
        help="Validate challenge-v1 release scale, challenge axes, representative baselines, and question-quality evidence",
    )
    challenge_acceptance.add_argument("--manifest", required=True)
    challenge_acceptance.add_argument("--output", required=True)
    challenge_acceptance.add_argument("--public-release-manifest", required=True)
    challenge_acceptance.add_argument("--release-validation-report", required=True)
    challenge_acceptance.add_argument("--public-release-validation-report", required=True)
    challenge_acceptance.add_argument("--representative-reports-dir", required=True)
    challenge_acceptance.add_argument("--question-craftsmanship-report", required=True)
    challenge_acceptance.add_argument("--query-construction-report", required=True)
    challenge_acceptance.add_argument("--probe-discriminativeness-report", required=True)
    challenge_acceptance.add_argument("--difficulty-calibration-report", required=True)
    challenge_acceptance.add_argument("--domain-construct-validity-report", required=True)
    challenge_acceptance.set_defaults(handler=cmd_challenge_release_acceptance)

    hidden_quarterly_acceptance = subparsers.add_parser(
        "hidden-quarterly-acceptance",
        help="Validate quarterly hidden refresh package semantics and hidden-only quality evidence",
    )
    hidden_quarterly_acceptance.add_argument("--manifest", required=True)
    hidden_quarterly_acceptance.add_argument("--output", required=True)
    hidden_quarterly_acceptance.add_argument("--release-validation-report", required=True)
    hidden_quarterly_acceptance.add_argument("--intrinsic-sanity-report", required=True)
    hidden_quarterly_acceptance.add_argument("--question-craftsmanship-report", required=True)
    hidden_quarterly_acceptance.add_argument("--query-construction-report", required=True)
    hidden_quarterly_acceptance.set_defaults(handler=cmd_hidden_quarterly_acceptance)

    intrinsic_sanity = subparsers.add_parser(
        "release-intrinsic-sanity",
        help="Summarize release-level oracle/no-memory intrinsic sanity from shard audit reports",
    )
    intrinsic_sanity.add_argument("--manifest", required=True)
    intrinsic_sanity.add_argument("--output", required=True)
    intrinsic_sanity.set_defaults(handler=cmd_release_intrinsic_sanity)

    agreement = subparsers.add_parser("agreement", help="Compute human audit annotation agreement")
    agreement.add_argument("--annotations", required=True)
    agreement.add_argument("--output", required=True)
    agreement.set_defaults(handler=cmd_agreement)

    prepare_human_audit = subparsers.add_parser("prepare-human-audit", help="Create pending double-annotation JSONL tasks from audit templates")
    prepare_human_audit.add_argument("--manifest")
    prepare_human_audit.add_argument("--templates", nargs="+")
    prepare_human_audit.add_argument("--annotator-id", action="append", dest="annotator_ids", required=True)
    prepare_human_audit.add_argument("--output-dir", required=True)
    prepare_human_audit.add_argument("--summary-output")
    prepare_human_audit.set_defaults(handler=cmd_prepare_human_audit)

    human_audit_progress = subparsers.add_parser(
        "human-audit-progress",
        help="Check double-annotation task coverage before finalizing a human audit",
    )
    human_audit_progress.add_argument("--task-manifest", required=True)
    human_audit_progress.add_argument("--annotations")
    human_audit_progress.add_argument("--output", required=True)
    human_audit_progress.set_defaults(handler=cmd_human_audit_progress)

    strong_ai_audit = subparsers.add_parser(
        "strong-ai-audit",
        help="Run Codex/AI self-audit gates over a human-audit task manifest or release manifest",
    )
    strong_ai_audit.add_argument("--task-manifest")
    strong_ai_audit.add_argument("--manifest")
    strong_ai_audit.add_argument("--expected-rows", type=int)
    strong_ai_audit.add_argument("--output", required=True)
    strong_ai_audit.add_argument("--row-ledger-output")
    strong_ai_audit.set_defaults(handler=cmd_strong_ai_audit)

    export_human_audit_sheets = subparsers.add_parser(
        "export-human-audit-annotation-sheets",
        help="Export per-annotator spreadsheet-friendly CSV sheets from a human-audit task package",
    )
    export_human_audit_sheets.add_argument("--task-manifest", required=True)
    export_human_audit_sheets.add_argument("--output-dir", required=True)
    export_human_audit_sheets.set_defaults(handler=cmd_export_human_audit_annotation_sheets)

    apply_human_audit_sheet = subparsers.add_parser(
        "apply-human-audit-annotation-sheet",
        help="Apply an edited annotator CSV sheet back into the canonical JSONL task file",
    )
    apply_human_audit_sheet.add_argument("--task-manifest", required=True)
    apply_human_audit_sheet.add_argument("--annotator-id", required=True)
    apply_human_audit_sheet.add_argument("--sheet", required=True)
    apply_human_audit_sheet.set_defaults(handler=cmd_apply_human_audit_annotation_sheet)

    apply_human_audit_packet = subparsers.add_parser(
        "apply-human-audit-annotator-packet",
        help="Apply a returned annotator packet directory or zip archive back into the canonical JSONL task file",
    )
    apply_human_audit_packet.add_argument("--task-manifest", required=True)
    apply_human_audit_packet.add_argument("--packet", required=True)
    apply_human_audit_packet.add_argument("--annotator-id")
    apply_human_audit_packet.set_defaults(handler=cmd_apply_human_audit_annotator_packet)

    apply_human_audit_adjudication = subparsers.add_parser(
        "apply-human-audit-adjudication-packet",
        help="Apply a returned adjudication packet directory or zip archive back into a human-audit bundle",
    )
    apply_human_audit_adjudication.add_argument("--bundle-dir", required=True)
    apply_human_audit_adjudication.add_argument("--packet", required=True)
    apply_human_audit_adjudication.add_argument("--adjudicator-id")
    apply_human_audit_adjudication.add_argument("--reconcile-when-ready", action="store_true")
    apply_human_audit_adjudication.add_argument("--signed-at")
    apply_human_audit_adjudication.add_argument("--annotation-guideline")
    apply_human_audit_adjudication.add_argument("--adjudication-policy")
    apply_human_audit_adjudication.add_argument("--output")
    apply_human_audit_adjudication.set_defaults(handler=cmd_apply_human_audit_adjudication_packet)

    ingest_human_audit_packets = subparsers.add_parser(
        "ingest-human-audit-return-packets",
        help="Apply one or more returned annotator packets into a human-audit bundle and optionally auto-reconcile when ready",
    )
    ingest_human_audit_packets.add_argument("--bundle-dir", required=True)
    ingest_human_audit_packets.add_argument("--packets", nargs="+", required=True)
    ingest_human_audit_packets.add_argument("--reconcile-when-ready", action="store_true")
    ingest_human_audit_packets.add_argument("--signed-at")
    ingest_human_audit_packets.add_argument("--annotation-guideline")
    ingest_human_audit_packets.add_argument("--adjudication-policy")
    ingest_human_audit_packets.add_argument("--output")
    ingest_human_audit_packets.set_defaults(handler=cmd_ingest_human_audit_return_packets)

    sync_human_audit_inbox = subparsers.add_parser(
        "sync-human-audit-return-inbox",
        help="Discover returned human-audit packets from inbox directories and sync them into a bundle",
    )
    sync_human_audit_inbox.add_argument("--bundle-dir", required=True)
    sync_human_audit_inbox.add_argument("--annotator-inbox", action="append", default=[])
    sync_human_audit_inbox.add_argument("--adjudication-inbox", action="append", default=[])
    sync_human_audit_inbox.add_argument("--reconcile-when-ready", action="store_true")
    sync_human_audit_inbox.add_argument("--signed-at")
    sync_human_audit_inbox.add_argument("--annotation-guideline")
    sync_human_audit_inbox.add_argument("--adjudication-policy")
    sync_human_audit_inbox.add_argument("--output")
    sync_human_audit_inbox.set_defaults(handler=cmd_sync_human_audit_return_inbox)

    watch_human_audit_inbox = subparsers.add_parser(
        "watch-human-audit-return-inbox",
        help="Repeatedly sync returned human-audit packets from bundle inbox directories and keep one watch summary current",
    )
    watch_human_audit_inbox.add_argument("--bundle-dir", required=True)
    watch_human_audit_inbox.add_argument("--annotator-inbox", action="append", default=[])
    watch_human_audit_inbox.add_argument("--adjudication-inbox", action="append", default=[])
    watch_human_audit_inbox.add_argument("--reconcile-when-ready", action="store_true")
    watch_human_audit_inbox.add_argument("--signed-at")
    watch_human_audit_inbox.add_argument("--annotation-guideline")
    watch_human_audit_inbox.add_argument("--adjudication-policy")
    watch_human_audit_inbox.add_argument("--interval-s", type=float, default=60.0)
    watch_human_audit_inbox.add_argument("--max-iterations", type=int, default=1)
    watch_human_audit_inbox.add_argument("--stop-when-ready", action="store_true")
    watch_human_audit_inbox.add_argument("--stop-when-rejected", action="store_true")
    watch_human_audit_inbox.add_argument("--output")
    watch_human_audit_inbox.set_defaults(handler=cmd_watch_human_audit_return_inbox)

    verify_human_audit = subparsers.add_parser(
        "verify-manifest-human-audit",
        help="Verify that a manifest's completed human audit is reproducible and write a machine-readable report",
    )
    verify_human_audit.add_argument("--manifest", required=True)
    verify_human_audit.add_argument("--output", required=True)
    verify_human_audit.set_defaults(handler=cmd_verify_manifest_human_audit)

    merge_human_audit = subparsers.add_parser(
        "merge-human-audit-annotations",
        help="Merge completed per-annotator task files into one annotation JSONL",
    )
    merge_human_audit.add_argument("--task-manifest", required=True)
    merge_human_audit.add_argument("--output", required=True)
    merge_human_audit.add_argument("--summary-output")
    merge_human_audit.set_defaults(handler=cmd_merge_human_audit_annotations)

    summarize_human_audit_disagreements_cmd = subparsers.add_parser(
        "summarize-human-audit-disagreements",
        help="Summarize item-level disagreement fields from completed double annotations",
    )
    summarize_human_audit_disagreements_cmd.add_argument("--task-manifest", required=True)
    summarize_human_audit_disagreements_cmd.add_argument("--annotations", required=True)
    summarize_human_audit_disagreements_cmd.add_argument("--output", required=True)
    summarize_human_audit_disagreements_cmd.set_defaults(handler=cmd_summarize_human_audit_disagreements)

    build_human_audit_adjudication = subparsers.add_parser(
        "build-human-audit-adjudication-package",
        help="Create adjudication tasks for disagreement items after completed double annotation",
    )
    build_human_audit_adjudication.add_argument("--task-manifest", required=True)
    build_human_audit_adjudication.add_argument("--annotations", required=True)
    build_human_audit_adjudication.add_argument("--output-dir", required=True)
    build_human_audit_adjudication.set_defaults(handler=cmd_build_human_audit_adjudication_package)

    generate_human_audit_attestation = subparsers.add_parser(
        "generate-human-audit-attestation",
        help="Build a human-audit attestation JSON from a ready-to-finalize task manifest and merged annotations",
    )
    generate_human_audit_attestation.add_argument("--task-manifest", required=True)
    generate_human_audit_attestation.add_argument("--annotations", required=True)
    generate_human_audit_attestation.add_argument("--output", required=True)
    generate_human_audit_attestation.add_argument("--signed-at", required=True)
    generate_human_audit_attestation.add_argument("--annotation-guideline", required=True)
    generate_human_audit_attestation.add_argument("--adjudication-policy", required=True)
    generate_human_audit_attestation.set_defaults(handler=cmd_generate_human_audit_attestation)

    bundle_human_audit = subparsers.add_parser(
        "build-human-audit-evidence-bundle",
        help="Create a self-contained human-audit handoff / evidence bundle with copied templates, tasks, and optional evidence files",
    )
    bundle_human_audit.add_argument("--output-dir", required=True)
    bundle_human_audit.add_argument("--manifest")
    bundle_human_audit.add_argument("--task-manifest")
    bundle_human_audit.add_argument("--annotations")
    bundle_human_audit.add_argument("--annotator-attestation")
    bundle_human_audit.add_argument("--adjudication")
    bundle_human_audit.set_defaults(handler=cmd_build_human_audit_evidence_bundle)

    verify_bundle_human_audit = subparsers.add_parser(
        "verify-human-audit-evidence-bundle",
        help="Verify that a human-audit evidence bundle is self-consistent and reproducible",
    )
    verify_bundle_human_audit.add_argument("--bundle-dir", required=True)
    verify_bundle_human_audit.add_argument("--output", required=True)
    verify_bundle_human_audit.set_defaults(handler=cmd_verify_human_audit_evidence_bundle)

    summarize_rejected_returns_human_audit = subparsers.add_parser(
        "summarize-human-audit-rejected-returns",
        help="Summarize currently quarantined invalid returned packets from a human-audit bundle",
    )
    summarize_rejected_returns_human_audit.add_argument("--bundle-dir", required=True)
    summarize_rejected_returns_human_audit.add_argument("--output", required=True)
    summarize_rejected_returns_human_audit.set_defaults(handler=cmd_summarize_human_audit_rejected_returns)

    reconcile_bundle_human_audit = subparsers.add_parser(
        "reconcile-human-audit-evidence-bundle",
        help="Sync a completed human-audit evidence bundle back into its canonical source manifest and task package",
    )
    reconcile_bundle_human_audit.add_argument("--bundle-dir", required=True)
    reconcile_bundle_human_audit.add_argument("--source-manifest")
    reconcile_bundle_human_audit.add_argument("--source-task-manifest")
    reconcile_bundle_human_audit.add_argument("--source-annotations-output")
    reconcile_bundle_human_audit.add_argument("--source-attestation-output")
    reconcile_bundle_human_audit.add_argument("--source-adjudication-output")
    reconcile_bundle_human_audit.add_argument("--source-agreement-output")
    reconcile_bundle_human_audit.add_argument("--signed-at")
    reconcile_bundle_human_audit.add_argument("--annotation-guideline")
    reconcile_bundle_human_audit.add_argument("--adjudication-policy")
    reconcile_bundle_human_audit.add_argument("--output")
    reconcile_bundle_human_audit.set_defaults(handler=cmd_reconcile_human_audit_evidence_bundle)

    sandbox_human_audit = subparsers.add_parser(
        "build-human-audit-sandbox-sample",
        help="Create a finalize-ready sandbox sample from a real human-audit task package without mutating the release manifest",
    )
    sandbox_human_audit.add_argument("--task-manifest", required=True)
    sandbox_human_audit.add_argument("--output-dir", required=True)
    sandbox_human_audit.add_argument("--num-items", type=int, default=2)
    sandbox_human_audit.add_argument("--signed-at", required=True)
    sandbox_human_audit.add_argument("--annotation-guideline", required=True)
    sandbox_human_audit.add_argument("--adjudication-policy", required=True)
    sandbox_human_audit.set_defaults(handler=cmd_build_human_audit_sandbox_sample)

    finalize_human_audit = subparsers.add_parser("finalize-human-audit", help="Verify annotations and mark a release human audit completed")
    finalize_human_audit.add_argument("--manifest", required=True)
    finalize_human_audit.add_argument("--annotations", required=True)
    finalize_human_audit.add_argument("--task-manifest", required=True)
    finalize_human_audit.add_argument("--annotator-attestation", required=True)
    finalize_human_audit.add_argument("--adjudication")
    finalize_human_audit.add_argument("--agreement-output")
    finalize_human_audit.set_defaults(handler=cmd_finalize_human_audit)

    template_summary = subparsers.add_parser("audit-template-summary", help="Summarize generated audit annotation templates")
    template_summary.add_argument("--templates", nargs="+", required=True)
    template_summary.add_argument("--output", required=True)
    template_summary.set_defaults(handler=cmd_audit_template_summary)

    completion_audit = subparsers.add_parser("completion-audit", help="Audit current artifacts against the final AutoMemoryBench plan")
    completion_audit.add_argument("--root", default=".")
    completion_audit.add_argument("--output", required=True)
    completion_audit.set_defaults(handler=cmd_completion_audit)

    git_hygiene = subparsers.add_parser(
        "git-hygiene-report",
        help="Write a machine-readable report for critical tracked/untracked benchmark assets and import edges",
    )
    git_hygiene.add_argument("--root", default=".")
    git_hygiene.add_argument("--output", required=True)
    git_hygiene.set_defaults(handler=cmd_git_hygiene_report)

    git_hygiene_plan = subparsers.add_parser(
        "build-git-hygiene-plan",
        help="Materialize per-batch pathspec files and stage scripts for the git-hygiene blocker",
    )
    git_hygiene_plan.add_argument("--root", default=".")
    git_hygiene_plan.add_argument("--output-dir", required=True)
    git_hygiene_plan.set_defaults(handler=cmd_git_hygiene_plan)

    objective_checklist = subparsers.add_parser(
        "objective-checklist",
        help="Map the AutoMemoryBench implementation objective to concrete evidence artifacts",
    )
    objective_checklist.add_argument("--root", default=".")
    objective_checklist.add_argument("--output", required=True)
    objective_checklist.set_defaults(handler=cmd_objective_checklist)

    evidence_readiness = subparsers.add_parser(
        "evidence-readiness",
        help="Summarize remaining evidence needed for AutoMemoryBench completion",
    )
    evidence_readiness.add_argument("--root", default=".")
    evidence_readiness.add_argument("--output", required=True)
    evidence_readiness.add_argument("--external-output-dir", default="reports/external")
    evidence_readiness.add_argument("--human-task-manifest")
    evidence_readiness.add_argument("--human-annotations")
    evidence_readiness.add_argument("--real-system-summary")
    evidence_readiness.add_argument("--real-system-sample-validation")
    evidence_readiness.add_argument("--real-system-sample-summary")
    evidence_readiness.add_argument("--integration-configs", nargs="+")
    evidence_readiness.add_argument("--integration-config-validation-report")
    evidence_readiness.set_defaults(handler=cmd_evidence_readiness)

    real_system_matrix = subparsers.add_parser(
        "validate-real-system-matrix",
        help="Validate real external memory-system matrix evidence",
    )
    real_system_matrix.add_argument("--summary", required=True)
    real_system_matrix.add_argument("--output", required=True)
    real_system_matrix.add_argument("--expected-benchmark-id")
    real_system_matrix.add_argument("--expected-release-split")
    real_system_matrix.set_defaults(handler=cmd_validate_real_system_matrix)

    merge_real_system_matrix = subparsers.add_parser(
        "merge-real-system-matrix",
        help="Merge one or more real-system matrix summaries into one canonical summary",
    )
    merge_real_system_matrix.add_argument("--summaries", nargs="+", required=True)
    merge_real_system_matrix.add_argument("--output", required=True)
    merge_real_system_matrix.add_argument("--expected-benchmark-id")
    merge_real_system_matrix.add_argument("--expected-release-split")
    merge_real_system_matrix.set_defaults(handler=cmd_merge_real_system_matrix)

    finalize_real_system_run_cmd = subparsers.add_parser(
        "finalize-real-system-run",
        help="Finalize one completed resumable real-system run into report.json and a one-system matrix summary",
    )
    finalize_real_system_run_cmd.add_argument("--manifest", required=True)
    finalize_real_system_run_cmd.add_argument("--split", required=True, choices=("public_dev", "public_test", "audit_subset", "hidden_test"))
    finalize_real_system_run_cmd.add_argument("--run-dir", required=True)
    finalize_real_system_run_cmd.add_argument("--summary-output")
    finalize_real_system_run_cmd.add_argument("--retrieval-k", type=int, default=DEFAULT_RETRIEVAL_K)
    finalize_real_system_run_cmd.set_defaults(handler=cmd_finalize_real_system_run)

    backfill_real_system_metadata_cmd = subparsers.add_parser(
        "backfill-real-system-run-metadata",
        help="Backfill run_metadata.json for an existing real-system run directory so it can be finalized later",
    )
    backfill_real_system_metadata_cmd.add_argument("--manifest", required=True)
    backfill_real_system_metadata_cmd.add_argument("--split", required=True, choices=("public_dev", "public_test", "audit_subset", "hidden_test"))
    backfill_real_system_metadata_cmd.add_argument("--config", required=True)
    backfill_real_system_metadata_cmd.add_argument("--run-dir", required=True)
    backfill_real_system_metadata_cmd.add_argument("--system-id")
    backfill_real_system_metadata_cmd.add_argument("--system-version")
    backfill_real_system_metadata_cmd.add_argument("--retrieval-k", type=int, default=DEFAULT_RETRIEVAL_K)
    backfill_real_system_metadata_cmd.add_argument("--command")
    backfill_real_system_metadata_cmd.add_argument("--no-resume", action="store_true")
    backfill_real_system_metadata_cmd.add_argument("--overwrite", action="store_true")
    backfill_real_system_metadata_cmd.set_defaults(handler=cmd_backfill_real_system_run_metadata)

    refresh_real_system_canonical_cmd = subparsers.add_parser(
        "refresh-real-system-canonical",
        help="Refresh provider run progress, backfill/finalize when possible, and materialize the best available canonical matrix view",
    )
    refresh_real_system_canonical_cmd.add_argument("--spec")
    refresh_real_system_canonical_cmd.add_argument("--manifest")
    refresh_real_system_canonical_cmd.add_argument("--split", choices=("public_dev", "public_test", "audit_subset", "hidden_test"))
    refresh_real_system_canonical_cmd.add_argument("--run", nargs=2, action="append", metavar=("CONFIG", "RUN_DIR"))
    refresh_real_system_canonical_cmd.add_argument("--output")
    refresh_real_system_canonical_cmd.add_argument("--config-validation-output")
    refresh_real_system_canonical_cmd.add_argument("--analysis-output")
    refresh_real_system_canonical_cmd.add_argument("--merged-summary-output")
    refresh_real_system_canonical_cmd.add_argument("--merged-validation-output")
    refresh_real_system_canonical_cmd.add_argument("--expected-benchmark-id")
    refresh_real_system_canonical_cmd.add_argument("--expected-release-split")
    refresh_real_system_canonical_cmd.add_argument("--retrieval-k", type=int)
    refresh_real_system_canonical_cmd.set_defaults(handler=cmd_refresh_real_system_canonical)

    watch_real_system_canonical_cmd = subparsers.add_parser(
        "watch-real-system-canonical",
        help="Repeatedly refresh canonical real-system artifacts and keep readiness/audit outputs current",
    )
    watch_real_system_canonical_cmd.add_argument("--spec")
    watch_real_system_canonical_cmd.add_argument("--manifest")
    watch_real_system_canonical_cmd.add_argument("--split", choices=("public_dev", "public_test", "audit_subset", "hidden_test"))
    watch_real_system_canonical_cmd.add_argument("--run", nargs=2, action="append", metavar=("CONFIG", "RUN_DIR"))
    watch_real_system_canonical_cmd.add_argument("--output")
    watch_real_system_canonical_cmd.add_argument("--config-validation-output")
    watch_real_system_canonical_cmd.add_argument("--analysis-output")
    watch_real_system_canonical_cmd.add_argument("--merged-summary-output")
    watch_real_system_canonical_cmd.add_argument("--merged-validation-output")
    watch_real_system_canonical_cmd.add_argument("--expected-benchmark-id")
    watch_real_system_canonical_cmd.add_argument("--expected-release-split")
    watch_real_system_canonical_cmd.add_argument("--retrieval-k", type=int)
    watch_real_system_canonical_cmd.add_argument("--root", default=".")
    watch_real_system_canonical_cmd.add_argument("--evidence-readiness-output")
    watch_real_system_canonical_cmd.add_argument("--completion-audit-output")
    watch_real_system_canonical_cmd.add_argument("--objective-checklist-output")
    watch_real_system_canonical_cmd.add_argument("--git-hygiene-output")
    watch_real_system_canonical_cmd.add_argument("--git-hygiene-plan-output-dir")
    watch_real_system_canonical_cmd.add_argument("--external-output-dir")
    watch_real_system_canonical_cmd.add_argument("--lock-file")
    watch_real_system_canonical_cmd.add_argument("--interval-s", type=float, default=60.0)
    watch_real_system_canonical_cmd.add_argument("--max-iterations", type=int, default=1)
    watch_real_system_canonical_cmd.add_argument("--stop-when-passed", action="store_true")
    watch_real_system_canonical_cmd.set_defaults(handler=cmd_watch_real_system_canonical)

    real_system_run_progress = subparsers.add_parser(
        "real-system-run-progress",
        help="Summarize progress for one resumable real-system run directory",
    )
    real_system_run_progress.add_argument("--manifest", required=True)
    real_system_run_progress.add_argument("--split", required=True, choices=("public_dev", "public_test", "audit_subset", "hidden_test"))
    real_system_run_progress.add_argument("--run-dir", required=True)
    real_system_run_progress.add_argument("--output", required=True)
    real_system_run_progress.set_defaults(handler=cmd_real_system_run_progress)


def cmd_validate(args: argparse.Namespace) -> None:
    benchmark = load_benchmark(args.benchmark)
    result = validate_benchmark(benchmark)
    print(print_validation(result.errors, result.warnings))
    if args.predictions:
        predictions = load_predictions(args.predictions)
        pred_result = validate_predictions(predictions, benchmark)
        print(print_validation(pred_result.errors, pred_result.warnings))
        if pred_result.errors:
            raise SystemExit(1)
    if result.errors:
        raise SystemExit(1)


def cmd_audit(args: argparse.Namespace) -> None:
    benchmark = load_benchmark(args.benchmark)
    result = validate_benchmark(benchmark)
    if result.errors:
        print(print_validation(result.errors, result.warnings))
        raise SystemExit(1)
    audit_report = audit_benchmark(benchmark)
    if args.output:
        audit_report = localize_report_contract(
            audit_report,
            output_path=args.output,
            project_root_hints=(args.benchmark,),
        )
        write_json(args.output, audit_report)
    print(f"benchmark_id: {audit_report['benchmark_id']}")
    print(f"num_cases: {audit_report['num_cases']}")
    print(f"num_queries: {audit_report['num_queries']}")
    print(f"num_memories: {audit_report['num_memories']}")
    print(f"quality_gates_passed: {audit_report['quality_gates_passed']}")
    print(f"construction_gates_passed: {audit_report['construction_gates_passed']}")
    print(f"data_quality_gates_passed: {audit_report['data_quality_gates_passed']}")


def cmd_lineage_audit(args: argparse.Namespace) -> None:
    try:
        report = write_lineage_audit(
            args.output,
            benchmark_path=args.benchmark,
            manifest_path=args.manifest,
            splits=args.split,
        )
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    print(f"status: {report['status']}")
    print(f"cases: {report['summary']['num_cases']}")
    print(f"queries: {report['summary']['num_queries']}")
    print(f"issues: {report['summary']['num_issues']}")
    print(f"output: {args.output}")
    if report["status"] != "passed":
        raise SystemExit(1)


def cmd_question_craftsmanship_audit(args: argparse.Namespace) -> None:
    try:
        report = write_question_craftsmanship_audit(
            args.output,
            benchmark_path=args.benchmark,
            manifest_path=args.manifest,
            split=args.split,
        )
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    print(f"status: {report['status']}")
    print(f"cases: {report['summary']['num_cases']}")
    print(f"queries: {report['summary']['num_queries']}")
    print(f"issues: {report['summary']['num_issues']}")
    print(f"output: {args.output}")
    if report["status"] != "passed":
        raise SystemExit(1)


def cmd_query_construction_audit(args: argparse.Namespace) -> None:
    try:
        report = write_query_construction_audit(
            args.output,
            benchmark_path=args.benchmark,
            manifest_path=args.manifest,
            split=args.split,
        )
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    print(f"status: {report['status']}")
    print(f"benchmark_id: {report['benchmark_id']}")
    print(f"queries: {report['summary']['num_queries']}")
    print(f"counterfactual_groups: {report['summary']['num_counterfactual_groups']}")
    print(f"issues: {report['summary']['num_total_issues']}")
    print(f"output: {args.output}")
    if report["status"] != "passed":
        raise SystemExit(1)


def cmd_probe_discriminativeness_audit(args: argparse.Namespace) -> None:
    report = write_probe_discriminativeness_audit(
        args.output,
        manifest_path=args.manifest,
        split=args.split,
        reports_dir=args.reports_dir,
    )
    print(f"status: {report['status']}")
    print(f"core_probe_types: {report['summary']['num_core_probe_types']}")
    print(f"passed_core_probe_types: {report['summary']['num_passed_core_probe_types']}")
    print(f"weak_probe_types: {len(report['weak_probe_types'])}")
    print(f"output: {args.output}")
    if report["status"] != "passed":
        raise SystemExit(1)


def cmd_difficulty_calibration_audit(args: argparse.Namespace) -> None:
    report = write_difficulty_calibration_audit(
        args.output,
        manifest_path=args.manifest,
        split=args.split,
        reports_dir=args.reports_dir,
    )
    print(f"status: {report['status']}")
    print(f"difficulty_levels: {report['summary']['num_difficulty_levels']}")
    print(f"failed_checks: {report['summary']['num_failed_checks']}")
    print(f"output: {args.output}")
    if report["status"] != "passed":
        raise SystemExit(1)


def cmd_domain_construct_validity_audit(args: argparse.Namespace) -> None:
    report = write_domain_construct_validity_audit(
        args.output,
        manifest_path=args.manifest,
        split=args.split,
        reports_dir=args.reports_dir,
    )
    print(f"status: {report['status']}")
    print(f"expected_domains: {report['summary']['num_expected_domains']}")
    print(f"weak_domains: {report['summary']['num_weak_domains']}")
    print(f"output: {args.output}")
    if report["status"] != "passed":
        raise SystemExit(1)


def cmd_foundation_validation_audit(args: argparse.Namespace) -> None:
    report = write_foundation_protocol_audit(
        args.output,
        args.reports,
        expected_benchmark_id=args.expected_benchmark_id,
        cohort_id=args.cohort_id,
        require_full_history=args.require_full_history,
    )
    print(f"status: {report['status']}")
    print(f"protocols: {', '.join(report['summary']['protocols_present'])}")
    print(f"passed_checks: {report['summary']['passed_checks']}")
    print(f"failed_checks: {report['summary']['failed_checks']}")
    print(f"output: {args.output}")
    if report["status"] != "passed":
        raise SystemExit(1)


def cmd_validate_release(args: argparse.Namespace) -> None:
    if args.output:
        report = write_release_validation(args.manifest, args.output)
    else:
        report = validate_release_artifacts(args.manifest)
    print(f"manifest: {report['manifest_path']}")
    print(f"ok: {report['ok']}")
    print(f"errors: {len(report['errors'])}")
    print(f"warnings: {len(report['warnings'])}")
    for split, split_report in report["split_reports"].items():
        if split in report.get("withheld_splits", {}):
            withheld = report["withheld_splits"][split]["split_report"]
            print(
                f"{split}: withheld=true "
                f"manifest_cases={withheld['num_cases']} manifest_queries={withheld['num_queries']}"
            )
        else:
            print(f"{split}: cases={split_report['num_cases']} queries={split_report['num_queries']}")
    if report["errors"]:
        raise SystemExit(1)


def cmd_main_dataset_acceptance(args: argparse.Namespace) -> None:
    report = write_main_dataset_acceptance(
        args.manifest,
        args.output,
        expected_benchmark_id=args.expected_benchmark_id,
        expected_profile_id=args.expected_profile_id,
        run_release_validation=args.run_release_validation,
        release_validation_report_path=args.release_validation_report,
        public_release_manifest_path=args.public_release_manifest,
        public_release_validation_report_path=args.public_release_validation_report,
        require_public_release_validation=args.require_public_release_validation,
        require_all_counterfactual_axes=args.require_all_counterfactual_axes,
        representative_reports_dir=args.representative_reports_dir,
        representative_split=args.representative_split,
        require_representative_baselines=args.require_representative_baselines,
        intrinsic_sanity_report_path=args.intrinsic_sanity_report,
        require_intrinsic_sanity=args.require_intrinsic_sanity,
        human_audit_verification_report_path=args.human_audit_verification_report,
        require_completed_human_audit=args.require_completed_human_audit,
        strong_ai_audit_report_path=args.strong_ai_audit_report,
        require_strong_ai_audit=args.require_strong_ai_audit,
        public_test_sanity_report_path=args.public_test_sanity_report,
        require_public_test_sanity=args.require_public_test_sanity,
        public_result_slices_report_path=args.public_result_slices_report,
        require_public_result_slices=args.require_public_result_slices,
        hidden_test_sanity_report_path=args.hidden_test_sanity_report,
        require_hidden_test_sanity=args.require_hidden_test_sanity,
        question_craftsmanship_report_path=args.question_craftsmanship_report,
        require_question_craftsmanship=args.require_question_craftsmanship,
        query_construction_report_path=args.query_construction_report,
        require_query_construction=args.require_query_construction,
        probe_discriminativeness_report_path=args.probe_discriminativeness_report,
        require_probe_discriminativeness=args.require_probe_discriminativeness,
        difficulty_calibration_report_path=args.difficulty_calibration_report,
        require_difficulty_calibration=args.require_difficulty_calibration,
        domain_construct_validity_report_path=args.domain_construct_validity_report,
        require_domain_construct_validity=args.require_domain_construct_validity,
        foundation_validation_report_path=args.foundation_validation_report,
        require_foundation_validation=args.require_foundation_validation,
        require_query_difficulty=args.require_query_difficulty,
    )
    print(f"status: {report['status']}")
    print(
        "summary: "
        f"passed={report['summary']['passed']} "
        f"failed={report['summary']['failed']} "
        f"warnings={report['summary']['warnings']}"
    )
    print(f"output: {args.output}")
    if report["errors"]:
        raise SystemExit(1)


def cmd_challenge_release_acceptance(args: argparse.Namespace) -> None:
    report = write_challenge_release_acceptance(
        args.manifest,
        args.output,
        public_release_manifest_path=args.public_release_manifest,
        release_validation_report_path=args.release_validation_report,
        public_release_validation_report_path=args.public_release_validation_report,
        representative_reports_dir=args.representative_reports_dir,
        question_craftsmanship_report_path=args.question_craftsmanship_report,
        query_construction_report_path=args.query_construction_report,
        probe_discriminativeness_report_path=args.probe_discriminativeness_report,
        difficulty_calibration_report_path=args.difficulty_calibration_report,
        domain_construct_validity_report_path=args.domain_construct_validity_report,
    )
    print(f"status: {report['status']}")
    print(
        "summary: "
        f"passed={report['summary']['passed']} "
        f"failed={report['summary']['failed']} "
        f"warnings={report['summary']['warnings']}"
    )
    print(f"output: {args.output}")
    if report["errors"]:
        raise SystemExit(1)


def cmd_hidden_quarterly_acceptance(args: argparse.Namespace) -> None:
    report = write_hidden_quarterly_release_acceptance(
        args.manifest,
        args.output,
        release_validation_report_path=args.release_validation_report,
        intrinsic_sanity_report_path=args.intrinsic_sanity_report,
        question_craftsmanship_report_path=args.question_craftsmanship_report,
        query_construction_report_path=args.query_construction_report,
    )
    print(f"status: {report['status']}")
    print(
        "summary: "
        f"passed={report['summary']['passed']} "
        f"failed={report['summary']['failed']} "
        f"warnings={report['summary']['warnings']}"
    )
    print(f"output: {args.output}")
    if report["errors"]:
        raise SystemExit(1)


def cmd_release_intrinsic_sanity(args: argparse.Namespace) -> None:
    report = write_release_intrinsic_sanity(args.manifest, args.output)
    print(f"status: {report['status']}")
    print(f"failed_checks: {report['summary']['failed_checks']}")
    print(f"warnings: {report['summary']['warnings']}")
    print(f"output: {args.output}")
    if report["errors"]:
        raise SystemExit(1)


def cmd_agreement(args: argparse.Namespace) -> None:
    annotations = load_audit_annotations(args.annotations)
    report = compute_agreement(annotations)
    write_json(args.output, report)
    print(f"annotations: {report['num_annotations']}")
    print(f"items: {report['num_items']}")
    print(f"annotators: {report['num_annotators']}")
    for field, metrics in report["fields"].items():
        percent = format_metric(metrics["percent_agreement"])
        kappa = format_metric(metrics["cohen_kappa"])
        print(f"{field}: percent_agreement={percent} cohen_kappa={kappa} pairs={metrics['num_pairs']}")


def cmd_prepare_human_audit(args: argparse.Namespace) -> None:
    try:
        template_paths = _audit_template_paths(args.manifest, args.templates)
        summary = write_double_annotation_task_package(template_paths, args.annotator_ids, args.output_dir)
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    if args.summary_output:
        write_json(args.summary_output, summary)
    print(f"templates: {summary['num_template_items']}")
    print(f"tasks: {summary['expected_annotations']}")
    print(f"annotators: {summary['num_annotators']}")
    print(f"completion_status: {summary['completion_status']}")


def cmd_human_audit_progress(args: argparse.Namespace) -> None:
    try:
        report = summarize_human_audit_progress(args.task_manifest, annotations_path=args.annotations)
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    write_json(args.output, report)
    print(f"status: {report['status']}")
    print(f"expected_annotations: {report['num_expected_annotations']}")
    print(f"task_file_started_annotations: {report['num_task_file_started_annotations']}")
    print(f"task_file_completed_annotations: {report['num_task_file_completed_annotations']}")
    print(f"matched_annotations: {report['num_matched_annotations']}")
    print(f"missing_annotations: {report['num_missing_annotations']}")
    print(f"ready_for_merge: {report['ready_for_merge']}")
    print(f"extra_annotations: {report['num_extra_annotations']}")
    print(f"ready_for_finalize: {report['ready_for_finalize']}")
    if report["errors"]:
        raise SystemExit(1)


def cmd_strong_ai_audit(args: argparse.Namespace) -> None:
    try:
        report = write_strong_ai_audit(
            args.output,
            task_manifest_path=args.task_manifest,
            manifest_path=args.manifest,
            expected_rows=args.expected_rows,
        )
        if args.row_ledger_output:
            ledger_summary = write_strong_ai_audit_ledger(
                args.row_ledger_output,
                task_manifest_path=args.task_manifest,
                manifest_path=args.manifest,
                expected_rows=args.expected_rows,
            )
            report["row_ledger"] = {
                key: value
                for key, value in ledger_summary.items()
                if key != "checks"
            }
            write_json(args.output, report)
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    print(f"status: {report['status']}")
    print(f"source_type: {report['source_type']}")
    print(f"rows: {report['summary']['num_rows']}")
    print(f"issues: {report['summary']['num_issues']}")
    print(f"failed_checks: {report['summary']['num_failed_checks']}")
    print(f"row_audit_digest: {report['summary']['row_audit_digest']}")
    if args.row_ledger_output:
        print(f"row_ledger_output: {args.row_ledger_output}")
        print(f"row_ledger_sha256: {report['row_ledger']['ledger_sha256']}")
    if report["status"] != "passed":
        raise SystemExit(1)


def cmd_export_human_audit_annotation_sheets(args: argparse.Namespace) -> None:
    try:
        summary = write_human_audit_annotation_sheets(args.task_manifest, args.output_dir)
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    print(f"sheets: {summary['num_sheets']}")
    print(f"rows: {summary['num_rows']}")
    print(f"output_dir: {summary['output_dir']}")


def cmd_apply_human_audit_annotation_sheet(args: argparse.Namespace) -> None:
    try:
        summary = apply_human_audit_annotation_sheet(
            args.task_manifest,
            args.annotator_id,
            args.sheet,
        )
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    print(f"annotator: {summary['annotator_id']}")
    print(f"status: {summary['status']}")
    print(f"started_annotations: {summary['num_started_annotations']}")
    print(f"completed_annotations: {summary['num_completed_annotations']}")
    print(f"task_file: {summary['task_file']}")


def cmd_apply_human_audit_annotator_packet(args: argparse.Namespace) -> None:
    try:
        summary = apply_human_audit_annotator_packet(
            args.task_manifest,
            args.packet,
            annotator_id=args.annotator_id,
        )
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    print(f"packet: {summary['packet_path']}")
    print(f"packet_type: {summary['packet_type']}")
    print(f"annotator: {summary['annotator_id']}")
    print(f"status: {summary['status']}")
    print(f"started_annotations: {summary['num_started_annotations']}")
    print(f"completed_annotations: {summary['num_completed_annotations']}")
    print(f"task_file: {summary['task_file']}")


def cmd_apply_human_audit_adjudication_packet(args: argparse.Namespace) -> None:
    try:
        summary = apply_human_audit_adjudication_packet(
            args.bundle_dir,
            args.packet,
            adjudicator_id=args.adjudicator_id,
            reconcile_when_ready=args.reconcile_when_ready,
            signed_at=args.signed_at,
            annotation_guideline=args.annotation_guideline,
            adjudication_policy=args.adjudication_policy,
        )
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    if args.output:
        write_json(args.output, summary)
    print(f"packet: {summary['packet_path']}")
    print(f"packet_type: {summary['packet_type']}")
    print(f"adjudicator: {summary['adjudicator_id']}")
    print(f"status: {summary['status']}")
    print(f"completed_rows: {summary['num_completed_rows']}")
    print(f"ready_for_finalize: {summary['ready_for_finalize']}")
    print(f"reconcile_status: {summary['reconcile_status']}")
    if args.output:
        print(f"output: {args.output}")


def cmd_ingest_human_audit_return_packets(args: argparse.Namespace) -> None:
    try:
        summary = ingest_human_audit_return_packets(
            args.bundle_dir,
            args.packets,
            reconcile_when_ready=args.reconcile_when_ready,
            signed_at=args.signed_at,
            annotation_guideline=args.annotation_guideline,
            adjudication_policy=args.adjudication_policy,
        )
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    if args.output:
        write_json(args.output, summary)
    print(f"packets: {summary['num_packets']}")
    print(f"applied_annotators: {','.join(summary['applied_annotators'])}")
    print(f"ready_for_merge: {summary['ready_for_merge']}")
    print(f"ready_for_finalize: {summary['ready_for_finalize']}")
    print(f"reconcile_status: {summary['reconcile_status']}")
    if args.output:
        print(f"output: {args.output}")


def cmd_sync_human_audit_return_inbox(args: argparse.Namespace) -> None:
    try:
        summary = sync_human_audit_return_inbox(
            args.bundle_dir,
            annotator_inboxes=args.annotator_inbox,
            adjudication_inboxes=args.adjudication_inbox,
            reconcile_when_ready=args.reconcile_when_ready,
            signed_at=args.signed_at,
            annotation_guideline=args.annotation_guideline,
            adjudication_policy=args.adjudication_policy,
        )
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    if args.output:
        write_json(args.output, summary)
    print(f"annotator_candidates: {len(summary['annotator_candidates'])}")
    print(f"adjudication_candidate: {summary['adjudication_candidate'] is not None}")
    print(f"bundle_status: {summary['bundle_status']}")
    print(f"ready_for_finalize: {summary['ready_for_finalize']}")
    if args.output:
        print(f"output: {args.output}")


def cmd_watch_human_audit_return_inbox(args: argparse.Namespace) -> None:
    if args.interval_s < 0:
        raise SystemExit("--interval-s must be non-negative")
    if args.max_iterations < 0:
        raise SystemExit("--max-iterations must be non-negative")
    try:
        summary = watch_human_audit_return_inbox(
            args.bundle_dir,
            annotator_inboxes=args.annotator_inbox,
            adjudication_inboxes=args.adjudication_inbox,
            reconcile_when_ready=args.reconcile_when_ready,
            signed_at=args.signed_at,
            annotation_guideline=args.annotation_guideline,
            adjudication_policy=args.adjudication_policy,
            interval_s=args.interval_s,
            max_iterations=args.max_iterations,
            stop_when_ready=args.stop_when_ready,
            stop_when_rejected=args.stop_when_rejected,
            output_path=args.output,
        )
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    print(f"iterations: {summary['iteration_count']}")
    print(f"bundle_status: {summary['bundle_status']}")
    print(f"ready_for_merge: {summary['ready_for_merge']}")
    print(f"ready_for_finalize: {summary['ready_for_finalize']}")
    print(f"stop_reason: {summary['stop_reason']}")
    print(f"stop_exit_code: {summary.get('stop_exit_code', 0)}")
    if summary.get("stop_reason") == "rejected_returns":
        print(f"rejected_returns_report: {summary.get('rejected_returns_report_file')}")
    if summary.get("stop_reason") in {"ready", "rejected_returns"}:
        if isinstance(summary.get("next_command"), str) and summary["next_command"].strip():
            print(f"next_command: {summary['next_command']}")
        if isinstance(summary.get("next_script"), str) and summary["next_script"].strip():
            print(f"next_script: {summary['next_script']}")
        if isinstance(summary.get("next_script_file"), str) and summary["next_script_file"].strip():
            print(f"next_script_file: {summary['next_script_file']}")
    if args.output:
        print(f"output: {args.output}")
    stop_exit_code = int(summary.get("stop_exit_code", 0) or 0)
    if stop_exit_code:
        raise SystemExit(stop_exit_code)


def cmd_verify_manifest_human_audit(args: argparse.Namespace) -> None:
    report = verify_manifest_human_audit(args.manifest)
    write_json(args.output, report)
    print(f"ok: {report['ok']}")
    print(f"num_template_items: {report['num_template_items']}")
    print(f"num_annotators: {report['num_annotators']}")
    print(f"output: {args.output}")
    if report["errors"]:
        raise SystemExit(1)


def cmd_merge_human_audit_annotations(args: argparse.Namespace) -> None:
    try:
        summary = merge_completed_human_audit_tasks(args.task_manifest, args.output)
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    if args.summary_output:
        write_json(args.summary_output, summary)
    print(f"annotations: {summary['num_annotations']}")
    print(f"items: {summary['num_template_items']}")
    print(f"annotators: {summary['num_annotators']}")
    print(f"ready_for_finalize: {summary['ready_for_finalize']}")
    if args.summary_output:
        print(f"summary: {args.summary_output}")


def cmd_summarize_human_audit_disagreements(args: argparse.Namespace) -> None:
    try:
        report = summarize_human_audit_disagreements(args.task_manifest, args.annotations)
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    write_json(args.output, report)
    print(f"adjudication_required: {report['adjudication_required']}")
    print(f"disagreement_items: {report['num_disagreement_items']}")
    print(f"disagreement_fields: {report['num_disagreement_fields']}")
    print(f"tied_fields: {report['num_tied_fields']}")
    print(f"output: {args.output}")


def cmd_build_human_audit_adjudication_package(args: argparse.Namespace) -> None:
    try:
        report = build_human_audit_adjudication_package(
            args.task_manifest,
            args.annotations,
            args.output_dir,
        )
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    print(f"status: {report['status']}")
    print(f"disagreement_items: {report['num_disagreement_items']}")
    print(f"disagreement_fields: {report['num_disagreement_fields']}")
    print(f"adjudication_required: {report['adjudication_required']}")
    print(f"output_dir: {args.output_dir}")


def cmd_generate_human_audit_attestation(args: argparse.Namespace) -> None:
    try:
        attestation = build_human_audit_attestation(
            args.task_manifest,
            args.annotations,
            args.output,
            annotation_guideline=args.annotation_guideline,
            adjudication_policy=args.adjudication_policy,
            signed_at=args.signed_at,
        )
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    print(f"annotators: {len(attestation['annotators'])}")
    print(f"task_manifest: {attestation['task_manifest_file']}")
    print(f"annotations: {attestation['annotations_file']}")
    print(f"output: {args.output}")


def cmd_build_human_audit_sandbox_sample(args: argparse.Namespace) -> None:
    try:
        sandbox = build_human_audit_sandbox_sample(
            args.task_manifest,
            args.output_dir,
            num_items=args.num_items,
            annotation_guideline=args.annotation_guideline,
            adjudication_policy=args.adjudication_policy,
            signed_at=args.signed_at,
        )
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    print(f"items: {sandbox['num_items']}")
    print(f"annotators: {sandbox['num_annotators']}")
    print(f"completed_annotations: {sandbox['completed_annotations_file']}")
    print(f"attestation: {sandbox['attestation_file']}")
    print(f"verification_ok: {sandbox['verification']['ok']}")
    print(f"output_dir: {args.output_dir}")


def cmd_build_human_audit_evidence_bundle(args: argparse.Namespace) -> None:
    try:
        bundle = build_human_audit_evidence_bundle(
            args.output_dir,
            manifest_path=args.manifest,
            task_manifest_path=args.task_manifest,
            annotations_path=args.annotations,
            annotator_attestation_path=args.annotator_attestation,
            adjudication_path=args.adjudication,
        )
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    print(f"task_manifest: {bundle['task_manifest_file']}")
    print(f"annotations: {bundle['annotations_file']}")
    print(f"progress: {bundle['progress_file']}")
    print(f"readme: {bundle['readme_file']}")
    print(f"annotation_sheets: {bundle['annotation_sheet_summary']['num_sheets']}")
    print(f"ready_for_merge: {bundle['progress']['ready_for_merge']}")
    print(f"ready_for_finalize: {bundle['progress']['ready_for_finalize']}")
    print(f"verification_ok: {bundle['verification']['ok'] if bundle['verification'] is not None else None}")
    print(f"output_dir: {args.output_dir}")


def cmd_verify_human_audit_evidence_bundle(args: argparse.Namespace) -> None:
    report = verify_human_audit_evidence_bundle(args.bundle_dir)
    write_json(args.output, report)
    print(f"ok: {report['ok']}")
    print(f"status: {report['status']}")
    print(f"task_manifest: {report['task_manifest_file']}")
    print(f"output: {args.output}")
    if report["errors"]:
        raise SystemExit(1)


def cmd_summarize_human_audit_rejected_returns(args: argparse.Namespace) -> None:
    try:
        report = summarize_human_audit_rejected_returns(args.bundle_dir)
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    write_json(args.output, report)
    summary = report["rejected_return_summary"]
    print(f"has_rejected_returns: {report['has_rejected_returns']}")
    print(f"rejected_annotator_packets: {summary['num_rejected_annotator_packets']}")
    print(f"rejected_adjudication_packets: {summary['num_rejected_adjudication_packets']}")
    print(f"output: {args.output}")


def cmd_reconcile_human_audit_evidence_bundle(args: argparse.Namespace) -> None:
    try:
        report = reconcile_human_audit_evidence_bundle(
            args.bundle_dir,
            source_manifest_path=args.source_manifest,
            source_task_manifest_path=args.source_task_manifest,
            source_annotations_output=args.source_annotations_output,
            source_attestation_output=args.source_attestation_output,
            source_adjudication_output=args.source_adjudication_output,
            source_agreement_output=args.source_agreement_output,
            annotation_guideline=args.annotation_guideline,
            adjudication_policy=args.adjudication_policy,
            signed_at=args.signed_at,
        )
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    if args.output:
        write_json(args.output, report)
    print(f"source_finalize_ok: {report['source_finalize_ok']}")
    print(f"bundle_verification_ok: {report['bundle_verification_ok']}")
    print(f"source_manifest: {report['source_manifest_file']}")
    print(f"source_annotations: {report['source_annotations_file']}")
    print(f"source_attestation: {report['source_annotator_attestation_file']}")
    if report["source_adjudication_file"]:
        print(f"source_adjudication: {report['source_adjudication_file']}")
    if report["source_agreement_metrics_file"]:
        print(f"source_agreement: {report['source_agreement_metrics_file']}")
    if args.output:
        print(f"output: {args.output}")


def cmd_finalize_human_audit(args: argparse.Namespace) -> None:
    try:
        report = finalize_human_audit_manifest(
            args.manifest,
            args.annotations,
            task_manifest_path=args.task_manifest,
            annotator_attestation_path=args.annotator_attestation,
            adjudication_path=args.adjudication,
            agreement_output=args.agreement_output,
        )
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    print("human_audit_status: completed")
    print(f"annotations: {report['num_annotations']}")
    print(f"items: {report['num_template_items']}")
    print(f"annotators: {report['num_annotators']}")


def cmd_completion_audit(args: argparse.Namespace) -> None:
    report = write_completion_audit(args.root, args.output)
    print(f"status: {report['status']}")
    print(
        "summary: "
        f"passed={report['summary'].get('passed', 0)} "
        f"partial={report['summary'].get('partial', 0)} "
        f"missing={report['summary'].get('missing', 0)}"
    )
    for check in report["checks"]:
        if check["status"] != "passed":
            print(f"{check['check_id']}: {check['status']}")


def cmd_git_hygiene_report(args: argparse.Namespace) -> None:
    report = write_git_hygiene_report(args.root, args.output)
    print(f"status: {report['status']}")
    print(
        "summary: "
        f"tracked_modifications={report['summary'].get('tracked_modification_count', 0)} "
        f"untracked_entries={report['summary'].get('untracked_entry_count', 0)} "
        f"imported_untracked_sources={report['summary'].get('imported_untracked_source_count', 0)}"
    )
    print(f"recommended_batches: {len(report['recommended_batches'])}")
    if report["recommended_batches"]:
        print(f"next_stage_batch: {report['recommended_batches'][0]['batch_id']}")
    if report["status"] != "passed":
        print(f"issues: {len(report['missing'])}")


def cmd_git_hygiene_plan(args: argparse.Namespace) -> None:
    manifest = write_git_hygiene_plan(args.root, args.output_dir)
    print(f"batches: {manifest['num_batches']}")
    print(f"output_dir: {manifest['output_dir']}")
    if manifest["batches"]:
        print(f"next_stage_command: {manifest['batches'][0]['stage_command']}")


def cmd_objective_checklist(args: argparse.Namespace) -> None:
    report = write_objective_checklist(args.root, args.output)
    print(f"status: {report['status']}")
    print(
        "summary: "
        f"passed={report['summary'].get('passed', 0)} "
        f"partial={report['summary'].get('partial', 0)} "
        f"missing={report['summary'].get('missing', 0)}"
    )
    for item in report["items"]:
        if item["status"] != "passed":
            print(f"{item['item_id']}: {item['status']}")


def cmd_evidence_readiness(args: argparse.Namespace) -> None:
    report = write_evidence_readiness_report(
        args.output,
        root=args.root,
        external_output_dir=args.external_output_dir,
        human_task_manifest=args.human_task_manifest,
        human_annotations=args.human_annotations,
        real_system_summary=args.real_system_summary,
        real_system_sample_validation=args.real_system_sample_validation,
        real_system_sample_summary=args.real_system_sample_summary,
        integration_configs=args.integration_configs,
        integration_config_validation_report=args.integration_config_validation_report,
    )
    blockers = report["completion_audit"]["blockers"]
    print(f"status: {report['status']}")
    print(f"completion_audit: {report['completion_audit']['status']}")
    print(f"blockers: {len(blockers)}")
    for action in report["next_actions"]:
        print(f"{action['blocker']}: {action['next_command']}")


def cmd_validate_real_system_matrix(args: argparse.Namespace) -> None:
    report = write_real_system_matrix_validation(
        args.summary,
        args.output,
        expected_benchmark_id=args.expected_benchmark_id,
        expected_release_split=args.expected_release_split,
    )
    print(f"status: {report['status']}")
    print(f"systems: {report['num_systems']}")
    print(f"framework_trace_artifacts: {report.get('num_framework_trace_artifacts', 0)}")
    print(f"covered_providers: {', '.join(report['covered_providers'])}")
    print(f"missing_providers: {', '.join(report['missing_providers'])}")
    print(f"errors: {len(report['errors'])}")
    if report["errors"]:
        raise SystemExit(1)


def cmd_merge_real_system_matrix(args: argparse.Namespace) -> None:
    report = write_merged_real_system_matrix_summary(
        args.summaries,
        args.output,
        expected_benchmark_id=args.expected_benchmark_id,
        expected_release_split=args.expected_release_split,
    )
    print(f"benchmark_id: {report['benchmark_id']}")
    print(f"release_split: {report['release_split']}")
    print(f"systems: {report['num_systems']}")
    print(f"output: {args.output}")


def cmd_finalize_real_system_run(args: argparse.Namespace) -> None:
    try:
        report = finalize_real_system_run(
            args.manifest,
            split=args.split,
            run_dir=args.run_dir,
            retrieval_k=args.retrieval_k,
            summary_output=args.summary_output,
        )
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    print(f"status: {report['status']}")
    print(f"system_id: {report['system_id']}")
    print(f"predictions: {report['num_predictions']}/{report['total_queries']}")
    print(f"report: {report['report_path']}")
    print(f"summary: {report['summary_path']}")


def cmd_backfill_real_system_run_metadata(args: argparse.Namespace) -> None:
    try:
        report = backfill_real_system_run_metadata(
            args.manifest,
            split=args.split,
            config_path=args.config,
            run_dir=args.run_dir,
            system_id=args.system_id,
            system_version=args.system_version,
            retrieval_k=args.retrieval_k,
            resume=not args.no_resume,
            command=args.command,
            overwrite=args.overwrite,
        )
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    print(f"status: {report['status']}")
    print(f"system_id: {report['system_id']}")
    print(f"run_metadata: {report['run_metadata_path']}")
    print(f"benchmark_id: {report['benchmark_id']}")


def cmd_refresh_real_system_canonical(args: argparse.Namespace) -> None:
    resolved = _resolve_refresh_real_system_canonical_args(args)
    report = write_real_system_canonical_refresh(
        resolved["manifest"],
        resolved["output"],
        split=resolved["split"],
        run_specs=resolved["run_specs"],
        config_validation_output=resolved["config_validation_output"],
        analysis_output=resolved.get("analysis_output"),
        merged_summary_output=resolved["merged_summary_output"],
        merged_validation_output=resolved["merged_validation_output"],
        expected_benchmark_id=resolved["expected_benchmark_id"],
        expected_release_split=resolved["expected_release_split"],
        retrieval_k=resolved["retrieval_k"],
    )
    print(f"status: {report['status']}")
    print(
        "summary: "
        f"runs={report['summary']['num_runs']} "
        f"running={report['summary']['num_running']} "
        f"ready_to_finalize={report['summary']['num_ready_to_finalize']} "
        f"summaries={report['summary']['num_available_summaries']}"
    )
    print(f"merged_summary: {report['merged_summary_path']}")
    print(f"merged_validation: {report['merged_validation_path']}")
    print(f"config_validation: {report['config_validation_path']}")
    print(f"analysis: {report['analysis_path']}")
    print(f"output: {resolved['output']}")
    if report["errors"]:
        raise SystemExit(1)


def cmd_watch_real_system_canonical(args: argparse.Namespace) -> None:
    resolved = _resolve_refresh_real_system_canonical_args(args)
    project_root = Path(args.root)
    evidence_output = args.evidence_readiness_output or str(project_root / "reports/examples/amst_evidence_readiness_current.json")
    completion_output = args.completion_audit_output or str(project_root / "reports/examples/amst_completion_audit_current.json")
    checklist_output = args.objective_checklist_output or str(project_root / "reports/examples/amst_objective_checklist_current.json")
    git_hygiene_output = args.git_hygiene_output or str(project_root / "reports/examples/amst_git_hygiene_current.json")
    git_hygiene_plan_output_dir = args.git_hygiene_plan_output_dir or str(project_root / "reports/git_hygiene/current")
    external_output_dir = args.external_output_dir or str(project_root / "reports/external")
    lock_file = Path(args.lock_file) if args.lock_file else _default_watch_real_system_canonical_lock_file(project_root)
    if args.interval_s < 0:
        raise SystemExit("--interval-s must be non-negative")
    if args.max_iterations < 0:
        raise SystemExit("--max-iterations must be non-negative")

    iteration = 0
    last_report: dict[str, object] | None = None
    last_external_refresh: dict[str, object] | None = None
    last_strict_main_support_currents: dict[str, object] | None = None
    last_main_quality_currents: dict[str, object] | None = None
    last_strict_main_quality_currents: dict[str, object] | None = None
    last_challenge_release_currents: dict[str, object] | None = None
    last_hidden_quarterly_release_currents: dict[str, object] | None = None
    last_challenge_acceptance_current_path: str | None = None
    last_hidden_quarterly_acceptance_current_path: str | None = None
    last_main_acceptance_current_path: str | None = None
    last_strict_main_acceptance_current_path: str | None = None
    try:
        _acquire_watch_real_system_canonical_lock(lock_file, project_root=project_root)
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    try:
        while True:
            iteration += 1
            report = write_real_system_canonical_refresh(
                resolved["manifest"],
                resolved["output"],
                split=resolved["split"],
                run_specs=resolved["run_specs"],
                config_validation_output=resolved["config_validation_output"],
                analysis_output=resolved.get("analysis_output"),
                merged_summary_output=resolved["merged_summary_output"],
                merged_validation_output=resolved["merged_validation_output"],
                expected_benchmark_id=resolved["expected_benchmark_id"],
                expected_release_split=resolved["expected_release_split"],
                retrieval_k=resolved["retrieval_k"],
            )
            write_git_hygiene_report(project_root, git_hygiene_output)
            write_git_hygiene_plan(project_root, git_hygiene_plan_output_dir)
            last_external_refresh = write_external_canonical_refresh(
                root=project_root,
                output_dir=external_output_dir,
                real_system_validation_path=report.get("merged_validation_path"),
            )
            write_evidence_readiness_report(
                evidence_output,
                root=project_root,
                external_output_dir=external_output_dir,
                real_system_summary=report.get("merged_summary_path"),
                integration_config_validation_report=report.get("config_validation_path"),
            )
            last_strict_main_support_currents = write_canonical_strict_main_release_support_currents(project_root)
            last_main_quality_currents = write_canonical_main_quality_audits_current(project_root)
            last_strict_main_quality_currents = write_canonical_strict_main_quality_audits_current(project_root)
            last_challenge_release_currents = write_canonical_challenge_release_currents(project_root)
            last_hidden_quarterly_release_currents = write_canonical_hidden_quarterly_release_currents(project_root)
            challenge_acceptance_current_path = project_root / "reports/examples/amst_challenge_v1_acceptance_current.json"
            challenge_acceptance_current = write_canonical_challenge_acceptance_current(
                project_root,
                challenge_acceptance_current_path,
            )
            last_challenge_acceptance_current_path = (
                str(challenge_acceptance_current_path.relative_to(project_root))
                if challenge_acceptance_current is not None
                else None
            )
            hidden_quarterly_acceptance_current_path = (
                project_root / "reports/examples/amst_hidden_quarterly_v1_acceptance_current.json"
            )
            hidden_quarterly_acceptance_current = write_canonical_hidden_quarterly_acceptance_current(
                project_root,
                hidden_quarterly_acceptance_current_path,
            )
            last_hidden_quarterly_acceptance_current_path = (
                str(hidden_quarterly_acceptance_current_path.relative_to(project_root))
                if hidden_quarterly_acceptance_current is not None
                else None
            )
            main_acceptance_current_path = project_root / "reports/examples/amst_main_v1_acceptance_current.json"
            acceptance_current = write_canonical_main_acceptance_current(project_root, main_acceptance_current_path)
            last_main_acceptance_current_path = (
                str(main_acceptance_current_path.relative_to(project_root)) if acceptance_current is not None else None
            )
            strict_main_acceptance_current_path = (
                project_root / "reports/examples/amst_main_v1_strict_acceptance_current.json"
            )
            strict_acceptance_current = write_canonical_strict_main_acceptance_current(
                project_root,
                strict_main_acceptance_current_path,
            )
            last_strict_main_acceptance_current_path = (
                str(strict_main_acceptance_current_path.relative_to(project_root))
                if strict_acceptance_current is not None
                else None
            )
            write_completion_audit(project_root, completion_output)
            write_objective_checklist(project_root, checklist_output)
            last_report = report
            print(
                f"iteration: {iteration} status={report['status']} "
                f"running={report['summary']['num_running']} "
                f"ready_to_finalize={report['summary']['num_ready_to_finalize']} "
                f"summaries={report['summary']['num_available_summaries']}",
                flush=True,
            )
            if report["errors"]:
                raise SystemExit(1)
            if args.stop_when_passed and report["status"] == "passed":
                break
            if args.max_iterations and iteration >= args.max_iterations:
                break
            if args.max_iterations == 0 or iteration < args.max_iterations:
                if args.interval_s > 0:
                    time.sleep(args.interval_s)
                continue
            break
    finally:
        _release_watch_real_system_canonical_lock(lock_file)

    assert last_report is not None
    assert last_external_refresh is not None
    print(f"merged_summary: {last_report['merged_summary_path']}")
    print(f"merged_validation: {last_report['merged_validation_path']}")
    print(f"config_validation: {last_report['config_validation_path']}")
    print(f"analysis: {last_report['analysis_path']}")
    print(f"refresh_output: {resolved['output']}")
    print(f"external_plan: {last_external_refresh['plan_output']}")
    print(f"external_validation: {last_external_refresh['validation_output']}")
    print(f"external_gap: {last_external_refresh['gap_output']}")
    print(f"external_expansion: {last_external_refresh['expansion_output']}")
    if "expansion_validation_output" in last_external_refresh:
        print(f"external_expansion_validation: {last_external_refresh['expansion_validation_output']}")
    if "expansion_handoff_output" in last_external_refresh:
        print(f"external_handoff_manifest: {last_external_refresh['expansion_handoff_output']}")
    if last_main_acceptance_current_path is not None:
        print(f"main_acceptance_current: {last_main_acceptance_current_path}")
    if last_strict_main_acceptance_current_path is not None:
        print(f"main_strict_acceptance_current: {last_strict_main_acceptance_current_path}")
    if last_challenge_acceptance_current_path is not None:
        print(f"challenge_acceptance_current: {last_challenge_acceptance_current_path}")
    if last_hidden_quarterly_acceptance_current_path is not None:
        print(f"hidden_quarterly_acceptance_current: {last_hidden_quarterly_acceptance_current_path}")
    if last_strict_main_support_currents is not None:
        print(
            "main_strict_support_currents: "
            + ", ".join(sorted(last_strict_main_support_currents.get("artifacts", {}).values()))
        )
    if last_main_quality_currents is not None:
        print(
            "main_quality_currents: "
            + ", ".join(sorted(last_main_quality_currents.get("artifacts", {}).values()))
        )
    if last_strict_main_quality_currents is not None:
        print(
            "main_strict_quality_currents: "
            + ", ".join(sorted(last_strict_main_quality_currents.get("artifacts", {}).values()))
        )
    if last_challenge_release_currents is not None:
        print(
            "challenge_release_currents: "
            + ", ".join(sorted(last_challenge_release_currents.get("artifacts", {}).values()))
        )
    if last_hidden_quarterly_release_currents is not None:
        print(
            "hidden_quarterly_release_currents: "
            + ", ".join(sorted(last_hidden_quarterly_release_currents.get("artifacts", {}).values()))
        )
    print(f"evidence_readiness: {evidence_output}")
    print(f"completion_audit: {completion_output}")
    print(f"objective_checklist: {checklist_output}")
    print(f"git_hygiene: {git_hygiene_output}")
    print(f"git_hygiene_plan: {git_hygiene_plan_output_dir}")
    print(f"lock_file: {lock_file}")


def cmd_real_system_run_progress(args: argparse.Namespace) -> None:
    report = write_real_system_run_progress(
        args.manifest,
        args.output,
        split=args.split,
        run_dir=args.run_dir,
    )
    print(f"status: {report['status']}")
    print(f"system_id: {report['system_id']}")
    print(f"predictions: {report['num_predictions']}/{report['total_queries']}")
    print(f"has_run_state: {report['has_run_state']}")
    live_state = report.get("live_state") or {}
    if live_state:
        print(f"live_phase: {live_state.get('phase')}")
        print(f"live_case: {live_state.get('case_id')}")
        if live_state.get("query_id"):
            print(f"live_query: {live_state.get('query_id')}")
    print(f"has_report: {report['has_report']}")
    print(f"output: {args.output}")
    if report["errors"]:
        raise SystemExit(1)


def cmd_audit_template_summary(args: argparse.Namespace) -> None:
    summary = summarize_audit_templates(args.templates)
    write_json(args.output, summary)
    print(f"templates: {summary['num_templates']}")
    print(f"records: {summary['num_records']}")
    print(f"cases: {summary['num_cases']}")
    print(f"queries: {summary['num_queries']}")
    print(f"ready_for_double_annotation: {summary['ready_for_double_annotation']}")


def _audit_template_paths(manifest_path: str | None, templates: list[str] | None) -> list[str]:
    if templates:
        return templates
    if not manifest_path:
        raise ValueError("prepare-human-audit requires --templates or --manifest")
    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    audit_plan = manifest.get("audit_plan", {}) if isinstance(manifest, dict) else {}
    paths: list[str] = []
    template_files = audit_plan.get("audit_template_files")
    if isinstance(template_files, dict):
        paths.extend(str(path) for _, path in sorted(template_files.items()))
    template_file = audit_plan.get("audit_template_file")
    if template_file:
        paths.append(str(template_file))
    if not paths:
        raise ValueError("manifest audit_plan does not include audit template files")
    return [str(_resolve_manifest_path(manifest_file.parent, path)) for path in paths]


def _resolve_manifest_path(manifest_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    return manifest_dir / path


def _load_refresh_spec(path: str) -> dict[str, object]:
    spec_path = Path(path)
    data = read_json(spec_path)
    if not isinstance(data, dict):
        raise SystemExit(f"{spec_path}: refresh spec must be a JSON object")
    base_dir = spec_path.parent
    resolved: dict[str, object] = {}
    for key in (
        "manifest",
        "output",
        "config_validation_output",
        "merged_summary_output",
        "merged_validation_output",
        "expected_benchmark_id",
        "expected_release_split",
        "split",
        "retrieval_k",
    ):
        if key not in data:
            continue
        value = data[key]
        if key in {"manifest", "output", "config_validation_output", "merged_summary_output", "merged_validation_output"} and isinstance(value, str):
            resolved_path = _resolve_manifest_path(base_dir, value)
            resolved[key] = os.path.normpath(str(resolved_path))
        else:
            resolved[key] = value
    runs = data.get("runs")
    if runs is not None:
        if not isinstance(runs, list):
            raise SystemExit(f"{spec_path}: spec.runs must be a list")
        resolved_runs: list[dict[str, str]] = []
        for index, item in enumerate(runs, start=1):
            if not isinstance(item, dict):
                raise SystemExit(f"{spec_path}: spec.runs[{index}] must be an object")
            config_path = item.get("config_path")
            run_dir = item.get("run_dir")
            if not isinstance(config_path, str) or not isinstance(run_dir, str):
                raise SystemExit(f"{spec_path}: spec.runs[{index}] must include string config_path and run_dir")
            resolved_item = {
                "config_path": os.path.normpath(str(_resolve_manifest_path(base_dir, config_path))),
                "run_dir": os.path.normpath(str(_resolve_manifest_path(base_dir, run_dir))),
            }
            for optional_key in ("summary_path", "progress_output"):
                value = item.get(optional_key)
                if isinstance(value, str):
                    resolved_item[optional_key] = os.path.normpath(str(_resolve_manifest_path(base_dir, value)))
            resolved_runs.append(resolved_item)
        resolved["runs"] = resolved_runs
    return resolved


def _resolve_refresh_real_system_canonical_args(args: argparse.Namespace) -> dict[str, object]:
    spec = _load_refresh_spec(args.spec) if getattr(args, "spec", None) else {}
    manifest = getattr(args, "manifest", None) or spec.get("manifest")
    split = getattr(args, "split", None) or spec.get("split")
    output = getattr(args, "output", None) or spec.get("output")
    config_validation_output = getattr(args, "config_validation_output", None) or spec.get("config_validation_output")
    analysis_output = getattr(args, "analysis_output", None) or spec.get("analysis_output")
    merged_summary_output = getattr(args, "merged_summary_output", None) or spec.get("merged_summary_output")
    merged_validation_output = getattr(args, "merged_validation_output", None) or spec.get("merged_validation_output")
    expected_benchmark_id = getattr(args, "expected_benchmark_id", None) or spec.get("expected_benchmark_id")
    expected_release_split = getattr(args, "expected_release_split", None) or spec.get("expected_release_split")
    retrieval_k = getattr(args, "retrieval_k", None)
    if retrieval_k is None:
        retrieval_k = int(spec.get("retrieval_k", DEFAULT_RETRIEVAL_K))
    raw_runs = getattr(args, "run", None) if getattr(args, "run", None) else spec.get("runs")
    if not manifest:
        raise SystemExit("refresh-real-system-canonical requires --manifest or spec.manifest")
    if not split:
        raise SystemExit("refresh-real-system-canonical requires --split or spec.split")
    if not output:
        raise SystemExit("refresh-real-system-canonical requires --output or spec.output")
    run_specs = _normalize_refresh_run_specs(raw_runs)
    if not run_specs:
        raise SystemExit("refresh-real-system-canonical requires --run entries or spec.runs")
    if not config_validation_output:
        config_validation_output = os.path.normpath(str(default_real_system_config_validation_output(output)))
    if not analysis_output:
        analysis_output = os.path.normpath(str(default_real_system_analysis_output(output)))
    return {
        "manifest": manifest,
        "split": split,
        "output": output,
        "config_validation_output": config_validation_output,
        "analysis_output": analysis_output,
        "merged_summary_output": merged_summary_output,
        "merged_validation_output": merged_validation_output,
        "expected_benchmark_id": expected_benchmark_id,
        "expected_release_split": expected_release_split,
        "retrieval_k": retrieval_k,
        "run_specs": run_specs,
    }


def _normalize_refresh_run_specs(raw_runs: object) -> list[dict[str, str]]:
    if not raw_runs:
        return []
    normalized: list[dict[str, str]] = []
    if isinstance(raw_runs, list):
        for item in raw_runs:
            if isinstance(item, dict):
                config_path = item.get("config_path")
                run_dir = item.get("run_dir")
                if isinstance(config_path, str) and isinstance(run_dir, str):
                    normalized_item = {"config_path": config_path, "run_dir": run_dir}
                    for optional_key in ("summary_path", "progress_output"):
                        value = item.get(optional_key)
                        if isinstance(value, str):
                            normalized_item[optional_key] = value
                    normalized.append(normalized_item)
                    continue
            if isinstance(item, (list, tuple)) and len(item) == 2:
                normalized.append({"config_path": str(item[0]), "run_dir": str(item[1])})
                continue
            raise SystemExit("refresh-real-system-canonical run entries must be (CONFIG RUN_DIR) pairs or spec objects")
        return normalized
    raise SystemExit("refresh-real-system-canonical run entries must be a list")
