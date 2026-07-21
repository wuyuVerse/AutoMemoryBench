"""One-shot workflow for building a complete AutoMemoryBench main release package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from amb.benchmark.artifact_contract import localize_report_contract
from amb.benchmark.evaluation.scoring import DEFAULT_RETRIEVAL_K
from amb.benchmark.generation.profiles import (
    CANONICAL_FINAL_MAIN_PROFILE_ID,
    is_canonical_final_main_profile,
)
from amb.benchmark.quality.annotation import write_double_annotation_task_package
from amb.benchmark.quality.human_audit import build_human_audit_evidence_bundle
from amb.benchmark.quality.human_audit import verify_human_audit_evidence_bundle
from amb.benchmark.release.failure_modes import build_release_failure_mode_diagnostics
from amb.benchmark.release.hidden_test_sanity import build_hidden_test_sanity_artifact
from amb.benchmark.release.packages import export_public_release_package
from amb.benchmark.release.public_test_baselines import build_public_test_baseline_artifacts
from amb.benchmark.release.public_result_slices import build_public_result_slice_artifacts
from amb.benchmark.release.public_test_sanity import build_public_test_sanity_summary
from amb.benchmark.release.public_test_summary import write_public_test_summary
from amb.benchmark.release.representative import build_representative_baseline_artifacts
from amb.benchmark.release.sharded import build_profile_release_shards
from amb.benchmark.release.splits import ReleaseConfig
from amb.benchmark.schemas.io import read_json
from amb.benchmark.schemas.io import write_json


MAIN_RELEASE_PROFILES = ("main-v1", "main-v1-strict")


def build_main_release_workflow(
    profile_id: str,
    output_dir: str | Path,
    *,
    existing_release_manifest_path: str | Path | None = None,
    public_output_dir: str | Path | None = None,
    reports_dir: str | Path = "reports/examples",
    release_config: ReleaseConfig | None = None,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    representative_split: str = "public_dev",
    bootstrap_samples: int = 200,
    seed: int | None = None,
    skip_representative_baselines: bool = False,
    require_completed_human_audit: bool = False,
    foundation_report_paths: tuple[str | Path, ...] = (),
    foundation_expected_benchmark_id: str | None = None,
    foundation_cohort_id: str | None = None,
    foundation_require_full_history: bool = False,
    require_foundation_validation: bool = False,
) -> dict[str, Any]:
    """Build, validate, and package a complete main-release workflow."""
    from amb.benchmark.quality.lineage import write_lineage_audit
    from amb.benchmark.quality.main_dataset_acceptance import write_main_dataset_acceptance
    from amb.benchmark.quality.difficulty_calibration import write_difficulty_calibration_audit
    from amb.benchmark.quality.domain_construct_validity import write_domain_construct_validity_audit
    from amb.benchmark.quality.foundation_validation import write_foundation_protocol_audit
    from amb.benchmark.quality.probe_discriminativeness import write_probe_discriminativeness_audit
    from amb.benchmark.quality.question_craftsmanship import write_question_craftsmanship_audit
    from amb.benchmark.quality.query_construction import write_query_construction_audit
    from amb.benchmark.quality.release_intrinsic_sanity import write_release_intrinsic_sanity
    from amb.benchmark.quality.release_validation import validate_release_artifacts
    from amb.benchmark.release.public_docs import generate_public_release_docs

    if profile_id not in MAIN_RELEASE_PROFILES:
        raise ValueError(
            f"build_main_release_workflow expects one of {MAIN_RELEASE_PROFILES}, got {profile_id!r}"
        )
    if require_foundation_validation and not foundation_report_paths:
        raise ValueError("foundation_report_paths must be provided when require_foundation_validation is True")

    cfg = release_config or ReleaseConfig()
    workflow_seed = cfg.seed if seed is None else seed
    output_root = Path(output_dir)
    public_root = Path(public_output_dir) if public_output_dir is not None else Path(f"{output_root}_public")
    reports_root = Path(reports_dir)
    reports_root.mkdir(parents=True, exist_ok=True)

    if existing_release_manifest_path is not None:
        release_manifest_path = Path(existing_release_manifest_path)
        release_manifest = read_json(release_manifest_path)
        if release_manifest.get("package_type") == "public_release_export":
            raise ValueError(
                "existing_release_manifest_path must point to a private release manifest, not a public release export"
            )
        actual_profile_id = str(release_manifest.get("profile_id"))
        if actual_profile_id != profile_id:
            raise ValueError(
                f"existing release manifest profile_id mismatch: expected {profile_id!r}, got {actual_profile_id!r}"
            )
    else:
        release_manifest = build_profile_release_shards(profile_id, output_root, cfg)
        release_manifest_path = Path(release_manifest["manifest_path"])

    public_manifest = export_public_release_package(release_manifest_path, public_root)
    public_manifest_path = Path(public_manifest["manifest_path"])

    private_validation_path = reports_root / f"{_report_prefix(release_manifest['benchmark_id'])}_release_validation.json"
    public_validation_path = reports_root / f"{_report_prefix(release_manifest['benchmark_id'])}_public_release_validation.json"
    private_validation = validate_release_artifacts(release_manifest_path)
    public_validation = validate_release_artifacts(public_manifest_path)
    write_json(private_validation_path, private_validation)
    write_json(public_validation_path, public_validation)

    representative = None
    if not skip_representative_baselines:
        representative = build_representative_baseline_artifacts(
            public_manifest_path,
            split=representative_split,
            output_dir=reports_root,
            retrieval_k=retrieval_k,
            bootstrap_samples=bootstrap_samples,
            seed=workflow_seed,
        )

    intrinsic_sanity_path = reports_root / f"{_report_prefix(release_manifest['benchmark_id'])}_intrinsic_sanity.json"
    intrinsic_sanity = write_release_intrinsic_sanity(release_manifest_path, intrinsic_sanity_path)

    public_test_baselines = build_public_test_baseline_artifacts(
        public_manifest_path,
        split="public_test",
        output_dir=reports_root,
        retrieval_k=retrieval_k,
    )
    public_test_report_paths = _resolve_localized_report_paths(
        public_test_baselines["report_paths"],
        artifact=public_test_baselines,
    )
    public_result_slices = build_public_result_slice_artifacts(
        public_test_report_paths,
        benchmark_id=f"{release_manifest['benchmark_id']}-public_test",
        release_split="public_test",
        json_output_path=reports_root / f"{_report_prefix(release_manifest['benchmark_id'])}_public_test_required_slices.json",
        markdown_output_path=reports_root / f"{_report_prefix(release_manifest['benchmark_id'])}_public_test_required_slices.md",
    )

    failure_mode_split = public_test_baselines["release_split"]
    failure_mode_diagnostics = build_release_failure_mode_diagnostics(
        public_manifest_path,
        split=failure_mode_split,
        output_dir=reports_root,
        retrieval_k=retrieval_k,
        existing_report_paths=public_test_report_paths,
    )
    public_test_sanity = build_public_test_sanity_summary(
        public_test_baselines,
        failure_mode_diagnostics,
        output_path=reports_root / f"{_report_prefix(release_manifest['benchmark_id'])}_public_test_sanity.json",
    )
    public_test_summary = write_public_test_summary(
        reports_root / f"{_report_prefix(release_manifest['benchmark_id'])}_public_test_summary.md",
        benchmark_id=release_manifest["benchmark_id"],
        profile_id=profile_id,
        public_manifest_path=public_manifest_path,
        intrinsic_sanity_path=intrinsic_sanity_path,
        representative_artifact=representative,
        public_result_slices_path=public_result_slices["markdown_path"],
        public_test_sanity_path=public_test_sanity["path"],
        failure_mode_diagnostics_path=failure_mode_diagnostics["diagnostics_path"],
    )

    question_craftsmanship_path = (
        reports_root / f"{_report_prefix(release_manifest['benchmark_id'])}_question_craftsmanship_audit.json"
    )
    question_craftsmanship = write_question_craftsmanship_audit(
        question_craftsmanship_path,
        manifest_path=release_manifest_path,
    )

    probe_discriminativeness_path = (
        reports_root / f"{_report_prefix(release_manifest['benchmark_id'])}_probe_discriminativeness_audit.json"
    )
    probe_discriminativeness = write_probe_discriminativeness_audit(
        probe_discriminativeness_path,
        manifest_path=public_manifest_path,
        split=representative_split,
        reports_dir=reports_root,
    )

    difficulty_calibration_path = (
        reports_root / f"{_report_prefix(release_manifest['benchmark_id'])}_difficulty_calibration_audit.json"
    )
    difficulty_calibration = write_difficulty_calibration_audit(
        difficulty_calibration_path,
        manifest_path=public_manifest_path,
        split=representative_split,
        reports_dir=reports_root,
    )

    domain_construct_validity_path = (
        reports_root / f"{_report_prefix(release_manifest['benchmark_id'])}_domain_construct_validity_audit.json"
    )
    domain_construct_validity = write_domain_construct_validity_audit(
        domain_construct_validity_path,
        manifest_path=public_manifest_path,
        split=representative_split,
        reports_dir=reports_root,
    )

    foundation_validation = None
    foundation_validation_path = None
    if foundation_report_paths:
        foundation_validation_path = (
            reports_root / f"{_report_prefix(release_manifest['benchmark_id'])}_foundation_validation_audit.json"
        )
        foundation_validation = write_foundation_protocol_audit(
            foundation_validation_path,
            foundation_report_paths,
            expected_benchmark_id=foundation_expected_benchmark_id,
            cohort_id=foundation_cohort_id,
            require_full_history=foundation_require_full_history,
        )

    hidden_test_sanity = build_hidden_test_sanity_artifact(
        release_manifest_path,
        output_dir=reports_root,
        retrieval_k=retrieval_k,
    )

    query_construction_path = reports_root / f"{_report_prefix(release_manifest['benchmark_id'])}_query_construction_audit.json"
    query_construction = write_query_construction_audit(
        query_construction_path,
        manifest_path=release_manifest_path,
    )

    acceptance_path = reports_root / f"{_report_prefix(release_manifest['benchmark_id'])}_acceptance.json"
    acceptance = write_main_dataset_acceptance(
        release_manifest_path,
        acceptance_path,
        expected_benchmark_id=release_manifest["benchmark_id"],
        expected_profile_id=profile_id,
        run_release_validation=True,
        release_validation_report_path=private_validation_path,
        public_release_manifest_path=public_manifest_path,
        public_release_validation_report_path=public_validation_path,
        require_public_release_validation=True,
        require_all_counterfactual_axes=(profile_id == "main-v1-strict"),
        representative_reports_dir=reports_root,
        representative_split=representative_split,
        require_representative_baselines=not skip_representative_baselines,
        intrinsic_sanity_report_path=intrinsic_sanity_path,
        require_intrinsic_sanity=True,
        require_completed_human_audit=require_completed_human_audit,
        public_test_sanity_report_path=public_test_sanity["path"],
        require_public_test_sanity=True,
        public_result_slices_report_path=public_result_slices["json_path"],
        require_public_result_slices=True,
        hidden_test_sanity_report_path=hidden_test_sanity["path"],
        require_hidden_test_sanity=True,
        question_craftsmanship_report_path=question_craftsmanship_path,
        require_question_craftsmanship=True,
        query_construction_report_path=query_construction_path,
        require_query_construction=True,
        probe_discriminativeness_report_path=probe_discriminativeness_path,
        require_probe_discriminativeness=True,
        difficulty_calibration_report_path=difficulty_calibration_path,
        require_difficulty_calibration=True,
        domain_construct_validity_report_path=domain_construct_validity_path,
        require_domain_construct_validity=True,
        foundation_validation_report_path=foundation_validation_path,
        require_foundation_validation=require_foundation_validation,
        require_query_difficulty=True,
    )

    lineage_path = reports_root / f"{_report_prefix(release_manifest['benchmark_id'])}_lineage_audit.json"
    lineage = write_lineage_audit(lineage_path, manifest_path=release_manifest_path)

    human_audit_artifacts = _build_human_audit_release_artifacts(
        public_manifest_path,
        reports_root,
        profile_id=profile_id,
    )
    human_audit_package = human_audit_artifacts["task_package"]
    human_audit_bundle = human_audit_artifacts["evidence_bundle"]
    human_audit_bundle_verification = human_audit_artifacts["evidence_bundle_verification"]
    human_audit_bundle_verification_path = human_audit_artifacts["evidence_bundle_verification_path"]

    workflow_manifest = {
        "schema_version": "amst-main-release-workflow-v1",
        "profile_id": profile_id,
        "canonical_final_main_profile_id": CANONICAL_FINAL_MAIN_PROFILE_ID,
        "is_canonical_final_main_profile": is_canonical_final_main_profile(profile_id),
        "benchmark_id": release_manifest["benchmark_id"],
        "profile_role": release_manifest.get("profile_role"),
        "canonical_final_main": bool(release_manifest.get("canonical_final_main")),
        "release_manifest_path": str(release_manifest_path),
        "public_manifest_path": str(public_manifest_path),
        "reports_dir": str(reports_root),
        "workflow_seed": workflow_seed,
        "release_config": {
            "seed": cfg.seed,
            "dev_fraction": cfg.dev_fraction,
            "audit_fraction": cfg.audit_fraction,
            "hidden_fraction": cfg.hidden_fraction,
        },
        "artifacts": {
            "release_validation": str(private_validation_path),
            "public_release_validation": str(public_validation_path),
            "intrinsic_sanity": str(intrinsic_sanity_path),
            "public_test_baselines": public_test_baselines["report_paths"],
            "public_result_slices": str(public_result_slices["json_path"]),
            "public_result_slices_markdown": str(public_result_slices["markdown_path"]),
            "failure_mode_diagnostics": str(failure_mode_diagnostics["diagnostics_path"]),
            "public_test_sanity": str(public_test_sanity["path"]),
            "public_test_summary": str(public_test_summary["path"]),
            "hidden_test_sanity": str(hidden_test_sanity["path"]),
            "acceptance": str(acceptance_path),
            "question_craftsmanship_audit": str(question_craftsmanship_path),
            "query_construction_audit": str(query_construction_path),
            "probe_discriminativeness_audit": str(probe_discriminativeness_path),
            "difficulty_calibration_audit": str(difficulty_calibration_path),
            "domain_construct_validity_audit": str(domain_construct_validity_path),
            "foundation_validation_audit": str(foundation_validation_path) if foundation_validation_path is not None else None,
            "lineage_audit": str(lineage_path),
            "human_audit_task_package": human_audit_package,
            "human_audit_evidence_bundle": human_audit_bundle,
            "human_audit_evidence_bundle_verification": str(human_audit_bundle_verification_path),
        },
        "release_validation_ok": bool(private_validation.get("ok")),
        "public_release_validation_ok": bool(public_validation.get("ok")),
        "intrinsic_sanity_status": intrinsic_sanity.get("status"),
        "acceptance_status": acceptance.get("status"),
        "public_result_slices_status": public_result_slices.get("status"),
        "hidden_test_sanity_status": hidden_test_sanity.get("status"),
        "question_craftsmanship_status": question_craftsmanship.get("status"),
        "query_construction_status": query_construction.get("status"),
        "probe_discriminativeness_status": probe_discriminativeness.get("status"),
        "difficulty_calibration_status": difficulty_calibration.get("status"),
        "domain_construct_validity_status": domain_construct_validity.get("status"),
        "foundation_validation_status": foundation_validation.get("status") if foundation_validation is not None else None,
        "lineage_status": lineage.get("status"),
        "human_audit_bundle_status": human_audit_bundle_verification.get("status"),
        "representative_baselines": representative,
        "public_test_baselines": public_test_baselines,
        "public_result_slices": public_result_slices,
        "failure_mode_diagnostics": failure_mode_diagnostics,
        "public_test_sanity": public_test_sanity,
        "public_test_summary": public_test_summary,
        "hidden_test_sanity": hidden_test_sanity,
        "question_craftsmanship_audit": question_craftsmanship,
        "query_construction_audit": query_construction,
        "probe_discriminativeness_audit": probe_discriminativeness,
        "difficulty_calibration_audit": difficulty_calibration,
        "domain_construct_validity_audit": domain_construct_validity,
        "foundation_validation_audit": foundation_validation,
        "human_audit_evidence_bundle": human_audit_bundle,
        "human_audit_evidence_bundle_verification": human_audit_bundle_verification,
    }
    workflow_manifest["workflow_ok"] = (
        workflow_manifest["release_validation_ok"]
        and workflow_manifest["public_release_validation_ok"]
        and workflow_manifest["intrinsic_sanity_status"] == "passed"
        and workflow_manifest["acceptance_status"] == "passed"
        and workflow_manifest["public_result_slices_status"] == "passed"
        and workflow_manifest["hidden_test_sanity_status"] == "passed"
        and workflow_manifest["question_craftsmanship_status"] == "passed"
        and workflow_manifest["query_construction_status"] == "passed"
        and workflow_manifest["probe_discriminativeness_status"] == "passed"
        and workflow_manifest["difficulty_calibration_status"] == "passed"
        and workflow_manifest["domain_construct_validity_status"] == "passed"
        and workflow_manifest["foundation_validation_status"] in (None, "passed")
        and workflow_manifest["lineage_status"] == "passed"
        and workflow_manifest["human_audit_bundle_status"] == "passed"
    )
    # Export-time docs are generated before representative baseline reports and
    # downstream public-test summaries exist. Regenerate the public docs at the
    # end of the workflow so cited report values stay synchronized with the
    # final canonical artifacts.
    generate_public_release_docs(public_manifest, public_root, source_manifest_path=release_manifest_path)
    workflow_manifest_path = reports_root / f"{_profile_prefix(profile_id)}_build_workflow_manifest.json"
    workflow_manifest["workflow_manifest_path"] = str(workflow_manifest_path)
    workflow_manifest_payload = localize_report_contract(
        workflow_manifest,
        output_path=workflow_manifest_path,
        project_root_hints=(release_manifest_path, public_manifest_path, reports_root),
    )
    write_json(workflow_manifest_path, workflow_manifest_payload)
    return workflow_manifest


def _profile_prefix(profile_id: str) -> str:
    return profile_id.replace("-", "_").replace("amst_", "")


def _report_prefix(benchmark_id: str) -> str:
    return benchmark_id.replace("-", "_")


def _resolve_localized_report_paths(
    report_paths: dict[str, str | Path],
    *,
    artifact: dict[str, Any],
) -> dict[str, Path]:
    output_path = artifact.get("output_path")
    root_ref = artifact.get("root")
    project_root: Path | None = None
    if output_path is not None and root_ref is not None:
        raw_root = Path(str(root_ref))
        project_root = raw_root if raw_root.is_absolute() else (Path(str(output_path)).parent / raw_root).resolve()

    resolved: dict[str, Path] = {}
    for key, raw_path in report_paths.items():
        path = Path(raw_path)
        if path.is_absolute() or path.exists() or project_root is None:
            resolved[key] = path
        else:
            resolved[key] = project_root / path
    return resolved


def _build_human_audit_task_package(
    public_manifest_path: Path,
    output_dir: Path,
    *,
    annotator_ids: tuple[str, str] = ("ann_a", "ann_b"),
) -> dict[str, Any]:
    manifest = read_json(public_manifest_path)
    audit_plan = manifest.get("audit_plan", {}) if isinstance(manifest, dict) else {}
    template_files = audit_plan.get("audit_template_files")
    template_paths: list[Path] = []
    if isinstance(template_files, dict):
        for _, raw_path in sorted(template_files.items()):
            template_paths.append(_resolve_manifest_path(public_manifest_path.parent, str(raw_path)))
    template_file = audit_plan.get("audit_template_file")
    if template_file:
        template_paths.append(_resolve_manifest_path(public_manifest_path.parent, str(template_file)))
    if not template_paths:
        raise ValueError("public manifest audit_plan does not include audit template files")

    summary = write_double_annotation_task_package(template_paths, annotator_ids, output_dir)
    task_manifest_path = output_dir / "task_manifest.json"
    return {
        "task_manifest_file": str(task_manifest_path),
        "summary": summary,
    }


def _build_human_audit_release_artifacts(
    public_manifest_path: Path,
    reports_root: Path,
    *,
    profile_id: str,
    annotator_ids: tuple[str, str] = ("ann_a", "ann_b"),
) -> dict[str, Any]:
    profile_prefix = f"{_profile_prefix(profile_id)}_public"
    task_package = _build_human_audit_task_package(
        public_manifest_path,
        reports_root / "human_audit_tasks" / profile_prefix,
        annotator_ids=annotator_ids,
    )
    evidence_bundle = build_human_audit_evidence_bundle(
        reports_root / "human_audit_bundle" / profile_prefix,
        manifest_path=public_manifest_path,
        task_manifest_path=task_package["task_manifest_file"],
    )
    verification_path = Path(evidence_bundle["bundle_dir"]) / "bundle_verification.json"
    verification = verify_human_audit_evidence_bundle(evidence_bundle["bundle_dir"])
    write_json(verification_path, verification)
    return {
        "task_package": task_package,
        "evidence_bundle": evidence_bundle,
        "evidence_bundle_verification": verification,
        "evidence_bundle_verification_path": str(verification_path),
    }


def _resolve_manifest_path(manifest_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    return manifest_dir / path
