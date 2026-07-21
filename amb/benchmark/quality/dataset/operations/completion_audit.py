"""Completion audit against the AutoMemoryBench final implementation plan."""

from __future__ import annotations

import ast
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from amb.benchmark.generation.domains.counterfactual import (
    COUNTERFACTUAL_EDIT_BY_AXIS,
    COUNTERFACTUAL_PROBE_FLIPS,
    RECOMMENDED_COUNTERFACTUAL_AXES,
)
from amb.benchmark.quality.representative_analysis import summarize_representative_analysis
from amb.benchmark.schemas.io import read_json, write_json


@dataclass(frozen=True)
class AuditCheck:
    check_id: str
    requirement: str
    status: str
    evidence: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    notes: str | None = None


@dataclass(frozen=True)
class CompletionAudit:
    objective: str
    root: str
    status: str
    checks: tuple[AuditCheck, ...] = field(default_factory=tuple)

    @property
    def summary(self) -> dict[str, int]:
        counts = {"passed": 0, "partial": 0, "missing": 0}
        for check in self.checks:
            counts[check.status] = counts.get(check.status, 0) + 1
        return counts


OBJECTIVE = (
    "Implement the AutoMemoryBench final plan with reproducible main dataset construction, "
    "quality validation, release artifacts, evaluation baselines, and effectiveness evidence."
)
GIT_HYGIENE_SCHEMA_VERSION = "amst-git-hygiene-v1"
GIT_HYGIENE_PLAN_SCHEMA_VERSION = "amst-git-hygiene-plan-v1"

_GIT_HYGIENE_SOURCE_ROOTS = ("amb", "agent_memory_benchmark", "amst_real_clients", "tests")
_GIT_HYGIENE_CONFIG_ROOTS = ("configs",)
_GIT_HYGIENE_SOURCE_SUFFIXES = {".py"}
_GIT_HYGIENE_CONFIG_SUFFIXES = {".json", ".toml", ".yaml", ".yml"}
_GIT_HYGIENE_SCHEMA_SPEC_PREFIXES = ("amb/benchmark/schemas/json", "agent_memory_benchmark/schemas/json")
_GIT_HYGIENE_SCHEMA_SPEC_SUFFIXES = {".json"}
_GIT_HYGIENE_DOMAIN_PACK_PREFIXES = ("data/domain_packs",)
_GIT_HYGIENE_DOMAIN_PACK_SUFFIXES = {".json"}
_GIT_HYGIENE_PROJECT_CONFIG_FILES = ("pyproject.toml",)
_GIT_HYGIENE_EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache"}
_GIT_HYGIENE_EXCLUDED_PREFIXES = (".venv", ".tmp_")
_GIT_HYGIENE_EXCLUDED_GENERATED_PREFIXES = ("reports/git_hygiene/current/",)
_GIT_HYGIENE_SECRET_PATTERNS = (
    ("openai_sk", re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")),
    ("github_pat", re.compile(r"\bghp_[A-Za-z0-9]{16,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{16,}\b")),
    ("json_secret_literal", re.compile(r'"(?:api_key|token|password|secret)"\s*:\s*"[^"]{12,}"', re.IGNORECASE)),
    ("python_secret_literal", re.compile(r"\b(?:api_key|token|password|secret)\b\s*=\s*['\"][^'\"]{12,}['\"]", re.IGNORECASE)),
)
_TRACKED_RUNTIME_BATCH_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "tracked_quality_release_runtime",
        "Tracked quality, release, CLI, and evaluator files with unstaged changes that should be staged together or shelved together.",
        (
            "amb/benchmark/quality/",
            "agent_memory_benchmark/quality/",
            "amb/benchmark/release/",
            "agent_memory_benchmark/release/",
            "amb/benchmark/interfaces/commands/",
            "agent_memory_benchmark/interfaces/commands/",
            "amb/benchmark/evaluation/",
            "agent_memory_benchmark/evaluation/",
        ),
    ),
    (
        "tracked_generation_runtime",
        "Tracked generation, compiler, domain-pack, and probe-construction files with unstaged changes.",
        ("amb/benchmark/generation/", "agent_memory_benchmark/generation/"),
    ),
    (
        "tracked_analysis_metrics_runtime",
        "Tracked analysis and metric implementation files with unstaged changes.",
        (
            "amb/benchmark/analysis/",
            "agent_memory_benchmark/analysis/",
            "amb/benchmark/metrics/",
            "agent_memory_benchmark/metrics/",
        ),
    ),
    (
        "tracked_schema_runtime",
        "Tracked schema and IO model files with unstaged changes.",
        ("amb/benchmark/schemas/", "agent_memory_benchmark/schemas/"),
    ),
)
_TRACKED_TEST_BATCH_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "tracked_tests_human_audit_and_governance",
        "Tracked regression tests covering human-audit, completion-audit, evidence-readiness, and governance flows.",
        (
            "tests/test_annotation.py",
            "tests/test_completion_audit.py",
            "tests/test_evidence_readiness.py",
            "tests/test_objective_checklist.py",
        ),
    ),
    (
        "tracked_tests_release_and_real_systems",
        "Tracked regression tests covering release workflows, CLI execution, and real-system evidence paths.",
        (
            "tests/integration_fixtures.py",
            "tests/test_integration_config_validation.py",
            "tests/test_real_system_evidence.py",
            "tests/test_release.py",
            "tests/test_release_docs.py",
            "tests/test_run_agent_cli.py",
        ),
    ),
    (
        "tracked_tests_generation_and_acceptance",
        "Tracked regression tests covering generation, schemas, domain packs, and release-acceptance gates.",
        (
            "tests/test_domain_packs.py",
            "tests/test_generation.py",
            "tests/test_json_schemas.py",
            "tests/test_main_dataset_acceptance.py",
            "tests/test_quality.py",
        ),
    ),
    (
        "tracked_tests_metrics_and_scoring",
        "Tracked regression tests covering metrics and scoring behavior.",
        (
            "tests/test_metrics.py",
            "tests/test_scoring.py",
        ),
    ),
)
_TRACKED_BENCHMARK_ASSET_BATCH_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "tracked_schema_spec_assets",
        "Tracked benchmark JSON schemas with unstaged changes.",
        ("amb/benchmark/schemas/json/", "agent_memory_benchmark/schemas/json/"),
    ),
    (
        "tracked_domain_pack_assets",
        "Tracked domain-pack benchmark assets with unstaged changes.",
        ("data/domain_packs/",),
    ),
    (
        "tracked_project_config_assets",
        "Tracked top-level project configuration files with unstaged changes.",
        ("pyproject.toml",),
    ),
)


def build_completion_audit(root: str | Path = ".") -> CompletionAudit:
    project = Path(root)
    checks = (
        _check_schema(project),
        _check_domain_packs(project),
        _check_compiler_validator(project),
        _check_renderer_probe_factory(project),
        _check_runner_scorer_analyzer(project),
        _check_main_release(project),
        _check_challenge_release(project),
        _check_quarterly_hidden_refresh(project),
        _check_effectiveness_reports(project),
        _check_public_docs(project),
        _check_human_audit(project),
        _check_external_benchmarks(project),
        _check_real_memory_systems(project),
        _check_git_hygiene(project),
    )
    status = "complete" if all(check.status == "passed" for check in checks) else "incomplete"
    return CompletionAudit(objective=OBJECTIVE, root=str(project.resolve()), status=status, checks=checks)


def write_completion_audit(root: str | Path, output: str | Path) -> dict[str, Any]:
    project = Path(root)
    audit = build_completion_audit(root)
    incomplete_check_ids = [check.check_id for check in audit.checks if check.status != "passed"]
    partial_check_ids = [check.check_id for check in audit.checks if check.status == "partial"]
    missing_check_ids = [check.check_id for check in audit.checks if check.status == "missing"]
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = project / output_path
    payload = {
        "objective": audit.objective,
        "root": _artifact_root_ref(output_path.parent, project),
        "status": audit.status,
        "summary": audit.summary,
        "incomplete_check_ids": incomplete_check_ids,
        "partial_check_ids": partial_check_ids,
        "missing_check_ids": missing_check_ids,
        "checks": [asdict(check) for check in audit.checks],
    }
    write_json(output, payload)
    return payload


def build_git_hygiene_report(root: str | Path = ".") -> dict[str, Any]:
    project = Path(root)
    state = _collect_git_hygiene_state(project)
    missing = _git_hygiene_missing_entries(state)
    status = "passed"
    if not state["git_dir_exists"] or not state["gitignore_exists"]:
        status = "missing"
    elif missing:
        status = "partial"
    report = {
        "schema_version": GIT_HYGIENE_SCHEMA_VERSION,
        "root": str(project.resolve()),
        "status": status,
        "evidence": list(state["evidence"]),
        "missing": list(missing),
        "summary": {
            "tracked_modification_count": state["tracked_dirty_count"],
            "tracked_staged_count": state["tracked_staged_count"],
            "tracked_unstaged_count": state["tracked_unstaged_count"],
            "untracked_entry_count": state["untracked_entry_count"],
            "imported_untracked_source_count": len(state["imported_untracked"]),
            "import_edge_count": state["import_edge_count"],
            "tracked_unstaged_implementation_source_count": len(state["tracked_unstaged_impl_sources"]),
            "tracked_unstaged_real_client_source_count": len(state["tracked_unstaged_client_sources"]),
            "tracked_unstaged_test_source_count": len(state["tracked_unstaged_test_sources"]),
            "tracked_unstaged_runtime_config_count": len(state["tracked_unstaged_configs"]),
            "tracked_unstaged_schema_spec_count": len(state["tracked_unstaged_schema_specs"]),
            "tracked_unstaged_domain_pack_count": len(state["tracked_unstaged_domain_packs"]),
            "tracked_unstaged_project_config_count": len(state["tracked_unstaged_project_configs"]),
            "untracked_implementation_source_count": len(state["untracked_impl_sources"]),
            "untracked_real_client_source_count": len(state["untracked_client_sources"]),
            "untracked_test_source_count": len(state["untracked_test_sources"]),
            "untracked_runtime_config_count": len(state["untracked_configs"]),
            "untracked_schema_spec_count": len(state["untracked_schema_specs"]),
            "untracked_domain_pack_count": len(state["untracked_domain_packs"]),
            "untracked_project_config_count": len(state["untracked_project_configs"]),
            "credential_literal_path_count": len(state["path_risks"]),
            "credential_literal_match_count": sum(len(risks) for risks in state["path_risks"].values()),
            "git_error_count": len(state["git_errors"]),
        },
        "categories": {
            "tracked_unstaged_implementation_sources": sorted(state["tracked_unstaged_impl_sources"]),
            "tracked_unstaged_real_client_sources": sorted(state["tracked_unstaged_client_sources"]),
            "tracked_unstaged_test_sources": sorted(state["tracked_unstaged_test_sources"]),
            "tracked_unstaged_runtime_configs": sorted(state["tracked_unstaged_configs"]),
            "tracked_unstaged_schema_specs": sorted(state["tracked_unstaged_schema_specs"]),
            "tracked_unstaged_domain_packs": sorted(state["tracked_unstaged_domain_packs"]),
            "tracked_unstaged_project_configs": sorted(state["tracked_unstaged_project_configs"]),
            "imported_untracked_sources": [
                {"path": path, "imported_by": list(importers)}
                for path, importers in state["imported_untracked"].items()
            ],
            "untracked_implementation_sources": sorted(state["untracked_impl_sources"] - set(state["imported_untracked"])),
            "untracked_real_client_sources": sorted(state["untracked_client_sources"] - set(state["imported_untracked"])),
            "untracked_test_sources": sorted(state["untracked_test_sources"]),
            "untracked_runtime_configs": sorted(state["untracked_configs"]),
            "untracked_schema_specs": sorted(state["untracked_schema_specs"]),
            "untracked_domain_packs": sorted(state["untracked_domain_packs"]),
            "untracked_project_configs": sorted(state["untracked_project_configs"]),
            "credential_literal_paths": [
                {"path": path, "risks": list(risks)}
                for path, risks in sorted(state["path_risks"].items())
            ],
            "git_errors": list(state["git_errors"]),
        },
        "recommended_batches": _git_hygiene_stage_batches(state),
        "notes": _git_hygiene_notes(state),
    }
    return report


def write_git_hygiene_report(root: str | Path, output: str | Path) -> dict[str, Any]:
    project = Path(root)
    report = build_git_hygiene_report(root)
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = project / output_path
    report["root"] = _artifact_root_ref(output_path.parent, project)
    write_json(output, report)
    return report


def write_git_hygiene_plan(root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    project = Path(root)
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = project / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    report = build_git_hygiene_report(project)
    root_ref = _artifact_root_ref(output_path, project)
    report["root"] = root_ref
    batches = []
    for batch in report["recommended_batches"]:
        batch_id = str(batch["batch_id"])
        paths = [str(path) for path in batch.get("paths", [])]
        pathspec_file = output_path / f"{batch_id}.paths.txt"
        stage_script = output_path / f"{batch_id}.stage.sh"
        pathspec_file.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
        review_required = bool(batch.get("review_required"))
        warning_comment = (
            "# review_required: credential-literal scan flagged one or more paths in this batch\n"
            if review_required
            else ""
        )
        stage_script.write_text(
            "\n".join(
                (
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    warning_comment.rstrip("\n"),
                    f'git add --pathspec-from-file="{pathspec_file}"',
                    "",
                )
            ),
            encoding="utf-8",
        )
        stage_script.chmod(0o755)
        batches.append(
            {
                "batch_id": batch_id,
                "description": batch.get("description"),
                "path_count": len(paths),
                "risk_level": batch.get("risk_level"),
                "review_required": review_required,
                "risk_entries": batch.get("risk_entries", []),
                "pathspec_file": str(pathspec_file),
                "stage_script": str(stage_script),
                "stage_command": f'git add --pathspec-from-file="{pathspec_file}"',
                "safe_stage_command": None if review_required else f'git add --pathspec-from-file="{pathspec_file}"',
            }
        )
    stage_all_script = output_path / "stage_all.sh"
    stage_all_lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for batch in batches:
        stage_all_lines.append(batch["stage_command"])
    stage_all_lines.append("")
    stage_all_script.write_text("\n".join(stage_all_lines), encoding="utf-8")
    stage_all_script.chmod(0o755)
    readme = output_path / "README.md"
    readme_lines = [
        "# Git Hygiene Stage Plan",
        "",
        f"source_report: `{root_ref}`",
        "",
        "Batches:",
    ]
    for batch in batches:
        readme_lines.extend(
            (
                f"- `{batch['batch_id']}`: {batch['path_count']} paths",
                f"  command: `{batch['stage_command']}`",
                f"  script: `{batch['stage_script']}`",
            )
        )
    readme_lines.extend(("", f"stage_all: `{stage_all_script}`", ""))
    readme.write_text("\n".join(readme_lines), encoding="utf-8")
    manifest = {
        "schema_version": GIT_HYGIENE_PLAN_SCHEMA_VERSION,
        "root": root_ref,
        "source_report": report,
        "output_dir": str(output_path),
        "num_batches": len(batches),
        "batches": batches,
        "stage_all_script": str(stage_all_script),
        "readme_file": str(readme),
    }
    write_json(output_path / "manifest.json", manifest)
    return manifest


def _check_schema(root: Path) -> AuditCheck:
    required = (
        "amb/benchmark/schemas/models.py",
        "amb/benchmark/schemas/state.py",
        "amb/benchmark/schemas/json/benchmark.schema.json",
        "amb/benchmark/schemas/json/prediction.schema.json",
        "amb/benchmark/schemas/json/report.schema.json",
        "amb/benchmark/schemas/json/release_manifest.schema.json",
    )
    return _file_check(
        root,
        "schema",
        "All benchmark, prediction, report, state-contract, and release schemas are versioned and machine-readable.",
        required,
    )


def _check_domain_packs(root: Path) -> AuditCheck:
    pack_dir = root / "data/domain_packs"
    packs = sorted(pack_dir.glob("*.json")) if pack_dir.exists() else []
    required_domains = {
        "personal_assistant",
        "office_collaboration",
        "coding_agent",
        "customer_support",
        "research_assistant",
        "devops_workflow",
        "education_tutoring",
        "multi_party_collaboration",
    }
    domains = {path.stem for path in packs}
    missing = sorted(required_domains - domains)
    malformed: list[str] = []
    expected_axes = list(RECOMMENDED_COUNTERFACTUAL_AXES)
    for path in packs:
        payload = read_json(path)
        rules = payload.get("counterfactual_rules", [])
        axes = [str(item.get("axis", "")) for item in rules if isinstance(item, dict)]
        if axes != expected_axes:
            malformed.append(f"{path.stem}:counterfactual_axes")
            continue
        for item in rules:
            axis = str(item.get("axis"))
            expected_edit = COUNTERFACTUAL_EDIT_BY_AXIS.get(axis)
            expected_flips = list(COUNTERFACTUAL_PROBE_FLIPS.get(axis, ()))
            if item.get("expected_counterfactual_edit") != expected_edit:
                malformed.append(f"{path.stem}:{axis}:expected_counterfactual_edit")
            if item.get("expected_probe_flip") != expected_flips:
                malformed.append(f"{path.stem}:{axis}:expected_probe_flip")
    missing_items = tuple(missing + malformed)
    return AuditCheck(
        "domain_packs",
        "Eight application-domain packs are available for the main dataset and expose the canonical five counterfactual axes.",
        "passed" if not missing_items else "missing",
        evidence=tuple(_rel(path, root) for path in packs),
        missing=missing_items,
    )


def _check_compiler_validator(root: Path) -> AuditCheck:
    required = (
        "amb/benchmark/generation/compilers/events.py",
        "amb/benchmark/generation/compilers/memories.py",
        "amb/benchmark/generation/compilers/edges.py",
        "amb/benchmark/generation/compilers/state_contracts.py",
        "amb/benchmark/generation/domains/counterfactual.py",
        "amb/benchmark/quality/validation.py",
        "amb/benchmark/quality/gates.py",
        "amb/benchmark/release/public_test_sanity.py",
        "amb/benchmark/quality/release_validation.py",
        "tests/test_quality.py",
        "tests/test_release.py",
        "tests/test_public_test_sanity.py",
    )
    return _file_check(
        root,
        "compiler_validator",
        "Compiler and validator cover event graphs, memories, state contracts, counterfactual edits, and quality gates.",
        required,
    )


def _check_renderer_probe_factory(root: Path) -> AuditCheck:
    required = (
        "amb/benchmark/generation/renderers/conversation.py",
        "amb/benchmark/generation/renderers/platform.py",
        "amb/benchmark/generation/renderers/tool.py",
        "amb/benchmark/generation/renderers/document.py",
        "amb/benchmark/generation/renderers/adversarial.py",
        "amb/benchmark/generation/probes/factory.py",
    )
    missing = _missing_files(root, required)
    probe_types = _probe_types_from_public_dev(root)
    renderer_coverage = _renderer_coverage_from_public_dev(root)
    required_probe_types = {
        "write_probe",
        "retrieval_probe",
        "answer_probe",
        "update_probe",
        "compression_probe",
        "forget_probe",
        "governance_probe",
        "tool_probe",
        "planning_probe",
        "evolution_probe",
        "governed_transfer_probe",
        "scope_contrast_probe",
        "conflict_resolution_probe",
        "cross_session_synthesis_probe",
        "adversarial_state_synthesis_probe",
        "temporal_causal_reconciliation_probe",
        "policy_temporal_state_probe",
    }
    required_renderers = {"platform", "tool", "document", "adversarial"}
    missing_probes = sorted(required_probe_types - probe_types) if probe_types else sorted(required_probe_types)
    missing_renderers = sorted(required_renderers - renderer_coverage) if renderer_coverage else sorted(required_renderers)
    status = "passed"
    missing_items = list(missing)
    notes = None
    if missing_probes:
        missing_items.extend(f"probe_type:{name}" for name in missing_probes)
    if missing_renderers:
        missing_items.extend(f"renderer_coverage:{name}" for name in missing_renderers)
    if missing or missing_probes or missing_renderers:
        status = "missing"
    else:
        notes = "Conversation, platform, tool, document, adversarial renderers, coverage evidence, and strict core probe types are present."
    return AuditCheck(
        "renderer_probe_factory",
        "Renderer and ProbeFactory expose natural traces and all ten required probe classes.",
        status,
        evidence=(
            tuple(str(path) for path in required if (root / path).exists())
            + tuple(f"probe_type:{name}" for name in sorted(probe_types))
            + tuple(f"renderer_coverage:{name}" for name in sorted(renderer_coverage))
        ),
        missing=tuple(missing_items),
        notes=notes,
    )


def _check_runner_scorer_analyzer(root: Path) -> AuditCheck:
    required = (
        "amb/benchmark/evaluation/adapters.py",
        "amb/benchmark/evaluation/runner.py",
        "amb/benchmark/evaluation/baselines.py",
        "amb/benchmark/evaluation/scoring.py",
        "amb/benchmark/analysis/report_analysis.py",
        "amb/benchmark/leaderboard/summary.py",
        "tests/test_runner.py",
        "tests/test_scoring.py",
        "tests/test_analyzer.py",
        "tests/test_leaderboard.py",
    )
    missing = _missing_files(root, required)
    status = "passed" if not missing else "missing"
    notes = "Deterministic diagnostic modes are implemented; full third-party System-Memory adapters are tracked separately." if not missing else None
    return AuditCheck(
        "runner_scorer_analyzer",
        "Runner, scorer, analyzer, deterministic baselines, statistics, and leaderboard summaries are implemented.",
        status,
        evidence=tuple(path for path in required if (root / path).exists()),
        missing=tuple(missing),
        notes=notes,
    )


def _check_main_release(root: Path) -> AuditCheck:
    manifest_path = root / "data/releases/amst_main_v1_strict/manifest.json"
    public_manifest_path = root / "data/releases/amst_main_v1_strict_public/manifest.json"
    validation_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_main_v1_strict_release_validation.json"
    )
    public_validation_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_main_v1_strict_public_release_validation.json"
    )
    acceptance_current_path = root / "reports/examples/amst_main_v1_strict_acceptance_current.json"
    acceptance_static_path = root / "reports/examples/amst_main_v1_strict_acceptance.json"
    acceptance_path = acceptance_current_path if acceptance_current_path.exists() else acceptance_static_path
    lineage_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_main_v1_strict_lineage_audit.json"
    )
    public_test_sanity_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_main_v1_strict_public_test_sanity.json"
    )
    public_result_slices_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_main_v1_strict_public_test_required_slices.json"
    )
    hidden_test_sanity_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_main_v1_strict_hidden_test_sanity.json"
    )
    foundation_validation_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_main_v1_strict_foundation_validation_audit.json"
    )
    question_craftsmanship_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_main_v1_strict_question_craftsmanship_audit.json"
    )
    query_construction_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_main_v1_strict_query_construction_audit.json"
    )
    probe_discriminativeness_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_main_v1_strict_probe_discriminativeness_audit.json"
    )
    difficulty_calibration_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_main_v1_strict_difficulty_calibration_audit.json"
    )
    domain_construct_validity_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_main_v1_strict_domain_construct_validity_audit.json"
    )
    core_paths = (
        manifest_path,
        public_manifest_path,
        validation_path,
        public_validation_path,
    )
    supporting_paths = (
        acceptance_path,
        lineage_path,
        public_test_sanity_path,
        public_result_slices_path,
        hidden_test_sanity_path,
        foundation_validation_path,
        question_craftsmanship_path,
        query_construction_path,
        probe_discriminativeness_path,
        difficulty_calibration_path,
        domain_construct_validity_path,
    )
    if any(not path.exists() for path in core_paths):
        return AuditCheck(
            "main_release",
            "Canonical final main release has 8 domains, 1200 base scenarios, 7200 case variants, full recommended counterfactual-axis coverage, public/hidden splits, and validation evidence.",
            "missing",
            missing=tuple(_rel(path, root) for path in core_paths if not path.exists()),
        )
    missing_supporting = tuple(_rel(path, root) for path in supporting_paths if not path.exists())
    if missing_supporting:
        return AuditCheck(
            "main_release",
            "Canonical final main release has 8 domains, 1200 base scenarios, 7200 case variants, full recommended counterfactual-axis coverage, public/hidden splits, and validation evidence.",
            "partial",
            evidence=tuple(
                _rel(path, root)
                for path in (*core_paths, *supporting_paths)
                if path.exists()
            ),
            missing=missing_supporting,
        )
    manifest = read_json(manifest_path)
    public_manifest = read_json(public_manifest_path)
    validation = read_json(validation_path)
    public_validation = read_json(public_validation_path)
    acceptance = read_json(acceptance_path)
    lineage = read_json(lineage_path)
    public_test_sanity = read_json(public_test_sanity_path) if public_test_sanity_path.exists() else {}
    public_result_slices = read_json(public_result_slices_path)
    hidden_test_sanity = read_json(hidden_test_sanity_path)
    foundation_validation = read_json(foundation_validation_path)
    question_craftsmanship = read_json(question_craftsmanship_path) if question_craftsmanship_path.exists() else {}
    query_construction = read_json(query_construction_path)
    probe_discriminativeness = read_json(probe_discriminativeness_path)
    difficulty_calibration = read_json(difficulty_calibration_path)
    domain_construct_validity = read_json(domain_construct_validity_path)
    expected = manifest.get("expected_generation_summary", {})
    split_reports = manifest.get("split_reports", {})
    axis_coverage = expected.get("counterfactual_axis_coverage", {})
    covered_axes = set(axis_coverage.get("covered_axes", []))
    ok = (
        expected.get("base_scenarios") == 1200
        and expected.get("num_cases") == 7200
        and expected.get("counterfactual_variants_per_base") == 5
        and bool(axis_coverage.get("covers_all_recommended_axes"))
        and {"current_value", "deletion_state", "authorization_state", "tool_result", "role_project_boundary"} <= covered_axes
        and split_reports.get("public_test", {}).get("num_cases") == 3600
        and split_reports.get("hidden_test", {}).get("num_cases") == 1440
        and bool(manifest.get("split_files", {}).get("hidden_test"))
        and public_manifest.get("split_files", {}).get("hidden_test") == {}
        and validation.get("ok") is True
        and public_validation.get("ok") is True
        and acceptance.get("status") == "passed"
        and lineage.get("status") == "passed"
        and public_test_sanity.get("status") == "passed"
        and public_result_slices.get("status") == "passed"
        and hidden_test_sanity.get("status") == "passed"
        and foundation_validation.get("status") == "passed"
        and question_craftsmanship.get("status") == "passed"
        and query_construction.get("status") == "passed"
        and probe_discriminativeness.get("status") == "passed"
        and difficulty_calibration.get("status") == "passed"
        and domain_construct_validity.get("status") == "passed"
    )
    return AuditCheck(
        "main_release",
        "Canonical final main release has 8 domains, 1200 base scenarios, 7200 case variants, full recommended counterfactual-axis coverage, public/hidden splits, and validation evidence.",
        "passed" if ok else "partial",
        evidence=(
            _rel(manifest_path, root),
            _rel(public_manifest_path, root),
            _rel(validation_path, root),
            _rel(public_validation_path, root),
            _rel(acceptance_path, root),
            _rel(lineage_path, root),
            _rel(public_test_sanity_path, root),
            _rel(public_result_slices_path, root),
            _rel(hidden_test_sanity_path, root),
            _rel(foundation_validation_path, root),
            _rel(question_craftsmanship_path, root),
            _rel(query_construction_path, root),
            _rel(probe_discriminativeness_path, root),
            _rel(difficulty_calibration_path, root),
            _rel(domain_construct_validity_path, root),
        ),
        missing=()
        if ok
        else (
            "expected canonical strict main-release counts/full counterfactual axis coverage/internal hidden split plus scrubbed public hidden split/release validations ok/acceptance passed/lineage passed/public_test_sanity passed/public_result_slices passed/hidden_test_sanity passed/foundation_validation passed/question_craftsmanship passed/query_construction passed/probe_discriminativeness passed/difficulty_calibration passed/domain_construct_validity passed",
        ),
    )


def _check_challenge_release(root: Path) -> AuditCheck:
    manifest_path = root / "data/releases/amst_challenge_v1/manifest.json"
    public_manifest_path = root / "data/releases/amst_challenge_v1_public/manifest.json"
    validation_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_challenge_v1_release_validation.json"
    )
    public_validation_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_challenge_v1_public_release_validation.json"
    )
    acceptance_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_challenge_v1_acceptance.json"
    )
    question_craftsmanship_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_challenge_v1_question_craftsmanship_audit.json"
    )
    query_construction_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_challenge_v1_query_construction_audit.json"
    )
    probe_discriminativeness_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_challenge_v1_probe_discriminativeness_audit.json"
    )
    difficulty_calibration_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_challenge_v1_difficulty_calibration_audit.json"
    )
    domain_construct_validity_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_challenge_v1_domain_construct_validity_audit.json"
    )
    core_paths = (
        manifest_path,
        public_manifest_path,
        validation_path,
        public_validation_path,
    )
    supporting_paths = (
        acceptance_path,
        question_craftsmanship_path,
        query_construction_path,
        probe_discriminativeness_path,
        difficulty_calibration_path,
        domain_construct_validity_path,
    )
    if any(not path.exists() for path in core_paths):
        return AuditCheck(
            "challenge_release",
            "Challenge release exists with stronger counterfactual axes and public package validation.",
            "missing",
            missing=tuple(_rel(path, root) for path in core_paths if not path.exists()),
        )
    missing_supporting = tuple(_rel(path, root) for path in supporting_paths if not path.exists())
    if missing_supporting:
        return AuditCheck(
            "challenge_release",
            "Challenge release exists with stronger counterfactual axes and public package validation.",
            "partial",
            evidence=tuple(_rel(path, root) for path in (*core_paths, *supporting_paths) if path.exists()),
            missing=missing_supporting,
        )
    manifest = read_json(manifest_path)
    public_manifest = read_json(public_manifest_path)
    validation = read_json(validation_path)
    public_validation = read_json(public_validation_path)
    acceptance = read_json(acceptance_path)
    question_craftsmanship = read_json(question_craftsmanship_path)
    query_construction = read_json(query_construction_path)
    probe_discriminativeness = read_json(probe_discriminativeness_path)
    difficulty_calibration = read_json(difficulty_calibration_path)
    domain_construct_validity = read_json(domain_construct_validity_path)
    expected = manifest.get("expected_generation_summary", {})
    axis_coverage = expected.get("counterfactual_axis_coverage", {})
    covered_axes = set(axis_coverage.get("covered_axes", []))
    ok = (
        expected.get("base_scenarios") == 240
        and expected.get("num_cases") == 1440
        and expected.get("counterfactual_variants_per_base") == 5
        and bool(axis_coverage.get("covers_all_recommended_axes"))
        and {"current_value", "deletion_state", "authorization_state", "tool_result", "role_project_boundary"} <= covered_axes
        and bool(manifest.get("split_files", {}).get("hidden_test"))
        and public_manifest.get("split_files", {}).get("hidden_test") == {}
        and validation.get("ok") is True
        and public_validation.get("ok") is True
        and acceptance.get("status") == "passed"
        and question_craftsmanship.get("status") == "passed"
        and query_construction.get("status") == "passed"
        and probe_discriminativeness.get("status") == "passed"
        and difficulty_calibration.get("status") == "passed"
        and domain_construct_validity.get("status") == "passed"
    )
    return AuditCheck(
        "challenge_release",
        "Challenge release exists with stronger counterfactual axes, withheld hidden split, and challenge-specific question-quality validation.",
        "passed" if ok else "partial",
        evidence=(
            _rel(manifest_path, root),
            _rel(public_manifest_path, root),
            _rel(validation_path, root),
            _rel(public_validation_path, root),
            _rel(acceptance_path, root),
            _rel(question_craftsmanship_path, root),
            _rel(query_construction_path, root),
            _rel(probe_discriminativeness_path, root),
            _rel(difficulty_calibration_path, root),
            _rel(domain_construct_validity_path, root),
        ),
        missing=()
        if ok
        else (
            "expected challenge-v1 counts/extended counterfactual axes/internal hidden split plus scrubbed public hidden split/release validations ok/acceptance passed/question_craftsmanship passed/query_construction passed/probe_discriminativeness passed/difficulty_calibration passed/domain_construct_validity passed",
        ),
    )


def _check_quarterly_hidden_refresh(root: Path) -> AuditCheck:
    manifest_path = root / "data/releases/amst_hidden_quarterly_v1/manifest.json"
    validation_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_hidden_quarterly_v1_release_validation.json"
    )
    intrinsic_sanity_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_hidden_quarterly_v1_release_intrinsic_sanity.json"
    )
    acceptance_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_hidden_quarterly_v1_acceptance.json"
    )
    question_craftsmanship_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_hidden_quarterly_v1_question_craftsmanship_audit.json"
    )
    query_construction_path = _prefer_current_example_artifact(
        root / "reports/examples/amst_hidden_quarterly_v1_query_construction_audit.json"
    )
    core_paths = (
        manifest_path,
        validation_path,
        intrinsic_sanity_path,
        acceptance_path,
        question_craftsmanship_path,
        query_construction_path,
    )
    if any(not path.exists() for path in core_paths):
        return AuditCheck(
            "quarterly_hidden_refresh",
            "Quarterly private hidden refresh package exists with 300 hidden scenarios from the same compiler family and hidden-only quality validation.",
            "missing" if not manifest_path.exists() else "partial",
            evidence=tuple(_rel(path, root) for path in core_paths if path.exists()),
            missing=tuple(_rel(path, root) for path in core_paths if not path.exists()),
        )
    manifest = read_json(manifest_path)
    validation = read_json(validation_path)
    intrinsic_sanity = read_json(intrinsic_sanity_path)
    acceptance = read_json(acceptance_path)
    question_craftsmanship = read_json(question_craftsmanship_path)
    query_construction = read_json(query_construction_path)
    refresh = manifest.get("quarterly_hidden_refresh", {})
    split_report = manifest.get("split_reports", {}).get("hidden_test", {})
    ok = (
        manifest.get("package_type") == "private_leaderboard_package"
        and int(refresh.get("num_hidden_scenarios", 0)) == 300
        and bool(refresh.get("same_compiler_family"))
        and int(split_report.get("num_cases", 0)) > 0
        and validation.get("ok") is True
        and intrinsic_sanity.get("status") == "passed"
        and acceptance.get("status") == "passed"
        and question_craftsmanship.get("status") == "passed"
        and query_construction.get("status") == "passed"
    )
    return AuditCheck(
        "quarterly_hidden_refresh",
        "Quarterly private hidden refresh package exists with 300 hidden scenarios from the same compiler family and hidden-only quality validation.",
        "passed" if ok else "partial",
        evidence=(
            _rel(manifest_path, root),
            _rel(validation_path, root),
            _rel(intrinsic_sanity_path, root),
            _rel(acceptance_path, root),
            _rel(question_craftsmanship_path, root),
            _rel(query_construction_path, root),
        ),
        missing=()
        if ok
        else (
            "expected private_leaderboard_package/300 hidden scenarios/same compiler family/non-empty hidden split/release_validation ok/release_intrinsic_sanity passed/acceptance passed/question_craftsmanship passed/query_construction passed",
        ),
    )


def _check_effectiveness_reports(root: Path) -> AuditCheck:
    reports = {
        "no_memory": _prefer_current_example_artifact(
            root / "reports/examples/amst_main_v1_strict_public_dev_no_memory_report.json"
        ),
        "full_history": _prefer_current_example_artifact(
            root / "reports/examples/amst_main_v1_strict_public_dev_full_history_report.json"
        ),
        "dense_memory": _prefer_current_example_artifact(
            root / "reports/examples/amst_main_v1_strict_public_dev_dense_memory_report.json"
        ),
        "hybrid_memory": _prefer_current_example_artifact(
            root / "reports/examples/amst_main_v1_strict_public_dev_hybrid_memory_report.json"
        ),
        "graph_memory": _prefer_current_example_artifact(
            root / "reports/examples/amst_main_v1_strict_public_dev_graph_memory_report.json"
        ),
        "oracle_memory": _prefer_current_example_artifact(
            root / "reports/examples/amst_main_v1_strict_public_dev_oracle_memory_report.json"
        ),
        "analysis": _prefer_current_example_artifact(
            root / "reports/examples/amst_main_v1_strict_public_dev_representative_baselines_analysis.json"
        ),
        "leaderboard": _prefer_current_example_artifact(
            root / "reports/examples/amst_main_v1_strict_public_dev_leaderboard.json"
        ),
        "failure_mode_diagnostics": _prefer_current_example_artifact(
            root / "reports/examples/amst_main_v1_strict_public_dev_failure_mode_diagnostics.json"
        ),
        "public_test_sanity": _prefer_current_example_artifact(
            root / "reports/examples/amst_main_v1_strict_public_test_sanity.json"
        ),
        "public_test_failure_mode_diagnostics": _prefer_current_example_artifact(
            root / "reports/examples/amst_main_v1_strict_public_test_failure_mode_diagnostics.json"
        ),
    }
    missing = [name for name, path in reports.items() if not path.exists()]
    if missing:
        return AuditCheck(
            "effectiveness_reports",
            "Representative baselines show solvability, memory dependence, cost, and system differentiation.",
            "missing",
            evidence=tuple(_rel(path, root) for path in reports.values() if path.exists()),
            missing=tuple(missing),
        )
    aggregates = {name: read_json(path).get("aggregate", {}) for name, path in reports.items() if name not in {"analysis", "leaderboard"}}
    analysis = read_json(reports["analysis"])
    if not isinstance(analysis.get("weight_sensitivity"), dict) or not analysis.get("weight_sensitivity", {}).get("profiles"):
        from amb.benchmark.analysis import build_weight_sensitivity_analysis

        analysis = dict(analysis)
        analysis["weight_sensitivity"] = build_weight_sensitivity_analysis(
            [
                (str(reports[name]), read_json(reports[name]))
                for name in ("no_memory", "full_history", "graph_memory", "oracle_memory")
            ]
        )
    leaderboard = read_json(reports["leaderboard"])
    failure_mode_diagnostics = read_json(reports["failure_mode_diagnostics"])
    public_test_sanity = read_json(reports["public_test_sanity"])
    public_test_failure_mode_diagnostics = read_json(reports["public_test_failure_mode_diagnostics"])
    analysis_summary = summarize_representative_analysis(analysis)
    oracle_amq = aggregates["oracle_memory"].get("lifecycle.amq", 0)
    no_memory_task = aggregates["no_memory"].get("task.task_success", 1)
    full_history_tokens = aggregates["full_history"].get("efficiency.input_tokens", 0)
    dense_amq = aggregates["dense_memory"].get("lifecycle.amq", 0)
    dense_recall = aggregates["dense_memory"].get("retrieval.recall_at_k", 0)
    hybrid_amq = aggregates["hybrid_memory"].get("lifecycle.amq", 0)
    graph_amq = aggregates["graph_memory"].get("lifecycle.amq", 0)
    graph_task = aggregates["graph_memory"].get("task.task_success", 0)
    no_memory_amq = aggregates["no_memory"].get("lifecycle.amq", 0)
    no_memory_recall = aggregates["no_memory"].get("retrieval.recall_at_k", 0)
    graph_recall = aggregates["graph_memory"].get("retrieval.recall_at_k", 0)
    graph_tokens = aggregates["graph_memory"].get("efficiency.input_tokens", 0)
    leaderboard_rows = leaderboard.get("rows", [])
    leaderboard_top = leaderboard_rows[0].get("system_id") if leaderboard_rows else None
    leaderboard_second = leaderboard_rows[1].get("system_id") if len(leaderboard_rows) > 1 else None
    analysis_pairs = {
        (str(item.get("baseline_system_id")), str(item.get("candidate_system_id")))
        for item in analysis.get("comparisons", [])
    }
    required_pairs = {
        ("no_memory", "full_history"),
        ("no_memory", "graph_memory"),
        ("no_memory", "oracle_memory"),
        ("full_history", "graph_memory"),
        ("full_history", "oracle_memory"),
        ("graph_memory", "oracle_memory"),
    }
    graph_memory_advantage = (
        graph_amq > no_memory_amq
        and graph_recall > no_memory_recall
        and graph_task >= no_memory_task
    )
    failure_mode_ok = (
        failure_mode_diagnostics.get("status") == "passed"
        or (
            failure_mode_diagnostics.get("status") == "failed"
            and tuple(failure_mode_diagnostics.get("errors") or ()) == ("graph_procedural_advantage",)
        )
    )
    ok = (
        oracle_amq >= 0.85
        and no_memory_task < 0.35
        and graph_memory_advantage
        and dense_recall > 0.0
        and hybrid_amq > dense_amq
        and full_history_tokens > graph_tokens
        and leaderboard_top == "oracle_memory"
        and leaderboard_second == "graph_memory"
        and required_pairs <= analysis_pairs
        and analysis_summary.get("bootstrap_samples_sufficient") is True
        and analysis_summary.get("report_bootstrap_cis_present") is True
        and analysis_summary.get("pairwise_stats_complete") is True
        and analysis_summary.get("key_memory_gains_statistically_visible") is True
        and analysis_summary.get("weight_sensitivity_profiles_complete") is True
        and analysis_summary.get("oracle_top_rank_stable_under_weight_shifts") is True
        and analysis_summary.get("key_memory_order_stable_under_weight_shifts") is True
        and failure_mode_ok
        and public_test_sanity.get("status") == "passed"
        and public_test_failure_mode_diagnostics.get("status") == "passed"
    )
    status = "passed" if ok else "partial"
    missing_items = () if ok else (
        "expected canonical strict public_dev representative baselines to show oracle high score/no-memory low task/graph AMQ+retrieval advantage/full_history higher cost/leaderboard oracle->graph ordering/bootstrap CI and paired-test evidence/weight-sensitive AMQ ordering robustness/required pairwise analysis/failure diagnostics passed-or-diagnostic-only/public_test_sanity passed",
    )
    notes = (
        f"full_history_input_tokens={full_history_tokens}; "
        f"graph_task_success={graph_task}; no_memory_task_success={no_memory_task}"
    )
    return AuditCheck(
        "effectiveness_reports",
        "Representative baselines show solvability, memory dependence, cost, and system differentiation.",
        status,
        evidence=tuple(_rel(path, root) for path in reports.values()),
        missing=missing_items,
        notes=notes,
    )


def _check_public_docs(root: Path) -> AuditCheck:
    required = (
        "data/releases/amst_main_v1_public/docs/dataset_card.md",
        "data/releases/amst_main_v1_public/docs/benchmark_card.md",
        "data/releases/amst_main_v1_public/docs/submission_guide.md",
        "data/releases/amst_main_v1_public/docs/memory_system_reporting_template.md",
        "data/releases/amst_main_v1_public/docs/annotation_guideline.md",
        "data/releases/amst_main_v1_public/docs/governance_privacy_statement.md",
        "data/releases/amst_main_v1_public/docs/reproducibility_checklist.md",
        "data/releases/amst_main_v1_strict_public/docs/dataset_card.md",
        "data/releases/amst_main_v1_strict_public/docs/benchmark_card.md",
        "data/releases/amst_main_v1_strict_public/docs/submission_guide.md",
        "data/releases/amst_main_v1_strict_public/docs/memory_system_reporting_template.md",
        "data/releases/amst_main_v1_strict_public/docs/annotation_guideline.md",
        "data/releases/amst_main_v1_strict_public/docs/governance_privacy_statement.md",
        "data/releases/amst_main_v1_strict_public/docs/reproducibility_checklist.md",
    )
    return _file_check(
        root,
        "public_docs",
        "Public releases include dataset card, benchmark card, submission guide, reporting template, annotation guide, governance statement, and reproducibility checklist.",
        required,
    )


def _check_human_audit(root: Path) -> AuditCheck:
    manifest_path = root / "data/releases/amst_main_v1_strict_public/manifest.json"
    tooling = (
        "amb/benchmark/quality/annotation.py",
        "tests/test_annotation.py",
    )
    bundle_artifacts = (
        "reports/human_audit_bundle/current/bundle_manifest.json",
        "reports/human_audit_bundle/current/handoff_manifest.json",
        "reports/human_audit_bundle/current/bundle_verification.json",
        "reports/human_audit_bundle/current/return_inbox_sync.json",
        "reports/human_audit_bundle/current/return_inbox_watch.json",
    )
    missing_tooling = _missing_files(root, tooling)
    if not manifest_path.exists():
        return AuditCheck(
            "human_audit",
            "Audit subset includes completed double annotations and agreement metrics.",
            "missing" if missing_tooling else "partial",
            evidence=tuple(path for path in tooling if (root / path).exists()),
            missing=tuple((*missing_tooling, _rel(manifest_path, root))),
            notes="Annotation tooling exists, but release audit templates/manifest are not available in this checkout." if not missing_tooling else None,
        )
    audit_plan = read_json(manifest_path).get("audit_plan", {})
    completed = audit_plan.get("human_audit_status") == "completed"
    has_metrics = _valid_agreement_metrics(audit_plan.get("agreement_metrics"))
    has_templates = bool(audit_plan.get("audit_template_files") or audit_plan.get("audit_template_file"))
    missing_bundle_artifacts = tuple(path for path in bundle_artifacts if not (root / path).exists())
    bundle_validation_errors = _human_audit_bundle_validation_errors(root)
    verification_errors: tuple[str, ...] = ()
    if completed and has_metrics:
        from amb.benchmark.quality.human_audit import verify_manifest_human_audit

        verification = verify_manifest_human_audit(manifest_path)
        verification_errors = tuple(verification["errors"])
    if (
        completed
        and has_metrics
        and not verification_errors
        and not missing_tooling
        and not missing_bundle_artifacts
        and not bundle_validation_errors
    ):
        status = "passed"
        missing = ()
        notes = f"human_audit_status={audit_plan.get('human_audit_status')}"
    elif completed and has_metrics:
        status = "partial"
        missing = tuple(
            dict.fromkeys(
                (
                    *missing_tooling,
                    *missing_bundle_artifacts,
                    *bundle_validation_errors,
                    "valid completed human audit evidence",
                    *verification_errors[:5],
                )
            )
        )
        notes = "Completed human-audit metadata is present, but the canonical bundle/operator evidence does not fully verify."
    elif not missing_tooling and has_templates:
        status = "partial"
        missing = tuple(
            dict.fromkeys(
                (
                    *missing_bundle_artifacts,
                    *bundle_validation_errors,
                    "completed double annotations",
                    "agreement metrics",
                )
            )
        )
        if not missing_bundle_artifacts and not bundle_validation_errors:
            notes = (
                f"human_audit_status={audit_plan.get('human_audit_status')}; "
                "canonical bundle/handoff workflow is ready, waiting on returned human evidence."
            )
        else:
            notes = (
                f"human_audit_status={audit_plan.get('human_audit_status')}; "
                "audit templates exist, but the canonical bundle/handoff workflow is incomplete."
            )
    else:
        status = "missing"
        missing = tuple(
            dict.fromkeys(
                (
                    *missing_tooling,
                    *missing_bundle_artifacts,
                    *bundle_validation_errors,
                    "audit templates",
                    "completed double annotations",
                    "agreement metrics",
                )
            )
        )
        notes = f"human_audit_status={audit_plan.get('human_audit_status')}"
    return AuditCheck(
        "human_audit",
        "Canonical final main audit subset includes completed double annotations and agreement metrics.",
        status,
        evidence=tuple(path for path in (*tooling, _rel(manifest_path, root), *bundle_artifacts) if (root / path).exists()),
        missing=missing,
        notes=notes,
    )


def _check_external_benchmarks(root: Path) -> AuditCheck:
    tool_groups = (
        ("amb/benchmark/analysis/external_scores.py", "agent_memory_benchmark/analysis/external_scores.py"),
        ("amb/benchmark/analysis/external_correlation.py", "agent_memory_benchmark/analysis/external_correlation.py"),
        ("amb/benchmark/analysis/external_protocol.py", "agent_memory_benchmark/analysis/external_protocol.py"),
        ("tests/test_external_correlation.py",),
    )
    reports = (
        "reports/external/locomo_correlation.json",
        "reports/external/longmemeval_correlation.json",
        "reports/external/mem2actbench_correlation.json",
        "reports/external/memoryagentbench_correlation.json",
    )
    validation_artifact = "reports/external/evidence_validation.json"
    expansion_artifacts = (
        "reports/external/cohort_expansion_plan.json",
        "reports/external/cohort_expansion_validation.json",
        "reports/external/handoff_manifest.json",
        "reports/external/return_inbox_sync.json",
        "reports/external/return_inbox_watch.json",
    )
    missing_tools = _missing_path_groups(root, tool_groups)
    missing_reports = _missing_files(root, reports)
    existing_reports = tuple(root / path for path in reports if (root / path).exists())
    invalid_reports = tuple(path for path in reports if (root / path).exists() and not _valid_external_correlation_report(root / path))
    if missing_tools:
        status = "missing"
        missing = tuple(missing_tools)
        notes = "External benchmark correlation tooling is not implemented."
    elif missing_reports:
        status = "partial"
        missing = tuple(missing_reports)
        notes = "Correlation tooling is implemented; real external benchmark run artifacts are still missing."
    elif invalid_reports:
        status = "partial"
        missing = invalid_reports
        notes = "External correlation files exist but do not contain enough common systems or rank-correlation evidence."
    else:
        from amb.benchmark.analysis.external_protocol import validate_external_evidence_set

        set_validation = validate_external_evidence_set(
            existing_reports,
            min_shared_systems=3,
            min_shared_control_systems=1,
            min_shared_real_memory_systems=1,
        )
        validation_errors = _external_validation_errors(root, validation_artifact)
        missing_expansion_artifacts = tuple(path for path in expansion_artifacts if not (root / path).exists())
        expansion_validation_errors = _external_expansion_validation_errors(root)
        combined_errors = tuple(
            dict.fromkeys(
                (
                    *missing_expansion_artifacts,
                    *validation_errors,
                    *expansion_validation_errors,
                    *(str(error) for error in set_validation["errors"]),
                )
            )
        )
        if combined_errors:
            status = "partial"
            missing = combined_errors
            if not missing_expansion_artifacts and not expansion_validation_errors:
                notes = (
                    "External correlation files and canonical expansion/handoff artifacts are present, but the "
                    "same-system evidence set is still below the completion-gate threshold and must include both "
                    "control anchors and real memory systems."
                )
            else:
                notes = (
                    "External correlation files exist, but the canonical same-system evidence set and/or the "
                    "expansion/handoff contract are still incomplete."
                )
        else:
            status = "passed"
            missing = ()
            notes = (
                "External correlation reports, validation artifact, and expansion/handoff contract are present for "
                "the same 3+ system cohort, including control anchors and at least one real memory system."
            )
    return AuditCheck(
        "external_benchmark_correlation",
        "Same systems are run on external memory benchmarks and AutoMemoryBench with rank-correlation evidence.",
        status,
        evidence=_existing_path_group_paths(root, tool_groups)
        + tuple(path for path in (*reports, validation_artifact, *expansion_artifacts) if (root / path).exists()),
        missing=missing,
        notes=notes,
    )


def _check_real_memory_systems(root: Path) -> AuditCheck:
    adapter_groups = (
        ("amb/benchmark/integrations/alignment.py", "agent_memory_benchmark/integrations/alignment.py"),
        ("amb/benchmark/integrations/base.py", "agent_memory_benchmark/integrations/base.py"),
        ("amb/benchmark/integrations/mem0.py", "agent_memory_benchmark/integrations/mem0.py"),
        ("amb/benchmark/integrations/letta.py", "agent_memory_benchmark/integrations/letta.py"),
        ("amb/benchmark/integrations/langmem.py", "agent_memory_benchmark/integrations/langmem.py"),
        ("amb/benchmark/integrations/zep_graphiti.py", "agent_memory_benchmark/integrations/zep_graphiti.py"),
        ("amb/benchmark/integrations/factory.py", "agent_memory_benchmark/integrations/factory.py"),
        ("tests/test_integrations.py",),
        ("tests/test_run_agent_cli.py",),
    )
    fixed_reports = (
        "reports/examples/amst_main_v1_public_dev_mem0_report.json",
        "reports/examples/amst_main_v1_public_dev_letta_report.json",
        "reports/examples/amst_main_v1_public_dev_langmem_report.json",
        "reports/examples/amst_main_v1_public_dev_zep_graphiti_report.json",
    )
    missing_adapters = _missing_path_groups(root, adapter_groups)
    report_paths, matrix_evidence, matrix_errors = _real_system_report_paths(root, fixed_reports)
    missing_reports = tuple(path for path in fixed_reports if not (root / path).exists()) if not matrix_evidence else ()
    invalid_reports = tuple(_rel(path, root) for path in report_paths if not _valid_real_system_report(path))
    if missing_adapters:
        status = "missing"
        missing = tuple(missing_adapters)
        notes = "External memory-system wrappers are not import-ready."
    elif matrix_errors:
        status = "partial"
        missing = matrix_errors
        notes = "Real-system matrix summary exists but is incomplete or invalid."
    elif missing_reports:
        status = "partial"
        missing = missing_reports
        notes = "Wrappers and fake-client smoke tests are present; real public-dev/public-test system run reports are still missing."
    elif invalid_reports:
        status = "partial"
        missing = invalid_reports
        notes = "Real-system report files exist but are incomplete or contain missing/extra predictions."
    else:
        status = "passed"
        missing = ()
        notes = "Wrappers and real-system run reports are present."
    return AuditCheck(
        "real_memory_system_integrations",
        "Representative external memory systems are wrapped and actually run on AutoMemoryBench splits.",
        status,
        evidence=_existing_path_group_paths(root, adapter_groups)
        + tuple(_rel(path, root) for path in report_paths)
        + matrix_evidence,
        missing=missing,
        notes=notes,
    )


def _check_git_hygiene(root: Path) -> AuditCheck:
    report = build_git_hygiene_report(root)
    return AuditCheck(
        "git_hygiene",
        "Repository metadata, ignore rules, and critical implementation assets are tracked or intentionally ignored.",
        report["status"],
        evidence=tuple(report["evidence"]),
        missing=tuple(report["missing"]),
        notes=str(report["notes"]) if report.get("notes") else None,
    )


def _collect_git_hygiene_state(root: Path) -> dict[str, Any]:
    git_dir = root / ".git"
    ignore = root / ".gitignore"
    evidence: list[str] = []
    if git_dir.exists():
        evidence.append(".git")
    if ignore.exists():
        evidence.append(".gitignore")

    tracked_paths = _git_ls_files(root)
    status_lines = _git_status_short(root)
    git_errors: list[str] = []
    if tracked_paths is None:
        git_errors.append("git:ls-files")
    if status_lines is None:
        git_errors.append("git:status")

    if status_lines is not None:
        tracked_dirty_count = 0
        tracked_staged_count = 0
        tracked_unstaged_count = 0
        untracked_entry_count = 0
        for line in status_lines:
            if line.startswith("??"):
                rel_text = line[3:].strip()
                if rel_text and _skip_git_hygiene_path(Path(rel_text)):
                    continue
                untracked_entry_count += 1
                continue
            entry = _parse_git_status_entry(line)
            if not entry:
                continue
            relative = Path(str(entry["path"]))
            if _skip_git_hygiene_path(relative):
                continue
            tracked_dirty_count += 1
            tracked_staged_count += int(bool(entry["staged"]))
            tracked_unstaged_count += int(bool(entry["unstaged"]))
        evidence.append(
            f"git_status:{'dirty' if (tracked_dirty_count or untracked_entry_count) else 'clean'}"
        )
    else:
        tracked_dirty_count = 0
        tracked_staged_count = 0
        tracked_unstaged_count = 0
        untracked_entry_count = 0

    imported_untracked: dict[str, tuple[str, ...]] = {}
    tracked_unstaged_impl_sources: set[str] = set()
    tracked_unstaged_client_sources: set[str] = set()
    tracked_unstaged_test_sources: set[str] = set()
    tracked_unstaged_configs: set[str] = set()
    tracked_unstaged_schema_specs: set[str] = set()
    tracked_unstaged_domain_packs: set[str] = set()
    tracked_unstaged_project_configs: set[str] = set()
    untracked_impl_sources: set[str] = set()
    untracked_client_sources: set[str] = set()
    untracked_test_sources: set[str] = set()
    untracked_configs: set[str] = set()
    untracked_schema_specs: set[str] = set()
    untracked_domain_packs: set[str] = set()
    untracked_project_configs: set[str] = set()
    import_edge_count = 0
    if status_lines is not None:
        tracked_unstaged_impl_sources = _tracked_git_hygiene_paths_from_status(
            status_lines,
            prefix="agent_memory_benchmark",
            suffixes=_GIT_HYGIENE_SOURCE_SUFFIXES,
            change_kind="unstaged",
        )
        tracked_unstaged_client_sources = _tracked_git_hygiene_paths_from_status(
            status_lines,
            prefix="amst_real_clients",
            suffixes=_GIT_HYGIENE_SOURCE_SUFFIXES,
            change_kind="unstaged",
        )
        tracked_unstaged_test_sources = _tracked_git_hygiene_paths_from_status(
            status_lines,
            prefix="tests",
            suffixes=_GIT_HYGIENE_SOURCE_SUFFIXES,
            change_kind="unstaged",
        )
        tracked_unstaged_configs = _tracked_git_hygiene_paths_from_status(
            status_lines,
            prefix="configs",
            suffixes=_GIT_HYGIENE_CONFIG_SUFFIXES,
            change_kind="unstaged",
        )
        for prefix in _GIT_HYGIENE_SCHEMA_SPEC_PREFIXES:
            tracked_unstaged_schema_specs |= _tracked_git_hygiene_paths_from_status(
                status_lines,
                prefix=prefix,
                suffixes=_GIT_HYGIENE_SCHEMA_SPEC_SUFFIXES,
                change_kind="unstaged",
            )
        tracked_unstaged_domain_packs = _tracked_git_hygiene_paths_from_status(
            status_lines,
            prefix="data/domain_packs",
            suffixes=_GIT_HYGIENE_DOMAIN_PACK_SUFFIXES,
            change_kind="unstaged",
        )
        tracked_unstaged_project_configs = _tracked_git_hygiene_explicit_paths_from_status(
            status_lines,
            allowed_paths=_GIT_HYGIENE_PROJECT_CONFIG_FILES,
            change_kind="unstaged",
        )
    if tracked_paths is not None:
        untracked_impl_sources = _untracked_git_hygiene_paths(
            root,
            tracked_paths,
            prefix="agent_memory_benchmark",
            suffixes=_GIT_HYGIENE_SOURCE_SUFFIXES,
        )
        untracked_client_sources = _untracked_git_hygiene_paths(
            root,
            tracked_paths,
            prefix="amst_real_clients",
            suffixes=_GIT_HYGIENE_SOURCE_SUFFIXES,
        )
        untracked_test_sources = _untracked_git_hygiene_paths(
            root,
            tracked_paths,
            prefix="tests",
            suffixes=_GIT_HYGIENE_SOURCE_SUFFIXES,
        )
        untracked_configs = _untracked_git_hygiene_paths(
            root,
            tracked_paths,
            prefix="configs",
            suffixes=_GIT_HYGIENE_CONFIG_SUFFIXES,
        )
        for prefix in _GIT_HYGIENE_SCHEMA_SPEC_PREFIXES:
            untracked_schema_specs |= _untracked_git_hygiene_paths(
                root,
                tracked_paths,
                prefix=prefix,
                suffixes=_GIT_HYGIENE_SCHEMA_SPEC_SUFFIXES,
            )
        untracked_domain_packs = _untracked_git_hygiene_paths(
            root,
            tracked_paths,
            prefix="data/domain_packs",
            suffixes=_GIT_HYGIENE_DOMAIN_PACK_SUFFIXES,
        )
        untracked_project_configs = _untracked_git_hygiene_explicit_paths(
            root,
            tracked_paths,
            allowed_paths=_GIT_HYGIENE_PROJECT_CONFIG_FILES,
        )
        imported_untracked = _tracked_to_untracked_imports(
            root,
            tracked_paths,
            untracked_impl_sources | untracked_client_sources,
        )
        import_edge_count = sum(len(importers) for importers in imported_untracked.values())
    tracked_candidates = (
        tracked_unstaged_impl_sources
        | tracked_unstaged_client_sources
        | tracked_unstaged_test_sources
        | tracked_unstaged_configs
        | tracked_unstaged_schema_specs
        | tracked_unstaged_domain_packs
        | tracked_unstaged_project_configs
        | untracked_impl_sources
        | untracked_client_sources
        | untracked_test_sources
        | untracked_configs
        | untracked_schema_specs
        | untracked_domain_packs
        | untracked_project_configs
    )
    path_risks = _scan_git_hygiene_path_risks(root, tracked_candidates)

    return {
        "git_dir_exists": git_dir.exists(),
        "gitignore_exists": ignore.exists(),
        "evidence": tuple(evidence),
        "tracked_paths": tracked_paths,
        "status_lines": status_lines,
        "git_errors": tuple(git_errors),
        "tracked_dirty_count": tracked_dirty_count,
        "tracked_staged_count": tracked_staged_count,
        "tracked_unstaged_count": tracked_unstaged_count,
        "untracked_entry_count": untracked_entry_count,
        "tracked_unstaged_impl_sources": tracked_unstaged_impl_sources,
        "tracked_unstaged_client_sources": tracked_unstaged_client_sources,
        "tracked_unstaged_test_sources": tracked_unstaged_test_sources,
        "tracked_unstaged_configs": tracked_unstaged_configs,
        "tracked_unstaged_schema_specs": tracked_unstaged_schema_specs,
        "tracked_unstaged_domain_packs": tracked_unstaged_domain_packs,
        "tracked_unstaged_project_configs": tracked_unstaged_project_configs,
        "imported_untracked": imported_untracked,
        "untracked_impl_sources": untracked_impl_sources,
        "untracked_client_sources": untracked_client_sources,
        "untracked_test_sources": untracked_test_sources,
        "untracked_configs": untracked_configs,
        "untracked_schema_specs": untracked_schema_specs,
        "untracked_domain_packs": untracked_domain_packs,
        "untracked_project_configs": untracked_project_configs,
        "path_risks": path_risks,
        "import_edge_count": import_edge_count,
    }


def _git_hygiene_missing_entries(state: dict[str, Any]) -> tuple[str, ...]:
    missing: list[str] = []
    if not state["git_dir_exists"]:
        missing.append(".git")
    if not state["gitignore_exists"]:
        missing.append(".gitignore")
    for path in sorted(state["tracked_unstaged_impl_sources"]):
        missing.append(f"tracked_unstaged_implementation_source:{path}")
    for path in sorted(state["tracked_unstaged_client_sources"]):
        missing.append(f"tracked_unstaged_real_client_source:{path}")
    for path in sorted(state["tracked_unstaged_test_sources"]):
        missing.append(f"tracked_unstaged_test_source:{path}")
    for path in sorted(state["tracked_unstaged_configs"]):
        missing.append(f"tracked_unstaged_runtime_config:{path}")
    for path in sorted(state["tracked_unstaged_schema_specs"]):
        missing.append(f"tracked_unstaged_schema_spec:{path}")
    for path in sorted(state["tracked_unstaged_domain_packs"]):
        missing.append(f"tracked_unstaged_domain_pack:{path}")
    for path in sorted(state["tracked_unstaged_project_configs"]):
        missing.append(f"tracked_unstaged_project_config:{path}")
    imported_paths = set(state["imported_untracked"])
    for path in sorted(imported_paths):
        importers = ",".join(state["imported_untracked"][path])
        missing.append(f"untracked_imported_source:{path}<-{importers}")
    for path in sorted(state["untracked_impl_sources"] - imported_paths):
        missing.append(f"untracked_implementation_source:{path}")
    for path in sorted(state["untracked_client_sources"] - imported_paths):
        missing.append(f"untracked_real_client_source:{path}")
    for path in sorted(state["untracked_test_sources"]):
        missing.append(f"untracked_test_source:{path}")
    for path in sorted(state["untracked_configs"]):
        missing.append(f"untracked_runtime_config:{path}")
    for path in sorted(state["untracked_schema_specs"]):
        missing.append(f"untracked_schema_spec:{path}")
    for path in sorted(state["untracked_domain_packs"]):
        missing.append(f"untracked_domain_pack:{path}")
    for path in sorted(state["untracked_project_configs"]):
        missing.append(f"untracked_project_config:{path}")
    missing.extend(state["git_errors"])
    return tuple(missing)


def _git_hygiene_notes(state: dict[str, Any]) -> str:
    note_parts = [
        "git status currently shows "
        f"{state['tracked_dirty_count']} tracked modifications and "
        f"{state['untracked_entry_count']} untracked entries. "
        f"Of the tracked changes, {state['tracked_staged_count']} are staged and "
        f"{state['tracked_unstaged_count']} remain unstaged."
    ]
    if state["tracked_paths"] is not None:
        note_parts.append(
            "Critical git-hygiene audit found "
            f"{len(state['tracked_unstaged_impl_sources'])} unstaged benchmark modules, "
            f"{len(state['tracked_unstaged_client_sources'])} unstaged real-client modules, "
            f"{len(state['tracked_unstaged_test_sources'])} unstaged test modules, and "
            f"{len(state['tracked_unstaged_configs'])} unstaged runtime configs, "
            f"{len(state['tracked_unstaged_schema_specs'])} unstaged schema specs, "
            f"{len(state['tracked_unstaged_domain_packs'])} unstaged domain packs, and "
            f"{len(state['tracked_unstaged_project_configs'])} unstaged project configs. "
            "It also found "
            f"{len(state['untracked_impl_sources'])} untracked benchmark modules, "
            f"{len(state['untracked_client_sources'])} untracked real-client modules, "
            f"{len(state['untracked_test_sources'])} untracked test modules, and "
            f"{len(state['untracked_configs'])} untracked runtime configs, "
            f"{len(state['untracked_schema_specs'])} untracked schema specs, "
            f"{len(state['untracked_domain_packs'])} untracked domain packs, and "
            f"{len(state['untracked_project_configs'])} untracked project configs."
        )
        if state["imported_untracked"]:
            note_parts.append(
                "Tracked Python code currently has "
                f"{state['import_edge_count']} import edges into "
                f"{len(state['imported_untracked'])} untracked source files."
            )
        if state["path_risks"]:
            note_parts.append(
                f"Credential-literal scan flagged {len(state['path_risks'])} critical paths for manual review before staging."
            )
    if state["git_errors"]:
        note_parts.append("Git command introspection failed for part of the audit.")
    return " ".join(note_parts)


def _git_hygiene_stage_batches(state: dict[str, Any]) -> list[dict[str, Any]]:
    tracked_runtime_paths = sorted(
        state["tracked_unstaged_impl_sources"]
        | state["tracked_unstaged_client_sources"]
        | state["tracked_unstaged_configs"]
    )
    tracked_test_paths = sorted(state["tracked_unstaged_test_sources"])
    imported_paths = sorted(state["imported_untracked"])
    non_imported_impl = sorted(state["untracked_impl_sources"] - set(state["imported_untracked"]))
    non_imported_clients = sorted(state["untracked_client_sources"] - set(state["imported_untracked"]))
    runtime_configs = sorted(state["untracked_configs"])
    tracked_asset_paths = sorted(
        state["tracked_unstaged_schema_specs"]
        | state["tracked_unstaged_domain_packs"]
        | state["tracked_unstaged_project_configs"]
    )
    path_risks = state["path_risks"]
    untracked_tests = sorted(state["untracked_test_sources"])
    untracked_asset_paths = sorted(
        state["untracked_schema_specs"]
        | state["untracked_domain_packs"]
        | state["untracked_project_configs"]
    )
    batches: list[dict[str, Any]] = []
    batches.extend(
        _git_hygiene_partitioned_batches(
            tracked_runtime_paths,
            _TRACKED_RUNTIME_BATCH_SPECS,
            fallback_batch_id="tracked_runtime_misc",
            fallback_description="Tracked benchmark/runtime files with unstaged changes that do not match a narrower subsystem batch.",
            path_risks=path_risks,
        )
    )
    batches.extend(
        _git_hygiene_partitioned_batches(
            tracked_test_paths,
            _TRACKED_TEST_BATCH_SPECS,
            fallback_batch_id="tracked_tests_misc",
            fallback_description="Tracked regression tests with unstaged changes that do not match a narrower feature-family batch.",
            path_risks=path_risks,
        )
    )
    batches.extend(
        _git_hygiene_partitioned_batches(
            tracked_asset_paths,
            _TRACKED_BENCHMARK_ASSET_BATCH_SPECS,
            fallback_batch_id="tracked_benchmark_assets_misc",
            fallback_description="Tracked non-Python benchmark assets with unstaged changes that do not match a narrower asset-family batch.",
            path_risks=path_risks,
        )
    )
    if imported_paths:
        batches.append(
            _git_hygiene_batch(
                "imported_runtime_sources",
                "Highest-priority sources already imported by tracked Python modules.",
                imported_paths,
                path_risks=path_risks,
            )
        )
    if non_imported_impl:
        batches.append(
            _git_hygiene_batch(
                "benchmark_support_sources",
                "Benchmark implementation sources present on disk but not yet tracked.",
                non_imported_impl,
                path_risks=path_risks,
            )
        )
    real_runtime_paths = [*non_imported_clients, *runtime_configs]
    if real_runtime_paths:
        batches.append(
            _git_hygiene_batch(
                "real_clients_and_configs",
                "Real-system client adapters and runtime config artifacts required by canonical runs.",
                real_runtime_paths,
                path_risks=path_risks,
            )
        )
    if untracked_tests:
        batches.append(
            _git_hygiene_batch(
                "regression_tests",
                "Untracked regression tests covering the new benchmark functionality.",
                untracked_tests,
                path_risks=path_risks,
            )
        )
    if untracked_asset_paths:
        batches.append(
            _git_hygiene_batch(
                "benchmark_assets",
                "Untracked benchmark assets such as schemas, domain packs, or project config that are required by the current implementation.",
                untracked_asset_paths,
                path_risks=path_risks,
            )
        )
    return batches


def _git_hygiene_partitioned_batches(
    paths: Iterable[str],
    specs: Iterable[tuple[str, str, tuple[str, ...]]],
    *,
    fallback_batch_id: str,
    fallback_description: str,
    path_risks: dict[str, tuple[str, ...]],
) -> list[dict[str, Any]]:
    remaining = sorted(set(paths))
    if not remaining:
        return []
    batches: list[dict[str, Any]] = []
    unmatched = set(remaining)
    for batch_id, description, prefixes in specs:
        group = [path for path in remaining if path in unmatched and _path_matches_any_prefix(path, prefixes)]
        if not group:
            continue
        unmatched.difference_update(group)
        batches.append(_git_hygiene_batch(batch_id, description, group, path_risks=path_risks))
    fallback_paths = sorted(path for path in remaining if path in unmatched)
    if fallback_paths:
        batches.append(
            _git_hygiene_batch(
                fallback_batch_id,
                fallback_description,
                fallback_paths,
                path_risks=path_risks,
            )
        )
    return batches


def _path_matches_any_prefix(path: str, prefixes: Iterable[str]) -> bool:
    for prefix in prefixes:
        if path == prefix or path.startswith(prefix):
            return True
    return False


def _git_hygiene_batch(
    batch_id: str,
    description: str,
    paths: list[str],
    *,
    path_risks: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    risk_entries: list[dict[str, Any]] = []
    for path in paths:
        risks = path_risks.get(path, ())
        if risks:
            risk_entries.append({"path": path, "risks": list(risks)})
    stage_command = _git_add_command(paths)
    safe_stage_command = None if risk_entries else stage_command
    return {
        "batch_id": batch_id,
        "description": description,
        "path_count": len(paths),
        "paths": list(paths),
        "risk_level": "review" if risk_entries else "low",
        "review_required": bool(risk_entries),
        "risk_entries": risk_entries,
        "suggested_command": stage_command,
        "safe_stage_command": safe_stage_command,
    }


def _git_add_command(paths: Iterable[str]) -> str:
    items = list(paths)
    return "git add " + " ".join(items)


def _scan_git_hygiene_path_risks(root: Path, paths: Iterable[str]) -> dict[str, tuple[str, ...]]:
    flagged: dict[str, tuple[str, ...]] = {}
    for path_text in sorted(set(paths)):
        path = root / path_text
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        risks = tuple(name for name, pattern in _GIT_HYGIENE_SECRET_PATTERNS if pattern.search(content))
        if risks:
            flagged[path_text] = risks
    return flagged


def _git_ls_files(root: Path) -> set[str] | None:
    lines = _git_command_lines(root, "ls-files")
    if lines is None:
        return None
    return set(lines)


def _git_status_short(root: Path) -> tuple[str, ...] | None:
    lines = _git_command_lines(root, "status", "--short")
    if lines is None:
        return None
    return tuple(line for line in lines if line.strip())


def _git_status_has_staged_change(line: str) -> bool:
    entry = _parse_git_status_entry(line)
    return bool(entry and entry["staged"])


def _git_status_has_unstaged_change(line: str) -> bool:
    entry = _parse_git_status_entry(line)
    return bool(entry and entry["unstaged"])


def _parse_git_status_entry(line: str) -> dict[str, Any] | None:
    if len(line) < 3:
        return None
    x = line[0]
    y = line[1]
    if x == "?" and y == "?":
        return None
    path_text = line[3:] if len(line) > 3 else ""
    if " -> " in path_text:
        path_text = path_text.split(" -> ", 1)[1]
    return {
        "staged": x not in {" ", "?"},
        "unstaged": y not in {" ", "?"},
        "path": path_text.strip(),
    }


def _git_command_lines(root: Path, *args: str) -> list[str] | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.splitlines()


def _tracked_git_hygiene_paths_from_status(
    status_lines: Iterable[str],
    *,
    prefix: str,
    suffixes: set[str],
    change_kind: str,
) -> set[str]:
    tracked: set[str] = set()
    for line in status_lines:
        entry = _parse_git_status_entry(line)
        if not entry:
            continue
        if change_kind == "staged" and not entry["staged"]:
            continue
        if change_kind == "unstaged" and not entry["unstaged"]:
            continue
        rel_text = str(entry["path"])
        if not rel_text:
            continue
        relative = Path(rel_text)
        if not _path_matches_any_prefix(relative.as_posix(), (prefix,)):
            continue
        if _skip_git_hygiene_path(relative):
            continue
        if relative.suffix not in suffixes:
            continue
        tracked.add(relative.as_posix())
    return tracked


def _tracked_git_hygiene_explicit_paths_from_status(
    status_lines: Iterable[str],
    *,
    allowed_paths: Iterable[str],
    change_kind: str,
) -> set[str]:
    allowed = set(allowed_paths)
    tracked: set[str] = set()
    for line in status_lines:
        entry = _parse_git_status_entry(line)
        if not entry:
            continue
        if change_kind == "staged" and not entry["staged"]:
            continue
        if change_kind == "unstaged" and not entry["unstaged"]:
            continue
        rel_text = str(entry["path"]).strip()
        if rel_text in allowed:
            tracked.add(rel_text)
    return tracked


def _untracked_git_hygiene_paths(root: Path, tracked_paths: set[str], prefix: str, suffixes: set[str]) -> set[str]:
    base_path = root / prefix
    if not base_path.exists():
        return set()
    untracked: set[str] = set()
    for path in base_path.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if _skip_git_hygiene_path(relative):
            continue
        rel_text = relative.as_posix()
        if path.suffix not in suffixes:
            continue
        if rel_text not in tracked_paths:
            untracked.add(rel_text)
    return untracked


def _untracked_git_hygiene_explicit_paths(root: Path, tracked_paths: set[str], allowed_paths: Iterable[str]) -> set[str]:
    untracked: set[str] = set()
    for rel_text in allowed_paths:
        if rel_text in tracked_paths:
            continue
        path = root / rel_text
        if path.exists() and path.is_file():
            untracked.add(rel_text)
    return untracked


def _skip_git_hygiene_path(relative: Path) -> bool:
    rel_text = relative.as_posix()
    if any(rel_text.startswith(prefix) for prefix in _GIT_HYGIENE_EXCLUDED_GENERATED_PREFIXES):
        return True
    if relative.parts[:2] == ("reports", "examples") and relative.name.endswith("_current.json"):
        return True
    return any(
        part in _GIT_HYGIENE_EXCLUDED_PARTS or part.startswith(_GIT_HYGIENE_EXCLUDED_PREFIXES)
        for part in relative.parts
    )


def _tracked_to_untracked_imports(
    root: Path,
    tracked_paths: set[str],
    untracked_source_paths: set[str],
) -> dict[str, tuple[str, ...]]:
    importers_by_target: dict[str, set[str]] = {}
    tracked_python = [
        path
        for path in tracked_paths
        if path.endswith(".py") and Path(path).parts and Path(path).parts[0] in _GIT_HYGIENE_SOURCE_ROOTS
    ]
    for rel_text in sorted(tracked_python):
        path = root / rel_text
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            for candidate in _import_candidates_for_node(Path(rel_text), node):
                if candidate in untracked_source_paths:
                    importers_by_target.setdefault(candidate, set()).add(rel_text)
    return {target: tuple(sorted(importers)) for target, importers in sorted(importers_by_target.items())}


def _import_candidates_for_node(relative: Path, node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        candidates: list[str] = []
        for alias in node.names:
            candidates.extend(_module_file_candidates(alias.name))
        return tuple(dict.fromkeys(candidates))
    if not isinstance(node, ast.ImportFrom):
        return ()
    package_parts = list(relative.parent.parts)
    if node.level:
        trim = max(0, len(package_parts) - (node.level - 1))
        base_parts = package_parts[:trim]
    else:
        base_parts = []
    module_parts = node.module.split(".") if node.module else []
    target_parts = [*base_parts, *module_parts]
    candidates: list[str] = []
    if target_parts:
        candidates.extend(_module_file_candidates(".".join(target_parts)))
    for alias in node.names:
        if alias.name == "*":
            continue
        if target_parts:
            candidates.extend(_module_file_candidates(".".join([*target_parts, alias.name])))
        elif base_parts:
            candidates.extend(_module_file_candidates(".".join([*base_parts, alias.name])))
    return tuple(dict.fromkeys(candidates))


def _module_file_candidates(module: str) -> tuple[str, str]:
    relative = module.replace(".", "/")
    return (f"{relative}.py", f"{relative}/__init__.py")


def _file_check(root: Path, check_id: str, requirement: str, required_paths: Iterable[str]) -> AuditCheck:
    required = tuple(required_paths)
    missing = _missing_files(root, required)
    return AuditCheck(
        check_id,
        requirement,
        "passed" if not missing else "missing",
        evidence=tuple(path for path in required if (root / path).exists()),
        missing=tuple(missing),
    )


def _missing_files(root: Path, paths: Iterable[str]) -> list[str]:
    return [path for path in paths if not (root / path).exists()]


def _missing_path_groups(root: Path, path_groups: Iterable[Iterable[str]]) -> tuple[str, ...]:
    missing: list[str] = []
    for group in path_groups:
        options = tuple(group)
        if options and not any((root / path).exists() for path in options):
            missing.append(options[0])
    return tuple(missing)


def _existing_path_group_paths(root: Path, path_groups: Iterable[Iterable[str]]) -> tuple[str, ...]:
    evidence: list[str] = []
    for group in path_groups:
        for path in group:
            if (root / path).exists():
                evidence.append(path)
                break
    return tuple(evidence)


def _probe_types_from_public_dev(root: Path) -> set[str]:
    manifest_path = root / "data/releases/amst_main_v1_public/manifest.json"
    if not manifest_path.exists():
        return set()
    manifest = read_json(manifest_path)
    split_files = manifest.get("split_files", {}).get("public_dev", {})
    probe_types: set[str] = set()
    if not isinstance(split_files, dict):
        return probe_types
    for raw_path in split_files.values():
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = manifest_path.parent / path
        if not path.exists():
            continue
        shard = read_json(path)
        for case in shard.get("cases", []):
            for query in case.get("queries", []):
                probe_type = query.get("probe_type")
                if probe_type:
                    probe_types.add(str(probe_type))
    return probe_types


def _renderer_coverage_from_public_dev(root: Path) -> set[str]:
    manifest_path = root / "data/releases/amst_main_v1_public/manifest.json"
    if not manifest_path.exists():
        return set()
    manifest = read_json(manifest_path)
    coverage = manifest.get("split_reports", {}).get("public_dev", {}).get("renderer_coverage", {})
    if not isinstance(coverage, dict):
        return set()
    return {
        str(name)
        for name, item in coverage.items()
        if isinstance(item, dict) and int(item.get("num_source_events", 0)) > 0
    }


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _prefer_current_example_artifact(path: Path) -> Path:
    current_path = path.with_name(f"{path.stem}_current{path.suffix}")
    return current_path if current_path.exists() else path


def _artifact_root_ref(base_dir: Path, project_root: Path) -> str:
    resolved_base_dir = base_dir.resolve()
    resolved_project_root = project_root.resolve()
    try:
        return Path(os.path.relpath(resolved_project_root, resolved_base_dir)).as_posix()
    except ValueError:
        return str(resolved_project_root)


def _valid_agreement_metrics(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    fields = value.get("fields")
    if not isinstance(fields, dict):
        return False
    required = ("evidence_sufficient", "answer_unique", "governance_boundary_clear", "trace_natural")
    for field in required:
        metrics = fields.get(field)
        if not isinstance(metrics, dict):
            return False
        if metrics.get("num_pairs", 0) <= 0:
            return False
        if metrics.get("cohen_kappa") is None:
            return False
    return True


def _valid_external_correlation_report(path: Path) -> bool:
    from amb.benchmark.analysis.external_protocol import validate_external_correlation_report

    return not validate_external_correlation_report(path)


def _valid_real_system_report(path: Path) -> bool:
    from amb.benchmark.quality.real_system import validate_real_system_report

    return not validate_real_system_report(path)


def _real_system_report_paths(root: Path, fixed_reports: tuple[str, ...]) -> tuple[tuple[Path, ...], tuple[str, ...], tuple[str, ...]]:
    from amb.benchmark.quality.real_system import (
        ordered_real_system_analysis_candidates,
        ordered_real_system_matrix_summary_candidates,
    )

    matrix_candidates = ordered_real_system_matrix_summary_candidates(root)
    for summary_path in matrix_candidates:
        if not summary_path.exists():
            continue
        report_paths, errors = _report_paths_from_real_system_matrix(root, summary_path)
        validation_evidence = ()
        validation_errors = ()
        validation_candidates = _ordered_real_system_matrix_validation_candidates(root, summary_path)
        validation_path = next((path for path in validation_candidates if path.exists()), None)
        if validation_path is None:
            if validation_candidates:
                validation_errors = (_rel(validation_candidates[0], root),)
        else:
            validation_evidence = (_rel(validation_path, root),)
            validation_errors = _real_system_matrix_validation_errors(root, summary_path, validation_path)
        config_validation_evidence = ()
        config_validation_errors = ()
        config_validation_candidates = _ordered_real_system_config_validation_candidates(root, summary_path)
        config_validation_path = next((path for path in config_validation_candidates if path.exists()), None)
        if config_validation_path is None:
            if config_validation_candidates:
                config_validation_errors = (_rel(config_validation_candidates[0], root),)
        else:
            config_validation_evidence = (_rel(config_validation_path, root),)
            config_validation_errors = _real_system_config_validation_errors(root, config_validation_path)
        analysis_evidence = ()
        analysis_errors = ()
        analysis_candidates = ordered_real_system_analysis_candidates(root, summary_path)
        analysis_path = next((path for path in analysis_candidates if path.exists()), None)
        if analysis_path is None:
            if analysis_candidates:
                analysis_errors = (_rel(analysis_candidates[0], root),)
        else:
            analysis_evidence = (_rel(analysis_path, root),)
            analysis_errors = _real_system_analysis_errors(root, analysis_path, expected_num_reports=len(report_paths))
        return (
            report_paths,
            (_rel(summary_path, root), *validation_evidence, *config_validation_evidence, *analysis_evidence),
            (*errors, *validation_errors, *config_validation_errors, *analysis_errors),
        )
    return tuple(root / path for path in fixed_reports if (root / path).exists()), (), ()


def _report_paths_from_real_system_matrix(root: Path, summary_path: Path) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    from amb.benchmark.quality.real_system import (
        _contract_root_from_summary,
        _read_summary_object,
        _resolve_report_path,
        validate_real_system_matrix_summary,
    )

    try:
        summary = _read_summary_object(summary_path)
    except ValueError:
        summary = {}
    contract_root = _contract_root_from_summary(summary_path, summary) if isinstance(summary, dict) else None
    validation = validate_real_system_matrix_summary(
        summary_path,
        expected_benchmark_id="amst-main-v1-public_dev",
        expected_release_split="public_dev",
    )
    report_paths = tuple(
        _resolve_report_path(summary_path, row["report_path"], contract_root=contract_root)
        for row in validation.get("systems", [])
        if isinstance(row, dict) and row.get("report_path")
    )
    return report_paths, tuple(str(error) for error in validation.get("errors", []))


def _ordered_real_system_matrix_validation_candidates(root: Path, summary_path: Path) -> tuple[Path, ...]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path.resolve())
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    name = summary_path.name
    if name.endswith("_matrix_summary.json"):
        add(summary_path.with_name(name.replace("_matrix_summary.json", "_matrix_validation.json")))
    if name.endswith("matrix_summary.json"):
        add(summary_path.with_name(name.replace("matrix_summary.json", "matrix_validation.json")))
    if name.endswith("_summary.json"):
        add(summary_path.with_name(name.replace("_summary.json", "_validation.json")))
    if summary_path.parent == root / "reports/real_system_runs":
        add(summary_path.parent / "current_matrix_validation.json")
        add(summary_path.parent / "canonical_matrix_validation.json")
    return tuple(candidates)


def _ordered_real_system_config_validation_candidates(root: Path, summary_path: Path) -> tuple[Path, ...]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path.resolve())
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    name = summary_path.name
    if name.endswith("_current_matrix_summary.json"):
        add(summary_path.with_name(name.replace("_current_matrix_summary.json", "_config_validation.json")))
    if name.endswith("_matrix_summary.json"):
        add(summary_path.with_name(name.replace("_matrix_summary.json", "_config_validation.json")))
    if name.endswith("matrix_summary.json"):
        add(summary_path.with_name(name.replace("matrix_summary.json", "refresh_config_validation.json")))
    if summary_path.parent == root / "reports/real_system_runs":
        add(summary_path.parent / "canonical_public_dev_refresh_config_validation.json")
        add(summary_path.parent / "current_refresh_config_validation.json")
        for path in sorted(summary_path.parent.glob("*config_validation.json")):
            add(path)
    return tuple(candidates)


def _real_system_matrix_validation_errors(root: Path, summary_path: Path, validation_path: Path) -> tuple[str, ...]:
    rel_validation = _rel(validation_path, root)
    try:
        report = read_json(validation_path)
    except Exception as exc:
        return (f"{rel_validation}: cannot read matrix validation: {exc}",)
    if not isinstance(report, dict):
        return (f"{rel_validation}: matrix validation must be a JSON object",)

    errors: list[str] = []
    if report.get("status") != "passed":
        errors.append(f"{rel_validation}: status must be passed")
    if report.get("errors"):
        errors.append(f"{rel_validation}: errors must be empty")
    matrix_summary = report.get("matrix_summary")
    if not isinstance(matrix_summary, str) or not _path_string_matches_file(
        matrix_summary,
        summary_path,
        root=root,
        anchor=validation_path.parent,
    ):
        errors.append(f"{rel_validation}: matrix_summary must reference {_rel(summary_path, root)}")
    return tuple(errors)


def _real_system_config_validation_errors(root: Path, validation_path: Path) -> tuple[str, ...]:
    rel_validation = _rel(validation_path, root)
    try:
        report = read_json(validation_path)
    except Exception as exc:
        return (f"{rel_validation}: cannot read config validation: {exc}",)
    if not isinstance(report, dict):
        return (f"{rel_validation}: config validation must be a JSON object",)

    errors: list[str] = []
    if report.get("status") != "passed":
        errors.append(f"{rel_validation}: status must be passed")
    if report.get("errors"):
        errors.append(f"{rel_validation}: errors must be empty")
    if int(report.get("num_configs", 0) or 0) <= 0:
        errors.append(f"{rel_validation}: num_configs must be positive")
    configs = report.get("configs")
    if not isinstance(configs, list) or not configs:
        errors.append(f"{rel_validation}: configs must be a non-empty list")
    return tuple(errors)


def _real_system_analysis_errors(root: Path, analysis_path: Path, *, expected_num_reports: int) -> tuple[str, ...]:
    from amb.benchmark.quality.real_system import summarize_real_system_analysis

    rel_analysis = _rel(analysis_path, root)
    try:
        report = read_json(analysis_path)
    except Exception as exc:
        return (f"{rel_analysis}: cannot read real-system analysis: {exc}",)
    if not isinstance(report, dict):
        return (f"{rel_analysis}: real-system analysis must be a JSON object",)

    summary = summarize_real_system_analysis(report)
    errors: list[str] = []
    if summary.get("schema_version") != "amst-real-system-analysis-v1":
        errors.append(f"{rel_analysis}: schema_version must be amst-real-system-analysis-v1")
    if int(summary.get("num_reports", 0) or 0) != expected_num_reports:
        errors.append(f"{rel_analysis}: num_reports must be {expected_num_reports}")
    if summary.get("bootstrap_samples_sufficient") is not True:
        errors.append(f"{rel_analysis}: bootstrap_samples must be at least 200")
    if summary.get("report_bootstrap_cis_present") is not True:
        errors.append(f"{rel_analysis}: report bootstrap CIs must be complete")
    if summary.get("pairwise_stats_complete") is not True:
        errors.append(f"{rel_analysis}: pairwise stats must be complete")
    if summary.get("quality_cost_frontier_complete") is not True:
        errors.append(f"{rel_analysis}: quality-cost frontier must be complete")
    if summary.get("weight_sensitivity_profiles_complete") is not True:
        errors.append(f"{rel_analysis}: weight_sensitivity profiles must be complete")
    return tuple(errors)


def _path_string_matches_file(candidate: str, expected: Path, *, root: Path, anchor: Path) -> bool:
    candidate_path = Path(candidate)
    expected_resolved = expected.resolve()
    search_paths = (candidate_path,) if candidate_path.is_absolute() else (root / candidate_path, anchor / candidate_path)
    return any(path.resolve() == expected_resolved for path in search_paths)


def _external_validation_errors(root: Path, validation_artifact: str) -> tuple[str, ...]:
    validation_path = root / validation_artifact
    if not validation_path.exists():
        return (validation_artifact,)
    try:
        report = read_json(validation_path)
    except Exception as exc:
        return (f"{validation_artifact}: cannot read evidence validation: {exc}",)
    if not isinstance(report, dict):
        return (f"{validation_artifact}: evidence validation must be a JSON object",)

    errors: list[str] = []
    if report.get("status") != "passed":
        errors.append(f"{validation_artifact}: status must be passed")
    if report.get("errors"):
        errors.append(f"{validation_artifact}: errors must be empty")
    num_shared_systems = report.get("num_shared_systems")
    if not isinstance(num_shared_systems, int) or num_shared_systems < 3:
        errors.append(f"{validation_artifact}: num_shared_systems must be at least 3")
    num_shared_control_systems = report.get("num_shared_control_systems")
    if not isinstance(num_shared_control_systems, int) or num_shared_control_systems < 1:
        errors.append(f"{validation_artifact}: num_shared_control_systems must be at least 1")
    num_shared_real_memory_systems = report.get("num_shared_real_memory_systems")
    if not isinstance(num_shared_real_memory_systems, int) or num_shared_real_memory_systems < 1:
        errors.append(f"{validation_artifact}: num_shared_real_memory_systems must be at least 1")
    covered_ids = report.get("covered_benchmark_ids")
    required_ids = ("locomo", "longmemeval", "mem2actbench", "memoryagentbench")
    if covered_ids != list(required_ids):
        errors.append(f"{validation_artifact}: covered_benchmark_ids must be {list(required_ids)}")
    if report.get("require_identical_systems") is not True:
        errors.append(f"{validation_artifact}: require_identical_systems must be true")
    if report.get("min_shared_systems") != 3:
        errors.append(f"{validation_artifact}: min_shared_systems must be 3")
    if report.get("min_shared_control_systems") != 1:
        errors.append(f"{validation_artifact}: min_shared_control_systems must be 1")
    if report.get("min_shared_real_memory_systems") != 1:
        errors.append(f"{validation_artifact}: min_shared_real_memory_systems must be 1")
    return tuple(errors)


def _external_expansion_validation_errors(root: Path) -> tuple[str, ...]:
    validation_artifact = "reports/external/cohort_expansion_validation.json"
    validation_path = root / validation_artifact
    if not validation_path.exists():
        return ()
    try:
        report = read_json(validation_path)
    except Exception as exc:
        return (f"{validation_artifact}: cannot read expansion validation: {exc}",)
    if not isinstance(report, dict):
        return (f"{validation_artifact}: expansion validation must be a JSON object",)

    errors: list[str] = []
    if report.get("status") != "passed":
        errors.append(f"{validation_artifact}: status must be passed")
    if report.get("errors"):
        errors.append(f"{validation_artifact}: errors must be empty")
    for key in (
        "return_inbox_sync_report_exists",
        "return_inbox_sync_report_matches",
        "return_inbox_watch_exists",
        "return_inbox_watch_matches",
        "handoff_manifest_exists",
        "handoff_manifest_matches",
    ):
        if report.get(key) is not True:
            errors.append(f"{validation_artifact}: {key} must be true")
    return tuple(errors)


def _human_audit_bundle_validation_errors(root: Path) -> tuple[str, ...]:
    verification_artifact = "reports/human_audit_bundle/current/bundle_verification.json"
    verification_path = root / verification_artifact
    if not verification_path.exists():
        return ()
    try:
        report = read_json(verification_path)
    except Exception as exc:
        return (f"{verification_artifact}: cannot read bundle verification: {exc}",)
    if not isinstance(report, dict):
        return (f"{verification_artifact}: bundle verification must be a JSON object",)

    errors: list[str] = []
    if report.get("status") != "passed":
        errors.append(f"{verification_artifact}: status must be passed")
    if report.get("ok") is not True:
        errors.append(f"{verification_artifact}: ok must be true")
    return tuple(errors)
