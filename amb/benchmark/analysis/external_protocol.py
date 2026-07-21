"""Evidence protocol for AutoMemoryBench external benchmark correlations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any, Iterable
import zipfile

from amb.benchmark.analysis.external_correlation import analyze_external_correlations, write_external_correlations
from amb.benchmark.analysis.external_scores import (
    EXTERNAL_SCORE_SCHEMA_VERSION,
    write_normalized_longmemeval_scores,
)
from amb.benchmark.analysis.statistics import numeric_or_none
from amb.benchmark.schemas.io import read_json, write_json

EXTERNAL_EVIDENCE_PLAN_SCHEMA_VERSION = "amst-external-evidence-plan-v1"
EXTERNAL_CORRELATION_BATCH_SCHEMA_VERSION = "amst-external-correlation-batch-v1"
EXTERNAL_EVIDENCE_VALIDATION_SCHEMA_VERSION = "amst-external-evidence-validation-v1"
EXTERNAL_EVIDENCE_GAP_REPORT_SCHEMA_VERSION = "amst-external-evidence-gap-report-v1"
EXTERNAL_COHORT_EXPANSION_PLAN_SCHEMA_VERSION = "amst-external-cohort-expansion-plan-v1"
EXTERNAL_COHORT_EXPANSION_VALIDATION_SCHEMA_VERSION = "amst-external-cohort-expansion-validation-v1"
EXTERNAL_COHORT_EXPANSION_HANDOFF_MANIFEST_SCHEMA_VERSION = "amst-external-cohort-expansion-handoff-manifest-v1"
EXTERNAL_COHORT_CANDIDATE_PACKET_SCHEMA_VERSION = "amst-external-cohort-candidate-packet-v1"
EXTERNAL_COHORT_RETURN_PACKET_SCHEMA_VERSION = "amst-external-cohort-return-packet-v1"
EXTERNAL_COHORT_RETURN_INBOX_SYNC_SCHEMA_VERSION = "amst-external-cohort-return-inbox-sync-v1"
EXTERNAL_COHORT_RETURN_INBOX_STATE_SCHEMA_VERSION = "amst-external-cohort-return-inbox-state-v1"
EXTERNAL_COHORT_REJECTED_RETURNS_REPORT_SCHEMA_VERSION = "amst-external-cohort-rejected-returns-v1"
EXTERNAL_COHORT_RETURN_INBOX_WATCH_SCHEMA_VERSION = "amst-external-cohort-return-inbox-watch-v1"
EXTERNAL_CANONICAL_REFRESH_SCHEMA_VERSION = "amst-external-canonical-refresh-v1"
EXTERNAL_COHORT_OPERATOR_WATCH_INTERVAL_S = 120.0
REQUIRED_EXTERNAL_RUN_CONFIG_FIELDS = ("benchmark_version", "split", "execution_protocol", "system_cohort")
EXTERNAL_COHORT_WATCH_STOP_EXIT_CODES = {
    "max_iterations": 0,
    "rejected_returns": 2,
    "ready": 3,
}


@dataclass(frozen=True)
class ExternalBenchmarkRequirement:
    benchmark_id: str
    score_artifact: str
    correlation_report: str
    recommended_metric: str
    requirement: str


DEFAULT_EXTERNAL_BENCHMARKS: tuple[tuple[str, str, str], ...] = (
    ("locomo", "f1", "Long-context conversational memory benchmark"),
    ("longmemeval", "accuracy", "Long-range memory retrieval and reasoning benchmark"),
    ("mem2actbench", "success_rate", "Memory-to-action agent benchmark"),
    ("memoryagentbench", "score", "Agent memory benchmark with task-level outcomes"),
)
CANONICAL_EXTERNAL_SMOKE_AMST_REPORTS: tuple[str, str] = (
    "reports/examples/amst_main_v1_strict_public_dev_no_memory_report.json",
    "reports/examples/amst_main_v1_strict_public_dev_oracle_memory_report.json",
)
CANONICAL_LONGMEMEVAL_SMOKE_MANIFEST = "configs/external/longmemeval_no_memory_oracle_smoke_manifest.json"
EXTERNAL_SMOKE_PATH_FIELD_NAMES = frozenset(
    {
        "source_artifact",
        "reference_artifact",
        "external_score_path",
        "external_source_artifact",
        "external_score_artifact",
        "correlation_report",
        "result_path",
        "reference_path",
    }
)

CONTROL_SYSTEM_ALIASES: dict[str, tuple[str, ...]] = {
    "no_memory": ("no_memory", "nomemory"),
    "sliding_window": ("sliding_window", "slidingwindow"),
    "full_history": ("full_history", "fullhistory", "longcontext", "fullcontext"),
    "rolling_summary": ("rolling_summary", "rollingsummary", "summarymemory", "summary"),
    "bm25": ("bm25", "bm25_memory", "bm25retrieval"),
    "dense_rag": ("dense_rag", "densememory", "dense_memory", "denseretrieval", "dense"),
    "hybrid_rag": ("hybrid_rag", "hybridmemory", "hybrid_memory", "hybridretrieval", "hybrid"),
    "oracle_retrieval": ("oracle_retrieval", "oracleretrieval"),
    "oracle_state": ("oracle_state", "oraclememory", "oracle_memory", "oraclestate"),
}

REAL_MEMORY_SYSTEM_ALIASES: dict[str, tuple[str, ...]] = {
    "memorybank": ("memorybank",),
    "generative_agents": ("generativeagents",),
    "letta_memgpt": ("letta", "memgpt", "lettamemgptlocalreal", "lettamemgptlocal"),
    "mem0": ("mem0", "mem0realsiliconflow", "mem0real", "mem0g"),
    "langmem": ("langmem", "langmemreal", "langmemmemory"),
    "zep_graphiti": ("zep", "graphiti", "zepgraphiti", "zepgraphitigraphiticorelocal"),
    "a_mem": ("amem", "amem*", "a_mem"),
    "memoryos": ("memoryos",),
    "lightmem": ("lightmem",),
    "memos": ("memos", "memcube"),
}


def build_external_evidence_plan(
    root: str | Path = ".",
    *,
    output_dir: str | Path = "reports/external",
    benchmark_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Describe the required external-correlation evidence and current readiness.

    The plan is intentionally separate from ``completion_audit``. It does not
    mark the project complete; it tells maintainers which real external score
    artifacts are still needed and which normalized artifacts are ready to
    correlate.
    """

    project = Path(root)
    base_dir = Path(output_dir)
    if not base_dir.is_absolute():
        base_dir = project / base_dir
    required_ids = tuple(benchmark_ids) if benchmark_ids is not None else tuple(item[0] for item in DEFAULT_EXTERNAL_BENCHMARKS)
    specs = {benchmark_id: (metric, requirement) for benchmark_id, metric, requirement in DEFAULT_EXTERNAL_BENCHMARKS}
    requirements = []
    counts = {"passed": 0, "ready": 0, "missing": 0, "invalid": 0}

    for benchmark_id in required_ids:
        metric, requirement = specs.get(benchmark_id, ("score", "External benchmark score artifact"))
        score_path = base_dir / f"{benchmark_id}_scores.json"
        correlation_path = base_dir / f"{benchmark_id}_correlation.json"
        score_errors = validate_normalized_external_score(score_path, expected_benchmark_id=benchmark_id) if score_path.exists() else ()
        correlation_errors = (
            validate_external_correlation_report(correlation_path, expected_benchmark_id=benchmark_id)
            if correlation_path.exists()
            else ()
        )
        pair_errors = (
            _validate_expected_score_alignment(project, correlation_path, score_path, benchmark_id=benchmark_id)
            if correlation_path.exists() and score_path.exists() and not score_errors and not correlation_errors
            else ()
        )
        if correlation_path.exists() and not correlation_errors and not score_errors and not pair_errors:
            status = "passed"
            missing: tuple[str, ...] = ()
            errors: tuple[str, ...] = ()
        elif correlation_path.exists() and (correlation_errors or pair_errors):
            status = "invalid"
            missing = ()
            errors = (*correlation_errors, *pair_errors)
        elif score_path.exists() and not score_errors:
            status = "ready"
            missing = (_rel(correlation_path, project),)
            errors = ()
        elif score_path.exists():
            status = "invalid"
            missing = (_rel(correlation_path, project),)
            errors = score_errors
        else:
            status = "missing"
            missing = (_rel(score_path, project), _rel(correlation_path, project))
            errors = ()
        counts[status] += 1
        requirements.append(
            {
                **asdict(
                    ExternalBenchmarkRequirement(
                        benchmark_id=benchmark_id,
                        score_artifact=_rel(score_path, project),
                        correlation_report=_rel(correlation_path, project),
                        recommended_metric=metric,
                        requirement=requirement,
                    )
                ),
                "status": status,
                "missing": list(missing),
                "errors": list(errors),
            }
        )

    overall = "passed" if counts["passed"] == len(requirements) else "incomplete"
    return {
        "schema_version": EXTERNAL_EVIDENCE_PLAN_SCHEMA_VERSION,
        "root": _project_root_ref(base_dir, project_root=project),
        "output_dir": _rel(base_dir, project),
        "status": overall,
        "summary": counts,
        "requirements": requirements,
    }


def write_external_evidence_plan(
    output: str | Path,
    *,
    root: str | Path = ".",
    output_dir: str | Path = "reports/external",
    benchmark_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    plan = build_external_evidence_plan(root, output_dir=output_dir, benchmark_ids=benchmark_ids)
    write_json(output, plan)
    return plan


def write_external_correlation_batch(
    amst_report_paths: Iterable[str | Path],
    external_score_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    amst_metric: str = "lifecycle.amq",
    external_metric: str = "score",
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 13013,
    strict: bool = True,
) -> dict[str, Any]:
    """Write one normalized external-correlation report per benchmark.

    ``strict=True`` rejects raw or under-covered score artifacts before they can
    be mistaken for completion evidence.
    """

    amst_reports = tuple(str(Path(path)) for path in amst_report_paths)
    score_paths = tuple(Path(path) for path in external_score_paths)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for score_path in score_paths:
        benchmark_id = _external_benchmark_id(score_path)
        score_errors = validate_normalized_external_score(score_path, expected_benchmark_id=benchmark_id)
        if score_errors and strict:
            raise ValueError(f"{score_path} is not a valid normalized external score artifact: {score_errors[0]}")
        output_path = target_dir / f"{benchmark_id}_correlation.json"
        report = write_external_correlations(
            amst_reports,
            [score_path],
            output_path,
            amst_metric=amst_metric,
            external_metric=external_metric,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        report_errors = validate_external_correlation_payload(report, expected_benchmark_id=benchmark_id)
        status = "passed" if not score_errors and not report_errors else "invalid"
        if report_errors and strict:
            raise ValueError(f"{score_path} does not provide valid correlation evidence: {report_errors[0]}")
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "status": status,
                "external_score_artifact": str(score_path),
                "correlation_report": str(output_path),
                "errors": [*score_errors, *report_errors],
            }
        )

    summary = {
        "schema_version": EXTERNAL_CORRELATION_BATCH_SCHEMA_VERSION,
        "status": "passed" if rows and all(row["status"] == "passed" for row in rows) else "incomplete",
        "amst_metric": amst_metric,
        "external_metric": external_metric,
        "num_amst_reports": len(amst_reports),
        "num_external_benchmarks": len(rows),
        "outputs": rows,
    }
    return summary


def write_external_correlation_batch_summary(
    summary: dict[str, Any],
    output: str | Path,
) -> dict[str, Any]:
    output_path = Path(output).resolve()
    normalized = json.loads(json.dumps(summary))
    project_root = _external_contract_project_root(output_path.parent)
    if project_root is not None:
        root_ref = _project_root_ref(output_path.parent, project_root=project_root)
        if root_ref is not None:
            normalized["root"] = root_ref
        outputs = normalized.get("outputs")
        if isinstance(outputs, list):
            for row in outputs:
                if not isinstance(row, dict):
                    continue
                for field in ("external_score_artifact", "correlation_report"):
                    raw_path = row.get(field)
                    if isinstance(raw_path, str) and raw_path:
                        row[field] = _normalize_project_relative_path_string(
                            raw_path,
                            project_root=project_root,
                            base_dir=output_path.parent,
                        )
    write_json(output_path, normalized)
    return normalized


def validate_external_evidence_set(
    correlation_paths: Iterable[str | Path],
    *,
    required_benchmark_ids: Iterable[str] | None = None,
    min_shared_systems: int = 2,
    require_identical_systems: bool = True,
    min_shared_control_systems: int = 0,
    min_shared_real_memory_systems: int = 0,
) -> dict[str, Any]:
    """Validate the external-correlation evidence as a benchmark set.

    Per-file checks are insufficient for the paper claim. The external evidence
    must show that the same system cohort was compared across AutoMemoryBench and the
    required external memory benchmarks.
    """

    required_ids = tuple(required_benchmark_ids) if required_benchmark_ids is not None else tuple(
        item[0] for item in DEFAULT_EXTERNAL_BENCHMARKS
    )
    errors: list[str] = []
    benchmark_systems: dict[str, set[str]] = {}
    report_rows: list[dict[str, Any]] = []
    if min_shared_systems < 1:
        errors.append("min_shared_systems must be at least 1")
    if min_shared_control_systems < 0:
        errors.append("min_shared_control_systems must be at least 0")
    if min_shared_real_memory_systems < 0:
        errors.append("min_shared_real_memory_systems must be at least 0")

    for raw_path in correlation_paths:
        path = Path(raw_path)
        report_errors = list(validate_external_correlation_report(path))
        entries = _external_correlation_entries(path)
        for benchmark_id, systems in entries.items():
            if benchmark_id in benchmark_systems:
                errors.append(f"duplicate external benchmark correlation evidence for {benchmark_id}")
            benchmark_systems[benchmark_id] = systems
        report_rows.append(
            {
                "path": str(path),
                "status": "passed" if not report_errors else "invalid",
                "benchmark_ids": sorted(entries),
                "errors": report_errors,
            }
        )
        errors.extend(f"{path}: {error}" for error in report_errors)

    missing_ids = sorted(set(required_ids) - set(benchmark_systems))
    if missing_ids:
        errors.append(f"missing required external benchmark correlations: {missing_ids}")

    shared_systems: list[str] = []
    shared_control_systems: list[str] = []
    shared_real_memory_systems: list[str] = []
    benchmark_control_systems: dict[str, list[str]] = {}
    benchmark_real_memory_systems: dict[str, list[str]] = {}
    if benchmark_systems:
        shared = set.intersection(*(set(systems) for systems in benchmark_systems.values()))
        shared_systems = sorted(shared)
        shared_control_systems = _classify_shared_systems(shared_systems, CONTROL_SYSTEM_ALIASES)
        shared_real_memory_systems = _classify_shared_systems(shared_systems, REAL_MEMORY_SYSTEM_ALIASES)
        benchmark_control_systems = {
            benchmark_id: _classify_shared_systems(sorted(systems), CONTROL_SYSTEM_ALIASES)
            for benchmark_id, systems in sorted(benchmark_systems.items())
        }
        benchmark_real_memory_systems = {
            benchmark_id: _classify_shared_systems(sorted(systems), REAL_MEMORY_SYSTEM_ALIASES)
            for benchmark_id, systems in sorted(benchmark_systems.items())
        }
        if len(shared) < min_shared_systems:
            errors.append(f"external evidence must share at least {min_shared_systems} systems across benchmarks")
        if len(shared_control_systems) < min_shared_control_systems:
            errors.append(
                f"external evidence must share at least {min_shared_control_systems} control systems across benchmarks"
            )
        if len(shared_real_memory_systems) < min_shared_real_memory_systems:
            errors.append(
                "external evidence must share at least "
                f"{min_shared_real_memory_systems} real memory systems across benchmarks"
            )
        if require_identical_systems:
            first_id = sorted(benchmark_systems)[0]
            reference = benchmark_systems[first_id]
            mismatched = {
                benchmark_id: sorted(systems)
                for benchmark_id, systems in sorted(benchmark_systems.items())
                if systems != reference
            }
            if mismatched:
                errors.append("external benchmark correlations must use the same system cohort")
    return {
        "schema_version": EXTERNAL_EVIDENCE_VALIDATION_SCHEMA_VERSION,
        "status": "passed" if not errors else "incomplete",
        "required_benchmark_ids": sorted(required_ids),
        "covered_benchmark_ids": sorted(benchmark_systems),
        "missing_benchmark_ids": missing_ids,
        "shared_systems": shared_systems,
        "num_shared_systems": len(shared_systems),
        "shared_control_systems": shared_control_systems,
        "num_shared_control_systems": len(shared_control_systems),
        "shared_real_memory_systems": shared_real_memory_systems,
        "num_shared_real_memory_systems": len(shared_real_memory_systems),
        "benchmark_systems": {
            benchmark_id: sorted(systems) for benchmark_id, systems in sorted(benchmark_systems.items())
        },
        "benchmark_control_systems": benchmark_control_systems,
        "benchmark_real_memory_systems": benchmark_real_memory_systems,
        "min_shared_systems": min_shared_systems,
        "min_shared_control_systems": min_shared_control_systems,
        "min_shared_real_memory_systems": min_shared_real_memory_systems,
        "require_identical_systems": require_identical_systems,
        "reports": report_rows,
        "errors": errors,
    }


def write_external_evidence_validation(
    correlation_paths: Iterable[str | Path],
    output: str | Path,
    *,
    required_benchmark_ids: Iterable[str] | None = None,
    min_shared_systems: int = 2,
    require_identical_systems: bool = True,
    min_shared_control_systems: int = 0,
    min_shared_real_memory_systems: int = 0,
) -> dict[str, Any]:
    report = validate_external_evidence_set(
        correlation_paths,
        required_benchmark_ids=required_benchmark_ids,
        min_shared_systems=min_shared_systems,
        require_identical_systems=require_identical_systems,
        min_shared_control_systems=min_shared_control_systems,
        min_shared_real_memory_systems=min_shared_real_memory_systems,
    )
    output_path = Path(output).resolve()
    root_ref = _external_root_ref(output_path.parent)
    if root_ref is not None:
        report["root"] = root_ref
    write_json(output, report)
    return report


def build_external_evidence_gap_report(
    correlation_paths: Iterable[str | Path],
    *,
    real_system_validation_path: str | Path | None = None,
    required_benchmark_ids: Iterable[str] | None = None,
    min_shared_systems: int = 3,
    require_identical_systems: bool = True,
    min_shared_control_systems: int = 1,
    min_shared_real_memory_systems: int = 1,
) -> dict[str, Any]:
    validation = validate_external_evidence_set(
        correlation_paths,
        required_benchmark_ids=required_benchmark_ids,
        min_shared_systems=min_shared_systems,
        require_identical_systems=require_identical_systems,
        min_shared_control_systems=min_shared_control_systems,
        min_shared_real_memory_systems=min_shared_real_memory_systems,
    )
    real_system_validation = _read_optional_json(real_system_validation_path)
    available_real_systems = _available_real_memory_systems(real_system_validation)
    available_real_providers = sorted(available_real_systems)
    benchmark_real_memory_systems = validation.get("benchmark_real_memory_systems", {})
    if not isinstance(benchmark_real_memory_systems, dict):
        benchmark_real_memory_systems = {}
    coverage_by_provider = {
        provider: sorted(
            benchmark_id
            for benchmark_id, providers in sorted(benchmark_real_memory_systems.items())
            if isinstance(providers, list) and provider in providers
        )
        for provider in available_real_providers
    }
    providers_missing_everywhere = sorted(
        provider for provider, covered in coverage_by_provider.items() if not covered
    )
    real_memory_gap = (
        validation.get("num_shared_real_memory_systems", 0) < min_shared_real_memory_systems
    )
    control_only_gap = bool(validation.get("shared_systems")) and not validation.get("shared_real_memory_systems")
    next_priority = (
        "Expand the external same-system cohort beyond control-only smoke anchors and add at least one AMST-validated real memory system."
        if real_memory_gap
        else "Add one more shared system across all required benchmarks."
    )
    if providers_missing_everywhere:
        next_priority = (
            next_priority
            + " Current AMST-validated providers missing from every external benchmark cohort: "
            + ", ".join(providers_missing_everywhere)
            + "."
        )
    return {
        "schema_version": EXTERNAL_EVIDENCE_GAP_REPORT_SCHEMA_VERSION,
        "status": validation["status"],
        "validation": validation,
        "completion_gate": {
            "min_shared_systems": min_shared_systems,
            "min_shared_control_systems": min_shared_control_systems,
            "min_shared_real_memory_systems": min_shared_real_memory_systems,
            "require_identical_systems": require_identical_systems,
        },
        "design_recommended_control_systems": sorted(CONTROL_SYSTEM_ALIASES),
        "design_recommended_real_memory_systems": sorted(REAL_MEMORY_SYSTEM_ALIASES),
        "amst_real_system_validation_path": str(real_system_validation_path) if real_system_validation_path else None,
        "amst_available_real_memory_systems": available_real_systems,
        "amst_available_real_memory_providers": available_real_providers,
        "external_real_memory_provider_coverage": coverage_by_provider,
        "providers_missing_from_all_external_benchmarks": providers_missing_everywhere,
        "control_only_smoke_cohort": control_only_gap,
        "recommended_next_priority": next_priority,
    }


def write_external_evidence_gap_report(
    correlation_paths: Iterable[str | Path],
    output: str | Path,
    *,
    real_system_validation_path: str | Path | None = None,
    required_benchmark_ids: Iterable[str] | None = None,
    min_shared_systems: int = 3,
    require_identical_systems: bool = True,
    min_shared_control_systems: int = 1,
    min_shared_real_memory_systems: int = 1,
) -> dict[str, Any]:
    report = build_external_evidence_gap_report(
        correlation_paths,
        real_system_validation_path=real_system_validation_path,
        required_benchmark_ids=required_benchmark_ids,
        min_shared_systems=min_shared_systems,
        require_identical_systems=require_identical_systems,
        min_shared_control_systems=min_shared_control_systems,
        min_shared_real_memory_systems=min_shared_real_memory_systems,
    )
    output_path = Path(output).resolve()
    root_ref = _external_root_ref(output_path.parent)
    if root_ref is not None:
        report["root"] = root_ref
    write_json(output, report)
    return report


def build_external_cohort_expansion_plan(
    correlation_paths: Iterable[str | Path],
    *,
    project_root: str | Path | None = None,
    real_system_validation_path: str | Path | None = None,
    required_benchmark_ids: Iterable[str] | None = None,
    min_shared_systems: int = 3,
    require_identical_systems: bool = True,
    min_shared_control_systems: int = 1,
    min_shared_real_memory_systems: int = 1,
) -> dict[str, Any]:
    project = Path(project_root) if project_root is not None else None
    correlation_path_list = tuple(Path(path) for path in correlation_paths)
    if correlation_path_list:
        base_dir = Path(os.path.commonpath([str(path.resolve().parent) for path in correlation_path_list]))
    elif project is not None:
        base_dir = project.resolve()
    else:
        base_dir = Path.cwd()
    validation = validate_external_evidence_set(
        correlation_path_list,
        required_benchmark_ids=required_benchmark_ids,
        min_shared_systems=min_shared_systems,
        require_identical_systems=require_identical_systems,
        min_shared_control_systems=min_shared_control_systems,
        min_shared_real_memory_systems=min_shared_real_memory_systems,
    )
    gap = build_external_evidence_gap_report(
        correlation_path_list,
        real_system_validation_path=real_system_validation_path,
        required_benchmark_ids=required_benchmark_ids,
        min_shared_systems=min_shared_systems,
        require_identical_systems=require_identical_systems,
        min_shared_control_systems=min_shared_control_systems,
        min_shared_real_memory_systems=min_shared_real_memory_systems,
    )
    real_system_validation = _read_optional_json(real_system_validation_path)
    required_ids = tuple(required_benchmark_ids) if required_benchmark_ids is not None else tuple(
        item[0] for item in DEFAULT_EXTERNAL_BENCHMARKS
    )
    available_real_systems = _available_real_memory_system_rows(real_system_validation)
    current_shared = list(validation.get("shared_systems", [])) if isinstance(validation.get("shared_systems"), list) else []
    current_shared_controls = list(validation.get("shared_control_systems", [])) if isinstance(validation.get("shared_control_systems"), list) else []
    current_shared_real_memory = list(validation.get("shared_real_memory_systems", [])) if isinstance(validation.get("shared_real_memory_systems"), list) else []
    coverage_by_provider = gap.get("external_real_memory_provider_coverage", {})
    if not isinstance(coverage_by_provider, dict):
        coverage_by_provider = {}

    benchmark_context = {
        benchmark_id: _external_benchmark_context(Path(path), benchmark_id)
        for benchmark_id, path in (
            (benchmark_id, _correlation_path_for_benchmark(correlation_path_list, benchmark_id)) for benchmark_id in required_ids
        )
    }
    provider_rows: list[dict[str, Any]] = []
    for row in available_real_systems:
        provider = row["provider"]
        system_id = row["system_id"]
        covered_benchmarks = coverage_by_provider.get(provider, [])
        if not isinstance(covered_benchmarks, list):
            covered_benchmarks = []
        missing_benchmarks = [benchmark_id for benchmark_id in required_ids if benchmark_id not in covered_benchmarks]
        target_system_cohort = [*current_shared]
        if system_id not in target_system_cohort:
            target_system_cohort.append(system_id)
        target_control_count = len(current_shared_controls)
        target_real_memory_count = len(current_shared_real_memory) + (0 if provider in current_shared_real_memory else 1)
        would_satisfy_gate = (
            len(target_system_cohort) >= min_shared_systems
            and target_control_count >= min_shared_control_systems
            and target_real_memory_count >= min_shared_real_memory_systems
        )
        regeneration_targets = []
        for benchmark_id in required_ids:
            context = benchmark_context[benchmark_id]
            regeneration_targets.append(
                {
                    "benchmark_id": benchmark_id,
                    "score_artifact": context["score_artifact"],
                    "correlation_report": context["correlation_report"],
                    "current_source_artifact": context.get("current_source_artifact"),
                    "current_system_cohort": list(context.get("current_system_cohort", [])),
                    "target_system_cohort": list(target_system_cohort),
                    "requires_regeneration": benchmark_id in missing_benchmarks or system_id not in context.get("current_system_cohort", []),
                }
            )
        command_bundle = (
            _external_candidate_command_bundle(
                project,
                provider=provider,
                system_id=system_id,
                real_system_report_path=row.get("report_path"),
                target_system_cohort=target_system_cohort,
                regeneration_targets=regeneration_targets,
            )
            if project is not None
            else None
        )
        provider_rows.append(
            {
                "provider": provider,
                "system_id": system_id,
                "real_system_report_path": row.get("report_path"),
                "covered_benchmarks": covered_benchmarks,
                "missing_benchmarks": missing_benchmarks,
                "num_missing_benchmarks": len(missing_benchmarks),
                "target_system_cohort": target_system_cohort,
                "target_shared_system_count": len(target_system_cohort),
                "target_shared_control_system_count": target_control_count,
                "target_shared_real_memory_system_count": target_real_memory_count,
                "would_satisfy_completion_gate": would_satisfy_gate,
                "regeneration_targets": regeneration_targets,
                "command_bundle": command_bundle,
            }
        )
    minimum_completion_candidates = [
        {
            "provider": row["provider"],
            "system_id": row["system_id"],
            "target_system_cohort": row["target_system_cohort"],
            "num_required_benchmark_regenerations": len(required_ids),
        }
        for row in provider_rows
        if row["would_satisfy_completion_gate"]
    ]
    if validation["status"] == "passed":
        status = "not_needed"
    elif minimum_completion_candidates:
        status = "ready"
    elif provider_rows:
        status = "partial"
    else:
        status = "blocked"
    return {
        "schema_version": EXTERNAL_COHORT_EXPANSION_PLAN_SCHEMA_VERSION,
        "root": _project_root_ref(base_dir, project_root=project),
        "status": status,
        "validation": validation,
        "gap": gap,
        "completion_gate": {
            "min_shared_systems": min_shared_systems,
            "min_shared_control_systems": min_shared_control_systems,
            "min_shared_real_memory_systems": min_shared_real_memory_systems,
            "require_identical_systems": require_identical_systems,
        },
        "current_shared_systems": current_shared,
        "current_shared_control_systems": current_shared_controls,
        "current_shared_real_memory_systems": current_shared_real_memory,
        "required_benchmark_ids": list(required_ids),
        "benchmark_context": [benchmark_context[benchmark_id] for benchmark_id in required_ids],
        "available_real_memory_candidates": provider_rows,
        "minimum_completion_candidates": minimum_completion_candidates,
        "recommended_completion_candidate": provider_rows[0] if provider_rows else None,
        "recommended_next_priority": gap.get("recommended_next_priority"),
    }


def write_external_cohort_expansion_plan(
    correlation_paths: Iterable[str | Path],
    output: str | Path,
    *,
    root: str | Path = ".",
    real_system_validation_path: str | Path | None = None,
    required_benchmark_ids: Iterable[str] | None = None,
    min_shared_systems: int = 3,
    require_identical_systems: bool = True,
    min_shared_control_systems: int = 1,
    min_shared_real_memory_systems: int = 1,
) -> dict[str, Any]:
    project = Path(root)
    project_root = project.resolve()
    report = build_external_cohort_expansion_plan(
        correlation_paths,
        project_root=project,
        real_system_validation_path=real_system_validation_path,
        required_benchmark_ids=required_benchmark_ids,
        min_shared_systems=min_shared_systems,
        require_identical_systems=require_identical_systems,
        min_shared_control_systems=min_shared_control_systems,
        min_shared_real_memory_systems=min_shared_real_memory_systems,
    )
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = project / output_path
    report["expansion_plan_file"] = _relative_or_absolute(output_path, output_path.parent)
    report["expansion_plan_path"] = _project_relative_path_or_absolute(output_path, project_root)
    return_inbox, return_archive, return_reject_archive = _ensure_external_return_dirs(output_path.parent)
    pending_return_packets = sorted((output_path.parent / "returns" / "inbox").glob("*.zip"))
    report["return_inbox"] = return_inbox
    report["return_archive"] = return_archive
    report["return_reject_archive"] = return_reject_archive
    report["return_inbox_sync_report_file"] = "return_inbox_sync.json"
    report["return_inbox_sync_report_path"] = _project_relative_path_or_absolute(
        output_path.parent / "return_inbox_sync.json",
        project_root,
    )
    report["return_inbox_state_file"] = "return_inbox_state.json"
    report["return_inbox_state_path"] = _project_relative_path_or_absolute(
        output_path.parent / "return_inbox_state.json",
        project_root,
    )
    report["return_inbox_watch_file"] = "return_inbox_watch.json"
    report["return_inbox_watch_output_file"] = _project_relative_path_or_absolute(
        output_path.parent / "return_inbox_watch.json",
        project_root,
    )
    report["pending_return_packets"] = [_relative_or_absolute(path, output_path.parent) for path in pending_return_packets]
    report["pending_return_packet_paths"] = [str(path.resolve()) for path in pending_return_packets]
    report["return_reject_archive_paths"] = {
        key: _project_relative_path_or_absolute(output_path.parent / value, project_root)
        for key, value in sorted(return_reject_archive.items())
        if isinstance(value, str) and value
    }
    return_inbox_state = _load_external_return_inbox_state(output_path.parent)
    rejected_report = write_external_cohort_rejected_returns_report(output_path.parent)
    report["rejected_return_summary"] = rejected_report["rejected_return_summary"]
    report["rejected_returns_report_file"] = "rejected_returns_report.json"
    report["rejected_returns_report_path"] = _project_relative_path_or_absolute(
        output_path.parent / "rejected_returns_report.json",
        project_root,
    )
    report["watch_stop_exit_codes"] = dict(EXTERNAL_COHORT_WATCH_STOP_EXIT_CODES)
    _write_initial_external_return_inbox_sync_report(output_path.parent, report)
    _write_external_return_inbox_state(output_path.parent, return_inbox_state)
    scripts = _write_external_operator_scripts(output_path.parent, report, repo_root=project)
    report["operator_scripts"] = scripts
    report["operator_script_files"] = {
        script_id: _project_relative_path_or_absolute(output_path.parent / Path(script_path), project_root)
        if not Path(script_path).is_absolute()
        else str(Path(script_path))
        for script_id, script_path in sorted(scripts.items())
    }
    report["operator_commands"] = _external_operator_commands(output_path.parent, report, repo_root=project)
    report["watch_stop_actions"] = _external_watch_stop_actions(
        output_path.parent,
        report["operator_scripts"],
        report["operator_script_files"],
        repo_root=project,
    )
    report["candidate_packets"] = _write_external_candidate_packets(output_path.parent, report, repo_root=project)
    recommended_candidate = report.get("recommended_completion_candidate")
    if isinstance(recommended_candidate, dict):
        provider = recommended_candidate.get("provider")
        if isinstance(provider, str) and provider:
            script_id = f"expand_with_{provider}"
            script_rel = scripts.get(script_id)
            if isinstance(script_rel, str) and script_rel:
                script_path = Path(script_rel)
                recommended_candidate["script_id"] = script_id
                recommended_candidate["script"] = script_rel
                recommended_candidate["script_file"] = (
                    _project_relative_path_or_absolute(output_path.parent / script_path, project_root)
                    if not script_path.is_absolute()
                    else str(script_path)
                )
            packet = report["candidate_packets"].get(provider) if isinstance(report.get("candidate_packets"), dict) else None
            if isinstance(packet, dict):
                for key in (
                    "packet_dir",
                    "packet_path",
                    "archive_file",
                    "archive_path",
                    "packet_manifest_file",
                    "packet_manifest_path",
                    "package_return_file",
                    "package_return_path",
                    "env_template_file",
                    "env_template_path",
                    "readme_file",
                    "readme_path",
                    "run_script_file",
                    "run_script_path",
                ):
                    if key in packet:
                        recommended_candidate[key] = packet[key]
    if isinstance(recommended_candidate, dict):
        report["recommended_completion_candidate"] = _external_materialized_candidate_summary(recommended_candidate)
    for key in ("minimum_completion_candidates", "available_real_memory_candidates"):
        candidates = report.get(key)
        if not isinstance(candidates, list):
            continue
        report[key] = [
            _external_materialized_candidate_summary(item) if isinstance(item, dict) else item
            for item in candidates
        ]
    readme_path = output_path.parent / "README.md"
    report["readme_file"] = _relative_or_absolute(readme_path, output_path.parent)
    report["readme_path"] = _project_relative_path_or_absolute(readme_path, project_root)
    _write_external_cohort_expansion_readme(readme_path, report)
    validation_path = output_path.parent / "cohort_expansion_validation.json"
    report["validation_file"] = _relative_or_absolute(validation_path, output_path.parent)
    report["validation_path"] = _project_relative_path_or_absolute(validation_path, project_root)
    handoff_manifest_path = output_path.parent / "handoff_manifest.json"
    report["handoff_manifest_file"] = _relative_or_absolute(handoff_manifest_path, output_path.parent)
    report["handoff_manifest_path"] = _project_relative_path_or_absolute(handoff_manifest_path, project_root)
    report.update(
        _external_recommended_next_refs(
            report,
            operator_commands=report["operator_commands"],
            operator_scripts=report["operator_scripts"],
            operator_script_files=report["operator_script_files"],
        )
    )
    _write_initial_external_return_inbox_sync_report(output_path.parent, report)
    _write_initial_external_return_inbox_watch_report(output_path.parent, report)
    write_json(
        handoff_manifest_path,
        _build_external_handoff_manifest(report, base_dir=output_path.parent, repo_root=project),
    )
    validation = write_external_cohort_expansion_validation(
        report,
        validation_path,
        root=output_path.parent,
    )
    report["validation_status"] = validation.get("status")
    report["validation_errors"] = list(validation.get("errors", [])) if isinstance(validation.get("errors"), list) else []
    write_json(
        handoff_manifest_path,
        _build_external_handoff_manifest(report, base_dir=output_path.parent, repo_root=project),
    )
    validation = write_external_cohort_expansion_validation(
        report,
        validation_path,
        root=output_path.parent,
    )
    report["validation_status"] = validation.get("status")
    report["validation_errors"] = list(validation.get("errors", [])) if isinstance(validation.get("errors"), list) else []
    write_json(output_path, report)
    return report


def build_external_cohort_expansion_validation(
    report: dict[str, Any],
    *,
    root: str | Path,
    require_artifacts: bool = True,
) -> dict[str, Any]:
    base_dir = Path(root)
    project_root = None
    project_root_raw = report.get("root")
    if isinstance(project_root_raw, str) and project_root_raw.strip():
        project_root = _resolve_reported_project_root(project_root_raw, base_dir=base_dir)
    if project_root is None:
        project_root = _external_contract_project_root(base_dir)
    errors: list[str] = []
    script_checks: dict[str, dict[str, Any]] = {}
    operator_scripts = report.get("operator_scripts")
    if not isinstance(operator_scripts, dict):
        operator_scripts = {}
        errors.append("operator_scripts must be present")
    operator_script_files = report.get("operator_script_files")
    if not isinstance(operator_script_files, dict):
        operator_script_files = {}
    for script_id, raw_script in sorted(operator_scripts.items()):
        if not isinstance(raw_script, str) or not raw_script.strip():
            errors.append(f"operator_scripts.{script_id} must be a non-empty string")
            continue
        script_path = Path(raw_script)
        script_file = base_dir / script_path if not script_path.is_absolute() else script_path
        exists = script_file.exists()
        declared_file = operator_script_files.get(script_id)
        if isinstance(declared_file, str) and declared_file:
            script_file_ref = declared_file
        elif project_root is not None:
            script_file_ref = _project_relative_path_or_absolute(script_file, project_root)
        else:
            script_file_ref = str(script_file.resolve()) if exists else str(script_file)
        script_checks[str(script_id)] = {
            "script": raw_script,
            "script_file": script_file_ref,
            "exists": exists,
        }
        if not exists and require_artifacts:
            errors.append(f"operator script missing: {raw_script}")
            continue
        if not exists:
            continue
        if isinstance(declared_file, str) and declared_file:
            declared_resolved = _resolve_external_contract_path(base_dir, declared_file)
            if declared_resolved.resolve() != script_file.resolve():
                errors.append(f"operator_script_files.{script_id} does not match operator_scripts.{script_id}")

    readme_file_raw = report.get("readme_file")
    readme_path = None
    readme_exists = False
    if isinstance(readme_file_raw, str) and readme_file_raw.strip():
        readme_path = base_dir / readme_file_raw
        readme_exists = readme_path.exists()
        if not readme_exists and require_artifacts:
            errors.append(f"readme missing: {readme_file_raw}")
    else:
        errors.append("readme_file must be present")
    return_inbox_exists = False
    return_archive_exists = False
    return_reject_archive_exists = False
    return_inbox = report.get("return_inbox") if isinstance(report.get("return_inbox"), dict) else {}
    return_archive = report.get("return_archive") if isinstance(report.get("return_archive"), dict) else {}
    return_reject_archive = (
        report.get("return_reject_archive") if isinstance(report.get("return_reject_archive"), dict) else {}
    )
    return_inbox_raw = return_inbox.get("candidate_inbox")
    return_archive_raw = return_archive.get("candidate_archive")
    return_reject_archive_raw = return_reject_archive.get("candidate_archive")
    if isinstance(return_inbox_raw, str) and return_inbox_raw.strip():
        return_inbox_exists = (base_dir / return_inbox_raw).exists()
    if isinstance(return_archive_raw, str) and return_archive_raw.strip():
        return_archive_exists = (base_dir / return_archive_raw).exists()
    if isinstance(return_reject_archive_raw, str) and return_reject_archive_raw.strip():
        return_reject_archive_exists = (base_dir / return_reject_archive_raw).exists()
    if require_artifacts:
        if not return_inbox_exists:
            errors.append("return_inbox.candidate_inbox must exist")
        if not return_archive_exists:
            errors.append("return_archive.candidate_archive must exist")
        if not return_reject_archive_exists:
            errors.append("return_reject_archive.candidate_archive must exist")
    rejected_returns_report_file_raw = report.get("rejected_returns_report_file")
    rejected_returns_report_path = None
    rejected_returns_report_exists = False
    rejected_returns_report_matches = False
    if isinstance(rejected_returns_report_file_raw, str) and rejected_returns_report_file_raw.strip():
        rejected_returns_report_path = base_dir / rejected_returns_report_file_raw
        rejected_returns_report_exists = rejected_returns_report_path.exists()
        if not rejected_returns_report_exists and require_artifacts:
            errors.append(f"rejected_returns_report missing: {rejected_returns_report_file_raw}")
        elif rejected_returns_report_exists:
            rejected_payload = _read_optional_json(rejected_returns_report_path)
            if rejected_payload is None:
                errors.append("rejected_returns_report.json must be a JSON object")
            else:
                rejected_returns_report_matches = (
                    rejected_payload.get("rejected_return_summary") == report.get("rejected_return_summary")
                )
                if not rejected_returns_report_matches:
                    errors.append("rejected_returns_report summary does not match expansion report")
    else:
        errors.append("rejected_returns_report_file must be present")
    return_inbox_sync_report_file_raw = report.get("return_inbox_sync_report_file")
    return_inbox_sync_report_path = None
    return_inbox_sync_report_exists = False
    return_inbox_sync_report_matches = False
    if isinstance(return_inbox_sync_report_file_raw, str) and return_inbox_sync_report_file_raw.strip():
        return_inbox_sync_report_path = base_dir / return_inbox_sync_report_file_raw
        return_inbox_sync_report_exists = return_inbox_sync_report_path.exists()
        if not return_inbox_sync_report_exists and require_artifacts:
            errors.append(f"return_inbox_sync_report missing: {return_inbox_sync_report_file_raw}")
        elif return_inbox_sync_report_exists:
            sync_payload = _read_optional_json(return_inbox_sync_report_path)
            if sync_payload is None:
                errors.append("return_inbox_sync.json must be a JSON object")
            else:
                return_inbox_sync_report_matches = (
                    _external_return_inbox_sync_report_summary(sync_payload)
                    == _expected_external_return_inbox_sync_report_summary(report, base_dir=base_dir)
                )
                if not return_inbox_sync_report_matches:
                    errors.append("return_inbox_sync summary does not match expansion report")
    else:
        errors.append("return_inbox_sync_report_file must be present")
    return_inbox_watch_file_raw = report.get("return_inbox_watch_file")
    return_inbox_watch_path = None
    return_inbox_watch_exists = False
    return_inbox_watch_matches = False
    if isinstance(return_inbox_watch_file_raw, str) and return_inbox_watch_file_raw.strip():
        return_inbox_watch_path = base_dir / return_inbox_watch_file_raw
        return_inbox_watch_exists = return_inbox_watch_path.exists()
        if not return_inbox_watch_exists and require_artifacts:
            errors.append(f"return_inbox_watch missing: {return_inbox_watch_file_raw}")
        elif return_inbox_watch_exists:
            watch_payload = _read_optional_json(return_inbox_watch_path)
            if watch_payload is None:
                errors.append("return_inbox_watch.json must be a JSON object")
            else:
                return_inbox_watch_matches = (
                    _external_return_inbox_watch_report_summary(watch_payload)
                    == _expected_external_return_inbox_watch_report_summary(report, watch_payload, base_dir=base_dir)
                )
                if not return_inbox_watch_matches:
                    errors.append("return_inbox_watch summary does not match expansion report")
    else:
        errors.append("return_inbox_watch_file must be present")
    handoff_manifest_file_raw = report.get("handoff_manifest_file")
    handoff_manifest_path = None
    handoff_manifest_exists = False
    handoff_manifest_matches = False
    if isinstance(handoff_manifest_file_raw, str) and handoff_manifest_file_raw.strip():
        handoff_manifest_path = base_dir / handoff_manifest_file_raw
        handoff_manifest_exists = handoff_manifest_path.exists()
        if not handoff_manifest_exists and require_artifacts:
            errors.append(f"handoff_manifest missing: {handoff_manifest_file_raw}")
        elif handoff_manifest_exists:
            handoff_payload = _read_optional_json(handoff_manifest_path)
            if handoff_payload is None:
                errors.append("handoff_manifest.json must be a JSON object")
            else:
                handoff_manifest_matches = _external_handoff_manifest_summary(handoff_payload) == _expected_external_handoff_manifest_summary(
                    report,
                    base_dir=base_dir,
                    repo_root=None,
                )
                if not handoff_manifest_matches:
                    errors.append("handoff_manifest summary does not match expansion report")
    else:
        errors.append("handoff_manifest_file must be present")

    recommended_candidate = (
        report.get("recommended_completion_candidate")
        if isinstance(report.get("recommended_completion_candidate"), dict)
        else None
    )
    recommended_validation: dict[str, Any] = {
        "required": report.get("status") == "ready",
        "present": recommended_candidate is not None,
    }
    candidate_packets = report.get("candidate_packets")
    packet_checks: dict[str, dict[str, Any]] = {}
    if isinstance(candidate_packets, dict):
        for provider, packet in sorted(candidate_packets.items()):
            if not isinstance(packet, dict):
                errors.append(f"candidate_packets.{provider} must be an object")
                continue
            archive_file = packet.get("archive_file")
            packet_dir = packet.get("packet_dir")
            packet_manifest_file = packet.get("packet_manifest_file")
            run_script_file = packet.get("run_script_file")
            package_return_file = packet.get("package_return_file")
            env_template_file = packet.get("env_template_file")
            readme_file = packet.get("readme_file")
            archive_path = base_dir / archive_file if isinstance(archive_file, str) and archive_file else None
            packet_dir_path = base_dir / packet_dir if isinstance(packet_dir, str) and packet_dir else None
            manifest_path = base_dir / packet_manifest_file if isinstance(packet_manifest_file, str) and packet_manifest_file else None
            run_script_path = base_dir / run_script_file if isinstance(run_script_file, str) and run_script_file else None
            package_return_path = base_dir / package_return_file if isinstance(package_return_file, str) and package_return_file else None
            env_template_path = base_dir / env_template_file if isinstance(env_template_file, str) and env_template_file else None
            readme_path_local = base_dir / readme_file if isinstance(readme_file, str) and readme_file else None
            manifest_payload = _read_optional_json(manifest_path) if manifest_path and manifest_path.exists() else None
            manifest_valid = False
            if isinstance(manifest_payload, dict):
                manifest_valid = True
                if manifest_payload.get("schema_version") != EXTERNAL_COHORT_CANDIDATE_PACKET_SCHEMA_VERSION:
                    manifest_valid = False
                if manifest_payload.get("provider") != provider:
                    manifest_valid = False
                if manifest_payload.get("system_id") != packet.get("system_id"):
                    manifest_valid = False
                expected_external_dir = (
                    _project_relative_path_or_absolute(base_dir, project_root)
                    if project_root is not None
                    else str(base_dir.resolve())
                )
                if manifest_payload.get("external_dir") != expected_external_dir:
                    manifest_valid = False
                expected_root = _project_root_ref(packet_dir_path, project_root=project_root) if packet_dir_path is not None else None
                if manifest_payload.get("root") != expected_root:
                    manifest_valid = False
            packet_checks[str(provider)] = {
                "archive_exists": bool(archive_path and archive_path.exists()),
                "packet_dir_exists": bool(packet_dir_path and packet_dir_path.exists()),
                "packet_manifest_exists": bool(manifest_path and manifest_path.exists()),
                "packet_manifest_valid": manifest_valid,
                "run_script_exists": bool(run_script_path and run_script_path.exists()),
                "package_return_exists": bool(package_return_path and package_return_path.exists()),
                "env_template_exists": bool(env_template_path and env_template_path.exists()),
                "readme_exists": bool(readme_path_local and readme_path_local.exists()),
                "archive_file": archive_file,
                "packet_dir": packet_dir,
            }
            if require_artifacts:
                if archive_path is None or not archive_path.exists():
                    errors.append(f"candidate packet archive missing for {provider}")
                if packet_dir_path is None or not packet_dir_path.exists():
                    errors.append(f"candidate packet directory missing for {provider}")
                if manifest_path is None or not manifest_path.exists():
                    errors.append(f"candidate packet manifest missing for {provider}")
                elif not manifest_valid:
                    errors.append(f"candidate packet manifest contract invalid for {provider}")
                if run_script_path is None or not run_script_path.exists():
                    errors.append(f"candidate packet run script missing for {provider}")
                if package_return_path is None or not package_return_path.exists():
                    errors.append(f"candidate packet package_return script missing for {provider}")
                if env_template_path is None or not env_template_path.exists():
                    errors.append(f"candidate packet env template missing for {provider}")
                if readme_path_local is None or not readme_path_local.exists():
                    errors.append(f"candidate packet readme missing for {provider}")
    elif report.get("status") == "ready" and require_artifacts:
        errors.append("candidate_packets must be present when expansion status is ready")
    if report.get("status") == "ready":
        if recommended_candidate is None:
            errors.append("recommended_completion_candidate must be present when expansion status is ready")
        else:
            provider = recommended_candidate.get("provider")
            system_id = recommended_candidate.get("system_id")
            script_id = recommended_candidate.get("script_id")
            script_rel = recommended_candidate.get("script")
            script_file_raw = recommended_candidate.get("script_file")
            command_bundle = (
                _external_candidate_contract_bundle(report, str(provider), external_dir=base_dir)
                if isinstance(provider, str) and provider
                else None
            )
            required_env_vars = command_bundle.get("required_env_vars") if isinstance(command_bundle, dict) else None
            missing_env_refs: list[str] = []
            script_exists = False
            if not isinstance(provider, str) or not provider:
                errors.append("recommended_completion_candidate.provider must be present")
            if not isinstance(system_id, str) or not system_id:
                errors.append("recommended_completion_candidate.system_id must be present")
            expected_script_id = f"expand_with_{provider}" if isinstance(provider, str) and provider else None
            if expected_script_id is not None and script_id != expected_script_id:
                errors.append("recommended_completion_candidate.script_id must match provider")
            if not isinstance(script_rel, str) or not script_rel.strip():
                errors.append("recommended_completion_candidate.script must be present")
            if not isinstance(script_file_raw, str) or not script_file_raw.strip():
                errors.append("recommended_completion_candidate.script_file must be present")
            else:
                candidate_script_file = _resolve_external_contract_path(base_dir, script_file_raw)
                script_exists = candidate_script_file.exists()
                if not script_exists and require_artifacts:
                    errors.append("recommended_completion_candidate.script_file does not exist")
                elif script_exists and isinstance(required_env_vars, dict):
                    script_text = candidate_script_file.read_text(encoding="utf-8")
                    for env_name in sorted(required_env_vars):
                        needle = f'${{{env_name}:?Set {env_name}}}'
                        if needle not in script_text:
                            missing_env_refs.append(str(env_name))
                if missing_env_refs:
                    errors.append(
                        "recommended completion script is missing required env guards: "
                        + ", ".join(missing_env_refs)
                    )
            if not isinstance(command_bundle, dict):
                errors.append("recommended completion candidate packet contract must be present")
            else:
                for required_field in (
                    "amst_report_paths",
                    "normalize_commands",
                    "correlate_command",
                    "refresh_command",
                    "required_env_vars",
                    "target_system_cohort",
                ):
                    if required_field not in command_bundle:
                        errors.append(f"recommended completion candidate packet contract.{required_field} must be present")
            recommended_validation.update(
                {
                    "provider": provider,
                    "system_id": system_id,
                    "script_id": script_id,
                    "script": script_rel,
                    "script_file": script_file_raw,
                    "script_exists": script_exists,
                    "missing_required_env_guards": missing_env_refs,
                    "num_required_env_vars": len(required_env_vars) if isinstance(required_env_vars, dict) else 0,
                    "archive_file": recommended_candidate.get("archive_file"),
                    "packet_dir": recommended_candidate.get("packet_dir"),
                }
            )
    status = "passed" if not errors else "incomplete"
    return {
        "schema_version": EXTERNAL_COHORT_EXPANSION_VALIDATION_SCHEMA_VERSION,
        "status": status,
        "root": _external_dir_ref(base_dir, project_root=project_root),
        "expansion_status": report.get("status"),
        "readme_file": readme_file_raw,
        "readme_path": report.get("readme_path")
        if isinstance(report.get("readme_path"), str)
        else str(readme_path.resolve())
        if readme_path is not None and readme_exists
        else str(readme_path)
        if readme_path is not None
        else None,
        "readme_exists": readme_exists,
        "return_inbox_exists": return_inbox_exists,
        "return_archive_exists": return_archive_exists,
        "return_reject_archive_exists": return_reject_archive_exists,
        "rejected_returns_report_file": rejected_returns_report_file_raw,
        "rejected_returns_report_path": report.get("rejected_returns_report_path")
        if isinstance(report.get("rejected_returns_report_path"), str)
        else (
            str(rejected_returns_report_path.resolve())
            if rejected_returns_report_path is not None and rejected_returns_report_exists
            else str(rejected_returns_report_path)
            if rejected_returns_report_path is not None
            else None
        ),
        "rejected_returns_report_exists": rejected_returns_report_exists,
        "rejected_returns_report_matches": rejected_returns_report_matches,
        "return_inbox_sync_report_file": return_inbox_sync_report_file_raw,
        "return_inbox_sync_report_path": report.get("return_inbox_sync_report_path")
        if isinstance(report.get("return_inbox_sync_report_path"), str)
        else (
            str(return_inbox_sync_report_path.resolve())
            if return_inbox_sync_report_path is not None and return_inbox_sync_report_exists
            else str(return_inbox_sync_report_path)
            if return_inbox_sync_report_path is not None
            else None
        ),
        "return_inbox_sync_report_exists": return_inbox_sync_report_exists,
        "return_inbox_sync_report_matches": return_inbox_sync_report_matches,
        "return_inbox_watch_file": return_inbox_watch_file_raw,
        "return_inbox_watch_path": report.get("return_inbox_watch_output_file")
        if isinstance(report.get("return_inbox_watch_output_file"), str)
        else (
            str(return_inbox_watch_path.resolve())
            if return_inbox_watch_path is not None and return_inbox_watch_exists
            else str(return_inbox_watch_path)
            if return_inbox_watch_path is not None
            else None
        ),
        "return_inbox_watch_exists": return_inbox_watch_exists,
        "return_inbox_watch_matches": return_inbox_watch_matches,
        "handoff_manifest_file": handoff_manifest_file_raw,
        "handoff_manifest_path": report.get("handoff_manifest_path")
        if isinstance(report.get("handoff_manifest_path"), str)
        else (
            str(handoff_manifest_path.resolve())
            if handoff_manifest_path is not None and handoff_manifest_exists
            else str(handoff_manifest_path)
            if handoff_manifest_path is not None
            else None
        ),
        "handoff_manifest_exists": handoff_manifest_exists,
        "handoff_manifest_matches": handoff_manifest_matches,
        "operator_scripts": script_checks,
        "candidate_packets": packet_checks,
        "recommended_completion_candidate": recommended_validation,
        "errors": errors,
    }


def write_external_cohort_expansion_validation(
    report: dict[str, Any],
    output: str | Path,
    *,
    root: str | Path,
) -> dict[str, Any]:
    validation = build_external_cohort_expansion_validation(report, root=root)
    write_json(output, validation)
    return validation


def _write_external_smoke_refresh(project: Path, base_dir: Path) -> dict[str, Any]:
    smoke_dir = base_dir / "smoke"
    if not smoke_dir.exists():
        return {"status": "not_present", "outputs": []}

    outputs: list[str] = []
    manifest_path = project / CANONICAL_LONGMEMEVAL_SMOKE_MANIFEST
    amst_report_paths = [project / raw_path for raw_path in CANONICAL_EXTERNAL_SMOKE_AMST_REPORTS]
    if manifest_path.exists() and all(path.exists() for path in amst_report_paths):
        score_path = smoke_dir / "longmemeval_no_memory_oracle_scores.json"
        summary_path = smoke_dir / "longmemeval_no_memory_oracle_batch_summary.json"
        write_normalized_longmemeval_scores(manifest_path, score_path)
        summary = write_external_correlation_batch(amst_report_paths, [score_path], smoke_dir)
        write_external_correlation_batch_summary(summary, summary_path)
        outputs.extend(
            [
                _rel(score_path, project),
                _rel(smoke_dir / "longmemeval_correlation.json", project),
                _rel(summary_path, project),
            ]
        )

    for smoke_json in sorted(smoke_dir.glob("*.json")):
        localized = _localize_external_smoke_json(smoke_json, project)
        if localized is None:
            continue
        write_json(smoke_json, localized)
        rel_path = _rel(smoke_json, project)
        if rel_path not in outputs:
            outputs.append(rel_path)

    return {"status": "passed", "outputs": outputs}


def _localize_external_smoke_json(path: Path, project_root: Path) -> dict[str, Any] | None:
    try:
        payload = read_json(path)
    except Exception:  # noqa: BLE001 - optional smoke fixtures should not fail the canonical refresh
        return None
    if isinstance(payload, dict):
        normalized = json.loads(json.dumps(payload))
    elif isinstance(payload, list) and path.name.endswith("_raw_scores.json"):
        normalized = {"rows": json.loads(json.dumps(payload))}
    else:
        return None
    root_ref = _project_root_ref(path.parent, project_root=project_root)
    if root_ref is not None:
        normalized["root"] = root_ref
    _normalize_smoke_named_paths(normalized, project_root=project_root, base_dir=path.parent)
    return normalized


def _normalize_smoke_named_paths(value: Any, *, project_root: Path, base_dir: Path) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(nested, str) and key in EXTERNAL_SMOKE_PATH_FIELD_NAMES and nested.strip():
                value[key] = _normalize_project_relative_path_string(
                    nested,
                    project_root=project_root,
                    base_dir=base_dir,
                )
            else:
                _normalize_smoke_named_paths(nested, project_root=project_root, base_dir=base_dir)
        return
    if isinstance(value, list):
        for item in value:
            _normalize_smoke_named_paths(item, project_root=project_root, base_dir=base_dir)


def write_external_canonical_refresh(
    *,
    root: str | Path = ".",
    output_dir: str | Path = "reports/external",
    benchmark_ids: Iterable[str] | None = None,
    plan_output: str | Path | None = None,
    validation_output: str | Path | None = None,
    gap_output: str | Path | None = None,
    expansion_output: str | Path | None = None,
    expansion_validation_output: str | Path | None = None,
    real_system_validation_path: str | Path | None = None,
    min_shared_systems: int = 3,
    min_shared_control_systems: int = 1,
    min_shared_real_memory_systems: int = 1,
    require_identical_systems: bool = True,
) -> dict[str, Any]:
    project = Path(root)
    base_dir = Path(output_dir)
    if not base_dir.is_absolute():
        base_dir = project / base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    benchmark_id_tuple = tuple(benchmark_ids) if benchmark_ids is not None else None
    plan_output_path = Path(plan_output) if plan_output is not None else base_dir / "evidence_plan.json"
    validation_output_path = Path(validation_output) if validation_output is not None else base_dir / "evidence_validation.json"
    gap_output_path = Path(gap_output) if gap_output is not None else base_dir / "evidence_gap_report.json"
    expansion_output_path = Path(expansion_output) if expansion_output is not None else base_dir / "cohort_expansion_plan.json"
    expansion_validation_output_path = (
        Path(expansion_validation_output)
        if expansion_validation_output is not None
        else base_dir / "cohort_expansion_validation.json"
    )
    if not plan_output_path.is_absolute():
        plan_output_path = project / plan_output_path
    if not validation_output_path.is_absolute():
        validation_output_path = project / validation_output_path
    if not gap_output_path.is_absolute():
        gap_output_path = project / gap_output_path
    if not expansion_output_path.is_absolute():
        expansion_output_path = project / expansion_output_path
    if not expansion_validation_output_path.is_absolute():
        expansion_validation_output_path = project / expansion_validation_output_path

    plan = write_external_evidence_plan(
        plan_output_path,
        root=project,
        output_dir=base_dir,
        benchmark_ids=benchmark_id_tuple,
    )
    correlation_paths = _resolve_external_correlation_paths(project, base_dir, benchmark_id_tuple)
    validation = write_external_evidence_validation(
        correlation_paths,
        validation_output_path,
        required_benchmark_ids=benchmark_id_tuple,
        min_shared_systems=min_shared_systems,
        require_identical_systems=require_identical_systems,
        min_shared_control_systems=min_shared_control_systems,
        min_shared_real_memory_systems=min_shared_real_memory_systems,
    )
    gap = write_external_evidence_gap_report(
        correlation_paths,
        gap_output_path,
        real_system_validation_path=real_system_validation_path,
        required_benchmark_ids=benchmark_id_tuple,
        min_shared_systems=min_shared_systems,
        require_identical_systems=require_identical_systems,
        min_shared_control_systems=min_shared_control_systems,
        min_shared_real_memory_systems=min_shared_real_memory_systems,
    )
    expansion = write_external_cohort_expansion_plan(
        correlation_paths,
        expansion_output_path,
        root=project,
        real_system_validation_path=real_system_validation_path,
        required_benchmark_ids=benchmark_id_tuple,
        min_shared_systems=min_shared_systems,
        require_identical_systems=require_identical_systems,
        min_shared_control_systems=min_shared_control_systems,
        min_shared_real_memory_systems=min_shared_real_memory_systems,
    )
    expansion_validation = write_external_cohort_expansion_validation(
        expansion,
        expansion_validation_output_path,
        root=expansion_output_path.parent,
    )
    smoke_refresh = _write_external_smoke_refresh(project, base_dir)
    statuses = (
        plan.get("status"),
        validation.get("status"),
        gap.get("status"),
        expansion_validation.get("status"),
    )
    overall = "passed" if all(status == "passed" for status in statuses) else "incomplete"
    report = {
        "schema_version": EXTERNAL_CANONICAL_REFRESH_SCHEMA_VERSION,
        "root": str(project.resolve()),
        "output_dir": _rel(base_dir, project),
        "status": overall,
        "benchmark_ids": list(benchmark_id_tuple) if benchmark_id_tuple is not None else [item[0] for item in DEFAULT_EXTERNAL_BENCHMARKS],
        "correlation_reports": [_rel(path, project) for path in correlation_paths],
        "plan_output": _rel(plan_output_path, project),
        "validation_output": _rel(validation_output_path, project),
        "gap_output": _rel(gap_output_path, project),
        "expansion_output": _rel(expansion_output_path, project),
        "expansion_validation_output": _rel(expansion_validation_output_path, project),
        "expansion_handoff_output": expansion.get("handoff_manifest_file"),
        "plan_status": plan.get("status"),
        "validation_status": validation.get("status"),
        "gap_status": gap.get("status"),
        "expansion_status": expansion.get("status"),
        "expansion_validation_status": expansion_validation.get("status"),
        "smoke_refresh_status": smoke_refresh.get("status"),
        "smoke_outputs": smoke_refresh.get("outputs", []),
    }
    return report


def build_external_cohort_return_packet(
    expansion_path: str | Path,
    *,
    provider: str,
    score_paths: dict[str, str | Path],
    output: str | Path,
) -> dict[str, Any]:
    expansion_file = Path(expansion_path).resolve()
    expansion = read_json(expansion_file)
    if not isinstance(expansion, dict):
        raise ValueError("external cohort expansion plan must be a JSON object")
    candidate = _external_candidate_from_expansion(expansion, provider)
    command_bundle = _external_candidate_contract_bundle(expansion, provider, external_dir=expansion_file.parent)
    if not isinstance(command_bundle, dict):
        raise ValueError(f"candidate {provider!r} does not have a candidate packet contract")
    candidate_packet = _external_candidate_packet(expansion, provider)
    candidate_packet_manifest_path = _external_candidate_packet_manifest_path(
        expansion,
        candidate_packet,
        provider=provider,
        external_dir=expansion_file.parent,
    )
    required_ids = [str(item) for item in expansion.get("required_benchmark_ids", [])]
    if not required_ids:
        raise ValueError("external cohort expansion plan must declare required_benchmark_ids")
    provided_ids = sorted(str(key) for key in score_paths)
    if sorted(required_ids) != provided_ids:
        raise ValueError(
            "return packet must include exactly the required benchmark ids: "
            + ", ".join(required_ids)
        )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    external_dir = expansion_file.parent
    target_cohort = candidate.get("target_system_cohort")
    if not isinstance(target_cohort, list):
        target_cohort = command_bundle.get("target_system_cohort")
    system_id = str(candidate.get("system_id") or "")
    copied_score_refs: dict[str, str] = {}
    source_score_refs: dict[str, str] = {}
    score_digests: dict[str, str] = {}
    with TemporaryDirectory(prefix=f"external-return-{provider}-") as temp_dir:
        packet_root = Path(temp_dir)
        scores_dir = packet_root / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        for benchmark_id in required_ids:
            source_path = Path(score_paths[benchmark_id]).resolve()
            payload = _validate_external_return_score_artifact(
                source_path,
                benchmark_id=benchmark_id,
                target_system_cohort=target_cohort,
                required_system_id=system_id,
            )
            target_path = scores_dir / f"{benchmark_id}_scores.json"
            shutil.copy2(source_path, target_path)
            copied_score_refs[benchmark_id] = _relative_or_absolute(target_path, packet_root)
            source_score_refs[benchmark_id] = str(source_path)
            score_digests[benchmark_id] = _file_sha256(target_path)
            if not isinstance(payload, dict):
                raise ValueError(f"{source_path} must be a JSON object")
        packet_manifest = {
            "schema_version": EXTERNAL_COHORT_RETURN_PACKET_SCHEMA_VERSION,
            "provider": provider,
            "system_id": system_id,
            "external_dir": str(external_dir.resolve()),
            "expansion_plan_file": str(expansion_file),
            "expansion_plan_sha256": _file_sha256(expansion_file),
            "candidate_packet_manifest_file": _relative_or_absolute(candidate_packet_manifest_path, external_dir),
            "candidate_packet_manifest_sha256": _file_sha256(candidate_packet_manifest_path),
            "target_system_cohort": list(target_cohort) if isinstance(target_cohort, list) else None,
            "required_benchmark_ids": required_ids,
            "score_files": copied_score_refs,
            "score_file_sha256": score_digests,
            "source_score_artifacts": source_score_refs,
        }
        write_json(packet_root / "packet_manifest.json", packet_manifest)
        (packet_root / "README.md").write_text(
            _external_return_packet_readme(provider=provider, system_id=system_id, required_benchmark_ids=required_ids),
            encoding="utf-8",
        )
        _write_zip_from_dir(packet_root, output_path)
    return {
        "schema_version": EXTERNAL_COHORT_RETURN_PACKET_SCHEMA_VERSION,
        "status": "passed",
        "provider": provider,
        "system_id": system_id,
        "packet_file": str(output_path),
        "required_benchmark_ids": required_ids,
        "score_files": source_score_refs,
    }


def apply_external_cohort_return_packet(
    packet_path: str | Path,
    *,
    expansion_path: str | Path,
    root: str | Path = ".",
    real_system_validation_path: str | Path | None = None,
) -> dict[str, Any]:
    packet_file = Path(packet_path).resolve()
    expansion_file = Path(expansion_path).resolve()
    expansion = read_json(expansion_file)
    if not isinstance(expansion, dict):
        raise ValueError("external cohort expansion plan must be a JSON object")
    project_root_value = expansion.get("root")
    external_dir = expansion_file.parent
    project_root = (
        _resolve_reported_project_root(project_root_value, base_dir=external_dir)
        if isinstance(project_root_value, str) and project_root_value
        else Path(root).resolve()
    )
    with TemporaryDirectory(prefix="external-return-apply-") as temp_dir:
        extracted_dir = Path(temp_dir)
        with zipfile.ZipFile(packet_file) as archive:
            archive.extractall(extracted_dir)
        manifest = read_json(extracted_dir / "packet_manifest.json")
        if not isinstance(manifest, dict):
            raise ValueError("external return packet manifest must be a JSON object")
        if manifest.get("schema_version") != EXTERNAL_COHORT_RETURN_PACKET_SCHEMA_VERSION:
            raise ValueError(
                f"external return packet schema_version must be {EXTERNAL_COHORT_RETURN_PACKET_SCHEMA_VERSION}"
            )
        provider = str(manifest.get("provider") or "")
        if not provider:
            raise ValueError("external return packet manifest provider is required")
        candidate = _external_candidate_from_expansion(expansion, provider)
        candidate_packet = _external_candidate_packet(expansion, provider)
        command_bundle = _external_candidate_contract_bundle(expansion, provider, external_dir=external_dir)
        if not isinstance(command_bundle, dict):
            raise ValueError(f"candidate {provider!r} does not have a candidate packet contract")
        target_cohort = candidate.get("target_system_cohort")
        if not isinstance(target_cohort, list):
            target_cohort = command_bundle.get("target_system_cohort")
        system_id = str(candidate.get("system_id") or "")
        if manifest.get("system_id") != system_id:
            raise ValueError("external return packet manifest system_id does not match expansion candidate")
        manifest_external_dir = manifest.get("external_dir")
        if not isinstance(manifest_external_dir, str) or Path(manifest_external_dir).resolve() != external_dir.resolve():
            raise ValueError("external return packet manifest external_dir does not match expansion external dir")
        manifest_expansion_file = manifest.get("expansion_plan_file")
        if not isinstance(manifest_expansion_file, str) or Path(manifest_expansion_file).resolve() != expansion_file.resolve():
            raise ValueError("external return packet manifest expansion_plan_file does not match target expansion plan")
        manifest_expansion_sha = manifest.get("expansion_plan_sha256")
        if not isinstance(manifest_expansion_sha, str) or not manifest_expansion_sha:
            raise ValueError("external return packet manifest expansion_plan_sha256 is required")
        if manifest_expansion_sha != _file_sha256(expansion_file):
            raise ValueError("external return packet manifest expansion_plan_sha256 does not match target expansion plan")
        candidate_packet_manifest_path = _external_candidate_packet_manifest_path(
            expansion,
            candidate_packet,
            provider=provider,
            external_dir=external_dir,
        )
        manifest_candidate_packet_file = manifest.get("candidate_packet_manifest_file")
        if not isinstance(manifest_candidate_packet_file, str) or not manifest_candidate_packet_file:
            raise ValueError("external return packet manifest candidate_packet_manifest_file is required")
        expected_candidate_packet_file = _relative_or_absolute(candidate_packet_manifest_path, external_dir)
        if manifest_candidate_packet_file != expected_candidate_packet_file:
            raise ValueError(
                "external return packet manifest candidate_packet_manifest_file does not match target candidate packet"
            )
        manifest_candidate_packet_sha = manifest.get("candidate_packet_manifest_sha256")
        if not isinstance(manifest_candidate_packet_sha, str) or not manifest_candidate_packet_sha:
            raise ValueError("external return packet manifest candidate_packet_manifest_sha256 is required")
        if manifest_candidate_packet_sha != _file_sha256(candidate_packet_manifest_path):
            raise ValueError(
                "external return packet manifest candidate_packet_manifest_sha256 does not match target candidate packet"
            )
        required_ids = [str(item) for item in expansion.get("required_benchmark_ids", [])]
        packet_required_ids = [str(item) for item in manifest.get("required_benchmark_ids", [])]
        if sorted(required_ids) != sorted(packet_required_ids):
            raise ValueError("external return packet required benchmark ids do not match expansion plan")
        score_files = manifest.get("score_files")
        if not isinstance(score_files, dict):
            raise ValueError("external return packet score_files must be an object")
        score_digests = manifest.get("score_file_sha256")
        if not isinstance(score_digests, dict):
            raise ValueError("external return packet score_file_sha256 must be an object")
        applied_score_files: dict[str, str] = {}
        copied_score_paths: list[Path] = []
        for benchmark_id in required_ids:
            raw_score_file = score_files.get(benchmark_id)
            if not isinstance(raw_score_file, str) or not raw_score_file:
                raise ValueError(f"external return packet missing score file for benchmark {benchmark_id!r}")
            source_score_path = extracted_dir / raw_score_file
            expected_digest = score_digests.get(benchmark_id)
            if not isinstance(expected_digest, str) or not expected_digest:
                raise ValueError(f"external return packet missing score_file_sha256 for benchmark {benchmark_id!r}")
            actual_digest = _file_sha256(source_score_path)
            if actual_digest != expected_digest:
                raise ValueError(f"external return packet score_file_sha256 mismatch for benchmark {benchmark_id!r}")
            _validate_external_return_score_artifact(
                source_score_path,
                benchmark_id=benchmark_id,
                target_system_cohort=target_cohort,
                required_system_id=system_id,
            )
            target_score_path = external_dir / f"{benchmark_id}_scores.json"
            shutil.copy2(source_score_path, target_score_path)
            copied_score_paths.append(target_score_path)
            applied_score_files[benchmark_id] = str(target_score_path.resolve())
    amst_report_paths = [
        _resolve_project_path(project_root, raw_path)
        for raw_path in command_bundle.get("amst_report_paths", [])
        if isinstance(raw_path, str) and raw_path
    ]
    correlation_batch = write_external_correlation_batch(amst_report_paths, copied_score_paths, external_dir)
    refresh = write_external_canonical_refresh(
        root=project_root,
        output_dir=external_dir,
        real_system_validation_path=(
            real_system_validation_path
            if real_system_validation_path is not None
            else project_root / "reports/real_system_runs/canonical_public_dev_refresh_current_matrix_validation.json"
        ),
        min_shared_systems=3,
        min_shared_control_systems=1,
        min_shared_real_memory_systems=1,
    )
    validation_path = external_dir / "evidence_validation.json"
    validation = read_json(validation_path) if validation_path.exists() else {}
    return {
        "schema_version": EXTERNAL_COHORT_RETURN_PACKET_SCHEMA_VERSION,
        "status": "passed",
        "provider": provider,
        "system_id": system_id,
        "packet_file": str(packet_file),
        "applied_score_files": applied_score_files,
        "correlation_batch_status": correlation_batch.get("status"),
        "refresh_status": refresh.get("status"),
        "completion_validation_status": validation.get("status") if isinstance(validation, dict) else None,
        "num_shared_systems": validation.get("num_shared_systems") if isinstance(validation, dict) else None,
        "num_shared_real_memory_systems": validation.get("num_shared_real_memory_systems") if isinstance(validation, dict) else None,
    }


def sync_external_cohort_return_inbox(
    expansion_path: str | Path,
    *,
    root: str | Path = ".",
    real_system_validation_path: str | Path | None = None,
) -> dict[str, Any]:
    expansion_file = Path(expansion_path).resolve()
    expansion = read_json(expansion_file)
    if not isinstance(expansion, dict):
        raise ValueError("external cohort expansion plan must be a JSON object")
    external_dir = expansion_file.parent
    project_root_value = expansion.get("root")
    project_root = (
        _resolve_reported_project_root(project_root_value, base_dir=external_dir)
        if isinstance(project_root_value, str) and project_root_value
        else Path(root).resolve()
    )
    return_inbox, return_archive, return_reject_archive = _ensure_external_return_dirs(external_dir)
    processed_state = _load_external_return_inbox_state(external_dir)
    inbox_dir = external_dir / str(return_inbox["candidate_inbox"])
    processed_dir = external_dir / str(return_archive["candidate_archive"])
    rejected_dir = external_dir / str(return_reject_archive["candidate_archive"])
    processed_packets: list[str] = []
    skipped_processed_packets: list[dict[str, str]] = []
    rejected_packets: list[dict[str, str]] = []
    skipped_rejected_packets: list[dict[str, str]] = []
    applied_reports: list[dict[str, Any]] = []
    for packet_file in sorted(inbox_dir.glob("*.zip")):
        packet_fingerprint = _file_sha256(packet_file)
        prior_processed = _external_processed_packet_entry(processed_state, packet_fingerprint)
        if prior_processed is not None:
            skipped_processed_packets.append(
                {
                    "packet_file": _relative_or_absolute(packet_file, external_dir),
                    "packet_path": str(packet_file.resolve()),
                    "packet_fingerprint": packet_fingerprint,
                    "processed_archive_file": str(prior_processed.get("processed_archive_file") or ""),
                }
            )
            continue
        prior_rejected = _external_rejected_packet_entry(processed_state, packet_fingerprint)
        if prior_rejected is not None:
            skipped_rejected_packets.append(
                {
                    "packet_file": _relative_or_absolute(packet_file, external_dir),
                    "packet_path": str(packet_file.resolve()),
                    "packet_fingerprint": packet_fingerprint,
                    "rejection_error": str(prior_rejected.get("rejection_error") or "previously rejected"),
                    "rejected_archive_file": str(prior_rejected.get("rejected_archive_file") or ""),
                }
            )
            continue
        try:
            applied = apply_external_cohort_return_packet(
                packet_file,
                expansion_path=expansion_file,
                root=root,
                real_system_validation_path=real_system_validation_path,
            )
            destination = _move_to_unique_destination(packet_file, processed_dir / packet_file.name)
            processed_archive_file = _relative_or_absolute(destination, external_dir)
            processed_packets.append(str(destination.resolve()))
            applied_reports.append(applied)
            _record_processed_external_return_packet(
                processed_state,
                packet_fingerprint=packet_fingerprint,
                archive_file=processed_archive_file,
                archive_path=str(destination.resolve()),
                provider=str(applied.get("provider") or ""),
                system_id=str(applied.get("system_id") or ""),
            )
        except Exception as exc:  # noqa: BLE001 - return inbox sync should quarantine bad packets
            destination = _move_to_unique_destination(packet_file, rejected_dir / packet_file.name)
            rejected_archive_file = _relative_or_absolute(destination, external_dir)
            rejected_packets.append(
                {
                    "packet_file": rejected_archive_file,
                    "packet_path": str(destination.resolve()),
                    "packet_fingerprint": packet_fingerprint,
                    "rejection_error": str(exc),
                }
            )
            _record_rejected_external_return_packet(
                processed_state,
                packet_fingerprint=packet_fingerprint,
                archive_file=rejected_archive_file,
                archive_path=str(destination.resolve()),
                rejection_error=str(exc),
            )
    _write_external_return_inbox_state(external_dir, processed_state)
    rejected_report = write_external_cohort_rejected_returns_report(
        external_dir,
        state=processed_state,
    )
    final_refresh: dict[str, Any] | None = None
    if processed_packets or rejected_packets:
        final_refresh = write_external_canonical_refresh(
            root=project_root,
            output_dir=external_dir,
            real_system_validation_path=(
                real_system_validation_path
                if real_system_validation_path is not None
                else project_root / "reports/real_system_runs/canonical_public_dev_refresh_current_matrix_validation.json"
            ),
            min_shared_systems=3,
            min_shared_control_systems=1,
            min_shared_real_memory_systems=1,
        )
    current_expansion = final_refresh if isinstance(final_refresh, dict) else expansion
    next_action = _external_recommended_next_action(current_expansion)
    status = "passed" if not rejected_report["rejected_return_summary"]["num_rejected_candidate_packets"] else "incomplete"
    pending_return_packets = sorted(inbox_dir.glob("*.zip"))
    report = {
        "schema_version": EXTERNAL_COHORT_RETURN_INBOX_SYNC_SCHEMA_VERSION,
        "status": status,
        "expansion_path": _project_relative_path_or_absolute(expansion_file, project_root),
        "processed_packets": processed_packets,
        "num_processed_packets": len(processed_packets),
        "skipped_processed_packets": skipped_processed_packets,
        "num_skipped_processed_packets": len(skipped_processed_packets),
        "rejected_packets": rejected_packets,
        "num_rejected_packets": len(rejected_packets),
        "skipped_rejected_packets": skipped_rejected_packets,
        "num_skipped_rejected_packets": len(skipped_rejected_packets),
        "rejected_return_summary": rejected_report["rejected_return_summary"],
        "rejected_returns_report_file": "rejected_returns_report.json",
        "applied_reports": applied_reports,
        "pending_return_packets": [_relative_or_absolute(path, external_dir) for path in pending_return_packets],
        "pending_return_packet_paths": [str(path.resolve()) for path in pending_return_packets],
        "return_inbox": return_inbox,
        "return_archive": return_archive,
        "return_reject_archive": return_reject_archive,
        "return_inbox_state_file": "return_inbox_state.json",
        "final_refresh_status": final_refresh.get("status") if isinstance(final_refresh, dict) else None,
        "next_command_id": next_action.get("next_command_id"),
        "next_command": next_action.get("next_command"),
        "next_script_id": next_action.get("next_script_id"),
        "next_script": next_action.get("next_script"),
        "next_script_file": next_action.get("next_script_file"),
    }
    write_json(external_dir / "return_inbox_sync.json", report)
    return report


def summarize_external_cohort_rejected_returns(
    external_dir: str | Path,
    *,
    state: dict[str, Any] | None = None,
    extra_entries: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    external_root = Path(external_dir).resolve()
    _, _, return_reject_archive = _ensure_external_return_dirs(external_root)
    rejected_dir = external_root / str(return_reject_archive["candidate_archive"])
    current_state = state if isinstance(state, dict) else _load_external_return_inbox_state(external_root)
    persisted = _read_optional_json(external_root / "rejected_returns_report.json")
    entries_by_file: dict[str, dict[str, Any]] = {}
    candidate_entries: list[Any] = []
    if isinstance(current_state.get("rejected_packets"), list):
        candidate_entries.extend(current_state["rejected_packets"])
    if isinstance(persisted, dict):
        if isinstance(persisted.get("rejected_candidate_packets"), list):
            candidate_entries.extend(persisted["rejected_candidate_packets"])
        raw_summary = persisted.get("rejected_return_summary")
        if isinstance(raw_summary, dict) and isinstance(raw_summary.get("rejected_candidate_packets"), list):
            candidate_entries.extend(raw_summary["rejected_candidate_packets"])
    if extra_entries is not None:
        candidate_entries.extend(list(extra_entries))
    for entry in _normalize_external_rejected_return_entries(external_root, candidate_entries):
        packet_path = Path(entry["packet_path"])
        if packet_path.exists():
            entries_by_file[entry["packet_file"]] = entry
    for packet_path in sorted(rejected_dir.glob("*.zip")):
        packet_file = _relative_or_absolute(packet_path, external_root)
        entries_by_file.setdefault(
            packet_file,
            {
                "packet_file": packet_file,
                "packet_path": str(packet_path.resolve()),
                "rejection_error": "unknown",
            },
        )
    rejected_candidate_packets = [entries_by_file[key] for key in sorted(entries_by_file)]
    return {
        "num_rejected_candidate_packets": len(rejected_candidate_packets),
        "rejected_candidate_packets": rejected_candidate_packets,
    }


def write_external_cohort_rejected_returns_report(
    external_dir: str | Path,
    output: str | Path | None = None,
    *,
    state: dict[str, Any] | None = None,
    extra_entries: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    external_root = Path(external_dir).resolve()
    output_path = Path(output).resolve() if output is not None else external_root / "rejected_returns_report.json"
    _, _, return_reject_archive = _ensure_external_return_dirs(external_root)
    summary = summarize_external_cohort_rejected_returns(external_root, state=state, extra_entries=extra_entries)
    report = {
        "schema_version": EXTERNAL_COHORT_REJECTED_RETURNS_REPORT_SCHEMA_VERSION,
        "root": _external_root_ref(external_root),
        "status": "passed" if summary["num_rejected_candidate_packets"] == 0 else "incomplete",
        "external_dir": _external_dir_ref(external_root),
        "return_reject_archive": return_reject_archive,
        "rejected_return_summary": summary,
        "num_rejected_candidate_packets": summary["num_rejected_candidate_packets"],
        "rejected_candidate_packets": summary["rejected_candidate_packets"],
    }
    write_json(output_path, report)
    return report


def _normalize_external_rejected_return_entries(
    external_dir: Path,
    raw_entries: Iterable[Any],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        packet_file = entry.get("packet_file")
        packet_path = entry.get("packet_path")
        if not isinstance(packet_file, str) or not packet_file.strip():
            if isinstance(packet_path, str) and packet_path.strip():
                path_value = Path(packet_path)
                packet_file = _relative_or_absolute(path_value, external_dir)
            else:
                continue
        if isinstance(packet_path, str) and packet_path.strip():
            packet_path_value = Path(packet_path)
            if not packet_path_value.is_absolute():
                packet_path_value = external_dir / packet_path_value
            resolved_packet_path = packet_path_value.resolve()
        else:
            resolved_packet_path = (external_dir / packet_file).resolve()
        rejection_error = entry.get("rejection_error")
        if not isinstance(rejection_error, str) or not rejection_error.strip():
            rejection_error = entry.get("error")
        normalized_entry = {
            "packet_file": str(packet_file),
            "packet_path": str(resolved_packet_path),
            "packet_fingerprint": str(entry.get("packet_fingerprint") or "") if entry.get("packet_fingerprint") else None,
            "rejection_error": str(rejection_error) if isinstance(rejection_error, str) and rejection_error.strip() else "unknown",
        }
        normalized.append(normalized_entry)
    return normalized


def watch_external_cohort_return_inbox(
    expansion_path: str | Path,
    *,
    root: str | Path = ".",
    real_system_validation_path: str | Path | None = None,
    interval_s: float = 60.0,
    max_iterations: int = 1,
    stop_when_ready: bool = False,
    stop_when_rejected: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if interval_s < 0:
        raise ValueError("interval_s must be non-negative")
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")

    expansion_file = Path(expansion_path).resolve()
    external_dir = expansion_file.parent
    output_file = Path(output_path).resolve() if output_path is not None else None
    project_root = Path(root).resolve()
    iteration_count = 0
    last_sync_summary: dict[str, Any] | None = None

    while True:
        iteration_count += 1
        last_sync_summary = sync_external_cohort_return_inbox(
            expansion_file,
            root=root,
            real_system_validation_path=real_system_validation_path,
        )
        expansion = read_json(expansion_file)
        if not isinstance(expansion, dict):
            raise ValueError("external cohort expansion plan must remain a JSON object during watch")
        ready = expansion.get("status") == "not_needed"
        rejected_return_summary = (
            last_sync_summary.get("rejected_return_summary", {})
            if isinstance(last_sync_summary, dict) and isinstance(last_sync_summary.get("rejected_return_summary"), dict)
            else {}
        )
        has_rejected_returns = bool(rejected_return_summary.get("num_rejected_candidate_packets"))

        stop_reason: str | None = None
        if stop_when_ready and ready:
            stop_reason = "ready"
        elif stop_when_rejected and has_rejected_returns:
            stop_reason = "rejected_returns"
        elif max_iterations > 0 and iteration_count >= max_iterations:
            stop_reason = "max_iterations"

        watch_stop_actions = _external_watch_stop_actions(
            external_dir,
            expansion.get("operator_scripts", {}) if isinstance(expansion.get("operator_scripts"), dict) else {},
            expansion.get("operator_script_files", {}) if isinstance(expansion.get("operator_script_files"), dict) else {},
            repo_root=project_root,
        )
        next_action = _external_watch_next_action(
            {
                "recommended_next_command_id": expansion.get("recommended_next_command_id"),
                "recommended_next_command": expansion.get("recommended_next_command"),
                "recommended_next_script_id": expansion.get("recommended_next_script_id"),
                "recommended_next_script": expansion.get("recommended_next_script"),
                "recommended_next_script_file": expansion.get("recommended_next_script_file"),
                "watch_stop_actions": watch_stop_actions,
            },
            stop_reason=stop_reason,
        )

        report = {
            "schema_version": EXTERNAL_COHORT_RETURN_INBOX_WATCH_SCHEMA_VERSION,
            "external_dir": _external_dir_ref(external_dir, project_root=project_root),
            "expansion_path": _project_relative_path_or_absolute(expansion_file, project_root),
            "expansion_status": expansion.get("status"),
            "return_inbox_state_file": expansion.get("return_inbox_state_file"),
            "return_inbox_sync_report_file": expansion.get("return_inbox_sync_report_file"),
            "rejected_returns_report_file": expansion.get("rejected_returns_report_file"),
            "iteration_count": iteration_count,
            "stop_when_ready": stop_when_ready,
            "stop_when_rejected": stop_when_rejected,
            "ready_for_completion": ready,
            "has_rejected_returns": has_rejected_returns,
            "pending_return_packets": last_sync_summary.get("pending_return_packets", []) if isinstance(last_sync_summary, dict) else [],
            "pending_return_packet_paths": last_sync_summary.get("pending_return_packet_paths", []) if isinstance(last_sync_summary, dict) else [],
            "rejected_return_summary": rejected_return_summary,
            "last_sync_summary": last_sync_summary,
            "watch_stop_exit_codes": dict(EXTERNAL_COHORT_WATCH_STOP_EXIT_CODES),
            "watch_stop_actions": watch_stop_actions,
            "stop_reason": stop_reason,
            "stop_exit_code": _external_watch_stop_exit_code(stop_reason),
            "stop_action": watch_stop_actions.get(stop_reason) if isinstance(stop_reason, str) else None,
            "next_command_id": next_action.get("next_command_id"),
            "next_command": next_action.get("next_command"),
            "next_script_id": next_action.get("next_script_id"),
            "next_script": next_action.get("next_script"),
            "next_script_file": next_action.get("next_script_file"),
        }
        if output_file is not None:
            write_json(output_file, report)
        if stop_reason is not None:
            return report
        if interval_s > 0:
            time.sleep(interval_s)


def validate_normalized_external_score(
    path: str | Path,
    *,
    expected_benchmark_id: str | None = None,
    min_systems: int = 2,
) -> tuple[str, ...]:
    try:
        data = read_json(path)
    except Exception as exc:
        return (f"cannot read normalized score artifact: {exc}",)
    errors: list[str] = []
    if not isinstance(data, dict):
        return ("normalized score artifact must be a JSON object",)
    if data.get("score_schema_version") != EXTERNAL_SCORE_SCHEMA_VERSION:
        errors.append(f"score_schema_version must be {EXTERNAL_SCORE_SCHEMA_VERSION}")
    benchmark_id = data.get("benchmark_id")
    if not benchmark_id:
        errors.append("benchmark_id is required")
    elif expected_benchmark_id is not None and benchmark_id != expected_benchmark_id:
        errors.append(f"benchmark_id must be {expected_benchmark_id}")
    if not data.get("source_artifact"):
        errors.append("source_artifact is required")
    systems = data.get("systems")
    if not isinstance(systems, list) or len(systems) < min_systems:
        errors.append(f"systems must contain at least {min_systems} scored systems")
        systems = []
    seen: set[str] = set()
    for index, row in enumerate(systems, start=1):
        if not isinstance(row, dict):
            errors.append(f"systems[{index}] must be an object")
            continue
        system_id = row.get("system_id")
        if not system_id:
            errors.append(f"systems[{index}].system_id is required")
        elif str(system_id) in seen:
            errors.append(f"duplicate system_id: {system_id}")
        else:
            seen.add(str(system_id))
        if numeric_or_none(row.get("score")) is None:
            errors.append(f"systems[{index}].score must be numeric")
    errors.extend(_validate_external_run_config(data.get("run_config"), seen))
    return tuple(errors)


def validate_external_correlation_report(
    path: str | Path,
    *,
    expected_benchmark_id: str | None = None,
    min_common_systems: int = 2,
) -> tuple[str, ...]:
    try:
        report = read_json(path)
    except Exception as exc:
        return (f"cannot read correlation report: {exc}",)
    return validate_external_correlation_payload(
        report,
        expected_benchmark_id=expected_benchmark_id,
        min_common_systems=min_common_systems,
    )


def validate_external_correlation_payload(
    report: dict[str, Any],
    *,
    expected_benchmark_id: str | None = None,
    min_common_systems: int = 2,
) -> tuple[str, ...]:
    if not isinstance(report, dict):
        return ("correlation report must be a JSON object",)
    errors: list[str] = []
    if report.get("analysis_schema_version") != "amst-external-correlation-v1":
        errors.append("analysis_schema_version must be amst-external-correlation-v1")
    results = report.get("external_results")
    if not isinstance(results, list) or not results:
        errors.append("external_results must be a non-empty list")
        return tuple(errors)
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            errors.append(f"external_results[{index}] must be an object")
            continue
        benchmark_id = item.get("external_benchmark_id")
        if item.get("external_score_schema_version") != EXTERNAL_SCORE_SCHEMA_VERSION:
            errors.append(f"external_results[{index}] missing normalized score schema provenance")
        if not benchmark_id:
            errors.append(f"external_results[{index}].external_benchmark_id is required")
        elif expected_benchmark_id is not None and benchmark_id != expected_benchmark_id:
            errors.append(f"external_results[{index}].external_benchmark_id must be {expected_benchmark_id}")
        if not item.get("external_source_artifact"):
            errors.append(f"external_results[{index}].external_source_artifact is required")
        common_systems = _optional_int(item.get("num_common_systems"))
        if common_systems is None or common_systems < min_common_systems:
            errors.append(f"external_results[{index}] must contain at least {min_common_systems} common systems")
        if item.get("spearman") is None:
            errors.append(f"external_results[{index}].spearman is required")
        if item.get("kendall_tau_b") is None:
            errors.append(f"external_results[{index}].kendall_tau_b is required")
        systems = item.get("systems")
        if not isinstance(systems, list):
            errors.append(f"external_results[{index}].systems must be a list")
            continue
        system_ids = _system_ids_from_common_rows(systems, errors, prefix=f"external_results[{index}].systems")
        if common_systems is not None and len(system_ids) != common_systems:
            errors.append(f"external_results[{index}].num_common_systems must match systems length")
        errors.extend(
            f"external_results[{index}].{error}"
            for error in _validate_external_run_config(item.get("external_run_config"), system_ids)
        )
    return tuple(errors)


def _validate_external_run_config(value: Any, system_ids: set[str]) -> tuple[str, ...]:
    errors: list[str] = []
    if not isinstance(value, dict) or not value:
        return ("run_config must be a non-empty object",)
    for field in REQUIRED_EXTERNAL_RUN_CONFIG_FIELDS:
        if value.get(field) in (None, "", [], {}):
            errors.append(f"run_config.{field} is required")
    cohort = value.get("system_cohort")
    if isinstance(cohort, list):
        cohort_ids = {str(item) for item in cohort if str(item).strip()}
        if len(cohort_ids) != len(cohort):
            errors.append("run_config.system_cohort must contain unique non-empty system ids")
        if system_ids and cohort_ids != system_ids:
            errors.append("run_config.system_cohort must match scored systems")
    elif cohort is not None:
        errors.append("run_config.system_cohort must be a list")
    return tuple(errors)


def _external_benchmark_id(path: Path) -> str:
    try:
        data = read_json(path)
    except Exception:
        data = None
    if isinstance(data, dict) and data.get("benchmark_id"):
        return str(data["benchmark_id"])
    stem = path.stem
    return stem[:-7] if stem.endswith("_scores") else stem


def _validate_expected_score_alignment(
    project: Path,
    correlation_path: Path,
    expected_score_path: Path,
    *,
    benchmark_id: str,
) -> tuple[str, ...]:
    try:
        report = read_json(correlation_path)
    except Exception as exc:
        return (f"{_rel(correlation_path, project)}: cannot read correlation report: {exc}",)
    try:
        score_artifact = read_json(expected_score_path)
    except Exception as exc:
        return (f"{_rel(expected_score_path, project)}: cannot read normalized score artifact: {exc}",)
    if not isinstance(report, dict) or not isinstance(score_artifact, dict):
        return ("canonical correlation/score artifacts must both be JSON objects",)

    results = report.get("external_results")
    if not isinstance(results, list):
        return (f"{_rel(correlation_path, project)}: external_results must be a list",)
    matching = [
        item for item in results if isinstance(item, dict) and str(item.get("external_benchmark_id")) == benchmark_id
    ]
    if len(matching) != 1:
        return (f"{_rel(correlation_path, project)}: expected exactly one external result for {benchmark_id}",)
    result = matching[0]

    errors: list[str] = []
    reported_score_path = result.get("external_score_path")
    if not isinstance(reported_score_path, str) or not reported_score_path.strip():
        errors.append(f"{_rel(correlation_path, project)}: external_score_path is required")
    elif not _path_matches_expected(reported_score_path, expected_score_path, project=project, anchor=correlation_path.parent):
        errors.append(
            f"{_rel(correlation_path, project)}: external_score_path must reference {_rel(expected_score_path, project)}"
        )
    if result.get("external_source_artifact") != score_artifact.get("source_artifact"):
        errors.append(
            f"{_rel(correlation_path, project)}: external_source_artifact must match {_rel(expected_score_path, project)}"
        )
    if result.get("external_run_config") != score_artifact.get("run_config"):
        errors.append(
            f"{_rel(correlation_path, project)}: external_run_config must match {_rel(expected_score_path, project)}"
        )
    score_systems = {
        str(item.get("system_id"))
        for item in score_artifact.get("systems", [])
        if isinstance(item, dict) and item.get("system_id")
    }
    result_systems = {
        str(item.get("system_id"))
        for item in result.get("systems", [])
        if isinstance(item, dict) and item.get("system_id")
    }
    if score_systems and score_systems != result_systems:
        errors.append(
            f"{_rel(correlation_path, project)}: scored systems must match {_rel(expected_score_path, project)}"
        )
    return tuple(errors)


def _path_matches_expected(candidate: str, expected: Path, *, project: Path, anchor: Path) -> bool:
    raw = Path(candidate)
    expected_resolved = expected.resolve()
    search_paths = (raw,) if raw.is_absolute() else (project / raw, anchor / raw)
    return any(path.resolve() == expected_resolved for path in search_paths)


def _external_correlation_entries(path: Path) -> dict[str, set[str]]:
    try:
        report = read_json(path)
    except Exception:
        return {}
    if not isinstance(report, dict):
        return {}
    entries: dict[str, set[str]] = {}
    results = report.get("external_results")
    if not isinstance(results, list):
        return entries
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        benchmark_id = item.get("external_benchmark_id")
        if not benchmark_id:
            continue
        systems = item.get("systems")
        if not isinstance(systems, list):
            continue
        row_errors: list[str] = []
        system_ids = _system_ids_from_common_rows(systems, row_errors, prefix=f"external_results[{index}].systems")
        if not system_ids:
            continue
        entries[str(benchmark_id)] = system_ids
    return entries


def _resolve_external_correlation_paths(
    project: Path,
    output_dir: Path,
    benchmark_ids: tuple[str, ...] | None,
) -> tuple[Path, ...]:
    required_ids = benchmark_ids if benchmark_ids is not None else tuple(item[0] for item in DEFAULT_EXTERNAL_BENCHMARKS)
    return tuple(output_dir / f"{benchmark_id}_correlation.json" for benchmark_id in required_ids)


def _normalize_system_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _classify_shared_systems(system_ids: list[str], aliases: dict[str, tuple[str, ...]]) -> list[str]:
    normalized = {_normalize_system_token(system_id) for system_id in system_ids}
    matched: list[str] = []
    for canonical_id, candidate_aliases in sorted(aliases.items()):
        alias_tokens = {_normalize_system_token(alias) for alias in candidate_aliases}
        if normalized & alias_tokens:
            matched.append(canonical_id)
    return matched


def _read_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    try:
        payload = read_json(candidate)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _available_real_memory_system_rows(real_system_validation: dict[str, Any] | None) -> list[dict[str, str]]:
    if not isinstance(real_system_validation, dict):
        return []
    systems = real_system_validation.get("systems")
    if not isinstance(systems, list):
        return []
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in systems:
        if not isinstance(row, dict):
            continue
        provider = row.get("provider")
        system_id = row.get("system_id")
        if not isinstance(provider, str) or not provider.strip():
            continue
        if not isinstance(system_id, str) or not system_id.strip():
            continue
        canonical_provider = _classify_shared_systems([provider, system_id], REAL_MEMORY_SYSTEM_ALIASES)
        if not canonical_provider:
            continue
        provider_id = canonical_provider[0]
        if provider_id in seen:
            continue
        seen.add(provider_id)
        item = {"provider": provider_id, "system_id": system_id}
        report_path = row.get("report_path")
        if isinstance(report_path, str) and report_path:
            item["report_path"] = report_path
        rows.append(item)
    return rows


def _available_real_memory_systems(real_system_validation: dict[str, Any] | None) -> dict[str, str]:
    return {row["provider"]: row["system_id"] for row in _available_real_memory_system_rows(real_system_validation)}


def _correlation_path_for_benchmark(correlation_paths: Iterable[str | Path], benchmark_id: str) -> Path:
    for raw_path in correlation_paths:
        path = Path(raw_path)
        if _external_benchmark_id(path) == benchmark_id:
            return path
    return Path(f"reports/external/{benchmark_id}_correlation.json")


def _external_benchmark_context(correlation_path: Path, benchmark_id: str) -> dict[str, Any]:
    score_path = correlation_path.parent / f"{benchmark_id}_scores.json"
    score_payload = _read_optional_json(score_path)
    score_run_config = score_payload.get("run_config") if isinstance(score_payload, dict) else None
    score_system_cohort = score_run_config.get("system_cohort") if isinstance(score_run_config, dict) else None
    context = {
        "benchmark_id": benchmark_id,
        "score_artifact": str(score_path),
        "correlation_report": str(correlation_path),
        "current_source_artifact": score_payload.get("source_artifact") if isinstance(score_payload, dict) else None,
        "current_system_cohort": list(score_system_cohort) if isinstance(score_system_cohort, list) else [],
        "current_run_config": score_run_config if isinstance(score_run_config, dict) else None,
    }
    correlation_payload = _read_optional_json(correlation_path)
    if isinstance(correlation_payload, dict):
        external_results = correlation_payload.get("external_results")
        if isinstance(external_results, list) and external_results:
            result = external_results[0]
            if isinstance(result, dict):
                run_config = result.get("external_run_config")
                system_cohort = run_config.get("system_cohort") if isinstance(run_config, dict) else None
                if isinstance(system_cohort, list) and system_cohort:
                    context["current_system_cohort"] = list(system_cohort)
                if isinstance(result.get("external_source_artifact"), str) and result["external_source_artifact"]:
                    context["current_source_artifact"] = result["external_source_artifact"]
                if isinstance(result.get("external_score_path"), str) and result["external_score_path"]:
                    context["score_artifact"] = result["external_score_path"]
    return context


def _external_candidate_command_bundle(
    project: Path,
    *,
    provider: str,
    system_id: str,
    real_system_report_path: str | None,
    target_system_cohort: list[str],
    regeneration_targets: list[dict[str, Any]],
) -> dict[str, Any]:
    amst_reports = [
        "reports/examples/amst_main_v1_strict_public_dev_no_memory_report.json",
        "reports/examples/amst_main_v1_strict_public_dev_oracle_memory_report.json",
    ]
    if isinstance(real_system_report_path, str) and real_system_report_path:
        amst_reports.append(real_system_report_path)
    normalize_commands: list[dict[str, str]] = []
    score_artifacts: list[str] = []
    required_env_vars: dict[str, str] = {}
    for item in regeneration_targets:
        benchmark_id = str(item["benchmark_id"])
        score_artifact = str(item["score_artifact"])
        score_artifacts.append(score_artifact)
        if benchmark_id == "longmemeval":
            manifest_env = f"LONGMEMEVAL_{provider.upper()}_MANIFEST"
            required_env_vars[manifest_env] = (
                f"Manifest path for longmemeval {provider} same-cohort scores using {system_id}."
            )
            normalize_commands.append(
                {
                    "benchmark_id": benchmark_id,
                    "score_artifact": score_artifact,
                    "command": (
                        "PYTHONPATH=. python -m agent_memory_benchmark normalize-longmemeval-scores "
                        f"--manifest {manifest_env} "
                        f"--output {score_artifact}"
                    ),
                }
            )
        else:
            raw_env = f"{benchmark_id.upper()}_{provider.upper()}_RAW"
            run_config_env = f"{benchmark_id.upper()}_{provider.upper()}_RUN_CONFIG"
            required_env_vars[raw_env] = f"Raw normalized-score input for {benchmark_id} {provider} same-cohort results."
            required_env_vars[run_config_env] = f"Run-config JSON for {benchmark_id} {provider} same-cohort results."
            normalize_commands.append(
                {
                    "benchmark_id": benchmark_id,
                    "score_artifact": score_artifact,
                    "command": (
                        "PYTHONPATH=. python -m agent_memory_benchmark normalize-external-scores "
                        f"--input {raw_env} "
                        f"--benchmark-id {benchmark_id} "
                        f"--output {score_artifact} "
                        f"--run-config {run_config_env}"
                    ),
                }
            )
    correlate_command = (
        "PYTHONPATH=. python -m agent_memory_benchmark external-correlation-batch "
        f"--amst-reports {' '.join(amst_reports)} "
        f"--external-scores {' '.join(score_artifacts)} "
        "--output-dir reports/external"
    )
    refresh_command = (
        "PYTHONPATH=. python -m agent_memory_benchmark refresh-external-canonical "
        "--root . --output-dir reports/external "
        "--real-system-validation reports/real_system_runs/canonical_public_dev_refresh_current_matrix_validation.json "
        "--min-shared-systems 3 --min-control-shared-systems 1 --min-real-memory-shared-systems 1"
    )
    return {
        "provider": provider,
        "system_id": system_id,
        "target_system_cohort": target_system_cohort,
        "amst_report_paths": amst_reports,
        "required_env_vars": required_env_vars,
        "normalize_commands": normalize_commands,
        "correlate_command": correlate_command,
        "refresh_command": refresh_command,
    }


def _external_operator_script_preamble(base_dir: Path, repo_root: Path | None = None) -> str:
    repo_root_text = str(repo_root) if repo_root is not None else ""
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'EXTERNAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"\n'
        f'REPO_ROOT_DEFAULT="{repo_root_text}"\n'
        'if [ -n "$REPO_ROOT_DEFAULT" ] && [ -d "$REPO_ROOT_DEFAULT/agent_memory_benchmark" ] && [ -f "$REPO_ROOT_DEFAULT/pyproject.toml" ]; then\n'
        '  REPO_ROOT="$REPO_ROOT_DEFAULT"\n'
        "else\n"
        '  REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"\n'
        "fi\n"
        'cd "$REPO_ROOT"\n\n'
    )


def _external_candidate_script_preamble(repo_root: Path | None = None) -> str:
    repo_root_text = str(repo_root) if repo_root is not None else ""
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'EXTERNAL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"\n'
        f'REPO_ROOT_DEFAULT="{repo_root_text}"\n'
        'if [ -n "$REPO_ROOT_DEFAULT" ] && [ -d "$REPO_ROOT_DEFAULT/agent_memory_benchmark" ] && [ -f "$REPO_ROOT_DEFAULT/pyproject.toml" ]; then\n'
        '  REPO_ROOT="$REPO_ROOT_DEFAULT"\n'
        "else\n"
        '  REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"\n'
        "fi\n"
        'cd "$REPO_ROOT"\n\n'
    )


def _write_external_operator_scripts(base_dir: Path, report: dict[str, Any], *, repo_root: Path | None = None) -> dict[str, str]:
    scripts_dir = base_dir / "bin"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    preamble = _external_operator_script_preamble(base_dir, repo_root=repo_root)
    scripts: dict[str, tuple[str, str]] = {
        "build_cohort_expansion_plan": (
            "build_cohort_expansion_plan.sh",
            preamble
            + "PYTHONPATH=. python -m agent_memory_benchmark build-external-cohort-expansion-plan "
            + '--correlations "$EXTERNAL_DIR"/*_correlation.json '
            + "--real-system-validation reports/real_system_runs/canonical_public_dev_refresh_current_matrix_validation.json "
            + '--output "$EXTERNAL_DIR/cohort_expansion_plan.json" '
            + "--min-shared-systems 3 --min-control-shared-systems 1 --min-real-memory-shared-systems 1\n",
        ),
        "refresh_canonical": (
            "refresh_canonical.sh",
            preamble
            + "PYTHONPATH=. python -m agent_memory_benchmark refresh-external-canonical "
            + '--root . --output-dir "$EXTERNAL_DIR" '
            + "--real-system-validation reports/real_system_runs/canonical_public_dev_refresh_current_matrix_validation.json "
            + "--min-shared-systems 3 --min-control-shared-systems 1 --min-real-memory-shared-systems 1\n",
        ),
        "apply_return_packet": (
            "apply_return_packet.sh",
            preamble
            + 'PACKET_PATH="${1:-}"' + "\n"
            + 'if [ -z "$PACKET_PATH" ]; then echo "usage: $0 /abs/path/to/provider_return.zip" >&2; exit 2; fi' + "\n"
            + "PYTHONPATH=. python -m agent_memory_benchmark apply-external-cohort-return-packet "
            + '--expansion "$EXTERNAL_DIR/cohort_expansion_plan.json" '
            + '--packet "$PACKET_PATH" '
            + '--output "$EXTERNAL_DIR/last_return_apply.json"' + "\n",
        ),
        "sync_return_inbox": (
            "sync_return_inbox.sh",
            preamble
            + "PYTHONPATH=. python -m agent_memory_benchmark sync-external-cohort-return-inbox "
            + '--expansion "$EXTERNAL_DIR/cohort_expansion_plan.json" '
            + '--output "$EXTERNAL_DIR/return_inbox_sync.json"' + "\n",
        ),
        "watch_return_inbox": (
            "watch_return_inbox.sh",
            preamble
            + "PYTHONPATH=. python -m agent_memory_benchmark watch-external-cohort-return-inbox "
            + '--expansion "$EXTERNAL_DIR/cohort_expansion_plan.json" '
            + '--interval-s 120 --max-iterations 0 --stop-when-ready --stop-when-rejected '
            + '--output "$EXTERNAL_DIR/return_inbox_watch.json"' + "\n",
        ),
        "review_rejected_returns": (
            "review_rejected_returns.sh",
            preamble
            + "PYTHONPATH=. python -m agent_memory_benchmark summarize-external-cohort-rejected-returns "
            + '--external-dir "$EXTERNAL_DIR" '
            + '--output "$EXTERNAL_DIR/rejected_returns_report.json"' + "\n",
        ),
    }
    candidates = report.get("available_real_memory_candidates")
    if isinstance(candidates, list):
        for item in candidates:
            if not isinstance(item, dict):
                continue
            provider = item.get("provider")
            bundle = item.get("command_bundle")
            if not isinstance(provider, str) or not provider or not isinstance(bundle, dict):
                continue
            script_id = f"expand_with_{provider}"
            filename = f"{script_id}.sh"
            lines = [preamble]
            required_env_vars = bundle.get("required_env_vars")
            if isinstance(required_env_vars, dict):
                for env_name, description in sorted(required_env_vars.items()):
                    if not isinstance(env_name, str) or not env_name:
                        continue
                    comment = str(description) if description is not None else ""
                    lines.append(f'# {env_name}: {comment}\n')
                    lines.append(f': "${{{env_name}:?Set {env_name}}}"\n')
            normalize_commands = bundle.get("normalize_commands")
            target_system_cohort = (
                [str(system_id) for system_id in bundle.get("target_system_cohort", []) if str(system_id).strip()]
                if isinstance(bundle.get("target_system_cohort"), list)
                else []
            )
            provider_system_id = str(bundle.get("system_id") or "")
            merge_enabled = len(target_system_cohort) > 1 and bool(provider_system_id)
            if merge_enabled:
                lines.append(f'TMP_DIR="${{EXTERNAL_DIR}}/runtime/{script_id}_$(date -u +%Y%m%dT%H%M%SZ)"\n')
                lines.append('mkdir -p "$TMP_DIR"\n')
            if isinstance(normalize_commands, list):
                for command_row in normalize_commands:
                    if not isinstance(command_row, dict):
                        continue
                    raw_command = command_row.get("command")
                    if not isinstance(raw_command, str) or not raw_command.strip():
                        continue
                    benchmark_id = str(command_row.get("benchmark_id") or "")
                    score_artifact = str(command_row.get("score_artifact") or "")
                    rendered_command = _external_command_with_env_refs(raw_command, required_env_vars)
                    if merge_enabled and benchmark_id and score_artifact:
                        provider_score = f"$TMP_DIR/{benchmark_id}_{provider}_scores.json"
                        existing_score = f"$TMP_DIR/{benchmark_id}_existing_scores.json"
                        rendered_command = _external_replace_output_arg(rendered_command, score_artifact, provider_score)
                        lines.append(f'cp {shlex.quote(score_artifact)} "{existing_score}"\n')
                        lines.append(rendered_command)
                        lines.append("\n")
                        lines.append(
                            "PYTHONPATH=. python -m agent_memory_benchmark merge-external-scores "
                            f'--input "{existing_score}" "{provider_score}" '
                            f"--benchmark-id {shlex.quote(benchmark_id)} "
                            f"--system-cohort {' '.join(shlex.quote(item) for item in target_system_cohort)} "
                            f"--replace-system-id {shlex.quote(provider_system_id)} "
                            f'--source-manifest "$EXTERNAL_DIR/{benchmark_id}_scores_source_manifest.json" '
                            f"--output {shlex.quote(score_artifact)}"
                        )
                    else:
                        lines.append(rendered_command)
                    lines.append("\n")
            correlate_command = bundle.get("correlate_command")
            if isinstance(correlate_command, str) and correlate_command.strip():
                lines.append(correlate_command)
                lines.append("\n")
            refresh_command = bundle.get("refresh_command")
            if isinstance(refresh_command, str) and refresh_command.strip():
                lines.append(refresh_command)
                lines.append("\n")
            scripts[script_id] = (filename, "".join(lines))

    refs: dict[str, str] = {}
    for script_id, (filename, content) in scripts.items():
        script_path = scripts_dir / filename
        script_path.write_text(content, encoding="utf-8")
        script_path.chmod(script_path.stat().st_mode | 0o111)
        refs[script_id] = _relative_or_absolute(script_path, base_dir)
    return refs


def _build_external_handoff_manifest(
    report: dict[str, Any],
    *,
    base_dir: Path,
    repo_root: Path | None,
) -> dict[str, Any]:
    operator_commands = _external_operator_commands(base_dir, report, repo_root=repo_root)
    recommended_candidate_raw = (
        report.get("recommended_completion_candidate")
        if isinstance(report.get("recommended_completion_candidate"), dict)
        else None
    )
    recommended_candidate = (
        _external_materialized_candidate_summary(recommended_candidate_raw)
        if isinstance(recommended_candidate_raw, dict)
        else None
    )
    operator_scripts = report.get("operator_scripts", {}) if isinstance(report.get("operator_scripts"), dict) else {}
    operator_script_files = (
        report.get("operator_script_files", {}) if isinstance(report.get("operator_script_files"), dict) else {}
    )
    watch_stop_actions = _external_watch_stop_actions(
        base_dir,
        operator_scripts,
        operator_script_files,
        repo_root=repo_root,
    )
    recommended_next = _external_recommended_next_refs(
        report,
        operator_commands=operator_commands,
        operator_scripts=operator_scripts,
        operator_script_files=operator_script_files,
    )
    runtime_snapshot = _external_handoff_runtime_snapshot(report, base_dir=base_dir)
    pending_return_packets = (
        list(report.get("pending_return_packets", []))
        if isinstance(report.get("pending_return_packets"), list)
        else []
    )
    rejected_return_summary = (
        report.get("rejected_return_summary")
        if isinstance(report.get("rejected_return_summary"), dict)
        else {}
    )
    return {
        "schema_version": EXTERNAL_COHORT_EXPANSION_HANDOFF_MANIFEST_SCHEMA_VERSION,
        "root": (
            report.get("root")
            if isinstance(report.get("root"), str)
            else _external_root_ref(base_dir, project_root=repo_root)
        ),
        "external_dir": _external_dir_ref(base_dir, project_root=repo_root),
        "status": report.get("status"),
        "completion_gate": report.get("completion_gate", {}),
        "current_shared_systems": report.get("current_shared_systems", []),
        "current_shared_control_systems": report.get("current_shared_control_systems", []),
        "current_shared_real_memory_systems": report.get("current_shared_real_memory_systems", []),
        "recommended_next_priority": report.get("recommended_next_priority"),
        "expansion_plan_file": report.get("expansion_plan_file"),
        "expansion_plan_path": report.get("expansion_plan_path"),
        "handoff_manifest_file": report.get("handoff_manifest_file"),
        "handoff_manifest_path": report.get("handoff_manifest_path"),
        "validation_file": report.get("validation_file"),
        "validation_path": report.get("validation_path"),
        "validation_status": report.get("validation_status"),
        "validation_errors": report.get("validation_errors", []),
        "readme_file": report.get("readme_file"),
        "readme_path": report.get("readme_path"),
        "return_inbox": report.get("return_inbox", {}),
        "return_archive": report.get("return_archive", {}),
        "return_reject_archive": report.get("return_reject_archive", {}),
        "return_reject_archive_paths": report.get("return_reject_archive_paths", {}),
        "return_inbox_state_file": report.get("return_inbox_state_file"),
        "return_inbox_state_path": report.get("return_inbox_state_path"),
        "return_inbox_sync_report_file": report.get("return_inbox_sync_report_file"),
        "return_inbox_sync_report_path": report.get("return_inbox_sync_report_path"),
        "return_inbox_watch_file": report.get("return_inbox_watch_file"),
        "return_inbox_watch_output_file": report.get("return_inbox_watch_output_file"),
        "pending_return_packets": pending_return_packets,
        "pending_return_packet_paths": report.get("pending_return_packet_paths", []),
        "rejected_return_summary": rejected_return_summary,
        "rejected_returns_report_file": report.get("rejected_returns_report_file"),
        "rejected_returns_report_path": report.get("rejected_returns_report_path"),
        "watch_stop_exit_codes": dict(EXTERNAL_COHORT_WATCH_STOP_EXIT_CODES),
        "watch_stop_actions": watch_stop_actions,
        "recommended_completion_candidate": recommended_candidate,
        "minimum_completion_candidates": [
            _external_materialized_candidate_summary(item)
            for item in report.get("minimum_completion_candidates", [])
            if isinstance(item, dict)
        ]
        if isinstance(report.get("minimum_completion_candidates"), list)
        else [],
        "candidate_packets": report.get("candidate_packets", {}),
        "operator_scripts": operator_scripts,
        "operator_script_files": operator_script_files,
        "operator_commands": operator_commands,
        **recommended_next,
        **runtime_snapshot,
    }


_EXTERNAL_HANDOFF_SIDECAR_NEXT_ACTION_FIELDS = (
    "next_command_id",
    "next_command",
    "next_script_id",
    "next_script",
    "next_script_file",
)


def _external_nonempty_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return str(value)
    return None


def _update_external_handoff_sidecar_snapshot(
    snapshot: dict[str, Any],
    prefix: str,
    payload: dict[str, Any] | None,
    *,
    fallback_status: str | None,
    fallback_next_action: dict[str, Any],
    include_stop_reason: bool = False,
) -> None:
    status = _external_nonempty_string(payload.get("status")) if isinstance(payload, dict) else None
    status = status or fallback_status
    if status is not None:
        snapshot[f"{prefix}_status"] = status
    if include_stop_reason and isinstance(payload, dict):
        stop_reason = _external_nonempty_string(payload.get("stop_reason"))
        if stop_reason is not None:
            snapshot[f"{prefix}_stop_reason"] = stop_reason
    for field in _EXTERNAL_HANDOFF_SIDECAR_NEXT_ACTION_FIELDS:
        value = _external_nonempty_string(payload.get(field)) if isinstance(payload, dict) else None
        if value is None:
            value = _external_nonempty_string(fallback_next_action.get(field))
        if value is not None:
            snapshot[f"{prefix}_{field}"] = value


def _external_handoff_runtime_snapshot(report: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    sync_payload = None
    raw_sync_path = report.get("return_inbox_sync_report_file")
    sync_path = (
        base_dir / raw_sync_path
        if isinstance(raw_sync_path, str) and raw_sync_path.strip()
        else base_dir / "return_inbox_sync.json"
    )
    if sync_path.exists():
        candidate = _read_optional_json(sync_path)
        sync_payload = candidate if isinstance(candidate, dict) else None

    watch_payload = None
    raw_watch_path = report.get("return_inbox_watch_file")
    watch_path = (
        base_dir / raw_watch_path
        if isinstance(raw_watch_path, str) and raw_watch_path.strip()
        else base_dir / "return_inbox_watch.json"
    )
    if watch_path.exists():
        candidate = _read_optional_json(watch_path)
        watch_payload = candidate if isinstance(candidate, dict) else None

    stop_reason = _external_nonempty_string(watch_payload.get("stop_reason")) if isinstance(watch_payload, dict) else None
    snapshot: dict[str, Any] = {}
    _update_external_handoff_sidecar_snapshot(
        snapshot,
        "return_inbox_sync",
        sync_payload,
        fallback_status="not_started",
        fallback_next_action=_external_recommended_next_action(report),
    )
    _update_external_handoff_sidecar_snapshot(
        snapshot,
        "return_inbox_watch",
        watch_payload,
        fallback_status="not_started",
        fallback_next_action=_external_watch_next_action(report, stop_reason=stop_reason),
        include_stop_reason=True,
    )
    return snapshot


def _external_recommended_next_refs(
    report: dict[str, Any],
    *,
    operator_commands: dict[str, str],
    operator_scripts: dict[str, str],
    operator_script_files: dict[str, str],
) -> dict[str, Any]:
    recommended_candidate = (
        report.get("recommended_completion_candidate")
        if isinstance(report.get("recommended_completion_candidate"), dict)
        else None
    )
    next_command_id = "build_cohort_expansion_plan"
    next_script_id = "build_cohort_expansion_plan"
    pending_return_packets = (
        list(report.get("pending_return_packets", []))
        if isinstance(report.get("pending_return_packets"), list)
        else []
    )
    rejected_return_summary = (
        report.get("rejected_return_summary")
        if isinstance(report.get("rejected_return_summary"), dict)
        else {}
    )
    has_rejected_returns = bool(rejected_return_summary.get("num_rejected_candidate_packets"))
    if has_rejected_returns:
        next_command_id = "review_rejected_returns"
        next_script_id = "review_rejected_returns"
    elif pending_return_packets:
        next_command_id = "sync_return_inbox"
        next_script_id = "sync_return_inbox"
    elif report.get("status") == "ready" and recommended_candidate is not None:
        provider = recommended_candidate.get("provider")
        if isinstance(provider, str) and provider:
            next_command_id = f"expand_with_{provider}"
            next_script_id = next_command_id
    elif report.get("status") == "not_needed":
        next_command_id = "refresh_canonical"
        next_script_id = "refresh_canonical"
    return {
        "recommended_next_command_id": next_command_id,
        "recommended_next_command": operator_commands.get(next_command_id),
        "recommended_next_script_id": next_script_id,
        "recommended_next_script": operator_scripts.get(next_script_id),
        "recommended_next_script_file": operator_script_files.get(next_script_id),
    }


def _write_initial_external_return_inbox_watch_report(external_dir: Path, report: dict[str, Any]) -> Path:
    watch_path = external_dir / "return_inbox_watch.json"
    existing = _read_optional_json(watch_path) if watch_path.exists() else None
    payload = dict(existing) if isinstance(existing, dict) else {}
    next_action = _external_watch_next_action(report)
    payload.update(
        {
            "schema_version": EXTERNAL_COHORT_RETURN_INBOX_WATCH_SCHEMA_VERSION,
            "root": (
                report.get("root")
                if isinstance(report.get("root"), str)
                else _external_root_ref(external_dir)
            ),
            "external_dir": _external_dir_ref(external_dir),
            "expansion_path": report.get("expansion_plan_path"),
            "expansion_status": report.get("status"),
            "return_inbox_state_file": report.get("return_inbox_state_file"),
            "return_inbox_sync_report_file": report.get("return_inbox_sync_report_file"),
            "rejected_returns_report_file": report.get("rejected_returns_report_file"),
            "watch_stop_exit_codes": dict(EXTERNAL_COHORT_WATCH_STOP_EXIT_CODES),
            "watch_stop_actions": (
                dict(report.get("watch_stop_actions"))
                if isinstance(report.get("watch_stop_actions"), dict)
                else {}
            ),
        }
    )
    payload.setdefault("status", "not_started")
    payload.setdefault("iteration_count", 0)
    payload.setdefault("interval_s", EXTERNAL_COHORT_OPERATOR_WATCH_INTERVAL_S)
    payload.setdefault("max_iterations", 0)
    payload.setdefault("stop_when_ready", True)
    payload.setdefault("stop_when_rejected", True)
    payload.setdefault("ready_for_completion", False)
    payload.setdefault(
        "has_rejected_returns",
        bool(
            isinstance(report.get("rejected_return_summary"), dict)
            and report["rejected_return_summary"].get("num_rejected_candidate_packets")
        ),
    )
    payload.setdefault(
        "pending_return_packets",
        list(report.get("pending_return_packets", [])) if isinstance(report.get("pending_return_packets"), list) else [],
    )
    payload.setdefault(
        "pending_return_packet_paths",
        list(report.get("pending_return_packet_paths", []))
        if isinstance(report.get("pending_return_packet_paths"), list)
        else [],
    )
    payload.setdefault(
        "rejected_return_summary",
        dict(report.get("rejected_return_summary")) if isinstance(report.get("rejected_return_summary"), dict) else {},
    )
    payload.setdefault("last_sync_summary", None)
    payload.setdefault("stop_reason", None)
    payload.setdefault("stop_exit_code", None)
    payload.setdefault("stop_action", None)
    payload["next_command_id"] = next_action.get("next_command_id")
    payload["next_command"] = next_action.get("next_command")
    payload["next_script_id"] = next_action.get("next_script_id")
    payload["next_script"] = next_action.get("next_script")
    payload["next_script_file"] = next_action.get("next_script_file")
    write_json(watch_path, payload)
    return watch_path


def _write_initial_external_return_inbox_sync_report(external_dir: Path, report: dict[str, Any]) -> Path:
    sync_path = external_dir / "return_inbox_sync.json"
    existing = _read_optional_json(sync_path) if sync_path.exists() else None
    payload = dict(existing) if isinstance(existing, dict) else {}
    next_action = _external_recommended_next_action(report)
    payload.update(
        {
            "schema_version": EXTERNAL_COHORT_RETURN_INBOX_SYNC_SCHEMA_VERSION,
            "root": (
                report.get("root")
                if isinstance(report.get("root"), str)
                else _external_root_ref(external_dir)
            ),
            "expansion_path": report.get("expansion_plan_path"),
            "rejected_returns_report_file": report.get("rejected_returns_report_file"),
            "return_inbox": report.get("return_inbox"),
            "return_archive": report.get("return_archive"),
            "return_reject_archive": report.get("return_reject_archive"),
            "next_command_id": next_action.get("next_command_id"),
            "next_command": next_action.get("next_command"),
            "next_script_id": next_action.get("next_script_id"),
            "next_script": next_action.get("next_script"),
            "next_script_file": next_action.get("next_script_file"),
        }
    )
    payload.setdefault("status", "not_started")
    payload.setdefault("processed_packets", [])
    payload.setdefault("num_processed_packets", 0)
    payload.setdefault("rejected_packets", [])
    payload.setdefault("num_rejected_packets", 0)
    payload.setdefault(
        "rejected_return_summary",
        dict(report.get("rejected_return_summary")) if isinstance(report.get("rejected_return_summary"), dict) else {},
    )
    payload.setdefault(
        "pending_return_packets",
        list(report.get("pending_return_packets", [])) if isinstance(report.get("pending_return_packets"), list) else [],
    )
    payload.setdefault(
        "pending_return_packet_paths",
        list(report.get("pending_return_packet_paths", []))
        if isinstance(report.get("pending_return_packet_paths"), list)
        else [],
    )
    payload.setdefault("final_refresh_status", None)
    write_json(sync_path, payload)
    return sync_path


def _external_operator_commands(base_dir: Path, report: dict[str, Any], *, repo_root: Path | None) -> dict[str, str]:
    project = repo_root.resolve() if repo_root is not None else None
    external_dir = base_dir.resolve()
    external_dir_ref = _rel(external_dir, project) if project is not None else str(external_dir)
    commands = {
        "build_cohort_expansion_plan": (
            "PYTHONPATH=. python -m agent_memory_benchmark build-external-cohort-expansion-plan "
            f"--correlations {external_dir_ref}/*_correlation.json "
            "--real-system-validation reports/real_system_runs/canonical_public_dev_refresh_current_matrix_validation.json "
            f"--output {external_dir_ref}/cohort_expansion_plan.json "
            "--min-shared-systems 3 --min-control-shared-systems 1 --min-real-memory-shared-systems 1"
        ),
        "apply_return_packet": (
            "PYTHONPATH=. python -m agent_memory_benchmark apply-external-cohort-return-packet "
            f"--expansion {external_dir_ref}/cohort_expansion_plan.json "
            "--packet /abs/path/to/provider_return.zip "
            f"--output {external_dir_ref}/last_return_apply.json"
        ),
        "refresh_canonical": (
            "PYTHONPATH=. python -m agent_memory_benchmark refresh-external-canonical "
            f"--root . --output-dir {external_dir_ref} "
            "--real-system-validation reports/real_system_runs/canonical_public_dev_refresh_current_matrix_validation.json "
            "--min-shared-systems 3 --min-control-shared-systems 1 --min-real-memory-shared-systems 1"
        ),
        "sync_return_inbox": (
            "PYTHONPATH=. python -m agent_memory_benchmark sync-external-cohort-return-inbox "
            f"--expansion {external_dir_ref}/cohort_expansion_plan.json "
            f"--output {external_dir_ref}/return_inbox_sync.json"
        ),
        "watch_return_inbox": (
            "PYTHONPATH=. python -m agent_memory_benchmark watch-external-cohort-return-inbox "
            f"--expansion {external_dir_ref}/cohort_expansion_plan.json "
            "--interval-s 120 --max-iterations 0 --stop-when-ready --stop-when-rejected "
            f"--output {external_dir_ref}/return_inbox_watch.json"
        ),
        "review_rejected_returns": (
            "PYTHONPATH=. python -m agent_memory_benchmark summarize-external-cohort-rejected-returns "
            f"--external-dir {external_dir_ref} "
            f"--output {external_dir_ref}/rejected_returns_report.json"
        ),
    }
    operator_script_files = (
        report.get("operator_script_files", {}) if isinstance(report.get("operator_script_files"), dict) else {}
    )
    for script_id, script_file in sorted(operator_script_files.items()):
        if script_id in commands:
            continue
        if isinstance(script_file, str) and script_file:
            commands[str(script_id)] = script_file
    return commands


def _write_external_candidate_packets(base_dir: Path, report: dict[str, Any], *, repo_root: Path | None = None) -> dict[str, dict[str, Any]]:
    packets_root = base_dir / "candidates"
    packets_root.mkdir(parents=True, exist_ok=True)
    project_root = repo_root.resolve() if repo_root is not None else None
    refs: dict[str, dict[str, Any]] = {}
    candidates = report.get("available_real_memory_candidates")
    if not isinstance(candidates, list):
        return refs
    for item in candidates:
        if not isinstance(item, dict):
            continue
        provider = item.get("provider")
        system_id = item.get("system_id")
        command_bundle = item.get("command_bundle")
        if not isinstance(provider, str) or not provider or not isinstance(system_id, str) or not system_id:
            continue
        if not isinstance(command_bundle, dict):
            continue
        packet_dir = packets_root / provider
        packet_dir.mkdir(parents=True, exist_ok=True)
        run_script_rel = Path("run.sh")
        run_script_path = packet_dir / run_script_rel
        env_template_rel = Path("env.template")
        env_template_path = packet_dir / env_template_rel
        readme_rel = Path("README.md")
        readme_path = packet_dir / readme_rel
        manifest_rel = Path("packet_manifest.json")
        manifest_path = packet_dir / manifest_rel
        package_return_rel = Path("package_return.sh")
        package_return_path = packet_dir / package_return_rel

        run_script_path.write_text(
            _render_external_candidate_run_script(
                provider=provider,
                command_bundle=command_bundle,
                repo_root=repo_root,
            ),
            encoding="utf-8",
        )
        run_script_path.chmod(run_script_path.stat().st_mode | 0o111)
        env_template_path.write_text(
            _render_external_candidate_env_template(provider=provider, command_bundle=command_bundle),
            encoding="utf-8",
        )
        readme_path.write_text(
            _render_external_candidate_readme(
                provider=provider,
                system_id=system_id,
                command_bundle=command_bundle,
            ),
            encoding="utf-8",
        )
        package_return_path.write_text(
            _render_external_candidate_package_return_script(
                provider=provider,
                required_benchmark_ids=report.get("required_benchmark_ids"),
                repo_root=repo_root,
            ),
            encoding="utf-8",
        )
        package_return_path.chmod(package_return_path.stat().st_mode | 0o111)
        packet_manifest = {
            "schema_version": EXTERNAL_COHORT_CANDIDATE_PACKET_SCHEMA_VERSION,
            "root": _project_root_ref(packet_dir, project_root=project_root) if project_root is not None else None,
            "external_dir": (
                _project_relative_path_or_absolute(base_dir, project_root)
                if project_root is not None
                else str(base_dir.resolve())
            ),
            "provider": provider,
            "system_id": system_id,
            "target_system_cohort": command_bundle.get("target_system_cohort"),
            "required_env_vars": command_bundle.get("required_env_vars"),
            "normalize_commands": command_bundle.get("normalize_commands"),
            "correlate_command": command_bundle.get("correlate_command"),
            "refresh_command": command_bundle.get("refresh_command"),
            "amst_report_paths": command_bundle.get("amst_report_paths"),
            "packet_files": {
                "run_script_file": str(run_script_rel),
                "package_return_file": str(package_return_rel),
                "env_template_file": str(env_template_rel),
                "readme_file": str(readme_rel),
            },
        }
        write_json(manifest_path, packet_manifest)
        archive_path = packets_root / f"{provider}.zip"
        _write_deterministic_zip(output_zip=archive_path, source_dir=packet_dir, root_dir=packets_root)
        packet_path_ref = (
            _project_relative_path_or_absolute(packet_dir, project_root)
            if project_root is not None
            else str(packet_dir.resolve())
        )
        archive_path_ref = (
            _project_relative_path_or_absolute(archive_path, project_root)
            if project_root is not None
            else str(archive_path.resolve())
        )
        manifest_path_ref = (
            _project_relative_path_or_absolute(manifest_path, project_root)
            if project_root is not None
            else str(manifest_path.resolve())
        )
        run_script_path_ref = (
            _project_relative_path_or_absolute(run_script_path, project_root)
            if project_root is not None
            else str(run_script_path.resolve())
        )
        package_return_path_ref = (
            _project_relative_path_or_absolute(package_return_path, project_root)
            if project_root is not None
            else str(package_return_path.resolve())
        )
        env_template_path_ref = (
            _project_relative_path_or_absolute(env_template_path, project_root)
            if project_root is not None
            else str(env_template_path.resolve())
        )
        readme_path_ref = (
            _project_relative_path_or_absolute(readme_path, project_root)
            if project_root is not None
            else str(readme_path.resolve())
        )
        refs[provider] = {
            "provider": provider,
            "system_id": system_id,
            "packet_dir": _relative_or_absolute(packet_dir, base_dir),
            "packet_path": packet_path_ref,
            "archive_file": f"candidates/{provider}.zip",
            "archive_path": archive_path_ref,
            "packet_manifest_file": _relative_or_absolute(manifest_path, base_dir),
            "packet_manifest_path": manifest_path_ref,
            "run_script_file": _relative_or_absolute(run_script_path, base_dir),
            "run_script_path": run_script_path_ref,
            "package_return_file": _relative_or_absolute(package_return_path, base_dir),
            "package_return_path": package_return_path_ref,
            "env_template_file": _relative_or_absolute(env_template_path, base_dir),
            "env_template_path": env_template_path_ref,
            "readme_file": _relative_or_absolute(readme_path, base_dir),
            "readme_path": readme_path_ref,
        }
    return refs


def _render_external_candidate_run_script(*, provider: str, command_bundle: dict[str, Any], repo_root: Path | None) -> str:
    preamble = _external_candidate_script_preamble(repo_root=repo_root)
    lines = [preamble]
    required_env_vars = command_bundle.get("required_env_vars")
    if isinstance(required_env_vars, dict):
        for env_name, description in sorted(required_env_vars.items()):
            if not isinstance(env_name, str) or not env_name:
                continue
            comment = str(description) if description is not None else ""
            lines.append(f"# {env_name}: {comment}\n")
            lines.append(f': "${{{env_name}:?Set {env_name}}}"\n')
    normalize_commands = command_bundle.get("normalize_commands")
    target_system_cohort = (
        [str(system_id) for system_id in command_bundle.get("target_system_cohort", []) if str(system_id).strip()]
        if isinstance(command_bundle.get("target_system_cohort"), list)
        else []
    )
    provider_system_id = str(command_bundle.get("system_id") or "")
    merge_enabled = len(target_system_cohort) > 1 and bool(provider_system_id)
    if merge_enabled:
        lines.append(f'TMP_DIR="${{EXTERNAL_DIR}}/runtime/expand_with_{provider}_$(date -u +%Y%m%dT%H%M%SZ)"\n')
        lines.append('mkdir -p "$TMP_DIR"\n')
    if isinstance(normalize_commands, list):
        for command_row in normalize_commands:
            if not isinstance(command_row, dict):
                continue
            raw_command = command_row.get("command")
            if not isinstance(raw_command, str) or not raw_command.strip():
                continue
            benchmark_id = str(command_row.get("benchmark_id") or "")
            score_artifact = str(command_row.get("score_artifact") or "")
            rendered_command = _external_command_with_env_refs(raw_command, required_env_vars)
            if merge_enabled and benchmark_id and score_artifact:
                provider_score = f"$TMP_DIR/{benchmark_id}_{provider}_scores.json"
                existing_score = f"$TMP_DIR/{benchmark_id}_existing_scores.json"
                rendered_command = _external_replace_output_arg(rendered_command, score_artifact, provider_score)
                lines.append(f'cp {shlex.quote(score_artifact)} "{existing_score}"\n')
                lines.append(rendered_command)
                lines.append("\n")
                lines.append(
                    "PYTHONPATH=. python -m agent_memory_benchmark merge-external-scores "
                    f'--input "{existing_score}" "{provider_score}" '
                    f"--benchmark-id {shlex.quote(benchmark_id)} "
                    f"--system-cohort {' '.join(shlex.quote(item) for item in target_system_cohort)} "
                    f"--replace-system-id {shlex.quote(provider_system_id)} "
                    f'--source-manifest "$EXTERNAL_DIR/{benchmark_id}_scores_source_manifest.json" '
                    f"--output {shlex.quote(score_artifact)}"
                )
            else:
                lines.append(rendered_command)
            lines.append("\n")
    correlate_command = command_bundle.get("correlate_command")
    if isinstance(correlate_command, str) and correlate_command.strip():
        lines.append(correlate_command)
        lines.append("\n")
    refresh_command = command_bundle.get("refresh_command")
    if isinstance(refresh_command, str) and refresh_command.strip():
        lines.append(refresh_command)
        lines.append("\n")
    return "".join(lines)


def _render_external_candidate_package_return_script(
    *,
    provider: str,
    required_benchmark_ids: Any,
    repo_root: Path | None,
) -> str:
    preamble = _external_candidate_script_preamble(repo_root=repo_root)
    benchmark_ids = [str(item) for item in required_benchmark_ids] if isinstance(required_benchmark_ids, list) else []
    output_name = f"{provider}_return.zip"
    command = [
        "PYTHONPATH=. python -m agent_memory_benchmark build-external-cohort-return-packet",
        '--expansion "$EXTERNAL_DIR/cohort_expansion_plan.json"',
        f"--provider {provider}",
    ]
    for benchmark_id in benchmark_ids:
        command.append(f'--score {benchmark_id}="$EXTERNAL_DIR/{benchmark_id}_scores.json"')
    command.append('--output "${1:-$EXTERNAL_DIR/returns/inbox/' + output_name + '}"')
    return (
        preamble
        + '# Package refreshed same-cohort score artifacts into a return packet for canonical ingest.' + "\n"
        + " ".join(command)
        + "\n"
    )


def _render_external_candidate_env_template(*, provider: str, command_bundle: dict[str, Any]) -> str:
    lines = [
        f"# Environment template for external cohort expansion candidate: {provider}\n",
        "# Fill these paths before running ./run.sh\n",
        "\n",
    ]
    required_env_vars = command_bundle.get("required_env_vars")
    if isinstance(required_env_vars, dict):
        for env_name, description in sorted(required_env_vars.items()):
            lines.append(f"# {description}\n")
            lines.append(f'export {env_name}="/abs/path/to/{env_name.lower()}"\n\n')
    return "".join(lines)


def _render_external_candidate_readme(*, provider: str, system_id: str, command_bundle: dict[str, Any]) -> str:
    lines = [
        f"# External Candidate Packet: {provider}\n",
        "\n",
        f"- provider: `{provider}`\n",
        f"- system_id: `{system_id}`\n",
        f"- target cohort: `{command_bundle.get('target_system_cohort')}`\n",
        "\n",
        "Files in this packet:\n",
        "- `run.sh`: execute the normalization, correlation, and canonical refresh sequence\n",
        "- `package_return.sh`: package refreshed score artifacts into a return packet zip\n",
        "- `env.template`: required environment variables to fill before execution\n",
        "- `packet_manifest.json`: machine-readable packet contract\n",
        "\n",
        "Operator flow:\n",
        "1. Fill `env.template` values in your shell environment.\n",
        "2. Run `./run.sh` from this packet directory or from the repo root.\n",
        "3. Run `./package_return.sh` to create a provider return packet under `reports/external/returns/inbox/`.\n",
        "4. Check refreshed outputs under `reports/external/`.\n",
    ]
    return "".join(lines)


def _external_return_packet_readme(*, provider: str, system_id: str, required_benchmark_ids: list[str]) -> str:
    return (
        f"# External Return Packet: {provider}\n\n"
        f"- provider: `{provider}`\n"
        f"- system_id: `{system_id}`\n"
        f"- required benchmarks: `{required_benchmark_ids}`\n\n"
        "This packet contains normalized external same-cohort score artifacts ready for canonical ingest.\n"
    )


def _write_external_cohort_expansion_readme(path: Path, report: dict[str, Any]) -> None:
    recommended_candidate = (
        report.get("recommended_completion_candidate")
        if isinstance(report.get("recommended_completion_candidate"), dict)
        else None
    )
    lines = [
        "# External Cohort Expansion\n",
        "\n",
        f"Status: `{report.get('status')}`\n",
        "\n",
        "This directory contains the completion-grade external same-system cohort expansion plan.\n",
        "\n",
    ]
    completion_gate = report.get("completion_gate")
    if isinstance(completion_gate, dict):
        lines.extend(
            [
                "Completion gate:\n",
                f"- shared systems: `>= {completion_gate.get('min_shared_systems')}`\n",
                f"- shared control systems: `>= {completion_gate.get('min_shared_control_systems')}`\n",
                f"- shared real-memory systems: `>= {completion_gate.get('min_shared_real_memory_systems')}`\n",
                "\n",
            ]
        )
    if recommended_candidate is not None:
        lines.extend(
            [
                "Recommended candidate:\n",
                f"- provider: `{recommended_candidate.get('provider')}`\n",
                f"- system_id: `{recommended_candidate.get('system_id')}`\n",
                f"- target cohort: `{recommended_candidate.get('target_system_cohort')}`\n",
                f"- operator script: `{recommended_candidate.get('script')}`\n",
                f"- candidate packet dir: `{recommended_candidate.get('packet_dir')}`\n",
                f"- candidate packet archive: `{recommended_candidate.get('archive_file')}`\n",
                f"- candidate packet manifest: `{recommended_candidate.get('packet_manifest_file')}`\n",
                f"- candidate run script: `{recommended_candidate.get('run_script_file')}`\n",
                f"- candidate env template: `{recommended_candidate.get('env_template_file')}`\n",
                f"- candidate README: `{recommended_candidate.get('readme_file')}`\n",
                f"- return packaging script: `{recommended_candidate.get('package_return_file')}`\n",
                "\n",
            ]
        )
    lines.extend(
        [
            "Operator handoff:\n",
            "- `handoff_manifest.json`: machine-readable operator contract with the recommended next script, packet, README, and validation refs\n",
            "- `README.md`: human-readable walkthrough for the same contract\n",
            "- `returns/inbox/`: drop provider return packet zips here, then run `bin/sync_return_inbox.sh`\n",
            "- `rejected_returns_report.json`: persisted triage report for invalid provider return packets\n",
            "\n",
        ]
    )
    if recommended_candidate is not None:
        lines.extend(
            [
                "Packet workflow:\n",
                "- Use the root operator script above for the recommended provider expansion.\n",
                "- Fill candidate inputs from the packet-local `env.template`.\n",
                "- Inspect packet-local execution details in the candidate `README.md` and `packet_manifest.json`.\n",
                "- After collecting refreshed external scores, package the return with `package_return.sh` and drop the zip into `returns/inbox/`.\n",
            ]
        )
    path.write_text("".join(lines), encoding="utf-8")


def _external_handoff_manifest_summary(payload: dict[str, Any]) -> dict[str, Any]:
    recommended_candidate = (
        payload.get("recommended_completion_candidate")
        if isinstance(payload.get("recommended_completion_candidate"), dict)
        else None
    )
    return {
        "schema_version": payload.get("schema_version"),
        "external_dir": payload.get("external_dir"),
        "status": payload.get("status"),
        "expansion_plan_file": payload.get("expansion_plan_file"),
        "validation_file": payload.get("validation_file"),
        "readme_file": payload.get("readme_file"),
        "return_inbox": payload.get("return_inbox"),
        "return_archive": payload.get("return_archive"),
        "return_reject_archive": payload.get("return_reject_archive"),
        "return_inbox_state_file": payload.get("return_inbox_state_file"),
        "rejected_return_summary": payload.get("rejected_return_summary"),
        "rejected_returns_report_file": payload.get("rejected_returns_report_file"),
        "return_inbox_sync_report_file": payload.get("return_inbox_sync_report_file"),
        "return_inbox_watch_file": payload.get("return_inbox_watch_file"),
        "pending_return_packets": payload.get("pending_return_packets"),
        "watch_stop_exit_codes": payload.get("watch_stop_exit_codes"),
        "recommended_completion_candidate": {
            "provider": recommended_candidate.get("provider"),
            "system_id": recommended_candidate.get("system_id"),
            "script_id": recommended_candidate.get("script_id"),
            "script": recommended_candidate.get("script"),
            "archive_file": recommended_candidate.get("archive_file"),
            "packet_dir": recommended_candidate.get("packet_dir"),
            "package_return_file": recommended_candidate.get("package_return_file"),
        }
        if recommended_candidate is not None
        else None,
        "candidate_packet_ids": sorted(payload.get("candidate_packets", {}).keys())
        if isinstance(payload.get("candidate_packets"), dict)
        else [],
        "operator_script_ids": sorted(payload.get("operator_scripts", {}).keys())
        if isinstance(payload.get("operator_scripts"), dict)
        else [],
        "operator_command_ids": sorted(payload.get("operator_commands", {}).keys())
        if isinstance(payload.get("operator_commands"), dict)
        else [],
        "recommended_next_command_id": payload.get("recommended_next_command_id"),
        "recommended_next_script_id": payload.get("recommended_next_script_id"),
        "return_inbox_sync_status": payload.get("return_inbox_sync_status"),
        "return_inbox_sync_next_command_id": payload.get("return_inbox_sync_next_command_id"),
        "return_inbox_sync_next_command": payload.get("return_inbox_sync_next_command"),
        "return_inbox_sync_next_script_id": payload.get("return_inbox_sync_next_script_id"),
        "return_inbox_sync_next_script": payload.get("return_inbox_sync_next_script"),
        "return_inbox_sync_next_script_file": payload.get("return_inbox_sync_next_script_file"),
        "return_inbox_watch_status": payload.get("return_inbox_watch_status"),
        "return_inbox_watch_stop_reason": payload.get("return_inbox_watch_stop_reason"),
        "return_inbox_watch_next_command_id": payload.get("return_inbox_watch_next_command_id"),
        "return_inbox_watch_next_command": payload.get("return_inbox_watch_next_command"),
        "return_inbox_watch_next_script_id": payload.get("return_inbox_watch_next_script_id"),
        "return_inbox_watch_next_script": payload.get("return_inbox_watch_next_script"),
        "return_inbox_watch_next_script_file": payload.get("return_inbox_watch_next_script_file"),
    }


def _expected_external_handoff_manifest_summary(
    report: dict[str, Any],
    *,
    base_dir: Path,
    repo_root: Path | None,
) -> dict[str, Any]:
    handoff = _build_external_handoff_manifest(report, base_dir=base_dir, repo_root=repo_root)
    return _external_handoff_manifest_summary(handoff)


def _external_return_inbox_sync_report_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload.get("schema_version"),
        "expansion_path": payload.get("expansion_path"),
        "return_inbox": payload.get("return_inbox"),
        "return_archive": payload.get("return_archive"),
        "return_reject_archive": payload.get("return_reject_archive"),
        "rejected_returns_report_file": payload.get("rejected_returns_report_file"),
        "next_command_id": payload.get("next_command_id"),
        "next_command": payload.get("next_command"),
        "next_script_id": payload.get("next_script_id"),
        "next_script": payload.get("next_script"),
        "next_script_file": payload.get("next_script_file"),
    }


def _expected_external_return_inbox_sync_report_summary(report: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    next_action = _external_recommended_next_action(report)
    return {
        "schema_version": EXTERNAL_COHORT_RETURN_INBOX_SYNC_SCHEMA_VERSION,
        "expansion_path": report.get("expansion_plan_path") or str((base_dir / "cohort_expansion_plan.json").resolve()),
        "return_inbox": report.get("return_inbox"),
        "return_archive": report.get("return_archive"),
        "return_reject_archive": report.get("return_reject_archive"),
        "rejected_returns_report_file": report.get("rejected_returns_report_file"),
        "next_command_id": next_action.get("next_command_id"),
        "next_command": next_action.get("next_command"),
        "next_script_id": next_action.get("next_script_id"),
        "next_script": next_action.get("next_script"),
        "next_script_file": next_action.get("next_script_file"),
    }


def _external_recommended_next_action(
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "next_command_id": report.get("recommended_next_command_id"),
        "next_command": report.get("recommended_next_command"),
        "next_script_id": report.get("recommended_next_script_id"),
        "next_script": report.get("recommended_next_script"),
        "next_script_file": report.get("recommended_next_script_file"),
    }


def _external_return_inbox_watch_report_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload.get("schema_version"),
        "external_dir": payload.get("external_dir"),
        "expansion_path": payload.get("expansion_path"),
        "return_inbox_state_file": payload.get("return_inbox_state_file"),
        "return_inbox_sync_report_file": payload.get("return_inbox_sync_report_file"),
        "rejected_returns_report_file": payload.get("rejected_returns_report_file"),
        "next_command_id": payload.get("next_command_id"),
        "next_command": payload.get("next_command"),
        "next_script_id": payload.get("next_script_id"),
        "next_script": payload.get("next_script"),
        "next_script_file": payload.get("next_script_file"),
        "watch_stop_exit_codes": payload.get("watch_stop_exit_codes"),
    }


def _external_watch_next_action(
    report: dict[str, Any],
    *,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    next_action = _external_recommended_next_action(report)
    if isinstance(stop_reason, str) and stop_reason.strip():
        watch_stop_actions = report.get("watch_stop_actions")
        stop_action = watch_stop_actions.get(stop_reason) if isinstance(watch_stop_actions, dict) else None
        if isinstance(stop_action, dict):
            for key in ("next_command_id", "next_command", "next_script_id", "next_script", "next_script_file"):
                value = stop_action.get(key)
                if isinstance(value, str) and value.strip():
                    next_action[key] = value
    return next_action


def _expected_external_return_inbox_watch_report_summary(
    report: dict[str, Any],
    payload: dict[str, Any],
    *,
    base_dir: Path,
) -> dict[str, Any]:
    project_root_raw = report.get("root")
    project_root = _resolve_reported_project_root(project_root_raw, base_dir=base_dir)
    next_action = _external_watch_next_action(
        report,
        stop_reason=payload.get("stop_reason") if isinstance(payload.get("stop_reason"), str) else None,
    )
    return {
        "schema_version": EXTERNAL_COHORT_RETURN_INBOX_WATCH_SCHEMA_VERSION,
        "external_dir": _external_dir_ref(base_dir, project_root=project_root),
        "expansion_path": report.get("expansion_plan_path") or str((base_dir / "cohort_expansion_plan.json").resolve()),
        "return_inbox_state_file": report.get("return_inbox_state_file"),
        "return_inbox_sync_report_file": report.get("return_inbox_sync_report_file"),
        "rejected_returns_report_file": report.get("rejected_returns_report_file"),
        "next_command_id": next_action.get("next_command_id"),
        "next_command": next_action.get("next_command"),
        "next_script_id": next_action.get("next_script_id"),
        "next_script": next_action.get("next_script"),
        "next_script_file": next_action.get("next_script_file"),
        "watch_stop_exit_codes": report.get("watch_stop_exit_codes"),
    }


def _external_return_inbox_state_path(external_dir: Path) -> Path:
    return external_dir / "return_inbox_state.json"


def _default_external_return_inbox_state(external_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": EXTERNAL_COHORT_RETURN_INBOX_STATE_SCHEMA_VERSION,
        "root": _external_root_ref(external_dir),
        "external_dir": _external_dir_ref(external_dir),
        "processed_packets": [],
        "rejected_packets": [],
    }


def _normalize_external_return_inbox_state_entry(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    packet_fingerprint = value.get("packet_fingerprint")
    packet_file = value.get("packet_file")
    packet_path = value.get("packet_path")
    if not isinstance(packet_fingerprint, str) or not packet_fingerprint.strip():
        return None
    if not isinstance(packet_file, str) or not packet_file.strip():
        return None
    if not isinstance(packet_path, str) or not packet_path.strip():
        return None
    normalized = {
        "packet_fingerprint": packet_fingerprint,
        "packet_file": packet_file,
        "packet_path": packet_path,
    }
    for key in (
        "provider",
        "system_id",
        "processed_archive_file",
        "processed_archive_path",
        "rejected_archive_file",
        "rejected_archive_path",
        "rejection_error",
    ):
        raw_value = value.get(key)
        if isinstance(raw_value, str) and raw_value.strip():
            normalized[key] = raw_value
    return normalized


def _load_external_return_inbox_state(external_dir: Path) -> dict[str, Any]:
    state_path = _external_return_inbox_state_path(external_dir)
    if not state_path.exists():
        return _default_external_return_inbox_state(external_dir)
    state = read_json(state_path)
    if not isinstance(state, dict):
        raise ValueError("external return_inbox_state.json must be a JSON object")
    if state.get("schema_version") != EXTERNAL_COHORT_RETURN_INBOX_STATE_SCHEMA_VERSION:
        raise ValueError(
            f"external return_inbox_state.json schema_version must be {EXTERNAL_COHORT_RETURN_INBOX_STATE_SCHEMA_VERSION}"
        )
    normalized = _default_external_return_inbox_state(external_dir)
    for field in ("processed_packets", "rejected_packets"):
        raw_entries = state.get(field)
        entries: list[dict[str, Any]] = []
        if isinstance(raw_entries, list):
            for entry in raw_entries:
                normalized_entry = _normalize_external_return_inbox_state_entry(entry)
                if normalized_entry is not None:
                    entries.append(normalized_entry)
        normalized[field] = entries
    return normalized


def _write_external_return_inbox_state(external_dir: Path, state: dict[str, Any]) -> Path:
    payload = _default_external_return_inbox_state(external_dir)
    for field in ("processed_packets", "rejected_packets"):
        payload[field] = []
        raw_entries = state.get(field)
        if isinstance(raw_entries, list):
            for entry in raw_entries:
                normalized_entry = _normalize_external_return_inbox_state_entry(entry)
                if normalized_entry is not None:
                    payload[field].append(normalized_entry)
    state_path = _external_return_inbox_state_path(external_dir)
    write_json(state_path, payload)
    return state_path


def _external_processed_packet_entry(processed_state: dict[str, Any], packet_fingerprint: str) -> dict[str, Any] | None:
    raw_entries = processed_state.get("processed_packets")
    if not isinstance(raw_entries, list):
        return None
    for entry in raw_entries:
        if isinstance(entry, dict) and entry.get("packet_fingerprint") == packet_fingerprint:
            return entry
    return None


def _external_rejected_packet_entry(processed_state: dict[str, Any], packet_fingerprint: str) -> dict[str, Any] | None:
    raw_entries = processed_state.get("rejected_packets")
    if not isinstance(raw_entries, list):
        return None
    for entry in raw_entries:
        if isinstance(entry, dict) and entry.get("packet_fingerprint") == packet_fingerprint:
            return entry
    return None


def _record_processed_external_return_packet(
    processed_state: dict[str, Any],
    *,
    packet_fingerprint: str,
    archive_file: str,
    archive_path: str,
    provider: str,
    system_id: str,
) -> None:
    entries = [
        entry
        for entry in processed_state.get("processed_packets", [])
        if isinstance(entry, dict) and entry.get("packet_fingerprint") != packet_fingerprint
    ]
    entries.append(
        {
            "packet_fingerprint": packet_fingerprint,
            "packet_file": archive_file,
            "packet_path": archive_path,
            "processed_archive_file": archive_file,
            "processed_archive_path": archive_path,
            "provider": provider,
            "system_id": system_id,
        }
    )
    processed_state["processed_packets"] = sorted(entries, key=lambda item: str(item.get("packet_fingerprint") or ""))


def _record_rejected_external_return_packet(
    processed_state: dict[str, Any],
    *,
    packet_fingerprint: str,
    archive_file: str,
    archive_path: str,
    rejection_error: str,
) -> None:
    entries = [
        entry
        for entry in processed_state.get("rejected_packets", [])
        if isinstance(entry, dict) and entry.get("packet_fingerprint") != packet_fingerprint
    ]
    entries.append(
        {
            "packet_fingerprint": packet_fingerprint,
            "packet_file": archive_file,
            "packet_path": archive_path,
            "rejected_archive_file": archive_file,
            "rejected_archive_path": archive_path,
            "rejection_error": rejection_error,
        }
    )
    processed_state["rejected_packets"] = sorted(entries, key=lambda item: str(item.get("packet_fingerprint") or ""))


def _external_watch_stop_actions(
    external_dir: Path,
    operator_scripts: dict[str, Any],
    operator_script_files: dict[str, Any],
    *,
    repo_root: Path | None,
) -> dict[str, dict[str, Any]]:
    commands = _external_operator_commands(
        external_dir,
        {"operator_script_files": operator_script_files},
        repo_root=repo_root,
    )
    def _script_refs(script_id: str) -> tuple[str | None, str | None]:
        raw_script = operator_scripts.get(script_id)
        raw_script_file = operator_script_files.get(script_id)
        script = str(raw_script) if isinstance(raw_script, str) and raw_script.strip() else None
        script_file = str(raw_script_file) if isinstance(raw_script_file, str) and raw_script_file.strip() else None
        return script, script_file

    watch_script, watch_script_file = _script_refs("watch_return_inbox")
    review_script, review_script_file = _script_refs("review_rejected_returns")
    refresh_script, refresh_script_file = _script_refs("refresh_canonical")
    return {
        "max_iterations": {
            "exit_code": EXTERNAL_COHORT_WATCH_STOP_EXIT_CODES["max_iterations"],
            "kind": "continue_waiting",
            "next_command_id": "watch_return_inbox",
            "next_command": commands.get("watch_return_inbox"),
            "next_script_id": "watch_return_inbox",
            "next_script": watch_script,
            "next_script_file": watch_script_file,
        },
        "rejected_returns": {
            "exit_code": EXTERNAL_COHORT_WATCH_STOP_EXIT_CODES["rejected_returns"],
            "kind": "triage_rejected_returns",
            "next_command_id": "review_rejected_returns",
            "next_command": commands.get("review_rejected_returns"),
            "next_script_id": "review_rejected_returns",
            "next_script": review_script,
            "next_script_file": review_script_file,
        },
        "ready": {
            "exit_code": EXTERNAL_COHORT_WATCH_STOP_EXIT_CODES["ready"],
            "kind": "completion_gate_satisfied",
            "next_command_id": "refresh_canonical",
            "next_command": commands.get("refresh_canonical"),
            "next_script_id": "refresh_canonical",
            "next_script": refresh_script,
            "next_script_file": refresh_script_file,
        },
    }


def _external_watch_stop_exit_code(stop_reason: str | None) -> int:
    if not isinstance(stop_reason, str) or not stop_reason.strip():
        return 0
    return int(EXTERNAL_COHORT_WATCH_STOP_EXIT_CODES.get(stop_reason, 0) or 0)


def _ensure_external_return_dirs(base_dir: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    inbox_dir = base_dir / "returns" / "inbox"
    processed_dir = base_dir / "returns" / "processed"
    rejected_dir = base_dir / "returns" / "rejected"
    for directory in (inbox_dir, processed_dir, rejected_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return (
        {"candidate_inbox": _relative_or_absolute(inbox_dir, base_dir)},
        {"candidate_archive": _relative_or_absolute(processed_dir, base_dir)},
        {"candidate_archive": _relative_or_absolute(rejected_dir, base_dir)},
    )


def _external_candidate_from_expansion(expansion: dict[str, Any], provider: str) -> dict[str, Any]:
    candidates = expansion.get("available_real_memory_candidates")
    if isinstance(candidates, list):
        for item in candidates:
            if isinstance(item, dict) and item.get("provider") == provider:
                return item
    raise ValueError(f"provider {provider!r} is not present in the external cohort expansion plan")


def _external_candidate_packet(expansion: dict[str, Any], provider: str) -> dict[str, Any]:
    candidate_packets = expansion.get("candidate_packets")
    if isinstance(candidate_packets, dict):
        packet = candidate_packets.get(provider)
        if isinstance(packet, dict):
            return packet
    raise ValueError(f"provider {provider!r} does not have a candidate packet in the external cohort expansion plan")


def _external_candidate_packet_manifest_path(
    expansion: dict[str, Any],
    candidate_packet: dict[str, Any],
    *,
    provider: str,
    external_dir: Path,
) -> Path:
    raw_path = candidate_packet.get("packet_manifest_path")
    if isinstance(raw_path, str) and raw_path:
        return _resolve_external_contract_path(external_dir, raw_path).resolve()
    raw_file = candidate_packet.get("packet_manifest_file")
    if isinstance(raw_file, str) and raw_file:
        return (external_dir / raw_file).resolve()
    raise ValueError(f"provider {provider!r} candidate packet is missing packet_manifest_file/path")


def _external_materialized_candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "provider",
        "system_id",
        "script_id",
        "script",
        "script_file",
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
        "target_system_cohort",
        "covered_benchmarks",
        "missing_benchmarks",
        "num_missing_benchmarks",
        "target_shared_system_count",
        "target_shared_control_system_count",
        "target_shared_real_memory_system_count",
        "would_satisfy_completion_gate",
        "real_system_report_path",
    ):
        if key in candidate:
            summary[key] = candidate[key]
    return summary


def _external_candidate_contract_bundle(
    expansion: dict[str, Any],
    provider: str,
    *,
    external_dir: Path,
) -> dict[str, Any] | None:
    candidate_packets = expansion.get("candidate_packets")
    packet = candidate_packets.get(provider) if isinstance(candidate_packets, dict) else None
    if isinstance(packet, dict):
        try:
            manifest_path = _external_candidate_packet_manifest_path(
                expansion,
                packet,
                provider=provider,
                external_dir=external_dir,
            )
        except ValueError:
            manifest_path = None
        if manifest_path is not None and manifest_path.exists():
            try:
                payload = read_json(manifest_path)
            except Exception:  # noqa: BLE001 - callers should degrade on malformed packet manifests
                payload = None
            if isinstance(payload, dict):
                return {
                    key: payload.get(key)
                    for key in (
                        "amst_report_paths",
                        "normalize_commands",
                        "correlate_command",
                        "refresh_command",
                        "required_env_vars",
                        "target_system_cohort",
                    )
                }
    recommended_candidate = (
        expansion.get("recommended_completion_candidate")
        if isinstance(expansion.get("recommended_completion_candidate"), dict)
        else None
    )
    if isinstance(recommended_candidate, dict) and recommended_candidate.get("provider") == provider:
        command_bundle = recommended_candidate.get("command_bundle")
        if isinstance(command_bundle, dict):
            return command_bundle
    for key in ("available_real_memory_candidates", "minimum_completion_candidates"):
        candidates = expansion.get(key)
        if not isinstance(candidates, list):
            continue
        for item in candidates:
            if not isinstance(item, dict) or item.get("provider") != provider:
                continue
            command_bundle = item.get("command_bundle")
            if isinstance(command_bundle, dict):
                return command_bundle
    return None


def _validate_external_return_score_artifact(
    path: Path,
    *,
    benchmark_id: str,
    target_system_cohort: Any,
    required_system_id: str,
) -> dict[str, Any]:
    errors = validate_normalized_external_score(path, expected_benchmark_id=benchmark_id)
    if errors:
        raise ValueError(f"{path} is not a valid normalized external score artifact: {errors[0]}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object")
    run_config = payload.get("run_config")
    system_cohort = run_config.get("system_cohort") if isinstance(run_config, dict) else None
    if not isinstance(system_cohort, list):
        raise ValueError(f"{path} run_config.system_cohort must be a list")
    actual_cohort = sorted(str(item) for item in system_cohort)
    expected_cohort = (
        sorted(str(item) for item in target_system_cohort)
        if isinstance(target_system_cohort, list)
        else []
    )
    if actual_cohort != expected_cohort:
        raise ValueError(f"{path} system cohort does not match expansion target cohort")
    systems = payload.get("systems")
    if not isinstance(systems, list) or required_system_id not in {
        str(item.get("system_id")) for item in systems if isinstance(item, dict) and item.get("system_id")
    }:
        raise ValueError(f"{path} does not contain required system_id {required_system_id!r}")
    return payload


def _write_zip_from_dir(source_dir: Path, output_zip: Path) -> None:
    _write_deterministic_zip(output_zip=output_zip, source_dir=source_dir, root_dir=source_dir)


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


def _resolve_project_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def _move_to_unique_destination(source: Path, destination: Path) -> Path:
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        return Path(shutil.move(str(source), str(destination)))
    stem = destination.stem
    suffix = destination.suffix
    index = 1
    while True:
        candidate = destination.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return Path(shutil.move(str(source), str(candidate)))
        index += 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _external_command_with_env_refs(command: str, required_env_vars: Any) -> str:
    rendered = command
    if isinstance(required_env_vars, dict):
        for env_name in sorted((str(key) for key in required_env_vars.keys()), key=len, reverse=True):
            rendered = re.sub(rf"(?<![A-Z0-9_]){re.escape(env_name)}(?![A-Z0-9_])", f'"${env_name}"', rendered)
    return rendered


def _external_replace_output_arg(command: str, old_output: str, new_output: str) -> str:
    pattern = re.compile(r"(?P<prefix>--output\s+)(?P<quote>['\"]?)(?P<path>\S+?)(?P=quote)(?=\s|$)")

    def replace(match: re.Match[str]) -> str:
        raw_path = match.group("path")
        if raw_path != old_output:
            return match.group(0)
        return f'{match.group("prefix")}"{new_output}"'

    updated = pattern.sub(replace, command, count=1)
    if updated == command:
        raise ValueError(f"cannot redirect command output for {old_output!r}: {command}")
    return updated


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _project_relative_path_or_absolute(path: Path, project_root: Path) -> str:
    return _rel(path.resolve(), project_root)


def _external_contract_project_root(base_dir: Path) -> Path | None:
    resolved_base_dir = base_dir.resolve()
    for candidate in (resolved_base_dir, *resolved_base_dir.parents):
        if candidate.name != "external":
            continue
        if candidate.parent.name != "reports":
            continue
        return candidate.parent.parent
    return None


def _external_dir_ref(external_dir: Path, *, project_root: Path | None = None) -> str:
    resolved_external_dir = external_dir.resolve()
    resolved_project_root = project_root.resolve() if project_root is not None else _external_contract_project_root(resolved_external_dir)
    if resolved_project_root is not None:
        return _project_relative_path_or_absolute(resolved_external_dir, resolved_project_root)
    return str(resolved_external_dir)


def _external_root_ref(external_dir: Path, *, project_root: Path | None = None) -> str | None:
    resolved_external_dir = external_dir.resolve()
    resolved_project_root = (
        project_root.resolve()
        if project_root is not None
        else _external_contract_project_root(resolved_external_dir)
    )
    if resolved_project_root is None:
        return None
    return _project_root_ref(resolved_external_dir, project_root=resolved_project_root)


def _project_root_ref(base_dir: Path, *, project_root: Path | None) -> str | None:
    if project_root is None:
        return None
    resolved_base_dir = base_dir.resolve()
    resolved_project_root = project_root.resolve()
    try:
        return Path(os.path.relpath(resolved_project_root, resolved_base_dir)).as_posix()
    except ValueError:
        return str(resolved_project_root)


def _resolve_reported_project_root(raw_root: str | None, *, base_dir: Path) -> Path | None:
    if not isinstance(raw_root, str) or not raw_root.strip():
        return None
    root_path = Path(raw_root)
    if root_path.is_absolute():
        return root_path.resolve()
    return (base_dir.resolve() / root_path).resolve()


def _normalize_project_relative_path_string(
    raw_path: str,
    *,
    project_root: Path,
    base_dir: Path,
) -> str:
    path = Path(raw_path)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend((project_root / path, base_dir / path, Path.cwd() / path))
    for candidate in candidates:
        if candidate.exists():
            return _project_relative_path_or_absolute(candidate.resolve(), project_root)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return _project_relative_path_or_absolute(path.resolve(), project_root)
    except OSError:
        return raw_path


def _resolve_external_contract_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    direct = base_dir / path
    if direct.exists():
        return direct
    project_root = _external_contract_project_root(base_dir)
    if project_root is not None:
        project_relative = project_root / path
        if project_relative.exists() or (path.parts and path.parts[0] == "reports"):
            return project_relative
    return direct


def _system_ids_from_common_rows(rows: list[Any], errors: list[str], *, prefix: str) -> set[str]:
    system_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            errors.append(f"{prefix}[{index}] must be an object")
            continue
        system_id = row.get("system_id")
        if not system_id:
            errors.append(f"{prefix}[{index}].system_id is required")
        elif str(system_id) in system_ids:
            errors.append(f"{prefix}[{index}].system_id is duplicated")
        else:
            system_ids.add(str(system_id))
        if numeric_or_none(row.get("amst_score")) is None:
            errors.append(f"{prefix}[{index}].amst_score must be numeric")
        if numeric_or_none(row.get("external_score")) is None:
            errors.append(f"{prefix}[{index}].external_score must be numeric")
    return system_ids


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)
