"""Evidence checks for real external memory-system AutoMemoryBench runs."""

from __future__ import annotations

from datetime import datetime, UTC
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from amb.benchmark.artifact_contract import localize_report_contract
from amb.benchmark.evaluation.framework_trace import (
    FRAMEWORK_TRACE_ARTIFACT_SCHEMA_VERSION,
    load_default_tool_runtime_contracts,
    validate_framework_trace_artifact_payload,
)
from amb.benchmark.evaluation.scoring import DEFAULT_RETRIEVAL_K
from amb.benchmark.integrations.config_validation import validate_integration_config_files
from amb.benchmark.integrations.factory import canonical_provider
from amb.benchmark.release.artifacts import artifact_info
from amb.benchmark.quality.run_metadata import (
    build_real_system_attestation,
    build_run_metadata,
    load_run_metadata,
    validate_real_system_run_metadata,
    validate_run_metadata,
)
from amb.benchmark.schemas.io import load_predictions, read_json, write_json

REAL_SYSTEM_EVIDENCE_SCHEMA_VERSION = "amst-real-system-evidence-v1"
REQUIRED_REAL_SYSTEM_PROVIDERS = ("mem0", "letta", "langmem", "zep_graphiti")
REAL_SYSTEM_ANALYSIS_SCHEMA_VERSION = "amst-real-system-analysis-v1"


def default_real_system_analysis_output(output: str | Path) -> Path:
    output_path = Path(output)
    if output_path.suffix == ".json":
        return output_path.with_name(f"{output_path.stem}_analysis.json")
    return output_path.with_name(f"{output_path.name}_analysis.json")


def _artifact_root_ref(base_dir: Path, project_root: Path) -> str:
    resolved_base_dir = base_dir.resolve()
    resolved_project_root = project_root.resolve()
    try:
        return Path(os.path.relpath(resolved_project_root, resolved_base_dir)).as_posix()
    except ValueError:
        return str(resolved_project_root)


def _contract_relative_or_absolute(contract_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(contract_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _infer_contract_root(paths: Iterable[str | Path]) -> Path:
    resolved: list[str] = []
    for raw_path in paths:
        if raw_path is None:
            continue
        path = Path(raw_path)
        resolved.append(str(path.resolve()))
    if not resolved:
        return Path(".").resolve()
    return Path(os.path.commonpath(resolved))


def _resolve_contract_path(raw_path: str | Path, *, contract_root: Path | None = None, anchor: Path | None = None) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if contract_root is not None:
        candidate = contract_root / path
        if candidate.exists():
            return candidate
    if anchor is not None:
        candidate = anchor / path
        if candidate.exists():
            return candidate
    if path.exists():
        return path
    return (contract_root / path) if contract_root is not None else (anchor / path if anchor is not None else path)


def ordered_real_system_matrix_summary_candidates(root: str | Path) -> tuple[Path, ...]:
    """Return existing canonical summary candidates ordered by preference."""

    project = Path(root)
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        if not path.exists():
            return
        key = str(path.resolve())
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    add(project / "reports/real_system_runs/canonical_public_dev_refresh_current_matrix_summary.json")
    add(project / "reports/real_system_runs/matrix_summary.json")
    add(project / "reports/examples/amst_main_v1_real_system_matrix_summary.json")

    refresh_candidates = sorted(
        (project / "reports/real_system_runs").glob("canonical_public_dev_refresh*_matrix_summary.json"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    for path in refresh_candidates:
        add(path)
    return tuple(candidates)


def ordered_real_system_analysis_candidates(
    root: str | Path,
    summary_path: str | Path | None = None,
) -> tuple[Path, ...]:
    """Return preferred real-system analysis candidates ordered by preference."""

    project = Path(root)
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path, *, allow_missing: bool = False) -> None:
        if not allow_missing and not path.exists():
            return
        key = str(path.resolve())
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    if summary_path is not None:
        summary = Path(summary_path)
        name = summary.name
        if name.endswith("_matrix_summary.json"):
            add(summary.with_name(name.replace("_matrix_summary.json", "_analysis.json")), allow_missing=True)
        if name.endswith("matrix_summary.json"):
            add(summary.with_name(name.replace("matrix_summary.json", "analysis.json")), allow_missing=True)

    add(project / "reports/real_system_runs/canonical_public_dev_refresh_current_analysis.json", allow_missing=True)
    add(project / "reports/real_system_runs/current_refresh_analysis.json", allow_missing=True)
    for path in sorted((project / "reports/real_system_runs").glob("*analysis.json")):
        add(path)
    return tuple(candidates)


def write_real_system_analysis(
    summary_path: str | Path,
    output: str | Path,
    *,
    expected_benchmark_id: str | None = None,
    expected_release_split: str | None = None,
    bootstrap_samples: int = 200,
    seed: int = 13,
) -> dict[str, Any]:
    from amb.benchmark.analysis import analyze_report_files

    summary_file = Path(summary_path)
    output_path = Path(output)
    report_paths = _real_system_report_paths_from_summary(
        summary_file,
        expected_benchmark_id=expected_benchmark_id,
        expected_release_split=expected_release_split,
    )
    analysis = analyze_report_files(
        [str(path) for path in report_paths],
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )
    analysis = localize_report_contract(
        analysis,
        output_path=output_path,
        project_root_hints=(summary_file, *report_paths),
    )
    analysis["schema_version"] = REAL_SYSTEM_ANALYSIS_SCHEMA_VERSION
    analysis["matrix_summary"] = _contract_relative_or_absolute(
        _infer_contract_root((output_path, summary_file)),
        summary_file,
    )
    write_json(output_path, analysis)
    return analysis


def summarize_real_system_analysis(
    analysis: dict[str, Any],
    *,
    min_bootstrap_samples: int = 200,
) -> dict[str, Any]:
    metrics = tuple(str(item) for item in analysis.get("metrics", ()) if isinstance(item, str))
    reports = [item for item in analysis.get("reports", ()) if isinstance(item, dict)]
    comparisons = [item for item in analysis.get("comparisons", ()) if isinstance(item, dict)]
    report_checks: dict[str, dict[str, Any]] = {}
    num_report_cis_present = 0
    for item in reports:
        system_id = str(item.get("system_id", "unknown"))
        ci_map = item.get("bootstrap_ci", {}) if isinstance(item.get("bootstrap_ci"), dict) else {}
        metric_checks: dict[str, Any] = {}
        for metric in metrics:
            ci = ci_map.get(metric, {}) if isinstance(ci_map, dict) else {}
            present = _real_system_ci_present(ci)
            if present:
                num_report_cis_present += 1
            metric_checks[metric] = {"present": present, "ci": ci}
        report_checks[system_id] = metric_checks

    pairwise_checks: dict[str, dict[str, Any]] = {}
    num_pairwise_metric_stats_present = 0
    for item in comparisons:
        baseline_system_id = str(item.get("baseline_system_id"))
        candidate_system_id = str(item.get("candidate_system_id"))
        metrics_map = item.get("metrics", {}) if isinstance(item.get("metrics"), dict) else {}
        key = f"{baseline_system_id}->{candidate_system_id}"
        metric_checks: dict[str, Any] = {}
        for metric in metrics:
            stats = metrics_map.get(metric, {}) if isinstance(metrics_map, dict) else {}
            ci = stats.get("bootstrap_ci", {}) if isinstance(stats, dict) else {}
            present = (
                isinstance(stats, dict)
                and stats.get("mean_difference") is not None
                and stats.get("p_value") is not None
                and _real_system_ci_present(ci)
                and _positive_int(stats.get("num_pairs"))
            )
            if present:
                num_pairwise_metric_stats_present += 1
            metric_checks[metric] = {"present": present, "stats": stats}
        pairwise_checks[key] = metric_checks

    frontier = analysis.get("quality_cost_frontier", {})
    points = frontier.get("points", ()) if isinstance(frontier, dict) else ()
    frontier_points_complete = (
        isinstance(points, list)
        and len(points) == len(reports)
        and all(
            isinstance(point, dict)
            and point.get("quality") is not None
            and point.get("cost_proxy") is not None
            for point in points
        )
    )
    frontier_nonempty = bool(frontier.get("frontier")) if isinstance(frontier, dict) else False
    weight_sensitivity = analysis.get("weight_sensitivity", {})
    expected_weight_profiles = (
        len(weight_sensitivity.get("default_weights", {})) + 1
        if isinstance(weight_sensitivity, dict) and weight_sensitivity.get("default_weights")
        else 0
    )
    weight_profiles_complete = (
        isinstance(weight_sensitivity, dict)
        and weight_sensitivity.get("present") is True
        and weight_sensitivity.get("num_complete_reports") == len(reports)
        and len(weight_sensitivity.get("profiles", ())) == expected_weight_profiles
        and len(weight_sensitivity.get("rank_stability", ())) == len(reports)
    )
    required_report_metric_count = len(reports) * len(metrics)
    required_pair_metric_count = (len(reports) * (len(reports) - 1) // 2) * len(metrics)
    return {
        "analysis_schema_version": analysis.get("analysis_schema_version"),
        "schema_version": analysis.get("schema_version"),
        "bootstrap_samples": _as_int(analysis.get("bootstrap_samples")),
        "ci_level": analysis.get("ci_level"),
        "num_reports": len(reports),
        "metrics": list(metrics),
        "bootstrap_samples_sufficient": _as_int(analysis.get("bootstrap_samples")) >= min_bootstrap_samples,
        "report_bootstrap_cis_present": num_report_cis_present == required_report_metric_count,
        "pairwise_stats_complete": num_pairwise_metric_stats_present == required_pair_metric_count,
        "quality_cost_frontier_complete": frontier_points_complete and frontier_nonempty,
        "weight_sensitivity_profiles_complete": weight_profiles_complete,
        "required_report_metric_count": required_report_metric_count,
        "num_report_cis_present": num_report_cis_present,
        "required_pair_metric_count": required_pair_metric_count,
        "num_pairwise_metric_stats_present": num_pairwise_metric_stats_present,
        "required_weight_profile_count": expected_weight_profiles,
        "num_weight_profiles": len(weight_sensitivity.get("profiles", ())) if isinstance(weight_sensitivity, dict) else 0,
        "report_checks": report_checks,
        "pairwise_checks": pairwise_checks,
    }


def validate_real_system_report(path: str | Path) -> tuple[str, ...]:
    """Validate one evaluated real-system AMST report."""

    report_path = Path(path)
    try:
        report = read_json(report_path)
    except Exception as exc:
        return (f"{report_path}: cannot read report: {exc}",)
    if not isinstance(report, dict):
        return (f"{report_path}: report must be a JSON object",)

    errors: list[str] = []
    aggregate = report.get("aggregate", {})
    if not isinstance(aggregate, dict) or aggregate.get("num_scored_queries", 0) <= 0:
        errors.append(f"{report_path}: aggregate.num_scored_queries must be positive")
    if report.get("missing_predictions"):
        errors.append(f"{report_path}: missing_predictions must be empty")
    if report.get("extra_predictions"):
        errors.append(f"{report_path}: extra_predictions must be empty")
    submission = report.get("submission", {})
    if not isinstance(submission, dict) or not submission.get("prediction_artifact"):
        errors.append(f"{report_path}: submission.prediction_artifact is required")
    if not isinstance(submission, dict) or not submission.get("run_metadata_artifact"):
        errors.append(f"{report_path}: submission.run_metadata_artifact is required")
    if isinstance(submission, dict) and submission.get("framework_trace_artifact") is not None:
        trace_artifact = submission.get("framework_trace_artifact")
        if not isinstance(trace_artifact, dict) or not trace_artifact.get("path"):
            errors.append(f"{report_path}: submission.framework_trace_artifact.path is required")
        else:
            trace_path = _resolve_contract_path(str(trace_artifact["path"]), anchor=report_path.parent)
            errors.extend(_validate_framework_trace_artifact(trace_path))
    run_metadata = report.get("run_metadata")
    if not isinstance(run_metadata, dict):
        errors.append(f"{report_path}: run_metadata is required")
    else:
        errors.extend(
            f"{report_path}: {error}"
            for error in validate_run_metadata(
                run_metadata,
                system_id=str(report.get("system_id")),
                benchmark_id=str(report.get("benchmark_id")),
                release_split=str(report.get("release_split")),
            )
        )
        errors.extend(f"{report_path}: {error}" for error in validate_real_system_run_metadata(run_metadata))
    for field in ("efficiency.input_tokens", "efficiency.output_tokens", "efficiency.latency_ms"):
        if not isinstance(aggregate, dict) or aggregate.get(field) is None:
            errors.append(f"{report_path}: aggregate.{field} is required")
    return tuple(errors)


def real_system_report_provider(path: str | Path) -> str | None:
    """Return provider attested by a real-system report, if present."""

    try:
        report = read_json(path)
    except Exception:
        return None
    if not isinstance(report, dict):
        return None
    metadata = report.get("run_metadata")
    if not isinstance(metadata, dict):
        return None
    attestation = metadata.get("real_system_attestation")
    if not isinstance(attestation, dict) or not attestation.get("provider"):
        return None
    return str(attestation["provider"])


def validate_real_system_matrix_summary(
    summary_path: str | Path,
    *,
    required_providers: Iterable[str] = REQUIRED_REAL_SYSTEM_PROVIDERS,
    expected_benchmark_id: str | None = None,
    expected_release_split: str | None = None,
) -> dict[str, Any]:
    """Validate a run-release-agent-matrix summary and referenced real reports."""

    source = Path(summary_path)
    errors: list[str] = []
    try:
        summary = read_json(source)
    except Exception as exc:
        return _matrix_report(
            source,
            (),
            {},
            (f"{source}: cannot read matrix summary: {exc}",),
            expected_benchmark_id=expected_benchmark_id,
            expected_release_split=expected_release_split,
        )
    if not isinstance(summary, dict):
        return _matrix_report(
            source,
            (),
            {},
            (f"{source}: matrix summary must be a JSON object",),
            expected_benchmark_id=expected_benchmark_id,
            expected_release_split=expected_release_split,
        )
    if summary.get("schema_version") != "amst-release-agent-matrix-v1":
        errors.append(f"{source}: schema_version must be amst-release-agent-matrix-v1")
    contract_root = _contract_root_from_summary(source, summary)
    if expected_benchmark_id is not None and summary.get("benchmark_id") != expected_benchmark_id:
        errors.append(f"{source}: benchmark_id must be {expected_benchmark_id}")
    if expected_release_split is not None and summary.get("release_split") != expected_release_split:
        errors.append(f"{source}: release_split must be {expected_release_split}")
    systems = summary.get("systems")
    if not isinstance(systems, list):
        errors.append(f"{source}: systems must be a list")
        return _matrix_report(
            source,
            (),
            {},
            tuple(errors),
            expected_benchmark_id=expected_benchmark_id,
            expected_release_split=expected_release_split,
        )

    provider_reports: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []
    num_framework_trace_artifacts = 0
    num_framework_trace_envelope_artifacts = 0
    for index, item in enumerate(systems, start=1):
        if not isinstance(item, dict):
            errors.append(f"{source}: systems[{index}] must be an object")
            continue
        system_id = str(item.get("system_id", f"system_{index}"))
        artifact = item.get("report_artifact")
        if not isinstance(artifact, dict) or not artifact.get("path"):
            errors.append(f"{source}: systems[{index}] report_artifact.path is required")
            continue
        report_path = _resolve_report_path(source, str(artifact["path"]), contract_root=contract_root)
        report_errors = list(validate_real_system_report(report_path))
        report = _read_report_object(report_path)
        if expected_benchmark_id is not None and report.get("benchmark_id") != expected_benchmark_id:
            report_errors.append(f"{report_path}: benchmark_id must be {expected_benchmark_id}")
        if expected_release_split is not None and report.get("release_split") != expected_release_split:
            report_errors.append(f"{report_path}: release_split must be {expected_release_split}")
        provider = real_system_report_provider(report_path)
        if provider:
            provider_reports.setdefault(provider, []).append(str(report_path))
        trace_path = None
        trace_has_envelope = False
        trace_errors: list[str] = []
        trace_artifact = item.get("framework_trace_artifact")
        if trace_artifact is not None:
            num_framework_trace_artifacts += 1
            if not isinstance(trace_artifact, dict) or not trace_artifact.get("path"):
                trace_errors.append(f"{source}: systems[{index}] framework_trace_artifact.path is required")
            else:
                trace_path = _resolve_report_path(source, str(trace_artifact["path"]), contract_root=contract_root)
                trace_has_envelope = _framework_trace_artifact_has_envelope(trace_path)
                if trace_has_envelope:
                    num_framework_trace_envelope_artifacts += 1
                trace_errors.extend(
                    _validate_framework_trace_artifact(
                        trace_path,
                        expected_records=int(item.get("num_predictions", 0) or 0),
                    )
                )
        rows.append(
            {
                "system_id": system_id,
                "provider": provider,
                "report_path": str(report_path),
                "framework_trace_path": str(trace_path) if trace_path is not None else None,
                "framework_trace_has_envelope": trace_has_envelope,
                **_real_system_matrix_artifact_binding(
                    report_path,
                    report=report,
                    trace_path=trace_path,
                ),
                "status": "passed" if not report_errors and not trace_errors else "invalid",
                "errors": list(report_errors) + list(trace_errors),
            }
        )
        errors.extend(report_errors)
        errors.extend(trace_errors)

    required = set(required_providers)
    missing_providers = sorted(required - set(provider_reports))
    if missing_providers:
        errors.append(f"{source}: missing required providers: {missing_providers}")
    return {
        "schema_version": REAL_SYSTEM_EVIDENCE_SCHEMA_VERSION,
        "matrix_summary": str(source),
        "root": _artifact_root_ref(source.parent, contract_root),
        "expected_benchmark_id": expected_benchmark_id,
        "expected_release_split": expected_release_split,
        "status": "passed" if not errors else "incomplete",
        "num_systems": len(rows),
        "num_framework_trace_artifacts": num_framework_trace_artifacts,
        "num_framework_trace_envelope_artifacts": num_framework_trace_envelope_artifacts,
        "required_providers": sorted(required),
        "covered_providers": sorted(provider_reports),
        "missing_providers": missing_providers,
        "systems": rows,
        "errors": errors,
    }


def write_real_system_matrix_validation(
    summary_path: str | Path,
    output: str | Path,
    *,
    expected_benchmark_id: str | None = None,
    expected_release_split: str | None = None,
) -> dict[str, Any]:
    report = validate_real_system_matrix_summary(
        summary_path,
        expected_benchmark_id=expected_benchmark_id,
        expected_release_split=expected_release_split,
    )
    output_path = Path(output)
    contract_root = _infer_real_system_validation_contract_root(output_path, report)
    _normalize_real_system_validation_contract(report, contract_root=contract_root)
    report["root"] = _artifact_root_ref(output_path.parent, contract_root)
    write_json(output, report)
    return report


def _validate_framework_trace_artifact(path: Path, *, expected_records: int | None = None) -> list[str]:
    try:
        payload = read_json(path)
    except Exception as exc:
        return [f"{path}: cannot read framework_trace_artifact: {exc}"]
    errors = validate_framework_trace_artifact_payload(
        payload,
        expected_records=expected_records,
        tool_runtime_contracts=load_default_tool_runtime_contracts(),
    )
    return [f"{path}: {error}" for error in errors]


def _framework_trace_artifact_has_envelope(path: Path) -> bool:
    try:
        payload = read_json(path)
    except Exception:
        return False
    return (
        isinstance(payload, dict)
        and payload.get("schema_version") == FRAMEWORK_TRACE_ARTIFACT_SCHEMA_VERSION
    )


def _real_system_matrix_artifact_binding(
    report_path: Path,
    *,
    report: dict[str, Any],
    trace_path: Path | None,
) -> dict[str, Any]:
    submission = report.get("submission", {}) if isinstance(report.get("submission"), dict) else {}
    role_paths: dict[str, str] = {"report": str(report_path)}
    role_sha256s: dict[str, str | None] = {"report": _sha256_if_exists(report_path)}
    hash_matches_report: dict[str, bool] = {"report": role_sha256s["report"] is not None}

    for role, field in (
        ("prediction", "prediction_artifact"),
        ("run_metadata", "run_metadata_artifact"),
    ):
        artifact = submission.get(field) if isinstance(submission, dict) else None
        resolved = _resolve_submission_artifact_path(report_path, artifact)
        role_paths[role] = str(resolved) if resolved is not None else ""
        actual_sha = _sha256_if_exists(resolved) if resolved is not None else None
        expected_sha = artifact.get("sha256") if isinstance(artifact, dict) else None
        role_sha256s[role] = actual_sha
        hash_matches_report[role] = bool(actual_sha and expected_sha and actual_sha == expected_sha)

    if trace_path is not None:
        role_paths["framework_trace"] = str(trace_path)
        trace_sha = _sha256_if_exists(trace_path)
        role_sha256s["framework_trace"] = trace_sha
        trace_artifact = submission.get("framework_trace_artifact") if isinstance(submission, dict) else None
        expected_trace_sha = trace_artifact.get("sha256") if isinstance(trace_artifact, dict) else None
        hash_matches_report["framework_trace"] = bool(
            trace_sha and expected_trace_sha and trace_sha == expected_trace_sha
        )

    materialized_roles = [role for role, sha in role_sha256s.items() if sha]
    valid_roles = [role for role, matched in hash_matches_report.items() if matched]
    return {
        "artifact_paths": role_paths,
        "artifact_sha256s": role_sha256s,
        "artifact_hash_matches_report": hash_matches_report,
        "num_materialized_artifacts": len(materialized_roles),
        "num_hash_matched_artifacts": len(valid_roles),
        "all_artifact_hashes_match_report": len(valid_roles) == len(role_paths),
        "artifact_bundle_fingerprint": _artifact_bundle_fingerprint(
            system_id=str(report.get("system_id", "")),
            provider=real_system_report_provider(report_path),
            role_paths=role_paths,
            role_sha256s=role_sha256s,
            hash_matches_report=hash_matches_report,
        ),
    }


def _resolve_submission_artifact_path(report_path: Path, artifact: Any) -> Path | None:
    if not isinstance(artifact, dict) or not artifact.get("path"):
        return None
    return _resolve_contract_path(str(artifact["path"]), anchor=report_path.parent)


def _sha256_if_exists(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_bundle_fingerprint(
    *,
    system_id: str,
    provider: str | None,
    role_paths: dict[str, str],
    role_sha256s: dict[str, str | None],
    hash_matches_report: dict[str, bool],
) -> str:
    payload = {
        "system_id": system_id,
        "provider": provider,
        "roles": [
            {
                "role": role,
                "path": Path(path).name if path else "",
                "sha256": role_sha256s.get(role),
                "hash_matches_report": hash_matches_report.get(role),
            }
            for role, path in sorted(role_paths.items())
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def default_real_system_config_validation_output(output: str | Path) -> Path:
    output_path = Path(output)
    name = output_path.name
    if name.endswith("_current.json"):
        target_name = f"{name[:-len('_current.json')]}_config_validation.json"
    elif name.endswith(".json"):
        target_name = f"{name[:-len('.json')]}_config_validation.json"
    else:
        target_name = f"{name}_config_validation.json"
    return output_path.with_name(target_name)


def write_real_system_config_validation(config_paths: Iterable[str | Path], output: str | Path) -> dict[str, Any]:
    report = validate_integration_config_files(config_paths)
    output_path = Path(output)
    contract_root = _infer_real_system_config_validation_contract_root(output_path, report)
    _normalize_real_system_config_validation_contract(report, contract_root=contract_root)
    report["root"] = _artifact_root_ref(output_path.parent, contract_root)
    write_json(output_path, report)
    return report


def _infer_real_system_config_validation_contract_root(output_path: Path, report: dict[str, Any]) -> Path:
    candidates: list[Path] = [output_path, output_path.parent]
    configs = report.get("configs")
    if isinstance(configs, list):
        for item in configs:
            if not isinstance(item, dict):
                continue
            raw_path = item.get("path")
            if isinstance(raw_path, str) and raw_path.strip():
                candidates.append(_resolve_contract_path(raw_path))
    return _infer_contract_root(candidates)


def _normalize_real_system_config_validation_contract(report: dict[str, Any], *, contract_root: Path) -> None:
    configs = report.get("configs")
    if not isinstance(configs, list):
        return
    for item in configs:
        if not isinstance(item, dict):
            continue
        raw_path = item.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            item["path"] = _contract_relative_or_absolute(
                contract_root,
                _resolve_contract_path(raw_path, contract_root=contract_root),
            )
        artifact = item.get("integration_config_artifact")
        if isinstance(artifact, dict):
            artifact_path = artifact.get("path")
            if isinstance(artifact_path, str) and artifact_path.strip():
                artifact["path"] = _contract_relative_or_absolute(
                    contract_root,
                    _resolve_contract_path(artifact_path, contract_root=contract_root),
                )


def _infer_real_system_validation_contract_root(output_path: Path, report: dict[str, Any]) -> Path:
    candidates: list[Path] = [output_path, output_path.parent]
    matrix_summary = report.get("matrix_summary")
    if isinstance(matrix_summary, str) and matrix_summary.strip():
        summary_path = _resolve_contract_path(matrix_summary)
        candidates.append(summary_path)
        try:
            summary = _read_summary_object(summary_path)
        except ValueError:
            summary = None
        if isinstance(summary, dict):
            return _contract_root_from_summary(summary_path, summary)
    return _infer_contract_root(candidates)


def _normalize_real_system_validation_contract(report: dict[str, Any], *, contract_root: Path) -> None:
    matrix_summary = report.get("matrix_summary")
    if isinstance(matrix_summary, str) and matrix_summary.strip():
        report["matrix_summary"] = _contract_relative_or_absolute(
            contract_root,
            _resolve_contract_path(matrix_summary, contract_root=contract_root),
        )
    systems = report.get("systems")
    if not isinstance(systems, list):
        return
    for row in systems:
        if not isinstance(row, dict):
            continue
        raw_path = row.get("report_path")
        if isinstance(raw_path, str) and raw_path.strip():
            row["report_path"] = _contract_relative_or_absolute(
                contract_root,
                _resolve_contract_path(raw_path, contract_root=contract_root),
            )


def summarize_real_system_run_progress(
    manifest_path: str | Path,
    *,
    split: str,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Summarize progress for one resumable real-system run directory."""

    manifest_file = Path(manifest_path)
    manifest = _read_summary_object(manifest_file)
    split_report = manifest.get("split_reports", {}).get(split, {})
    total_queries = int(split_report.get("num_queries", 0) or 0)
    total_cases = int(split_report.get("num_cases", 0) or 0)

    run_root = Path(run_dir)
    predictions_path = run_root / "predictions.json"
    metadata_path = run_root / "run_metadata.json"
    run_state_path = run_root / "run_state.json"
    report_path = run_root / "report.json"
    errors: list[str] = []

    system_id = run_root.name
    num_predictions = 0
    if predictions_path.exists():
        try:
            predictions = load_predictions(predictions_path)
            system_id = predictions.system_id or system_id
            num_predictions = len(predictions.predictions)
            query_ids = [item.query_id for item in predictions.predictions]
            if len(query_ids) != len(set(query_ids)):
                errors.append(f"{predictions_path}: duplicate prediction query_id detected")
        except Exception as exc:
            errors.append(f"{predictions_path}: cannot load predictions: {exc}")
    live_state: dict[str, Any] | None = None
    if run_state_path.exists():
        try:
            raw_state = read_json(run_state_path)
            if isinstance(raw_state, dict):
                live_state = raw_state
            else:
                errors.append(f"{run_state_path}: run state must be a JSON object")
        except Exception as exc:
            errors.append(f"{run_state_path}: cannot read run state: {exc}")
    report_validation_errors = validate_real_system_report(report_path) if report_path.exists() else ()
    errors.extend(report_validation_errors)

    if report_path.exists() and not report_validation_errors:
        status = "completed"
    elif predictions_path.exists():
        status = "predictions_complete_report_missing" if total_queries > 0 and num_predictions >= total_queries else "running_or_partial"
    else:
        status = "not_started"
    if errors:
        status = "invalid" if report_path.exists() else "running_with_errors"

    progress_ratio = None
    if total_queries > 0:
        progress_ratio = min(max(num_predictions / total_queries, 0.0), 1.0)

    return {
        "schema_version": REAL_SYSTEM_EVIDENCE_SCHEMA_VERSION,
        "status": status,
        "manifest": str(manifest_file),
        "benchmark_id": f"{manifest.get('benchmark_id', 'release')}-{split}",
        "release_split": split,
        "system_id": system_id,
        "run_dir": str(run_root),
        "predictions_path": str(predictions_path),
        "run_metadata_path": str(metadata_path),
        "run_state_path": str(run_state_path),
        "report_path": str(report_path),
        "has_predictions": predictions_path.exists(),
        "has_run_metadata": metadata_path.exists(),
        "has_run_state": run_state_path.exists(),
        "has_report": report_path.exists(),
        "num_predictions": num_predictions,
        "total_queries": total_queries,
        "total_cases": total_cases,
        "progress_ratio": progress_ratio,
        "checkpoint_modified_at": _iso_mtime(predictions_path) if predictions_path.exists() else None,
        "run_state_modified_at": _iso_mtime(run_state_path) if run_state_path.exists() else None,
        "live_state": live_state,
        "errors": errors,
    }


def write_real_system_run_progress(
    manifest_path: str | Path,
    output: str | Path,
    *,
    split: str,
    run_dir: str | Path,
) -> dict[str, Any]:
    report = summarize_real_system_run_progress(manifest_path, split=split, run_dir=run_dir)
    write_json(output, report)
    return report


def refresh_real_system_canonical_matrix(
    manifest_path: str | Path,
    *,
    split: str,
    run_specs: Iterable[dict[str, Any]],
    merged_summary_output: str | Path | None = None,
    merged_validation_output: str | Path | None = None,
    expected_benchmark_id: str | None = None,
    expected_release_split: str | None = None,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
) -> dict[str, Any]:
    """Refresh per-provider run state and materialize the best available canonical matrix artifacts."""

    from amb.benchmark.release.evaluation import release_split_benchmark_id

    manifest_file = Path(manifest_path)
    expected_benchmark = expected_benchmark_id or release_split_benchmark_id(manifest_file, split)
    expected_split = expected_release_split or split
    run_rows: list[dict[str, Any]] = []
    summary_paths: list[Path] = []

    for raw_spec in run_specs:
        if not isinstance(raw_spec, dict):
            raise ValueError("each run spec must be a mapping with config_path and run_dir")
        config_path = Path(str(raw_spec.get("config_path") or ""))
        run_dir = Path(str(raw_spec.get("run_dir") or ""))
        if not config_path:
            raise ValueError("run spec config_path is required")
        if not run_dir:
            raise ValueError("run spec run_dir is required")

        config = read_json(config_path)
        if not isinstance(config, dict):
            raise ValueError(f"{config_path}: integration config must be a JSON object")
        provider = canonical_provider(str(config.get("provider", ""))) or str(raw_spec.get("provider") or "")
        if not provider:
            raise ValueError(f"{config_path}: integration config provider is required")
        summary_path = Path(str(raw_spec.get("summary_path") or _default_run_summary_path(run_dir)))
        progress_output = raw_spec.get("progress_output")
        progress_output_path = Path(str(progress_output)) if progress_output else None
        row: dict[str, Any] = {
            "provider": provider,
            "config_path": str(config_path),
            "run_dir": str(run_dir),
            "summary_path": str(summary_path),
            "progress_output": str(progress_output_path) if progress_output_path is not None else None,
            "actions": [],
            "errors": [],
        }
        progress_before = summarize_real_system_run_progress(manifest_file, split=split, run_dir=run_dir)
        row["progress_before"] = progress_before
        progress_after = dict(progress_before)

        if not progress_before.get("has_run_metadata"):
            try:
                backfill_real_system_run_metadata(
                    manifest_file,
                    split=split,
                    config_path=config_path,
                    run_dir=run_dir,
                    retrieval_k=retrieval_k,
                )
                row["actions"].append("backfilled_run_metadata")
                progress_after = summarize_real_system_run_progress(manifest_file, split=split, run_dir=run_dir)
            except Exception as exc:
                row["errors"].append(str(exc))

        should_finalize = (
            not row["errors"]
            and (
                progress_after.get("status") == "predictions_complete_report_missing"
                or (progress_after.get("status") == "completed" and not summary_path.exists())
            )
        )
        if should_finalize:
            try:
                finalize_real_system_run(
                    manifest_file,
                    split=split,
                    run_dir=run_dir,
                    retrieval_k=retrieval_k,
                    summary_output=summary_path,
                )
                row["actions"].append("finalized_run")
                progress_after = summarize_real_system_run_progress(manifest_file, split=split, run_dir=run_dir)
            except Exception as exc:
                row["errors"].append(str(exc))

        should_refinalize = (
            not row["errors"]
            and progress_after.get("has_run_metadata")
            and int(progress_after.get("num_predictions") or 0) >= int(progress_after.get("total_queries") or 0) > 0
            and _completed_run_artifacts_need_repair(progress_after, summary_path)
        )
        if should_refinalize:
            try:
                finalize_real_system_run(
                    manifest_file,
                    split=split,
                    run_dir=run_dir,
                    retrieval_k=retrieval_k,
                    summary_output=summary_path,
                )
                row["actions"].append("refinalized_run")
                progress_after = summarize_real_system_run_progress(manifest_file, split=split, run_dir=run_dir)
            except Exception as exc:
                row["errors"].append(str(exc))

        row["progress_after"] = progress_after
        if summary_path.exists():
            row["summary_exists"] = True
            if summary_path not in summary_paths:
                summary_paths.append(summary_path)
        else:
            row["summary_exists"] = False
        if progress_output_path is not None:
            try:
                write_json(progress_output_path, progress_after)
                row["actions"].append("wrote_progress_snapshot")
            except Exception as exc:
                row["errors"].append(str(exc))
        _project_refresh_progress_fields(row, progress_after)
        row["status"] = _refresh_row_status(row["errors"], progress_after, row["summary_exists"])
        run_rows.append(row)

    merged_summary_path = Path(merged_summary_output) if merged_summary_output is not None else None
    merged_validation_path = Path(merged_validation_output) if merged_validation_output is not None else None
    merged_summary: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    merge_errors: list[str] = []
    if summary_paths:
        try:
            if merged_summary_path is None:
                first_parent = summary_paths[0].parent
                merged_summary_path = first_parent / "canonical_matrix_summary.json"
            merged_summary = write_merged_real_system_matrix_summary(
                summary_paths,
                merged_summary_path,
                expected_benchmark_id=expected_benchmark,
                expected_release_split=expected_split,
            )
            if merged_validation_path is None:
                merged_validation_path = merged_summary_path.parent / "canonical_matrix_validation.json"
            validation = write_real_system_matrix_validation(
                merged_summary_path,
                merged_validation_path,
                expected_benchmark_id=expected_benchmark,
                expected_release_split=expected_split,
            )
        except Exception as exc:
            merge_errors.append(str(exc))

    running = sum(1 for row in run_rows if row["progress_after"].get("status") == "running_or_partial")
    ready_to_finalize = sum(1 for row in run_rows if row["progress_after"].get("status") == "predictions_complete_report_missing")
    completed = sum(1 for row in run_rows if row["progress_after"].get("status") == "completed")
    invalid = sum(1 for row in run_rows if row["status"] == "invalid")
    overall_status = "passed"
    if merge_errors or invalid:
        overall_status = "invalid"
    elif validation is None or validation.get("status") != "passed":
        overall_status = "partial"
    if running or ready_to_finalize:
        overall_status = "partial" if overall_status == "passed" else overall_status

    return {
        "schema_version": REAL_SYSTEM_EVIDENCE_SCHEMA_VERSION,
        "status": overall_status,
        "manifest": str(manifest_file),
        "expected_benchmark_id": expected_benchmark,
        "expected_release_split": expected_split,
        "summary": {
            "num_runs": len(run_rows),
            "num_running": running,
            "num_ready_to_finalize": ready_to_finalize,
            "num_completed": completed,
            "num_invalid": invalid,
            "num_available_summaries": len(summary_paths),
        },
        "runs": run_rows,
        "available_summary_paths": [str(path) for path in summary_paths],
        "merged_summary_path": str(merged_summary_path) if merged_summary_path is not None else None,
        "merged_validation_path": str(merged_validation_path) if merged_validation_path is not None else None,
        "merged_summary": merged_summary,
        "merged_validation": validation,
        "errors": merge_errors,
    }


def write_real_system_canonical_refresh(
    manifest_path: str | Path,
    output: str | Path,
    *,
    split: str,
    run_specs: Iterable[dict[str, Any]],
    config_validation_output: str | Path | None = None,
    analysis_output: str | Path | None = None,
    merged_summary_output: str | Path | None = None,
    merged_validation_output: str | Path | None = None,
    expected_benchmark_id: str | None = None,
    expected_release_split: str | None = None,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
) -> dict[str, Any]:
    run_specs_tuple = tuple(run_specs)
    report = refresh_real_system_canonical_matrix(
        manifest_path,
        split=split,
        run_specs=run_specs_tuple,
        merged_summary_output=merged_summary_output,
        merged_validation_output=merged_validation_output,
        expected_benchmark_id=expected_benchmark_id,
        expected_release_split=expected_release_split,
        retrieval_k=retrieval_k,
    )
    output_path = Path(output)
    config_validation_path = (
        Path(config_validation_output)
        if config_validation_output is not None
        else default_real_system_config_validation_output(output_path)
    )
    analysis_path = (
        Path(analysis_output)
        if analysis_output is not None
        else default_real_system_analysis_output(output_path)
    )
    config_validation = write_real_system_config_validation(
        [str(spec["config_path"]) for spec in run_specs_tuple if isinstance(spec, dict) and spec.get("config_path")],
        config_validation_path,
    )
    report["config_validation_path"] = str(config_validation_path)
    report["config_validation_status"] = config_validation.get("status")
    report["config_validation_errors"] = list(config_validation.get("errors", []))
    report["config_validation_num_configs"] = int(config_validation.get("num_configs", 0) or 0)
    merged_summary_path = report.get("merged_summary_path")
    if isinstance(merged_summary_path, str) and merged_summary_path.strip():
        analysis = write_real_system_analysis(
            merged_summary_path,
            analysis_path,
            expected_benchmark_id=expected_benchmark_id,
            expected_release_split=expected_release_split,
        )
        report["analysis_path"] = str(analysis_path)
        report["analysis_status"] = "passed"
        report["analysis_num_reports"] = int(analysis.get("num_reports", 0) or 0)
        report["analysis"] = analysis
    else:
        report["analysis_path"] = str(analysis_path)
        report["analysis_status"] = "missing"
        report["analysis_num_reports"] = 0
        report["analysis_errors"] = ["merged_summary_path is required to build real-system analysis"]
    contract_root = _infer_real_system_refresh_contract_root(output_path, report)
    _normalize_real_system_refresh_contract(report, contract_root=contract_root)
    report["root"] = _artifact_root_ref(output_path.parent, contract_root)
    write_json(output, report)
    return report


def _infer_real_system_refresh_contract_root(output_path: Path, report: dict[str, Any]) -> Path:
    candidates: list[Path] = [output_path, output_path.parent]
    for key in ("manifest", "merged_summary_path", "merged_validation_path", "config_validation_path", "analysis_path"):
        raw_path = report.get(key)
        if isinstance(raw_path, str) and raw_path.strip():
            candidates.append(_resolve_contract_path(raw_path))
    for raw_path in report.get("available_summary_paths", []):
        if isinstance(raw_path, str) and raw_path.strip():
            candidates.append(_resolve_contract_path(raw_path))
    for row in report.get("runs", []):
        if not isinstance(row, dict):
            continue
        for key in ("config_path", "run_dir", "summary_path", "progress_output"):
            raw_path = row.get(key)
            if isinstance(raw_path, str) and raw_path.strip():
                candidates.append(_resolve_contract_path(raw_path))
    return _infer_contract_root(candidates)


def _normalize_real_system_refresh_contract(report: dict[str, Any], *, contract_root: Path) -> None:
    for key in ("manifest", "merged_summary_path", "merged_validation_path", "config_validation_path", "analysis_path"):
        raw_path = report.get(key)
        if isinstance(raw_path, str) and raw_path.strip():
            report[key] = _contract_relative_or_absolute(
                contract_root,
                _resolve_contract_path(raw_path, contract_root=contract_root),
            )
    available_summary_paths = report.get("available_summary_paths")
    if isinstance(available_summary_paths, list):
        report["available_summary_paths"] = [
            _contract_relative_or_absolute(contract_root, _resolve_contract_path(raw_path, contract_root=contract_root))
            if isinstance(raw_path, str) and raw_path.strip()
            else raw_path
            for raw_path in available_summary_paths
        ]
    for row in report.get("runs", []):
        if not isinstance(row, dict):
            continue
        for key in ("config_path", "run_dir", "summary_path", "progress_output"):
            raw_path = row.get(key)
            if isinstance(raw_path, str) and raw_path.strip():
                row[key] = _contract_relative_or_absolute(
                    contract_root,
                    _resolve_contract_path(raw_path, contract_root=contract_root),
                )
        for progress_key in ("progress_before", "progress_after"):
            progress = row.get(progress_key)
            if not isinstance(progress, dict):
                continue
            for key in ("predictions_path", "report_path", "run_metadata_path", "run_state_path"):
                raw_path = progress.get(key)
                if isinstance(raw_path, str) and raw_path.strip():
                    progress[key] = _contract_relative_or_absolute(
                        contract_root,
                        _resolve_contract_path(raw_path, contract_root=contract_root),
                    )
    merged_summary = report.get("merged_summary")
    if isinstance(merged_summary, dict):
        _normalize_real_system_summary_contract(merged_summary, contract_root=contract_root)
    merged_validation = report.get("merged_validation")
    if isinstance(merged_validation, dict):
        _normalize_real_system_validation_contract(merged_validation, contract_root=contract_root)


def backfill_real_system_run_metadata(
    manifest_path: str | Path,
    *,
    split: str,
    config_path: str | Path,
    run_dir: str | Path,
    system_id: str | None = None,
    system_version: str | None = None,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    resume: bool = True,
    command: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Backfill run_metadata.json for an existing real-system run directory."""

    from amb.benchmark.release.evaluation import release_split_benchmark_id

    manifest_file = Path(manifest_path)
    config_file = Path(config_path)
    run_root = Path(run_dir)
    metadata_path = run_root / "run_metadata.json"
    predictions_path = run_root / "predictions.json"

    if metadata_path.exists() and not overwrite:
        raise ValueError(f"{metadata_path}: already exists; use overwrite=True to replace it")
    if not config_file.exists():
        raise ValueError(f"{config_file}: integration config does not exist")

    config = read_json(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"{config_file}: integration config must be a JSON object")
    provider = canonical_provider(str(config.get("provider", "")))
    if not provider:
        raise ValueError(f"{config_file}: integration config provider is required")
    execution_mode = str(config.get("execution_mode") or "integration_smoke")
    config_artifact = artifact_info(config_file)
    real_system_attestation = build_real_system_attestation(
        config.get("real_system_attestation") if "real_system_attestation" in config else None,
        config_artifact,
        config_provider=provider,
        config_client_factory=str(config.get("client_factory", "")),
    )
    if execution_mode == "real_system" and real_system_attestation is None:
        raise ValueError(f"{config_file}: real_system execution_mode requires real_system_attestation")
    if execution_mode != "real_system" and real_system_attestation is not None:
        raise ValueError(f"{config_file}: real_system_attestation requires execution_mode=real_system")

    resolved_system_id = str(system_id or config.get("system_id") or run_root.name)
    if predictions_path.exists():
        predictions = load_predictions(predictions_path)
        if predictions.system_id and predictions.system_id != resolved_system_id:
            raise ValueError(
                f"{predictions_path}: predictions system_id mismatch, expected {resolved_system_id!r}, got {predictions.system_id!r}"
            )
        if predictions.system_id:
            resolved_system_id = predictions.system_id

    resolved_system_version = str(system_version or config.get("system_version") or "unspecified")
    resolved_command = command or _default_real_system_matrix_command(
        manifest_file,
        split=split,
        config_path=config_file,
        output_dir=run_root.parent,
        system_version=resolved_system_version,
        retrieval_k=retrieval_k,
        resume=resume,
    )
    metadata = build_run_metadata(
        system_id=resolved_system_id,
        system_version=resolved_system_version,
        benchmark_id=release_split_benchmark_id(manifest_file, split),
        release_split=split,
        command=resolved_command,
        dependencies={
            "integration_provider": provider,
            "integration_config": str(config_file),
            "agent_memory_benchmark": "local",
        },
        execution_mode=execution_mode,
        integration_config_artifact=config_artifact,
        real_system_attestation=real_system_attestation,
    )
    errors = list(
        validate_run_metadata(
            metadata,
            system_id=resolved_system_id,
            benchmark_id=release_split_benchmark_id(manifest_file, split),
            release_split=split,
        )
    )
    errors.extend(validate_real_system_run_metadata(metadata))
    if errors:
        raise ValueError(f"{metadata_path}: invalid backfilled run metadata: {'; '.join(errors[:3])}")
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(metadata_path, metadata)
    return {
        "schema_version": REAL_SYSTEM_EVIDENCE_SCHEMA_VERSION,
        "status": "backfilled",
        "run_dir": str(run_root),
        "system_id": resolved_system_id,
        "benchmark_id": metadata["benchmark_id"],
        "release_split": split,
        "run_metadata_path": str(metadata_path),
        "integration_config_artifact": config_artifact,
        "real_system_attestation": real_system_attestation,
    }


def finalize_real_system_run(
    manifest_path: str | Path,
    *,
    split: str,
    run_dir: str | Path,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    summary_output: str | Path | None = None,
) -> dict[str, Any]:
    """Finalize one resumable real-system run into a validated report and one-system matrix summary."""

    from amb.benchmark.release.evaluation import evaluate_release_split_predictions, release_split_benchmark_id

    progress = summarize_real_system_run_progress(manifest_path, split=split, run_dir=run_dir)
    run_root = Path(run_dir)
    predictions_path = run_root / "predictions.json"
    metadata_path = run_root / "run_metadata.json"
    report_path = run_root / "report.json"
    summary_path = Path(summary_output) if summary_output is not None else run_root.parent / "matrix_summary.json"
    manifest_file = Path(manifest_path)

    if not predictions_path.exists():
        raise ValueError(f"{predictions_path}: predictions checkpoint is required")
    if not metadata_path.exists():
        raise ValueError(f"{metadata_path}: run metadata is required")
    if progress["total_queries"] <= 0:
        raise ValueError(f"{manifest_file}: split {split!r} must define a positive num_queries")
    if progress["num_predictions"] < progress["total_queries"]:
        raise ValueError(
            f"{predictions_path}: run is incomplete ({progress['num_predictions']}/{progress['total_queries']} predictions)"
        )

    report = evaluate_release_split_predictions(
        manifest_file,
        predictions_path,
        split=split,
        retrieval_k=retrieval_k,
        run_metadata_path=metadata_path,
    )
    write_json(report_path, report)
    report_errors = validate_real_system_report(report_path)
    if report_errors:
        raise ValueError(f"{report_path}: invalid finalized report: {'; '.join(report_errors[:3])}")

    predictions = load_predictions(predictions_path)
    run_metadata = load_run_metadata(metadata_path)
    system_id = predictions.system_id or str(run_metadata.get("system_id") or run_root.name)
    summary = {
        "schema_version": "amst-release-agent-matrix-v1",
        "release_manifest": str(manifest_file),
        "release_manifest_artifact": artifact_info(manifest_file),
        "release_split": split,
        "benchmark_id": release_split_benchmark_id(manifest_file, split),
        "num_systems": 1,
        "systems": [
            {
                "system_id": system_id,
                "system_version": str(run_metadata.get("system_version") or "unspecified"),
                "prediction_artifact": artifact_info(predictions_path),
                "run_metadata_artifact": artifact_info(metadata_path),
                "report_artifact": artifact_info(report_path),
                "aggregate": report["aggregate"],
                "num_predictions": len(predictions.predictions),
                "missing_predictions": report.get("missing_predictions", []),
                "extra_predictions": report.get("extra_predictions", []),
            }
        ],
    }
    write_json(summary_path, summary)
    return {
        "schema_version": REAL_SYSTEM_EVIDENCE_SCHEMA_VERSION,
        "status": "finalized",
        "benchmark_id": summary["benchmark_id"],
        "release_split": split,
        "system_id": system_id,
        "num_predictions": len(predictions.predictions),
        "total_queries": int(progress["total_queries"]),
        "report_generated": True,
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def merge_real_system_matrix_summaries(
    summary_paths: Iterable[str | Path],
    *,
    expected_benchmark_id: str | None = None,
    expected_release_split: str | None = None,
) -> dict[str, Any]:
    """Merge one or more single/multi-system matrix summaries into one summary."""

    sources = [Path(path) for path in summary_paths]
    if not sources:
        raise ValueError("at least one matrix summary path is required")

    merged_systems: list[dict[str, Any]] = []
    seen_system_ids: set[str] = set()
    benchmark_id: str | None = None
    release_split: str | None = None
    release_manifest: str | None = None
    release_manifest_artifact: dict[str, Any] | None = None

    for source in sources:
        summary = _read_summary_object(source)
        source_contract_root = _contract_root_from_summary(source, summary)
        if summary.get("schema_version") != "amst-release-agent-matrix-v1":
            raise ValueError(f"{source}: schema_version must be amst-release-agent-matrix-v1")

        source_benchmark_id = str(summary.get("benchmark_id") or "")
        source_release_split = str(summary.get("release_split") or "")
        if not source_benchmark_id:
            raise ValueError(f"{source}: benchmark_id is required")
        if not source_release_split:
            raise ValueError(f"{source}: release_split is required")
        if expected_benchmark_id is not None and source_benchmark_id != expected_benchmark_id:
            raise ValueError(f"{source}: benchmark_id must be {expected_benchmark_id}")
        if expected_release_split is not None and source_release_split != expected_release_split:
            raise ValueError(f"{source}: release_split must be {expected_release_split}")
        if benchmark_id is None:
            benchmark_id = source_benchmark_id
        elif source_benchmark_id != benchmark_id:
            raise ValueError(f"{source}: benchmark_id mismatch, expected {benchmark_id}")
        if release_split is None:
            release_split = source_release_split
        elif source_release_split != release_split:
            raise ValueError(f"{source}: release_split mismatch, expected {release_split}")

        source_manifest = str(summary.get("release_manifest") or "")
        if source_manifest:
            if release_manifest is None:
                release_manifest = source_manifest
            elif source_manifest != release_manifest:
                raise ValueError(f"{source}: release_manifest mismatch, expected {release_manifest}")
        source_manifest_artifact = summary.get("release_manifest_artifact")
        if source_manifest_artifact is not None:
            if not isinstance(source_manifest_artifact, dict):
                raise ValueError(f"{source}: release_manifest_artifact must be an object")
            if release_manifest_artifact is None:
                release_manifest_artifact = dict(source_manifest_artifact)
            elif source_manifest_artifact != release_manifest_artifact:
                raise ValueError(f"{source}: release_manifest_artifact mismatch across summaries")

        systems = summary.get("systems")
        if not isinstance(systems, list) or not systems:
            raise ValueError(f"{source}: systems must be a non-empty list")
        for index, item in enumerate(systems, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"{source}: systems[{index}] must be an object")
            system_id = str(item.get("system_id") or "")
            if not system_id:
                raise ValueError(f"{source}: systems[{index}] system_id is required")
            if system_id in seen_system_ids:
                raise ValueError(f"{source}: duplicate system_id across summaries: {system_id}")
            artifact = item.get("report_artifact")
            if not isinstance(artifact, dict) or not artifact.get("path"):
                raise ValueError(f"{source}: systems[{index}] report_artifact.path is required")
            report_path = _resolve_report_path(source, str(artifact["path"]), contract_root=source_contract_root)
            report_errors = validate_real_system_report(report_path)
            if report_errors:
                raise ValueError(f"{source}: systems[{index}] invalid report: {'; '.join(report_errors[:3])}")
            report = _read_report_object(report_path)
            if benchmark_id is not None and report.get("benchmark_id") != benchmark_id:
                raise ValueError(f"{report_path}: benchmark_id must be {benchmark_id}")
            if release_split is not None and report.get("release_split") != release_split:
                raise ValueError(f"{report_path}: release_split must be {release_split}")
            seen_system_ids.add(system_id)
            merged_systems.append(dict(item))

    return {
        "schema_version": "amst-release-agent-matrix-v1",
        "release_manifest": release_manifest,
        "release_manifest_artifact": release_manifest_artifact,
        "release_split": release_split,
        "benchmark_id": benchmark_id,
        "num_systems": len(merged_systems),
        "systems": merged_systems,
        "source_summaries": [str(path) for path in sources],
    }


def write_merged_real_system_matrix_summary(
    summary_paths: Iterable[str | Path],
    output: str | Path,
    *,
    expected_benchmark_id: str | None = None,
    expected_release_split: str | None = None,
) -> dict[str, Any]:
    summary = merge_real_system_matrix_summaries(
        summary_paths,
        expected_benchmark_id=expected_benchmark_id,
        expected_release_split=expected_release_split,
    )
    output_path = Path(output)
    contract_root = _infer_real_system_summary_contract_root(output_path, summary)
    _normalize_real_system_summary_contract(summary, contract_root=contract_root)
    summary["root"] = _artifact_root_ref(output_path.parent, contract_root)
    write_json(output, summary)
    return summary


def _infer_real_system_summary_contract_root(output_path: Path, summary: dict[str, Any]) -> Path:
    candidates: list[Path] = [output_path, output_path.parent]
    release_manifest = summary.get("release_manifest")
    if isinstance(release_manifest, str) and release_manifest.strip():
        candidates.append(_resolve_contract_path(release_manifest))
    for raw_path in summary.get("source_summaries", []):
        if isinstance(raw_path, str) and raw_path.strip():
            candidates.append(_resolve_contract_path(raw_path))
    return _infer_contract_root(candidates)


def _normalize_real_system_summary_contract(summary: dict[str, Any], *, contract_root: Path) -> None:
    release_manifest = summary.get("release_manifest")
    if isinstance(release_manifest, str) and release_manifest.strip():
        summary["release_manifest"] = _contract_relative_or_absolute(
            contract_root,
            _resolve_contract_path(release_manifest, contract_root=contract_root),
        )
    release_manifest_artifact = summary.get("release_manifest_artifact")
    if isinstance(release_manifest_artifact, dict):
        raw_path = release_manifest_artifact.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            release_manifest_artifact["path"] = _contract_relative_or_absolute(
                contract_root,
                _resolve_contract_path(raw_path, contract_root=contract_root),
            )
    source_summaries = summary.get("source_summaries")
    if isinstance(source_summaries, list):
        summary["source_summaries"] = [
            _contract_relative_or_absolute(contract_root, _resolve_contract_path(raw_path, contract_root=contract_root))
            if isinstance(raw_path, str) and raw_path.strip()
            else raw_path
            for raw_path in source_summaries
        ]
    systems = summary.get("systems")
    if not isinstance(systems, list):
        return
    for row in systems:
        if not isinstance(row, dict):
            continue
        for artifact_key in ("prediction_artifact", "run_metadata_artifact", "report_artifact", "framework_trace_artifact"):
            artifact = row.get(artifact_key)
            if not isinstance(artifact, dict):
                continue
            raw_path = artifact.get("path")
            if isinstance(raw_path, str) and raw_path.strip():
                artifact["path"] = _contract_relative_or_absolute(
                    contract_root,
                    _resolve_contract_path(raw_path, contract_root=contract_root),
                )


def _matrix_report(
    summary_path: Path,
    rows: Iterable[dict[str, Any]],
    provider_reports: dict[str, list[str]],
    errors: tuple[str, ...],
    *,
    expected_benchmark_id: str | None = None,
    expected_release_split: str | None = None,
) -> dict[str, Any]:
    required = set(REQUIRED_REAL_SYSTEM_PROVIDERS)
    row_list = list(rows)
    contract_root = summary_path.parent.resolve()
    return {
        "schema_version": REAL_SYSTEM_EVIDENCE_SCHEMA_VERSION,
        "matrix_summary": str(summary_path),
        "root": _artifact_root_ref(summary_path.parent, contract_root),
        "expected_benchmark_id": expected_benchmark_id,
        "expected_release_split": expected_release_split,
        "status": "passed" if not errors else "incomplete",
        "num_systems": len(row_list),
        "required_providers": sorted(required),
        "covered_providers": sorted(provider_reports),
        "missing_providers": sorted(required - set(provider_reports)),
        "systems": row_list,
        "errors": list(errors),
    }


def _contract_root_from_summary(summary_path: Path, summary: dict[str, Any]) -> Path:
    candidates: list[Path] = [summary_path, summary_path.parent]
    raw_root = summary.get("root")
    if isinstance(raw_root, str) and raw_root.strip():
        candidates.append(_resolve_contract_path(raw_root, anchor=summary_path.parent))
    release_manifest = summary.get("release_manifest")
    if isinstance(release_manifest, str) and release_manifest.strip():
        candidates.append(_resolve_contract_path(release_manifest, anchor=summary_path.parent))
    return _infer_contract_root(candidates)


def _resolve_report_path(summary_path: Path, raw_path: str, *, contract_root: Path | None = None) -> Path:
    return _resolve_contract_path(raw_path, contract_root=contract_root, anchor=summary_path.parent)


def _real_system_report_paths_from_summary(
    summary_path: Path,
    *,
    expected_benchmark_id: str | None = None,
    expected_release_split: str | None = None,
) -> tuple[Path, ...]:
    validation = validate_real_system_matrix_summary(
        summary_path,
        expected_benchmark_id=expected_benchmark_id,
        expected_release_split=expected_release_split,
    )
    rows = validation.get("systems", ())
    if not isinstance(rows, list):
        raise ValueError(f"{summary_path}: matrix validation did not return system rows")
    report_paths = [
        Path(str(row["report_path"]))
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("report_path"), str) and row.get("report_path")
    ]
    if not report_paths:
        raise ValueError(f"{summary_path}: no report paths were resolved from the matrix summary")
    return tuple(report_paths)


def _read_report_object(path: Path) -> dict[str, Any]:
    try:
        report = read_json(path)
    except Exception:
        return {}
    return report if isinstance(report, dict) else {}


def _read_summary_object(path: Path) -> dict[str, Any]:
    try:
        summary = read_json(path)
    except Exception as exc:
        raise ValueError(f"{path}: cannot read matrix summary: {exc}") from exc
    if not isinstance(summary, dict):
        raise ValueError(f"{path}: matrix summary must be a JSON object")
    return summary


def _real_system_ci_present(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("lower") is not None
        and value.get("mean") is not None
        and value.get("upper") is not None
        and _positive_int(value.get("num_observations"))
    )


def _positive_int(value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat().replace("+00:00", "Z")


def _refresh_row_status(errors: list[str], progress: dict[str, Any], summary_exists: bool) -> str:
    if errors:
        return "invalid"
    progress_status = str(progress.get("status"))
    if summary_exists and progress_status == "completed":
        return "materialized"
    if progress_status == "predictions_complete_report_missing":
        return "ready_to_finalize"
    if progress_status == "running_or_partial":
        return "running"
    if progress_status == "completed":
        return "completed"
    if progress_status == "not_started":
        return "not_started"
    if progress_status == "running_with_errors":
        return "invalid"
    return progress_status or "unknown"


def _completed_run_artifacts_need_repair(progress: dict[str, Any], summary_path: Path) -> bool:
    report_path = Path(str(progress.get("report_path") or ""))
    if progress.get("status") == "invalid" and report_path.exists():
        return True
    if progress.get("status") == "completed" and summary_path.exists():
        try:
            _read_summary_object(summary_path)
        except ValueError:
            return True
    return False


def _project_refresh_progress_fields(row: dict[str, Any], progress: dict[str, Any]) -> None:
    for key in (
        "system_id",
        "benchmark_id",
        "release_split",
        "num_predictions",
        "total_queries",
        "progress_ratio",
        "has_predictions",
        "has_report",
        "has_run_metadata",
        "has_run_state",
        "predictions_path",
        "report_path",
        "run_metadata_path",
        "run_state_path",
        "checkpoint_modified_at",
        "run_state_modified_at",
    ):
        row[key] = progress.get(key)
    row["live_state"] = dict(progress.get("live_state") or {})


def _default_run_summary_path(run_dir: Path) -> Path:
    parent_default = run_dir.parent / "matrix_summary.json"
    if parent_default.exists():
        return parent_default
    return run_dir.parent / f"{run_dir.name}_matrix_summary.json"


def _default_real_system_matrix_command(
    manifest_path: Path,
    *,
    split: str,
    config_path: Path,
    output_dir: Path,
    system_version: str,
    retrieval_k: int,
    resume: bool,
) -> str:
    command = (
        "python -m agent_memory_benchmark run-release-agent-matrix "
        f"--manifest {manifest_path} --split {split} --configs {config_path} "
        f"--output-dir {output_dir} --system-version {system_version} --retrieval-k {retrieval_k}"
    )
    if resume:
        command = f"{command} --resume"
    return command
