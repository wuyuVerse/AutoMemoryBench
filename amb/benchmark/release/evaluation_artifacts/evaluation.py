"""Evaluate baselines over release split shards."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
import time
from typing import Any, Callable

from amb.benchmark.evaluation.adapters import BlackBoxAgent
from amb.benchmark.evaluation.framework_trace import (
    framework_trace_artifact_payload,
    framework_trace_record,
    load_default_tool_runtime_contracts,
    validate_framework_trace_artifact_payload,
    validate_framework_trace_records,
)
from amb.benchmark.evaluation.baselines import make_baseline
from amb.benchmark.evaluation.agent_systems import bind_agent_system_metadata
from amb.benchmark.evaluation.runner import run_case_with_agent, write_prediction_checkpoint, write_run_state_checkpoint
from amb.benchmark.evaluation.scoring import (
    DEFAULT_RETRIEVAL_K,
    Scorer,
    aggregate_by,
    aggregate_reports,
    counterfactual_report,
)
from amb.benchmark.metrics.task_judges import load_task_judge_plugin
from amb.benchmark.quality.run_metadata import build_run_metadata, load_run_metadata, run_metadata_artifact, validate_run_metadata
from amb.benchmark.quality.validation import validate_benchmark, validate_predictions
from amb.benchmark.release.artifacts import artifact_info
from amb.benchmark.release.fingerprint import release_split_contract_fingerprint
from amb.benchmark.release.splits import RELEASE_SPLITS
from amb.benchmark.schemas.io import load_benchmark, load_predictions, read_json, write_json
from amb.benchmark.schemas.models import Benchmark, PredictionSet, QueryPrediction, SCHEMA_VERSION


def run_release_split_agent(
    manifest_path: str | Path,
    *,
    split: str,
    agent: BlackBoxAgent,
    system_id: str,
    resume_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    state_path: str | Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    framework_trace_callback: Callable[[dict[str, Any]], None] | None = None,
    emit_resume_traces: bool = True,
    agent_system: dict[str, Any] | None = None,
    case_ids: set[str] | None = None,
    domains: set[str] | None = None,
    failure_predictions: bool = False,
) -> PredictionSet:
    """Run a streaming-system-memory agent over every shard in a release split."""

    if split not in RELEASE_SPLITS:
        raise ValueError(f"unknown release split {split!r}")
    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    if split in manifest.get("withheld_splits", {}):
        visibility = manifest.get("visibility", {}).get(split, "withheld")
        raise ValueError(f"split {split!r} is withheld in this package ({visibility}); use the private leaderboard package")
    entries = _split_entries(manifest.get("split_files", {}).get(split))
    if not entries:
        raise ValueError(f"manifest has no benchmark artifacts for split {split!r}")
    selected_benchmarks: list[tuple[str, Benchmark]] = []
    total_cases = 0
    total_queries = 0
    for label, raw_path in entries:
        path = _resolve_path(raw_path, manifest_file.parent)
        benchmark = load_benchmark(path)
        benchmark = _filter_benchmark(benchmark, case_ids=case_ids, domains=domains)
        if not benchmark.cases:
            continue
        benchmark_result = validate_benchmark(benchmark)
        if benchmark_result.errors:
            raise ValueError(f"{split}/{label} benchmark validation failed: {benchmark_result.errors}")
        selected_benchmarks.append((label, benchmark))
        total_cases += len(benchmark.cases)
        total_queries += sum(len(case.queries) for case in benchmark.cases)
    if not selected_benchmarks:
        raise ValueError(f"no cases selected for split {split!r}")
    if case_ids is not None or domains is not None:
        missing_case_ids = sorted(case_ids - {case.case_id for _, benchmark in selected_benchmarks for case in benchmark.cases}) if case_ids else []
        if missing_case_ids:
            raise ValueError(f"case_ids not found in split {split!r}: {', '.join(missing_case_ids[:20])}")
    else:
        split_report = manifest.get("split_reports", {}).get(split, {})
        total_cases = int(split_report.get("num_cases", 0) or 0)
        total_queries = int(split_report.get("num_queries", 0) or 0)

    prediction_by_query: dict[str, QueryPrediction] = {}
    if resume_path is not None and Path(resume_path).exists():
        existing = load_predictions(resume_path)
        if existing.system_id != system_id:
            raise ValueError(
                f"resume predictions system_id mismatch: expected {system_id!r}, got {existing.system_id!r}"
            )
        prediction_by_query = _prediction_map(existing)
    seen_query_ids: set[str] = set()
    query_order: list[str] = []
    completed_cases = 0
    for label, benchmark in selected_benchmarks:
        for case in benchmark.cases:
            case_query_ids = [query.query_id for query in case.queries]
            for query_id in case_query_ids:
                if query_id in seen_query_ids:
                    raise ValueError(f"duplicate prediction query_id across release shards: {query_id}")
                seen_query_ids.add(query_id)
                query_order.append(query_id)
            if case_query_ids and all(query_id in prediction_by_query for query_id in case_query_ids):
                if framework_trace_callback is not None and emit_resume_traces:
                    for query_id in case_query_ids:
                        framework_trace_callback(
                            framework_trace_record(
                                agent_system=agent_system or {},
                                case_id=case.case_id,
                                query_id=query_id,
                                raw_response=_raw_response_from_prediction(prediction_by_query[query_id]),
                            )
                        )
                _write_live_run_state(
                    state_path,
                    system_id=system_id,
                    split=split,
                    case_id=case.case_id,
                    completed_cases=completed_cases + 1,
                    total_cases=total_cases,
                    num_predictions=len(prediction_by_query),
                    total_queries=total_queries,
                    status="running",
                    phase="resume_skip",
                    event_type="resume_skip_case",
                )
                completed_cases += 1
                _emit_progress(
                    progress_callback,
                    system_id=system_id,
                    case_id=case.case_id,
                    completed_cases=completed_cases,
                    total_cases=total_cases,
                    num_predictions=len(prediction_by_query),
                    total_queries=total_queries,
                    resumed=True,
                )
                continue
            for query_id in case_query_ids:
                prediction_by_query.pop(query_id, None)
            _write_live_run_state(
                state_path,
                system_id=system_id,
                split=split,
                case_id=case.case_id,
                completed_cases=completed_cases,
                total_cases=total_cases,
                num_predictions=len(prediction_by_query),
                total_queries=total_queries,
                status="running",
                phase="case_started",
                event_type="case_start",
            )

            def _record_case_prediction(prediction: QueryPrediction) -> None:
                prediction_by_query[prediction.query_id] = prediction
                if checkpoint_path is not None:
                    write_prediction_checkpoint(
                        checkpoint_path,
                        system_id=system_id,
                        predictions=_ordered_predictions(prediction_by_query, query_order),
                    )

            case_predictions = run_case_with_agent(
                case,
                agent,
                framework_trace_callback=framework_trace_callback,
                agent_system=agent_system,
                state_callback=lambda item: _write_live_run_state(
                    state_path,
                    system_id=system_id,
                    split=split,
                    case_id=str(item.get("case_id") or case.case_id),
                    completed_cases=completed_cases,
                    total_cases=total_cases,
                    num_predictions=len(prediction_by_query),
                    total_queries=total_queries,
                    status="running",
                    **{key: value for key, value in item.items() if key != "case_id"},
                ),
                failure_predictions=failure_predictions,
                prediction_callback=_record_case_prediction,
            )
            for prediction in case_predictions:
                prediction_by_query[prediction.query_id] = prediction
            if checkpoint_path is not None:
                write_prediction_checkpoint(
                    checkpoint_path,
                    system_id=system_id,
                    predictions=_ordered_predictions(prediction_by_query, query_order),
                )
            _write_live_run_state(
                state_path,
                system_id=system_id,
                split=split,
                case_id=case.case_id,
                completed_cases=completed_cases + 1,
                total_cases=total_cases,
                num_predictions=len(prediction_by_query),
                total_queries=total_queries,
                status="running",
                phase="case_completed",
                event_type="case_finish",
            )
            completed_cases += 1
            _emit_progress(
                progress_callback,
                system_id=system_id,
                case_id=case.case_id,
                completed_cases=completed_cases,
                total_cases=total_cases,
                num_predictions=len(prediction_by_query),
                total_queries=total_queries,
                resumed=False,
            )
        shard_predictions = PredictionSet(
            schema_version=SCHEMA_VERSION,
            system_id=system_id,
            predictions=tuple(
                prediction_by_query[query_id]
                for case in benchmark.cases
                for query_id in [query.query_id for query in case.queries]
                if query_id in prediction_by_query
            ),
        )
        prediction_result = validate_predictions(shard_predictions, benchmark)
        if prediction_result.errors:
            raise ValueError(f"{split}/{label} prediction validation failed: {prediction_result.errors}")
    _write_live_run_state(
        state_path,
        system_id=system_id,
        split=split,
        case_id=None,
        completed_cases=completed_cases,
        total_cases=total_cases,
        num_predictions=len(prediction_by_query),
        total_queries=total_queries,
        status="completed",
        phase="completed",
        event_type="run_completed",
    )
    return PredictionSet(
        schema_version=SCHEMA_VERSION,
        system_id=system_id,
        predictions=tuple(_ordered_predictions(prediction_by_query, query_order)),
    )


def run_release_split_agent_with_retries(
    manifest_path: str | Path,
    *,
    split: str,
    agent_factory: Callable[[], BlackBoxAgent],
    system_id: str,
    resume_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    state_path: str | Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    retry_callback: Callable[[dict[str, Any]], None] | None = None,
    framework_trace_callback: Callable[[dict[str, Any]], None] | None = None,
    emit_resume_traces: bool = True,
    agent_system: dict[str, Any] | None = None,
    max_run_attempts: int = 1,
    retry_backoff_s: float = 0.0,
    case_ids: set[str] | None = None,
    domains: set[str] | None = None,
    failure_predictions: bool = False,
) -> tuple[PredictionSet, int]:
    """Run a release split with fresh agents across attempts and resume from checkpoints when available."""

    if max_run_attempts <= 0:
        raise ValueError("max_run_attempts must be positive")
    if retry_backoff_s < 0:
        raise ValueError("retry_backoff_s must be non-negative")

    last_error: Exception | None = None
    for attempt in range(1, max_run_attempts + 1):
        active_resume_path = resume_path if attempt == 1 else (checkpoint_path or resume_path)
        try:
            predictions = run_release_split_agent(
                manifest_path,
                split=split,
                agent=agent_factory(),
                system_id=system_id,
                resume_path=active_resume_path,
                checkpoint_path=checkpoint_path,
                state_path=state_path,
                progress_callback=progress_callback,
                framework_trace_callback=framework_trace_callback,
                emit_resume_traces=emit_resume_traces,
                agent_system=agent_system,
                case_ids=case_ids,
                domains=domains,
                failure_predictions=failure_predictions,
            )
            return predictions, attempt
        except Exception as exc:
            last_error = exc
            will_retry = attempt < max_run_attempts
            if retry_callback is not None:
                retry_callback(
                    {
                        "system_id": system_id,
                        "attempt": attempt,
                        "max_run_attempts": max_run_attempts,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "will_retry": will_retry,
                    }
                )
            if not will_retry:
                raise
            if retry_backoff_s > 0:
                time.sleep(retry_backoff_s)
    assert last_error is not None
    raise last_error


def release_split_benchmark_id(manifest_path: str | Path, split: str) -> str:
    manifest = read_json(manifest_path)
    return f"{manifest.get('benchmark_id', 'release')}-{split}"


def evaluate_release_split_baselines(
    manifest_path: str | Path,
    *,
    split: str,
    baseline_kinds: tuple[str, ...] | list[str],
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    baseline_top_k: int | None = None,
    task_judge_plugin: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Run multiple deterministic baselines over one split with one shard load/validation pass."""

    requested_kinds = tuple(dict.fromkeys(str(kind) for kind in baseline_kinds))
    if not requested_kinds:
        raise ValueError("baseline_kinds must contain at least one baseline")

    manifest_file, manifest, prepared_shards = _load_release_split_benchmarks(manifest_path, split=split)
    scorer = Scorer(
        retrieval_k=retrieval_k,
        task_judge_plugin=load_task_judge_plugin(task_judge_plugin),
    )
    query_reports_by_kind: dict[str, list[dict[str, Any]]] = {kind: [] for kind in requested_kinds}
    shard_reports_by_kind: dict[str, dict[str, Any]] = {kind: {} for kind in requested_kinds}
    missing_predictions_by_kind: dict[str, list[str]] = {kind: [] for kind in requested_kinds}
    extra_predictions_by_kind: dict[str, list[str]] = {kind: [] for kind in requested_kinds}

    for label, path, benchmark in prepared_shards:
        for kind in requested_kinds:
            predictions = make_baseline(benchmark, kind, top_k=baseline_top_k)
            prediction_result = validate_predictions(predictions, benchmark)
            if prediction_result.errors:
                raise ValueError(f"{split}/{label} prediction validation failed: {prediction_result.errors}")
            shard_report = scorer.score(benchmark, predictions)
            shard_reports_by_kind[kind][label] = {
                "benchmark_id": benchmark.benchmark_id,
                "path": str(path),
                "num_cases": len(benchmark.cases),
                "num_scored_queries": shard_report["aggregate"].get("num_scored_queries", 0),
                "aggregate": shard_report["aggregate"],
            }
            query_reports_by_kind[kind].extend(shard_report["queries"])
            missing_predictions_by_kind[kind].extend(shard_report.get("missing_predictions", []))
            extra_predictions_by_kind[kind].extend(shard_report.get("extra_predictions", []))

    benchmark_id = f"{manifest.get('benchmark_id', 'release')}-{split}"
    release_contract = release_split_contract_fingerprint(manifest_file, split)
    task_judge_metadata = scorer.task_judge_plugin.report_metadata()
    scoring_config = _scoring_config(retrieval_k=retrieval_k, task_judge_metadata=task_judge_metadata)
    return {
        kind: {
            "schema_version": manifest.get("schema_version", "1.0.0"),
            "benchmark_id": benchmark_id,
            "release_manifest": str(manifest_file),
            "release_split": split,
            "release_contract_fingerprint": release_contract,
            "system_id": kind,
            "scoring_config": scoring_config,
            "baseline_config": {"top_k": baseline_top_k},
            "aggregate": aggregate_reports(query_reports_by_kind[kind]),
            "task_judge": _summarize_task_judge_metadata(
                query_reports_by_kind[kind],
                task_judge_metadata,
            ),
            "counterfactual": counterfactual_report(query_reports_by_kind[kind]),
            "by_task_type": aggregate_by(query_reports_by_kind[kind], "task_type"),
            "by_probe_type": aggregate_by(query_reports_by_kind[kind], "probe_type"),
            "by_domain": aggregate_by(query_reports_by_kind[kind], "domain"),
            "by_difficulty": aggregate_by(query_reports_by_kind[kind], "difficulty_level"),
            "by_memory_requirement": aggregate_by(query_reports_by_kind[kind], "memory_requirement"),
            "shards": shard_reports_by_kind[kind],
            "queries": query_reports_by_kind[kind],
            "missing_predictions": sorted(missing_predictions_by_kind[kind]),
            "extra_predictions": sorted(extra_predictions_by_kind[kind]),
            "duplicate_predictions": [],
        }
        for kind in requested_kinds
    }


def run_release_split_agent_experiment(
    manifest_path: str | Path,
    *,
    split: str,
    agent: BlackBoxAgent,
    system_id: str,
    output_dir: str | Path,
    system_version: str = "unspecified",
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    task_judge_plugin: str | None = None,
    command: str = "",
    dependencies: dict[str, Any] | None = None,
    execution_mode: str = "integration_smoke",
    integration_config_artifact: dict[str, Any] | None = None,
    real_system_attestation: dict[str, Any] | None = None,
    agent_system: dict[str, Any] | None = None,
    framework_trace_path: str | Path | None = None,
    resume: bool = False,
    checkpoint: bool | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run, evaluate, and package one external-agent release-split experiment."""

    output = Path(output_dir) / _safe_name(system_id)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.json"
    metadata_path = output / "run_metadata.json"
    report_path = output / "report.json"
    resolved_command = command or _default_release_agent_matrix_command(
        manifest_path,
        split=split,
        output_dir=output_dir,
        resume=resume or bool(checkpoint),
    )
    bound_agent_system = bind_agent_system_metadata(
        agent_system,
        system_id=system_id,
        system_version=system_version,
        execution_mode=execution_mode,
    )
    metadata = build_run_metadata(
        system_id=system_id,
        system_version=system_version,
        benchmark_id=release_split_benchmark_id(manifest_path, split),
        release_split=split,
        command=resolved_command,
        dependencies=dependencies,
        execution_mode=execution_mode,
        integration_config_artifact=integration_config_artifact,
        real_system_attestation=real_system_attestation,
        release_contract_fingerprint=release_split_contract_fingerprint(manifest_path, split),
        agent_system=bound_agent_system,
    )
    release_contract = metadata.get("release_contract_fingerprint")
    write_json(metadata_path, metadata)

    checkpoint_enabled = resume if checkpoint is None else checkpoint
    framework_traces: list[dict[str, Any]] = []
    predictions = run_release_split_agent(
        manifest_path,
        split=split,
        agent=agent,
        system_id=system_id,
        resume_path=prediction_path if resume else None,
        checkpoint_path=prediction_path if checkpoint_enabled else None,
        state_path=output / "run_state.json",
        progress_callback=progress_callback,
        framework_trace_callback=framework_traces.append if framework_trace_path is not None else None,
        agent_system=bound_agent_system,
    )
    write_json(prediction_path, asdict(predictions))
    resolved_framework_trace_path = Path(framework_trace_path) if framework_trace_path is not None else None
    if resolved_framework_trace_path is not None:
        _write_framework_trace_artifact(
            resolved_framework_trace_path,
            framework_traces=framework_traces,
            expected_records=len(predictions.predictions),
            system_id=system_id,
            benchmark_id=str(metadata["benchmark_id"]),
            release_split=split,
            run_id=str(metadata["timestamp"]),
            release_contract_fingerprint=release_contract,
        )
    return _package_release_split_agent_experiment(
        manifest_path,
        split=split,
        prediction_path=prediction_path,
        metadata_path=metadata_path,
        report_path=report_path,
        framework_trace_path=resolved_framework_trace_path,
        system_id=system_id,
        system_version=system_version,
        retrieval_k=retrieval_k,
        task_judge_plugin=task_judge_plugin,
    )


def run_release_split_agent_experiment_with_retries(
    manifest_path: str | Path,
    *,
    split: str,
    agent_factory: Callable[[], BlackBoxAgent],
    system_id: str,
    output_dir: str | Path,
    system_version: str = "unspecified",
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    task_judge_plugin: str | None = None,
    command: str = "",
    dependencies: dict[str, Any] | None = None,
    execution_mode: str = "integration_smoke",
    integration_config_artifact: dict[str, Any] | None = None,
    real_system_attestation: dict[str, Any] | None = None,
    agent_system: dict[str, Any] | None = None,
    framework_trace_path: str | Path | None = None,
    resume: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    retry_callback: Callable[[dict[str, Any]], None] | None = None,
    max_run_attempts: int = 1,
    retry_backoff_s: float = 0.0,
) -> dict[str, Any]:
    """Run, evaluate, and package one release-split experiment with automatic fresh-agent retries."""

    output = Path(output_dir) / _safe_name(system_id)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.json"
    metadata_path = output / "run_metadata.json"
    report_path = output / "report.json"
    resolved_command = command or _default_release_agent_matrix_command(
        manifest_path,
        split=split,
        output_dir=output_dir,
        resume=resume or max_run_attempts > 1,
        max_run_attempts=max_run_attempts,
        retry_backoff_s=retry_backoff_s,
    )
    bound_agent_system = bind_agent_system_metadata(
        agent_system,
        system_id=system_id,
        system_version=system_version,
        execution_mode=execution_mode,
    )
    metadata = build_run_metadata(
        system_id=system_id,
        system_version=system_version,
        benchmark_id=release_split_benchmark_id(manifest_path, split),
        release_split=split,
        command=resolved_command,
        dependencies=dependencies,
        execution_mode=execution_mode,
        integration_config_artifact=integration_config_artifact,
        real_system_attestation=real_system_attestation,
        release_contract_fingerprint=release_split_contract_fingerprint(manifest_path, split),
        agent_system=bound_agent_system,
    )
    release_contract = metadata.get("release_contract_fingerprint")
    write_json(metadata_path, metadata)

    framework_traces: list[dict[str, Any]] = []
    predictions, attempt_count = run_release_split_agent_with_retries(
        manifest_path,
        split=split,
        agent_factory=agent_factory,
        system_id=system_id,
        resume_path=prediction_path if resume else None,
        checkpoint_path=prediction_path if (resume or max_run_attempts > 1) else None,
        state_path=output / "run_state.json",
        progress_callback=progress_callback,
        retry_callback=retry_callback,
        framework_trace_callback=framework_traces.append if framework_trace_path is not None else None,
        agent_system=bound_agent_system,
        max_run_attempts=max_run_attempts,
        retry_backoff_s=retry_backoff_s,
    )
    write_json(prediction_path, asdict(predictions))
    resolved_framework_trace_path = Path(framework_trace_path) if framework_trace_path is not None else None
    if resolved_framework_trace_path is not None:
        _write_framework_trace_artifact(
            resolved_framework_trace_path,
            framework_traces=framework_traces,
            expected_records=len(predictions.predictions),
            system_id=system_id,
            benchmark_id=str(metadata["benchmark_id"]),
            release_split=split,
            run_id=str(metadata["timestamp"]),
            release_contract_fingerprint=release_contract,
        )
    result = _package_release_split_agent_experiment(
        manifest_path,
        split=split,
        prediction_path=prediction_path,
        metadata_path=metadata_path,
        report_path=report_path,
        framework_trace_path=resolved_framework_trace_path,
        system_id=system_id,
        system_version=system_version,
        retrieval_k=retrieval_k,
        task_judge_plugin=task_judge_plugin,
    )
    result["execution_attempts"] = attempt_count
    return result


def _package_release_split_agent_experiment(
    manifest_path: str | Path,
    *,
    split: str,
    prediction_path: Path,
    metadata_path: Path,
    report_path: Path,
    framework_trace_path: Path | None = None,
    system_id: str,
    system_version: str,
    retrieval_k: int,
    task_judge_plugin: str | None = None,
) -> dict[str, Any]:
    predictions = load_predictions(prediction_path)
    report = evaluate_release_split_predictions(
        manifest_path,
        prediction_path,
        split=split,
        retrieval_k=retrieval_k,
        task_judge_plugin=task_judge_plugin,
        run_metadata_path=metadata_path,
        framework_trace_path=framework_trace_path,
    )
    if framework_trace_path is not None:
        submission = report.setdefault("submission", {})
        if isinstance(submission, dict):
            submission["framework_trace_artifact"] = artifact_info(framework_trace_path)
    write_json(report_path, report)
    result = {
        "system_id": system_id,
        "system_version": system_version,
        "prediction_artifact": artifact_info(prediction_path),
        "run_metadata_artifact": artifact_info(metadata_path),
        "report_artifact": artifact_info(report_path),
        "aggregate": report["aggregate"],
        "num_predictions": len(predictions.predictions),
        "missing_predictions": report.get("missing_predictions", []),
        "extra_predictions": report.get("extra_predictions", []),
    }
    if framework_trace_path is not None:
        result["framework_trace_artifact"] = artifact_info(framework_trace_path)
    return result


def _write_framework_trace_artifact(
    path: Path,
    *,
    framework_traces: list[dict[str, Any]],
    expected_records: int,
    system_id: str,
    benchmark_id: str,
    release_split: str,
    run_id: str,
    release_contract_fingerprint: str | dict[str, Any] | None,
) -> None:
    if len(framework_traces) != expected_records:
        raise ValueError(
            f"framework trace record count mismatch: expected {expected_records}, got {len(framework_traces)}"
        )
    trace_errors = validate_framework_trace_records(
        framework_traces,
        tool_runtime_contracts=load_default_tool_runtime_contracts(),
    )
    if trace_errors:
        raise ValueError(f"framework trace validation failed: {trace_errors}")
    payload = framework_trace_artifact_payload(
        framework_traces,
        system_id=system_id,
        benchmark_id=benchmark_id,
        release_split=release_split,
        run_id=run_id,
        release_contract_fingerprint=release_contract_fingerprint,
    )
    artifact_errors = validate_framework_trace_artifact_payload(
        payload,
        expected_records=expected_records,
        require_envelope=True,
        tool_runtime_contracts=load_default_tool_runtime_contracts(),
    )
    if artifact_errors:
        raise ValueError(f"framework trace artifact envelope validation failed: {artifact_errors}")
    write_json(path, payload)


def evaluate_release_split_baseline(
    manifest_path: str | Path,
    *,
    split: str,
    baseline_kind: str,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    baseline_top_k: int | None = None,
    task_judge_plugin: str | None = None,
) -> dict[str, Any]:
    """Run a deterministic baseline over every benchmark shard in one split."""
    return evaluate_release_split_baselines(
        manifest_path,
        split=split,
        baseline_kinds=(baseline_kind,),
        retrieval_k=retrieval_k,
        baseline_top_k=baseline_top_k,
        task_judge_plugin=task_judge_plugin,
    )[baseline_kind]


def evaluate_release_split_predictions(
    manifest_path: str | Path,
    predictions_path: str | Path,
    *,
    split: str,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    run_metadata_path: str | Path | None = None,
    framework_trace_path: str | Path | None = None,
    task_judge_plugin: str | None = None,
    case_ids: set[str] | None = None,
    domains: set[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a submitted prediction file over every benchmark shard in one split."""

    if split not in RELEASE_SPLITS:
        raise ValueError(f"unknown release split {split!r}")
    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    if split in manifest.get("withheld_splits", {}):
        visibility = manifest.get("visibility", {}).get(split, "withheld")
        raise ValueError(f"split {split!r} is withheld in this package ({visibility}); use the private leaderboard package")
    entries = _split_entries(manifest.get("split_files", {}).get(split))
    if not entries:
        raise ValueError(f"manifest has no benchmark artifacts for split {split!r}")

    predictions_file = Path(predictions_path)
    predictions = load_predictions(predictions_file)
    run_metadata = load_run_metadata(run_metadata_path) if run_metadata_path is not None else None
    prediction_by_query = _prediction_map(predictions)
    scorer = Scorer(
        retrieval_k=retrieval_k,
        task_judge_plugin=load_task_judge_plugin(task_judge_plugin),
    )
    query_reports: list[dict[str, Any]] = []
    shard_reports: dict[str, Any] = {}
    missing_predictions: list[str] = []
    seen_query_ids: set[str] = set()
    selected_case_ids: set[str] = set()

    for label, raw_path in entries:
        path = _resolve_path(raw_path, manifest_file.parent)
        benchmark = load_benchmark(path)
        benchmark = _filter_benchmark(benchmark, case_ids=case_ids, domains=domains)
        if not benchmark.cases:
            continue
        selected_case_ids.update(case.case_id for case in benchmark.cases)
        benchmark_result = validate_benchmark(benchmark)
        if benchmark_result.errors:
            raise ValueError(f"{split}/{label} benchmark validation failed: {benchmark_result.errors}")
        query_ids = {query.query_id for case in benchmark.cases for query in case.queries}
        seen_query_ids.update(query_ids)
        shard_predictions = PredictionSet(
            schema_version=predictions.schema_version,
            system_id=predictions.system_id,
            predictions=tuple(prediction_by_query[query_id] for query_id in sorted(query_ids & prediction_by_query.keys())),
        )
        prediction_result = validate_predictions(shard_predictions, benchmark)
        if prediction_result.errors:
            raise ValueError(f"{split}/{label} prediction validation failed: {prediction_result.errors}")
        shard_report = scorer.score(benchmark, shard_predictions)
        shard_reports[label] = {
            "benchmark_id": benchmark.benchmark_id,
            "path": str(path),
            "num_cases": len(benchmark.cases),
            "num_scored_queries": shard_report["aggregate"].get("num_scored_queries", 0),
            "num_missing_predictions": len(shard_report.get("missing_predictions", [])),
            "aggregate": shard_report["aggregate"],
        }
        query_reports.extend(shard_report["queries"])
        missing_predictions.extend(shard_report.get("missing_predictions", []))
    if case_ids is not None:
        missing_case_ids = sorted(case_ids - selected_case_ids)
        if missing_case_ids:
            raise ValueError(f"case_ids not found in split {split!r}: {', '.join(missing_case_ids[:20])}")
    if not selected_case_ids:
        raise ValueError(f"no cases selected for split {split!r}")

    extra_predictions = sorted(set(prediction_by_query) - seen_query_ids)
    benchmark_id = f"{manifest.get('benchmark_id', 'release')}-{split}"
    if run_metadata is not None:
        metadata_errors = validate_run_metadata(
            run_metadata,
            system_id=predictions.system_id,
            benchmark_id=benchmark_id,
            release_split=split,
        )
        if metadata_errors:
            raise ValueError(f"run metadata validation failed: {metadata_errors}")
    if framework_trace_path is not None:
        framework_trace_file = Path(framework_trace_path)
        framework_trace = read_json(framework_trace_file)
        trace_errors = list(
            validate_framework_trace_artifact_payload(
                framework_trace,
                expected_records=len(predictions.predictions),
                require_envelope=True,
                tool_runtime_contracts=load_default_tool_runtime_contracts(),
            )
        )
        release_contract = release_split_contract_fingerprint(manifest_file, split)
        expected_contract = str(release_contract.get("query_contract_sha256") or "")
        trace_errors.extend(
            _validate_framework_trace_submission_binding(
                framework_trace,
                system_id=predictions.system_id,
                benchmark_id=benchmark_id,
                release_split=split,
                release_contract_fingerprint=expected_contract,
                prediction_query_ids=[prediction.query_id for prediction in predictions.predictions],
                run_metadata=run_metadata,
            )
        )
        if trace_errors:
            raise ValueError(f"framework trace artifact validation failed: {trace_errors}")
    task_judge_metadata = scorer.task_judge_plugin.report_metadata()
    scoring_config = _scoring_config(retrieval_k=retrieval_k, task_judge_metadata=task_judge_metadata)

    return {
        "schema_version": manifest.get("schema_version", "1.0.0"),
        "benchmark_id": benchmark_id,
        "release_manifest": str(manifest_file),
        "release_split": split,
        "system_id": predictions.system_id,
        "scoring_config": scoring_config,
        "submission": {
            "prediction_file": str(predictions_file),
            "prediction_artifact": artifact_info(predictions_file) if predictions_file.exists() else None,
            "manifest_artifact": artifact_info(manifest_file) if manifest_file.exists() else None,
            "run_metadata_file": str(run_metadata_path) if run_metadata_path is not None else None,
            "run_metadata_artifact": run_metadata_artifact(run_metadata_path) if run_metadata_path is not None else None,
            "framework_trace_file": str(framework_trace_path) if framework_trace_path is not None else None,
            "framework_trace_artifact": artifact_info(Path(framework_trace_path))
            if framework_trace_path is not None and Path(framework_trace_path).exists()
            else None,
            "scoring_config": scoring_config,
        },
        "run_metadata": run_metadata,
        "aggregate": aggregate_reports(query_reports),
        "task_judge": _summarize_task_judge_metadata(query_reports, task_judge_metadata),
        "counterfactual": counterfactual_report(query_reports),
        "by_task_type": aggregate_by(query_reports, "task_type"),
        "by_probe_type": aggregate_by(query_reports, "probe_type"),
        "by_domain": aggregate_by(query_reports, "domain"),
        "by_difficulty": aggregate_by(query_reports, "difficulty_level"),
        "by_memory_requirement": aggregate_by(query_reports, "memory_requirement"),
        "shards": shard_reports,
        "queries": query_reports,
        "missing_predictions": sorted(missing_predictions),
        "extra_predictions": extra_predictions,
        "duplicate_predictions": [],
    }


def _validate_framework_trace_submission_binding(
    payload: dict[str, Any],
    *,
    system_id: str,
    benchmark_id: str,
    release_split: str,
    release_contract_fingerprint: str,
    prediction_query_ids: list[str],
    run_metadata: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return ()
    checks = (
        ("system_id", system_id, "predictions.system_id"),
        ("benchmark_id", benchmark_id, "release benchmark_id"),
        ("release_split", release_split, "requested split"),
        (
            "release_contract_fingerprint",
            release_contract_fingerprint,
            "current release contract fingerprint",
        ),
    )
    errors: list[str] = []
    for field, expected, label in checks:
        actual = payload.get(field)
        if isinstance(expected, str) and expected and actual != expected:
            errors.append(
                f"framework_trace_artifact.{field} must match {label}: "
                f"expected {expected!r}, got {actual!r}"
            )
    if isinstance(run_metadata, dict):
        errors.extend(_validate_framework_trace_run_metadata_binding(payload, run_metadata))
    records = payload.get("framework_traces")
    if isinstance(records, list):
        trace_query_ids = [
            str(record.get("query_id"))
            for record in records
            if isinstance(record, dict) and isinstance(record.get("query_id"), str)
        ]
        expected_counts = Counter(prediction_query_ids)
        actual_counts = Counter(trace_query_ids)
        missing = sorted((expected_counts - actual_counts).elements())
        extra = sorted((actual_counts - expected_counts).elements())
        duplicate_trace_query_ids = sorted(
            query_id for query_id, count in actual_counts.items() if count > 1
        )
        if missing:
            errors.append(
                "framework_trace_artifact.framework_traces query_id missing predictions: "
                f"{missing[:10]}"
            )
        if extra:
            errors.append(
                "framework_trace_artifact.framework_traces query_id not present in predictions: "
                f"{extra[:10]}"
            )
        if duplicate_trace_query_ids:
            errors.append(
                "framework_trace_artifact.framework_traces query_id must be unique: "
                f"{duplicate_trace_query_ids[:10]}"
            )
        if (
            len(trace_query_ids) == len(prediction_query_ids)
            and not missing
            and not extra
            and trace_query_ids != prediction_query_ids
        ):
            errors.append(
                "framework_trace_artifact.framework_traces query_id order must match predictions order"
            )
    return tuple(errors)


def _validate_framework_trace_run_metadata_binding(
    payload: dict[str, Any],
    run_metadata: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    metadata_checks = (
        ("system_id", run_metadata.get("system_id"), "run_metadata.system_id"),
        ("benchmark_id", run_metadata.get("benchmark_id"), "run_metadata.benchmark_id"),
        ("release_split", run_metadata.get("release_split"), "run_metadata.release_split"),
        ("run_id", run_metadata.get("timestamp"), "run_metadata.timestamp"),
    )
    for field, expected, label in metadata_checks:
        actual = payload.get(field)
        if isinstance(expected, str) and expected and actual != expected:
            errors.append(
                f"framework_trace_artifact.{field} must match {label}: "
                f"expected {expected!r}, got {actual!r}"
            )
    release_contract = run_metadata.get("release_contract_fingerprint")
    if isinstance(release_contract, dict):
        expected_contract = str(release_contract.get("query_contract_sha256") or "")
        actual_contract = payload.get("release_contract_fingerprint")
        if expected_contract and actual_contract != expected_contract:
            errors.append(
                "framework_trace_artifact.release_contract_fingerprint must match "
                "run_metadata.release_contract_fingerprint.query_contract_sha256: "
                f"expected {expected_contract!r}, got {actual_contract!r}"
            )
    return errors


def _load_release_split_benchmarks(
    manifest_path: str | Path,
    *,
    split: str,
) -> tuple[Path, dict[str, Any], list[tuple[str, Path, Any]]]:
    if split not in RELEASE_SPLITS:
        raise ValueError(f"unknown release split {split!r}")
    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    if split in manifest.get("withheld_splits", {}):
        visibility = manifest.get("visibility", {}).get(split, "withheld")
        raise ValueError(f"split {split!r} is withheld in this package ({visibility}); use the private leaderboard package")
    entries = _split_entries(manifest.get("split_files", {}).get(split))
    if not entries:
        raise ValueError(f"manifest has no benchmark artifacts for split {split!r}")

    prepared_shards: list[tuple[str, Path, Any]] = []
    for label, raw_path in entries:
        path = _resolve_path(raw_path, manifest_file.parent)
        benchmark = load_benchmark(path)
        benchmark_result = validate_benchmark(benchmark)
        if benchmark_result.errors:
            raise ValueError(f"{split}/{label} benchmark validation failed: {benchmark_result.errors}")
        prepared_shards.append((label, path, benchmark))
    return manifest_file, manifest, prepared_shards


def _split_entries(value: Any) -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [("benchmark", value)]
    if isinstance(value, dict):
        return [(str(label), str(path)) for label, path in sorted(value.items())]
    return []


def _prediction_map(predictions: PredictionSet) -> dict[str, QueryPrediction]:
    prediction_by_query: dict[str, QueryPrediction] = {}
    duplicates: list[str] = []
    for prediction in predictions.predictions:
        if prediction.query_id in prediction_by_query:
            duplicates.append(prediction.query_id)
        prediction_by_query[prediction.query_id] = prediction
    if duplicates:
        raise ValueError(f"duplicate prediction query_id(s): {', '.join(sorted(duplicates))}")
    return prediction_by_query


def _filter_benchmark(
    benchmark: Benchmark,
    *,
    case_ids: set[str] | None,
    domains: set[str] | None,
) -> Benchmark:
    if case_ids is None and domains is None:
        return benchmark
    selected = []
    for case in benchmark.cases:
        if case_ids is not None and case.case_id not in case_ids:
            continue
        if domains is not None and case.domain not in domains:
            continue
        selected.append(case)
    return replace(benchmark, cases=tuple(selected))


def _raw_response_from_prediction(prediction: QueryPrediction) -> dict[str, Any]:
    cost = {key: value for key, value in asdict(prediction.cost).items() if value is not None}
    return {
        "memory_needed": prediction.memory_needed,
        "activated_memory_ids": list(prediction.activated_memory_ids),
        "response": prediction.response,
        "compression_summary": prediction.compression_summary,
        "tool_name": prediction.tool_name,
        "parameters": dict(prediction.parameters),
        "memory_operations": [
            {key: value for key, value in asdict(item).items() if value is not None}
            for item in prediction.memory_operations
        ],
        "cost": cost,
    }


def _resolve_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return manifest_dir / path


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value).strip("._") or "system"


def _summarize_task_judge_metadata(
    query_reports: list[dict[str, Any]],
    report_metadata: dict[str, Any],
) -> dict[str, Any]:
    query_metadata = [
        item.get("diagnostics", {}).get("task_judge")
        for item in query_reports
        if isinstance(item.get("diagnostics", {}).get("task_judge"), dict)
    ]
    plugin_ids = sorted(
        {
            str(item.get("plugin_id"))
            for item in query_metadata
            if isinstance(item, dict) and item.get("plugin_id")
        }
    )
    summary = dict(report_metadata)
    summary.update(
        {
            "num_scored_queries": len(query_reports),
            "num_queries_with_metadata": len(query_metadata),
            "all_queries_have_metadata": len(query_metadata) == len(query_reports),
            "all_queries_share_plugin": len(plugin_ids) == 1 and len(query_metadata) == len(query_reports),
            "query_plugin_ids": plugin_ids,
        }
    )
    return summary


def _scoring_config(*, retrieval_k: int, task_judge_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "retrieval_k": retrieval_k,
        "task_judge_plugin_id": task_judge_metadata.get("plugin_id"),
        "task_judge_plugin_kind": task_judge_metadata.get("plugin_kind"),
        "task_judge_rule_version": task_judge_metadata.get("rule_version"),
    }


def _default_release_agent_matrix_command(
    manifest_path: str | Path,
    *,
    split: str,
    output_dir: str | Path,
    resume: bool,
    max_run_attempts: int | None = None,
    retry_backoff_s: float | None = None,
) -> str:
    command = (
        "python -m agent_memory_benchmark run-release-agent-matrix "
        f"--manifest {manifest_path} --split {split} --output-dir {output_dir}"
    )
    if resume:
        command += " --resume"
    if max_run_attempts is not None:
        command += f" --max-run-attempts {max_run_attempts}"
    if retry_backoff_s is not None:
        command += f" --retry-backoff-s {retry_backoff_s}"
    return command


def _ordered_predictions(
    prediction_by_query: dict[str, QueryPrediction],
    query_order: list[str],
) -> list[QueryPrediction]:
    return [prediction_by_query[query_id] for query_id in query_order if query_id in prediction_by_query]


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    system_id: str,
    case_id: str,
    completed_cases: int,
    total_cases: int,
    num_predictions: int,
    total_queries: int,
    resumed: bool,
) -> None:
    if callback is None:
        return
    callback(
        {
            "system_id": system_id,
            "case_id": case_id,
            "completed_cases": completed_cases,
            "total_cases": total_cases,
            "num_predictions": num_predictions,
            "total_queries": total_queries,
            "resumed": resumed,
        }
    )


def _write_live_run_state(
    state_path: str | Path | None,
    *,
    system_id: str,
    split: str,
    case_id: str | None,
    completed_cases: int,
    total_cases: int,
    num_predictions: int,
    total_queries: int,
    status: str,
    phase: str,
    event_type: str,
    **extra: Any,
) -> None:
    if state_path is None:
        return
    payload = {
        "schema_version": "amst-real-system-run-state-v1",
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "system_id": system_id,
        "release_split": split,
        "status": status,
        "phase": phase,
        "event_type": event_type,
        "case_id": case_id,
        "completed_cases": completed_cases,
        "total_cases": total_cases,
        "num_predictions": num_predictions,
        "total_queries": total_queries,
    }
    payload.update({key: value for key, value in extra.items() if value is not None})
    write_run_state_checkpoint(state_path, payload)
