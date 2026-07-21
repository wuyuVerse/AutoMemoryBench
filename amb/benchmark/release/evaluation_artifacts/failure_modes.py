"""Release-level failure-mode diagnostics for memory baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from amb.benchmark.artifact_contract import localize_report_contract
from amb.benchmark.evaluation.scoring import DEFAULT_RETRIEVAL_K
from amb.benchmark.release.evaluation import evaluate_release_split_baseline
from amb.benchmark.schemas.io import read_json, write_json

FAILURE_MODE_BASELINES = (
    "dense_memory",
    "hybrid_memory",
    "full_history",
    "graph_memory",
    "oracle_memory",
)

FAILURE_MODE_PROBES = (
    "forget_probe",
    "governance_probe",
    "update_probe",
    "planning_probe",
    "tool_probe",
    "evolution_probe",
)

FAILURE_MODE_SCHEMA_VERSION = "amst-release-failure-modes-v1"


def build_release_failure_mode_diagnostics(
    manifest_path: str | Path,
    *,
    split: str = "public_dev",
    output_dir: str | Path = "reports/examples",
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    baseline_kinds: tuple[str, ...] = FAILURE_MODE_BASELINES,
    existing_report_paths: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    manifest_benchmark_id = str(manifest.get("benchmark_id", "release"))
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    prefix = _artifact_prefix(manifest_benchmark_id, split)
    report_paths: dict[str, str] = {}
    for kind, path in (existing_report_paths or {}).items():
        if kind in baseline_kinds:
            report_paths[kind] = str(path)
    for kind in baseline_kinds:
        if kind in report_paths and Path(report_paths[kind]).exists():
            continue
        report = evaluate_release_split_baseline(
            manifest_file,
            split=split,
            baseline_kind=kind,
            retrieval_k=retrieval_k,
        )
        path = output_root / f"{prefix}_{kind}_report.json"
        report = localize_report_contract(
            report,
            output_path=path,
            project_root_hints=(manifest_path,),
        )
        write_json(path, report)
        report_paths[kind] = str(path)

    diagnostics = summarize_failure_mode_reports(
        report_paths,
        benchmark_id=f"{manifest_benchmark_id}-{split}",
        release_split=split,
    )
    diagnostics_path = output_root / f"{prefix}_failure_mode_diagnostics.json"
    diagnostics = localize_report_contract(
        diagnostics,
        output_path=diagnostics_path,
        project_root_hints=tuple(report_paths.values()),
    )
    write_json(diagnostics_path, diagnostics)
    diagnostics["diagnostics_path"] = str(diagnostics_path)
    return diagnostics


def summarize_failure_mode_reports(
    report_paths: dict[str, str | Path],
    *,
    benchmark_id: str,
    release_split: str,
) -> dict[str, Any]:
    missing = [kind for kind in FAILURE_MODE_BASELINES if kind not in report_paths]
    reports = {kind: read_json(path) for kind, path in report_paths.items()}
    missing += [kind for kind in FAILURE_MODE_BASELINES if kind not in reports]
    if missing:
        return {
            "schema_version": FAILURE_MODE_SCHEMA_VERSION,
            "benchmark_id": benchmark_id,
            "release_split": release_split,
            "baseline_kinds": list(report_paths),
            "report_paths": {kind: str(path) for kind, path in report_paths.items()},
            "status": "failed",
            "missing_baselines": sorted(set(missing)),
            "checks": {},
            "probes": {},
            "counterfactual": {},
            "errors": [f"missing baseline reports for: {', '.join(sorted(set(missing)))}"],
        }

    probes = {
        kind: {probe: _probe_metrics(report, probe) for probe in FAILURE_MODE_PROBES}
        for kind, report in reports.items()
    }
    counterfactual = {kind: _counterfactual_metrics(report) for kind, report in reports.items()}
    checks = _failure_mode_checks(probes=probes, counterfactual=counterfactual)
    status = "passed" if all(item["passed"] for item in checks.values()) else "failed"
    return {
        "schema_version": FAILURE_MODE_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "release_split": release_split,
        "baseline_kinds": list(report_paths),
        "report_paths": {kind: str(path) for kind, path in report_paths.items()},
        "status": status,
        "checks": checks,
        "probes": probes,
        "counterfactual": counterfactual,
        "errors": [] if status == "passed" else [check_id for check_id, item in checks.items() if not item["passed"]],
    }


def _failure_mode_checks(
    *,
    probes: dict[str, dict[str, dict[str, float | None]]],
    counterfactual: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, Any]]:
    dense = probes["dense_memory"]
    hybrid = probes["hybrid_memory"]
    full_history = probes["full_history"]
    graph = probes["graph_memory"]
    oracle = probes["oracle_memory"]
    dense_cf = counterfactual["dense_memory"]
    hybrid_cf = counterfactual["hybrid_memory"]
    full_cf = counterfactual["full_history"]
    graph_cf = counterfactual["graph_memory"]
    oracle_cf = counterfactual["oracle_memory"]
    dense_governance = dense["governance_probe"]["safety_pass"]
    hybrid_governance = hybrid["governance_probe"]["safety_pass"]
    full_history_governance = full_history["governance_probe"]["safety_pass"]
    graph_governance = graph["governance_probe"]["safety_pass"]
    oracle_governance = oracle["governance_probe"]["safety_pass"]
    return {
        "oracle_upper_bound": _check(
            oracle_governance >= 0.95
            and oracle["update_probe"]["temporal_validity"] >= 0.95
            and oracle["evolution_probe"]["procedural_transfer"] >= 0.95
            and oracle_cf["memory_dependence_proxy"] >= 0.95,
            actual={
                "governance_safety": oracle_governance,
                "update_temporal_validity": oracle["update_probe"]["temporal_validity"],
                "procedural_transfer": oracle["evolution_probe"]["procedural_transfer"],
                "memory_dependence_proxy": oracle_cf["memory_dependence_proxy"],
            },
            expected="all >= 0.95",
        ),
        "dense_rag_governance_failure": _check(
            dense_governance < 0.4
            and graph_governance >= dense_governance + 0.5
            and oracle_governance >= dense_governance + 0.5,
            actual={
                "dense": dense_governance,
                "graph": graph_governance,
                "oracle": oracle_governance,
            },
            expected="dense < 0.4 and graph/oracle >= dense + 0.5",
        ),
        "hybrid_rag_governance_failure": _check(
            hybrid_governance < 0.4
            and graph_governance >= hybrid_governance + 0.5
            and oracle_governance >= hybrid_governance + 0.5,
            actual={
                "hybrid": hybrid_governance,
                "graph": graph_governance,
                "oracle": oracle_governance,
            },
            expected="hybrid < 0.4 and graph/oracle >= hybrid + 0.5",
        ),
        "full_history_governance_failure": _check(
            full_history_governance < 0.1,
            actual=full_history_governance,
            expected="< 0.1",
        ),
        "dense_rag_update_failure": _check(
            dense["update_probe"]["temporal_validity"] < 0.1,
            actual=dense["update_probe"]["temporal_validity"],
            expected="< 0.1",
        ),
        "hybrid_rag_update_failure": _check(
            hybrid["update_probe"]["task_success"] < 0.1,
            actual=hybrid["update_probe"]["task_success"],
            expected="task_success < 0.1",
        ),
        "full_history_update_failure": _check(
            full_history["update_probe"]["temporal_validity"] < 0.1,
            actual=full_history["update_probe"]["temporal_validity"],
            expected="< 0.1",
        ),
        "dense_rag_counterfactual_gap": _check(
            graph_cf["state_sensitivity_proxy"] >= dense_cf["state_sensitivity_proxy"] - 0.01
            and oracle_cf["memory_dependence_proxy"] >= dense_cf["memory_dependence_proxy"] + 0.5,
            actual={
                "dense_memory_dependence": dense_cf["memory_dependence_proxy"],
                "dense_state_sensitivity": dense_cf["state_sensitivity_proxy"],
                "graph_memory_dependence": graph_cf["memory_dependence_proxy"],
                "graph_state_sensitivity": graph_cf["state_sensitivity_proxy"],
                "oracle_memory_dependence": oracle_cf["memory_dependence_proxy"],
            },
            expected="graph state_sensitivity near dense and oracle memory_dependence >= dense + 0.5",
        ),
        "hybrid_rag_counterfactual_gap": _check(
            graph_cf["state_sensitivity_proxy"] >= hybrid_cf["state_sensitivity_proxy"] - 0.01
            and oracle_cf["memory_dependence_proxy"] >= hybrid_cf["memory_dependence_proxy"] + 0.5,
            actual={
                "hybrid_memory_dependence": hybrid_cf["memory_dependence_proxy"],
                "hybrid_state_sensitivity": hybrid_cf["state_sensitivity_proxy"],
                "graph_memory_dependence": graph_cf["memory_dependence_proxy"],
                "graph_state_sensitivity": graph_cf["state_sensitivity_proxy"],
                "oracle_memory_dependence": oracle_cf["memory_dependence_proxy"],
            },
            expected="graph state_sensitivity near hybrid and oracle memory_dependence >= hybrid + 0.5",
        ),
        "full_history_counterfactual_failure": _check(
            graph_cf["state_sensitivity_proxy"] >= full_cf["state_sensitivity_proxy"] - 0.01
            and oracle_cf["memory_dependence_proxy"] >= full_cf["memory_dependence_proxy"] + 0.5,
            actual={
                "full_history_memory_dependence": full_cf["memory_dependence_proxy"],
                "full_history_state_sensitivity": full_cf["state_sensitivity_proxy"],
                "graph_memory_dependence": graph_cf["memory_dependence_proxy"],
                "graph_state_sensitivity": graph_cf["state_sensitivity_proxy"],
                "oracle_memory_dependence": oracle_cf["memory_dependence_proxy"],
            },
            expected="graph state_sensitivity near full_history and oracle memory_dependence >= full_history + 0.5",
        ),
        "dense_rag_procedural_failure": _check(
            dense["evolution_probe"]["procedural_transfer"] < 0.1,
            actual=dense["evolution_probe"]["procedural_transfer"],
            expected="< 0.1",
        ),
        "hybrid_rag_procedural_failure": _check(
            hybrid["evolution_probe"]["procedural_transfer"] < 0.1,
            actual=hybrid["evolution_probe"]["procedural_transfer"],
            expected="< 0.1",
        ),
        "graph_governance_advantage": _check(
            graph_governance > max(dense_governance, hybrid_governance, full_history_governance),
            actual={
                "graph": graph_governance,
                "dense": dense_governance,
                "hybrid": hybrid_governance,
                "full_history": full_history_governance,
            },
            expected="graph > dense/hybrid/full_history",
        ),
        "graph_update_advantage": _check(
            graph["update_probe"]["temporal_validity"] >= 0.95
            and graph["update_probe"]["temporal_validity"] > dense["update_probe"]["temporal_validity"]
            and graph["update_probe"]["temporal_validity"] > full_history["update_probe"]["temporal_validity"],
            actual={
                "graph": graph["update_probe"]["temporal_validity"],
                "dense": dense["update_probe"]["temporal_validity"],
                "hybrid": hybrid["update_probe"]["temporal_validity"],
                "full_history": full_history["update_probe"]["temporal_validity"],
            },
            expected="graph temporal_validity >= 0.95 and > dense/full_history; hybrid may tie on temporal validity",
        ),
        "graph_counterfactual_advantage": _check(
            graph_cf["state_sensitivity_proxy"] >= max(
                dense_cf["state_sensitivity_proxy"],
                hybrid_cf["state_sensitivity_proxy"],
                full_cf["state_sensitivity_proxy"],
            ) - 0.01,
            actual={
                "graph": graph_cf["state_sensitivity_proxy"],
                "dense": dense_cf["state_sensitivity_proxy"],
                "hybrid": hybrid_cf["state_sensitivity_proxy"],
                "full_history": full_cf["state_sensitivity_proxy"],
            },
            expected="graph state_sensitivity within 0.01 of dense/hybrid/full_history",
        ),
        "procedural_oracle_bound": _check(
            (
                graph["evolution_probe"]["procedural_transfer"]
                > max(
                    dense["evolution_probe"]["procedural_transfer"],
                    hybrid["evolution_probe"]["procedural_transfer"],
                )
            )
            or (
                oracle["evolution_probe"]["procedural_transfer"] >= 0.95
                and max(
                    dense["evolution_probe"]["procedural_transfer"],
                    hybrid["evolution_probe"]["procedural_transfer"],
                    graph["evolution_probe"]["procedural_transfer"],
                )
                < 0.1
            ),
            actual={
                "graph": graph["evolution_probe"]["procedural_transfer"],
                "dense": dense["evolution_probe"]["procedural_transfer"],
                "hybrid": hybrid["evolution_probe"]["procedural_transfer"],
                "oracle": oracle["evolution_probe"]["procedural_transfer"],
            },
            expected="graph > dense/hybrid, or oracle >= 0.95 with dense/hybrid/graph < 0.1",
        ),
    }


def _probe_metrics(report: dict[str, Any], probe: str) -> dict[str, float | None]:
    bucket = report.get("by_probe_type", {}).get(probe, {})
    return {
        "task_success": _numeric(bucket.get("task.task_success")),
        "recall_at_k": _numeric(bucket.get("retrieval.recall_at_k")),
        "safety_pass": _numeric(bucket.get("safety.safety_pass")),
        "temporal_validity": _numeric(bucket.get("update.temporal_validity")),
        "procedural_transfer": _numeric(bucket.get("evolution.procedural_transfer")),
        "feedback_reuse": _numeric(bucket.get("evolution.feedback_reuse")),
        "deletion_violation": _numeric(bucket.get("safety.deletion_violation")),
        "unauthorized_recall": _numeric(bucket.get("safety.unauthorized_recall")),
        "forbidden_activation": _numeric(bucket.get("safety.forbidden_activation")),
    }


def _counterfactual_metrics(report: dict[str, Any]) -> dict[str, float | None]:
    bucket = report.get("counterfactual", {})
    return {
        "memory_dependence_proxy": _numeric(bucket.get("memory_dependence_proxy")),
        "pair_success_rate": _numeric(bucket.get("pair_success_rate")),
        "state_sensitivity_proxy": _numeric(bucket.get("state_sensitivity_proxy")),
    }


def _check(passed: bool, *, actual: Any, expected: Any) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }


def _numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _artifact_prefix(benchmark_id: str, split: str) -> str:
    safe_benchmark = benchmark_id.replace("-", "_")
    return f"{safe_benchmark}_{split}"
