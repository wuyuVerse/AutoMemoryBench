"""Machine-readable private hidden-test sanity verdict for AutoMemoryBench releases."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any

from amb.benchmark.artifact_contract import localize_report_contract
from amb.benchmark.evaluation.scoring import DEFAULT_RETRIEVAL_K
from amb.benchmark.release.failure_modes import build_release_failure_mode_diagnostics
from amb.benchmark.release.public_test_baselines import build_public_test_baseline_artifacts
from amb.benchmark.schemas.io import read_json, write_json

HIDDEN_TEST_SANITY_SCHEMA_VERSION = "amst-hidden-test-sanity-v1"
GRAPH_HIDDEN_REQUIRES_MEMORY_RECALL_MIN = 0.50


def build_hidden_test_sanity_artifact(
    manifest_path: str | Path,
    *,
    output_dir: str | Path = "reports/examples",
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
) -> dict[str, Any]:
    """Build private hidden-test baselines, diagnostics, and sanity summary."""

    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    _require_private_hidden_manifest(manifest)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    benchmark_id = str(manifest.get("benchmark_id", "release"))
    prefix = _artifact_prefix(benchmark_id, "hidden_test")

    hidden_baselines = build_public_test_baseline_artifacts(
        manifest_file,
        split="hidden_test",
        output_dir=output_root,
        retrieval_k=retrieval_k,
    )
    failure_mode_diagnostics = build_release_failure_mode_diagnostics(
        manifest_file,
        split="hidden_test",
        output_dir=output_root,
        retrieval_k=retrieval_k,
        existing_report_paths=hidden_baselines["report_paths"],
    )
    return build_hidden_test_sanity_summary(
        manifest,
        hidden_baselines,
        failure_mode_diagnostics,
        output_path=output_root / f"{prefix}_sanity.json",
    )


def build_hidden_test_sanity_summary(
    manifest: dict[str, Any],
    hidden_test_baselines: dict[str, Any],
    failure_mode_diagnostics: dict[str, Any],
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build a private hidden-test sanity verdict from precomputed artifacts."""

    _require_private_hidden_manifest(manifest)
    benchmark_id = str(hidden_test_baselines.get("benchmark_id", f"{manifest.get('benchmark_id', 'release')}-hidden_test"))
    release_split = str(hidden_test_baselines.get("release_split", "hidden_test"))
    summary_metrics = dict(hidden_test_baselines.get("summary_metrics", {}))
    hidden_enrichment = dict(manifest.get("hidden_enrichment_summary", {}))
    checks = _checks(summary_metrics, hidden_enrichment, manifest, failure_mode_diagnostics)
    status = "passed" if all(item["passed"] for item in checks.values()) else "failed"

    payload = {
        "schema_version": HIDDEN_TEST_SANITY_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "release_split": release_split,
        "baseline_kinds": list(hidden_test_baselines.get("baseline_kinds", [])),
        "summary_metrics": summary_metrics,
        "hidden_enrichment_status": hidden_enrichment.get("status"),
        "hidden_enrichment_checks": dict(hidden_enrichment.get("checks", {})),
        "failure_mode_diagnostics_path": failure_mode_diagnostics.get("diagnostics_path"),
        "failure_mode_status": failure_mode_diagnostics.get("status"),
        "checks": checks,
        "status": status,
        "summary": {
            "passed": sum(1 for item in checks.values() if item["passed"]),
            "failed": sum(1 for item in checks.values() if not item["passed"]),
        },
    }
    payload = localize_report_contract(
        payload,
        output_path=output_path,
        project_root_hints=(
            *(hidden_test_baselines.get("report_paths", {}) or {}).values(),
            failure_mode_diagnostics.get("diagnostics_path"),
        ),
    )
    output_file = Path(output_path).resolve()
    package_root = output_file.parent.parent if output_file.parent.name == "reports" else output_file.parent
    payload["root"] = Path(os.path.relpath(package_root, output_file.parent)).as_posix()
    write_json(output_path, payload)
    payload["path"] = str(Path(output_path))
    return payload


def _checks(
    summary_metrics: dict[str, dict[str, float | None]],
    hidden_enrichment: dict[str, Any],
    manifest: dict[str, Any],
    failure_mode_diagnostics: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    no_memory = summary_metrics.get("no_memory", {})
    full_history = summary_metrics.get("full_history", {})
    dense = summary_metrics.get("dense_memory", {})
    hybrid = summary_metrics.get("hybrid_memory", {})
    graph = summary_metrics.get("graph_memory", {})
    oracle = summary_metrics.get("oracle_memory", {})
    probes = failure_mode_diagnostics.get("probes", {})
    graph_probes = probes.get("graph_memory", {})
    slice_counts = {
        "requires_memory": {
            name: _num(metrics.get("requires_memory_num_queries"))
            for name, metrics in summary_metrics.items()
        },
        "no_memory_required": {
            name: _num(metrics.get("no_memory_required_num_queries"))
            for name, metrics in summary_metrics.items()
        },
    }
    enrichment_checks = dict(hidden_enrichment.get("checks", {}))
    visibility = str(manifest.get("visibility", {}).get("hidden_test", ""))
    return {
        "private_hidden_split_present": _check(
            _has_materialized_hidden_split(manifest),
            actual=_hidden_split_descriptor(manifest),
            expected="materialized hidden_test shard(s) in a private release manifest",
        ),
        "hidden_visibility_private": _check(
            visibility == "private_leaderboard_only",
            actual=visibility,
            expected="private_leaderboard_only",
        ),
        "hidden_enrichment_summary_present": _check(
            bool(hidden_enrichment),
            actual=bool(hidden_enrichment),
            expected=True,
        ),
        "hidden_counterfactual_enrichment": _check(
            bool(enrichment_checks.get("counterfactual_share_gt_public_test")),
            actual=enrichment_checks.get("counterfactual_share_gt_public_test"),
            expected=True,
        ),
        "hidden_governance_enrichment": _check(
            bool(enrichment_checks.get("governance_share_gt_public_test")),
            actual=enrichment_checks.get("governance_share_gt_public_test"),
            expected=True,
        ),
        "hidden_cross_subject_enrichment": _check(
            bool(enrichment_checks.get("cross_subject_share_gt_public_test")),
            actual=enrichment_checks.get("cross_subject_share_gt_public_test"),
            expected=True,
        ),
        "required_and_control_slices_present": _check(
            all(value > 0 for value in slice_counts["requires_memory"].values())
            and all(value > 0 for value in slice_counts["no_memory_required"].values())
            and len(set(slice_counts["requires_memory"].values())) == 1
            and len(set(slice_counts["no_memory_required"].values())) == 1,
            actual=slice_counts,
            expected="all baselines expose nonzero requires_memory/no_memory_required slices with consistent counts",
        ),
        "oracle_high_amq": _check(
            _num(oracle.get("amq")) >= 0.95,
            actual=_num(oracle.get("amq")),
            expected=">= 0.95",
        ),
        "oracle_high_requires_memory_task": _check(
            _num(oracle.get("requires_memory_task_success")) >= 0.95,
            actual=_num(oracle.get("requires_memory_task_success")),
            expected=">= 0.95",
        ),
        "no_memory_control_task_nontrivial": _check(
            _num(no_memory.get("no_memory_required_task_success")) >= 0.35,
            actual=_num(no_memory.get("no_memory_required_task_success")),
            expected=">= 0.35",
        ),
        "no_memory_low_requires_memory_task": _check(
            _num(no_memory.get("requires_memory_task_success")) <= 0.05,
            actual=_num(no_memory.get("requires_memory_task_success")),
            expected="<= 0.05",
        ),
        "no_memory_zero_requires_memory_recall": _check(
            _num(no_memory.get("requires_memory_recall_at_k")) == 0.0,
            actual=_num(no_memory.get("requires_memory_recall_at_k")),
            expected="== 0.0",
        ),
        "graph_memory_high_amq": _check(
            _num(graph.get("amq")) >= 0.45,
            actual=_num(graph.get("amq")),
            expected=">= 0.45",
        ),
        "graph_memory_beats_no_memory_requires_memory_task": _check(
            _num(graph.get("requires_memory_task_success")) >= _num(no_memory.get("requires_memory_task_success")) + 0.10,
            actual={
                "graph": _num(graph.get("requires_memory_task_success")),
                "no_memory": _num(no_memory.get("requires_memory_task_success")),
            },
            expected="graph requires_memory task >= no_memory + 0.10",
        ),
        "graph_memory_beats_no_memory_amq": _check(
            _num(graph.get("amq")) > _num(no_memory.get("amq")),
            actual={"graph": _num(graph.get("amq")), "no_memory": _num(no_memory.get("amq"))},
            expected="graph amq > no_memory",
        ),
        "graph_memory_requires_memory_recall_high": _check(
            _num(graph.get("requires_memory_recall_at_k")) >= GRAPH_HIDDEN_REQUIRES_MEMORY_RECALL_MIN,
            actual=_num(graph.get("requires_memory_recall_at_k")),
            expected=f">= {GRAPH_HIDDEN_REQUIRES_MEMORY_RECALL_MIN:.2f}",
        ),
        "graph_memory_beats_dense_amq": _check(
            _num(graph.get("amq")) > _num(dense.get("amq")),
            actual={"graph": _num(graph.get("amq")), "dense": _num(dense.get("amq"))},
            expected="graph > dense",
        ),
        "graph_memory_beats_hybrid_amq": _check(
            _num(graph.get("amq")) > _num(hybrid.get("amq")),
            actual={"graph": _num(graph.get("amq")), "hybrid": _num(hybrid.get("amq"))},
            expected="graph > hybrid",
        ),
        "full_history_hidden_failure_visible": _check(
            _num(full_history.get("amq")) <= 0.20
            and _num(full_history.get("requires_memory_task_success")) <= 0.10
            and _num(full_history.get("memory_dependence_proxy")) <= 0.10
            and _num(full_history.get("safety_pass")) < 0.10,
            actual={
                "amq": _num(full_history.get("amq")),
                "requires_memory_task_success": _num(full_history.get("requires_memory_task_success")),
                "memory_dependence_proxy": _num(full_history.get("memory_dependence_proxy")),
                "safety_pass": _num(full_history.get("safety_pass")),
            },
            expected={
                "amq": "<= 0.20",
                "requires_memory_task_success": "<= 0.10",
                "memory_dependence_proxy": "<= 0.10",
                "safety_pass": "< 0.10",
            },
        ),
        "graph_counterfactual_advantage": _check(
            _num(graph.get("memory_dependence_proxy")) > max(
                _num(no_memory.get("memory_dependence_proxy")),
                _num(full_history.get("memory_dependence_proxy")),
                _num(dense.get("memory_dependence_proxy")),
                _num(hybrid.get("memory_dependence_proxy")),
            ),
            actual={
                "graph": _num(graph.get("memory_dependence_proxy")),
                "no_memory": _num(no_memory.get("memory_dependence_proxy")),
                "full_history": _num(full_history.get("memory_dependence_proxy")),
                "dense": _num(dense.get("memory_dependence_proxy")),
                "hybrid": _num(hybrid.get("memory_dependence_proxy")),
            },
            expected="graph > no_memory/full_history/dense/hybrid",
        ),
        "graph_counterfactual_high": _check(
            _num(graph.get("memory_dependence_proxy")) >= 0.01,
            actual=_num(graph.get("memory_dependence_proxy")),
            expected=">= 0.01",
        ),
        "graph_governance_high": _check(
            _num(_probe_metric(graph_probes, "governance_probe", "safety_pass")) >= 0.95,
            actual=_num(_probe_metric(graph_probes, "governance_probe", "safety_pass")),
            expected=">= 0.95",
        ),
        "graph_update_temporal_high": _check(
            _num(_probe_metric(graph_probes, "update_probe", "temporal_validity")) >= 0.95,
            actual=_num(_probe_metric(graph_probes, "update_probe", "temporal_validity")),
            expected=">= 0.95",
        ),
        "graph_procedural_high": _check(
            _num(_probe_metric(graph_probes, "evolution_probe", "procedural_transfer")) >= 0.50,
            actual=_num(_probe_metric(graph_probes, "evolution_probe", "procedural_transfer")),
            expected=">= 0.50",
        ),
        "failure_mode_diagnostics_passed": _check(
            failure_mode_diagnostics.get("status") == "passed",
            actual=failure_mode_diagnostics.get("status"),
            expected="passed",
        ),
    }


def _require_private_hidden_manifest(manifest: dict[str, Any]) -> None:
    if str(manifest.get("package_type")) == "public_release_export":
        raise ValueError("hidden_test sanity requires a private release manifest, not a public export")
    if not _has_materialized_hidden_split(manifest):
        raise ValueError("hidden_test sanity requires materialized hidden_test artifacts in the private manifest")
    if str(manifest.get("visibility", {}).get("hidden_test")) != "private_leaderboard_only":
        raise ValueError("hidden_test sanity requires hidden_test visibility to remain private_leaderboard_only")


def _has_materialized_hidden_split(manifest: dict[str, Any]) -> bool:
    hidden_files = manifest.get("split_files", {}).get("hidden_test")
    if isinstance(hidden_files, str):
        return bool(hidden_files)
    if isinstance(hidden_files, dict):
        return bool(hidden_files)
    return False


def _hidden_split_descriptor(manifest: dict[str, Any]) -> Any:
    hidden_files = manifest.get("split_files", {}).get("hidden_test")
    if isinstance(hidden_files, dict):
        return sorted(hidden_files)
    return hidden_files


def _check(passed: bool, *, actual: Any, expected: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "actual": actual, "expected": expected}


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _artifact_prefix(benchmark_id: str, split: str) -> str:
    safe_benchmark = benchmark_id.replace("-", "_")
    return f"{safe_benchmark}_{split}"


def _probe_metric(probes: dict[str, Any], probe_type: str, field: str) -> float:
    value = probes.get(probe_type, {}).get(field, 0.0)
    return _num(value)
