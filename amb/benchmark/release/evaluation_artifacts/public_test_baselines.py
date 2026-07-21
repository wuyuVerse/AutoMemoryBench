"""Public-test baseline artifacts for release-level sanity and reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from amb.benchmark.artifact_contract import localize_report_contract
from amb.benchmark.release.evaluation import evaluate_release_split_baselines
from amb.benchmark.release.fingerprint import release_split_contract_fingerprint
from amb.benchmark.schemas.io import read_json, write_json

PUBLIC_TEST_BASELINES = (
    "no_memory",
    "full_history",
    "dense_memory",
    "hybrid_memory",
    "graph_memory",
    "oracle_memory",
)


def build_public_test_baseline_artifacts(
    manifest_path: str | Path,
    *,
    split: str = "public_test",
    output_dir: str | Path = "reports/examples",
    retrieval_k: int = 8,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    manifest_benchmark_id = str(manifest.get("benchmark_id", "release"))
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    prefix = _artifact_prefix(manifest_benchmark_id, split)
    report_paths: dict[str, str] = {}
    summaries: dict[str, dict[str, float | None]] = {}

    reports = evaluate_release_split_baselines(
        manifest_file,
        split=split,
        baseline_kinds=PUBLIC_TEST_BASELINES,
        retrieval_k=retrieval_k,
    )
    release_contract = release_split_contract_fingerprint(manifest_file, split)

    for kind in PUBLIC_TEST_BASELINES:
        report = reports[kind]
        path = output_root / f"{prefix}_{kind}_report.json"
        report = localize_report_contract(
            report,
            output_path=path,
            project_root_hints=(manifest_path,),
        )
        write_json(path, report)
        report_paths[kind] = str(path)
        summaries[kind] = _summary_metrics(report)

    result = {
        "benchmark_id": f"{manifest_benchmark_id}-{split}",
        "release_split": split,
        "release_contract_fingerprint": release_contract,
        "baseline_kinds": list(PUBLIC_TEST_BASELINES),
        "report_paths": report_paths,
        "summary_metrics": summaries,
    }
    result_path = output_root / f"{prefix}_baselines.json"
    result = localize_report_contract(
        result,
        output_path=result_path,
        project_root_hints=(manifest_path, *report_paths.values()),
    )
    write_json(result_path, result)
    result["output_path"] = str(result_path)
    return result


def _summary_metrics(report: dict[str, Any]) -> dict[str, float | None]:
    aggregate = report.get("aggregate", {})
    counterfactual = report.get("counterfactual", {})
    by_requirement = report.get("by_memory_requirement", {})
    requires_memory = by_requirement.get("requires_memory", {})
    no_memory_required = by_requirement.get("no_memory_required", {})
    return {
        "amq": _numeric(aggregate.get("lifecycle.amq")),
        "task_success": _numeric(aggregate.get("task.task_success")),
        "recall_at_k": _numeric(aggregate.get("retrieval.recall_at_k")),
        "requires_memory_num_queries": _numeric(requires_memory.get("num_scored_queries")),
        "requires_memory_amq": _numeric(requires_memory.get("lifecycle.amq")),
        "requires_memory_task_success": _numeric(requires_memory.get("task.task_success")),
        "requires_memory_recall_at_k": _numeric(requires_memory.get("retrieval.recall_at_k")),
        "no_memory_required_num_queries": _numeric(no_memory_required.get("num_scored_queries")),
        "no_memory_required_amq": _numeric(no_memory_required.get("lifecycle.amq")),
        "no_memory_required_task_success": _numeric(no_memory_required.get("task.task_success")),
        "no_memory_required_recall_at_k": _numeric(no_memory_required.get("retrieval.recall_at_k")),
        "safety_pass": _numeric(aggregate.get("safety.safety_pass")),
        "input_tokens": _numeric(aggregate.get("efficiency.input_tokens")),
        "memory_dependence_proxy": _numeric(counterfactual.get("memory_dependence_proxy")),
    }


def _numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _artifact_prefix(benchmark_id: str, split: str) -> str:
    safe_benchmark = benchmark_id.replace("-", "_")
    return f"{safe_benchmark}_{split}"
