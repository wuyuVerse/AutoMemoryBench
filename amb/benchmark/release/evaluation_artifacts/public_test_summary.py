"""Human-readable public-test construction summary for AutoMemoryBench releases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from amb.benchmark.schemas.io import read_json


def write_public_test_summary(
    output_path: str | Path,
    *,
    benchmark_id: str,
    profile_id: str,
    public_manifest_path: str | Path,
    intrinsic_sanity_path: str | Path,
    representative_artifact: dict[str, Any] | None,
    public_result_slices_path: str | Path,
    public_test_sanity_path: str | Path,
    failure_mode_diagnostics_path: str | Path,
) -> dict[str, Any]:
    public_manifest = read_json(public_manifest_path)
    intrinsic = read_json(intrinsic_sanity_path)
    public_test_sanity = read_json(public_test_sanity_path)
    failure = read_json(failure_mode_diagnostics_path)
    representative = representative_artifact or {}

    split_reports = public_manifest.get("split_reports", {})
    public_test = split_reports.get("public_test", {})
    public_dev = split_reports.get("public_dev", {})

    text = _render_summary(
        benchmark_id=benchmark_id,
        profile_id=profile_id,
        public_manifest=public_manifest,
        public_dev=public_dev,
        public_test=public_test,
        intrinsic=intrinsic,
        representative=representative,
        public_result_slices_path=str(public_result_slices_path),
        public_test_sanity=public_test_sanity,
        failure=failure,
    )
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text, encoding="utf-8")
    return {
        "path": str(output_file),
        "benchmark_id": benchmark_id,
        "profile_id": profile_id,
        "release_split": "public_test",
    }


def _render_summary(
    *,
    benchmark_id: str,
    profile_id: str,
    public_manifest: dict[str, Any],
    public_dev: dict[str, Any],
    public_test: dict[str, Any],
    intrinsic: dict[str, Any],
    representative: dict[str, Any],
    public_result_slices_path: str,
    public_test_sanity: dict[str, Any],
    failure: dict[str, Any],
) -> str:
    expected = public_manifest.get("expected_generation_summary", {})
    axis = expected.get("counterfactual_axis_coverage", {})
    covered_axes = ", ".join(axis.get("covered_axes", [])) or "none"
    intrinsic_split = intrinsic.get("split_intrinsic_sanity", {}).get("public_test", {})
    checks = failure.get("checks", {})
    representative_top = representative.get("top_system_id", "n/a")
    sanity_checks = public_test_sanity.get("checks", {})

    return f"""# AutoMemoryBench {profile_id} Public-Test Construction Summary

## 1. Release Scope

- benchmark_id: `{benchmark_id}`
- profile_id: `{profile_id}`
- public_dev cases/queries: `{public_dev.get("num_cases", 0)}` / `{public_dev.get("num_queries", 0)}`
- public_test cases/queries: `{public_test.get("num_cases", 0)}` / `{public_test.get("num_queries", 0)}`
- domains: `{expected.get("num_domains", 0)}`
- counterfactual variants per base: `{expected.get("counterfactual_variants_per_base", "n/a")}`
- covered counterfactual axes: `{covered_axes}`

## 2. Construction Gates

- intrinsic sanity status: `{intrinsic.get("status", "unknown")}`
- public_test oracle_solvability: `{intrinsic_split.get("gates", {}).get("oracle_solvability")}`
- public_test no_memory_unsolvability: `{intrinsic_split.get("gates", {}).get("no_memory_unsolvability")}`
- representative-baseline top system on public_dev: `{representative_top}`
- public_test sanity status: `{public_test_sanity.get("status", "unknown")}`
- oracle high AMQ: `{sanity_checks.get("oracle_high_amq", {}).get("actual")}`
- no_memory zero recall: `{sanity_checks.get("no_memory_zero_recall", {}).get("actual")}`
- graph memory beats no_memory AMQ: `{sanity_checks.get("graph_memory_beats_no_memory_amq", {}).get("actual")}`
- full_history cost > graph memory: `{sanity_checks.get("full_history_costs_more_than_graph_memory", {}).get("actual")}`

## 3. Public-Test Failure Modes

The public-test diagnostic artifact confirms the intended benchmark behavior:

- dense RAG governance failure: `{checks.get("dense_rag_governance_failure", {}).get("actual")}`
- hybrid RAG governance failure: `{checks.get("hybrid_rag_governance_failure", {}).get("actual")}`
- full-history counterfactual dependence: `{checks.get("full_history_counterfactual_failure", {}).get("actual")}`
- graph counterfactual advantage: `{checks.get("graph_counterfactual_advantage", {}).get("actual")}`
- graph procedural advantage: `{checks.get("graph_procedural_advantage", {}).get("actual")}`

These diagnostics support the claim that the release can expose:

- governance failures on deletion/authorization-sensitive probes;
- update failures under changed memory state;
- weak counterfactual dependence for retrieval-style systems;
- procedural-memory gaps that are not visible from answer-only performance.

## 4. Primary Artifacts

- public manifest: `{public_manifest.get("manifest_path", "manifest.json")}`
- intrinsic sanity: `reports/examples/{benchmark_id.replace("-", "_")}_intrinsic_sanity.json`
- representative baselines: `reports/examples/{benchmark_id.replace("-", "_")}_public_dev_*`
- public-test sanity: `reports/examples/{benchmark_id.replace("-", "_")}_public_test_sanity.json`
- required slice tables: `{public_result_slices_path}`
- failure-mode diagnostics: `reports/examples/{benchmark_id.replace("-", "_")}_public_test_failure_mode_diagnostics.json`
"""
