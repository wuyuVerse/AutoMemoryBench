"""Release planning, materialization, packaging, and release-level evaluation."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


_BASE = Path(__file__).resolve().parent
__path__ = [
    str(_BASE),
    str(_BASE / "core"),
    str(_BASE / "builders"),
    str(_BASE / "evaluation_artifacts"),
    str(_BASE / "packaging"),
]

_EXPORTS = {
    "MAIN_RELEASE_PROFILES": ("amb.benchmark.release.workflow", "MAIN_RELEASE_PROFILES"),
    "RELEASE_SPLITS": ("amb.benchmark.release.splits", "RELEASE_SPLITS"),
    "ReleaseConfig": ("amb.benchmark.release.splits", "ReleaseConfig"),
    "assign_release_splits": ("amb.benchmark.release.splits", "assign_release_splits"),
    "build_main_release_workflow": ("amb.benchmark.release.workflow", "build_main_release_workflow"),
    "build_release_failure_mode_diagnostics": (
        "amb.benchmark.release.failure_modes",
        "build_release_failure_mode_diagnostics",
    ),
    "build_hidden_test_sanity_artifact": (
        "amb.benchmark.release.hidden_test_sanity",
        "build_hidden_test_sanity_artifact",
    ),
    "build_hidden_test_sanity_summary": (
        "amb.benchmark.release.hidden_test_sanity",
        "build_hidden_test_sanity_summary",
    ),
    "build_profile_release_shards": ("amb.benchmark.release.sharded", "build_profile_release_shards"),
    "build_public_test_baseline_artifacts": (
        "amb.benchmark.release.public_test_baselines",
        "build_public_test_baseline_artifacts",
    ),
    "build_public_result_slice_artifacts": (
        "amb.benchmark.release.public_result_slices",
        "build_public_result_slice_artifacts",
    ),
    "build_public_test_sanity_summary": (
        "amb.benchmark.release.public_test_sanity",
        "build_public_test_sanity_summary",
    ),
    "build_quarterly_hidden_refresh_package": (
        "amb.benchmark.release.private_leaderboard",
        "build_quarterly_hidden_refresh_package",
    ),
    "build_representative_baseline_artifacts": (
        "amb.benchmark.release.representative",
        "build_representative_baseline_artifacts",
    ),
    "build_release": ("amb.benchmark.release.materialized", "build_release"),
    "build_sharded_release": ("amb.benchmark.release.sharded", "build_sharded_release"),
    "evaluate_release_split_baseline": (
        "amb.benchmark.release.evaluation",
        "evaluate_release_split_baseline",
    ),
    "evaluate_release_split_baselines": (
        "amb.benchmark.release.evaluation",
        "evaluate_release_split_baselines",
    ),
    "evaluate_release_split_predictions": (
        "amb.benchmark.release.evaluation",
        "evaluate_release_split_predictions",
    ),
    "export_public_release_package": ("amb.benchmark.release.packages", "export_public_release_package"),
    "planned_release_summary": ("amb.benchmark.release.splits", "planned_release_summary"),
    "release_split_benchmark_id": ("amb.benchmark.release.evaluation", "release_split_benchmark_id"),
    "run_release_split_agent": ("amb.benchmark.release.evaluation", "run_release_split_agent"),
    "run_release_split_agent_experiment": (
        "amb.benchmark.release.evaluation",
        "run_release_split_agent_experiment",
    ),
    "run_release_split_agent_experiment_with_retries": (
        "amb.benchmark.release.evaluation",
        "run_release_split_agent_experiment_with_retries",
    ),
    "run_release_split_agent_with_retries": (
        "amb.benchmark.release.evaluation",
        "run_release_split_agent_with_retries",
    ),
    "sample_release_split": ("amb.benchmark.release.sampling", "sample_release_split"),
    "split_count_mapping": ("amb.benchmark.release.splits", "split_count_mapping"),
    "summarize_public_result_slices": (
        "amb.benchmark.release.public_result_slices",
        "summarize_public_result_slices",
    ),
    "validate_public_result_slices_payload": (
        "amb.benchmark.release.public_result_slices",
        "validate_public_result_slices_payload",
    ),
    "write_public_test_summary": ("amb.benchmark.release.public_test_summary", "write_public_test_summary"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value
