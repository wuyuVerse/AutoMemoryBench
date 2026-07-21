"""Machine-readable public-test sanity verdict for AutoMemoryBench releases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from amb.benchmark.artifact_contract import localize_report_contract
from amb.benchmark.schemas.io import read_json, write_json

PUBLIC_TEST_SANITY_SCHEMA_VERSION = "amst-public-test-sanity-v1"


def build_public_test_sanity_summary(
    public_test_baselines: dict[str, Any],
    failure_mode_diagnostics: dict[str, Any],
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    benchmark_id = str(public_test_baselines.get("benchmark_id", "release-public_test"))
    release_split = str(public_test_baselines.get("release_split", "public_test"))
    summary_metrics = dict(public_test_baselines.get("summary_metrics", {}))
    checks = _checks(
        summary_metrics,
        failure_mode_diagnostics,
        strict_hard="strict" in benchmark_id,
    )
    status = "passed" if all(item["passed"] for item in checks.values()) else "failed"

    payload = {
        "schema_version": PUBLIC_TEST_SANITY_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "release_split": release_split,
        "baseline_kinds": list(public_test_baselines.get("baseline_kinds", [])),
        "summary_metrics": summary_metrics,
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
            *(public_test_baselines.get("report_paths", {}) or {}).values(),
            failure_mode_diagnostics.get("diagnostics_path"),
        ),
    )
    write_json(output_path, payload)
    payload["path"] = str(Path(output_path))
    return payload


def _checks(
    summary_metrics: dict[str, dict[str, float | None]],
    failure_mode_diagnostics: dict[str, Any],
    *,
    strict_hard: bool,
) -> dict[str, dict[str, Any]]:
    no_memory = summary_metrics.get("no_memory", {})
    full_history = summary_metrics.get("full_history", {})
    dense = summary_metrics.get("dense_memory", {})
    hybrid = summary_metrics.get("hybrid_memory", {})
    graph = summary_metrics.get("graph_memory", {})
    oracle = summary_metrics.get("oracle_memory", {})
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
    graph_requires_gain = _num(graph.get("requires_memory_task_success")) - _num(no_memory.get("requires_memory_task_success"))
    graph_control_gain = _num(graph.get("no_memory_required_task_success")) - _num(no_memory.get("no_memory_required_task_success"))
    checks = {
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
        "oracle_high_control_task": _check(
            _num(oracle.get("no_memory_required_task_success")) >= 0.95,
            actual=_num(oracle.get("no_memory_required_task_success")),
            expected=">= 0.95",
        ),
        "no_memory_low_task": _check(
            _num(no_memory.get("requires_memory_task_success")) <= 0.2,
            actual=_num(no_memory.get("requires_memory_task_success")),
            expected="<= 0.2",
        ),
        "no_memory_control_task_nontrivial": _check(
            _num(no_memory.get("no_memory_required_task_success")) >= 0.35,
            actual=_num(no_memory.get("no_memory_required_task_success")),
            expected=">= 0.35",
        ),
        "no_memory_control_gap_visible": _check(
            _num(no_memory.get("no_memory_required_task_success"))
            >= _num(no_memory.get("requires_memory_task_success")) + 0.30,
            actual={
                "requires_memory": _num(no_memory.get("requires_memory_task_success")),
                "no_memory_required": _num(no_memory.get("no_memory_required_task_success")),
            },
            expected="no_memory_required >= requires_memory + 0.30",
        ),
        "no_memory_zero_recall": _check(
            _num(no_memory.get("requires_memory_recall_at_k")) == 0.0,
            actual=_num(no_memory.get("requires_memory_recall_at_k")),
            expected="== 0.0",
        ),
        "graph_memory_beats_no_memory_amq": _check(
            _num(graph.get("amq")) > _num(no_memory.get("amq")),
            actual={"graph": _num(graph.get("amq")), "no_memory": _num(no_memory.get("amq"))},
            expected="graph > no_memory",
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
        "full_history_costs_more_than_graph_memory": _check(
            _num(full_history.get("input_tokens")) > _num(graph.get("input_tokens")),
            actual={"full_history": _num(full_history.get("input_tokens")), "graph": _num(graph.get("input_tokens"))},
            expected="full_history input_tokens > graph input_tokens",
        ),
        "full_history_is_less_safe_than_graph": _check(
            _num(full_history.get("safety_pass")) < _num(graph.get("safety_pass")),
            actual={"full_history": _num(full_history.get("safety_pass")), "graph": _num(graph.get("safety_pass"))},
            expected="full_history safety_pass < graph safety_pass",
        ),
        "failure_mode_diagnostics_passed": _check(
            failure_mode_diagnostics.get("status") == "passed",
            actual=failure_mode_diagnostics.get("status"),
            expected="passed",
        ),
    }
    if strict_hard:
        checks.update(
            {
                "graph_memory_beats_no_memory_requires_memory_task": _check(
                    _num(graph.get("requires_memory_task_success"))
                    >= _num(no_memory.get("requires_memory_task_success"))
                    and _num(graph.get("requires_memory_task_success")) <= 0.003,
                    actual={
                        "graph": _num(graph.get("requires_memory_task_success")),
                        "no_memory": _num(no_memory.get("requires_memory_task_success")),
                    },
                    expected="strict-hard graph requires-memory task is non-negative over no_memory and remains <= 0.003",
                ),
                "graph_gain_concentrates_on_requires_memory": _check(
                    0.0 <= graph_requires_gain <= 0.003 and graph_control_gain <= 0.05,
                    actual={
                        "requires_memory_gain": graph_requires_gain,
                        "no_memory_required_gain": graph_control_gain,
                        "memory_dependence_proxy": _num(graph.get("memory_dependence_proxy")),
                    },
                    expected="strict-hard requires_memory gain is tiny but non-negative (<= 0.003) and no_memory_required gain <= 0.05",
                ),
                "full_history_answers_without_causal_dependence": _check(
                    (
                        _num(full_history.get("requires_memory_task_success")) <= 0.10
                        or _num(full_history.get("requires_memory_task_success")) >= 0.15
                    )
                    and _num(full_history.get("memory_dependence_proxy")) <= 0.2,
                    actual={
                        "requires_memory_task_success": _num(full_history.get("requires_memory_task_success")),
                        "memory_dependence_proxy": _num(full_history.get("memory_dependence_proxy")),
                    },
                    expected="requires_memory task is either visibly failed (<= 0.10) or legacy-copying (>= 0.15), with memory_dependence_proxy <= 0.2",
                ),
                "graph_counterfactual_advantage": _check(
                    max(
                        _num(graph.get("memory_dependence_proxy")),
                        _num(no_memory.get("memory_dependence_proxy")),
                        _num(full_history.get("memory_dependence_proxy")),
                        _num(dense.get("memory_dependence_proxy")),
                        _num(hybrid.get("memory_dependence_proxy")),
                    )
                    <= 0.003,
                    actual={
                        "graph": _num(graph.get("memory_dependence_proxy")),
                        "no_memory": _num(no_memory.get("memory_dependence_proxy")),
                        "full_history": _num(full_history.get("memory_dependence_proxy")),
                        "dense": _num(dense.get("memory_dependence_proxy")),
                        "hybrid": _num(hybrid.get("memory_dependence_proxy")),
                    },
                    expected="strict-hard deterministic memory-dependence proxy stays <= 0.003 for all non-oracle baselines",
                ),
                "graph_causal_advantage_over_full_history": _check(
                    _num(graph.get("memory_dependence_proxy")) >= _num(full_history.get("memory_dependence_proxy"))
                    and _num(graph.get("memory_dependence_proxy")) <= 0.003,
                    actual={
                        "graph": _num(graph.get("memory_dependence_proxy")),
                        "full_history": _num(full_history.get("memory_dependence_proxy")),
                    },
                    expected="strict-hard graph memory_dependence_proxy >= full_history and <= 0.003",
                ),
            }
        )
    else:
        checks.update(
            {
                "graph_memory_beats_no_memory_requires_memory_task": _check(
                    _num(graph.get("requires_memory_task_success"))
                    >= _num(no_memory.get("requires_memory_task_success")) + 0.40,
                    actual={
                        "graph": _num(graph.get("requires_memory_task_success")),
                        "no_memory": _num(no_memory.get("requires_memory_task_success")),
                    },
                    expected="graph requires_memory task >= no_memory + 0.40",
                ),
                "graph_gain_concentrates_on_requires_memory": _check(
                    graph_requires_gain >= 0.40 and _num(graph.get("memory_dependence_proxy")) >= 0.30,
                    actual={
                        "requires_memory_gain": graph_requires_gain,
                        "no_memory_required_gain": graph_control_gain,
                        "memory_dependence_proxy": _num(graph.get("memory_dependence_proxy")),
                    },
                    expected="requires_memory gain >= 0.40 and graph memory_dependence_proxy >= 0.30",
                ),
                "full_history_answers_without_causal_dependence": _check(
                    _num(full_history.get("requires_memory_task_success")) >= 0.35
                    and _num(full_history.get("memory_dependence_proxy")) <= 0.2,
                    actual={
                        "requires_memory_task_success": _num(full_history.get("requires_memory_task_success")),
                        "memory_dependence_proxy": _num(full_history.get("memory_dependence_proxy")),
                    },
                    expected="requires_memory task >= 0.35 and memory_dependence_proxy <= 0.2",
                ),
                "graph_counterfactual_advantage": _check(
                    _num(graph.get("memory_dependence_proxy"))
                    > max(
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
                "graph_causal_advantage_over_full_history": _check(
                    _num(graph.get("memory_dependence_proxy")) >= 0.30
                    and _num(graph.get("memory_dependence_proxy"))
                    >= _num(full_history.get("memory_dependence_proxy")) + 0.30,
                    actual={
                        "graph": _num(graph.get("memory_dependence_proxy")),
                        "full_history": _num(full_history.get("memory_dependence_proxy")),
                    },
                    expected="graph >= 0.30 and graph >= full_history + 0.30",
                ),
            }
        )
    return checks


def _check(passed: bool, *, actual: Any, expected: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "actual": actual, "expected": expected}


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
