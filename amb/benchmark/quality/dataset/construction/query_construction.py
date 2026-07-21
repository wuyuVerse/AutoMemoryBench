"""Machine-readable audit for query construction quality on benchmark artifacts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from amb.benchmark.quality.artifact_contract import localize_report_contract
from amb.benchmark.quality.gates import (
    _counterfactual_issues,
    _counterfactual_prompt_signature,
    _query_construction_issues,
)
from amb.benchmark.quality.main_dataset_acceptance import _split_entries
from amb.benchmark.release.splits import RELEASE_SPLITS
from amb.benchmark.schemas.io import load_benchmark, read_json, write_json
from amb.benchmark.schemas.models import Benchmark, Case, Query

QUERY_CONSTRUCTION_AUDIT_SCHEMA_VERSION = "amst-query-construction-audit-v1"


def audit_query_construction_benchmark(benchmark: Benchmark) -> dict[str, Any]:
    pairs = _benchmark_pairs(benchmark)
    report = _build_query_construction_report(
        benchmark_id=benchmark.benchmark_id,
        release_split=None,
        pairs=pairs,
    )
    report["source_type"] = "benchmark"
    return report


def audit_query_construction_release(
    manifest_path: str | Path,
    *,
    split: str | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    splits = (split,) if split is not None else RELEASE_SPLITS
    split_files = manifest.get("split_files", {})

    pairs: list[tuple[str, Case, Query]] = []
    per_split_counts: dict[str, int] = {}
    shard_counts: dict[str, int] = {}
    for split_name in splits:
        entries = _split_entries(split_files.get(split_name))
        if not entries:
            continue
        for label, raw_path in entries:
            path = _resolve_manifest_path(manifest_file.parent, str(raw_path))
            benchmark = load_benchmark(path)
            shard_counts[f"{split_name}:{label}"] = len(benchmark.cases)
            current_pairs = _benchmark_pairs(benchmark, split_name=split_name)
            per_split_counts[split_name] = per_split_counts.get(split_name, 0) + len(current_pairs)
            pairs.extend(current_pairs)

    report = _build_query_construction_report(
        benchmark_id=str(manifest.get("benchmark_id", "release")),
        release_split=split,
        pairs=pairs,
    )
    report["source_type"] = "release_manifest"
    report["manifest_path"] = str(manifest_file)
    report["split_query_counts"] = {name: per_split_counts[name] for name in sorted(per_split_counts)}
    report["shard_case_counts"] = {name: shard_counts[name] for name in sorted(shard_counts)}
    return report


def write_query_construction_audit(
    output: str | Path,
    *,
    benchmark_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    if bool(benchmark_path) == bool(manifest_path):
        raise ValueError("provide exactly one of benchmark_path or manifest_path")
    if benchmark_path is not None:
        benchmark = load_benchmark(benchmark_path)
        report = audit_query_construction_benchmark(benchmark)
        report["benchmark_path"] = str(Path(benchmark_path))
    else:
        report = audit_query_construction_release(manifest_path, split=split)
    report = localize_report_contract(
        report,
        output_path=output,
        project_root_hints=(benchmark_path, manifest_path),
    )
    write_json(output, report)
    return report


def _build_query_construction_report(
    *,
    benchmark_id: str,
    release_split: str | None,
    pairs: list[tuple[str, Case, Query]],
) -> dict[str, Any]:
    query_pairs = [(case, query) for _, case, query in pairs]
    counterfactual_groups: dict[str, list[tuple[Case, Query]]] = defaultdict(list)
    for _, case, query in pairs:
        if query.counterfactual_group_id:
            counterfactual_groups[str(query.counterfactual_group_id)].append((case, query))

    query_construction_issues = _query_construction_issues(query_pairs)
    counterfactual_issues = _counterfactual_issues(counterfactual_groups)
    all_issues = [*query_construction_issues, *counterfactual_issues]

    by_code: dict[str, int] = {}
    by_gate: dict[str, int] = {}
    by_probe_type: dict[str, int] = {}
    for issue in all_issues:
        code = str(issue.get("code", "unknown"))
        gate = str(issue.get("gate", "unknown"))
        by_code[code] = by_code.get(code, 0) + 1
        by_gate[gate] = by_gate.get(gate, 0) + 1
        probe_type = issue.get("probe_type")
        if probe_type is not None:
            name = str(probe_type)
            by_probe_type[name] = by_probe_type.get(name, 0) + 1

    counterfactual_group_sizes = sorted(len(members) for members in counterfactual_groups.values())
    prompt_variants_per_group = _prompt_variants_per_group(counterfactual_groups)
    scoring_variants_per_group = _field_variants_per_group(counterfactual_groups, field="scoring_rule")
    task_variants_per_group = _field_variants_per_group(counterfactual_groups, field="task_type")
    probe_variants_per_group = _field_variants_per_group(counterfactual_groups, field="probe_type")

    no_issue_codes = {
        "gold_memory_missing_required_fragment",
        "gold_memory_has_no_direct_support",
        "overdeclared_gold_memory",
        "missing_adversarial_competitor",
        "missing_state_grounded_competitor",
        "hard_query_missing_compositional_support",
        "hard_query_literal_sloting",
        "singleton_counterfactual_group",
        "counterfactual_group_has_no_expected_difference",
        "counterfactual_group_not_comparable",
        "low_prompt_skeleton_diversity",
    }
    checks = {
        "no_query_construction_issues": _check(not query_construction_issues, len(query_construction_issues), 0),
        "no_counterfactual_comparability_issues": _check(not counterfactual_issues, len(counterfactual_issues), 0),
        "no_gold_minimality_issues": _check(
            not any(code in by_code for code in ("gold_memory_missing_required_fragment", "gold_memory_has_no_direct_support", "overdeclared_gold_memory")),
            {code: by_code.get(code, 0) for code in ("gold_memory_missing_required_fragment", "gold_memory_has_no_direct_support", "overdeclared_gold_memory")},
            "all zero",
        ),
        "no_adversarial_competitor_issues": _check(by_code.get("missing_adversarial_competitor", 0) == 0, by_code.get("missing_adversarial_competitor", 0), 0),
        "no_state_grounded_competitor_issues": _check(
            by_code.get("missing_state_grounded_competitor", 0) == 0,
            by_code.get("missing_state_grounded_competitor", 0),
            0,
        ),
        "no_hard_query_shortcut_issues": _check(
            by_code.get("hard_query_missing_compositional_support", 0) == 0
            and by_code.get("hard_query_literal_sloting", 0) == 0,
            {
                "hard_query_missing_compositional_support": by_code.get("hard_query_missing_compositional_support", 0),
                "hard_query_literal_sloting": by_code.get("hard_query_literal_sloting", 0),
            },
            "all zero",
        ),
        "counterfactual_groups_have_multiple_members": _check(
            bool(counterfactual_group_sizes) and min(counterfactual_group_sizes) >= 2,
            min(counterfactual_group_sizes) if counterfactual_group_sizes else 0,
            ">= 2",
        ),
        "counterfactual_groups_have_prompt_signature": _check(
            bool(prompt_variants_per_group) and min(prompt_variants_per_group.values()) >= 1,
            max(prompt_variants_per_group.values()) if prompt_variants_per_group else 0,
            ">= 1",
        ),
        "counterfactual_groups_single_scoring_rule": _check(
            bool(scoring_variants_per_group) and max(scoring_variants_per_group.values()) == 1,
            max(scoring_variants_per_group.values()) if scoring_variants_per_group else 0,
            1,
        ),
        "counterfactual_groups_single_task_type": _check(
            bool(task_variants_per_group) and max(task_variants_per_group.values()) == 1,
            max(task_variants_per_group.values()) if task_variants_per_group else 0,
            1,
        ),
        "counterfactual_groups_single_probe_type": _check(
            bool(probe_variants_per_group) and max(probe_variants_per_group.values()) == 1,
            max(probe_variants_per_group.values()) if probe_variants_per_group else 0,
            1,
        ),
        "counterfactual_groups_share_target_slot": _check(
            by_code.get("counterfactual_group_missing_shared_target_slot", 0) == 0,
            by_code.get("counterfactual_group_missing_shared_target_slot", 0),
            0,
        ),
        "counterfactual_groups_change_target_slot_state": _check(
            by_code.get("counterfactual_group_target_slot_state_static", 0) == 0,
            by_code.get("counterfactual_group_target_slot_state_static", 0),
            0,
        ),
    }

    return {
        "schema_version": QUERY_CONSTRUCTION_AUDIT_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "release_split": release_split,
        "status": "passed" if all(item["passed"] for item in checks.values()) else "failed",
        "summary": {
            "num_cases": len({case.case_id for _, case, _ in pairs}),
            "num_queries": len(query_pairs),
            "num_counterfactual_groups": len(counterfactual_groups),
            "num_query_construction_issues": len(query_construction_issues),
            "num_counterfactual_issues": len(counterfactual_issues),
            "num_total_issues": len(all_issues),
            "issue_codes_present": sorted(code for code in by_code if code in no_issue_codes and by_code.get(code, 0) > 0),
        },
        "checks": checks,
        "issue_counts": {
            "by_gate": {name: by_gate[name] for name in sorted(by_gate)},
            "by_code": {name: by_code[name] for name in sorted(by_code)},
            "by_probe_type": {name: by_probe_type[name] for name in sorted(by_probe_type)},
        },
        "counterfactual_group_stats": {
            "min_group_size": min(counterfactual_group_sizes) if counterfactual_group_sizes else 0,
            "max_group_size": max(counterfactual_group_sizes) if counterfactual_group_sizes else 0,
            "prompt_variants_per_group_max": max(prompt_variants_per_group.values()) if prompt_variants_per_group else 0,
            "scoring_rule_variants_per_group_max": max(scoring_variants_per_group.values()) if scoring_variants_per_group else 0,
            "task_type_variants_per_group_max": max(task_variants_per_group.values()) if task_variants_per_group else 0,
            "probe_type_variants_per_group_max": max(probe_variants_per_group.values()) if probe_variants_per_group else 0,
        },
        "sample_issues": all_issues[:50],
    }


def _benchmark_pairs(benchmark: Benchmark, *, split_name: str | None = None) -> list[tuple[str, Case, Query]]:
    pairs: list[tuple[str, Case, Query]] = []
    for case in benchmark.cases:
        for query in case.queries:
            pairs.append((split_name or "benchmark", case, query))
    return pairs


def _prompt_variants_per_group(groups: dict[str, list[tuple[Case, Query]]]) -> dict[str, int]:
    return {
        group_id: len({_counterfactual_prompt_signature(query.prompt) for _, query in members})
        for group_id, members in sorted(groups.items())
    }


def _field_variants_per_group(
    groups: dict[str, list[tuple[Case, Query]]],
    *,
    field: str,
) -> dict[str, int]:
    return {
        group_id: len({getattr(query, field) for _, query in members})
        for group_id, members in sorted(groups.items())
    }


def _check(passed: bool, actual: Any, expected: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "actual": actual, "expected": expected}


def _resolve_manifest_path(manifest_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    return manifest_dir / path
