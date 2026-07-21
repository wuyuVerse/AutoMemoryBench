"""Lineage audit from probes back to state, memory, events, and turns."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from amb.benchmark.quality.artifact_contract import localize_report_contract
from amb.benchmark.release.splits import RELEASE_SPLITS
from amb.benchmark.schemas.io import load_benchmark, read_json, write_json
from amb.benchmark.schemas.models import Benchmark, Case, ExpectedBehavior, MemoryUnit, Query

LINEAGE_AUDIT_SCHEMA_VERSION = "amst-lineage-audit-v1"


def audit_benchmark_lineage(benchmark: Benchmark) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    counts = _empty_counts()

    for case in benchmark.cases:
        _audit_case_lineage(case, issues, counts)

    return _lineage_report(
        benchmark_id=benchmark.benchmark_id,
        source=str(benchmark.benchmark_id),
        counts=counts,
        issues=issues,
    )


def audit_release_lineage(
    manifest_path: str | Path,
    *,
    splits: Iterable[str] | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    requested_splits = tuple(splits or RELEASE_SPLITS)
    issues: list[dict[str, Any]] = []
    counts = _empty_counts()
    shard_reports = []

    for split in requested_splits:
        entries = _split_entries(manifest.get("split_files", {}).get(split))
        if not entries:
            if split not in manifest.get("withheld_splits", {}):
                issues.append(
                    _issue(
                        "missing_split_artifact",
                        split=split,
                        detail=f"split {split} has no lineage-auditable benchmark artifacts",
                    )
                )
            continue
        for label, raw_path in entries:
            path = _resolve_path(raw_path, manifest_file.parent)
            if not path.exists():
                issues.append(
                    _issue(
                        "missing_shard_artifact",
                        split=split,
                        shard=label,
                        detail=f"shard artifact does not exist: {raw_path}",
                    )
                )
                continue
            benchmark = load_benchmark(path)
            before_issues = len(issues)
            before_queries = counts["num_queries"]
            for case in benchmark.cases:
                _audit_case_lineage(case, issues, counts, split=split, shard=label)
            shard_reports.append(
                {
                    "split": split,
                    "shard": label,
                    "path": str(path),
                    "num_queries": counts["num_queries"] - before_queries,
                    "num_issues": len(issues) - before_issues,
                }
            )

    report = _lineage_report(
        benchmark_id=str(manifest.get("benchmark_id")),
        source=str(manifest_file),
        counts=counts,
        issues=issues,
    )
    report["split_filter"] = list(requested_splits)
    report["shards"] = shard_reports
    return report


def write_lineage_audit(
    output: str | Path,
    *,
    benchmark_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    splits: Iterable[str] | None = None,
) -> dict[str, Any]:
    if bool(benchmark_path) == bool(manifest_path):
        raise ValueError("provide exactly one of benchmark_path or manifest_path")
    if benchmark_path is not None:
        report = audit_benchmark_lineage(load_benchmark(benchmark_path))
    else:
        report = audit_release_lineage(str(manifest_path), splits=splits)
    report = localize_report_contract(
        report,
        output_path=output,
        project_root_hints=(benchmark_path, manifest_path),
    )
    write_json(output, report)
    return report


def _audit_case_lineage(
    case: Case,
    issues: list[dict[str, Any]],
    counts: dict[str, int],
    *,
    split: str | None = None,
    shard: str | None = None,
) -> None:
    counts["num_cases"] += 1
    turn_ids = {turn.turn_id for session in case.sessions for turn in session.turns}
    event_ids = {event.event_id for event in case.events}
    memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units}
    contracts_by_id = {contract.state_contract_id: contract for contract in case.state_contracts}

    for event in case.events:
        counts["num_events"] += 1
        if not event.source_turn_ids:
            _append_issue(issues, case, None, "event_missing_source_turn", "event has no source_turn_ids", split, shard, event.event_id)
        for turn_id in event.source_turn_ids:
            if turn_id not in turn_ids:
                _append_issue(issues, case, None, "event_missing_turn", f"event references missing turn {turn_id}", split, shard, event.event_id)

    for memory in case.gold_memory_units:
        counts["num_memories"] += 1
        _audit_memory_lineage(case, memory, turn_ids, event_ids, issues, counts, split, shard)

    for query in case.queries:
        counts["num_queries"] += 1
        _audit_query_lineage(case, query, memory_by_id, contracts_by_id, issues, counts, split, shard)


def _audit_memory_lineage(
    case: Case,
    memory: MemoryUnit,
    turn_ids: set[str],
    event_ids: set[str],
    issues: list[dict[str, Any]],
    counts: dict[str, int],
    split: str | None,
    shard: str | None,
) -> None:
    if memory.source_turn_ids:
        counts["memories_with_source_turns"] += 1
    else:
        _append_issue(issues, case, None, "memory_missing_source_turn", "memory has no source_turn_ids", split, shard, memory.memory_id)
    if memory.source_event_ids:
        counts["memories_with_source_events"] += 1
    else:
        _append_issue(issues, case, None, "memory_missing_source_event", "memory has no source_event_ids", split, shard, memory.memory_id)
    for turn_id in memory.source_turn_ids:
        if turn_id not in turn_ids:
            _append_issue(issues, case, None, "memory_missing_turn", f"memory references missing turn {turn_id}", split, shard, memory.memory_id)
    for event_id in memory.source_event_ids:
        if event_id not in event_ids:
            _append_issue(issues, case, None, "memory_missing_event", f"memory references missing event {event_id}", split, shard, memory.memory_id)


def _audit_query_lineage(
    case: Case,
    query: Query,
    memory_by_id: dict[str, MemoryUnit],
    contracts_by_id: dict[str, Any],
    issues: list[dict[str, Any]],
    counts: dict[str, int],
    split: str | None,
    shard: str | None,
) -> None:
    contract = contracts_by_id.get(query.state_contract_id or "")
    if query.state_contract_id:
        counts["queries_with_state_contract"] += 1
    else:
        _append_issue(issues, case, query, "query_missing_state_contract", "query has no state_contract_id", split, shard)
    if query.scoring_rule:
        counts["queries_with_scoring_rule"] += 1
    else:
        _append_issue(issues, case, query, "query_missing_scoring_rule", "query has no scoring_rule", split, shard)
    if query.probe_type:
        counts["queries_with_probe_type"] += 1
    else:
        _append_issue(issues, case, query, "query_missing_probe_type", "query has no probe_type", split, shard)

    active = set(contract.active_memory_ids) if contract is not None else set()
    blocked = set() if contract is None else (
        set(contract.deleted_memory_ids)
        | set(contract.forbidden_memory_ids)
        | set(contract.restricted_memory_ids)
        | set(contract.superseded_memory_ids)
    )
    for memory_id in query.gold_memory_ids:
        memory = memory_by_id.get(memory_id)
        if memory is None:
            _append_issue(issues, case, query, "query_missing_gold_memory", f"query references missing gold memory {memory_id}", split, shard)
            continue
        counts["query_memory_links"] += 1
        if contract is not None and memory_id not in active and not query.expected_behavior.should_refuse:
            _append_issue(
                issues,
                case,
                query,
                "gold_memory_not_active_in_contract",
                f"gold memory {memory_id} is not active in the query state contract",
                split,
                shard,
                memory_id,
            )
    for memory_id in query.forbidden_memory_ids:
        memory = memory_by_id.get(memory_id)
        if memory is None:
            _append_issue(issues, case, query, "query_missing_forbidden_memory", f"query references missing forbidden memory {memory_id}", split, shard)
            continue
        counts["query_forbidden_memory_links"] += 1
        if contract is not None and memory_id not in blocked:
            _append_issue(
                issues,
                case,
                query,
                "forbidden_memory_not_blocked_in_contract",
                f"forbidden memory {memory_id} is not blocked in the query state contract",
                split,
                shard,
                memory_id,
            )
    if _expected_behavior_has_target(query.expected_behavior):
        counts["queries_with_expected_target"] += 1
    else:
        _append_issue(issues, case, query, "query_missing_expected_target", "query expected_behavior has no deterministic target", split, shard)
    if _expected_behavior_supported(query.expected_behavior, query.gold_memory_ids, memory_by_id):
        counts["queries_with_supported_expected_behavior"] += 1
    elif query.requires_memory and not query.expected_behavior.should_refuse:
        _append_issue(
            issues,
            case,
            query,
            "expected_behavior_not_supported_by_gold_memory",
            "expected behavior is not supported by gold memory content or metadata",
            split,
            shard,
        )


def _expected_behavior_has_target(expected: ExpectedBehavior) -> bool:
    return bool(
        expected.must_include
        or expected.must_not_include
        or expected.should_refuse
        or expected.tool_name
        or expected.parameters
        or expected.behavior_type == "no_memory"
    )


def _expected_behavior_supported(
    expected: ExpectedBehavior,
    gold_memory_ids: tuple[str, ...],
    memory_by_id: dict[str, MemoryUnit],
) -> bool:
    if expected.should_refuse:
        return True
    if expected.behavior_type == "no_memory":
        return True
    if not expected.must_include:
        return bool(expected.tool_name or expected.parameters)
    gold_text = "\n".join(memory_by_id[memory_id].content for memory_id in gold_memory_ids if memory_id in memory_by_id)
    metadata_text = " ".join(str(value) for value in expected.parameters.values())
    return all(_contains(gold_text, fragment) or _contains(metadata_text, fragment) for fragment in expected.must_include)


def _lineage_report(benchmark_id: str, source: str, counts: dict[str, int], issues: list[dict[str, Any]]) -> dict[str, Any]:
    rates = {
        "memory_source_event_coverage": _rate(counts["memories_with_source_events"], counts["num_memories"]),
        "memory_source_turn_coverage": _rate(counts["memories_with_source_turns"], counts["num_memories"]),
        "query_state_contract_coverage": _rate(counts["queries_with_state_contract"], counts["num_queries"]),
        "query_scoring_rule_coverage": _rate(counts["queries_with_scoring_rule"], counts["num_queries"]),
        "query_probe_type_coverage": _rate(counts["queries_with_probe_type"], counts["num_queries"]),
        "expected_behavior_support_coverage": _rate(counts["queries_with_supported_expected_behavior"], counts["num_queries"]),
    }
    issue_counts: dict[str, int] = {}
    for issue in issues:
        code = str(issue["code"])
        issue_counts[code] = issue_counts.get(code, 0) + 1
    return {
        "schema_version": LINEAGE_AUDIT_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "source": source,
        "status": "passed" if not issues else "failed",
        "summary": {
            **counts,
            "num_issues": len(issues),
            "issue_counts": dict(sorted(issue_counts.items())),
            "coverage_rates": rates,
        },
        "issues": issues[:100],
        "truncated_issues": max(0, len(issues) - 100),
    }


def _empty_counts() -> dict[str, int]:
    return {
        "num_cases": 0,
        "num_events": 0,
        "num_memories": 0,
        "num_queries": 0,
        "memories_with_source_turns": 0,
        "memories_with_source_events": 0,
        "queries_with_state_contract": 0,
        "queries_with_scoring_rule": 0,
        "queries_with_probe_type": 0,
        "queries_with_expected_target": 0,
        "queries_with_supported_expected_behavior": 0,
        "query_memory_links": 0,
        "query_forbidden_memory_links": 0,
    }


def _append_issue(
    issues: list[dict[str, Any]],
    case: Case,
    query: Query | None,
    code: str,
    detail: str,
    split: str | None,
    shard: str | None,
    object_id: str | None = None,
) -> None:
    issues.append(
        _issue(
            code,
            split=split,
            shard=shard,
            case_id=case.case_id,
            query_id=query.query_id if query else None,
            object_id=object_id,
            detail=detail,
        )
    )


def _issue(code: str, **kwargs: Any) -> dict[str, Any]:
    return {"code": code, **{key: value for key, value in kwargs.items() if value is not None}}


def _contains(text: str, fragment: str) -> bool:
    return fragment.strip().lower() in text.lower()


def _rate(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else numerator / denominator


def _split_entries(value: Any) -> tuple[tuple[str, str], ...]:
    if isinstance(value, str):
        return (("benchmark", value),)
    if isinstance(value, dict):
        return tuple((str(label), str(path)) for label, path in sorted(value.items()) if path)
    return ()


def _resolve_path(raw_path: str, manifest_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    return manifest_dir / path
