"""Dataset coverage audit for benchmark construction quality."""

from __future__ import annotations

from collections import Counter
from typing import Any

from amb.benchmark.generation.renderers import RENDERER_EVENT_TYPES
from amb.benchmark.schemas.models import Benchmark
from amb.benchmark.quality.gates import quality_checks


def audit_benchmark(benchmark: Benchmark) -> dict[str, Any]:
    domains: Counter[str] = Counter()
    task_types: Counter[str] = Counter()
    probe_types: Counter[str] = Counter()
    memory_types: Counter[str] = Counter()
    privacy_levels: Counter[str] = Counter()
    difficulty_keys: Counter[str] = Counter()
    query_difficulty_levels: Counter[str] = Counter()
    query_difficulty_keys: Counter[str] = Counter()
    event_types: Counter[str] = Counter()
    counterfactual_edits: Counter[str] = Counter()
    stress_families: Counter[str] = Counter()
    stress_tags: Counter[str] = Counter()

    num_cases = len(benchmark.cases)
    num_queries = 0
    num_memories = 0
    num_events = 0
    num_event_edges = 0
    num_state_contracts = 0
    memory_required_queries = 0
    no_memory_queries = 0
    refusal_queries = 0
    sensitive_memories = 0
    deleted_memories = 0
    stale_capable_memories = 0
    state_bound_queries = 0
    forbidden_probe_queries = 0
    cross_subject_cases = 0
    governance_cases = 0
    counterfactual_cases = 0

    for case in benchmark.cases:
        domains[case.domain] += 1
        difficulty_keys.update(case.difficulty.values.keys())
        edit = str(case.difficulty.values.get("counterfactual_edit", "base"))
        counterfactual_edits[edit] += 1
        stress_family = str(case.difficulty.values.get("stress_family", "routine"))
        stress_families[stress_family] += 1
        tags = tuple(str(tag) for tag in case.difficulty.values.get("stress_tags", ()) if str(tag))
        stress_tags.update(tags)
        governance_cases += int("governance" in tags)
        cross_subject_cases += int("cross_subject" in tags or case.domain == "multi_party_collaboration")
        counterfactual_cases += int(edit != "base" or "counterfactual" in tags)
        num_event_edges += len(case.event_edges)
        num_state_contracts += len(case.state_contracts)
        for event in case.events:
            num_events += 1
            event_types[event.event_type] += 1
        for memory in case.gold_memory_units:
            num_memories += 1
            memory_types[memory.type] += 1
            privacy_levels[memory.privacy_level] += 1
            sensitive_memories += int(memory.is_sensitive)
            deleted_memories += int(memory.should_delete)
            stale_capable_memories += int(memory.valid_until is not None)
        for query in case.queries:
            num_queries += 1
            task_types[query.task_type] += 1
            if query.probe_type:
                probe_types[query.probe_type] += 1
            difficulty = dict(query.difficulty or {})
            query_difficulty_levels[str(difficulty.get("level", "missing"))] += 1
            query_difficulty_keys.update(difficulty.keys())
            memory_required_queries += int(query.requires_memory)
            no_memory_queries += int(not query.requires_memory)
            refusal_queries += int(query.expected_behavior.should_refuse)
            state_bound_queries += int(query.state_contract_id is not None)
            forbidden_probe_queries += int(bool(query.forbidden_memory_ids))

    gates = {
        "has_cases": num_cases > 0,
        "has_queries": num_queries > 0,
        "has_memory_required_queries": memory_required_queries > 0,
        "has_no_memory_queries": no_memory_queries > 0,
        "has_refusal_queries": refusal_queries > 0,
        "has_sensitive_memories": sensitive_memories > 0,
        "has_deleted_memories": deleted_memories > 0,
        "has_stale_memory_candidates": stale_capable_memories > 0,
        "has_multiple_domains": len(domains) >= 2,
        "has_multiple_task_types": len(task_types) >= 3,
        "has_multiple_memory_types": len(memory_types) >= 3,
    }
    construction_gates = {
        "has_event_graph": num_events > 0,
        "has_event_edges": num_event_edges > 0,
        "has_state_contracts": num_state_contracts > 0,
        "has_state_bound_queries": state_bound_queries > 0,
        "has_forbidden_probe_queries": forbidden_probe_queries > 0,
        "has_update_events": event_types["fact_update"] > 0,
        "has_deletion_events": event_types["deletion_request"] > 0,
        "has_governance_events": event_types["governance_rule"] > 0,
        "has_tool_events": event_types["tool_result"] > 0,
        "has_distractor_events": event_types["distractor"] > 0,
    }

    quality_report = quality_checks(benchmark)

    return {
        "benchmark_id": benchmark.benchmark_id,
        "num_cases": num_cases,
        "num_queries": num_queries,
        "num_memories": num_memories,
        "num_events": num_events,
        "num_event_edges": num_event_edges,
        "num_state_contracts": num_state_contracts,
        "domains": dict(sorted(domains.items())),
        "task_types": dict(sorted(task_types.items())),
        "probe_types": dict(sorted(probe_types.items())),
        "query_difficulty_levels": dict(sorted(query_difficulty_levels.items())),
        "query_difficulty_keys": sorted(query_difficulty_keys),
        "memory_types": dict(sorted(memory_types.items())),
        "privacy_levels": dict(sorted(privacy_levels.items())),
        "event_types": dict(sorted(event_types.items())),
        "counterfactual_edits": dict(sorted(counterfactual_edits.items())),
        "stress_families": dict(sorted(stress_families.items())),
        "stress_tags": dict(sorted(stress_tags.items())),
        "renderer_coverage": _renderer_coverage(event_types),
        "difficulty_keys": sorted(difficulty_keys),
        "coverage": {
            "memory_required_queries": memory_required_queries,
            "no_memory_queries": no_memory_queries,
            "refusal_queries": refusal_queries,
            "sensitive_memories": sensitive_memories,
            "deleted_memories": deleted_memories,
            "stale_capable_memories": stale_capable_memories,
            "state_bound_queries": state_bound_queries,
            "forbidden_probe_queries": forbidden_probe_queries,
            "governance_cases": governance_cases,
            "counterfactual_cases": counterfactual_cases,
            "cross_subject_cases": cross_subject_cases,
        },
        "quality_gates": gates,
        "quality_gates_passed": all(gates.values()),
        "data_quality_gates": quality_report["gates"],
        "data_quality_gates_passed": quality_report["passed"],
        "data_quality_summary": quality_report["summary"],
        "data_quality_issues": quality_report["issues"],
        "construction_gates": construction_gates,
        "construction_gates_passed": all(construction_gates.values()),
    }


def _renderer_coverage(event_types: Counter[str]) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for renderer, supported_event_types in sorted(RENDERER_EVENT_TYPES.items()):
        counts = {
            event_type: event_types[event_type]
            for event_type in supported_event_types
            if event_types[event_type] > 0
        }
        coverage[renderer] = {
            "source_event_types": counts,
            "num_source_events": sum(counts.values()),
            "provenance_field": "source_event_id",
        }
    return coverage
