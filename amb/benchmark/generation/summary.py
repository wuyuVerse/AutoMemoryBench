"""Construction summaries for generated and planned AutoMemoryBench datasets."""

from __future__ import annotations

from collections import Counter
from typing import Any

from amb.benchmark.generation.domains.counterfactual import counterfactual_axis_coverage
from amb.benchmark.generation.domains.specs import select_domain_specs
from amb.benchmark.generation.types import GenerationConfig
from amb.benchmark.schemas.models import Benchmark


EVENTS_PER_CASE = 25
MEMORIES_PER_CASE = 24
QUERIES_PER_CASE = 21
STATE_CONTRACTS_PER_CASE = 3


def expected_generation_summary(config: GenerationConfig) -> dict[str, Any]:
    domains = select_domain_specs(config.domains)
    num_domains = len(domains)
    base_scenarios = num_domains * config.case_count_per_domain
    counterfactual_scenarios = base_scenarios * config.counterfactual_variants_per_case
    total_case_variants = base_scenarios + counterfactual_scenarios
    return {
        "benchmark_id": config.benchmark_id,
        "name": config.name,
        "seed": config.seed,
        "num_domains": num_domains,
        "domains": [spec.domain for spec in domains],
        "base_scenarios_per_domain": config.case_count_per_domain,
        "base_scenarios": base_scenarios,
        "counterfactual_variants_per_base": config.counterfactual_variants_per_case,
        "counterfactual_scenarios": counterfactual_scenarios,
        "counterfactual_axis_coverage": counterfactual_axis_coverage(
            config.counterfactual_variants_per_case,
            base_scenarios=base_scenarios,
        ),
        "num_cases": total_case_variants,
        "total_case_variants": total_case_variants,
        "events_per_case": EVENTS_PER_CASE,
        "memories_per_case": MEMORIES_PER_CASE,
        "queries_per_case": QUERIES_PER_CASE,
        "state_contracts_per_case": STATE_CONTRACTS_PER_CASE,
        "base_events": base_scenarios * EVENTS_PER_CASE,
        "base_memories": base_scenarios * MEMORIES_PER_CASE,
        "base_queries": base_scenarios * QUERIES_PER_CASE,
        "total_events_with_counterfactuals": total_case_variants * EVENTS_PER_CASE,
        "total_memories_with_counterfactuals": total_case_variants * MEMORIES_PER_CASE,
        "total_queries_with_counterfactuals": total_case_variants * QUERIES_PER_CASE,
        "total_state_contracts_with_counterfactuals": total_case_variants * STATE_CONTRACTS_PER_CASE,
        "final_main_scale_checks": _final_main_scale_checks(
            base_scenarios=base_scenarios,
            counterfactual_scenarios=counterfactual_scenarios,
            base_events=base_scenarios * EVENTS_PER_CASE,
            base_memories=base_scenarios * MEMORIES_PER_CASE,
            base_queries=base_scenarios * QUERIES_PER_CASE,
        ),
    }


def benchmark_construction_summary(benchmark: Benchmark) -> dict[str, Any]:
    domains = Counter(case.domain for case in benchmark.cases)
    variants = Counter(case.difficulty.values.get("counterfactual_variant", 0) for case in benchmark.cases)
    edits = Counter(str(case.difficulty.values.get("counterfactual_edit", "base")) for case in benchmark.cases)
    return {
        "benchmark_id": benchmark.benchmark_id,
        "name": benchmark.name,
        "num_domains": len(domains),
        "domains": dict(sorted(domains.items())),
        "base_scenarios": variants.get(0, 0),
        "counterfactual_scenarios": sum(count for variant, count in variants.items() if variant),
        "counterfactual_edits": dict(sorted(edits.items())),
        "num_cases": len(benchmark.cases),
        "total_case_variants": len(benchmark.cases),
        "num_queries": sum(len(case.queries) for case in benchmark.cases),
        "num_memories": sum(len(case.gold_memory_units) for case in benchmark.cases),
        "num_events": sum(len(case.events) for case in benchmark.cases),
        "num_event_edges": sum(len(case.event_edges) for case in benchmark.cases),
        "num_state_contracts": sum(len(case.state_contracts) for case in benchmark.cases),
        "num_sessions": sum(len(case.sessions) for case in benchmark.cases),
        "num_turns": sum(len(session.turns) for case in benchmark.cases for session in case.sessions),
        "probe_types": dict(
            sorted(Counter(query.probe_type for case in benchmark.cases for query in case.queries).items())
        ),
        "event_types": dict(
            sorted(Counter(event.event_type for case in benchmark.cases for event in case.events).items())
        ),
    }


def _final_main_scale_checks(
    *,
    base_scenarios: int,
    counterfactual_scenarios: int,
    base_events: int,
    base_memories: int,
    base_queries: int,
) -> dict[str, bool]:
    return {
        "base_scenarios_eq_1200": base_scenarios == 1200,
        "counterfactual_scenarios_ge_2400": counterfactual_scenarios >= 2400,
        "base_events_in_24k_to_60k": 24_000 <= base_events <= 60_000,
        "base_memories_in_18k_to_48k": 18_000 <= base_memories <= 48_000,
        "base_queries_in_9600_to_30000": 9_600 <= base_queries <= 30_000,
    }
