"""Event-graph-first benchmark generation for AutoMemoryBench main datasets."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import random
from typing import Any, Mapping

from amb.benchmark.generation.compilers import (
    build_event_graph,
    compile_event_edges,
    compile_memories,
    compile_model_events,
    compile_state_contracts,
)
from amb.benchmark.generation.domains.counterfactual import counterfactual_axes_for_variants
from amb.benchmark.generation.domains.specs import counterfactual_spec, select_domain_specs
from amb.benchmark.generation.probes.factory import compile_queries
from amb.benchmark.generation.renderers import compile_sessions, event_source_turns
from amb.benchmark.generation.stress import apply_stress_profile
from amb.benchmark.generation.types import CaseGroupPlan, DomainSpec, GenerationConfig
from amb.benchmark.schemas.models import Benchmark, Case, Difficulty, SCHEMA_VERSION, ScenarioMetadata, ScenarioTimeSpan


def generate_main_dataset(case_count_per_domain: int = 10, seed: int = 13) -> Benchmark:
    """Generate a deterministic AutoMemoryBench main-dataset slice."""

    return generate_benchmark(GenerationConfig(case_count_per_domain=case_count_per_domain, seed=seed))


def generate_benchmark(config: GenerationConfig | Mapping[str, Any] | None = None) -> Benchmark:
    """Generate a Benchmark compatible with :mod:`amb.benchmark.schemas.models`.

    The generator is event-graph-first: each case starts from typed lifecycle
    events, then sessions, gold memory units, and probes are compiled from the
    same event graph.
    """

    cfg = _coerce_config(config)
    if cfg.case_count_per_domain < 1:
        raise ValueError("case_count_per_domain must be >= 1")
    if cfg.counterfactual_variants_per_case < 0:
        raise ValueError("counterfactual_variants_per_case must be >= 0")

    specs = select_domain_specs(cfg.domains)
    counterfactual_axes = _counterfactual_axes(cfg.counterfactual_variants_per_case)
    cases: list[Case] = []
    for plan in plan_case_groups(cfg):
        cases.extend(generate_case_group(cfg, plan, counterfactual_axes=counterfactual_axes, specs=specs))

    return Benchmark(
        schema_version=SCHEMA_VERSION,
        benchmark_id=cfg.benchmark_id,
        name=cfg.name,
        cases=tuple(cases),
    )


def plan_case_groups(config: GenerationConfig | Mapping[str, Any] | None = None) -> tuple[CaseGroupPlan, ...]:
    """Plan deterministic base scenario seeds without materializing cases."""

    cfg = _coerce_config(config)
    rng = random.Random(cfg.seed)
    plans: list[CaseGroupPlan] = []
    for spec in select_domain_specs(cfg.domains):
        for index in range(cfg.case_count_per_domain):
            plans.append(
                CaseGroupPlan(
                    domain=spec.domain,
                    index=index,
                    case_seed=rng.randrange(1_000_000_000),
                    counterfactual_group_id=f"cf_{spec.domain}_{index + 1:04d}",
                )
            )
    return tuple(plans)


def generate_case_group(
    config: GenerationConfig | Mapping[str, Any],
    plan: CaseGroupPlan,
    *,
    counterfactual_axes: frozenset[str] | None = None,
    specs: tuple[DomainSpec, ...] | None = None,
) -> tuple[Case, ...]:
    """Generate one base scenario plus configured counterfactual variants."""

    cfg = _coerce_config(config)
    specs_by_domain = {spec.domain: spec for spec in (specs or select_domain_specs(cfg.domains))}
    base_spec = _case_indexed_spec(specs_by_domain[plan.domain], plan.index)
    spec, stress_profile = apply_stress_profile(base_spec, plan.counterfactual_group_id)
    axes = counterfactual_axes or _counterfactual_axes(cfg.counterfactual_variants_per_case)
    cases = [
        _generate_case(
            spec,
            plan.index,
            plan.case_seed,
            group_id=plan.counterfactual_group_id,
            counterfactual_index=0,
            counterfactual_axes=axes,
            stress_profile=stress_profile,
        )
    ]
    for cf_index in range(1, cfg.counterfactual_variants_per_case + 1):
        cases.append(
            _generate_case(
                counterfactual_spec(spec, cf_index),
                plan.index,
                plan.case_seed,
                group_id=plan.counterfactual_group_id,
                counterfactual_index=cf_index,
                counterfactual_axes=axes,
                stress_profile=stress_profile,
            )
        )
    return tuple(cases)


def _coerce_config(config: GenerationConfig | Mapping[str, Any] | None) -> GenerationConfig:
    if config is None:
        return GenerationConfig()
    if isinstance(config, GenerationConfig):
        return config
    domains = config.get("domains")
    return GenerationConfig(
        case_count_per_domain=int(config.get("case_count_per_domain", 10)),
        seed=int(config.get("seed", 13)),
        benchmark_id=str(config.get("benchmark_id", "amst-main-generated")),
        name=str(config.get("name", "AutoMemoryBench Main Generated")),
        domains=tuple(str(item) for item in domains) if domains is not None else None,
        counterfactual_variants_per_case=int(config.get("counterfactual_variants_per_case", 2)),
    )


def _generate_case(
    spec: DomainSpec,
    index: int,
    seed: int,
    *,
    group_id: str,
    counterfactual_index: int,
    counterfactual_axes: frozenset[str],
    stress_profile,
) -> Case:
    variant = seed % 997
    case_id = f"case_{spec.domain}_{index + 1:04d}"
    if counterfactual_index:
        case_id = f"{case_id}_cf{counterfactual_index:02d}"
    start = datetime(2026, 1, 1, 9, 0, tzinfo=UTC) + timedelta(days=index * 11 + variant % 5)
    events = build_event_graph(spec, start, variant)
    turn_by_event = event_source_turns(case_id, events)
    sessions = compile_sessions(spec, case_id, events)
    memories = compile_memories(case_id, events, turn_by_event)
    queries = compile_queries(spec, case_id, events, start + timedelta(days=30), group_id, counterfactual_axes)
    model_events = compile_model_events(case_id, events, turn_by_event)
    event_edges = compile_event_edges(events)
    state_contracts = compile_state_contracts(case_id, events)
    difficulty = Difficulty(
        {
            "num_sessions": len(sessions),
            "num_turns": sum(len(session.turns) for session in sessions),
            "context_tokens": 1100 + (variant % 9) * 90,
            "distractor_ratio": 0.20,
            "conflict_count": 1,
            "sensitive_memory_count": 1,
            "deleted_memory_count": 1,
            "reasoning_hops": 2 + (variant % 2),
            "event_count": len(events),
            "counterfactual_group_id": group_id,
            "counterfactual_variant": counterfactual_index,
            "counterfactual_edit": spec.counterfactual_edit,
            "counterfactual_axes_enabled": tuple(sorted(counterfactual_axes)),
            "stress_family": stress_profile.family,
            "stress_tags": stress_profile.tags,
            "hidden_priority": stress_profile.hidden_priority,
        }
    )
    return Case(
        case_id=case_id,
        domain=spec.domain,
        sessions=sessions,
        gold_memory_units=memories,
        queries=queries,
        events=model_events,
        event_edges=event_edges,
        state_contracts=state_contracts,
        difficulty=difficulty,
        scenario_id=case_id,
        scenario=_scenario_metadata(
            case_id=case_id,
            spec=spec,
            start=start,
            events=events,
            difficulty=difficulty.values,
            seed=seed,
        ),
    )


def _counterfactual_axes(variants_per_case: int) -> frozenset[str]:
    return counterfactual_axes_for_variants(variants_per_case)


def _case_indexed_spec(spec: DomainSpec, index: int) -> DomainSpec:
    """Derive a deterministic same-domain scenario variant.

    The first scenario keeps the canonical domain-pack wording unchanged for
    stable examples. Later scenarios vary subjects and answer-bearing values so
    large releases do not become near-duplicate copies of one domain template.
    """

    if index == 0:
        return spec
    tag = _case_workspace_name(index)
    return replace(
        spec,
        stable_item=_with_scope(spec.stable_item, tag),
        stable_value=_with_scope(spec.stable_value, tag),
        mutable_item=_with_scope(spec.mutable_item, tag),
        old_value=_with_scope(spec.old_value, tag),
        new_value=_with_scope(spec.new_value, tag),
        counterfactual_new_value=_with_scope(spec.counterfactual_new_value, tag),
        deletion_item=_with_scope(spec.deletion_item, tag),
        deleted_value=_with_scope(spec.deleted_value, tag),
        sensitive_item=_with_scope(spec.sensitive_item, tag),
        sensitive_value=_with_secret_scope(spec.sensitive_value, index),
        tool_name=_with_tool_scope(spec.tool_name, index),
        tool_result=_with_scope(spec.tool_result, tag),
        plan_goal=_with_scope(spec.plan_goal, tag),
        plan_constraint=_with_scope(spec.plan_constraint, tag),
        procedure=_with_scope(spec.procedure, tag),
        feedback=_with_scope(spec.feedback, tag),
        task_result=_with_scope(spec.task_result, tag),
        governance_rule=f"{spec.governance_rule} Apply this rule to the {tag}.",
        distractor=_with_scope(spec.distractor, tag),
    )


def _with_scope(value: str, tag: str) -> str:
    return f"{value} for the {tag}"


def _with_secret_scope(value: str, index: int) -> str:
    return f"{value}-s{index + 1:02d}"


def _with_tool_scope(value: str, index: int) -> str:
    return value


def _case_workspace_name(index: int) -> str:
    adjectives = (
        "Atlas",
        "Beacon",
        "Cedar",
        "Delta",
        "Ember",
        "Fjord",
        "Harbor",
        "Indigo",
        "Juniper",
        "Keystone",
        "Lumen",
        "Marble",
        "Nimbus",
        "Orchid",
        "Pioneer",
        "Quarry",
        "River",
        "Summit",
        "Tundra",
        "Northstar",
        "Vertex",
        "Willow",
        "Yarrow",
        "Zephyr",
    )
    nouns = (
        "workspace",
        "project",
        "account",
        "course",
        "portfolio",
        "request board",
        "brief",
        "launch",
    )
    adjective = adjectives[index % len(adjectives)]
    noun = nouns[(index // len(adjectives)) % len(nouns)]
    cycle = index // (len(adjectives) * len(nouns))
    return f"{adjective} {noun}" if cycle == 0 else f"{adjective} {noun} {cycle + 1}"


def _scenario_metadata(
    *,
    case_id: str,
    spec: DomainSpec,
    start: datetime,
    events: tuple,
    difficulty: dict[str, Any],
    seed: int,
) -> ScenarioMetadata:
    end = max(event.timestamp for event in events)
    return ScenarioMetadata(
        scenario_id=case_id,
        domain=spec.domain,
        actors=(spec.actor, "assistant", "policy"),
        groups=(spec.domain,),
        tools=(spec.tool_name,),
        time_span=ScenarioTimeSpan(
            start=start.isoformat().replace("+00:00", "Z"),
            end=end.isoformat().replace("+00:00", "Z"),
        ),
        memory_policy={
            "sensitive_memory_handling": "do_not_store_or_recall_when_forbidden",
            "deleted_memory_handling": "do_not_recall_deleted",
        },
        difficulty=dict(difficulty),
        generation_seed=seed,
    )
