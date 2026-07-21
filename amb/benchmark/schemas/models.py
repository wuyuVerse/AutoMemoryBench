"""Dataclasses for benchmark and prediction schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class Turn:
    turn_id: str
    role: str
    content: str


@dataclass(frozen=True)
class Session:
    session_id: str
    timestamp: str | None
    turns: tuple[Turn, ...]


@dataclass(frozen=True)
class Event:
    event_id: str
    event_type: str
    timestamp: str
    subject: str
    predicate: str
    object: str
    source_turn_ids: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EventEdge:
    source_event_id: str
    target_event_id: str
    edge_type: str


@dataclass(frozen=True)
class MemoryUnit:
    memory_id: str
    type: str
    content: str
    source_turn_ids: tuple[str, ...]
    scenario_id: str | None = None
    memory_type: str | None = None
    canonical_form: dict[str, Any] = field(default_factory=dict)
    source_event_ids: tuple[str, ...] = ()
    source_trace_ids: tuple[str, ...] = ()
    valid_from: str | None = None
    valid_until: str | None = None
    status: str = "active"
    importance: int | None = None
    confidence: float | None = None
    should_store: bool = True
    should_write: bool | None = None
    should_delete: bool = False
    privacy_level: str = "normal"
    sensitivity: str | None = None
    authorization_scope: str = "same_user"
    should_retrieve_for: tuple[str, ...] = ()
    should_not_retrieve_for: tuple[str, ...] = ()
    update_of: str | None = None
    invalidates: tuple[str, ...] = ()
    forget_policy: str | None = None
    expected_use: str | None = None

    @property
    def is_sensitive(self) -> bool:
        level = (self.sensitivity or self.privacy_level).lower()
        return level in {"sensitive", "restricted", "forbidden"}


@dataclass(frozen=True)
class ExpectedBehavior:
    must_include: tuple[str, ...] = ()
    must_not_include: tuple[str, ...] = ()
    should_refuse: bool = False
    behavior_type: str = "answer"
    tool_name: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioTimeSpan:
    start: str
    end: str


@dataclass(frozen=True)
class ScenarioMetadata:
    scenario_id: str
    domain: str
    actors: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    time_span: ScenarioTimeSpan | None = None
    memory_policy: dict[str, Any] = field(default_factory=dict)
    difficulty: dict[str, Any] = field(default_factory=dict)
    generation_seed: int | None = None


@dataclass(frozen=True)
class StateTransition:
    from_memory_id: str
    to_memory_id: str
    transition_type: str
    trigger_event_id: str


@dataclass(frozen=True)
class MemoryStateContract:
    state_contract_id: str
    timestamp: str
    scenario_id: str | None = None
    active_memory_ids: tuple[str, ...] = ()
    inactive_memory_ids: tuple[str, ...] = ()
    deleted_memory_ids: tuple[str, ...] = ()
    forbidden_memory_ids: tuple[str, ...] = ()
    superseded_memory_ids: tuple[str, ...] = ()
    restricted_memory_ids: tuple[str, ...] = ()
    required_governance_rules: tuple[str, ...] = ()
    transitions: tuple[StateTransition, ...] = ()


@dataclass(frozen=True)
class Query:
    query_id: str
    timestamp: str | None
    prompt: str
    task_type: str
    requires_memory: bool
    gold_memory_ids: tuple[str, ...]
    expected_behavior: ExpectedBehavior
    state_contract_id: str | None = None
    forbidden_memory_ids: tuple[str, ...] = ()
    counterfactual_group_id: str | None = None
    memory_dependency: str = "strong"
    probe_type: str | None = None
    scoring_rule: str | None = None
    difficulty: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Difficulty:
    values: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Case:
    case_id: str
    domain: str
    sessions: tuple[Session, ...]
    gold_memory_units: tuple[MemoryUnit, ...]
    queries: tuple[Query, ...]
    events: tuple[Event, ...] = ()
    event_edges: tuple[EventEdge, ...] = ()
    state_contracts: tuple[MemoryStateContract, ...] = ()
    difficulty: Difficulty = field(default_factory=Difficulty)
    scenario_id: str | None = None
    scenario: ScenarioMetadata | None = None


@dataclass(frozen=True)
class Benchmark:
    schema_version: str
    benchmark_id: str
    name: str
    cases: tuple[Case, ...]


@dataclass(frozen=True)
class MemoryOperation:
    operation: str
    memory_id: str | None = None
    content: str | None = None


@dataclass(frozen=True)
class Cost:
    input_tokens: float | None = None
    output_tokens: float | None = None
    latency_ms: float | None = None
    retrieval_latency_ms: float | None = None
    storage_bytes: float | None = None


@dataclass(frozen=True)
class QueryPrediction:
    query_id: str
    memory_needed: bool | None
    activated_memory_ids: tuple[str, ...]
    response: str
    compression_summary: str | None = None
    tool_name: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    memory_operations: tuple[MemoryOperation, ...] = ()
    cost: Cost = field(default_factory=Cost)


@dataclass(frozen=True)
class PredictionSet:
    schema_version: str
    system_id: str
    predictions: tuple[QueryPrediction, ...]


@dataclass(frozen=True)
class QueryContext:
    case: Case
    query: Query
    gold_memories: tuple[MemoryUnit, ...]
    all_memories_by_id: dict[str, MemoryUnit]
