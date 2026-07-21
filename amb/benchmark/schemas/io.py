"""JSON loading and writing for benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from amb.benchmark.schemas.models import (
    Benchmark,
    Case,
    Cost,
    Difficulty,
    Event,
    EventEdge,
    ExpectedBehavior,
    MemoryOperation,
    MemoryStateContract,
    MemoryUnit,
    PredictionSet,
    Query,
    QueryPrediction,
    ScenarioMetadata,
    ScenarioTimeSpan,
    Session,
    StateTransition,
    Turn,
)


class ArtifactError(ValueError):
    """Raised when an input artifact cannot be parsed."""


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


def load_benchmark(path: str | Path) -> Benchmark:
    data = read_json(path)
    try:
        cases = tuple(_case(item) for item in data["cases"])
        return Benchmark(
            schema_version=str(data.get("schema_version", "")),
            benchmark_id=str(data["benchmark_id"]),
            name=str(data.get("name", data["benchmark_id"])),
            cases=cases,
        )
    except KeyError as exc:
        raise ArtifactError(f"Benchmark missing required field: {exc.args[0]}") from exc


def load_predictions(path: str | Path) -> PredictionSet:
    data = read_json(path)
    try:
        return PredictionSet(
            schema_version=str(data.get("schema_version", "")),
            system_id=str(data["system_id"]),
            predictions=tuple(_prediction(item) for item in data["predictions"]),
        )
    except KeyError as exc:
        raise ArtifactError(f"Prediction file missing required field: {exc.args[0]}") from exc


def _case(data: dict[str, Any]) -> Case:
    difficulty = data.get("difficulty", {})
    if isinstance(difficulty, dict) and isinstance(difficulty.get("values"), dict):
        difficulty = difficulty["values"]
    return Case(
        case_id=str(data["case_id"]),
        domain=str(data["domain"]),
        sessions=tuple(_session(item) for item in data.get("sessions", [])),
        gold_memory_units=tuple(_memory(item) for item in data.get("gold_memory_units", [])),
        queries=tuple(_query(item) for item in data.get("queries", [])),
        events=tuple(_event(item) for item in data.get("events", [])),
        event_edges=tuple(_event_edge(item) for item in data.get("event_edges", [])),
        state_contracts=tuple(_state_contract(item) for item in data.get("state_contracts", [])),
        difficulty=Difficulty(dict(difficulty)),
        scenario_id=_optional_str(data.get("scenario_id")),
        scenario=_scenario(data.get("scenario")),
    )


def _session(data: dict[str, Any]) -> Session:
    return Session(
        session_id=str(data["session_id"]),
        timestamp=_optional_str(data.get("timestamp")),
        turns=tuple(_turn(item) for item in data.get("turns", [])),
    )


def _turn(data: dict[str, Any]) -> Turn:
    return Turn(turn_id=str(data["turn_id"]), role=str(data["role"]), content=str(data["content"]))


def _memory(data: dict[str, Any]) -> MemoryUnit:
    return MemoryUnit(
        memory_id=str(data["memory_id"]),
        type=str(data["type"]),
        content=str(data["content"]),
        source_turn_ids=tuple(str(item) for item in data.get("source_turn_ids", [])),
        scenario_id=_optional_str(data.get("scenario_id")),
        memory_type=_optional_str(data.get("memory_type")),
        canonical_form=dict(data.get("canonical_form", {})),
        source_event_ids=tuple(str(item) for item in data.get("source_event_ids", [])),
        source_trace_ids=tuple(str(item) for item in data.get("source_trace_ids", [])),
        valid_from=_optional_str(data.get("valid_from")),
        valid_until=_optional_str(data.get("valid_until")),
        status=str(data.get("status", "active")),
        importance=_optional_int(data.get("importance")),
        confidence=_optional_float(data.get("confidence")),
        should_store=bool(data.get("should_store", True)),
        should_write=_optional_bool(data.get("should_write")),
        should_delete=bool(data.get("should_delete", False)),
        privacy_level=str(data.get("privacy_level", "normal")),
        sensitivity=_optional_str(data.get("sensitivity")),
        authorization_scope=str(data.get("authorization_scope", "same_user")),
        should_retrieve_for=tuple(str(item) for item in data.get("should_retrieve_for", [])),
        should_not_retrieve_for=tuple(str(item) for item in data.get("should_not_retrieve_for", [])),
        update_of=_optional_str(data.get("update_of")),
        invalidates=tuple(str(item) for item in data.get("invalidates", [])),
        forget_policy=_optional_str(data.get("forget_policy")),
        expected_use=_optional_str(data.get("expected_use")),
    )


def _query(data: dict[str, Any]) -> Query:
    behavior = data.get("expected_behavior", {})
    return Query(
        query_id=str(data["query_id"]),
        timestamp=_optional_str(data.get("timestamp")),
        prompt=str(data["prompt"]),
        task_type=str(data["task_type"]),
        requires_memory=bool(data["requires_memory"]),
        gold_memory_ids=tuple(str(item) for item in data.get("gold_memory_ids", [])),
        expected_behavior=ExpectedBehavior(
            must_include=tuple(str(item) for item in behavior.get("must_include", [])),
            must_not_include=tuple(str(item) for item in behavior.get("must_not_include", [])),
            should_refuse=bool(behavior.get("should_refuse", False)),
            behavior_type=str(behavior.get("behavior_type", "answer")),
            tool_name=_optional_str(behavior.get("tool_name")),
            parameters=dict(behavior.get("parameters", {})),
        ),
        state_contract_id=_optional_str(data.get("state_contract_id")),
        forbidden_memory_ids=tuple(str(item) for item in data.get("forbidden_memory_ids", [])),
        counterfactual_group_id=_optional_str(data.get("counterfactual_group_id")),
        memory_dependency=str(data.get("memory_dependency", "strong")),
        probe_type=_optional_str(data.get("probe_type")),
        scoring_rule=_optional_str(data.get("scoring_rule")),
        difficulty=dict(data.get("difficulty", {})),
    )


def _event(data: dict[str, Any]) -> Event:
    return Event(
        event_id=str(data["event_id"]),
        event_type=str(data["event_type"]),
        timestamp=str(data["timestamp"]),
        subject=str(data["subject"]),
        predicate=str(data["predicate"]),
        object=str(data["object"]),
        source_turn_ids=tuple(str(item) for item in data.get("source_turn_ids", [])),
        attributes=dict(data.get("attributes", {})),
    )


def _event_edge(data: dict[str, Any]) -> EventEdge:
    return EventEdge(
        source_event_id=str(data["source_event_id"]),
        target_event_id=str(data["target_event_id"]),
        edge_type=str(data["edge_type"]),
    )


def _state_contract(data: dict[str, Any]) -> MemoryStateContract:
    return MemoryStateContract(
        state_contract_id=str(data["state_contract_id"]),
        timestamp=str(data["timestamp"]),
        scenario_id=_optional_str(data.get("scenario_id")),
        active_memory_ids=tuple(str(item) for item in data.get("active_memory_ids", [])),
        inactive_memory_ids=tuple(str(item) for item in data.get("inactive_memory_ids", [])),
        deleted_memory_ids=tuple(str(item) for item in data.get("deleted_memory_ids", [])),
        forbidden_memory_ids=tuple(str(item) for item in data.get("forbidden_memory_ids", [])),
        superseded_memory_ids=tuple(str(item) for item in data.get("superseded_memory_ids", [])),
        restricted_memory_ids=tuple(str(item) for item in data.get("restricted_memory_ids", [])),
        required_governance_rules=tuple(str(item) for item in data.get("required_governance_rules", [])),
        transitions=tuple(_transition(item) for item in data.get("transitions", [])),
    )


def _transition(data: dict[str, Any]) -> StateTransition:
    return StateTransition(
        from_memory_id=str(data["from_memory_id"]),
        to_memory_id=str(data["to_memory_id"]),
        transition_type=str(data["transition_type"]),
        trigger_event_id=str(data["trigger_event_id"]),
    )


def _scenario(data: Any) -> ScenarioMetadata | None:
    if not isinstance(data, dict):
        return None
    time_span = data.get("time_span")
    return ScenarioMetadata(
        scenario_id=str(data["scenario_id"]),
        domain=str(data["domain"]),
        actors=tuple(str(item) for item in data.get("actors", [])),
        groups=tuple(str(item) for item in data.get("groups", [])),
        tools=tuple(str(item) for item in data.get("tools", [])),
        time_span=(
            ScenarioTimeSpan(start=str(time_span["start"]), end=str(time_span["end"]))
            if isinstance(time_span, dict) and "start" in time_span and "end" in time_span
            else None
        ),
        memory_policy=dict(data.get("memory_policy", {})),
        difficulty=dict(data.get("difficulty", {})),
        generation_seed=_optional_int(data.get("generation_seed")),
    )


def _prediction(data: dict[str, Any]) -> QueryPrediction:
    cost = data.get("cost", {})
    return QueryPrediction(
        query_id=str(data["query_id"]),
        memory_needed=_optional_bool(data.get("memory_needed")),
        activated_memory_ids=tuple(str(item) for item in data.get("activated_memory_ids", [])),
        response=str(data.get("response", "")),
        compression_summary=_optional_str(data.get("compression_summary")),
        tool_name=_optional_str(data.get("tool_name")),
        parameters=dict(data.get("parameters", {})),
        memory_operations=tuple(_operation(item) for item in data.get("memory_operations", [])),
        cost=Cost(
            input_tokens=_optional_float(cost.get("input_tokens")),
            output_tokens=_optional_float(cost.get("output_tokens")),
            latency_ms=_optional_float(cost.get("latency_ms")),
            retrieval_latency_ms=_optional_float(cost.get("retrieval_latency_ms")),
            storage_bytes=_optional_float(cost.get("storage_bytes")),
        ),
    )


def _operation(data: dict[str, Any]) -> MemoryOperation:
    return MemoryOperation(
        operation=str(data["operation"]),
        memory_id=_optional_str(data.get("memory_id")),
        content=_optional_str(data.get("content")),
    )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)
