"""Offline runner for black-box AutoMemoryBench agent adapters."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any, Callable
import uuid

from amb.benchmark.evaluation.adapters import BlackBoxAgent
from amb.benchmark.evaluation.framework_trace import framework_trace_record
from amb.benchmark.schemas.models import (
    Case,
    Benchmark,
    Cost,
    MemoryOperation,
    PredictionSet,
    QueryPrediction,
    SCHEMA_VERSION,
)


def run_case_with_agent(
    case: Case,
    agent: BlackBoxAgent,
    *,
    state_callback: Callable[[dict[str, Any]], None] | None = None,
    framework_trace_callback: Callable[[dict[str, Any]], None] | None = None,
    agent_system: dict[str, Any] | None = None,
    failure_predictions: bool = False,
    prediction_callback: Callable[[QueryPrediction], None] | None = None,
) -> tuple[QueryPrediction, ...]:
    """Run one agent over one case and return case-local predictions."""

    predictions: list[QueryPrediction] = []
    if hasattr(agent, "set_case_reference"):
        agent.set_case_reference(case)
    agent.reset(case.case_id)
    total_turns = sum(len(session.turns) for session in case.sessions)
    observed_turns = 0
    for session in sorted(case.sessions, key=lambda item: item.timestamp or ""):
        for turn in session.turns:
            observed_turns += 1
            _emit_state(
                state_callback,
                {
                    "event_type": "observe",
                    "phase": "observing",
                    "case_id": case.case_id,
                    "session_id": session.session_id,
                    "turn_id": turn.turn_id,
                    "observed_turns": observed_turns,
                    "total_turns": total_turns,
                },
            )
            agent.observe(
                {
                    "case_id": case.case_id,
                    "domain": case.domain,
                    "session_id": session.session_id,
                    "timestamp": session.timestamp,
                    "turn_id": turn.turn_id,
                    "role": turn.role,
                    "content": turn.content,
                }
            )
    total_queries = len(case.queries)
    for index, query in enumerate(case.queries, start=1):
        _emit_state(
            state_callback,
            {
                "event_type": "query_start",
                "phase": "querying",
                "case_id": case.case_id,
                "query_id": query.query_id,
                "query_index": index,
                "case_total_queries": total_queries,
            },
        )
        request = {
            "case_id": case.case_id,
            "domain": case.domain,
            "query_id": query.query_id,
            "timestamp": query.timestamp,
            "prompt": query.prompt,
            "task_type": query.task_type,
            "probe_type": query.probe_type,
            "scoring_rule": query.scoring_rule,
            "state_contract_id": query.state_contract_id,
            "memory_dependency": query.memory_dependency,
        }
        try:
            raw = agent.answer_or_act(request)
        except Exception as exc:
            if not failure_predictions:
                raise
            raw = _failure_response(query.query_id, exc)
        _emit_framework_trace(
            framework_trace_callback,
            agent,
            agent_system=agent_system,
            case_id=case.case_id,
            query_id=query.query_id,
            raw_response=raw,
        )
        prediction = _prediction_from_response(query.query_id, raw)
        predictions.append(prediction)
        if prediction_callback is not None:
            prediction_callback(prediction)
        _emit_state(
            state_callback,
            {
                "event_type": "query_finish",
                "phase": "querying",
                "case_id": case.case_id,
                "query_id": query.query_id,
                "query_index": index,
                "case_total_queries": total_queries,
            },
        )
    return tuple(predictions)


def run_black_box_agent(
    benchmark: Benchmark,
    agent: BlackBoxAgent,
    system_id: str,
    *,
    framework_trace_callback: Callable[[dict[str, Any]], None] | None = None,
    agent_system: dict[str, Any] | None = None,
) -> PredictionSet:
    """Run an agent in streaming-system-memory mode over a benchmark."""

    predictions: list[QueryPrediction] = []
    for case in benchmark.cases:
        predictions.extend(
            run_case_with_agent(
                case,
                agent,
                framework_trace_callback=framework_trace_callback,
                agent_system=agent_system,
            )
        )
    return PredictionSet(schema_version=SCHEMA_VERSION, system_id=system_id, predictions=tuple(predictions))


def write_prediction_checkpoint(
    checkpoint_path: str | Path,
    *,
    system_id: str,
    predictions: list[QueryPrediction] | tuple[QueryPrediction, ...],
) -> None:
    """Write one resumable prediction checkpoint."""
    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "system_id": system_id,
        "predictions": [asdict(item) for item in predictions],
    }
    temp_path = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        temp_path.replace(target)
    finally:
        temp_path.unlink(missing_ok=True)


def write_run_state_checkpoint(state_path: str | Path, state: dict[str, Any]) -> None:
    """Write one resumable live-run state checkpoint."""
    target = Path(state_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        temp_path.replace(target)
    finally:
        temp_path.unlink(missing_ok=True)


def _prediction_from_response(query_id: str, raw: dict[str, Any]) -> QueryPrediction:
    if not isinstance(raw, dict):
        raise TypeError(f"agent response for {query_id} must be a dict")
    if "response" not in raw:
        legacy_keys = sorted({"answer", "retrieved_memory_ids", "supporting_evidence"} & set(raw))
        if legacy_keys:
            raise ValueError(
                "agent response for "
                f"{query_id} uses legacy keys {legacy_keys}; return benchmark-compatible "
                "keys response, activated_memory_ids, memory_needed, parameters, "
                "memory_operations, and cost"
            )
        raise ValueError(f"agent response for {query_id} must include a response field")
    if "retrieved_memory_ids" in raw and "activated_memory_ids" not in raw:
        raise ValueError(
            "agent response for "
            f"{query_id} uses retrieved_memory_ids without activated_memory_ids; "
            "return activated_memory_ids to identify memories actually used"
        )
    return QueryPrediction(
        query_id=query_id,
        memory_needed=_optional_bool(raw.get("memory_needed")),
        activated_memory_ids=tuple(str(item) for item in raw.get("activated_memory_ids", [])),
        response=str(raw.get("response", "")),
        compression_summary=None if raw.get("compression_summary") is None else str(raw.get("compression_summary")),
        tool_name=None if raw.get("tool_name") is None else str(raw.get("tool_name")),
        parameters=dict(raw.get("parameters", {})),
        memory_operations=tuple(_memory_operation(item) for item in raw.get("memory_operations", [])),
        cost=_cost(raw.get("cost", {})),
    )


def _failure_response(query_id: str, exc: Exception) -> dict[str, Any]:
    return {
        "response": "",
        "memory_needed": None,
        "activated_memory_ids": [],
        "parameters": {
            "amb_query_failed": True,
            "amb_failure_query_id": query_id,
            "amb_failure_type": type(exc).__name__,
            "amb_failure_message": str(exc)[:1000],
        },
        "memory_operations": [],
        "cost": {},
    }


def _memory_operation(raw: dict[str, Any]) -> MemoryOperation:
    return MemoryOperation(
        operation=str(raw.get("operation", "skip")),
        memory_id=None if raw.get("memory_id") is None else str(raw.get("memory_id")),
        content=None if raw.get("content") is None else str(raw.get("content")),
    )


def _cost(raw: dict[str, Any]) -> Cost:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("cost must be a dict")
    return Cost(
        input_tokens=_optional_float(raw.get("input_tokens")),
        output_tokens=_optional_float(raw.get("output_tokens")),
        latency_ms=_optional_float(raw.get("latency_ms")),
        retrieval_latency_ms=_optional_float(raw.get("retrieval_latency_ms")),
        storage_bytes=_optional_float(raw.get("storage_bytes")),
    )


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _emit_state(callback: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]) -> None:
    if callback is not None:
        callback(dict(payload))


def _emit_framework_trace(
    callback: Callable[[dict[str, Any]], None] | None,
    agent: BlackBoxAgent,
    *,
    agent_system: dict[str, Any] | None,
    case_id: str,
    query_id: str,
    raw_response: dict[str, Any],
) -> None:
    if callback is None:
        return
    exported_trace = None
    export_trace = getattr(agent, "export_trace", None)
    if callable(export_trace):
        exported = export_trace()
        if isinstance(exported, dict):
            exported_trace = exported
    callback(
        framework_trace_record(
            agent_system=agent_system or {},
            case_id=case_id,
            query_id=query_id,
            raw_response=raw_response,
            exported_trace=exported_trace,
        )
    )
