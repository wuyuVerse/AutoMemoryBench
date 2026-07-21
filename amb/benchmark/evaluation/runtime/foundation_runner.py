"""Run OpenAI-compatible foundation models on AutoMemoryBench protocols."""

from __future__ import annotations

from dataclasses import asdict
import json
import re
import time
from pathlib import Path
from typing import Any

from amb.benchmark.evaluation.openai_compatible import OpenAICompatibleChatClient, parse_json_response
from amb.benchmark.release.evaluation import _resolve_path, _split_entries
from amb.benchmark.schemas.io import load_benchmark, read_json
from amb.benchmark.schemas.models import (
    Benchmark,
    Cost,
    MemoryOperation,
    PredictionSet,
    QueryPrediction,
    SCHEMA_VERSION,
)


FOUNDATION_PROTOCOLS = ("query_only", "full_history", "oracle_state")


def run_foundation_model(
    benchmark: Benchmark,
    *,
    client: OpenAICompatibleChatClient,
    model: str,
    protocol: str,
    system_id: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> PredictionSet:
    if protocol not in FOUNDATION_PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}")
    predictions: list[QueryPrediction] = []
    for case in benchmark.cases:
        state_contracts = {contract.state_contract_id: contract for contract in case.state_contracts}
        memories_by_id = {memory.memory_id: memory for memory in case.gold_memory_units}
        for query in case.queries:
            start = time.perf_counter()
            messages = _build_messages(case, query, memories_by_id, state_contracts, protocol)
            completion = client.create_chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json=True,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            payload = parse_json_response(completion.content)
            predictions.append(
                _prediction_from_payload(
                    query_id=query.query_id,
                    payload=payload,
                    protocol=protocol,
                    memories_by_id=memories_by_id,
                    cost=_cost_from_usage(completion.usage, latency_ms),
                )
            )
    return PredictionSet(schema_version=SCHEMA_VERSION, system_id=system_id, predictions=tuple(predictions))


def run_foundation_model_release_split(
    manifest_path: str | Path,
    *,
    split: str,
    client: OpenAICompatibleChatClient,
    model: str,
    protocol: str,
    system_id: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> PredictionSet:
    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    entries = _split_entries(manifest.get("split_files", {}).get(split))
    if not entries:
        raise ValueError(f"manifest has no benchmark artifacts for split {split!r}")
    predictions: list[QueryPrediction] = []
    seen_query_ids: set[str] = set()
    for _, raw_path in entries:
        benchmark = load_benchmark(_resolve_path(raw_path, manifest_file.parent))
        result = run_foundation_model(
            benchmark,
            client=client,
            model=model,
            protocol=protocol,
            system_id=system_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for prediction in result.predictions:
            if prediction.query_id in seen_query_ids:
                raise ValueError(f"duplicate prediction query_id across release shards: {prediction.query_id}")
            seen_query_ids.add(prediction.query_id)
            predictions.append(prediction)
    return PredictionSet(schema_version=SCHEMA_VERSION, system_id=system_id, predictions=tuple(predictions))


def _build_messages(
    case: Any,
    query: Any,
    memories_by_id: dict[str, Any],
    state_contracts: dict[str, Any],
    protocol: str,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": _system_prompt()}]
    if protocol == "query_only":
        user_content = _query_only_prompt(case, query)
    elif protocol == "full_history":
        user_content = _full_history_prompt(case, query)
    else:
        user_content = _oracle_state_prompt(case, query, memories_by_id, state_contracts)
    messages.append({"role": "user", "content": user_content})
    return messages


def _system_prompt() -> str:
    return (
        "You are being evaluated on agent memory tasks.\n"
        "Return exactly one JSON object and nothing else.\n"
        "Required keys: memory_needed, activated_memory_ids, response, compression_summary, tool_name, parameters, memory_operations.\n"
        "Rules:\n"
        "- memory_needed: true or false.\n"
        "- activated_memory_ids: list of memory ids actually used. Use [] if none or unknown.\n"
        "- response: only the final user-facing answer. Do not include JSON, metadata, keys, or analysis in response.\n"
        "- compression_summary: string or null. For compression_probe, put the scored durable-memory summary here. For all other probes, use null.\n"
        "- tool_name: string or null.\n"
        "- parameters: object.\n"
        "- memory_operations: list of objects with keys operation, memory_id, content. operation must be one of write, update, delete, skip.\n"
        "- If the task requires refusal, refuse in the response and do not reveal forbidden or sensitive content.\n"
        "- Never place another JSON object inside response.\n"
        "- If the available context is insufficient, still return valid JSON and answer conservatively."
    )


def _query_only_prompt(case: Any, query: Any) -> str:
    payload = {
        "benchmark_mode": "AutoMemoryBench foundation-model evaluation",
        "protocol": "query_only",
        "case_id": case.case_id,
        "domain": case.domain,
        "task_type": query.task_type,
        "probe_type": query.probe_type,
        "query_id": query.query_id,
        "timestamp": query.timestamp,
        "user_query": query.prompt,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _full_history_prompt(case: Any, query: Any) -> str:
    sessions = []
    for session in sorted(case.sessions, key=lambda item: item.timestamp or ""):
        sessions.append(
            {
                "session_id": session.session_id,
                "timestamp": session.timestamp,
                "turns": [{"turn_id": turn.turn_id, "role": turn.role, "content": turn.content} for turn in session.turns],
            }
        )
    payload = {
        "benchmark_mode": "AutoMemoryBench foundation-model evaluation",
        "protocol": "full_history",
        "case_id": case.case_id,
        "domain": case.domain,
        "task_type": query.task_type,
        "probe_type": query.probe_type,
        "query_id": query.query_id,
        "timestamp": query.timestamp,
        "conversation_history": sessions,
        "user_query": query.prompt,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _oracle_state_prompt(
    case: Any,
    query: Any,
    memories_by_id: dict[str, Any],
    state_contracts: dict[str, Any],
) -> str:
    contract = state_contracts.get(query.state_contract_id) if query.state_contract_id else None
    allowed_ids = list(contract.active_memory_ids) if contract is not None else []
    blocked_ids = {
        "forbidden_memory_ids": list(contract.forbidden_memory_ids) if contract is not None else [],
        "deleted_memory_ids": list(contract.deleted_memory_ids) if contract is not None else [],
        "superseded_memory_ids": list(contract.superseded_memory_ids) if contract is not None else [],
        "restricted_memory_ids": list(contract.restricted_memory_ids) if contract is not None else [],
    }
    memory_state = []
    for memory_id in allowed_ids:
        memory = memories_by_id.get(memory_id)
        if memory is None:
            continue
        memory_state.append(
            {
                "memory_id": memory.memory_id,
                "type": memory.type,
                "content": memory.content,
                "valid_from": memory.valid_from,
                "valid_until": memory.valid_until,
                "status": memory.status,
                "authorization_scope": memory.authorization_scope,
                "privacy_level": memory.privacy_level,
            }
        )
    payload = {
        "benchmark_mode": "AutoMemoryBench foundation-model evaluation",
        "protocol": "oracle_state",
        "case_id": case.case_id,
        "domain": case.domain,
        "task_type": query.task_type,
        "probe_type": query.probe_type,
        "query_id": query.query_id,
        "timestamp": query.timestamp,
        "memory_state": memory_state,
        "blocked_state": blocked_ids,
        "query_forbidden_memory_ids": list(query.forbidden_memory_ids),
        "user_query": query.prompt,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _prediction_from_payload(
    *,
    query_id: str,
    payload: dict[str, Any],
    protocol: str,
    memories_by_id: dict[str, Any],
    cost: Cost,
) -> QueryPrediction:
    del protocol
    payload = _maybe_unwrap_nested_payload(payload)
    known_ids = set(memories_by_id)
    operations = tuple(_coerce_memory_operation(item) for item in payload.get("memory_operations", []))
    operations = _align_memory_operations(operations, memories_by_id)
    activated = _coerce_activated_memory_ids(payload.get("activated_memory_ids"), known_ids)
    activated = _augment_activated_memory_ids(
        activated,
        payload=payload,
        operations=operations,
        memories_by_id=memories_by_id,
    )
    if not operations:
        operations = (MemoryOperation(operation="skip", memory_id=None, content=None),)
    tool_name = payload.get("tool_name")
    return QueryPrediction(
        query_id=query_id,
        memory_needed=_optional_bool(payload.get("memory_needed")),
        activated_memory_ids=activated,
        response=str(payload.get("response", "")),
        compression_summary=None if payload.get("compression_summary") is None else str(payload.get("compression_summary")),
        tool_name=None if tool_name is None else str(tool_name),
        parameters=_coerce_parameters(payload.get("parameters")),
        memory_operations=operations,
        cost=cost,
    )


def _coerce_activated_memory_ids(value: Any, known_ids: set[str]) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    for item in value:
        memory_id = str(item)
        if memory_id in known_ids and memory_id not in seen:
            ordered.append(memory_id)
            seen.add(memory_id)
    return tuple(ordered)


def _coerce_parameters(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _coerce_memory_operation(value: Any) -> MemoryOperation:
    if not isinstance(value, dict):
        return MemoryOperation(operation="skip", memory_id=None, content=None)
    operation = str(value.get("operation", "skip"))
    if operation not in {"write", "update", "delete", "skip"}:
        operation = "skip"
    memory_id = value.get("memory_id")
    content = value.get("content")
    return MemoryOperation(
        operation=operation,
        memory_id=None if memory_id is None else str(memory_id),
        content=None if content is None else str(content),
    )


def _maybe_unwrap_nested_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) == {"response"} and isinstance(payload.get("response"), str):
        nested = parse_json_response(str(payload["response"]))
        if _looks_like_prediction_payload(nested):
            return nested
    response = payload.get("response")
    if isinstance(response, str):
        nested = parse_json_response(response)
        if _looks_like_prediction_payload(nested):
            return nested
    return payload


def _looks_like_prediction_payload(payload: dict[str, Any]) -> bool:
    return isinstance(payload, dict) and any(
        key in payload
        for key in ("memory_needed", "activated_memory_ids", "compression_summary", "tool_name", "parameters", "memory_operations")
    )


def _align_memory_operations(
    operations: tuple[MemoryOperation, ...],
    memories_by_id: dict[str, Any],
) -> tuple[MemoryOperation, ...]:
    aligned: list[MemoryOperation] = []
    for operation in operations:
        memory_id = operation.memory_id if operation.memory_id in memories_by_id else None
        if memory_id is None and operation.content:
            memory_id = _best_memory_match(operation.content, memories_by_id)
        aligned.append(
            MemoryOperation(
                operation=operation.operation,
                memory_id=memory_id,
                content=operation.content,
            )
        )
    return tuple(aligned)


def _augment_activated_memory_ids(
    activated: tuple[str, ...],
    *,
    payload: dict[str, Any],
    operations: tuple[MemoryOperation, ...],
    memories_by_id: dict[str, Any],
) -> tuple[str, ...]:
    ordered = list(activated)
    seen = set(activated)
    for operation in operations:
        if operation.memory_id and operation.memory_id not in seen:
            ordered.append(operation.memory_id)
            seen.add(operation.memory_id)
    candidate_texts: list[str] = []
    response = payload.get("response")
    if isinstance(response, str) and response:
        candidate_texts.append(response)
    parameters = payload.get("parameters")
    if isinstance(parameters, dict):
        candidate_texts.extend(str(value) for value in parameters.values() if value is not None)
    candidate_texts.extend(operation.content for operation in operations if operation.content)
    for text in candidate_texts:
        memory_id = _best_memory_match(text, memories_by_id)
        if memory_id and memory_id not in seen:
            ordered.append(memory_id)
            seen.add(memory_id)
    return tuple(ordered)


def _best_memory_match(text: str, memories_by_id: dict[str, Any]) -> str | None:
    candidate_norm = _normalize_text(text)
    candidate_tokens = _content_tokens(text)
    best_memory_id: str | None = None
    best_score = (0.0, 0)
    for memory_id, memory in memories_by_id.items():
        memory_text = getattr(memory, "content", "")
        if not memory_text:
            continue
        memory_norm = _normalize_text(memory_text)
        if memory_norm and (memory_norm in candidate_norm or candidate_norm in memory_norm and len(candidate_norm) >= 24):
            return memory_id
        memory_tokens = _content_tokens(memory_text)
        if not memory_tokens:
            continue
        overlap = candidate_tokens & memory_tokens
        recall = len(overlap) / len(memory_tokens)
        score = (recall, len(overlap))
        if _is_good_memory_match(recall, len(overlap), len(memory_tokens)) and score > best_score:
            best_memory_id = memory_id
            best_score = score
    return best_memory_id


def _is_good_memory_match(recall: float, overlap_count: int, memory_token_count: int) -> bool:
    if overlap_count >= max(4, memory_token_count) and recall >= 0.6:
        return True
    if overlap_count >= 4 and recall >= 0.6:
        return True
    return overlap_count >= 3 and recall >= 0.8


def _normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9#._-]+", text.lower()))


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "by",
        "do",
        "for",
        "from",
        "i",
        "if",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "please",
        "repeat",
        "shared",
        "should",
        "still",
        "that",
        "the",
        "this",
        "to",
        "use",
        "using",
        "was",
        "what",
        "with",
        "without",
    }
    return {token for token in re.findall(r"[a-z0-9#._-]+", text.lower()) if token not in stopwords}


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _cost_from_usage(usage: dict[str, Any], latency_ms: float) -> Cost:
    input_tokens = _first_number(usage, ("prompt_tokens", "input_tokens", "total_prompt_tokens"))
    output_tokens = _first_number(usage, ("completion_tokens", "output_tokens", "total_completion_tokens"))
    return Cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        retrieval_latency_ms=None,
        storage_bytes=None,
    )


def _first_number(source: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def prediction_set_to_dict(predictions: PredictionSet) -> dict[str, Any]:
    return asdict(predictions)
