"""Framework-level trace records for agent-system comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from amb.benchmark.evaluation.tool_runtime import (
    ToolRuntimeContract,
    load_tool_runtime_contract,
    validate_tool_call_against_runtime,
)


FRAMEWORK_TRACE_SCHEMA_VERSION = "amst-agent-framework-trace-v1"
FRAMEWORK_TRACE_ARTIFACT_SCHEMA_VERSION = "amst-framework-trace-artifact-v1"
DEFAULT_TOOL_RUNTIME_CONTRACTS = {
    "automemorybench_tool_runtime_v1": Path("configs/tool_runtime/automemorybench_tool_runtime_v1.json")
}

REQUIRED_TRACE_FIELDS = (
    "schema_version",
    "case_id",
    "query_id",
    "framework_id",
    "framework_runtime",
    "orchestration_mode",
    "memory_backend_id",
    "tool_runtime_id",
    "memory_ops",
    "retrieval_hits",
    "tool_calls",
    "planner_trace",
    "handoff_trace",
    "cost",
)

LIST_TRACE_FIELDS = (
    "memory_ops",
    "retrieval_hits",
    "tool_calls",
    "planner_trace",
    "handoff_trace",
)
ASYNC_RESUME_LIST_STATE_FIELDS = (
    "async_events",
    "retry_events",
    "stream_events",
)
ASYNC_RESUME_STRING_STATE_FIELDS = (
    "checkpoint_id",
    "resume_from",
    "state_persistence_mode",
)
ASYNC_EVENT_STRING_FIELDS = (
    "event_id",
    "event_type",
)
RETRY_EVENT_STRING_FIELDS = (
    "retry_id",
    "reason",
)
STREAM_EVENT_STRING_FIELDS = (
    "stream_id",
    "chunk_type",
)
STREAM_EVENT_NUMERIC_FIELDS = (
    "chunk_index",
)
PLANNER_TRACE_STRING_FIELDS = (
    "step",
    "decision",
    "executor",
)
PLANNER_TRACE_LIST_FIELDS = (
    "input_memory_ids",
    "output_memory_ids",
    "tool_call_ids",
)
COST_TRACE_NUMERIC_FIELDS = (
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "retrieval_latency_ms",
    "storage_bytes",
    "tool_call_count",
    "memory_op_count",
)
MEMORY_OP_STRING_FIELDS = (
    "operation",
    "memory_id",
)
MEMORY_OP_LIST_FIELDS = (
    "source_memory_ids",
)
RETRIEVAL_HIT_STRING_FIELDS = (
    "memory_id",
)
RETRIEVAL_HIT_NUMERIC_FIELDS = (
    "score",
    "rank",
)
TRACE_ARTIFACT_ENVELOPE_STRING_FIELDS = (
    "schema_version",
    "system_id",
    "benchmark_id",
    "release_split",
    "run_id",
    "release_contract_fingerprint",
    "query_ids_sha256",
)


def framework_trace_record(
    *,
    agent_system: dict[str, Any],
    case_id: str,
    query_id: str,
    raw_response: dict[str, Any],
    exported_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one normalized framework trace record.

    Agent adapters may expose richer fields through ``export_trace`` or directly
    in their raw response. Missing optional traces are represented as empty
    lists so downstream audits can distinguish "supported but no event" from a
    missing trace file.
    """

    trace = exported_trace if isinstance(exported_trace, dict) else {}
    record = {
        "schema_version": FRAMEWORK_TRACE_SCHEMA_VERSION,
        "case_id": case_id,
        "query_id": query_id,
        "framework_id": str(
            trace.get("framework_id")
            or agent_system.get("agent_framework")
            or agent_system.get("provider")
            or "unspecified"
        ),
        "framework_version": str(
            trace.get("framework_version")
            or agent_system.get("agent_runtime_version")
            or agent_system.get("system_version")
            or "unspecified"
        ),
        "framework_runtime": str(
            trace.get("framework_runtime")
            or agent_system.get("agent_runtime")
            or "unspecified"
        ),
        "orchestration_mode": str(
            trace.get("orchestration_mode")
            or agent_system.get("orchestration_mode")
            or "unspecified"
        ),
        "model_id": str(trace.get("model_id") or agent_system.get("model_backend") or "unspecified"),
        "memory_backend_id": str(
            trace.get("memory_backend_id")
            or agent_system.get("memory_backend")
            or "unspecified"
        ),
        "tool_runtime_id": str(trace.get("tool_runtime_id") or agent_system.get("tool_runtime_id") or "unspecified"),
        "session_id": str(trace.get("session_id") or case_id),
        "user_id": str(trace.get("user_id") or "unspecified"),
        "namespace": _list(trace.get("namespace")),
        "message_history_policy": str(trace.get("message_history_policy") or "unspecified"),
        "memory_ops": _list(trace.get("memory_ops") or raw_response.get("memory_operations")),
        "retrieval_hits": _list(trace.get("retrieval_hits") or raw_response.get("retrieval_hits")),
        "tool_calls": _list(trace.get("tool_calls") or raw_response.get("tool_calls")),
        "planner_trace": _list(trace.get("planner_trace") or raw_response.get("planner_trace")),
        "handoff_trace": _list(trace.get("handoff_trace") or raw_response.get("handoff_trace")),
        "cost": _dict(trace.get("cost") or raw_response.get("cost")),
        "framework_state": _dict(trace.get("framework_state")),
    }
    return record


def validate_framework_trace_record(
    record: dict[str, Any],
    *,
    tool_runtime_contracts: dict[str, ToolRuntimeContract] | None = None,
    require_handoff_boundary: bool = False,
    require_async_resume_identity: bool = False,
) -> tuple[str, ...]:
    """Validate one framework trace record for T0 adapter evidence."""

    errors: list[str] = []
    if not isinstance(record, dict):
        return ("framework_trace record must be an object",)
    for field in REQUIRED_TRACE_FIELDS:
        value = record.get(field)
        if value is None or value == "":
            errors.append(f"framework_trace.{field} is required")
    if record.get("schema_version") != FRAMEWORK_TRACE_SCHEMA_VERSION:
        errors.append(f"framework_trace.schema_version must be {FRAMEWORK_TRACE_SCHEMA_VERSION}")
    for field in LIST_TRACE_FIELDS:
        if field in record and not isinstance(record[field], list):
            errors.append(f"framework_trace.{field} must be a list")
    if "namespace" in record and not isinstance(record["namespace"], list):
        errors.append("framework_trace.namespace must be a list")
    if isinstance(record.get("namespace"), list):
        errors.extend(_validate_string_list(record["namespace"], "framework_trace.namespace"))
    if "cost" in record and not isinstance(record["cost"], dict):
        errors.append("framework_trace.cost must be an object")
    if isinstance(record.get("cost"), dict):
        errors.extend(_validate_cost_trace(record["cost"]))
        errors.extend(_validate_cost_count_consistency(record))
    if "framework_state" in record and not isinstance(record["framework_state"], dict):
        errors.append("framework_trace.framework_state must be an object")
    if isinstance(record.get("framework_state"), dict):
        errors.extend(_validate_async_resume_state(record["framework_state"]))
    errors.extend(_validate_memory_ops(record.get("memory_ops", [])))
    errors.extend(_validate_retrieval_hits(record.get("retrieval_hits", [])))
    errors.extend(_validate_planner_trace(record.get("planner_trace", [])))
    errors.extend(_validate_planner_trace_references(record))
    errors.extend(_validate_handoff_trace(record.get("handoff_trace", [])))
    errors.extend(_validate_handoff_trace_references(record))
    if require_handoff_boundary:
        errors.extend(_validate_handoff_boundary(record))
    if require_async_resume_identity:
        errors.extend(_validate_async_resume_identity(record))
    errors.extend(_validate_tool_calls_against_runtime(record, tool_runtime_contracts))
    return tuple(errors)


def validate_framework_trace_records(
    records: list[dict[str, Any]],
    *,
    tool_runtime_contracts: dict[str, ToolRuntimeContract] | None = None,
    require_handoff_boundary: bool = False,
    require_async_resume_identity: bool = False,
) -> tuple[str, ...]:
    if not isinstance(records, list):
        return ("framework_trace must be a list",)
    errors: list[str] = []
    for index, record in enumerate(records):
        errors.extend(
            f"framework_trace[{index}].{error}"
            for error in validate_framework_trace_record(
                record,
                tool_runtime_contracts=tool_runtime_contracts,
                require_handoff_boundary=require_handoff_boundary,
                require_async_resume_identity=require_async_resume_identity,
            )
        )
    return tuple(errors)


def validate_framework_trace_artifact_payload(
    payload: dict[str, Any],
    *,
    expected_records: int | None = None,
    require_envelope: bool = False,
    tool_runtime_contracts: dict[str, ToolRuntimeContract] | None = None,
    require_handoff_boundary: bool = False,
    require_async_resume_identity: bool = False,
) -> tuple[str, ...]:
    """Validate a framework-trace artifact envelope and contained records.

    Existing smoke artifacts only contain ``framework_traces``. Set
    ``require_envelope=True`` for claimable T1+ artifacts that must bind traces
    to the concrete system, split, run, and release contract.
    """

    if not isinstance(payload, dict):
        return ("framework_trace_artifact must be a JSON object",)
    errors: list[str] = []
    records = payload.get("framework_traces")
    if not isinstance(records, list):
        errors.append("framework_traces must be a list")
        records = []
    else:
        errors.extend(
            validate_framework_trace_records(
                records,
                tool_runtime_contracts=tool_runtime_contracts,
                require_handoff_boundary=require_handoff_boundary,
                require_async_resume_identity=require_async_resume_identity,
            )
        )
    if expected_records is not None and expected_records > 0 and len(records) != expected_records:
        errors.append(f"framework_traces count must match num_predictions {expected_records}")
    if require_envelope:
        for field in TRACE_ARTIFACT_ENVELOPE_STRING_FIELDS:
            if not isinstance(payload.get(field), str) or not payload.get(field):
                errors.append(f"framework_trace_artifact.{field} must be a non-empty string")
        if not isinstance(payload.get("record_count"), int) or payload.get("record_count", 0) < 0:
            errors.append("framework_trace_artifact.record_count must be a non-negative integer")
        if (
            isinstance(payload.get("schema_version"), str)
            and payload.get("schema_version") != FRAMEWORK_TRACE_ARTIFACT_SCHEMA_VERSION
        ):
            errors.append(
                "framework_trace_artifact.schema_version must be "
                f"{FRAMEWORK_TRACE_ARTIFACT_SCHEMA_VERSION}"
            )
        if isinstance(payload.get("record_count"), int) and payload.get("record_count") != len(records):
            errors.append("framework_trace_artifact.record_count must match framework_traces length")
        if isinstance(payload.get("query_ids_sha256"), str) and payload.get("query_ids_sha256"):
            expected_query_digest = _query_ids_sha256(records)
            if payload["query_ids_sha256"] != expected_query_digest:
                errors.append("framework_trace_artifact.query_ids_sha256 must match framework_traces query_id order")
    return tuple(errors)


def framework_trace_artifact_payload(
    framework_traces: list[dict[str, Any]],
    *,
    system_id: str,
    benchmark_id: str,
    release_split: str,
    run_id: str,
    release_contract_fingerprint: str | dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a framework-trace artifact with claimable provenance envelope."""

    return {
        "schema_version": FRAMEWORK_TRACE_ARTIFACT_SCHEMA_VERSION,
        "system_id": system_id,
        "benchmark_id": benchmark_id,
        "release_split": release_split,
        "run_id": run_id,
        "release_contract_fingerprint": _release_contract_fingerprint_label(
            release_contract_fingerprint
        ),
        "record_count": len(framework_traces),
        "query_ids_sha256": _query_ids_sha256(framework_traces),
        "framework_traces": framework_traces,
    }


def _release_contract_fingerprint_label(value: str | dict[str, Any] | None) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        digest = value.get("query_contract_sha256") or value.get("sha256")
        if isinstance(digest, str) and digest:
            return digest
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return "not_applicable"


def _query_ids_sha256(records: list[dict[str, Any]]) -> str:
    query_ids = [str(record.get("query_id") or "") for record in records]
    encoded = json.dumps(query_ids, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_default_tool_runtime_contracts(*, root: str | Path = ".") -> dict[str, ToolRuntimeContract]:
    """Load packaged tool-runtime contracts for framework trace validation."""

    resolved_root = Path(root)
    contracts: dict[str, ToolRuntimeContract] = {}
    for runtime_id, relative_path in DEFAULT_TOOL_RUNTIME_CONTRACTS.items():
        path = resolved_root / relative_path
        if path.exists():
            contracts[runtime_id] = load_tool_runtime_contract(path)
    return contracts


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _validate_tool_calls_against_runtime(
    record: dict[str, Any],
    contracts: dict[str, ToolRuntimeContract] | None,
) -> list[str]:
    if not contracts:
        return []
    runtime_id = str(record.get("tool_runtime_id") or "")
    if runtime_id in {"", "none", "unspecified"}:
        return []
    if runtime_id not in contracts:
        return [f"framework_trace.tool_runtime_id has no loaded contract: {runtime_id}"]
    errors: list[str] = []
    tool_calls = record.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        return []
    contract = contracts[runtime_id]
    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            errors.append(f"framework_trace.tool_calls[{index}] must be an object")
            continue
        errors.extend(
            f"framework_trace.tool_calls[{index}].{error}"
            for error in validate_tool_call_against_runtime(tool_call, contract)
        )
    return errors


def _validate_handoff_trace(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    errors: list[str] = []
    for index, event in enumerate(value):
        prefix = f"framework_trace.handoff_trace[{index}]"
        if not isinstance(event, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for field in ("from_agent", "to_agent", "reason"):
            if not isinstance(event.get(field), str) or not event.get(field):
                errors.append(f"{prefix}.{field} must be a non-empty string")
        for field in ("forwarded_memory_ids", "filtered_memory_ids"):
            if field in event and not isinstance(event[field], list):
                errors.append(f"{prefix}.{field} must be a list")
            elif field in event:
                errors.extend(_validate_string_list(event[field], f"{prefix}.{field}"))
    return errors


def _validate_handoff_trace_references(record: dict[str, Any]) -> list[str]:
    handoff_trace = record.get("handoff_trace", [])
    if not isinstance(handoff_trace, list):
        return []
    memory_ids = _memory_ids_in_trace(record)
    errors: list[str] = []
    for index, event in enumerate(handoff_trace):
        if not isinstance(event, dict):
            continue
        prefix = f"framework_trace.handoff_trace[{index}]"
        for field in ("forwarded_memory_ids", "filtered_memory_ids"):
            values = event.get(field)
            if not isinstance(values, list):
                continue
            for value_index, memory_id in enumerate(values):
                if isinstance(memory_id, str) and memory_id and memory_id not in memory_ids:
                    errors.append(f"{prefix}.{field}[{value_index}] references unknown memory_id: {memory_id}")
    return errors


def _validate_handoff_boundary(record: dict[str, Any]) -> list[str]:
    handoff_trace = record.get("handoff_trace", [])
    if not isinstance(handoff_trace, list) or not handoff_trace:
        return []
    errors: list[str] = []
    namespace = record.get("namespace")
    if not isinstance(namespace, list) or not namespace:
        errors.append("framework_trace.namespace must be a non-empty list for handoff boundary validation")
    policy = record.get("message_history_policy")
    if not isinstance(policy, str) or not policy or policy == "unspecified":
        errors.append("framework_trace.message_history_policy must be specified for handoff boundary validation")
    has_boundary_ids = False
    for event in handoff_trace:
        if not isinstance(event, dict):
            continue
        for field in ("forwarded_memory_ids", "filtered_memory_ids"):
            values = event.get(field)
            if isinstance(values, list) and values:
                has_boundary_ids = True
                break
        if has_boundary_ids:
            break
    if not has_boundary_ids:
        errors.append("framework_trace.handoff_trace must include forwarded_memory_ids or filtered_memory_ids")
    return errors


def _validate_planner_trace(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    errors: list[str] = []
    for index, event in enumerate(value):
        prefix = f"framework_trace.planner_trace[{index}]"
        if not isinstance(event, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for field in PLANNER_TRACE_STRING_FIELDS:
            if field in event and (not isinstance(event[field], str) or not event[field]):
                errors.append(f"{prefix}.{field} must be a non-empty string")
        for field in PLANNER_TRACE_LIST_FIELDS:
            if field in event and not isinstance(event[field], list):
                errors.append(f"{prefix}.{field} must be a list")
            elif field in event:
                errors.extend(_validate_string_list(event[field], f"{prefix}.{field}"))
    return errors


def _validate_planner_trace_references(record: dict[str, Any]) -> list[str]:
    planner_trace = record.get("planner_trace", [])
    if not isinstance(planner_trace, list):
        return []
    memory_ids = _memory_ids_in_trace(record)
    tool_call_ids = _tool_call_ids_in_trace(record)
    errors: list[str] = []
    for index, event in enumerate(planner_trace):
        if not isinstance(event, dict):
            continue
        prefix = f"framework_trace.planner_trace[{index}]"
        for field in ("input_memory_ids", "output_memory_ids"):
            values = event.get(field)
            if not isinstance(values, list):
                continue
            for value_index, memory_id in enumerate(values):
                if isinstance(memory_id, str) and memory_id and memory_id not in memory_ids:
                    errors.append(f"{prefix}.{field}[{value_index}] references unknown memory_id: {memory_id}")
        values = event.get("tool_call_ids")
        if isinstance(values, list):
            for value_index, tool_call_id in enumerate(values):
                if isinstance(tool_call_id, str) and tool_call_id and tool_call_id not in tool_call_ids:
                    errors.append(
                        f"{prefix}.tool_call_ids[{value_index}] references unknown tool_call_id: {tool_call_id}"
                    )
    return errors


def _memory_ids_in_trace(record: dict[str, Any]) -> set[str]:
    memory_ids: set[str] = set()
    for field in ("memory_ops", "retrieval_hits"):
        values = record.get(field, [])
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, dict) and isinstance(value.get("memory_id"), str) and value["memory_id"]:
                memory_ids.add(value["memory_id"])
    return memory_ids


def _tool_call_ids_in_trace(record: dict[str, Any]) -> set[str]:
    tool_call_ids: set[str] = set()
    values = record.get("tool_calls", [])
    if not isinstance(values, list):
        return tool_call_ids
    for value in values:
        if not isinstance(value, dict):
            continue
        for field in ("tool_call_id", "id"):
            if isinstance(value.get(field), str) and value[field]:
                tool_call_ids.add(value[field])
    return tool_call_ids


def _validate_async_resume_state(value: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ASYNC_RESUME_LIST_STATE_FIELDS:
        if field in value and not isinstance(value[field], list):
            errors.append(f"framework_trace.framework_state.{field} must be a list")
        elif field in value:
            errors.extend(_validate_async_resume_events(field, value[field]))
    for field in ASYNC_RESUME_STRING_STATE_FIELDS:
        if field in value and (not isinstance(value[field], str) or not value[field]):
            errors.append(f"framework_trace.framework_state.{field} must be a non-empty string")
    return errors


def _validate_async_resume_identity(record: dict[str, Any]) -> list[str]:
    state = record.get("framework_state")
    if not isinstance(state, dict) or not state:
        return ["framework_trace.framework_state must contain async/resume state for T4 validation"]
    errors: list[str] = []
    if not any(
        key in state and state[key] not in (None, "", [], {})
        for key in ASYNC_RESUME_LIST_STATE_FIELDS + ASYNC_RESUME_STRING_STATE_FIELDS
    ):
        errors.append("framework_trace.framework_state must contain async/resume evidence for T4 validation")
    missing_identity = [
        key
        for key in ASYNC_RESUME_STRING_STATE_FIELDS
        if not isinstance(state.get(key), str) or not state[key]
    ]
    if missing_identity:
        errors.append(
            "framework_trace.framework_state must include checkpoint_id, resume_from, "
            "and state_persistence_mode for T4 validation"
        )
    return errors


def _validate_async_resume_events(field: str, value: list[Any]) -> list[str]:
    errors: list[str] = []
    for index, event in enumerate(value):
        prefix = f"framework_trace.framework_state.{field}[{index}]"
        if not isinstance(event, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if field == "async_events":
            errors.extend(_validate_optional_string_fields(event, ASYNC_EVENT_STRING_FIELDS, prefix))
        elif field == "retry_events":
            errors.extend(_validate_optional_string_fields(event, RETRY_EVENT_STRING_FIELDS, prefix))
        elif field == "stream_events":
            errors.extend(_validate_optional_string_fields(event, STREAM_EVENT_STRING_FIELDS, prefix))
            for numeric_field in STREAM_EVENT_NUMERIC_FIELDS:
                if numeric_field in event and (
                    not isinstance(event[numeric_field], (int, float))
                    or isinstance(event[numeric_field], bool)
                    or event[numeric_field] < 0
                ):
                    errors.append(f"{prefix}.{numeric_field} must be a non-negative number")
    return errors


def _validate_optional_string_fields(
    event: dict[str, Any],
    fields: tuple[str, ...],
    prefix: str,
) -> list[str]:
    errors: list[str] = []
    for field in fields:
        if field in event and (not isinstance(event[field], str) or not event[field]):
            errors.append(f"{prefix}.{field} must be a non-empty string")
    return errors


def _validate_cost_trace(value: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in COST_TRACE_NUMERIC_FIELDS:
        if field in value and (
            not isinstance(value[field], (int, float))
            or isinstance(value[field], bool)
            or value[field] < 0
        ):
            errors.append(f"framework_trace.cost.{field} must be a non-negative number")
    return errors


def _validate_cost_count_consistency(record: dict[str, Any]) -> list[str]:
    cost = record.get("cost", {})
    if not isinstance(cost, dict):
        return []
    errors: list[str] = []
    if (
        isinstance(record.get("tool_calls"), list)
        and "tool_call_count" in cost
        and isinstance(cost["tool_call_count"], (int, float))
        and not isinstance(cost["tool_call_count"], bool)
        and cost["tool_call_count"] != len(record["tool_calls"])
    ):
        errors.append("framework_trace.cost.tool_call_count must match len(tool_calls)")
    if (
        isinstance(record.get("memory_ops"), list)
        and "memory_op_count" in cost
        and isinstance(cost["memory_op_count"], (int, float))
        and not isinstance(cost["memory_op_count"], bool)
        and cost["memory_op_count"] != len(record["memory_ops"])
    ):
        errors.append("framework_trace.cost.memory_op_count must match len(memory_ops)")
    return errors


def _validate_memory_ops(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    errors: list[str] = []
    for index, event in enumerate(value):
        prefix = f"framework_trace.memory_ops[{index}]"
        if not isinstance(event, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for field in MEMORY_OP_STRING_FIELDS:
            if field in event and (not isinstance(event[field], str) or not event[field]):
                errors.append(f"{prefix}.{field} must be a non-empty string")
        for field in MEMORY_OP_LIST_FIELDS:
            if field in event and not isinstance(event[field], list):
                errors.append(f"{prefix}.{field} must be a list")
            elif field in event:
                errors.extend(_validate_string_list(event[field], f"{prefix}.{field}"))
    return errors


def _validate_retrieval_hits(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    errors: list[str] = []
    for index, hit in enumerate(value):
        prefix = f"framework_trace.retrieval_hits[{index}]"
        if not isinstance(hit, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for field in RETRIEVAL_HIT_STRING_FIELDS:
            if field in hit and (not isinstance(hit[field], str) or not hit[field]):
                errors.append(f"{prefix}.{field} must be a non-empty string")
        for field in RETRIEVAL_HIT_NUMERIC_FIELDS:
            if field in hit and (
                not isinstance(hit[field], (int, float))
                or isinstance(hit[field], bool)
                or hit[field] < 0
            ):
                errors.append(f"{prefix}.{field} must be a non-negative number")
    return errors


def _validate_string_list(value: list[Any], field: str) -> list[str]:
    errors: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            errors.append(f"{field}[{index}] must be a non-empty string")
    return errors
