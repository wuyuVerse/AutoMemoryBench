"""Compile graph events into standard memory units."""

from __future__ import annotations

from amb.benchmark.generation.compilers.formatting import importance, memory_content
from amb.benchmark.generation.types import GraphEvent
from amb.benchmark.schemas.models import MemoryUnit


def compile_memories(
    case_id: str,
    events: tuple[GraphEvent, ...],
    turn_by_event: dict[str, str] | None = None,
) -> tuple[MemoryUnit, ...]:
    if turn_by_event is None:
        from amb.benchmark.generation.renderers.conversation import event_source_turns

        turn_by_event = event_source_turns(case_id, events)
    memories: list[MemoryUnit] = []
    update_event = next(event for event in events if event.event_type == "fact_update")
    deleted_by = {
        event.supersedes: event
        for event in events
        if event.event_type == "deletion_request" and event.supersedes
    }
    superseded_event_ids = _superseded_event_ids(events, update_event)
    for event in events:
        if event.event_type == "distractor":
            continue
        memory_id = f"m_{event.event_id}"
        deletion_event = deleted_by.get(event.event_id)
        valid_until = None
        if event.event_id in superseded_event_ids:
            valid_until = update_event.timestamp.isoformat().replace("+00:00", "Z")
        if deletion_event:
            valid_until = deletion_event.timestamp.isoformat().replace("+00:00", "Z")
        memories.append(
            MemoryUnit(
                memory_id=memory_id,
                type=event.memory_type,
                memory_type=event.memory_type,
                content=memory_content(event),
                source_turn_ids=(turn_by_event[event.event_id],),
                scenario_id=case_id,
                canonical_form=_canonical_form(event),
                source_event_ids=(event.event_id,),
                source_trace_ids=(turn_by_event[event.event_id],),
                valid_from=event.timestamp.isoformat().replace("+00:00", "Z"),
                valid_until=valid_until,
                status=_memory_status(
                    event,
                    deleted_by=deleted_by,
                    superseded_event_ids=superseded_event_ids,
                ),
                importance=importance(event),
                confidence=1.0,
                should_store=event.should_store and not deletion_event,
                should_write=event.should_store and not deletion_event,
                should_delete=event.should_delete or bool(deletion_event),
                privacy_level=event.privacy_level,
                sensitivity=event.privacy_level,
                authorization_scope=_authorization_scope(event),
                should_retrieve_for=_should_retrieve_for(event),
                should_not_retrieve_for=_should_not_retrieve_for(event),
                update_of=f"m_{event.supersedes}" if event.event_type == "fact_update" else None,
                invalidates=_invalidated_memory_ids(event),
                forget_policy=_forget_policy(event),
                expected_use=_expected_use(event),
            )
        )
    return tuple(memories)


def _superseded_event_ids(events: tuple[GraphEvent, ...], update_event: GraphEvent) -> set[str]:
    if not update_event.supersedes:
        return set()
    by_id = {event.event_id: event for event in events}
    target = by_id.get(update_event.supersedes)
    if target is None:
        return set()
    superseded = {update_event.supersedes}
    for event in events:
        if (
            event.event_type == "fact_reinforcement"
            and event.subject == target.subject
            and event.value == target.value
            and event.timestamp <= update_event.timestamp
        ):
            superseded.add(event.event_id)
    return superseded


def _memory_status(
    event: GraphEvent,
    *,
    deleted_by: dict[str, GraphEvent],
    superseded_event_ids: set[str],
) -> str:
    if event.event_id in deleted_by:
        return "deleted"
    if event.event_id in superseded_event_ids:
        return "superseded"
    if event.should_delete:
        return "deleted"
    if not event.should_store:
        return "forbidden"
    return "active"


def _invalidated_memory_ids(event: GraphEvent) -> tuple[str, ...]:
    if event.event_type in {"fact_update", "deletion_request"} and event.supersedes:
        return (f"m_{event.supersedes}",)
    return ()


def _canonical_form(event: GraphEvent) -> dict[str, str]:
    return {
        "subject": event.subject,
        "predicate": _canonical_predicate(event),
        "object": event.value,
    }


def _canonical_predicate(event: GraphEvent) -> str:
    if event.event_type == "near_miss_context" and event.subject == "feedback":
        return "feedback"
    mapping = {
        "fact_introduction": "states",
        "fact_reinforcement": "reinforces",
        "fact_update": "updates_to",
        "conflict_event": "conflicts_with",
        "expiry_event": "expires",
        "deletion_request": "deletes",
        "retention_confirmation": "retains",
        "sensitive_disclosure": "forbids_use",
        "authorized_sensitive_memory": "authorizes_use",
        "authorization_event": "authorizes_scope",
        "tool_result": "tool_returns",
        "tool_outcome_event": "tool_confirms",
        "planning_constraint": "constrains",
        "document_note": "documents",
        "procedural_event": "procedure",
        "feedback_event": "feedback",
        "reflective_event": "reflects",
        "task_result_event": "results_in",
        "governance_rule": "governs",
        "platform_message": "confirms",
    }
    return mapping.get(event.event_type, "states")


def _authorization_scope(event: GraphEvent) -> str:
    if event.event_type in {"authorization_event", "authorized_sensitive_memory"}:
        return "same_user_same_project"
    if event.privacy_level in {"restricted", "sensitive", "forbidden"}:
        return "governed_scope_only"
    return "same_user"


def _should_retrieve_for(event: GraphEvent) -> tuple[str, ...]:
    by_type = {
        "tool_result": ("tool_call",),
        "tool_outcome_event": ("tool_call", "task_execution"),
        "planning_constraint": ("planning",),
        "document_note": ("planning", "answering"),
        "procedural_event": ("planning", "policy_reuse"),
        "feedback_event": ("policy_reuse",),
        "reflective_event": ("policy_reuse",),
        "fact_update": ("answering", "retrieval", "state_update"),
        "fact_introduction": ("answering",),
        "authorized_sensitive_memory": ("governance",),
        "authorization_event": ("governance",),
        "governance_rule": ("governance", "safety"),
        "task_result_event": ("task_execution", "policy_reuse"),
    }
    return by_type.get(event.event_type, ())


def _should_not_retrieve_for(event: GraphEvent) -> tuple[str, ...]:
    if event.event_type == "near_miss_context":
        return ("answering", "retrieval", "planning", "tool_call", "policy_reuse")
    if event.event_type in {"deletion_request", "expiry_event", "sensitive_disclosure", "conflict_event"}:
        return ("answering", "planning", "tool_call")
    return ()


def _forget_policy(event: GraphEvent) -> str:
    if event.event_type == "deletion_request":
        return "delete_immediately"
    if event.event_type == "expiry_event":
        return "expire_before_final_state"
    if event.event_type == "sensitive_disclosure":
        return "do_not_store"
    return "retain_until_updated_or_deleted"


def _expected_use(event: GraphEvent) -> str:
    mapping = {
        "fact_introduction": "answer_current_fact",
        "fact_update": "answer_current_fact",
        "tool_result": "condition_tool_parameters",
        "planning_constraint": "condition_plan",
        "document_note": "support_plan",
        "procedural_event": "support_policy_reuse",
        "feedback_event": "support_policy_reuse",
        "reflective_event": "support_policy_reuse",
        "task_result_event": "support_policy_reuse",
        "governance_rule": "govern_output",
        "authorized_sensitive_memory": "condition_governed_answer",
        "authorization_event": "condition_governed_answer",
    }
    return mapping.get(event.event_type, "support_context")
