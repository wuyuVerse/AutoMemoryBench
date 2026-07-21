"""Formatting and classification helpers for compiled memory artifacts."""

from __future__ import annotations

from amb.benchmark.generation.types import GraphEvent


def memory_content(event: GraphEvent) -> str:
    if event.event_type == "governance_rule":
        return event.value
    if event.event_type == "authorization_event":
        return f"Authorization scope: {event.value}."
    if event.event_type == "tool_result":
        return f"{event.actor} returned: {event.value}."
    if event.event_type == "tool_outcome_event":
        return f"Tool outcome: {event.value}."
    if event.event_type == "planning_constraint":
        return f"For {event.subject}, the constraint is {event.value}."
    if event.event_type == "document_note":
        return f"Document note for {event.subject}: {event.value}."
    if event.event_type == "procedural_event":
        return f"Procedure: {event.value}."
    if event.event_type == "feedback_event":
        return f"Feedback: {event.value}."
    if event.event_type == "reflective_event":
        return f"Reflection: {event.value}."
    if event.event_type == "task_result_event":
        return f"Task result: {event.value}."
    if event.event_type == "platform_message":
        return f"Platform message for {event.subject}: {event.value}."
    if event.event_type == "fact_reinforcement":
        return f"Reinforced {event.subject}: {event.value}."
    if event.event_type == "conflict_event":
        return f"Unverified conflicting {event.subject}: {event.value}."
    if event.event_type == "expiry_event":
        return f"Expired {event.subject}: {event.value}."
    if event.event_type == "deletion_request":
        return f"Deleted {event.subject}: {event.value}."
    if event.event_type == "retention_confirmation":
        return f"Retained {event.subject}: {event.value}."
    if event.event_type == "authorized_sensitive_memory":
        return f"Authorized {event.subject}: {event.value}."
    return f"{event.subject} is {event.value}."


def memory_status(event: GraphEvent, update_event: GraphEvent, deleted_by: dict[str, GraphEvent] | None = None) -> str:
    del update_event
    if deleted_by and event.event_id in deleted_by:
        return "deleted"
    if event.should_delete:
        return "deleted"
    if not event.should_store:
        return "forbidden"
    return "active"


def event_predicate(event: GraphEvent) -> str:
    if event.event_type == "fact_reinforcement":
        return "reinforces"
    if event.event_type == "fact_update":
        return "updates_to"
    if event.event_type == "conflict_event":
        return "conflicts_with"
    if event.event_type == "expiry_event":
        return "expires"
    if event.event_type == "deletion_request":
        return "delete"
    if event.event_type == "retention_confirmation":
        return "retain"
    if event.event_type == "sensitive_disclosure":
        return "must_not_store"
    if event.event_type == "authorized_sensitive_memory":
        return "authorized_to_store"
    if event.event_type == "authorization_event":
        return "authorizes_scope"
    if event.event_type == "tool_result":
        return "returned"
    if event.event_type == "tool_outcome_event":
        return "confirmed"
    if event.event_type == "planning_constraint":
        return "constrains"
    if event.event_type == "document_note":
        return "documents"
    if event.event_type == "procedural_event":
        return "procedure"
    if event.event_type == "feedback_event":
        return "feedback"
    if event.event_type == "reflective_event":
        return "reflects"
    if event.event_type == "task_result_event":
        return "resulted_in"
    if event.event_type == "platform_message":
        return "confirms"
    if event.event_type == "governance_rule":
        return "governs"
    return "is"


def importance(event: GraphEvent) -> int:
    if event.privacy_level != "normal" or event.should_delete:
        return 5
    if event.event_type in {
        "fact_update",
        "tool_result",
        "planning_constraint",
        "procedural_event",
        "feedback_event",
        "authorization_event",
        "tool_outcome_event",
        "document_note",
        "reflective_event",
    }:
        return 4
    return 3
