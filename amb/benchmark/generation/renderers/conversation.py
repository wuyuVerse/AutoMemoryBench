"""Conversation renderer and source-turn mapping for AutoMemoryBench traces."""

from __future__ import annotations

from amb.benchmark.generation.types import DomainSpec, GraphEvent
from amb.benchmark.schemas.models import Session, Turn


EVENTS_PER_SESSION = 5


def compile_sessions(spec: DomainSpec, case_id: str, events: tuple[GraphEvent, ...]) -> tuple[Session, ...]:
    """Render graph events into chronological multi-session conversation traces."""

    sessions: list[Session] = []
    for session_index, group in enumerate(_event_groups(events), start=1):
        session_id = f"{case_id}:s{session_index}"
        turns: list[Turn] = []
        for turn_index, event in enumerate(group, start=1):
            turn_id = f"{session_id}:t{turn_index}"
            turns.append(Turn(turn_id, "user", render_user_turn(event)))
            if event.event_type in _ACK_EVENT_TYPES:
                turns.append(Turn(f"{turn_id}a", "assistant", render_ack(event)))
        sessions.append(Session(session_id, group[0].timestamp.isoformat().replace("+00:00", "Z"), tuple(turns)))
    return tuple(sessions)


def event_source_turns(case_id: str, events: tuple[GraphEvent, ...]) -> dict[str, str]:
    """Return the user turn that introduced each graph event."""

    mapping: dict[str, str] = {}
    for session_index, group in enumerate(_event_groups(events), start=1):
        session_id = f"{case_id}:s{session_index}"
        for turn_index, event in enumerate(group, start=1):
            mapping[event.event_id] = f"{session_id}:t{turn_index}"
    return mapping


def render_user_turn(event: GraphEvent) -> str:
    if event.event_type == "fact_reinforcement":
        return f"Confirm this still matters: {event.subject} remains {event.value}."
    if event.event_type == "fact_update":
        return f"Update this: {event.subject} is now {event.value}."
    if event.event_type == "conflict_event":
        return f"I heard a conflicting but unverified note about {event.subject}: {event.value}."
    if event.event_type == "expiry_event":
        return f"This temporary context has expired: {event.value}."
    if event.event_type == "deletion_request":
        return f"Forget and delete the {event.subject}: {event.value}."
    if event.event_type == "retention_confirmation":
        return f"Keep remembering the {event.subject}: {event.value}."
    if event.event_type == "authorization_event":
        return f"Access note: {event.value}."
    if event.event_type == "sensitive_disclosure":
        return f"Temporary sensitive {event.subject}: {event.value}. Do not save or repeat it."
    if event.event_type == "authorized_sensitive_memory":
        return f"You are authorized to remember this {event.subject}: {event.value}."
    if event.event_type == "tool_result":
        return f"Tool result from {event.actor}: {event.value}."
    if event.event_type == "tool_outcome_event":
        return f"Tool outcome confirmation: {event.value}."
    if event.event_type == "planning_constraint":
        return f"For {event.subject}, remember this constraint: {event.value}."
    if event.event_type == "document_note":
        return f"Document note for {event.subject}: {event.value}."
    if event.event_type == "procedural_event":
        return f"Procedure to remember: {event.value}."
    if event.event_type == "feedback_event":
        return f"Feedback for future tasks: {event.value}."
    if event.event_type == "reflective_event":
        return f"Reflection to preserve: {event.value}."
    if event.event_type == "task_result_event":
        return f"Task outcome: {event.value}."
    if event.event_type == "governance_rule":
        return f"Memory policy: {event.value}"
    if event.event_type == "platform_message":
        return f"Platform follow-up on {event.subject}: {event.value}."
    if event.event_type == "distractor":
        return event.value
    return f"Remember that {event.subject} is {event.value}."


def render_ack(event: GraphEvent) -> str:
    if event.event_type == "deletion_request":
        return "I will delete that and avoid using it later."
    if event.event_type == "retention_confirmation":
        return "I will retain that for later."
    if event.event_type == "sensitive_disclosure":
        return "I will not save or repeat that sensitive value."
    if event.event_type == "authorized_sensitive_memory":
        return "I will retain that sensitive detail only for approved work here."
    if event.event_type == "authorization_event":
        return "I will follow that access limit."
    if event.event_type == "expiry_event":
        return "I will treat that temporary context as expired."
    return "I will use the updated value going forward."


def _event_groups(events: tuple[GraphEvent, ...]) -> tuple[tuple[GraphEvent, ...], ...]:
    ordered = tuple(sorted(events, key=lambda event: event.timestamp))
    return tuple(
        ordered[index : index + EVENTS_PER_SESSION]
        for index in range(0, len(ordered), EVENTS_PER_SESSION)
    )


_ACK_EVENT_TYPES = {
    "fact_update",
    "expiry_event",
    "deletion_request",
    "retention_confirmation",
    "authorization_event",
    "sensitive_disclosure",
    "authorized_sensitive_memory",
}
