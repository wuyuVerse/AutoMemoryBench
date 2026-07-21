"""Compile semantic event edges for AutoMemoryBench event graphs."""

from __future__ import annotations

from amb.benchmark.generation.types import GraphEvent
from amb.benchmark.schemas.models import EventEdge


def compile_event_edges(events: tuple[GraphEvent, ...]) -> tuple[EventEdge, ...]:
    edges: list[EventEdge] = []
    ordered = sorted(events, key=lambda event: event.timestamp)
    for previous, current in zip(ordered, ordered[1:]):
        edges.append(EventEdge(previous.event_id, current.event_id, "temporal_before"))

    introductions_by_subject: dict[str, list[GraphEvent]] = {}
    for event in events:
        if event.event_type == "fact_introduction":
            introductions_by_subject.setdefault(event.subject, []).append(event)

    for event in events:
        if event.supersedes and event.supersedes.startswith("e_old_"):
            edges.append(EventEdge(event.supersedes, event.event_id, "superseded_by"))
            if event.event_type == "fact_update":
                edges.append(EventEdge(event.supersedes, event.event_id, "updates"))
        if event.event_type == "deletion_request" and event.supersedes:
            edges.append(EventEdge(event.supersedes, event.event_id, "invalidates"))
        if event.event_type == "retention_confirmation" and event.supersedes:
            edges.append(EventEdge(event.supersedes, event.event_id, "supports"))
        if event.event_type == "fact_reinforcement":
            target = next(
                (
                    intro
                    for intro in introductions_by_subject.get(event.subject, ())
                    if intro.value == event.value and intro.timestamp <= event.timestamp
                ),
                None,
            )
            if target:
                edges.append(EventEdge(event.event_id, target.event_id, "supports"))

    governance = next((event for event in events if event.event_type == "governance_rule"), None)
    sensitive = next((event for event in events if event.event_type == "sensitive_disclosure"), None)
    authorization = next((event for event in events if event.event_type == "authorization_event"), None)
    if governance and sensitive:
        edges.append(EventEdge(governance.event_id, sensitive.event_id, "forbids"))
    if authorization and sensitive:
        edges.append(EventEdge(authorization.event_id, sensitive.event_id, "forbids"))
    authorized_sensitive = next((event for event in events if event.event_type == "authorized_sensitive_memory"), None)
    if governance and authorized_sensitive:
        edges.append(EventEdge(governance.event_id, authorized_sensitive.event_id, "authorizes"))
    if authorization and authorized_sensitive:
        edges.append(EventEdge(authorization.event_id, authorized_sensitive.event_id, "authorizes"))

    distractor = next((event for event in events if event.event_type == "distractor"), None)
    stable = next((event for event in events if event.event_id.startswith("e_stable_")), None)
    if distractor and stable:
        edges.append(EventEdge(distractor.event_id, stable.event_id, "distracts"))

    old = next((event for event in events if event.event_id.startswith("e_old_")), None)
    update = next((event for event in events if event.event_type == "fact_update"), None)
    conflict = next((event for event in events if event.event_type == "conflict_event"), None)
    if old and update:
        edges.append(EventEdge(old.event_id, update.event_id, "same_entity_as"))
    if conflict and old:
        edges.append(EventEdge(conflict.event_id, old.event_id, "contradicts"))
    if conflict and update:
        edges.append(EventEdge(conflict.event_id, update.event_id, "contradicts"))

    tool = next((event for event in events if event.event_type == "tool_result"), None)
    tool_outcome = next((event for event in events if event.event_type == "tool_outcome_event"), None)
    document = next((event for event in events if event.event_type == "document_note"), None)
    plan = next((event for event in events if event.event_type == "planning_constraint"), None)
    procedure = next((event for event in events if event.event_type == "procedural_event"), None)
    feedback = next((event for event in events if event.event_type == "feedback_event"), None)
    reflection = next((event for event in events if event.event_type == "reflective_event"), None)
    task_result = next((event for event in events if event.event_type == "task_result_event"), None)
    platform = next((event for event in events if event.event_type == "platform_message"), None)
    if update and tool:
        edges.append(EventEdge(tool.event_id, update.event_id, "depends_on"))
    if tool and tool_outcome:
        edges.append(EventEdge(tool_outcome.event_id, tool.event_id, "supports"))
    if plan and document:
        edges.append(EventEdge(document.event_id, plan.event_id, "supports"))
    if plan and procedure:
        edges.append(EventEdge(procedure.event_id, plan.event_id, "depends_on"))
    if procedure and feedback:
        edges.append(EventEdge(feedback.event_id, procedure.event_id, "supports"))
    if feedback and reflection:
        edges.append(EventEdge(reflection.event_id, feedback.event_id, "supports"))
    if feedback and task_result:
        edges.append(EventEdge(feedback.event_id, task_result.event_id, "supports"))
    if task_result and platform:
        edges.append(EventEdge(platform.event_id, task_result.event_id, "supports"))

    return tuple(edges)
