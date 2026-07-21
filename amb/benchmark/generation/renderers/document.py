"""Document-style trace renderer."""

from __future__ import annotations

from typing import Any

from amb.benchmark.generation.types import DomainSpec, GraphEvent
from amb.benchmark.generation.renderers.records import event_timestamp, event_value

DOCUMENT_EVENT_TYPES = ("document_note", "procedural_event", "planning_constraint", "reflective_event")


def render_document_snippets(spec: DomainSpec, case_id: str, events: tuple[GraphEvent, ...]) -> tuple[dict[str, Any], ...]:
    """Render durable notes as document snippets with source-event provenance."""

    records: list[dict[str, Any]] = []
    for index, event in enumerate(sorted(events, key=lambda item: item.timestamp), start=1):
        if event.event_type not in DOCUMENT_EVENT_TYPES:
            continue
        records.append(
            {
                "trace_id": f"{case_id}:doc:{index:03d}",
                "source_event_id": event.event_id,
                "timestamp": event_timestamp(event),
                "document_id": f"{case_id}:{spec.domain}:notes",
                "section": _section_for_event(event),
                "text": f"{event.subject}: {event_value(event)}",
            }
        )
    return tuple(records)


def _section_for_event(event: GraphEvent) -> str:
    if event.event_type == "procedural_event":
        return "procedure"
    if event.event_type == "planning_constraint":
        return "constraints"
    if event.event_type == "reflective_event":
        return "reflection"
    return "notes"
