"""Tool trace renderer."""

from __future__ import annotations

from typing import Any

from amb.benchmark.generation.types import DomainSpec, GraphEvent
from amb.benchmark.generation.renderers.records import event_actor, event_timestamp, event_value

TOOL_EVENT_TYPES = ("tool_result", "tool_outcome_event")


def render_tool_records(spec: DomainSpec, case_id: str, events: tuple[GraphEvent, ...]) -> tuple[dict[str, Any], ...]:
    """Render tool result and outcome events as structured tool records."""

    records: list[dict[str, Any]] = []
    for index, event in enumerate(sorted(events, key=lambda item: item.timestamp), start=1):
        if event.event_type not in TOOL_EVENT_TYPES:
            continue
        records.append(
            {
                "trace_id": f"{case_id}:tool:{index:03d}",
                "source_event_id": event.event_id,
                "timestamp": event_timestamp(event),
                "tool_name": spec.tool_name,
                "actor": event_actor(event, spec.actor),
                "status": "ok",
                "parameters": {
                    "subject": event.subject,
                    "current_state": spec.new_value,
                },
                "result": event_value(event),
            }
        )
    return tuple(records)
