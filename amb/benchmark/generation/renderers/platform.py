"""Platform-style trace renderer."""

from __future__ import annotations

from typing import Any

from amb.benchmark.generation.types import DomainSpec, GraphEvent
from amb.benchmark.generation.renderers.records import event_actor, event_timestamp, event_value

PLATFORM_CHANNELS = {
    "platform_message": "chat",
    "authorization_event": "admin_console",
    "planning_constraint": "project_board",
    "task_result_event": "task_tracker",
    "feedback_event": "comment_thread",
    "conflict_event": "chat",
    "distractor": "chat",
}
PLATFORM_EVENT_TYPES = tuple(PLATFORM_CHANNELS)


def render_platform_messages(spec: DomainSpec, case_id: str, events: tuple[GraphEvent, ...]) -> tuple[dict[str, Any], ...]:
    """Render selected events as multi-platform messages.

    The return value is intentionally schema-light: generated benchmark cases
    keep canonical event/source ids, while downstream challenge packs can embed
    these records as Slack, email, issue, calendar, or ticket traces.
    """

    records: list[dict[str, Any]] = []
    for index, event in enumerate(sorted(events, key=lambda item: item.timestamp), start=1):
        platform = _platform_for_event(event)
        if platform is None:
            continue
        records.append(
            {
                "trace_id": f"{case_id}:platform:{index:03d}",
                "source_event_id": event.event_id,
                "timestamp": event_timestamp(event),
                "platform": platform,
                "actor": event_actor(event, spec.actor),
                "subject": event.subject,
                "text": _message_text(platform, event),
            }
        )
    return tuple(records)


def _platform_for_event(event: GraphEvent) -> str | None:
    return PLATFORM_CHANNELS.get(event.event_type)


def _message_text(platform: str, event: GraphEvent) -> str:
    if platform == "admin_console":
        return f"Access policy update for {event.subject}: {event_value(event)}"
    if platform == "project_board":
        return f"Project constraint recorded for {event.subject}: {event_value(event)}"
    if platform == "task_tracker":
        return f"Task result for {event.subject}: {event_value(event)}"
    if platform == "comment_thread":
        return f"Feedback note: {event_value(event)}"
    return f"{event.subject}: {event_value(event)}"
