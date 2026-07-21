"""Compile internal graph events into release-facing event records."""

from __future__ import annotations

from amb.benchmark.generation.compilers.formatting import event_predicate
from amb.benchmark.generation.types import GraphEvent
from amb.benchmark.schemas.models import Event as ModelEvent


def compile_model_events(
    case_id: str,
    events: tuple[GraphEvent, ...],
    turn_by_event: dict[str, str] | None = None,
) -> tuple[ModelEvent, ...]:
    if turn_by_event is None:
        from amb.benchmark.generation.renderers.conversation import event_source_turns

        turn_by_event = event_source_turns(case_id, events)
    return tuple(
        ModelEvent(
            event_id=event.event_id,
            event_type=event.event_type,
            timestamp=event.timestamp.isoformat().replace("+00:00", "Z"),
            subject=event.subject,
            predicate=event_predicate(event),
            object=event.value,
            source_turn_ids=(turn_by_event[event.event_id],),
            attributes={
                "actor": event.actor,
                "memory_type": event.memory_type,
                "privacy_level": event.privacy_level,
                "should_store": event.should_store,
                "should_delete": event.should_delete,
                **({"supersedes": event.supersedes} if event.supersedes else {}),
            },
        )
        for event in events
    )
