"""Compile query-time memory-state contracts from graph events."""

from __future__ import annotations

from datetime import datetime

from amb.benchmark.generation.types import GraphEvent
from amb.benchmark.schemas.models import MemoryStateContract, StateTransition


def compile_state_contracts(case_id: str, events: tuple[GraphEvent, ...]) -> tuple[MemoryStateContract, ...]:
    update_event = next(event for event in events if event.event_type == "fact_update")
    deletion_or_retention = next(
        event for event in events if event.event_type in {"deletion_request", "retention_confirmation"}
    )
    final_timestamp = max(event.timestamp for event in events)
    return (
        _state_contract_at(case_id, "state_after_update", update_event.timestamp, events),
        _state_contract_at(case_id, "state_after_delete_or_retain", deletion_or_retention.timestamp, events),
        _state_contract_at(case_id, "state_final", final_timestamp, events),
    )


def _state_contract_at(
    case_id: str,
    label: str,
    timestamp: datetime,
    events: tuple[GraphEvent, ...],
) -> MemoryStateContract:
    update_event = next(event for event in events if event.event_type == "fact_update")
    superseded_event_ids = _superseded_event_ids(events, update_event)
    deleted_sources = {
        event.supersedes
        for event in events
        if event.event_type == "deletion_request" and event.supersedes and event.timestamp <= timestamp
    }
    active: list[str] = []
    inactive: list[str] = []
    deleted: list[str] = []
    forbidden: list[str] = []
    superseded: list[str] = []
    restricted: list[str] = []
    for event in events:
        if event.event_type == "distractor":
            continue
        memory_id = f"m_{event.event_id}"
        if event.timestamp > timestamp:
            inactive.append(memory_id)
        elif event.event_id in superseded_event_ids and update_event.timestamp <= timestamp:
            superseded.append(memory_id)
            inactive.append(memory_id)
        elif event.event_id in deleted_sources:
            deleted.append(memory_id)
            forbidden.append(memory_id)
        elif event.event_id.startswith("e_near_miss_update_"):
            forbidden.append(memory_id)
            inactive.append(memory_id)
        elif event.should_delete:
            deleted.append(memory_id)
            forbidden.append(memory_id)
        elif event.privacy_level in {"sensitive", "restricted", "forbidden"}:
            forbidden.append(memory_id)
            restricted.append(memory_id)
        elif not event.should_store:
            forbidden.append(memory_id)
            inactive.append(memory_id)
        elif event.should_store:
            active.append(memory_id)
    return MemoryStateContract(
        state_contract_id=f"{case_id}:{label}",
        timestamp=timestamp.isoformat().replace("+00:00", "Z"),
        scenario_id=case_id,
        active_memory_ids=tuple(active),
        inactive_memory_ids=tuple(inactive),
        deleted_memory_ids=tuple(deleted),
        forbidden_memory_ids=tuple(forbidden),
        superseded_memory_ids=tuple(superseded),
        restricted_memory_ids=tuple(restricted),
        required_governance_rules=_required_governance_rules(deleted=deleted, forbidden=forbidden, restricted=restricted),
        transitions=tuple(_state_transitions_at(events, timestamp)),
    )


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


def _state_transitions_at(events: tuple[GraphEvent, ...], timestamp: datetime) -> list[StateTransition]:
    transitions: list[StateTransition] = []
    for event in events:
        if event.timestamp > timestamp or not event.supersedes:
            continue
        if event.event_type == "fact_update":
            transitions.append(
                StateTransition(
                    from_memory_id=f"m_{event.supersedes}",
                    to_memory_id=f"m_{event.event_id}",
                    transition_type="update",
                    trigger_event_id=event.event_id,
                )
            )
        elif event.event_type == "deletion_request":
            transitions.append(
                StateTransition(
                    from_memory_id=f"m_{event.supersedes}",
                    to_memory_id=f"m_{event.event_id}",
                    transition_type="delete",
                    trigger_event_id=event.event_id,
                )
            )
        elif event.event_type == "retention_confirmation":
            transitions.append(
                StateTransition(
                    from_memory_id=f"m_{event.supersedes}",
                    to_memory_id=f"m_{event.event_id}",
                    transition_type="retain",
                    trigger_event_id=event.event_id,
                )
            )
    return transitions


def _required_governance_rules(
    *,
    deleted: list[str],
    forbidden: list[str],
    restricted: list[str],
) -> tuple[str, ...]:
    rules = ["same_user_only"]
    if deleted:
        rules.append("do_not_recall_deleted")
    if forbidden or restricted:
        rules.append("respect_authorization_scope")
    return tuple(rules)
