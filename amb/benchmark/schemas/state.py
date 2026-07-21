"""Memory-state replay and state-contract utilities."""

from __future__ import annotations

from dataclasses import dataclass

from amb.benchmark.schemas.models import Case, Event, EventEdge, MemoryStateContract


@dataclass(frozen=True)
class ReplayedState:
    active_memory_ids: frozenset[str]
    inactive_memory_ids: frozenset[str]
    deleted_memory_ids: frozenset[str]
    forbidden_memory_ids: frozenset[str]
    superseded_memory_ids: frozenset[str]
    restricted_memory_ids: frozenset[str]


def replay_state_at(case: Case, timestamp: str) -> ReplayedState:
    """Replay memory status at a query time from memory validity metadata.

    This is intentionally deterministic and conservative: it uses memory-unit
    validity/status fields plus update/delete flags. Event-edge replay can be
    layered on top once richer domain packs are available.
    """

    active: set[str] = set()
    inactive: set[str] = set()
    deleted: set[str] = set()
    forbidden: set[str] = set()
    superseded: set[str] = set()
    restricted: set[str] = set()

    invalidated_by_update = {
        invalidated
        for memory in case.gold_memory_units
        if _started(memory.valid_from, timestamp)
        for invalidated in memory.invalidates
    }

    for memory in case.gold_memory_units:
        memory_id = memory.memory_id
        if not _started(memory.valid_from, timestamp):
            inactive.add(memory_id)
            continue
        if memory.should_delete or memory.status == "deleted":
            deleted.add(memory_id)
            forbidden.add(memory_id)
            continue
        if memory.is_sensitive or memory.status == "forbidden":
            forbidden.add(memory_id)
        if memory.privacy_level.lower() == "restricted" or memory.status == "restricted":
            restricted.add(memory_id)
        if memory_id in invalidated_by_update or memory.status == "superseded":
            superseded.add(memory_id)
            inactive.add(memory_id)
            continue
        if _is_active(memory.valid_from, memory.valid_until, timestamp) and memory.status == "active":
            if memory_id not in forbidden and memory_id not in restricted:
                active.add(memory_id)
        else:
            inactive.add(memory_id)

    return ReplayedState(
        active_memory_ids=frozenset(active),
        inactive_memory_ids=frozenset(inactive),
        deleted_memory_ids=frozenset(deleted),
        forbidden_memory_ids=frozenset(forbidden),
        superseded_memory_ids=frozenset(superseded),
        restricted_memory_ids=frozenset(restricted),
    )


def state_contract_differences(case: Case, contract: MemoryStateContract) -> list[str]:
    replayed = replay_state_at(case, contract.timestamp)
    differences: list[str] = []
    for field in (
        "active_memory_ids",
        "inactive_memory_ids",
        "deleted_memory_ids",
        "forbidden_memory_ids",
        "superseded_memory_ids",
        "restricted_memory_ids",
    ):
        expected = set(getattr(contract, field))
        actual = set(getattr(replayed, field))
        if expected != actual:
            differences.append(
                f"{contract.state_contract_id} {field} mismatch: expected={sorted(expected)} actual={sorted(actual)}"
            )
    return differences


def event_graph_state_contract_differences(case: Case, contract: MemoryStateContract) -> list[str]:
    """Cross-check a state contract against event records and event edges.

    `replay_state_at` verifies the contract against compiled memory metadata.
    This function independently replays the release-facing event graph fields
    and verifies state-transition evidence in event edges, so a bad memory
    compiler cannot silently make state contracts self-consistent.
    """

    differences = _temporal_edge_differences(case.event_edges, {event.event_id: event for event in case.events})
    replayed = _replay_event_graph_state_at(case, contract.timestamp)
    for field in (
        "active_memory_ids",
        "inactive_memory_ids",
        "deleted_memory_ids",
        "forbidden_memory_ids",
        "superseded_memory_ids",
        "restricted_memory_ids",
    ):
        expected = set(getattr(contract, field))
        actual = set(getattr(replayed, field))
        if expected != actual:
            differences.append(
                f"{contract.state_contract_id} event_graph {field} mismatch: expected={sorted(expected)} actual={sorted(actual)}"
            )
    event_by_id = {event.event_id: event for event in case.events}
    differences.extend(_transition_edge_differences(contract, case.event_edges, event_by_id))
    differences.extend(_governance_edge_differences(contract, case.event_edges, event_by_id))
    return differences


def _replay_event_graph_state_at(case: Case, timestamp: str) -> ReplayedState:
    events = tuple(event for event in case.events if event.event_type != "distractor")
    event_by_id = {event.event_id: event for event in events}
    update_superseded = _event_graph_superseded_ids(events, event_by_id, timestamp)
    deleted_sources = {
        str(event.attributes["supersedes"])
        for event in events
        if event.event_type == "deletion_request"
        and event.attributes.get("supersedes")
        and event.timestamp <= timestamp
    }

    active: set[str] = set()
    inactive: set[str] = set()
    deleted: set[str] = set()
    forbidden: set[str] = set()
    superseded: set[str] = set()
    restricted: set[str] = set()
    for event in events:
        memory_id = f"m_{event.event_id}"
        if event.timestamp > timestamp:
            inactive.add(memory_id)
            continue
        if event.event_id in update_superseded:
            superseded.add(memory_id)
            inactive.add(memory_id)
            continue
        if event.event_id in deleted_sources or bool(event.attributes.get("should_delete")):
            deleted.add(memory_id)
            forbidden.add(memory_id)
            continue

        privacy = str(event.attributes.get("privacy_level", "normal")).lower()
        should_store = bool(event.attributes.get("should_store", True))
        if privacy in {"sensitive", "restricted", "forbidden"}:
            forbidden.add(memory_id)
            restricted.add(memory_id)
            continue
        if not should_store:
            forbidden.add(memory_id)
            inactive.add(memory_id)
            continue
        active.add(memory_id)

    return ReplayedState(
        active_memory_ids=frozenset(active),
        inactive_memory_ids=frozenset(inactive),
        deleted_memory_ids=frozenset(deleted),
        forbidden_memory_ids=frozenset(forbidden),
        superseded_memory_ids=frozenset(superseded),
        restricted_memory_ids=frozenset(restricted),
    )


def _event_graph_superseded_ids(
    events: tuple[Event, ...],
    event_by_id: dict[str, Event],
    timestamp: str,
) -> set[str]:
    superseded: set[str] = set()
    for event in events:
        if (
            event.event_type != "fact_update"
            or not event.attributes.get("supersedes")
            or event.timestamp > timestamp
        ):
            continue
        target_id = str(event.attributes["supersedes"])
        target = event_by_id.get(target_id)
        if target is None:
            continue
        superseded.add(target_id)
        for candidate in events:
            if (
                candidate.event_type == "fact_reinforcement"
                and candidate.subject == target.subject
                and candidate.object == target.object
                and candidate.timestamp <= event.timestamp
            ):
                superseded.add(candidate.event_id)
    return superseded


def _temporal_edge_differences(edges: tuple[EventEdge, ...], event_by_id: dict[str, Event]) -> list[str]:
    differences: list[str] = []
    for edge in edges:
        source = event_by_id.get(edge.source_event_id)
        target = event_by_id.get(edge.target_event_id)
        if source is None or target is None:
            continue
        if edge.edge_type == "temporal_before" and source.timestamp > target.timestamp:
            differences.append(
                f"temporal_before edge violates timestamp order: {edge.source_event_id}>{edge.target_event_id}"
            )
        if edge.edge_type in {"superseded_by", "invalidates"} and source.timestamp > target.timestamp:
            differences.append(
                f"{edge.edge_type} edge violates causal timestamp order: {edge.source_event_id}>{edge.target_event_id}"
            )
    return differences


def _transition_edge_differences(
    contract: MemoryStateContract,
    edges: tuple[EventEdge, ...],
    event_by_id: dict[str, Event],
) -> list[str]:
    edge_types = {(edge.source_event_id, edge.target_event_id, edge.edge_type) for edge in edges}
    transition_keys = {
        (_event_id_from_memory_id(transition.from_memory_id), transition.trigger_event_id, transition.transition_type)
        for transition in contract.transitions
    }
    differences: list[str] = []
    for transition in contract.transitions:
        from_event = _event_id_from_memory_id(transition.from_memory_id)
        to_event = _event_id_from_memory_id(transition.to_memory_id)
        trigger_event = transition.trigger_event_id
        if transition.transition_type == "update":
            if (from_event, trigger_event, "superseded_by") not in edge_types:
                differences.append(f"{contract.state_contract_id} update transition lacks superseded_by edge for {from_event}->{trigger_event}")
            if to_event != trigger_event:
                differences.append(f"{contract.state_contract_id} update transition to_memory does not match trigger event {trigger_event}")
        elif transition.transition_type == "delete":
            if (from_event, trigger_event, "invalidates") not in edge_types:
                differences.append(f"{contract.state_contract_id} delete transition lacks invalidates edge for {from_event}->{trigger_event}")
            if to_event != trigger_event:
                differences.append(f"{contract.state_contract_id} delete transition to_memory does not match trigger event {trigger_event}")
        elif transition.transition_type == "retain":
            if (from_event, trigger_event, "supports") not in edge_types:
                differences.append(f"{contract.state_contract_id} retain transition lacks supports edge for {from_event}->{trigger_event}")
            if to_event != trigger_event:
                differences.append(f"{contract.state_contract_id} retain transition to_memory does not match trigger event {trigger_event}")
    for edge in edges:
        target = event_by_id.get(edge.target_event_id)
        if target is None or target.timestamp > contract.timestamp:
            continue
        if edge.edge_type == "superseded_by" and target.event_type == "fact_update":
            if (edge.source_event_id, edge.target_event_id, "update") not in transition_keys:
                differences.append(
                    f"{contract.state_contract_id} lacks update transition for edge {edge.source_event_id}->{edge.target_event_id}"
                )
        elif edge.edge_type == "invalidates" and target.event_type == "deletion_request":
            if (edge.source_event_id, edge.target_event_id, "delete") not in transition_keys:
                differences.append(
                    f"{contract.state_contract_id} lacks delete transition for edge {edge.source_event_id}->{edge.target_event_id}"
                )
        elif edge.edge_type == "supports" and target.event_type == "retention_confirmation":
            if (edge.source_event_id, edge.target_event_id, "retain") not in transition_keys:
                differences.append(
                    f"{contract.state_contract_id} lacks retain transition for edge {edge.source_event_id}->{edge.target_event_id}"
                )
    return differences


def _governance_edge_differences(
    contract: MemoryStateContract,
    edges: tuple[EventEdge, ...],
    event_by_id: dict[str, Event],
) -> list[str]:
    incoming = {(edge.target_event_id, edge.edge_type) for edge in edges}
    blocked = set(contract.deleted_memory_ids) | set(contract.forbidden_memory_ids) | set(contract.restricted_memory_ids)
    differences: list[str] = []
    for event in event_by_id.values():
        if event.timestamp > contract.timestamp:
            continue
        memory_id = f"m_{event.event_id}"
        if event.event_type == "sensitive_disclosure":
            if (event.event_id, "forbids") not in incoming:
                differences.append(f"{contract.state_contract_id} sensitive event lacks incoming forbids edge: {event.event_id}")
            if memory_id not in blocked:
                differences.append(f"{contract.state_contract_id} sensitive memory is not blocked: {memory_id}")
        elif event.event_type == "authorized_sensitive_memory":
            if (event.event_id, "authorizes") not in incoming:
                differences.append(f"{contract.state_contract_id} authorized sensitive event lacks incoming authorizes edge: {event.event_id}")
            if memory_id in blocked:
                differences.append(f"{contract.state_contract_id} authorized sensitive memory is blocked: {memory_id}")
    return differences


def _event_id_from_memory_id(memory_id: str) -> str:
    return memory_id[2:] if memory_id.startswith("m_") else memory_id


def _is_active(valid_from: str | None, valid_until: str | None, timestamp: str) -> bool:
    return _started(valid_from, timestamp) and not _expired(valid_until, timestamp)


def _started(valid_from: str | None, timestamp: str) -> bool:
    return valid_from is None or valid_from <= timestamp


def _expired(valid_until: str | None, timestamp: str) -> bool:
    return valid_until is not None and valid_until <= timestamp
