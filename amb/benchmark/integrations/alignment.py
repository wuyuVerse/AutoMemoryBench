"""Post-hoc alignment from provider memory hits to AMST memory ids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from amb.benchmark.integrations.base import MemoryRecord
from amb.benchmark.schemas.models import Case


@dataclass(frozen=True)
class AlignmentIndex:
    turn_to_memory_ids: dict[str, tuple[str, ...]]
    event_to_memory_ids: dict[str, tuple[str, ...]]
    content_to_memory_id: dict[str, str]


def build_alignment_index(case: Case) -> AlignmentIndex:
    turn_map: dict[str, list[str]] = {}
    event_map: dict[str, list[str]] = {}
    content_map: dict[str, str] = {}
    for memory in case.gold_memory_units:
        for turn_id in memory.source_turn_ids:
            turn_map.setdefault(turn_id, []).append(memory.memory_id)
        for event_id in memory.source_event_ids:
            event_map.setdefault(event_id, []).append(memory.memory_id)
        content_map[_fingerprint(memory.content)] = memory.memory_id
    return AlignmentIndex(
        turn_to_memory_ids={key: tuple(values) for key, values in turn_map.items()},
        event_to_memory_ids={key: tuple(values) for key, values in event_map.items()},
        content_to_memory_id=content_map,
    )


def align_records(records: list[MemoryRecord], index: AlignmentIndex | None) -> tuple[str, ...]:
    if index is None:
        return tuple(record.memory_id for record in records)
    aligned: list[str] = []
    seen: set[str] = set()
    for record in records:
        for memory_id in _record_memory_ids(record, index):
            if memory_id not in seen:
                aligned.append(memory_id)
                seen.add(memory_id)
    return tuple(aligned)


def _record_memory_ids(record: MemoryRecord, index: AlignmentIndex) -> tuple[str, ...]:
    metadata = record.metadata or {}
    ids: list[str] = []
    for key in ("source_turn_ids", "turn_ids"):
        for turn_id in _as_tuple(metadata.get(key)):
            ids.extend(index.turn_to_memory_ids.get(str(turn_id), ()))
    for key in ("source_turn_id", "turn_id"):
        value = metadata.get(key)
        if value is not None:
            ids.extend(index.turn_to_memory_ids.get(str(value), ()))
    for key in ("source_event_ids", "event_ids"):
        for event_id in _as_tuple(metadata.get(key)):
            ids.extend(index.event_to_memory_ids.get(str(event_id), ()))
    for key in ("source_event_id", "event_id"):
        value = metadata.get(key)
        if value is not None:
            ids.extend(index.event_to_memory_ids.get(str(value), ()))
    content_hit = index.content_to_memory_id.get(_fingerprint(record.content))
    if content_hit:
        ids.append(content_hit)
    return tuple(ids)


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _fingerprint(text: str) -> str:
    return " ".join(text.lower().split())
