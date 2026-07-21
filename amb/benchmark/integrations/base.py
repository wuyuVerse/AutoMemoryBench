"""Duck-typed adapter for external memory-system clients.

This module intentionally has no dependency on Mem0, Letta, LangMem, Zep, or
Graphiti packages. Named integrations configure likely method names and call
patterns, while tests use fake clients with the same minimal behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from time import perf_counter
from typing import Any, Iterable

from amb.benchmark.schemas.models import Case


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    content: str
    score: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class IntegrationConfig:
    system_id: str
    top_k: int = 5
    add_methods: tuple[str, ...] = ("add", "add_memory", "save", "put")
    search_methods: tuple[str, ...] = ("search", "retrieve", "query")
    delete_methods: tuple[str, ...] = ("delete", "delete_memory", "remove")
    reset_methods: tuple[str, ...] = ("reset", "clear")
    export_methods: tuple[str, ...] = ("export_memory", "get_all", "list", "all")
    observe_roles: tuple[str, ...] = ("user", "assistant")


class ExternalMemoryAgent:
    """Streaming AMST agent backed by a duck-typed external memory client."""

    def __init__(self, client: Any, config: IntegrationConfig) -> None:
        if config.top_k <= 0:
            raise ValueError("top_k must be positive")
        self.client = client
        self.config = config
        self.case_id: str | None = None
        self.observation_count = 0
        self.write_count = 0
        self._alignment_index = None

    def reset(self, case_id: str) -> None:
        self.case_id = case_id
        self.observation_count = 0
        self.write_count = 0
        method = _first_method(self.client, self.config.reset_methods)
        if method is not None:
            _call_first(
                (
                    lambda: method(case_id=case_id),
                    lambda: method(user_id=case_id),
                    lambda: method(case_id),
                    lambda: method(),
                )
            )

    def set_case_reference(self, case: Case) -> None:
        """Register AMST metadata for post-hoc provider-id alignment.

        The external client never receives gold memory ids. Alignment is applied
        only after retrieval returns provider hits, using source turn/event
        metadata captured during `observe`.
        """

        from amb.benchmark.integrations.alignment import build_alignment_index

        self._alignment_index = build_alignment_index(case)

    def observe(self, observation: dict[str, Any]) -> None:
        self.observation_count += 1
        if observation.get("role") not in self.config.observe_roles:
            return
        content = str(observation.get("content", ""))
        if not content:
            return
        method = _first_method(self.client, self.config.add_methods)
        if method is None:
            return
        metadata = {
            "case_id": observation.get("case_id", self.case_id),
            "domain": observation.get("domain"),
            "session_id": observation.get("session_id"),
            "turn_id": observation.get("turn_id"),
            "timestamp": observation.get("timestamp"),
            "role": observation.get("role"),
        }
        result = _call_first(
            (
                lambda: method(content, user_id=self.case_id, metadata=metadata),
                lambda: method(messages=[{"role": observation.get("role"), "content": content}], user_id=self.case_id, metadata=metadata),
                lambda: method(content, metadata=metadata),
                lambda: method(content),
                lambda: method(observation),
            )
        )
        if result is not _NO_RESULT:
            self.write_count += 1

    def answer_or_act(self, probe: dict[str, Any]) -> dict[str, Any]:
        prompt = str(probe.get("prompt", ""))
        start = perf_counter()
        records = self.retrieve(prompt, int(probe.get("top_k") or self.config.top_k))
        latency_ms = (perf_counter() - start) * 1000.0
        response = _response_from_records(records)
        activated_memory_ids = self._aligned_memory_ids(records)
        return {
            "memory_needed": bool(records),
            "activated_memory_ids": list(activated_memory_ids),
            "response": response,
            "memory_operations": [],
            "cost": {
                "input_tokens": _token_proxy(prompt),
                "output_tokens": _token_proxy(response),
                "latency_ms": latency_ms,
                "retrieval_latency_ms": latency_ms,
                "storage_bytes": float(sum(len(record.content.encode("utf-8")) for record in records)),
            },
        }

    def retrieve(self, query: str, k: int) -> list[MemoryRecord]:
        method = _first_method(self.client, self.config.search_methods)
        if method is None:
            return []
        raw = _call_first(
            (
                lambda: method(query=query, user_id=self.case_id, limit=k),
                lambda: method(query=query, user_id=self.case_id, top_k=k),
                lambda: method(query, user_id=self.case_id, limit=k),
                lambda: method(query, user_id=self.case_id, top_k=k),
                lambda: method(query, k),
                lambda: method(query),
            )
        )
        if raw is _NO_RESULT:
            return []
        return _normalize_records(raw)[:k]

    def delete(self, deletion_request: dict[str, Any]) -> dict[str, Any]:
        method = _first_method(self.client, self.config.delete_methods)
        if method is None:
            return {"deleted": False, "reason": "delete method unavailable"}
        memory_id = deletion_request.get("memory_id")
        result = _call_first(
            (
                lambda: method(memory_id=memory_id, user_id=self.case_id),
                lambda: method(memory_id),
                lambda: method(deletion_request),
            )
        )
        return {"deleted": result is not _NO_RESULT, "raw_result": None if result is _NO_RESULT else result}

    def export_memory(self) -> list[dict[str, Any]]:
        method = _first_method(self.client, self.config.export_methods)
        if method is None:
            return []
        raw = _call_first(
            (
                lambda: method(user_id=self.case_id),
                lambda: method(self.case_id),
                lambda: method(),
            )
        )
        if raw is _NO_RESULT:
            return []
        return [record.__dict__ for record in _normalize_records(raw)]

    def export_trace(self) -> dict[str, Any]:
        return {
            "system_id": self.config.system_id,
            "case_id": self.case_id,
            "observations_seen": self.observation_count,
            "writes_attempted": self.write_count,
            "client_type": type(self.client).__name__,
        }

    def _aligned_memory_ids(self, records: list[MemoryRecord]) -> tuple[str, ...]:
        from amb.benchmark.integrations.alignment import align_records

        return align_records(records, self._alignment_index)


def _first_method(client: Any, names: Iterable[str]):
    for name in names:
        method = getattr(client, name, None)
        if callable(method):
            return method
    return None


class _NoResult:
    pass


_NO_RESULT = _NoResult()


def _call_first(attempts):
    last_error: TypeError | None = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        return _NO_RESULT
    return _NO_RESULT


def _normalize_records(raw: Any) -> list[MemoryRecord]:
    items = _unwrap_items(raw)
    return [_record_from_item(item, index) for index, item in enumerate(items, start=1)]


def _unwrap_items(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        for key in ("results", "memories", "items", "data"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
        return [raw]
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    return [raw]


def _record_from_item(item: Any, index: int) -> MemoryRecord:
    if isinstance(item, dict):
        memory_id = _first_value(item, ("memory_id", "id", "uuid", "key"))
        content = _first_value(item, ("content", "memory", "text", "value", "summary"))
        score = _optional_float(_first_value(item, ("score", "relevance", "similarity")))
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else None
    else:
        memory_id = _first_attr(item, ("memory_id", "id", "uuid", "key"))
        content = _first_attr(item, ("content", "memory", "text", "value", "summary"))
        score = _optional_float(_first_attr(item, ("score", "relevance", "similarity")))
        metadata = getattr(item, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = None
    content = str(content or item)
    memory_id = str(memory_id or _stable_record_id(content, index))
    return MemoryRecord(memory_id=memory_id, content=content, score=score, metadata=metadata)


def _first_value(values: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = values.get(key)
        if value is not None:
            return value
    return None


def _first_attr(item: Any, names: Iterable[str]) -> Any:
    for name in names:
        value = getattr(item, name, None)
        if value is not None:
            return value
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stable_record_id(content: str, index: int) -> str:
    digest = sha1(content.encode("utf-8")).hexdigest()[:12]
    return f"external:{index}:{digest}"


def _response_from_records(records: list[MemoryRecord]) -> str:
    if not records:
        return "I do not have enough retrieved memory to answer."
    return "\n".join(record.content for record in records)


def _token_proxy(text: str) -> float:
    return float(max(1, len(text.split())))
