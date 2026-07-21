"""Official framework-SDK OpenAI Agents session-memory client factory for AutoMemoryBench runs."""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path
from typing import Any


class OpenAIAgentsSessionMemoryClient:
    """Thin adapter over the official Agents SDK ``Session`` protocol.

    This is a conversation-history/session-memory baseline. It is not ChatGPT
    saved memory and is not a semantic long-term memory implementation.
    """

    def __init__(self, session: Any, *, default_limit: int = 5) -> None:
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        self.session = session
        self.default_limit = default_limit
        self.case_id: str | None = None

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        if hasattr(self.session, "clear_session"):
            _run_async(self.session.clear_session())
        return {"ok": True, "session_id": getattr(self.session, "session_id", self.case_id)}

    def add(
        self,
        content: Any = None,
        *,
        messages: Any = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        items = _items_from_payload(content=content, messages=messages, metadata=metadata)
        _run_async(self.session.add_items(items))
        return {"added": len(items), "session_id": getattr(self.session, "session_id", self.case_id), "items": items}

    def search(self, query: str | None = None, *, limit: int | None = None, top_k: int | None = None, **_: Any) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required for OpenAI session-memory search")
        k = int(limit or top_k or self.default_limit)
        items = _run_async(self.session.get_items(limit=None))
        scored = [_scored_item(idx, item, query=str(query)) for idx, item in enumerate(items or [], start=1)]
        return [item for item in sorted(scored, key=lambda item: item["score"], reverse=True) if item["score"] > 0][:k]

    def get_all(self, *, limit: int | None = None, **_: Any) -> list[dict[str, Any]]:
        items = _run_async(self.session.get_items(limit=limit))
        return [
            {
                "id": f"openai-session-item-{idx}",
                "content": _item_content(item),
                "metadata": {"role": _item_role(item), "raw_item": item},
            }
            for idx, item in enumerate(items or [], start=1)
        ]

    def delete(self, memory_id: str | None = None, **_: Any) -> dict[str, Any]:
        if hasattr(self.session, "pop_item") and (memory_id is None or memory_id == "latest"):
            item = _run_async(self.session.pop_item())
            return {"deleted": item is not None, "memory_id": memory_id or "latest"}
        return {
            "deleted": False,
            "memory_id": memory_id,
            "reason": "OpenAI Agents SDK Session protocol exposes pop_item/clear_session, not arbitrary semantic-memory delete",
        }


def create_client(
    *,
    session_id: str = "amst_openai_session_memory",
    db_path: str | None = None,
    default_limit: int = 5,
) -> OpenAIAgentsSessionMemoryClient:
    """Create an OpenAI Agents SDK SQLiteSession-backed memory baseline."""

    from agents import SQLiteSession  # type: ignore

    resolved_db_path = Path(db_path) if db_path else Path(tempfile.mkdtemp(prefix="amst-openai-session-")) / "session.db"
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
    return OpenAIAgentsSessionMemoryClient(SQLiteSession(session_id, db_path=str(resolved_db_path)), default_limit=default_limit)


def _run_async(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError("OpenAI Agents SDK session adapter cannot run blocking AMST wrapper inside an active event loop") from None


def _items_from_payload(*, content: Any, messages: Any, metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    if messages is not None:
        values = messages if isinstance(messages, list) else [messages]
        return [_normalize_message(value, metadata=metadata) for value in values]
    if content is None:
        raise ValueError("content or messages is required")
    return [_normalize_message({"role": (metadata or {}).get("role") or "user", "content": str(content)}, metadata=metadata)]


def _normalize_message(message: Any, *, metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(message, dict):
        message = {"role": "user", "content": str(message)}
    item = {"role": str(message.get("role") or "user"), "content": str(message.get("content", ""))}
    if metadata:
        item["metadata"] = dict(metadata)
    return item


def _scored_item(idx: int, item: Any, *, query: str) -> dict[str, Any]:
    content = _item_content(item)
    score = _lexical_overlap(query, content)
    return {
        "id": f"openai-session-item-{idx}",
        "content": content,
        "score": score,
        "metadata": {"role": _item_role(item), "raw_item": item, "memory_kind": "agents_sdk_session_history"},
    }


def _item_content(item: Any) -> str:
    if isinstance(item, dict):
        content = item.get("content", "")
        if isinstance(content, list):
            return " ".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
        return str(content)
    return str(item)


def _item_role(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("role") or "")
    return ""


def _lexical_overlap(left: str, right: str) -> float:
    left_terms = {term for term in re.findall(r"[a-z0-9_]+", left.lower()) if len(term) > 2}
    right_terms = {term for term in re.findall(r"[a-z0-9_]+", right.lower()) if len(term) > 2}
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(len(left_terms), 1)
