"""Local official-source MemU client factory for AutoMemoryBench integration.

MemU (github.com/NevaMind-AI/memU) is an agentic memory framework: explicit
``create_memory_item`` / ``retrieve`` over an OpenAI-compatible chat+embed
endpoint. Both calls are async; we wrap them with ``asyncio.run`` per AMB call.

MemU uses one base_url for chat AND embedding (cosine vector retrieval), so it
runs cleanly on providers that serve both — i.e. the configured chat models
(glm/deepseek), where our bge-m3 embedder also lives. provider-specific chat
models are skipped (their endpoints don't serve bge-m3).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from amb.clients.core.common import ensure_site_packages_from_venv, ensure_source_path

DEFAULT_MEMU_VENV = Path(__file__).resolve().parents[4] / ".venv-memu"
DEFAULT_MEMU_SOURCE = "related_work/repos/memU/src"


class MemUOfficialSourceClient:
    """Adapter over MemU's MemoryService (create_memory_item / retrieve)."""

    def __init__(self, *, llm_profiles: dict[str, Any], default_limit: int = 5) -> None:
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        self._llm_profiles = llm_profiles
        self.default_limit = default_limit
        self.case_id: str | None = None
        self._svc: Any = None
        self._MemoryService: Any = None

    def _new_service(self) -> Any:
        if self._MemoryService is None:
            from memu.app import MemoryService  # type: ignore
            self._MemoryService = MemoryService
        return self._MemoryService(llm_profiles=self._llm_profiles)

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        self._svc = self._new_service()
        return {"ok": True}

    def add(
        self,
        content: Any = None,
        *,
        messages: Any = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if self._svc is None:
            self._svc = self._new_service()
        text = _content_from_payload(content=content, messages=messages)
        asyncio.run(self._svc.create_memory_item(
            memory_type="conversation",
            memory_content=text,
            memory_categories=["general"],
        ))
        return {"added": 1, "user_id": user_id or self.case_id}

    def search(
        self,
        query: str | None = None,
        *,
        limit: int | None = None,
        top_k: int | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required for MemU search")
        if self._svc is None:
            return []
        k = int(limit or top_k or self.default_limit)
        result = asyncio.run(self._svc.retrieve(queries=[{"role": "user", "content": str(query)}]))
        items = (result or {}).get("items") if isinstance(result, dict) else None
        out: list[dict[str, Any]] = []
        for it in (items or [])[:k]:
            if isinstance(it, dict):
                text = it.get("content") or it.get("memory_content") or it.get("text") or it.get("summary") or ""
                meta = {kk: vv for kk, vv in it.items() if kk not in ("content", "memory_content", "text", "summary")}
            else:
                text = str(it); meta = {}
            out.append({"id": meta.get("id", f"memu-{len(out)+1}"), "content": str(text), "metadata": meta})
        return out

    def get_all(self, **_: Any) -> list[dict[str, Any]]:
        return []

    def delete(self, memory_id: str | None = None, **_: Any) -> dict[str, Any]:
        return {"deleted": False, "memory_id": memory_id}


def create_client(
    *,
    source_root: str = DEFAULT_MEMU_SOURCE,
    default_limit: int = 5,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    api_key: str | None = None,
    embedding_model: str | None = None,
    **_: Any,
) -> MemUOfficialSourceClient:
    """Create a MemU client on our OpenAI-compatible endpoint (chat + embed share base_url)."""
    ensure_site_packages_from_venv(DEFAULT_MEMU_VENV)
    ensure_source_path(source_root)

    key = api_key or (os.environ.get(api_key_env or "OPENAI_API_KEY") or "")
    profiles = {"default": {
        "provider": "openai",
        "client_backend": "openai",
        "base_url": base_url,
        "api_key": key,
        "chat_model": model or "gpt-4.1-mini",
        "embed_model": embedding_model or "BAAI/bge-m3",
    }}
    return MemUOfficialSourceClient(llm_profiles=profiles, default_limit=default_limit)


def _content_from_payload(*, content: Any, messages: Any) -> str:
    if messages is not None:
        values = messages if isinstance(messages, list) else [messages]
        parts = []
        for v in values:
            if isinstance(v, dict):
                parts.append(f"{v.get('role', 'user')}: {v.get('content', '')}")
            else:
                parts.append(str(v))
        return "\n".join(parts)
    if content is None:
        raise ValueError("content or messages is required")
    return str(content)
