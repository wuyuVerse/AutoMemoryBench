"""Official managed-service Hindsight API client factory for AutoMemoryBench integration runs."""

from __future__ import annotations

import asyncio
from typing import Any

from amb.clients.core.common import effective_api_key, ensure_source_path


class HindsightOfficialClient:
    """Thin adapter over Hindsight's official Python client surface."""

    def __init__(self, client: Any, *, bank_id: str, default_limit: int = 5) -> None:
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        self.client = client
        self.bank_id = bank_id
        self.default_limit = default_limit
        self.case_id: str | None = None

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        return {"ok": True}

    def add(self, content: Any = None, *, messages: Any = None, metadata: dict[str, Any] | None = None, **_: Any) -> Any:
        text = _content_from_payload(content=content, messages=messages)
        return _run_async(self.client.aretain(bank_id=self.bank_id, content=text, metadata=metadata or {}))

    def search(self, query: str | None = None, *, limit: int | None = None, top_k: int | None = None, **_: Any) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required for Hindsight recall")
        max_tokens = int(limit or top_k or self.default_limit) * 512
        response = _run_async(self.client.arecall(bank_id=self.bank_id, query=str(query), max_tokens=max_tokens))
        results = getattr(response, "results", []) or []
        return [
            {
                "id": str(getattr(item, "id", idx)),
                "content": str(getattr(item, "text", item)),
                "score": getattr(item, "score", None),
            }
            for idx, item in enumerate(results, start=1)
        ]


def create_client(
    *,
    source_root: str = "related_work/repos/hindsight/hindsight-integrations/openai-agents",
    hindsight_api_url: str = "http://localhost:8888",
    api_key: str | None = None,
    api_key_env: str = "HINDSIGHT_API_KEY",
    bank_id: str = "amst",
    default_limit: int = 5,
) -> HindsightOfficialClient:
    """Create a client using Hindsight's official OpenAI Agents integration."""

    ensure_source_path(source_root)
    from hindsight_openai_agents._client import resolve_client  # type: ignore

    resolved_key = effective_api_key(api_key=api_key, api_key_env=api_key_env)
    client = resolve_client(None, hindsight_api_url, resolved_key)
    return HindsightOfficialClient(client, bank_id=bank_id, default_limit=default_limit)


def _run_async(awaitable: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError("Hindsight official client cannot run blocking AMST wrapper inside an active event loop") from None


def _content_from_payload(*, content: Any, messages: Any) -> str:
    if messages is not None:
        if isinstance(messages, list):
            return "\n".join(str(item.get("content", item)) if isinstance(item, dict) else str(item) for item in messages)
        if isinstance(messages, dict):
            return str(messages.get("content", messages))
        return str(messages)
    if content is None:
        raise ValueError("content or messages is required")
    return str(content)
