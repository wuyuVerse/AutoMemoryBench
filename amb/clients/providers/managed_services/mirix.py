"""Official managed-service MIRIX client factory for AutoMemoryBench integration runs."""

from __future__ import annotations

from typing import Any

from amb.clients.core.common import effective_api_key


class MirixOfficialClient:
    """Thin adapter over MIRIX's official ``MirixClient`` memory API."""

    def __init__(
        self,
        client: Any,
        *,
        default_user_id: str = "amst-user",
        default_limit: int = 5,
        initialize_provider: str | None = None,
        initialize_config: dict[str, Any] | None = None,
    ) -> None:
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        self.client = client
        self.default_user_id = default_user_id
        self.default_limit = default_limit
        self.case_id: str | None = None
        self._written: list[dict[str, Any]] = []
        if initialize_config is not None:
            self.client.initialize_meta_agent(config=initialize_config)
        elif initialize_provider:
            self.client.initialize_meta_agent(provider=initialize_provider)

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        self._written = []
        return {"ok": True, "user_id": self._user_id(user_id)}

    def add(
        self,
        content: Any = None,
        *,
        messages: Any = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> Any:
        formatted = _messages_from_payload(content=content, messages=messages)
        resolved_user_id = self._user_id(user_id)
        result = self.client.add(user_id=resolved_user_id, messages=formatted)
        self._written.append({"user_id": resolved_user_id, "messages": formatted, "metadata": metadata or {}})
        return {"result": result, "user_id": resolved_user_id}

    def search(
        self,
        query: str | None = None,
        *,
        user_id: str | None = None,
        limit: int | None = None,
        top_k: int | None = None,
        memory_type: str | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required for MIRIX search")
        resolved_user_id = self._user_id(user_id)
        k = int(limit or top_k or self.default_limit)
        if hasattr(self.client, "retrieve_with_conversation"):
            raw = self.client.retrieve_with_conversation(
                user_id=resolved_user_id,
                messages=[{"role": "user", "content": [{"type": "text", "text": str(query)}]}],
                limit=k,
            )
        else:
            kwargs: dict[str, Any] = {"user_id": resolved_user_id, "query": str(query), "limit": k}
            if memory_type:
                kwargs["memory_type"] = memory_type
            raw = self.client.search(**kwargs)
        return _normalize_results(raw)

    def get_all(self, **_: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": f"mirix-observation-{idx}",
                "content": _flatten_messages(row["messages"]),
                "metadata": {"user_id": row["user_id"], **dict(row.get("metadata") or {})},
            }
            for idx, row in enumerate(self._written, start=1)
        ]

    def delete(self, memory_id: str | None = None, **_: Any) -> dict[str, Any]:
        return {
            "deleted": False,
            "reason": "official MIRIX AMST path does not expose stable per-memory delete",
            "memory_id": memory_id,
        }

    def _user_id(self, user_id: str | None = None) -> str:
        return str(user_id or self.case_id or self.default_user_id)


def create_client(
    *,
    api_key: str | None = None,
    api_key_env: str = "MIRIX_API_KEY",
    base_url: str | None = None,
    default_user_id: str = "amst-user",
    default_limit: int = 5,
    initialize_provider: str | None = None,
    initialize_config: dict[str, Any] | None = None,
) -> MirixOfficialClient:
    """Create a MIRIX client using the official ``mirix-client`` package."""

    from mirix import MirixClient  # type: ignore

    resolved_key = effective_api_key(api_key=api_key, api_key_env=api_key_env)
    kwargs: dict[str, Any] = {}
    if resolved_key:
        kwargs["api_key"] = resolved_key
    if base_url:
        kwargs["base_url"] = base_url
    client = MirixClient(**kwargs)
    return MirixOfficialClient(
        client,
        default_user_id=default_user_id,
        default_limit=default_limit,
        initialize_provider=initialize_provider,
        initialize_config=initialize_config,
    )


def _messages_from_payload(*, content: Any, messages: Any) -> list[dict[str, Any]]:
    if messages is not None:
        if isinstance(messages, list):
            return [_normalize_message(item) for item in messages]
        if isinstance(messages, dict):
            return [_normalize_message(messages)]
        return [_normalize_message({"role": "user", "content": str(messages)})]
    if content is None:
        raise ValueError("content or messages is required")
    return [_normalize_message({"role": "user", "content": str(content)})]


def _normalize_message(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        return {"role": "user", "content": [{"type": "text", "text": str(message)}]}
    role = str(message.get("role") or "user")
    content = message.get("content", "")
    if isinstance(content, list):
        return {"role": role, "content": content}
    return {"role": role, "content": [{"type": "text", "text": str(content)}]}


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            chunks.extend(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
        else:
            chunks.append(str(content))
    return "\n".join(chunks)


def _normalize_results(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict) and isinstance(raw.get("results"), list):
        items = raw["results"]
    elif isinstance(raw, dict) and isinstance(raw.get("memories"), dict):
        items = []
        for memory_type, payload in raw["memories"].items():
            values = payload.get("results", payload.get("items", [])) if isinstance(payload, dict) else payload
            if isinstance(values, list):
                for value in values:
                    items.append({"memory_type": memory_type, "value": value})
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        if isinstance(item, dict):
            value = item.get("value", item)
            if isinstance(value, dict):
                content = value.get("summary") or value.get("details") or value.get("content") or value.get("caption") or value
                score = value.get("score")
                item_id = value.get("id", idx)
            else:
                content = item.get("summary") or item.get("details") or item.get("content") or value
                score = item.get("score")
                item_id = item.get("id", idx)
            memory_type = item.get("memory_type")
        else:
            content = getattr(item, "summary", getattr(item, "content", item))
            score = getattr(item, "score", None)
            item_id = getattr(item, "id", idx)
            memory_type = getattr(item, "memory_type", None)
        rows.append({"id": str(item_id), "content": str(content), "score": score, "metadata": {"memory_type": memory_type}})
    return rows
