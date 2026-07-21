"""Official managed-service Memobase SDK client factory for AutoMemoryBench runs."""

from __future__ import annotations

from typing import Any

from amb.clients.core.common import effective_api_key


class MemobaseOfficialClient:
    """Thin adapter over Memobase's official user/blob/context API."""

    def __init__(
        self,
        *,
        client: Any,
        chat_blob_cls: Any,
        default_user_id: str | None = None,
        default_limit: int = 5,
        flush_sync: bool = True,
    ) -> None:
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        self.client = client
        self.chat_blob_cls = chat_blob_cls
        self.default_user_id = default_user_id
        self.default_limit = default_limit
        self.flush_sync = flush_sync
        self.case_id: str | None = None
        self.user_id: str | None = default_user_id
        self._user: Any | None = None
        self._written: list[dict[str, Any]] = []

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        self.user_id = user_id or self.default_user_id
        self._user = None
        self._written = []
        user = self._ensure_user()
        return {"ok": True, "user_id": self.user_id or _user_id_from_object(user)}

    def add(
        self,
        content: Any = None,
        *,
        messages: Any = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        formatted = _messages_from_payload(content=content, messages=messages, metadata=metadata)
        user = self._ensure_user(user_id=user_id)
        blob = self.chat_blob_cls(messages=formatted)
        result = user.insert(blob)
        if self.flush_sync and hasattr(user, "flush"):
            user.flush(sync=True)
        row = {
            "user_id": self.user_id or _user_id_from_object(user),
            "messages": formatted,
            "metadata": metadata or {},
            "result": result,
        }
        self._written.append(row)
        return {"result": result, "user_id": row["user_id"]}

    def search(
        self,
        query: str | None = None,
        *,
        user_id: str | None = None,
        limit: int | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required for Memobase context retrieval")
        user = self._ensure_user(user_id=user_id)
        chats = [{"role": "user", "content": str(query)}]
        kwargs: dict[str, Any] = {"chats": chats}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        raw = user.context(**kwargs)
        rows = _normalize_context(raw, limit=int(limit or top_k or self.default_limit))
        return rows

    def get_all(self, *, user_id: str | None = None, **_: Any) -> list[dict[str, Any]]:
        user = self._ensure_user(user_id=user_id)
        profile = user.profile() if hasattr(user, "profile") else None
        rows = [
            {
                "id": f"memobase-observation-{idx}",
                "content": _flatten_messages(row["messages"]),
                "metadata": {"user_id": row["user_id"], **dict(row.get("metadata") or {})},
            }
            for idx, row in enumerate(self._written, start=1)
        ]
        if profile:
            rows.append({"id": "memobase-profile", "content": str(profile), "metadata": {"kind": "profile"}})
        return rows

    def delete(self, memory_id: str | None = None, *, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        user = self._ensure_user(user_id=user_id)
        if memory_id and hasattr(user, "delete"):
            return {"deleted": bool(user.delete(memory_id)), "memory_id": memory_id}
        return {
            "deleted": False,
            "reason": "official Memobase AMST path does not expose a stable per-blob delete in the adapter contract",
            "memory_id": memory_id,
        }

    def _ensure_user(self, *, user_id: str | None = None) -> Any:
        if user_id and user_id != self.user_id:
            self.user_id = user_id
            self._user = None
        if self._user is not None:
            return self._user
        if self.user_id:
            self._user = self.client.get_user(self.user_id)
            return self._user
        profile = {"amst_case_id": self.case_id or "amst-case"}
        self._user = self.client.add_user(profile=profile)
        self.user_id = _user_id_from_object(self._user)
        return self._user


def create_client(
    *,
    api_key: str | None = None,
    api_key_env: str = "MEMOBASE_API_KEY",
    project_url: str | None = None,
    project_url_env: str = "MEMOBASE_PROJECT_URL",
    default_user_id: str | None = None,
    default_limit: int = 5,
    flush_sync: bool = True,
) -> MemobaseOfficialClient:
    """Create a Memobase client using the official ``memobase`` SDK."""

    from memobase import ChatBlob, MemoBaseClient  # type: ignore

    resolved_key = effective_api_key(api_key=api_key, api_key_env=api_key_env)
    if not resolved_key:
        raise RuntimeError(f"{api_key_env} is required for Memobase official SDK runs")
    import os

    resolved_project_url = project_url or os.getenv(project_url_env)
    if not resolved_project_url:
        raise RuntimeError(f"{project_url_env} or project_url is required for Memobase official SDK runs")
    client = MemoBaseClient(project_url=resolved_project_url, api_key=resolved_key)
    return MemobaseOfficialClient(
        client=client,
        chat_blob_cls=ChatBlob,
        default_user_id=default_user_id,
        default_limit=default_limit,
        flush_sync=flush_sync,
    )


def _messages_from_payload(*, content: Any, messages: Any, metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    if messages is not None:
        values = messages if isinstance(messages, list) else [messages]
        return [_normalize_message(value) for value in values]
    if content is None:
        raise ValueError("content or messages is required")
    role = str((metadata or {}).get("role") or "user")
    return [{"role": role, "content": str(content)}]


def _normalize_message(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        return {"role": "user", "content": str(message)}
    return {"role": str(message.get("role") or "user"), "content": str(message.get("content", ""))}


def _normalize_context(raw: Any, *, limit: int) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        items = raw.get("contexts", raw.get("results", raw.get("memories", raw)))
    else:
        items = raw
    if isinstance(items, list):
        values = items[:limit]
    else:
        values = [items]
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(values, start=1):
        if isinstance(item, dict):
            content = item.get("content") or item.get("memory") or item.get("text") or item
            score = item.get("score")
            item_id = item.get("id", idx)
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        else:
            content = item
            score = None
            item_id = idx
            metadata = {}
        rows.append({"id": str(item_id), "content": str(content), "score": score, "metadata": metadata})
    return rows


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(message.get("content", "")) for message in messages)


def _user_id_from_object(user: Any) -> str:
    for attr in ("id", "user_id", "uid"):
        value = getattr(user, attr, None)
        if value:
            return str(value)
    return "memobase-user"
