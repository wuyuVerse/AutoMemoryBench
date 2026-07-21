"""Official managed-service REST API client factory for Supermemory AutoMemoryBench runs."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from amb.clients.core.common import effective_api_key


class SupermemoryOfficialClient:
    """Thin adapter over Supermemory's documented REST memory API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.supermemory.ai",
        default_limit: int = 5,
        container_tags: list[str] | None = None,
        search_api_version: str = "v3",
        ingest_settle_seconds: float = 15.0,
    ) -> None:
        if not api_key:
            raise RuntimeError("SUPERMEMORY_API_KEY is required for Supermemory official API runs")
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        if search_api_version not in {"v3", "v4"}:
            raise ValueError("search_api_version must be 'v3' or 'v4'")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_limit = default_limit
        self.container_tags = list(container_tags or [])
        self.search_api_version = search_api_version
        # Supermemory ingests asynchronously (~10s to index); settle once before
        # the first search of a case so add-then-search recall isn't spuriously 0.
        self.ingest_settle_seconds = max(0.0, ingest_settle_seconds)
        self._last_add_ts: float | None = None
        self._settled = False
        self.case_id: str | None = None
        self._written: list[dict[str, Any]] = []

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        self._written = []
        self._last_add_ts = None
        self._settled = False
        return {"ok": True, "container_tags": self._container_tags(user_id=user_id)}

    def add(
        self,
        content: Any = None,
        *,
        messages: Any = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        text = _text_from_payload(content=content, messages=messages)
        body: dict[str, Any] = {"content": text}
        tags = self._container_tags(user_id=user_id)
        if tags:
            # /v3/documents ingest takes a single containerTag (camelCase wire key)
            body["containerTag"] = tags[0]
        if metadata:
            body["metadata"] = metadata
        response = self._request("POST", "/v3/documents", body)
        self._last_add_ts = time.monotonic()
        self._settled = False
        self._written.append({"content": text, "metadata": metadata or {}, "response": response, "containerTags": tags})
        return response if isinstance(response, dict) else {"result": response}

    def search(
        self,
        query: str | None = None,
        *,
        user_id: str | None = None,
        limit: int | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required for Supermemory search")
        # Wait once for async ingestion to settle before the first search of a case.
        if not self._settled and self._last_add_ts is not None and self.ingest_settle_seconds > 0:
            elapsed = time.monotonic() - self._last_add_ts
            if elapsed < self.ingest_settle_seconds:
                time.sleep(self.ingest_settle_seconds - elapsed)
            self._settled = True
        k = int(limit or top_k or self.default_limit)
        body: dict[str, Any] = {"q": str(query), "limit": k}
        tags = self._container_tags(user_id=user_id)
        if tags:
            if self.search_api_version == "v4":
                body["containerTag"] = tags[0]
            else:
                body["containerTags"] = tags
        if threshold is not None:
            body["threshold"] = threshold
        raw = self._request("POST", f"/{self.search_api_version}/search", body)
        return _normalize_results(raw)

    def get_all(self, **_: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": str(row.get("response", {}).get("id", f"supermemory-observation-{idx}")),
                "content": row["content"],
                "metadata": {**dict(row.get("metadata") or {}), "containerTags": row.get("containerTags", [])},
            }
            for idx, row in enumerate(self._written, start=1)
        ]

    def delete(self, memory_id: str | None = None, **_: Any) -> dict[str, Any]:
        if not memory_id:
            raise ValueError("memory_id is required for Supermemory delete")
        raw = self._request("DELETE", f"/v3/documents/{memory_id}", None)
        return raw if isinstance(raw, dict) else {"deleted": True, "result": raw}

    def _container_tags(self, *, user_id: str | None = None) -> list[str]:
        tags = list(self.container_tags)
        resolved_user = user_id or self.case_id
        if resolved_user:
            tags.append(str(resolved_user))
        return tags

    def _request(self, method: str, path: str, body: dict[str, Any] | None) -> Any:
        data = None if body is None else json.dumps(body).encode("utf-8")
        # Honor free-tier rate limits: retry on HTTP 429 with the server-advised
        # backoff (retryAfterSeconds), plus a couple of transient-5xx retries.
        max_attempts = 6
        for attempt in range(max_attempts):
            request = urllib.request.Request(
                f"{self.base_url}{path}",
                data=data,
                method=method,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read().decode("utf-8")
                return json.loads(payload) if payload else {}
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if retryable and attempt < max_attempts - 1:
                    wait = 60.0
                    try:
                        wait = float(json.loads(detail).get("retryAfterSeconds", 0)) or wait
                    except Exception:
                        wait = float(exc.headers.get("Retry-After", 0) or 0) or min(60.0, 2.0 * (attempt + 1))
                    time.sleep(min(wait, 90.0))
                    continue
                raise RuntimeError(f"Supermemory API {method} {path} failed with HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                if attempt < max_attempts - 1:
                    time.sleep(min(30.0, 2.0 * (attempt + 1)))
                    continue
                raise RuntimeError(f"Supermemory API {method} {path} network error: {exc}") from exc
        return {}


def create_client(
    *,
    api_key: str | None = None,
    api_key_env: str = "SUPERMEMORY_API_KEY",
    base_url: str = "https://api.supermemory.ai",
    default_limit: int = 5,
    container_tags: list[str] | None = None,
    search_api_version: str = "v3",
    ingest_settle_seconds: float = 15.0,
) -> SupermemoryOfficialClient:
    """Create a Supermemory client using its official documented REST API."""

    resolved_key = effective_api_key(api_key=api_key, api_key_env=api_key_env)
    if not resolved_key:
        raise RuntimeError(f"{api_key_env} is required for Supermemory official API runs")
    return SupermemoryOfficialClient(
        api_key=resolved_key,
        base_url=base_url,
        default_limit=default_limit,
        container_tags=container_tags,
        search_api_version=search_api_version,
        ingest_settle_seconds=ingest_settle_seconds,
    )


def _text_from_payload(*, content: Any, messages: Any) -> str:
    if messages is not None:
        values = messages if isinstance(messages, list) else [messages]
        chunks = []
        for message in values:
            if isinstance(message, dict):
                chunks.append(str(message.get("content", "")))
            else:
                chunks.append(str(message))
        return "\n".join(chunks)
    if content is None:
        raise ValueError("content or messages is required")
    return str(content)


def _normalize_results(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        items = raw.get("results", raw.get("memories", []))
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    if isinstance(items, dict):
        items = list(items.values())
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(items if isinstance(items, list) else [items], start=1):
        if isinstance(item, dict):
            content = item.get("memory") or item.get("content") or item.get("chunk") or item.get("title") or item
            score = item.get("similarity", item.get("score"))
            item_id = item.get("id", item.get("documentId", idx))
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        else:
            content = item
            score = None
            item_id = idx
            metadata = {}
        rows.append({"id": str(item_id), "content": str(content), "score": score, "metadata": metadata})
    return rows
