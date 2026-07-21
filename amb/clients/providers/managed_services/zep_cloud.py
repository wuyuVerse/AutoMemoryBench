"""Official Zep Cloud (app.getzep.com) graph-memory client factory for AutoMemoryBench runs.

Uses the official ``zep-cloud`` SDK: per-case Zep user + ``graph.add`` (text ingest) +
``graph.search`` (fact/edge retrieval). Zep ingests asynchronously (~10-15s to build the
graph), so we settle once before the first search of a case. Credentials live only in
env (ZEP_API_KEY); never in the committed config.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from amb.clients.core.common import effective_api_key, ensure_site_packages_from_venv


def _retry(fn, *, attempts: int = 6):
    """Call fn() with backoff on transient/rate-limit (429) errors common to free tiers."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - SDK raises varied error types
            last = exc
            msg = str(exc).lower()
            status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            rate = status == 429 or "429" in msg or "rate limit" in msg or "too many requests" in msg
            transient = rate or "timeout" in msg or "temporarily" in msg or (isinstance(status, int) and 500 <= status < 600)
            if not transient or i == attempts - 1:
                raise
            time.sleep(min(60.0, (15.0 if rate else 2.0) * (i + 1)))
    raise last  # pragma: no cover


class ZepCloudClient:
    """Thin adapter over the official Zep Cloud graph memory API."""

    def __init__(
        self,
        *,
        api_key: str,
        default_limit: int = 5,
        ingest_settle_seconds: float = 16.0,
        scope: str = "edges",
        reranker: str | None = None,
    ) -> None:
        if not api_key:
            raise RuntimeError("ZEP_API_KEY is required for Zep Cloud runs")
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        from zep_cloud.client import Zep  # type: ignore

        self._zep = Zep(api_key=api_key)
        self.default_limit = default_limit
        self.ingest_settle_seconds = max(0.0, ingest_settle_seconds)
        self.scope = scope
        self.reranker = reranker
        self.user_id: str | None = None
        self._last_add_ts: float | None = None
        self._settled = False

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        base = str(case_id or user_id or "amb")
        # Fresh graph per case: unique user id avoids stale cross-case memory.
        self.user_id = f"{base}_{uuid.uuid4().hex[:10]}"
        try:
            _retry(lambda: self._zep.user.add(user_id=self.user_id))
        except Exception:
            pass
        self._last_add_ts = None
        self._settled = False
        return {"ok": True, "user_id": self.user_id}

    def add(self, content: Any = None, *, messages: Any = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        if not self.user_id:
            self.reset(user_id=user_id)
        text = _text_from_payload(content=content, messages=messages)
        _retry(lambda: self._zep.graph.add(user_id=self.user_id, type="text", data=text))
        self._last_add_ts = time.monotonic()
        self._settled = False
        return {"ok": True, "user_id": self.user_id}

    def search(
        self,
        query: str | None = None,
        *,
        user_id: str | None = None,
        limit: int | None = None,
        top_k: int | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required for Zep Cloud search")
        if not self.user_id:
            raise RuntimeError("Zep Cloud search requires reset()/add() to establish a user graph first")
        if not self._settled and self._last_add_ts is not None and self.ingest_settle_seconds > 0:
            elapsed = time.monotonic() - self._last_add_ts
            if elapsed < self.ingest_settle_seconds:
                time.sleep(self.ingest_settle_seconds - elapsed)
            self._settled = True
        k = int(limit or top_k or self.default_limit)
        kwargs: dict[str, Any] = {"user_id": self.user_id, "query": str(query), "scope": self.scope, "limit": k}
        if self.reranker:
            kwargs["reranker"] = self.reranker
        result = _retry(lambda: self._zep.graph.search(**kwargs))
        return _normalize_results(result, scope=self.scope)

    def get_all(self, *, user_id: str | None = None, **_: Any) -> list[dict[str, Any]]:
        if not self.user_id:
            return []
        try:
            edges = self._zep.graph.search(user_id=self.user_id, query="*", scope="edges", limit=50)
            return _normalize_results(edges, scope="edges")
        except Exception:
            return []

    def delete(self, memory_id: str | None = None, **_: Any) -> dict[str, Any]:
        # Zep graph deletion is per-user; per-fact delete isn't part of the AMB flow.
        return {"ok": True, "note": "zep graph delete is per-user; no-op for per-memory delete"}


def create_client(
    *,
    api_key: str | None = None,
    api_key_env: str = "ZEP_API_KEY",
    venv_root: str | None = ".venv-zep",
    default_limit: int = 5,
    ingest_settle_seconds: float = 16.0,
    scope: str = "edges",
    reranker: str | None = None,
    **_: Any,
) -> ZepCloudClient:
    """Create a Zep Cloud client using the official zep-cloud SDK."""

    if venv_root:
        ensure_site_packages_from_venv(Path(venv_root))
    resolved_key = effective_api_key(api_key=api_key, api_key_env=api_key_env)
    if not resolved_key:
        raise RuntimeError(f"{api_key_env} is required for Zep Cloud runs")
    return ZepCloudClient(
        api_key=resolved_key,
        default_limit=default_limit,
        ingest_settle_seconds=ingest_settle_seconds,
        scope=scope,
        reranker=reranker,
    )


def _text_from_payload(*, content: Any, messages: Any) -> str:
    if messages is not None:
        values = messages if isinstance(messages, list) else [messages]
        chunks = []
        for message in values:
            if isinstance(message, dict):
                role = message.get("role", "")
                chunks.append(f"{role}: {message.get('content', '')}" if role else str(message.get("content", "")))
            else:
                chunks.append(str(message))
        return "\n".join(chunks)
    if content is None:
        raise ValueError("content or messages is required")
    return str(content)


def _normalize_results(result: Any, *, scope: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    items = getattr(result, scope, None) or getattr(result, "edges", None) or getattr(result, "nodes", None) or []
    for idx, item in enumerate(items, start=1):
        fact = getattr(item, "fact", None) or getattr(item, "name", None) or getattr(item, "summary", None) or str(item)
        item_id = getattr(item, "uuid_", None) or getattr(item, "uuid", None) or idx
        score = getattr(item, "score", None)
        rows.append({"id": str(item_id), "content": str(fact), "score": score, "metadata": {}})
    return rows
