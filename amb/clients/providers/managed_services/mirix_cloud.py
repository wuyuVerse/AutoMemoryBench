"""Official Mirix Cloud (app.mirix.io) client factory for AutoMemoryBench runs.

Uses the official ``mirix`` SDK (MirixClient): ``initialize_meta_agent`` (LLM-backed
memory agent), ``add`` (messages → extracted memories) and ``retrieve_with_conversation``
(query → relevant memories). Mirix ingests asynchronously, so we settle once before the
first retrieve of a case. Credentials live only in env (MIRIX_API_KEY).

NOTE: as of 2026-06, the api.mirix.io TLS certificate is expired server-side; we relax
SSL verification for this client's HTTP session so runs aren't blocked by their cert bug.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from amb.clients.core.common import effective_api_key, ensure_site_packages_from_venv


def _retry(fn, *, attempts: int = 6):
    """Call fn() with backoff on transient/rate-limit (429) errors common to free tiers."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            msg = str(exc).lower()
            status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            rate = status == 429 or "429" in msg or "rate limit" in msg or "too many requests" in msg
            transient = rate or "timeout" in msg or "temporarily" in msg or (isinstance(status, int) and 500 <= status < 600)
            if not transient or i == attempts - 1:
                raise
            time.sleep(min(60.0, (15.0 if rate else 2.0) * (i + 1)))
    raise last  # pragma: no cover


def _relax_ssl() -> None:
    import ssl
    try:
        import urllib3  # type: ignore
        urllib3.disable_warnings()
    except Exception:
        pass
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore
    try:
        import requests  # type: ignore
        _orig = requests.Session.request

        def _patched(self, *a, **k):  # noqa: ANN001
            k.setdefault("verify", False)
            return _orig(self, *a, **k)

        requests.Session.request = _patched  # type: ignore
    except Exception:
        pass


class MirixCloudClient:
    """Thin adapter over the official Mirix Cloud memory API."""

    def __init__(
        self,
        *,
        api_key: str,
        provider: str = "openai",
        default_limit: int = 5,
        ingest_settle_seconds: float = 12.0,
    ) -> None:
        if not api_key:
            raise RuntimeError("MIRIX_API_KEY is required for Mirix Cloud runs")
        _relax_ssl()
        from mirix import MirixClient  # type: ignore

        self._client = MirixClient(api_key=api_key)
        _retry(lambda: self._client.initialize_meta_agent(provider=provider))
        self.default_limit = default_limit
        self.ingest_settle_seconds = max(0.0, ingest_settle_seconds)
        self.user_id: str | None = None
        self._last_add_ts: float | None = None
        self._settled = False

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.user_id = str(case_id or user_id or "amb_user")
        self._last_add_ts = None
        self._settled = False
        return {"ok": True, "user_id": self.user_id}

    def add(self, content: Any = None, *, messages: Any = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        uid = str(user_id or self.user_id or "amb_user")
        self.user_id = uid
        payload = _messages_payload(content=content, messages=messages)
        _retry(lambda: self._client.add(messages=payload, user_id=uid))
        self._last_add_ts = time.monotonic()
        self._settled = False
        return {"ok": True, "user_id": uid}

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
            raise ValueError("query is required for Mirix search")
        uid = str(user_id or self.user_id or "amb_user")
        if not self._settled and self._last_add_ts is not None and self.ingest_settle_seconds > 0:
            elapsed = time.monotonic() - self._last_add_ts
            if elapsed < self.ingest_settle_seconds:
                time.sleep(self.ingest_settle_seconds - elapsed)
            self._settled = True
        result = _retry(lambda: self._client.retrieve_with_conversation(
            messages=[{"role": "user", "content": [{"type": "text", "text": str(query)}]}],
            user_id=uid,
        ))
        return _normalize_results(result, limit=int(limit or top_k or self.default_limit))

    def get_all(self, *, user_id: str | None = None, **_: Any) -> list[dict[str, Any]]:
        return []

    def delete(self, memory_id: str | None = None, **_: Any) -> dict[str, Any]:
        return {"ok": True, "note": "mirix cloud per-memory delete not used in AMB flow"}


def create_client(
    *,
    api_key: str | None = None,
    api_key_env: str = "MIRIX_API_KEY",
    venv_root: str | None = ".venv-mirix",
    provider: str = "openai",
    default_limit: int = 5,
    ingest_settle_seconds: float = 12.0,
    **_: Any,
) -> MirixCloudClient:
    """Create a Mirix Cloud client using the official mirix SDK."""

    if venv_root:
        ensure_site_packages_from_venv(Path(venv_root))
    resolved_key = effective_api_key(api_key=api_key, api_key_env=api_key_env)
    if not resolved_key:
        raise RuntimeError(f"{api_key_env} is required for Mirix Cloud runs")
    return MirixCloudClient(
        api_key=resolved_key,
        provider=provider,
        default_limit=default_limit,
        ingest_settle_seconds=ingest_settle_seconds,
    )


def _messages_payload(*, content: Any, messages: Any) -> list[dict[str, Any]]:
    if messages is not None:
        out = []
        values = messages if isinstance(messages, list) else [messages]
        for m in values:
            if isinstance(m, dict):
                role = m.get("role", "user")
                text = m.get("content", "")
                if isinstance(text, list):
                    text = " ".join(str(p.get("text", p) if isinstance(p, dict) else p) for p in text)
                out.append({"role": role, "content": [{"type": "text", "text": str(text)}]})
            else:
                out.append({"role": "user", "content": [{"type": "text", "text": str(m)}]})
        return out
    if content is None:
        raise ValueError("content or messages is required")
    return [{"role": "user", "content": [{"type": "text", "text": str(content)}]}]


def _normalize_results(result: Any, *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mems = (result or {}).get("memories", {}) if isinstance(result, dict) else {}
    idx = 0
    for mtype, bucket in (mems.items() if isinstance(mems, dict) else []):
        if not isinstance(bucket, dict):
            continue
        for key in ("relevant", "recent"):
            for item in (bucket.get(key) or []):
                if isinstance(item, dict):
                    text = item.get("summary") or item.get("content") or item.get("text") or item.get("details") or str(item)
                    iid = item.get("id", idx)
                else:
                    text = str(item)
                    iid = idx
                idx += 1
                rows.append({"id": str(iid), "content": str(text), "score": None, "metadata": {"memory_type": mtype}})
                if len(rows) >= limit:
                    return rows
    return rows
