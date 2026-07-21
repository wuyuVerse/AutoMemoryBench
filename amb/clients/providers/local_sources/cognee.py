"""Local official-source Cognee client factory for AutoMemoryBench integration runs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from amb.clients.core.common import ensure_site_packages_from_venv, ensure_source_path

# Cognee's heavy deps (sqlalchemy, cognee, ...) are installed in a dedicated venv,
# not the default interpreter; add its site-packages before importing cognee.
DEFAULT_COGNEE_VENV = Path(__file__).resolve().parents[4] / ".venv-cognee"


class CogneeOfficialSourceClient:
    """Thin adapter over Cognee's official ``add -> cognify -> search`` API."""

    def __init__(
        self,
        cognee_module: Any,
        *,
        dataset_prefix: str = "amst_cognee",
        chunk_size: int | None = None,
        default_limit: int = 5,
        run_cognify_on_add: bool = True,
        search_type: Any = None,
    ) -> None:
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        self.cognee = cognee_module
        self.dataset_prefix = dataset_prefix
        self.chunk_size = chunk_size
        self.default_limit = default_limit
        self.run_cognify_on_add = run_cognify_on_add
        self.search_type = search_type
        self.case_id: str | None = None
        self._written: list[dict[str, Any]] = []

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        self._written = []
        return {"ok": True, "dataset_name": self._dataset_name(user_id=user_id)}

    def add(
        self,
        content: Any = None,
        *,
        messages: Any = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        text = _content_from_payload(content=content, messages=messages)
        dataset_name = self._dataset_name(user_id=user_id, metadata=metadata)
        add_result = _run_async(self.cognee.add(text, dataset_name=dataset_name))
        cognify_result = None
        if self.run_cognify_on_add:
            kwargs: dict[str, Any] = {"datasets": [dataset_name]}
            if self.chunk_size is not None:
                kwargs["chunk_size"] = self.chunk_size
            cognify_result = _run_async(self.cognee.cognify(**kwargs))
        self._written.append({"content": text, "dataset_name": dataset_name, "metadata": metadata or {}})
        return {
            "result": add_result,
            "cognify_result": cognify_result,
            "dataset_name": dataset_name,
            "user_id": user_id or self.case_id,
        }

    def search(
        self,
        query: str | None = None,
        *,
        limit: int | None = None,
        top_k: int | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required for Cognee search")
        dataset_name = self._dataset_name(user_id=user_id, metadata=metadata)
        k = int(limit or top_k or self.default_limit)
        kwargs: dict[str, Any] = {"query_text": str(query), "top_k": k, "datasets": [dataset_name]}
        if self.search_type is not None:
            kwargs["query_type"] = self.search_type
        raw = _run_async(self.cognee.search(**kwargs))
        return _normalize_search_results(raw)

    def get_all(self, **_: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": f"cognee-observation-{idx}",
                "content": row["content"],
                "metadata": {"dataset_name": row["dataset_name"], **dict(row.get("metadata") or {})},
            }
            for idx, row in enumerate(self._written, start=1)
        ]

    def delete(self, memory_id: str | None = None, **_: Any) -> dict[str, Any]:
        return {
            "deleted": False,
            "reason": "official Cognee AMST path does not expose stable per-memory delete",
            "memory_id": memory_id,
        }

    def _dataset_name(self, *, user_id: str | None = None, metadata: dict[str, Any] | None = None) -> str:
        suffix = user_id or self.case_id or (metadata or {}).get("case_id") or "default"
        safe_suffix = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(suffix))
        return f"{self.dataset_prefix}_{safe_suffix}"


def create_client(
    *,
    source_root: str = "related_work/repos/MemoryAgentBench",
    dataset_prefix: str = "amst_cognee",
    chunk_size: int | None = 1024,
    default_limit: int = 5,
    run_cognify_on_add: bool = True,
    search_type: str | None = None,
    venv_root: str | None = None,
    **_: Any,
) -> CogneeOfficialSourceClient:
    """Create a client from the local Cognee source used by MemoryAgentBench."""

    ensure_site_packages_from_venv(Path(venv_root) if venv_root else DEFAULT_COGNEE_VENV)
    ensure_source_path(source_root)
    import cognee  # type: ignore

    resolved_search_type = None
    if search_type:
        resolved_search_type = getattr(cognee.SearchType, search_type)
    return CogneeOfficialSourceClient(
        cognee,
        dataset_prefix=dataset_prefix,
        chunk_size=chunk_size,
        default_limit=default_limit,
        run_cognify_on_add=run_cognify_on_add,
        search_type=resolved_search_type,
    )


def _run_async(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError("Cognee official client cannot run blocking AMST wrapper inside an active event loop") from None


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


def _normalize_search_results(raw: Any) -> list[dict[str, Any]]:
    items = raw if isinstance(raw, list) else [raw]
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        if isinstance(item, dict):
            content = item.get("content") or item.get("text") or item.get("result") or item
            score = item.get("score")
            item_id = item.get("id", idx)
        else:
            content = getattr(item, "content", getattr(item, "text", item))
            score = getattr(item, "score", None)
            item_id = getattr(item, "id", idx)
        rows.append({"id": str(item_id), "content": str(content), "score": score})
    return rows
