"""Local Evermind client factory for AutoMemoryBench integration.

Evermind (evermind.ai, pip `evermind`) is a 2026 langchain-based agent-memory
system (authors report 93% LoCoMo / 83% LongMemEval-S). Its MemoryManager takes
langchain LLM + Embeddings + VectorStore, so we wire ChatOpenAI(matrix model) +
OpenAIEmbeddings(bge-m3) + an in-memory vector store — clean ×8
(LLM varies, embedder fixed). ingest/query back AMB's add/search.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from amb.clients.core.common import ensure_site_packages_from_venv

DEFAULT_EVERMIND_VENV = Path(__file__).resolve().parents[4] / ".venv-ever"


class EvermindOfficialSourceClient:
    """Adapter over Evermind MemoryManager (ingest/query) on an in-memory vector store."""

    def __init__(self, *, build_kwargs: dict[str, Any], default_limit: int = 5) -> None:
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        self._bk = build_kwargs
        self.default_limit = default_limit
        self.case_id: str | None = None
        self._mgr: Any = None

    def _new_mgr(self) -> Any:
        from evermind import MemoryManager, MnemonConfig  # type: ignore
        from langchain_core.vectorstores import InMemoryVectorStore
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        bk = self._bk
        chat_retries = _int_env("AMB_EVERMIND_CHAT_MAX_RETRIES", 6)
        embedding_retries = _int_env("AMB_EVERMIND_EMBEDDING_MAX_RETRIES", 10)
        retry_min_seconds = _int_env("AMB_EVERMIND_EMBEDDING_RETRY_MIN_SECONDS", 8)
        retry_max_seconds = _int_env("AMB_EVERMIND_EMBEDDING_RETRY_MAX_SECONDS", 60)
        timeout_seconds = _float_env("AMB_EVERMIND_OPENAI_TIMEOUT_SECONDS", 60.0)
        llm = ChatOpenAI(model=bk["model"], base_url=bk["base_url"], api_key=bk["api_key"],
                         temperature=0.0, max_retries=chat_retries, timeout=timeout_seconds)
        emb = OpenAIEmbeddings(model=bk["embedding_model"], base_url=bk["embedding_base_url"],
                               api_key=bk["embedding_api_key"], check_embedding_ctx_length=False,
                               max_retries=embedding_retries, timeout=timeout_seconds,
                               retry_min_seconds=retry_min_seconds, retry_max_seconds=retry_max_seconds)
        emb = _RetryingEmbeddings(
            emb,
            max_attempts=_int_env("AMB_EVERMIND_EMBEDDING_TRANSPORT_ATTEMPTS", 12),
            sleep_seconds=_float_env("AMB_EVERMIND_EMBEDDING_TRANSPORT_SLEEP_SECONDS", 30.0),
        )
        vs = InMemoryVectorStore(emb)
        # enable_semantic_memory requires a GraphStore (Neo4j etc.); run vector-only
        # (episodic) memory so the system needs no external graph DB on the cluster.
        cfg = MnemonConfig(enable_semantic_memory=False)
        return MemoryManager(config=cfg, vector_store=vs, llm=llm, embedding_model=emb)

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        self._mgr = self._new_mgr()
        return {"ok": True}

    def add(self, content: Any = None, *, messages: Any = None, user_id: str | None = None,
            metadata: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        if self._mgr is None:
            self._mgr = self._new_mgr()
        text = _content_from_payload(content=content, messages=messages)
        try:
            mid = self._mgr.ingest(text, metadata=metadata or {})
        except Exception:
            mid = None
        return {"id": mid, "user_id": user_id or self.case_id}

    def search(self, query: str | None = None, *, limit: int | None = None,
               top_k: int | None = None, **_: Any) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required for Evermind search")
        if self._mgr is None:
            return []
        k = int(limit or top_k or self.default_limit)
        # synthesize_answer=False → return retrieved memories (not an LLM-written answer)
        res = self._mgr.query(str(query), synthesize_answer=False)
        mems = (getattr(res, "retrieved_memories", None) or getattr(res, "memories", None)
                or getattr(res, "results", None) or [])
        out: list[dict[str, Any]] = []
        for m in list(mems)[:k]:
            text = (getattr(m, "content", None) or getattr(m, "text", None)
                    or (m.get("content") if isinstance(m, dict) else None) or str(m))
            out.append({"id": getattr(m, "id", f"evermind-{len(out)+1}"), "content": str(text), "metadata": {}})
        return out

    def get_all(self, **_: Any) -> list[dict[str, Any]]:
        return []

    def delete(self, memory_id: str | None = None, **_: Any) -> dict[str, Any]:
        return {"deleted": False, "memory_id": memory_id}


def create_client(*, default_limit: int = 5, model: str | None = None, base_url: str | None = None,
                  api_key_env: str | None = None, api_key: str | None = None,
                  embedding_model: str | None = None, embedding_base_url: str | None = None,
                  embedding_api_key_env: str | None = None, **_: Any) -> EvermindOfficialSourceClient:
    """Create an Evermind client: chat LLM = matrix model, embedder = the reference embedder (bge-m3)."""
    ensure_site_packages_from_venv(DEFAULT_EVERMIND_VENV)
    key = api_key or (os.environ.get(api_key_env or "OPENAI_API_KEY") or "")
    emb_key = os.environ.get(embedding_api_key_env or "OPENAI_API_KEY") or key
    bk = {
        "model": model or "gpt-4.1-mini", "base_url": base_url, "api_key": key,
        "embedding_model": embedding_model or "BAAI/bge-m3",
        "embedding_base_url": embedding_base_url or "https://api.siliconflow.cn/v1",
        "embedding_api_key": emb_key,
    }
    return EvermindOfficialSourceClient(build_kwargs=bk, default_limit=default_limit)


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


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


class _RetryingEmbeddings:
    """Transport retry shim around the official OpenAIEmbeddings backend."""

    def __init__(self, backend: Any, *, max_attempts: int, sleep_seconds: float) -> None:
        self.backend = backend
        self.max_attempts = max(1, max_attempts)
        self.sleep_seconds = max(0.0, sleep_seconds)

    def embed_query(self, text: str) -> list[float]:
        return self._call(self.backend.embed_query, text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._call(self.backend.embed_documents, texts)

    def _call(self, fn: Any, *args: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn(*args)
            except Exception as exc:  # noqa: BLE001 - provider SDK raises several 429 classes.
                if not _is_rate_limit_error(exc) or attempt >= self.max_attempts:
                    raise
                last_exc = exc
                time.sleep(self.sleep_seconds)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("embedding retry wrapper exhausted without an exception")


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return status_code == 429 or "rate limit" in text or "too many requests" in text or "rpm limit" in text
