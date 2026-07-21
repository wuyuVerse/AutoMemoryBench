"""Local Synap client factory for AutoMemoryBench integration.

Synap (github.com/veeeceee/synap) is a 2026 agentic-memory library (SOTA on
LongMemEval per its authors) with a pluggable provider design: an LLMProvider
(generate) and EmbeddingProvider (embed/embed_batch). We supply OpenAI-compatible
providers so the chat LLM varies per matrix model (×8) while the embedder stays
fixed on the reference embedder (bge-m3) — clean N×N decoupling, no base_url conflict.

SemanticMemory.store/search backs AMB's add/search contract over an in-memory
MemoryGraph (no external server/DB).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any


def _run(coro):
    """Run a possibly-async result to completion (Synap store/search are coroutines)."""
    if asyncio.iscoroutine(coro):
        return asyncio.run(coro)
    return coro

from amb.clients.core.common import ensure_site_packages_from_venv

DEFAULT_SYNAP_VENV = Path(__file__).resolve().parents[4] / ".venv-synap"


def _openai_client(base_url: str | None, api_key: str):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI(api_key=api_key)


def _make_providers(*, model, base_url, api_key, embedding_model, embedding_base_url, embedding_api_key):
    """Build Synap LLM/Embedding providers backed by OpenAI-compatible endpoints."""
    from synap import LLMProvider, EmbeddingProvider  # type: ignore

    class _LLM(LLMProvider):
        def __init__(self):
            self._c = _openai_client(base_url, api_key); self._m = model

        async def generate(self, prompt: str, output_schema: dict[str, Any] | None = None) -> str:
            try:
                kw: dict[str, Any] = {"model": self._m,
                                      "messages": [{"role": "user", "content": str(prompt)}],
                                      "temperature": 0.0, "max_tokens": 1024}
                if output_schema:
                    kw["response_format"] = {"type": "json_object"}
                r = self._c.chat.completions.create(**kw)
                return r.choices[0].message.content or ""
            except Exception:
                return ""

    class _Embed(EmbeddingProvider):
        def __init__(self):
            self._c = _openai_client(embedding_base_url, embedding_api_key); self._m = embedding_model

        async def embed(self, text: str) -> list[float]:
            return (await self.embed_batch([text]))[0]

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            try:
                r = self._c.embeddings.create(model=self._m, input=[str(t) for t in texts])
                return [d.embedding for d in r.data]
            except Exception:
                # bge-m3 dim 1024; zero vectors keep the graph consistent on transient errors
                return [[0.0] * 1024 for _ in texts]

    return _LLM(), _Embed()


class SynapOfficialSourceClient:
    """Adapter over Synap SemanticMemory (store/search) on an in-memory graph."""

    def __init__(self, *, providers_kwargs: dict[str, Any], default_limit: int = 5) -> None:
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        self._pk = providers_kwargs
        self.default_limit = default_limit
        self.case_id: str | None = None
        self._mem: Any = None

    def _new_mem(self) -> Any:
        from synap import SemanticMemory, MemoryGraph  # type: ignore
        llm, embed = _make_providers(**self._pk)
        return SemanticMemory(MemoryGraph(), embed, llm)

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        self._mem = self._new_mem()
        return {"ok": True}

    def add(self, content: Any = None, *, messages: Any = None, user_id: str | None = None,
            metadata: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        if self._mem is None:
            self._mem = self._new_mem()
        text = _content_from_payload(content=content, messages=messages)
        try:
            mid = _run(self._mem.store(text, check_contradictions=False))
        except TypeError:
            mid = _run(self._mem.store(text))
        return {"id": mid, "user_id": user_id or self.case_id}

    def search(self, query: str | None = None, *, limit: int | None = None,
               top_k: int | None = None, **_: Any) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required for Synap search")
        if self._mem is None:
            return []
        k = int(limit or top_k or self.default_limit)
        res = _run(self._mem.search(str(query), max_nodes=k))
        nodes = getattr(res, "nodes", None) or []
        out: list[dict[str, Any]] = []
        for n in nodes[:k]:
            text = (getattr(n, "content", None) or getattr(n, "text", None)
                    or getattr(n, "value", None) or "")
            out.append({"id": getattr(n, "id", f"synap-{len(out)+1}"), "content": str(text), "metadata": {}})
        return out

    def get_all(self, **_: Any) -> list[dict[str, Any]]:
        return []

    def delete(self, memory_id: str | None = None, **_: Any) -> dict[str, Any]:
        return {"deleted": False, "memory_id": memory_id}


def create_client(*, default_limit: int = 5, model: str | None = None, base_url: str | None = None,
                  api_key_env: str | None = None, api_key: str | None = None,
                  embedding_model: str | None = None, embedding_base_url: str | None = None,
                  embedding_api_key_env: str | None = None, **_: Any) -> SynapOfficialSourceClient:
    """Create a Synap client: chat LLM = matrix model, embedder = the reference embedder (bge-m3)."""
    ensure_site_packages_from_venv(DEFAULT_SYNAP_VENV)
    key = api_key or (os.environ.get(api_key_env or "OPENAI_API_KEY") or "")
    emb_key = os.environ.get(embedding_api_key_env or "OPENAI_API_KEY") or key
    pk = {
        "model": model or "gpt-4.1-mini", "base_url": base_url, "api_key": key,
        "embedding_model": embedding_model or "BAAI/bge-m3",
        "embedding_base_url": embedding_base_url or "https://api.siliconflow.cn/v1",
        "embedding_api_key": emb_key,
    }
    return SynapOfficialSourceClient(providers_kwargs=pk, default_limit=default_limit)


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
