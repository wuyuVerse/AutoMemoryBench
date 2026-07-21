"""Local in-process LightRAG (HKUDS) memory client factory for AutoMemoryBench.

LightRAG (`lightrag-hku`, github.com/HKUDS/LightRAG) is a graph-RAG engine: on add it
runs LLM entity/relationship extraction into a local knowledge graph + vector index
(nano-vectordb + networkx, file-backed, NO daemon), and on search does hybrid graph +
vector retrieval. We drive it in-process, matching the other library-style adapters.

Both the LLM and the embedder are OpenAI-compatible via explicit base_url/api_key, so the
chat model varies per matrix model (×8) and the embedder stays fixed on bge-m3 (1024-dim).
Each case gets an isolated `working_dir` (fresh KG/vector index → no cross-case leakage).

NOTE: requires `.venv-lightrag` (`pip install lightrag-hku` with `--constraint /dev/null`
to bypass the global pytz pin). LightRAG's bundled `openai_embed` hardcodes a 1536-dim
guard that rejects bge-m3's 1024 dims, so we embed via a raw AsyncOpenAI client wrapped in
`EmbeddingFunc(embedding_dim=1024)` instead.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

from amb.clients.core.common import ensure_site_packages_from_venv

DEFAULT_LIGHTRAG_VENV = Path(__file__).resolve().parents[4] / ".venv-lightrag"


def _run(coro):
    """Run an async coroutine to completion from sync adapter methods."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio  # type: ignore
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
    except RuntimeError:
        pass
    return asyncio.run(coro)


class LightRAGClient:
    """Thin adapter driving LightRAG in-process for AMB reset/add/search."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None,
        api_key: str,
        embedding_model: str,
        embedding_base_url: str | None,
        embedding_api_key: str,
        embedding_dims: int = 1024,
        default_limit: int = 5,
    ) -> None:
        self._model = model
        self._base_url = base_url or "https://api.siliconflow.cn/v1"
        self._api_key = api_key
        self._embed_model = embedding_model
        self._embed_base = embedding_base_url or "https://api.siliconflow.cn/v1"
        self._embed_key = embedding_api_key
        self._embed_dims = int(embedding_dims)
        self.default_limit = default_limit
        self._rag = None
        self._wd: str | None = None
        self.case_id: str | None = None

    def _build(self, workspace: str):
        import numpy as np  # noqa: F401
        from lightrag import LightRAG  # type: ignore
        from lightrag.llm.openai import openai_complete_if_cache  # type: ignore
        from lightrag.utils import EmbeddingFunc  # type: ignore
        from lightrag.kg.shared_storage import initialize_pipeline_status  # type: ignore
        from openai import AsyncOpenAI  # type: ignore

        model = self._model
        base_url = self._base_url
        api_key = self._api_key
        embed_model = self._embed_model
        embed_base = self._embed_base
        embed_key = self._embed_key

        async def llm_func(prompt, system_prompt=None, history_messages=None, **kw):
            history_messages = history_messages or []
            # drop LightRAG-internal kwargs the OpenAI wrapper doesn't accept
            clean = {k: v for k, v in kw.items() if k not in ("hashing_kv", "keyword_extraction")}
            return await openai_complete_if_cache(
                model, prompt, system_prompt=system_prompt,
                history_messages=history_messages, base_url=base_url, api_key=api_key, **clean,
            )

        async def embed_func(texts):
            import numpy as np
            cli = AsyncOpenAI(base_url=embed_base, api_key=embed_key)
            resp = await cli.embeddings.create(model=embed_model, input=list(texts))
            return np.array([d.embedding for d in resp.data])

        rag = LightRAG(
            working_dir=workspace,
            llm_model_func=llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=self._embed_dims, max_token_size=8192, func=embed_func,
            ),
        )
        _run(rag.initialize_storages())
        _run(initialize_pipeline_status())
        return rag

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        # fresh KG/vector workspace per case → no cross-case leakage, no shared DB
        self._wd = tempfile.mkdtemp(prefix="lightrag_ws_")
        self._rag = self._build(self._wd)
        return {"ok": True, "workspace": self._wd}

    def _ensure(self):
        if self._rag is None:
            self.reset()
        return self._rag

    def add(self, content: Any = None, *, messages: Any = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        rag = self._ensure()
        text = _text_payload(content=content, messages=messages)
        if not text.strip():
            return {"ok": True, "skipped": "empty"}
        try:
            _run(rag.ainsert(text))
        except Exception:
            pass
        return {"ok": True}

    def search(self, query: str | None = None, *, user_id: str | None = None, limit: int | None = None,
               top_k: int | None = None, **_: Any) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required for LightRAG search")
        rag = self._ensure()
        k = int(limit or top_k or self.default_limit)
        try:
            from lightrag import QueryParam  # type: ignore
            resp = _run(rag.aquery(str(query), param=QueryParam(mode="hybrid", top_k=k)))
        except Exception:
            return []
        text = str(resp) if resp is not None else ""
        if not text.strip():
            return []
        return [{"id": "1", "content": text, "score": None, "metadata": {}}]

    def get_all(self, *, user_id: str | None = None, **_: Any) -> list[dict[str, Any]]:
        return []

    def delete(self, memory_id: str | None = None, **_: Any) -> dict[str, Any]:
        return {"ok": True, "note": "lightrag workspace delete not used in AMB flow"}


def create_client(
    *,
    venv_root: str | None = None,
    model: str = "deepseek-ai/DeepSeek-V4-Flash",
    base_url: str | None = "https://api.siliconflow.cn/v1",
    api_key_env: str = "OPENAI_API_KEY",
    embedding_model: str = "BAAI/bge-m3",
    embedding_base_url: str | None = "https://api.siliconflow.cn/v1",
    embedding_api_key_env: str = "OPENAI_API_KEY",
    embedding_dims: int = 1024,
    default_limit: int = 5,
    **_: Any,
) -> LightRAGClient:
    vroot = Path(venv_root) if venv_root else DEFAULT_LIGHTRAG_VENV
    ensure_site_packages_from_venv(vroot)
    api_key = os.environ.get(api_key_env, "") or os.environ.get("OPENAI_API_KEY", "")
    embed_key = os.environ.get(embedding_api_key_env, "") or api_key
    return LightRAGClient(
        model=model, base_url=base_url, api_key=api_key,
        embedding_model=embedding_model, embedding_base_url=embedding_base_url, embedding_api_key=embed_key,
        embedding_dims=embedding_dims, default_limit=default_limit,
    )


def _text_payload(*, content: Any, messages: Any) -> str:
    if messages is not None:
        vals = messages if isinstance(messages, list) else [messages]
        parts: list[str] = []
        for m in vals:
            if isinstance(m, dict):
                role = m.get("role", "user")
                text = m.get("content", "")
                if isinstance(text, list):
                    text = " ".join(str(p.get("text", p) if isinstance(p, dict) else p) for p in text)
                parts.append(f"{role}: {text}")
            else:
                parts.append(str(m))
        return "\n".join(parts)
    if content is not None:
        return str(content)
    raise ValueError("content or messages is required")
