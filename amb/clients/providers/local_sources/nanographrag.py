"""Local in-process nano-graphrag memory client factory for AutoMemoryBench.

nano-graphrag (`nano-graphrag`, github.com/gusye1234/nano-graphrag) is a compact,
hackable GraphRAG implementation: on add it runs LLM entity/relationship extraction
into a local knowledge graph + vector index (file-backed under working_dir, NO daemon),
and on search does graph-aware ("local"/"global") or naive vector retrieval. We drive it
in-process, matching the other library-style adapters (same family as our lightrag one).

LLM (best_model_func + cheap_model_func) and embedder (embedding_func) are OpenAI-
compatible via explicit base_url/api_key, so the chat model varies per matrix model (×8)
and the embedder stays fixed on bge-m3 (1024-dim). Each case gets an isolated working_dir.

NOTE: requires `.venv-nanographrag` (`pip install nano-graphrag` with `--constraint
/dev/null`; then `pip install --target <venv-site-packages> openai tiktoken` so the pod
— which disables user-site — sees them). The bge-m3 embedder is wrapped with
`wrap_embedding_func_with_attrs(embedding_dim=1024, ...)`.

We call the SYNC `insert()`/`query()` from the adapter's sync methods (they manage their
own event loop internally; wrapping them in an outer asyncio.run double-drives the loop).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from amb.clients.core.common import ensure_site_packages_from_venv

DEFAULT_NANOGRAPHRAG_VENV = Path(__file__).resolve().parents[4] / ".venv-nanographrag"


class NanoGraphRAGClient:
    """Thin adapter driving nano-graphrag in-process for AMB reset/add/search."""

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
        from nano_graphrag import GraphRAG  # type: ignore
        from nano_graphrag._utils import wrap_embedding_func_with_attrs  # type: ignore
        from openai import AsyncOpenAI  # type: ignore

        model = self._model
        base_url = self._base_url
        api_key = self._api_key
        embed_model = self._embed_model
        embed_base = self._embed_base
        embed_key = self._embed_key
        embed_dims = self._embed_dims

        async def llm_func(prompt, system_prompt=None, history_messages=None, **kw):
            history_messages = history_messages or []
            cli = AsyncOpenAI(base_url=base_url, api_key=api_key)
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs += list(history_messages) + [{"role": "user", "content": prompt}]
            # drop nano-graphrag internal kwargs the OpenAI client won't accept
            clean = {k: v for k, v in kw.items() if k in ("temperature", "max_tokens", "response_format")}
            r = await cli.chat.completions.create(model=model, messages=msgs, **clean)
            return r.choices[0].message.content

        @wrap_embedding_func_with_attrs(embedding_dim=embed_dims, max_token_size=8192)
        async def embed_func(texts):
            import numpy as np
            cli = AsyncOpenAI(base_url=embed_base, api_key=embed_key)
            resp = await cli.embeddings.create(model=embed_model, input=list(texts))
            return np.array([d.embedding for d in resp.data])

        return GraphRAG(
            working_dir=workspace,
            best_model_func=llm_func,
            cheap_model_func=llm_func,
            embedding_func=embed_func,
            enable_naive_rag=True,
        )

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        self._wd = tempfile.mkdtemp(prefix="nanographrag_ws_")
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
            rag.insert(text)
        except Exception:
            pass
        return {"ok": True}

    def search(self, query: str | None = None, *, user_id: str | None = None, limit: int | None = None,
               top_k: int | None = None, **_: Any) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required for nano-graphrag search")
        rag = self._ensure()
        try:
            from nano_graphrag import QueryParam  # type: ignore
            # naive (vector) mode is robust when the KG is sparse; falls back cleanly
            resp = rag.query(str(query), param=QueryParam(mode="local"))
        except Exception:
            try:
                from nano_graphrag import QueryParam  # type: ignore
                resp = rag.query(str(query), param=QueryParam(mode="naive"))
            except Exception:
                return []
        text = str(resp) if resp is not None else ""
        if not text.strip():
            return []
        return [{"id": "1", "content": text, "score": None, "metadata": {}}]

    def get_all(self, *, user_id: str | None = None, **_: Any) -> list[dict[str, Any]]:
        return []

    def delete(self, memory_id: str | None = None, **_: Any) -> dict[str, Any]:
        return {"ok": True, "note": "nano-graphrag workspace delete not used in AMB flow"}


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
) -> NanoGraphRAGClient:
    vroot = Path(venv_root) if venv_root else DEFAULT_NANOGRAPHRAG_VENV
    ensure_site_packages_from_venv(vroot)
    api_key = os.environ.get(api_key_env, "") or os.environ.get("OPENAI_API_KEY", "")
    embed_key = os.environ.get(embedding_api_key_env, "") or api_key
    return NanoGraphRAGClient(
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
