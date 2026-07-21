"""Local in-process txtai memory client factory for AutoMemoryBench.

txtai (`txtai`, github.com/neuml/txtai) is a mature embeddings-database / semantic search
engine. We use it as a pure vector memory: `Embeddings.index([...])` on add and
`Embeddings.search(query, k)` on retrieve. Storage is in-memory (Faiss + content store),
NO daemon, NO graph build — retrieval is sub-second, so it completes an AMB case well
within pod lifetime (unlike graph-RAG systems that rebuild communities per memory).

The embedder is OpenAI-compatible via txtai's `method="external"` transform hook: we feed
it bge-m3 through the configured embedding endpoint, so the embedding stays fixed on bge-m3 across
the ×8 model axis. txtai itself does no LLM generation — this is an embedding-recall memory
(the matrix "model" labels the column; the memory backend is model-agnostic on the embed
side, consistent with how the benchmark treats retrieval-only memories).

NOTE: requires `.venv-txtai` (`pip install txtai` with `--constraint /dev/null`; then
`pip install --target <venv-site-packages> openai` so the pod — which disables user-site —
sees it). `content=True` so search returns the stored text, not just ids.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from amb.clients.core.common import ensure_site_packages_from_venv

DEFAULT_TXTAI_VENV = Path(__file__).resolve().parents[4] / ".venv-txtai"


class TxtaiClient:
    """Thin adapter driving txtai Embeddings in-process for AMB reset/add/search."""

    def __init__(
        self,
        *,
        embedding_model: str,
        embedding_base_url: str | None,
        embedding_api_key: str,
        default_limit: int = 5,
        **_: Any,
    ) -> None:
        self._embed_model = embedding_model
        self._embed_base = embedding_base_url or "https://api.siliconflow.cn/v1"
        self._embed_key = embedding_api_key
        self.default_limit = default_limit
        self._emb = None
        self._docs: list[str] = []
        self._pending: list[str] = []
        self.case_id: str | None = None

    def _build(self):
        import numpy as np  # noqa: F401
        from txtai import Embeddings  # type: ignore
        from openai import OpenAI  # type: ignore

        embed_model = self._embed_model
        base_url = self._embed_base
        api_key = self._embed_key
        cli = OpenAI(base_url=base_url, api_key=api_key)

        def transform(data):
            import numpy as np
            resp = cli.embeddings.create(model=embed_model, input=list(data))
            return np.array([d.embedding for d in resp.data])

        return Embeddings({"method": "external", "transform": transform, "content": True})

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        self._emb = self._build()
        self._docs = []
        self._pending = []
        return {"ok": True}

    def _ensure(self):
        if self._emb is None:
            self.reset()
        return self._emb

    def add(self, content: Any = None, *, messages: Any = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self._ensure()
        text = _text_payload(content=content, messages=messages)
        if not text.strip():
            return {"ok": True, "skipped": "empty"}
        # buffer + reindex so each search sees all docs so far
        self._pending.append(text)
        return {"ok": True}

    def _flush(self):
        emb = self._ensure()
        if self._pending:
            start = len(self._docs)
            rows = [(str(start + i), t) for i, t in enumerate(self._pending)]
            self._docs.extend(self._pending)
            self._pending = []
            try:
                emb.upsert(rows)
            except Exception:
                # fallback: full reindex
                try:
                    emb.index([(str(i), t) for i, t in enumerate(self._docs)])
                except Exception:
                    pass

    def search(self, query: str | None = None, *, user_id: str | None = None, limit: int | None = None,
               top_k: int | None = None, **_: Any) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required for txtai search")
        emb = self._ensure()
        self._flush()
        k = int(limit or top_k or self.default_limit)
        try:
            res = emb.search(str(query), k)
        except Exception:
            return []
        rows: list[dict[str, Any]] = []
        for r in res or []:
            if isinstance(r, dict):
                text = r.get("text") or r.get("data") or ""
                rid = r.get("id")
                score = r.get("score")
            else:
                # tuple (id, score)
                rid = r[0] if len(r) > 0 else None
                score = r[1] if len(r) > 1 else None
                text = self._docs[int(rid)] if rid is not None and str(rid).isdigit() and int(rid) < len(self._docs) else ""
            if not str(text).strip():
                continue
            rows.append({"id": str(rid), "content": str(text), "score": score, "metadata": {}})
        return rows

    def get_all(self, *, user_id: str | None = None, **_: Any) -> list[dict[str, Any]]:
        return [{"id": str(i), "content": t, "metadata": {}} for i, t in enumerate(self._docs)]

    def delete(self, memory_id: str | None = None, **_: Any) -> dict[str, Any]:
        return {"ok": True, "note": "txtai in-memory index delete not used in AMB flow"}


def create_client(
    *,
    venv_root: str | None = None,
    model: str = "zai-org/GLM-5.2",  # accepted for matrix-symmetry; txtai does no LLM gen
    base_url: str | None = "https://api.siliconflow.cn/v1",
    api_key_env: str = "OPENAI_API_KEY",
    embedding_model: str = "BAAI/bge-m3",
    embedding_base_url: str | None = "https://api.siliconflow.cn/v1",
    embedding_api_key_env: str = "OPENAI_API_KEY",
    embedding_dims: int = 1024,
    default_limit: int = 5,
    **_: Any,
) -> TxtaiClient:
    vroot = Path(venv_root) if venv_root else DEFAULT_TXTAI_VENV
    ensure_site_packages_from_venv(vroot)
    api_key = os.environ.get(api_key_env, "") or os.environ.get("OPENAI_API_KEY", "")
    embed_key = os.environ.get(embedding_api_key_env, "") or api_key
    return TxtaiClient(
        embedding_model=embedding_model, embedding_base_url=embedding_base_url,
        embedding_api_key=embed_key, default_limit=default_limit,
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
