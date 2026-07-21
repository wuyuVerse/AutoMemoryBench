"""Local Memoripy client factory for AutoMemoryBench integration.

Memoripy (pip `memoripy`) is a library-style agent-memory system with short/long-term
memory + concept graph. Its ChatModel/EmbeddingModel are pure ABCs, so we supply
OpenAI-compatible implementations: the chat LLM varies per matrix model (×8) while the
embedder stays fixed on the reference embedder (bge-m3) — clean N×N decoupling. Storage is the
in-memory InMemoryStorage (no external server/DB).

add -> MemoryManager.add_interaction(prompt, output, embedding, concepts)
search -> MemoryManager.retrieve_relevant_interactions(query)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from amb.clients.core.common import ensure_site_packages_from_venv

DEFAULT_MEMORIPY_VENV = Path(__file__).resolve().parents[4] / ".venv-memoripy"


def _openai_client(base_url: str | None, api_key: str):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI(api_key=api_key)


class MemoripyClient:
    """Thin adapter exposing AMB reset/add/search/get_all/delete over Memoripy."""

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
        from memoripy import MemoryManager, ChatModel, EmbeddingModel, InMemoryStorage  # type: ignore
        import numpy as np

        chat_client = _openai_client(base_url, api_key)
        embed_client = _openai_client(embedding_base_url, embedding_api_key)

        class _Chat(ChatModel):
            def invoke(self, messages: list) -> str:
                try:
                    r = chat_client.chat.completions.create(
                        model=model, messages=messages, temperature=0.0, max_tokens=1024
                    )
                    return r.choices[0].message.content or ""
                except Exception:
                    return ""

            def extract_concepts(self, text: str) -> list[str]:
                try:
                    r = chat_client.chat.completions.create(
                        model=model,
                        messages=[{
                            "role": "user",
                            "content": (
                                "Extract up to 8 key concepts (short noun phrases) from the text. "
                                "Return ONLY a JSON array of strings.\n\n" + str(text)
                            ),
                        }],
                        temperature=0.0,
                        max_tokens=256,
                        response_format={"type": "json_object"},
                    )
                    raw = r.choices[0].message.content or "[]"
                    data = json.loads(raw)
                    if isinstance(data, dict):
                        for v in data.values():
                            if isinstance(v, list):
                                return [str(x) for x in v][:8]
                        return []
                    return [str(x) for x in data][:8] if isinstance(data, list) else []
                except Exception:
                    return []

        class _Embed(EmbeddingModel):
            def __init__(self) -> None:
                self.dimension = embedding_dims

            def get_embedding(self, text: str):
                try:
                    r = embed_client.embeddings.create(model=embedding_model, input=str(text))
                    return np.array(r.data[0].embedding, dtype=float)
                except Exception:
                    return np.zeros(embedding_dims, dtype=float)

            def initialize_embedding_dimension(self):
                return embedding_dims

        self._np = np
        self._mm = MemoryManager(chat_model=_Chat(), embedding_model=_Embed(), storage=InMemoryStorage())
        self.default_limit = default_limit
        self.case_id: str | None = None

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        from memoripy import InMemoryStorage  # type: ignore
        # fresh storage per case
        self._mm.storage = InMemoryStorage()
        try:
            self._mm.initialize_memory()
        except Exception:
            pass
        return {"ok": True}

    def add(self, content: Any = None, *, messages: Any = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        text = _text_from_payload(content=content, messages=messages)
        emb = self._mm.get_embedding(text)
        try:
            concepts = self._mm.chat_model.extract_concepts(text)
        except Exception:
            concepts = []
        self._mm.add_interaction(prompt=text, output="", embedding=emb, concepts=concepts)
        return {"ok": True}

    def search(self, query: str | None = None, *, user_id: str | None = None, limit: int | None = None,
               top_k: int | None = None, **_: Any) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required for Memoripy search")
        k = int(limit or top_k or self.default_limit)
        try:
            hits = self._mm.retrieve_relevant_interactions(str(query), similarity_threshold=0, exclude_last_n=0)
        except Exception:
            hits = []
        rows: list[dict[str, Any]] = []
        for idx, h in enumerate(hits[:k], start=1):
            if isinstance(h, dict):
                content = h.get("prompt") or h.get("output") or h.get("content") or str(h)
                cid = h.get("id", idx)
            else:
                content = str(h)
                cid = idx
            rows.append({"id": str(cid), "content": str(content), "score": None, "metadata": {}})
        return rows

    def get_all(self, *, user_id: str | None = None, **_: Any) -> list[dict[str, Any]]:
        try:
            hist = self._mm.storage.load_history()
            items = hist[0] if isinstance(hist, (list, tuple)) and hist else []
            return [{"id": str(i), "content": str(x.get("prompt", x) if isinstance(x, dict) else x), "metadata": {}}
                    for i, x in enumerate(items)]
        except Exception:
            return []

    def delete(self, memory_id: str | None = None, **_: Any) -> dict[str, Any]:
        return {"ok": True, "note": "memoripy per-memory delete not used in AMB flow"}


def create_client(
    *,
    venv_root: str | None = None,
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    embedding_model: str = "BAAI/bge-m3",
    embedding_base_url: str | None = "https://api.siliconflow.cn/v1",
    embedding_api_key_env: str = "OPENAI_API_KEY",
    embedding_dims: int = 1024,
    default_limit: int = 5,
    **_: Any,
) -> MemoripyClient:
    ensure_site_packages_from_venv(Path(venv_root) if venv_root else DEFAULT_MEMORIPY_VENV)
    api_key = os.environ.get(api_key_env, "") or os.environ.get("OPENAI_API_KEY", "")
    embed_key = os.environ.get(embedding_api_key_env, "") or api_key
    return MemoripyClient(
        model=model, base_url=base_url, api_key=api_key,
        embedding_model=embedding_model, embedding_base_url=embedding_base_url, embedding_api_key=embed_key,
        embedding_dims=embedding_dims, default_limit=default_limit,
    )


def _text_from_payload(*, content: Any, messages: Any) -> str:
    if messages is not None:
        vals = messages if isinstance(messages, list) else [messages]
        out = []
        for m in vals:
            out.append(str(m.get("content", "")) if isinstance(m, dict) else str(m))
        return "\n".join(out)
    if content is None:
        raise ValueError("content or messages is required")
    return str(content)
