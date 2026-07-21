"""Local in-process MemEngine (RUC-NLPIR) memory client factory for AutoMemoryBench.

MemEngine (`memengine`, github.com/nuster1128/MemEngine) is a unified library of LLM
memory models. We drive its ``LTMemory`` (long-term memory) model as a pure vector
memory: ``store(text)`` on add and ``recall(query)`` on retrieve, backed by
``LinearStorage`` + ``TextRetrieval`` (cosine top-k) + ``ConcateUtilization``.

The embedder is injected as a custom ``APIEncoder(BaseEncoder)`` that calls our
siliconflow OpenAI-compatible endpoint with bge-m3 and returns an L2-normalized torch
tensor of shape ``[1, dim]`` (the shape MemEngine's cosine retrieval expects). It is
patched onto ``memengine.function.Encoder`` and ``.Retrieval`` so the config method
name ``APIEncoder`` resolves. MemEngine's LTMemory does NO LLM generation — this is an
embedding-recall memory, so the embed backbone stays fixed on bge-m3 across the ×8 model
axis (the matrix "model" labels the column; the memory backend is model-agnostic on the
embed side, consistent with how the benchmark treats retrieval-only memories like txtai).

``recall`` returns the ranked memories concatenated by ``ConcateUtilization``. We set the
utilization ``sep`` to an ASCII record-separator (\\x1e) with ``index=False`` so the
concatenation can be split cleanly back into one row per retrieved memory in rank order.

NOTE: requires ``.venv-memengine`` (``pip install memengine`` with ``langchain==0.3.27``
+ ``langchain-core==0.3.29``; then ``pip install --target <venv-site-packages> openai``
so the pod — which disables user-site — sees it).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from amb.clients.core.common import ensure_site_packages_from_venv

DEFAULT_MEMENGINE_VENV = Path(__file__).resolve().parents[4] / ".venv-memengine"

# ASCII record separator: unlikely to appear inside a memory fact, so it cleanly delimits
# the memories that ConcateUtilization joins in rank order.
_SEP = "\x1e"


class MemEngineClient:
    """Thin adapter driving MemEngine LTMemory in-process for AMB reset/add/search."""

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
        self._mem = None
        self._docs: list[str] = []
        self.case_id: str | None = None

    def _build(self):
        import numpy as np
        import torch
        from openai import OpenAI  # type: ignore
        import memengine.function.Encoder as ENC  # type: ignore
        import memengine.function.Retrieval as RET  # type: ignore
        from memengine import MemoryConfig, LTMemory  # type: ignore

        embed_model = self._embed_model
        cli = OpenAI(base_url=self._embed_base, api_key=self._embed_key)

        class APIEncoder(ENC.BaseEncoder):
            def __init__(self, config):
                super().__init__(config)
                self.model_name = getattr(config, "name", embed_model)
                self.device = "cpu"

            def __call__(self, text, return_type="tensor"):
                resp = cli.embeddings.create(model=self.model_name, input=[text])
                v = np.array([d.embedding for d in resp.data], dtype="float32")
                v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
                return v if return_type == "numpy" else torch.from_numpy(v)

            def reset(self):
                pass

        # Patch onto both modules so config method name "APIEncoder" resolves wherever
        # MemEngine looks it up (encoder factory + retrieval-embedded encoder).
        ENC.APIEncoder = APIEncoder
        RET.APIEncoder = APIEncoder

        cfg = {
            "global_config": {"usable_gpu": ""},
            "storage": {"method": "LinearStorage"},
            "recall": {
                "method": "LTMemoryRecall",
                "empty_memory": "None",
                "truncation": {"method": "LMTruncation", "mode": "word", "number": 2000},
                "utilization": {
                    "method": "ConcateUtilization",
                    "list_config": {"index": False, "sep": _SEP},
                    "prefix": "",
                    "suffix": "",
                },
                "text_retrieval": {
                    "method": "TextRetrieval",
                    "mode": "cosine",
                    "topk": self.default_limit,
                    "encoder": {"method": "APIEncoder", "name": embed_model},
                },
            },
            "store": {"method": "LTMemoryStore"},
            "display": {"method": "ScreenDisplay"},
        }
        return LTMemory(MemoryConfig(cfg))

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        self._mem = self._build()
        self._docs = []
        return {"ok": True}

    def _ensure(self):
        if self._mem is None:
            self.reset()
        return self._mem

    def add(self, content: Any = None, *, messages: Any = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        mem = self._ensure()
        text = _text_payload(content=content, messages=messages)
        if not text.strip():
            return {"ok": True, "skipped": "empty"}
        mem.store(text)
        self._docs.append(text)
        return {"ok": True, "id": str(len(self._docs) - 1)}

    def search(self, query: str | None = None, *, user_id: str | None = None, limit: int | None = None,
               top_k: int | None = None, **_: Any) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required for MemEngine search")
        mem = self._ensure()
        if not self._docs:
            return []
        try:
            out = mem.recall(str(query))
        except Exception:
            return []
        k = int(limit or top_k or self.default_limit)
        parts = [p.strip() for p in str(out).split(_SEP) if p.strip()]
        return [{"id": str(i), "content": p, "metadata": {}} for i, p in enumerate(parts[:k])]

    def get_all(self, *, user_id: str | None = None, **_: Any) -> list[dict[str, Any]]:
        return [{"id": str(i), "content": t, "metadata": {}} for i, t in enumerate(self._docs)]

    def delete(self, memory_id: str | None = None, **_: Any) -> dict[str, Any]:
        return {"ok": True, "note": "memengine in-memory LTMemory delete not used in AMB flow"}


def _text_payload(*, content: Any, messages: Any) -> str:
    if messages is not None:
        if isinstance(messages, list):
            return "\n".join(
                str(m.get("content", m)) if isinstance(m, dict) else str(m) for m in messages
            )
        if isinstance(messages, dict):
            return str(messages.get("content", messages))
        return str(messages)
    if content is None:
        return ""
    return str(content)


def create_client(
    *,
    venv_root: str | None = None,
    model: str = "zai-org/GLM-5.2",  # accepted for matrix-symmetry; LTMemory does no LLM gen
    base_url: str | None = "https://api.siliconflow.cn/v1",
    api_key_env: str = "OPENAI_API_KEY",
    embedding_model: str = "BAAI/bge-m3",
    embedding_base_url: str | None = "https://api.siliconflow.cn/v1",
    embedding_api_key_env: str = "OPENAI_API_KEY",
    embedding_dims: int = 1024,
    default_limit: int = 5,
    **_: Any,
) -> MemEngineClient:
    vroot = Path(venv_root) if venv_root else DEFAULT_MEMENGINE_VENV
    ensure_site_packages_from_venv(vroot)
    api_key = os.environ.get(api_key_env, "") or os.environ.get("OPENAI_API_KEY", "")
    embed_key = os.environ.get(embedding_api_key_env, "") or api_key
    return MemEngineClient(
        embedding_model=embedding_model, embedding_base_url=embedding_base_url,
        embedding_api_key=embed_key, default_limit=default_limit,
    )
