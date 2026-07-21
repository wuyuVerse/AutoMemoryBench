"""Local in-process ReMe (ModelScope) memory client factory for AutoMemoryBench.

ReMe (`reme-ai`, github.com/modelscope/ReMe) is a note-based agent-memory "OS" that
normally runs as an HTTP daemon. Its daemon startup is fragile under our sandbox, so we
drive it **in-process**: build the ReMe Application (`ReMe(**config)`), `await app.start()`,
and invoke its jobs directly — `auto_memory` (record conversation facts into daily notes)
for add, and `search` (hybrid vector+BM25 over the note workspace) for retrieval. This
matches the in-process adapter pattern used by the other library-style systems.

LLM + embedder are OpenAI-compatible via env (LLM_BACKEND/LLM_BASE_URL + EMBEDDING_*),
so the chat model varies per matrix model (×8) and the embedder stays fixed on bge-m3.
Each case gets an isolated note workspace (no external DB/server).

NOTE: requires `.venv-reme` (reme-ai + agentscope) with the as_embedding `dimensions`
patch applied (agentscope OpenAIEmbeddingModel needs `dimensions` as a direct kwarg).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

from amb.clients.core.common import ensure_site_packages_from_venv

DEFAULT_REME_VENV = Path(__file__).resolve().parents[4] / ".venv-reme"


def _prioritize_venv(venv_root: Path) -> None:
    """Force the ReMe venv's site-packages ahead of system dist-packages.

    On the cluster pod, the system `attr`/`attrs` (`/usr/local/lib/.../dist-packages`)
    can shadow the venv's newer versions (agentscope/jsonschema need attrs>=24.2's
    `ClassProps`), causing `ImportError: cannot import name 'ClassProps'`. We move the
    venv site-packages to the FRONT of sys.path and evict any already-imported stale
    `attr*` modules so ReMe's imports resolve against the venv.
    """
    import sys
    import sysconfig  # noqa: F401
    site = None
    for cand in venv_root.glob("lib/python*/site-packages"):
        site = str(cand)
        break
    if not site:
        return
    # de-dup then prepend
    sys.path[:] = [p for p in sys.path if p != site]
    sys.path.insert(0, site)
    for name in list(sys.modules):
        if name == "attr" or name == "attrs" or name.startswith("attr.") or name.startswith("attrs."):
            mod = sys.modules.get(name)
            f = getattr(mod, "__file__", "") or ""
            if site not in f:  # only evict the shadowing (non-venv) copy
                sys.modules.pop(name, None)


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


class ReMeClient:
    """Thin adapter driving ReMe jobs in-process for AMB reset/add/search."""

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
        # ReMe reads model/embedder from env at config-resolve time.
        os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")
        os.environ["LLM_BACKEND"] = "openai"
        os.environ["LLM_MODEL_NAME"] = model
        if base_url:
            os.environ["LLM_BASE_URL"] = base_url
        os.environ["LLM_API_KEY"] = api_key
        os.environ["EMBEDDING_BACKEND"] = "openai"
        os.environ["EMBEDDING_MODEL_NAME"] = embedding_model
        if embedding_base_url:
            os.environ["EMBEDDING_BASE_URL"] = embedding_base_url
        os.environ["EMBEDDING_API_KEY"] = embedding_api_key
        os.environ["EMBEDDING_DIMENSIONS"] = str(embedding_dims)

        self.default_limit = default_limit
        self._model = model
        self._app = None
        self._ws: str | None = None
        self.case_id: str | None = None

    def _build_app(self, workspace: str):
        from reme.reme import ReMe  # type: ignore
        from reme.config import resolve_app_config  # type: ignore

        cfg = resolve_app_config(workspace=workspace)
        app = ReMe(**cfg)
        _run(app.start())
        return app

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        # fresh note workspace per case → no cross-case leakage, no shared DB
        self._ws = tempfile.mkdtemp(prefix="reme_ws_")
        os.environ["REME_WORKSPACE"] = self._ws
        os.environ["REME_HOME"] = self._ws
        self._app = self._build_app(self._ws)
        return {"ok": True, "workspace": self._ws}

    def _jobs(self):
        if self._app is None:
            self.reset()
        return self._app.context.jobs

    def add(self, content: Any = None, *, messages: Any = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        jobs = self._jobs()
        msgs = _messages_payload(content=content, messages=messages)
        try:
            _run(jobs["auto_memory"](messages=msgs, session_id=str(self.case_id or "amb")))
        except Exception:
            pass
        # make freshly-written notes searchable
        try:
            if "reindex" in jobs:
                _run(jobs["reindex"]())
        except Exception:
            pass
        return {"ok": True}

    def search(self, query: str | None = None, *, user_id: str | None = None, limit: int | None = None,
               top_k: int | None = None, **_: Any) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required for ReMe search")
        jobs = self._jobs()
        k = int(limit or top_k or self.default_limit)
        try:
            if "reindex" in jobs:
                _run(jobs["reindex"]())
        except Exception:
            pass
        try:
            resp = _run(jobs["search"](query=str(query), limit=k))
        except Exception:
            return []
        return _normalize_search(resp, k)

    def get_all(self, *, user_id: str | None = None, **_: Any) -> list[dict[str, Any]]:
        return []

    def delete(self, memory_id: str | None = None, **_: Any) -> dict[str, Any]:
        return {"ok": True, "note": "reme note-workspace delete not used in AMB flow"}


def create_client(
    *,
    venv_root: str | None = None,
    model: str = "zai-org/GLM-5.2",
    base_url: str | None = "https://api.siliconflow.cn/v1",
    api_key_env: str = "OPENAI_API_KEY",
    embedding_model: str = "BAAI/bge-m3",
    embedding_base_url: str | None = "https://api.siliconflow.cn/v1",
    embedding_api_key_env: str = "OPENAI_API_KEY",
    embedding_dims: int = 1024,
    default_limit: int = 5,
    **_: Any,
) -> ReMeClient:
    vroot = Path(venv_root) if venv_root else DEFAULT_REME_VENV
    _prioritize_venv(vroot)
    ensure_site_packages_from_venv(vroot)
    api_key = os.environ.get(api_key_env, "") or os.environ.get("OPENAI_API_KEY", "")
    embed_key = os.environ.get(embedding_api_key_env, "") or api_key
    return ReMeClient(
        model=model, base_url=base_url, api_key=api_key,
        embedding_model=embedding_model, embedding_base_url=embedding_base_url, embedding_api_key=embed_key,
        embedding_dims=embedding_dims, default_limit=default_limit,
    )


def _messages_payload(*, content: Any, messages: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if messages is not None:
        vals = messages if isinstance(messages, list) else [messages]
        for m in vals:
            if isinstance(m, dict):
                role = m.get("role", "user")
                text = m.get("content", "")
                if isinstance(text, list):
                    text = " ".join(str(p.get("text", p) if isinstance(p, dict) else p) for p in text)
                out.append({"role": role, "name": role, "content": str(text)})
            else:
                out.append({"role": "user", "name": "user", "content": str(m)})
    elif content is not None:
        out.append({"role": "user", "name": "user", "content": str(content)})
    else:
        raise ValueError("content or messages is required")
    return out


def _normalize_search(resp: Any, k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    md = getattr(resp, "metadata", None) or {}
    results = md.get("results") if isinstance(md, dict) else None
    if results:
        for idx, r in enumerate(results, start=1):
            if isinstance(r, dict):
                text = r.get("content") or r.get("text") or r.get("snippet") or r.get("chunk") or str(r)
                rid = r.get("id", idx)
                score = r.get("score")
            else:
                text, rid, score = str(r), idx, None
            rows.append({"id": str(rid), "content": str(text), "score": score, "metadata": {}})
            if len(rows) >= k:
                break
    if not rows:
        ans = getattr(resp, "answer", "") or ""
        if ans.strip():
            rows.append({"id": "1", "content": str(ans), "score": None, "metadata": {}})
    return rows
