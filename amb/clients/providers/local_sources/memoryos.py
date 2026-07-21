"""Local official-source MemoryOS client factory for AutoMemoryBench integration runs."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from amb.clients.core.common import effective_api_key, ensure_site_packages_from_venv, ensure_source_path, resolve_path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MEMORYOS_VENV = PROJECT_ROOT / ".venv-memoryos"


class MemoryOSOfficialSourceClient:
    """Thin adapter over the official ``memoryos.Memoryos`` Python API."""

    def __init__(
        self,
        memoryos: Any | None = None,
        *,
        default_limit: int = 5,
        memoryos_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        if memoryos is None and memoryos_factory is None:
            raise ValueError("memoryos or memoryos_factory is required")
        self.memoryos = memoryos
        self._memoryos_factory = memoryos_factory
        self.default_limit = default_limit
        self.case_id: str | None = None
        self._written: list[dict[str, Any]] = []

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        self._written = []
        if self._memoryos_factory is not None:
            if not self.case_id:
                raise ValueError("case_id or user_id is required for MemoryOS reset")
            self.memoryos = self._memoryos_factory(str(self.case_id))
        return {"ok": True, "user_id": getattr(self.memoryos, "user_id", self.case_id)}

    def add(
        self,
        content: Any = None,
        *,
        messages: Any = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        pairs = _qa_pairs_from_payload(content=content, messages=messages, metadata=metadata)
        results: list[dict[str, Any]] = []
        for pair in pairs:
            self.memoryos.add_memory(
                user_input=pair["user_input"],
                agent_response=pair["agent_response"],
                timestamp=pair.get("timestamp"),
                meta_data=metadata or {},
            )
            row = {"user_id": user_id or self.case_id, **pair, "metadata": metadata or {}}
            self._written.append(row)
            results.append(row)
        return {"added": len(results), "user_id": user_id or self.case_id, "results": results}

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
            raise ValueError("query is required for MemoryOS retrieval")
        k = int(limit or top_k or self.default_limit)
        retriever = getattr(self.memoryos, "retriever", None)
        if retriever is None or not hasattr(retriever, "retrieve_context"):
            raw = self.memoryos.get_response(query=str(query))
            return [{"id": "memoryos-response", "content": str(raw), "score": None, "metadata": {"kind": "response"}}]
        raw = retriever.retrieve_context(user_query=str(query), user_id=user_id or getattr(self.memoryos, "user_id", None))
        results = _normalize_retrieval(raw, limit=k)
        # MemoryOS tiers memories short->mid->long; retrieve_context only searches
        # mid/long-term, so memories observed within a case (which sit in short-term
        # until promoted) are NEVER retrieved -> the model answers "no memory". Include
        # short-term memory directly so freshly-observed turns are retrievable.
        st = getattr(self.memoryos, "short_term_memory", None)
        st_items = list(getattr(st, "memory", []) or []) if st is not None else []
        st_results: list[dict[str, Any]] = []
        for idx, qa in enumerate(st_items):
            ui = str(qa.get("user_input") or "").strip()
            ar = str(qa.get("agent_response") or "").strip()
            content = (ui + ("\n" + ar if ar else "")).strip()
            if content:
                st_results.append({"id": f"memoryos-short-{idx}", "content": content,
                                   "score": None, "metadata": {"tier": "short_term",
                                   "timestamp": qa.get("timestamp")}})
        # short-term (recent, in-case) first, then mid/long-term retrievals; dedup by content
        seen = set(); merged: list[dict[str, Any]] = []
        for r in st_results + results:
            ckey = (r.get("content") or "")[:200]
            if ckey and ckey not in seen:
                seen.add(ckey); merged.append(r)
        return merged[:k] if k else merged

    def get_all(self, *, user_id: str | None = None, **_: Any) -> list[dict[str, Any]]:
        rows = [
            {
                "id": f"memoryos-observation-{idx}",
                "content": f"{row['user_input']}\n{row['agent_response']}".strip(),
                "metadata": {"user_id": row.get("user_id"), **dict(row.get("metadata") or {})},
            }
            for idx, row in enumerate(self._written, start=1)
        ]
        if hasattr(self.memoryos, "get_user_profile_summary"):
            profile = self.memoryos.get_user_profile_summary()
            if profile:
                rows.append({"id": "memoryos-user-profile", "content": str(profile), "metadata": {"kind": "profile"}})
        if hasattr(self.memoryos, "get_memory_stats"):
            rows.append({"id": "memoryos-stats", "content": str(self.memoryos.get_memory_stats()), "metadata": {"kind": "stats"}})
        return rows

    def delete(self, memory_id: str | None = None, *, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        return {
            "deleted": False,
            "reason": "official MemoryOS Python API does not expose a stable per-memory delete in the adapter contract",
            "memory_id": memory_id,
        }


def create_client(
    *,
    source_root: str = "related_work/repos/MemoryOS/memoryos-pypi",
    venv_root: str | None = None,
    extra_venv_roots: list[str] | tuple[str, ...] = (),
    api_key: str | None = None,
    api_key_env: str = "MEMORYOS_OPENAI_API_KEY",
    fallback_envs: list[str] | tuple[str, ...] = ("OPENAI_API_KEY", "AMST_OPENAI_COMPAT_API_KEY"),
    base_url: str | None = None,
    base_url_env: str = "MEMORYOS_OPENAI_BASE_URL",
    embedding_base_url: str | None = None,      # default: same endpoint as the chat LLM
    embedding_api_key_env: str | None = None,   # default: same key as the chat LLM
    data_storage_path: str | None = None,
    user_id: str = "amst_memoryos_user",
    assistant_id: str = "amst_memoryos_assistant",
    llm_model: str = "gpt-4o-mini",
    embedding_model: str | None = None,         # matrix-injected embedder name; overrides embedding_model_name when set
    embedding_model_name: str = "all-MiniLM-L6-v2",
    short_term_capacity: int = 10,
    mid_term_capacity: int = 2000,
    long_term_knowledge_capacity: int = 100,
    retrieval_queue_capacity: int = 7,
    mid_term_heat_threshold: float = 5.0,
    mid_term_similarity_threshold: float = 0.6,
    default_limit: int = 5,
    clear_case_storage_on_reset: bool = True,
) -> MemoryOSOfficialSourceClient:
    """Create a MemoryOS client from the official local source tree."""

    ensure_site_packages_from_venv(Path(venv_root) if venv_root else DEFAULT_MEMORYOS_VENV)
    for extra_venv_root in extra_venv_roots:
        ensure_site_packages_from_venv(Path(extra_venv_root))
    import_root = ensure_source_path(source_root)
    _prioritize_memoryos_source(import_root)
    _purge_memoryos_import_conflicts(import_root)
    from memoryos import Memoryos  # type: ignore

    resolved_key = effective_api_key(
        api_key=api_key,
        api_key_env=api_key_env,
        fallback_envs=tuple(fallback_envs),
    )
    if not resolved_key:
        raise RuntimeError(f"{api_key_env}, OPENAI_API_KEY, or AMST_OPENAI_COMPAT_API_KEY is required for MemoryOS official runs")
    resolved_base_url = base_url or os.getenv(base_url_env)
    # The embedder (e.g. bge-m3) may live on a different endpoint/key than the chat
    # LLM (e.g. chat and embeddings on your OpenAI-compatible endpoint(s)). Default to the LLM's
    # endpoint/key so behaviour is unchanged when these params are None.
    resolved_embedding_key = (
        effective_api_key(
            api_key=None,
            api_key_env=embedding_api_key_env,
            fallback_envs=(api_key_env, "OPENAI_API_KEY"),
        )
        if embedding_api_key_env
        else resolved_key
    )
    resolved_embedding_base_url = embedding_base_url or resolved_base_url
    # The systems x models matrix injects the embedder name as `embedding_model`
    # (matching the sibling mem0 adapter); honour it while staying backward
    # compatible with the historical `embedding_model_name` param when unset.
    resolved_embedding_model_name = embedding_model or embedding_model_name
    storage_path = _storage_path(data_storage_path)

    def build_memoryos(case_id: str) -> Any:
        safe_case_id = _safe_memoryos_id(case_id)
        case_user_id = f"{_safe_memoryos_id(user_id)}_{safe_case_id}"
        case_assistant_id = f"{_safe_memoryos_id(assistant_id)}_{safe_case_id}"
        if clear_case_storage_on_reset:
            _clear_memoryos_case_storage(storage_path, user_id=case_user_id, assistant_id=case_assistant_id)
        # Route the embedder to its own endpoint/key (when provided) without touching
        # the chat LLM's openai_api_key/openai_base_url below. MemoryOS forwards
        # embedding_model_kwargs to its embedding backend only.
        embedding_model_kwargs: dict[str, Any] = {}
        if embedding_base_url is not None or embedding_api_key_env is not None:
            embedding_model_kwargs = {
                "openai_base_url": resolved_embedding_base_url,
                "openai_api_key": resolved_embedding_key,
            }
        memoryos_kwargs: dict[str, Any] = {}
        if embedding_model_kwargs:
            memoryos_kwargs["embedding_model_kwargs"] = embedding_model_kwargs
        return Memoryos(
            user_id=case_user_id,
            openai_api_key=resolved_key,
            openai_base_url=resolved_base_url,
            data_storage_path=str(storage_path),
            assistant_id=case_assistant_id,
            short_term_capacity=short_term_capacity,
            mid_term_capacity=mid_term_capacity,
            long_term_knowledge_capacity=long_term_knowledge_capacity,
            retrieval_queue_capacity=retrieval_queue_capacity,
            mid_term_heat_threshold=mid_term_heat_threshold,
            mid_term_similarity_threshold=mid_term_similarity_threshold,
            llm_model=llm_model,
            embedding_model_name=resolved_embedding_model_name,
            **memoryos_kwargs,
        )

    return MemoryOSOfficialSourceClient(
        build_memoryos(user_id),
        default_limit=default_limit,
        memoryos_factory=build_memoryos,
    )


def _storage_path(raw: str | None) -> Path:
    if raw:
        path = resolve_path(raw, Path(raw))
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(tempfile.mkdtemp(prefix="amst-memoryos-"))


def _prioritize_memoryos_source(import_root: Path) -> None:
    raw = str(import_root)
    while raw in sys.path:
        sys.path.remove(raw)
    sys.path.insert(0, raw)


def _purge_memoryos_import_conflicts(import_root: Path) -> None:
    """MemoryOS uses top-level fallback imports that conflict with benchmark repos."""

    root = str(import_root.resolve())
    for name in ("memoryos", "utils", "prompts", "short_term", "mid_term", "long_term", "updater", "retriever"):
        module = sys.modules.get(name)
        module_file = str(getattr(module, "__file__", "") or "")
        if module is None:
            continue
        if module_file and Path(module_file).resolve().as_posix().startswith(root):
            continue
        sys.modules.pop(name, None)


def _safe_memoryos_id(raw: str) -> str:
    value = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(raw))
    return value or "default"


def _clear_memoryos_case_storage(storage_path: Path, *, user_id: str, assistant_id: str) -> None:
    for path in (
        storage_path / "users" / user_id,
        storage_path / "assistants" / assistant_id,
    ):
        if path.exists():
            shutil.rmtree(path)


def _qa_pairs_from_payload(*, content: Any, messages: Any, metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    timestamp = str((metadata or {}).get("timestamp") or "") or None
    if messages is not None:
        values = messages if isinstance(messages, list) else [messages]
        normalized = [_normalize_message(value) for value in values]
        pairs: list[dict[str, Any]] = []
        pending_user: str | None = None
        for message in normalized:
            if message["role"] == "user":
                if pending_user is not None:
                    pairs.append({"user_input": pending_user, "agent_response": "", "timestamp": timestamp})
                pending_user = message["content"]
            elif message["role"] == "assistant" and pending_user is not None:
                pairs.append({"user_input": pending_user, "agent_response": message["content"], "timestamp": timestamp})
                pending_user = None
            else:
                pairs.append({"user_input": message["content"], "agent_response": "", "timestamp": timestamp})
        if pending_user is not None:
            pairs.append({"user_input": pending_user, "agent_response": "", "timestamp": timestamp})
        return pairs
    if content is None:
        raise ValueError("content or messages is required")
    role = str((metadata or {}).get("role") or "user")
    if role == "assistant":
        return [{"user_input": "", "agent_response": str(content), "timestamp": timestamp}]
    return [{"user_input": str(content), "agent_response": str((metadata or {}).get("agent_response") or ""), "timestamp": timestamp}]


def _normalize_message(message: Any) -> dict[str, str]:
    if not isinstance(message, dict):
        return {"role": "user", "content": str(message)}
    return {"role": str(message.get("role") or "user"), "content": str(message.get("content", ""))}


def _normalize_retrieval(raw: Any, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return [{"id": "memoryos-result-1", "content": str(raw), "score": None, "metadata": {}}]
    rows: list[dict[str, Any]] = []
    for idx, page in enumerate(raw.get("retrieved_pages") or [], start=1):
        rows.append(
            {
                "id": f"memoryos-page-{idx}",
                "content": f"{page.get('user_input', '')}\n{page.get('agent_response', '')}".strip(),
                "score": page.get("score"),
                "metadata": {"kind": "mid_term_page", "timestamp": page.get("timestamp"), "meta_info": page.get("meta_info")},
            }
        )
    for idx, item in enumerate(raw.get("retrieved_user_knowledge") or [], start=1):
        rows.append(
            {
                "id": f"memoryos-user-knowledge-{idx}",
                "content": str(item.get("knowledge", item)),
                "score": item.get("score"),
                "metadata": {"kind": "user_knowledge", "timestamp": item.get("timestamp")},
            }
        )
    for idx, item in enumerate(raw.get("retrieved_assistant_knowledge") or [], start=1):
        rows.append(
            {
                "id": f"memoryos-assistant-knowledge-{idx}",
                "content": str(item.get("knowledge", item)),
                "score": item.get("score"),
                "metadata": {"kind": "assistant_knowledge", "timestamp": item.get("timestamp")},
            }
        )
    return rows[:limit]
