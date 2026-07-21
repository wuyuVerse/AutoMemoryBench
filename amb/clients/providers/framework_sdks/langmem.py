"""Official framework-SDK LangMem client factory for AutoMemoryBench integration runs."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

from amb.clients.core.common import (
    effective_api_key as resolve_api_key,
    ensure_site_packages_from_venv,
    import_or_raise,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LANGMEM_VENV = PROJECT_ROOT / ".venv-langmem"
DEFAULT_TIKTOKEN_CACHE = PROJECT_ROOT / ".cache" / "tiktoken"


class LangMemRealClient:
    """Thin adapter exposing LangMem-like store methods to AMST."""

    def __init__(
        self,
        backend: Any,
        *,
        default_limit: int = 5,
        transient_max_attempts: int = 8,
        transient_base_sleep: float = 2.0,
        transient_max_sleep: float = 60.0,
    ) -> None:
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        if transient_max_attempts <= 0:
            raise ValueError("transient_max_attempts must be positive")
        if transient_base_sleep <= 0:
            raise ValueError("transient_base_sleep must be positive")
        if transient_max_sleep <= 0:
            raise ValueError("transient_max_sleep must be positive")
        self.backend = backend
        self.default_limit = default_limit
        self.transient_max_attempts = transient_max_attempts
        self.transient_base_sleep = transient_base_sleep
        self.transient_max_sleep = transient_max_sleep

    def reset(self, **_: Any) -> dict[str, Any]:
        return {"ok": True}

    def clear(self) -> dict[str, Any]:
        return {"ok": True}

    def put(
        self,
        content: Any = None,
        *,
        messages: Any = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        payload = _normalize_payload(content=content, messages=messages)
        if not user_id:
            raise ValueError("user_id is required for LangMem put")
        if _is_memory_store_manager(self.backend):
            metadata_dict = dict(metadata or {})
            key = str(metadata_dict.get("memory_id") or metadata_dict.get("turn_id") or f"{user_id}:{abs(hash(str(payload)))}")
            result = self._call_with_transient_retry(
                lambda: self.backend.put(
                    key,
                    {"memory": payload, "metadata": metadata_dict},
                    config=_langgraph_config(str(user_id)),
                )
            )
            return {"key": key, "text": str(payload), "metadata": metadata_dict, "raw_result": result}
        method = _resolve_method(self.backend, "put", "add", "store", "save")
        return self._call_with_transient_retry(
            lambda: _call_variants(
                (
                    lambda: method(payload, user_id=str(user_id), metadata=dict(metadata or {})),
                    lambda: method(content=payload, user_id=str(user_id), metadata=dict(metadata or {})),
                    lambda: method(memory=payload, user_id=str(user_id), metadata=dict(metadata or {})),
                    lambda: method(payload, metadata=dict(metadata or {})),
                    lambda: method(payload),
                )
            )
        )

    def search(
        self,
        query: str | None = None,
        *,
        user_id: str | None = None,
        limit: int | None = None,
        top_k: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not query:
            raise ValueError("query is required for LangMem search")
        if not user_id:
            raise ValueError("user_id is required for LangMem search")
        k = int(limit or top_k or self.default_limit)
        if _is_memory_store_manager(self.backend):
            items = self._call_with_transient_retry(
                lambda: self.backend.search(
                    query=str(query),
                    limit=k,
                    config=_langgraph_config(str(user_id)),
                )
            )
            return {"items": [_memory_store_search_item(item) for item in items]}
        method = _resolve_method(self.backend, "search", "retrieve", "query")
        return self._call_with_transient_retry(
            lambda: _call_variants(
                (
                    lambda: method(query=str(query), user_id=str(user_id), limit=k),
                    lambda: method(query=str(query), user_id=str(user_id), top_k=k),
                    lambda: method(str(query), user_id=str(user_id), limit=k),
                    lambda: method(str(query), user_id=str(user_id), top_k=k),
                    lambda: method(str(query), k),
                    lambda: method(str(query)),
                )
            )
        )

    def list(self, *, user_id: str | None = None, limit: int = 100, **_: Any) -> list[dict[str, Any]] | dict[str, Any]:
        if not user_id:
            raise ValueError("user_id is required for LangMem list")
        if _is_memory_store_manager(self.backend):
            items = self._call_with_transient_retry(
                lambda: self.backend.search(
                    query=None,
                    limit=int(limit),
                    config=_langgraph_config(str(user_id)),
                )
            )
            return [_memory_store_search_item(item) for item in items]
        method = _resolve_method(self.backend, "list", "all", "get_all", "export_memory")
        return self._call_with_transient_retry(
            lambda: _call_variants(
                (
                    lambda: method(user_id=str(user_id), limit=int(limit)),
                    lambda: method(user_id=str(user_id)),
                    lambda: method(str(user_id)),
                    lambda: method(),
                )
            )
        )

    def delete(self, memory_id: str | None = None, *, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        if not memory_id:
            raise ValueError("memory_id is required for LangMem delete")
        if _is_memory_store_manager(self.backend):
            self._call_with_transient_retry(
                lambda: self.backend.delete(str(memory_id), config=_langgraph_config(str(user_id or "global")))
            )
            return {"ok": True, "key": str(memory_id)}
        method = _resolve_method(self.backend, "delete", "remove")
        return self._call_with_transient_retry(
            lambda: _call_variants(
                (
                    lambda: method(memory_id=str(memory_id), user_id=str(user_id)) if user_id else method(memory_id=str(memory_id)),
                    lambda: method(str(memory_id)),
                )
            )
        )

    def _call_with_transient_retry(self, fn):
        return _call_with_transient_retry(
            fn,
            max_attempts=self.transient_max_attempts,
            base_sleep=self.transient_base_sleep,
            max_sleep=self.transient_max_sleep,
        )


def create_client(
    *,
    backend: Any | None = None,
    backend_factory: str | None = None,
    venv_root: str | None = None,
    default_limit: int = 5,
    model: str = "openai:gpt-4o-mini",
    store: Any | None = None,
    base_url: str | None = None,                # chat LLM endpoint (OpenAI-compatible)
    api_key_env: str | None = None,             # env var holding the chat LLM key
    embedding_model: str | None = None,         # embedder model name (default BAAI/bge-m3)
    embedding_base_url: str | None = None,      # default: same endpoint as the LLM
    embedding_api_key_env: str | None = None,   # default: same key as the LLM
    chat_timeout: float | None = None,
    chat_max_retries: int | None = None,
    embedding_timeout: float | None = None,
    embedding_max_retries: int | None = None,
    transient_max_attempts: int | None = None,
    transient_base_sleep: float | None = None,
    transient_max_sleep: float | None = None,
    **factory_kwargs: Any,
) -> LangMemRealClient:
    _ensure_tiktoken_cache()
    # The chat LLM may live on a DIFFERENT endpoint/key than the embedder
    # (e.g. chat and embeddings on your OpenAI-compatible endpoint(s)). When a chat override is
    # supplied, build an explicit OpenAI-compatible chat model pinned to that
    # base_url/key and pass the MODEL INSTANCE to the langmem factory, so the
    # chat model does NOT inherit (and we never mutate) the ambient OPENAI_*
    # env — which must stay pointed at the embedder's endpoint. With no
    # override (both None) behavior is identical to before: the plain
    # "openai:<model>" string is passed through and langgraph resolves it from
    # the ambient OpenAI env.
    # langchain (used by _resolve_chat_model when base_url/api_key_env override the
    # chat endpoint) lives in the langmem venv, so its site-packages MUST be on the
    # path BEFORE resolving the chat model — otherwise `import langchain.chat_models`
    # raises ModuleNotFoundError under the default interpreter.
    ensure_site_packages_from_venv(Path(venv_root) if venv_root else DEFAULT_LANGMEM_VENV)
    model = _resolve_chat_model(
        model,
        base_url=base_url,
        api_key_env=api_key_env,
        timeout=chat_timeout,
        max_retries=chat_max_retries,
    )
    if backend is None:
        ensure_site_packages_from_venv(Path(venv_root) if venv_root else DEFAULT_LANGMEM_VENV)
        if backend_factory:
            module_name, _, attr_name = str(backend_factory).partition(":")
            module = import_or_raise(module_name)
            factory = getattr(module, attr_name)
            backend = _call_langmem_factory(
                factory,
                model=model,
                store=store,
                factory_kwargs=factory_kwargs,
                embedding_model=embedding_model,
                embedding_base_url=embedding_base_url,
                embedding_api_key_env=embedding_api_key_env,
                embedding_timeout=embedding_timeout,
                embedding_max_retries=embedding_max_retries,
            )
        else:
            module = import_or_raise("langmem")
            factory = (
                getattr(module, "create_store", None)
                or getattr(module, "Client", None)
                or getattr(module, "create_memory_store_manager", None)
            )
            if factory is None:
                raise AttributeError("langmem module does not expose create_store, Client, or create_memory_store_manager")
            backend = _call_langmem_factory(
                factory,
                model=model,
                store=store,
                factory_kwargs=factory_kwargs,
                embedding_model=embedding_model,
                embedding_base_url=embedding_base_url,
                embedding_api_key_env=embedding_api_key_env,
                embedding_timeout=embedding_timeout,
                embedding_max_retries=embedding_max_retries,
            )
    return LangMemRealClient(
        backend,
        default_limit=default_limit,
        transient_max_attempts=_int_env("AMB_LANGMEM_TRANSIENT_MAX_ATTEMPTS", transient_max_attempts, default=12),
        transient_base_sleep=_float_env("AMB_LANGMEM_TRANSIENT_BASE_SLEEP", transient_base_sleep, default=2.0),
        transient_max_sleep=_float_env("AMB_LANGMEM_TRANSIENT_MAX_SLEEP", transient_max_sleep, default=120.0),
    )


def _ensure_tiktoken_cache() -> None:
    """Pin tiktoken to the repo-shared cache on offline workers."""

    cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR")
    if cache_dir and Path(cache_dir).exists():
        return
    if DEFAULT_TIKTOKEN_CACHE.exists():
        os.environ["TIKTOKEN_CACHE_DIR"] = str(DEFAULT_TIKTOKEN_CACHE)


def _normalize_payload(*, content: Any, messages: Any) -> Any:
    if messages is not None:
        return messages
    if isinstance(content, (str, list, dict)):
        return content
    if content is None:
        raise ValueError("content or messages is required")
    return str(content)


def _resolve_method(target: Any, *names: str):
    for name in names:
        method = getattr(target, name, None)
        if callable(method):
            return method
    raise AttributeError(f"backend does not expose any of methods: {', '.join(names)}")


def _call_variants(variants):
    last_error: TypeError | None = None
    for attempt in variants:
        try:
            return attempt()
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("no call variants supplied")


def _resolve_chat_model(
    model: Any,
    *,
    base_url: str | None,
    api_key_env: str | None,
    timeout: float | None,
    max_retries: int | None,
) -> Any:
    """Return the chat model for the langmem factory.

    When no chat override is given (base_url and api_key_env both None), the
    original value (typically an ``"openai:<model>"`` string) is returned
    unchanged so the langmem/langchain factory resolves it from the ambient
    OpenAI environment — identical to the prior behavior. When an override IS
    given and ``model`` is a string, build an explicit chat model instance via
    ``init_chat_model`` pinned to the chat endpoint/key so the chat model is
    decoupled from the embedder's endpoint/key (and the ambient OPENAI_* env is
    neither read nor mutated).
    """

    if base_url is None and api_key_env is None:
        return model
    if not isinstance(model, str):
        # Already an instantiated chat model: nothing to override.
        return model

    # Accept "openai:<model>" or a bare "<model>"; init_chat_model with an
    # explicit model_provider does not want the "openai:" prefix.
    chat_model_name = model.split(":", 1)[1] if model.startswith("openai:") else model
    resolved_chat_key = resolve_api_key(
        api_key=None,
        api_key_env=api_key_env or "OPENAI_API_KEY",
        fallback_envs=("OPENAI_API_KEY",),
    )
    langchain_openai_module = import_or_raise("langchain_openai")
    return langchain_openai_module.ChatOpenAI(
        model=chat_model_name,
        base_url=base_url,
        api_key=resolved_chat_key,
        timeout=_openai_timeout(
            total=float(timeout or os.environ.get("AMB_LANGMEM_CHAT_TIMEOUT", "120")),
            connect=float(os.environ.get("AMB_LANGMEM_CONNECT_TIMEOUT", "30")),
        ),
        max_retries=int(max_retries or os.environ.get("AMB_LANGMEM_CHAT_MAX_RETRIES", "4")),
    )


def _call_langmem_factory(
    factory: Any,
    *,
    model: Any,
    store: Any | None,
    factory_kwargs: dict[str, Any],
    embedding_model: str | None = None,
    embedding_base_url: str | None = None,
    embedding_api_key_env: str | None = None,
    embedding_timeout: float | None = None,
    embedding_max_retries: int | None = None,
) -> Any:
    if not callable(factory):
        return factory
    if getattr(factory, "__name__", "") == "create_memory_store_manager":
        # FIX (adapter correctness, 0625 audit): a bare InMemoryStore() has no
        # embedding index, so langgraph silently disables semantic search and
        # returns memories in insertion order (FIFO) regardless of query — which
        # collapsed langmem's recall to noise (7.9% AMQ artifact). Build the store
        # WITH an embedding index so search is actually semantic. Config-driven so
        # the embed model/dims can match the deployment's embedding endpoint
        # (default: siliconflow-served BAAI/bge-m3 @ 1024, same as the mem0 setup).
        # NOTE: requires a single-case smoke-test before a full re-run.
        embed = factory_kwargs.pop("index_embed", None) or os.environ.get(
            "AMB_LANGMEM_INDEX_EMBED", "openai:BAAI/bge-m3"
        )
        dims = int(factory_kwargs.pop("index_dims", 0) or os.environ.get("AMB_LANGMEM_INDEX_DIMS", "1024"))
        # The embedder may live on a DIFFERENT endpoint/key than the chat LLM
        # (e.g. chat and embeddings on your OpenAI-compatible endpoint(s) bge-m3). When an embedding
        # override is supplied we build an explicit OpenAI-compatible Embeddings
        # instance pinned to that base_url/key so the embedder does NOT inherit the
        # chat LLM's OPENAI_BASE_URL/OPENAI_API_KEY. With no override (both None) the
        # behavior is identical to before: the plain "openai:..." string is passed
        # through and langgraph resolves it from the ambient OpenAI env.
        embed = _resolve_embedder(
            embed,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            embedding_api_key_env=embedding_api_key_env,
            timeout=embedding_timeout,
            max_retries=embedding_max_retries,
        )
        if store is None:
            memory_module = import_or_raise("langgraph.store.memory")
            store = memory_module.InMemoryStore(index={"dims": dims, "embed": embed})
        return factory(model, store=store, **factory_kwargs)
    return factory(**factory_kwargs)


def _resolve_embedder(
    embed: Any,
    *,
    embedding_model: str | None = None,
    embedding_base_url: str | None,
    embedding_api_key_env: str | None,
    timeout: float | None,
    max_retries: int | None,
) -> Any:
    """Return the embedder for the store index.

    When no embedding override is given (model, base_url and key_env all None),
    the original value (typically an ``"openai:<model>"`` string) is returned
    unchanged so langgraph resolves it from the ambient OpenAI environment —
    identical to the prior behavior. When an override IS given and ``embed`` is
    an ``"openai:<model>"`` string, build an explicit ``OpenAIEmbeddings`` client
    pinned to the embedding endpoint/key (and model name) so the embedder is
    decoupled from the chat LLM's endpoint/key. An explicit ``embedding_model``
    takes precedence over the model name parsed from the ``embed`` string.
    """

    if embedding_model is None and embedding_base_url is None and embedding_api_key_env is None:
        return embed
    if not (isinstance(embed, str) and embed.startswith("openai:")):
        # Non-OpenAI / already-instantiated embedder: nothing to override.
        return embed

    embedding_model = embedding_model or embed.split(":", 1)[1]
    # Fall back to the LLM key envs (then OPENAI_API_KEY) when no dedicated
    # embedding key env is configured.
    resolved_embedding_key = resolve_api_key(
        api_key=None,
        api_key_env=embedding_api_key_env or "OPENAI_API_KEY",
        fallback_envs=("OPENAI_API_KEY",),
    )
    openai_embeddings_module = import_or_raise("langchain_openai")
    return openai_embeddings_module.OpenAIEmbeddings(
        model=embedding_model,
        base_url=embedding_base_url or os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL"),
        api_key=resolved_embedding_key,
        timeout=_openai_timeout(
            total=float(timeout or os.environ.get("AMB_LANGMEM_EMBEDDING_TIMEOUT", "120")),
            connect=float(os.environ.get("AMB_LANGMEM_CONNECT_TIMEOUT", "30")),
        ),
        max_retries=int(max_retries or os.environ.get("AMB_LANGMEM_EMBEDDING_MAX_RETRIES", "4")),
    )


def _openai_timeout(*, total: float, connect: float) -> Any:
    httpx_module = import_or_raise("httpx")
    return httpx_module.Timeout(total, connect=connect)


def _is_memory_store_manager(backend: Any) -> bool:
    put = getattr(backend, "put", None)
    search = getattr(backend, "search", None)
    delete = getattr(backend, "delete", None)
    return callable(put) and callable(search) and callable(delete) and getattr(type(backend), "__name__", "") == "MemoryStoreManager"


def _langgraph_config(user_id: str) -> dict[str, Any]:
    return {"configurable": {"langgraph_user_id": user_id}}


def _memory_store_search_item(item: Any) -> dict[str, Any]:
    value = getattr(item, "value", {}) or {}
    metadata = value.get("metadata") if isinstance(value, dict) else None
    text = value.get("memory") if isinstance(value, dict) else None
    return {
        "key": str(getattr(item, "key", "")),
        "text": str(text or value),
        "score": getattr(item, "score", None),
        "metadata": metadata if isinstance(metadata, dict) else None,
    }


def _call_with_transient_retry(fn, *, max_attempts: int, base_sleep: float, max_sleep: float = 60.0):
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - provider SDK exceptions vary.
            if not _is_transient_provider_error(exc) or attempt + 1 >= max_attempts:
                raise
            last_exc = exc
            retry_after = _retry_after_seconds(exc)
            sleep_s = retry_after if retry_after is not None else base_sleep * (2 ** attempt)
            time.sleep(min(max(sleep_s, 0.1), max_sleep))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("LangMem transient retry loop exited without result")


def _is_transient_provider_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if response_status in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    return any(
        marker in name or marker in message
        for marker in (
            "apitimerouterror",
            "apitimeouterror",
            "connecttimeout",
            "readtimeout",
            "ratelimit",
            "rate limit",
            "rpm limit",
            "tpm limit",
            "too many requests",
            "temporarily unavailable",
            "timeout",
            "timed out",
            "connection reset",
            "service unavailable",
        )
    )


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_env(name: str, value: int | None, *, default: int) -> int:
    if value is not None:
        return int(value)
    raw = os.getenv(name)
    if raw:
        return int(raw)
    return default


def _float_env(name: str, value: float | None, *, default: float) -> float:
    if value is not None:
        return float(value)
    raw = os.getenv(name)
    if raw:
        return float(raw)
    return default
