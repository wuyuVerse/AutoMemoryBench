"""Official managed-service Zep/Graphiti client factory for AutoMemoryBench integration runs."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from amb.clients.core.common import effective_api_key as resolve_api_key, ensure_site_packages_from_venv


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ZEP_GRAPHITI_VENV = PROJECT_ROOT / ".venv-zep-graphiti"
_DEFAULT_MODULE_CANDIDATES = ("zep_cloud", "graphiti_core", "graphiti", "zep")
_DEFAULT_FACTORY_NAMES = ("create_client", "Client", "Graphiti", "Zep", "ZepClient")


class ZepGraphitiRealClient:
    """Thin adapter exposing episode-style graph memory methods to AutoMemoryBench."""

    def __init__(
        self,
        backend: Any,
        *,
        default_limit: int = 5,
        transient_max_attempts: int = 6,
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

    def add_episode(
        self,
        content: Any = None,
        *,
        messages: Any = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        payload = _normalize_payload(content=content, messages=messages)
        resolved_user_id = _resolve_user_id(user_id=user_id, metadata=metadata)
        if _is_graphiti_core_backend(self.backend):
            _prepare_graphiti_backend(self.backend, resolved_user_id)
            payload_text = _payload_text(payload)
            if _graphiti_episode_only_mode(self.backend):
                return _graphiti_store_episode_only(
                    self.backend,
                    payload_text=payload_text,
                    user_id=resolved_user_id,
                    metadata=metadata,
                )
            try:
                result = self._call_with_transient_retry(
                    lambda: _coerce_sync(
                        self.backend.add_episode(
                            name=_graphiti_episode_name(resolved_user_id, metadata),
                            episode_body=payload_text,
                            source_description=_source_description(metadata),
                            reference_time=_reference_time(metadata),
                            group_id=resolved_user_id,
                        )
                    )
                )
            except RuntimeError as exc:
                if _should_use_graphiti_episode_only_fallback(self.backend, exc):
                    return _graphiti_store_episode_only(
                        self.backend,
                        payload_text=payload_text,
                        user_id=resolved_user_id,
                        metadata=metadata,
                    )
                raise
            return _normalize_graphiti_add_result(result, payload_text)
        if _is_zep_cloud_backend(self.backend):
            payload_text = _payload_text(payload)
            result = self._call_with_transient_retry(
                lambda: self.backend.graph.add(
                    data=payload_text,
                    type="text",
                    created_at=_reference_time(metadata).isoformat(),
                    metadata=dict(metadata or {}),
                    source_description=_source_description(metadata),
                    user_id=resolved_user_id,
                )
            )
            return _normalize_zep_episode(result)
        method = _resolve_method(self.backend, "add_episode", "add", "add_memory", "save")
        return _call_variants(
            (
                lambda: method(payload, user_id=resolved_user_id, metadata=dict(metadata or {})),
                lambda: method(content=payload, user_id=resolved_user_id, metadata=dict(metadata or {})),
                lambda: method(text=payload, user_id=resolved_user_id, metadata=dict(metadata or {})),
                lambda: method(episode=payload, user_id=resolved_user_id, metadata=dict(metadata or {})),
                lambda: method(payload, metadata=dict(metadata or {})),
                lambda: method(payload),
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
            raise ValueError("query is required for Zep/Graphiti search")
        if not user_id:
            raise ValueError("user_id is required for Zep/Graphiti search")
        k = int(limit or top_k or self.default_limit)
        if _is_graphiti_core_backend(self.backend):
            _prepare_graphiti_backend(self.backend, str(user_id))
            if _graphiti_episode_only_mode(self.backend):
                episodes = self.get_episodes(user_id=str(user_id), limit=max(20, k * 4))
                return {"data": _rank_episode_hits(str(query), episodes, k)}
            try:
                result = self._call_with_transient_retry(
                    lambda: _coerce_sync(self.backend.search(str(query), group_ids=[str(user_id)], num_results=k))
                )
            except RuntimeError as exc:
                if _is_graphiti_missing_index_error(exc):
                    episodes = self.get_episodes(user_id=str(user_id), limit=max(20, k * 4))
                    return {"data": _rank_episode_hits(str(query), episodes, k)}
                raise
            return {"data": [_graphiti_edge_to_dict(item) for item in result]}
        if _is_zep_cloud_backend(self.backend):
            result = self._call_with_transient_retry(
                lambda: self.backend.graph.search(query=str(query), limit=k, scope="episodes", user_id=str(user_id))
            )
            return {"data": _normalize_zep_search_results(result)}
        method = _resolve_method(self.backend, "search", "search_memory", "search_episodes", "retrieve", "query")
        return _call_variants(
            (
                lambda: method(query=str(query), user_id=str(user_id), limit=k),
                lambda: method(query=str(query), user_id=str(user_id), top_k=k),
                lambda: method(str(query), user_id=str(user_id), limit=k),
                lambda: method(str(query), user_id=str(user_id), top_k=k),
                lambda: method(str(query), k),
                lambda: method(str(query)),
            )
        )

    def get_episodes(self, *, user_id: str | None = None, limit: int = 100, **_: Any) -> list[dict[str, Any]] | dict[str, Any]:
        if not user_id:
            raise ValueError("user_id is required for Zep/Graphiti get_episodes")
        if _is_graphiti_core_backend(self.backend):
            _prepare_graphiti_backend(self.backend, str(user_id))
            result = self._call_with_transient_retry(
                lambda: _coerce_sync(
                    self.backend.retrieve_episodes(
                        reference_time=datetime.now(UTC),
                        last_n=int(limit),
                        group_ids=[str(user_id)],
                    )
                )
            )
            return [_graphiti_episode_to_dict(item) for item in result]
        if _is_zep_cloud_backend(self.backend):
            result = self._call_with_transient_retry(
                lambda: self.backend.graph.episode.get_by_user_id(str(user_id), lastn=int(limit))
            )
            return _normalize_zep_episode_collection(result)
        method = _resolve_method(self.backend, "get_episodes", "get_all", "list", "export_memory")
        return _call_variants(
            (
                lambda: method(user_id=str(user_id), limit=int(limit)),
                lambda: method(user_id=str(user_id)),
                lambda: method(str(user_id)),
                lambda: method(),
            )
        )

    def delete_episode(self, memory_id: str | None = None, **_: Any) -> dict[str, Any]:
        if not memory_id:
            raise ValueError("memory_id is required for Zep/Graphiti delete_episode")
        if _is_graphiti_core_backend(self.backend):
            if hasattr(self.backend, "driver") and getattr(self.backend.driver, "_database", None) is None:
                setattr(self.backend.driver, "_database", "")
            self._call_with_transient_retry(lambda: _coerce_sync(self.backend.remove_episode(str(memory_id))))
            return {"ok": True, "uuid": str(memory_id)}
        if _is_zep_cloud_backend(self.backend):
            result = self._call_with_transient_retry(lambda: self.backend.graph.episode.delete(str(memory_id)))
            return {"ok": True, "uuid": str(memory_id), "message": getattr(result, "message", None)}
        method = _resolve_method(self.backend, "delete_episode", "delete", "remove", "delete_memory")
        return _call_variants(
            (
                lambda: method(memory_id=str(memory_id)),
                lambda: method(episode_id=str(memory_id)),
                lambda: method(str(memory_id)),
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
    module_name: str | None = None,
    venv_root: str | None = None,
    default_limit: int = 5,
    api_key: str | None = None,
    api_key_env: str = "ZEP_API_KEY",
    base_url: str | None = None,
    embedding_base_url: str | None = None,      # default: same endpoint as the chat LLM
    embedding_api_key_env: str | None = None,   # default: same key as the chat LLM
    transient_max_attempts: int | None = None,
    transient_base_sleep: float | None = None,
    transient_max_sleep: float | None = None,
    llm_timeout: float | None = None,
    embedding_timeout: float | None = None,
    openai_max_retries: int | None = None,
    **factory_kwargs: Any,
) -> ZepGraphitiRealClient:
    """Create a real Zep/Graphiti-backed client or wrap a supplied backend."""

    if backend is None:
        ensure_site_packages_from_venv(Path(venv_root) if venv_root else DEFAULT_ZEP_GRAPHITI_VENV)
        resolved_api_key = resolve_api_key(
            api_key=api_key,
            api_key_env=api_key_env,
            fallback_envs=("GRAPHITI_API_KEY",),
        )
        if resolved_api_key:
            factory_kwargs.setdefault("api_key", resolved_api_key)
        if base_url and "base_url" not in factory_kwargs:
            factory_kwargs["base_url"] = base_url
        # The embedder may live on a different endpoint/key than the chat LLM
        # (e.g. chat and embeddings on your OpenAI-compatible endpoint(s) bge-m3). Default to the
        # LLM's endpoint/key when these are not supplied.
        resolved_embedding_key = (
            resolve_api_key(
                api_key=None,
                api_key_env=embedding_api_key_env,
                fallback_envs=(api_key_env, "OPENAI_API_KEY"),
            )
            if embedding_api_key_env
            else resolved_api_key
        )
        if embedding_base_url and "embedding_base_url" not in factory_kwargs:
            factory_kwargs["embedding_base_url"] = embedding_base_url
        if resolved_embedding_key and "embedding_api_key" not in factory_kwargs:
            factory_kwargs["embedding_api_key"] = resolved_embedding_key
        if backend_factory:
            module_ref, _, attr_name = str(backend_factory).partition(":")
            module = importlib.import_module(module_ref)
            factory = getattr(module, attr_name)
            backend = _invoke_factory(factory, factory_kwargs)
        else:
            if llm_timeout is not None:
                factory_kwargs.setdefault("llm_timeout", llm_timeout)
            if embedding_timeout is not None:
                factory_kwargs.setdefault("embedding_timeout", embedding_timeout)
            if openai_max_retries is not None:
                factory_kwargs.setdefault("openai_max_retries", openai_max_retries)
            backend = _create_default_backend(module_name=module_name, factory_kwargs=factory_kwargs)
    return ZepGraphitiRealClient(
        backend,
        default_limit=default_limit,
        transient_max_attempts=_int_env(
            "AMB_ZEP_GRAPHITI_TRANSIENT_MAX_ATTEMPTS",
            transient_max_attempts,
            default=10,
        ),
        transient_base_sleep=_float_env(
            "AMB_ZEP_GRAPHITI_TRANSIENT_BASE_SLEEP",
            transient_base_sleep,
            default=2.0,
        ),
        transient_max_sleep=_float_env(
            "AMB_ZEP_GRAPHITI_TRANSIENT_MAX_SLEEP",
            transient_max_sleep,
            default=120.0,
        ),
    )


def _explicit_module_is_importable(module_name: str | None) -> bool:
    if not module_name:
        return False
    if module_name in sys.modules:
        return True
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        return False
    return True


def _create_default_backend(*, module_name: str | None, factory_kwargs: dict[str, Any]) -> Any:
    module_candidates = [module_name] if module_name else list(_DEFAULT_MODULE_CANDIDATES)
    last_error: Exception | None = None
    for candidate in module_candidates:
        if candidate == "graphiti_core":
            try:
                return _create_graphiti_core_backend(dict(factory_kwargs))
            except (ModuleNotFoundError, TypeError, ValueError) as exc:
                last_error = exc
                continue
            except RuntimeError as exc:
                if module_name:
                    raise
                last_error = exc
                continue
        try:
            module = importlib.import_module(candidate)
        except ModuleNotFoundError as exc:
            last_error = exc
            continue
        for attr_name in _DEFAULT_FACTORY_NAMES:
            factory = getattr(module, attr_name, None)
            if factory is None:
                continue
            if callable(factory):
                try:
                    return _invoke_factory(factory, factory_kwargs)
                except (TypeError, ValueError) as exc:
                    last_error = exc
                    continue
            return factory
    if last_error is not None:
        raise ModuleNotFoundError(
            "required Zep/Graphiti backend is not importable; install one of "
            f"{', '.join(_DEFAULT_MODULE_CANDIDATES)} into the configured venv first"
        ) from last_error
    raise AttributeError(
        "no supported backend factory found; expected one of "
        f"{', '.join(_DEFAULT_FACTORY_NAMES)} on modules {', '.join(module_candidates)}"
    )


def _normalize_payload(*, content: Any, messages: Any) -> Any:
    if messages is not None:
        return messages
    if isinstance(content, (str, list, dict)):
        return content
    if content is None:
        raise ValueError("content or messages is required")
    return str(content)


def _resolve_user_id(*, user_id: str | None, metadata: dict[str, Any] | None) -> str:
    """Resolve the benchmark-scoped memory namespace without using labels."""

    if user_id:
        return str(user_id)
    if isinstance(metadata, dict):
        for key in ("case_id", "session_id"):
            value = metadata.get(key)
            if value:
                return str(value)
    return "amb_default_user"


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
            return _coerce_sync(attempt())
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("no call variants supplied")


def _coerce_sync(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    return asyncio.run(value)


def _call_with_transient_retry(fn, *, max_attempts: int, base_sleep: float, max_sleep: float = 60.0) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - SDKs expose different transient exception classes.
            last_error = exc
            if attempt >= max_attempts or not _is_transient_exception(exc):
                raise
            retry_after = _retry_after_seconds(exc)
            sleep_s = retry_after if retry_after is not None else base_sleep * (2 ** (attempt - 1))
            time.sleep(min(max(sleep_s, 0.1), max_sleep))
    if last_error is not None:
        raise last_error
    raise RuntimeError("transient retry called with no attempts")


def _is_transient_exception(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if response_status in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    names = {cls.__name__.lower() for cls in type(exc).mro()}
    if any(
        token in name
        for name in names
        for token in ("timeout", "connection", "rate", "serviceunavailable", "internalserver")
    ):
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "timed out",
            "timeout",
            "temporarily unavailable",
            "rate limit",
            "ratelimit",
            "too many requests",
            "tpm limit",
            "rpm limit",
            "connection reset",
            "connection aborted",
            "502",
            "503",
            "504",
        )
    )


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    if value is None:
        value = headers.get("Retry-After")
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


def _invoke_factory(factory: Any, factory_kwargs: dict[str, Any]) -> Any:
    if not callable(factory):
        return factory
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(**factory_kwargs)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return factory(**factory_kwargs)
    filtered_kwargs = {
        key: value
        for key, value in factory_kwargs.items()
        if key in signature.parameters
    }
    return factory(**filtered_kwargs)


def _is_graphiti_core_backend(backend: Any) -> bool:
    module_name = getattr(type(backend), "__module__", "")
    return "graphiti_core" in module_name or getattr(type(backend), "__name__", "") == "Graphiti"


def _is_zep_cloud_backend(backend: Any) -> bool:
    graph = getattr(backend, "graph", None)
    episode = getattr(graph, "episode", None)
    return (
        graph is not None
        and callable(getattr(graph, "add", None))
        and callable(getattr(graph, "search", None))
        and callable(getattr(episode, "get_by_user_id", None))
        and callable(getattr(episode, "delete", None))
    )


def _payload_text(payload: Any) -> str:
    return payload if isinstance(payload, str) else str(payload)


def _reference_time(metadata: dict[str, Any] | None) -> datetime:
    raw = None if not isinstance(metadata, dict) else metadata.get("timestamp")
    if isinstance(raw, str) and raw:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        except ValueError:
            pass
    return datetime.now(UTC)


def _graphiti_episode_name(user_id: str, metadata: dict[str, Any] | None) -> str:
    if isinstance(metadata, dict):
        turn_id = metadata.get("turn_id")
        if turn_id:
            return f"{user_id}:{turn_id}"
    return f"{user_id}:episode"


def _source_description(metadata: dict[str, Any] | None) -> str:
    if not isinstance(metadata, dict):
        return "AutoMemoryBench interaction"
    role = metadata.get("role")
    domain = metadata.get("domain")
    if role and domain:
        return f"AutoMemoryBench {domain} {role} turn"
    if role:
        return f"AutoMemoryBench {role} turn"
    return "AutoMemoryBench interaction"


def _normalize_graphiti_add_result(result: Any, payload_text: str) -> dict[str, Any]:
    episode = getattr(result, "episode", None)
    if episode is None and isinstance(result, dict):
        episode = result.get("episode")
    if episode is None:
        return {"summary": payload_text}
    normalized = _graphiti_episode_to_dict(episode)
    return {"uuid": normalized["uuid"], "summary": normalized["summary"], "metadata": normalized.get("metadata")}


def _graphiti_edge_to_dict(item: Any) -> dict[str, Any]:
    return {
        "uuid": str(getattr(item, "uuid", "")),
        "summary": str(getattr(item, "fact", None) or getattr(item, "name", None) or item),
        "metadata": {
            "group_id": getattr(item, "group_id", None),
            "episodes": [str(value) for value in (getattr(item, "episodes", None) or [])],
        },
    }


def _graphiti_episode_to_dict(item: Any) -> dict[str, Any]:
    return {
        "uuid": str(getattr(item, "uuid", "")),
        "summary": str(getattr(item, "content", None) or getattr(item, "name", None) or item),
        "metadata": {
            "group_id": getattr(item, "group_id", None),
            "source_description": getattr(item, "source_description", None),
        },
    }


def _normalize_zep_episode(item: Any) -> dict[str, Any]:
    return {
        "uuid": str(getattr(item, "uuid_", None) or getattr(item, "uuid", None) or ""),
        "summary": str(getattr(item, "content", None) or getattr(item, "summary", None) or item),
        "metadata": getattr(item, "metadata", None) if isinstance(getattr(item, "metadata", None), dict) else None,
    }


def _normalize_zep_episode_collection(result: Any) -> list[dict[str, Any]]:
    items = getattr(result, "episodes", result)
    if not isinstance(items, list):
        items = [items]
    return [_normalize_zep_episode(item) for item in items]


def _normalize_zep_search_results(result: Any) -> list[dict[str, Any]]:
    for field in ("episodes", "edges", "observations", "nodes", "context"):
        items = getattr(result, field, None)
        if not items:
            continue
        if not isinstance(items, list):
            items = [items]
        if field == "edges":
            return [_graphiti_edge_to_dict(item) for item in items]
        return [_normalize_zep_episode(item) for item in items]
    return []


def _create_graphiti_core_backend(factory_kwargs: dict[str, Any]) -> Any:
    module = importlib.import_module("graphiti_core")
    graphiti_cls = getattr(module, "Graphiti")
    driver_kind = str(factory_kwargs.pop("graph_driver_kind", factory_kwargs.pop("driver_kind", "neo4j"))).lower()
    llm_client_mode = str(factory_kwargs.pop("llm_client_mode", "openai")).lower()
    llm_model = factory_kwargs.pop("llm_model", factory_kwargs.pop("model", None))
    small_model = factory_kwargs.pop("small_model", llm_model)
    reranker_model = factory_kwargs.pop("reranker_model", llm_model)
    structured_response_format = str(
        factory_kwargs.pop("structured_response_format", "json_schema")
    ).lower()
    llm_max_tokens = int(factory_kwargs.pop("llm_max_tokens", 16384))
    llm_timeout = float(factory_kwargs.pop("llm_timeout", os.getenv("AMB_ZEP_GRAPHITI_LLM_TIMEOUT", "180")))
    embedding_timeout = float(factory_kwargs.pop("embedding_timeout", os.getenv("AMB_ZEP_GRAPHITI_EMBEDDING_TIMEOUT", "120")))
    connect_timeout = float(factory_kwargs.pop("connect_timeout", os.getenv("AMB_ZEP_GRAPHITI_CONNECT_TIMEOUT", "60")))
    openai_max_retries = int(factory_kwargs.pop("openai_max_retries", os.getenv("AMB_ZEP_GRAPHITI_OPENAI_MAX_RETRIES", "8")))
    embedding_model = factory_kwargs.pop("embedding_model", None)
    embedding_dim = factory_kwargs.pop("embedding_dim", factory_kwargs.pop("embedding_dims", None))
    store_raw_episode_content = bool(factory_kwargs.pop("store_raw_episode_content", True))
    bootstrap_indices = bool(factory_kwargs.pop("bootstrap_indices", False))
    episode_only_fallback = bool(factory_kwargs.pop("episode_only_fallback", False))
    api_key = factory_kwargs.pop("api_key", None) or os.getenv("OPENAI_API_KEY")
    base_url = factory_kwargs.pop("base_url", None) or os.getenv("OPENAI_BASE_URL")
    embedding_base_url = factory_kwargs.pop("embedding_base_url", None)
    embedding_api_key = factory_kwargs.pop("embedding_api_key", None)

    driver = None
    backend_kwargs: dict[str, Any] = {
        "store_raw_episode_content": store_raw_episode_content,
    }
    if driver_kind == "kuzu":
        kuzu_module = importlib.import_module("graphiti_core.driver.kuzu_driver")
        kuzu_driver_cls = getattr(kuzu_module, "KuzuDriver")
        database = str(factory_kwargs.pop("database", factory_kwargs.pop("group_database", "")))
        db_path = str(
            os.getenv("AMB_ZEP_GRAPHITI_DB")
            or os.getenv("AMB_ZEP_GRAPHITI_GRAPH_STORE_PATH")
            or factory_kwargs.pop("graph_store_path", factory_kwargs.pop("db", ":memory:"))
        )
        if db_path != ":memory:":
            if _bool_env("AMB_ZEP_GRAPHITI_CLEAN_DB_ON_START"):
                _remove_kuzu_db_path(Path(db_path))
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        driver = kuzu_driver_cls(db=db_path)
        setattr(driver, "_database", database)
        driver.clone = lambda database_name: driver.with_database(str(database_name))
        backend_kwargs["graph_driver"] = driver
    else:
        if "uri" not in factory_kwargs:
            raise ValueError("graphiti_core non-kuzu backend requires uri")
        backend_kwargs["uri"] = factory_kwargs.pop("uri")
        if "user" in factory_kwargs:
            backend_kwargs["user"] = factory_kwargs.pop("user")
        if "password" in factory_kwargs:
            backend_kwargs["password"] = factory_kwargs.pop("password")

    llm_config_module = importlib.import_module("graphiti_core.llm_client.config")
    llm_config_cls = getattr(llm_config_module, "LLMConfig")
    llm_config = llm_config_cls(
        api_key=api_key,
        base_url=base_url,
        model=llm_model,
        small_model=small_model,
        temperature=0,
        max_tokens=llm_max_tokens,
    )
    llm_openai_client = _openai_async_client(
        api_key=api_key,
        base_url=base_url,
        timeout=llm_timeout,
        connect_timeout=connect_timeout,
        max_retries=openai_max_retries,
    )
    if llm_client_mode == "generic":
        llm_client_module = importlib.import_module("graphiti_core.llm_client.openai_generic_client")
        llm_client_cls = getattr(llm_client_module, "OpenAIGenericClient")
        if structured_response_format in {"json_object", "object"}:
            llm_client_cls = _json_object_structured_generic_client_cls(llm_client_cls)
        elif structured_response_format not in {"json_schema", "schema"}:
            raise ValueError(
                "structured_response_format must be one of: json_schema, json_object"
            )
        llm_client = llm_client_cls(config=llm_config, client=llm_openai_client, max_tokens=llm_max_tokens)
    else:
        llm_client_module = importlib.import_module("graphiti_core.llm_client.openai_client")
        llm_client_cls = getattr(llm_client_module, "OpenAIClient")
        llm_client = llm_client_cls(config=llm_config, client=llm_openai_client, max_tokens=llm_max_tokens)

    embedder_module = importlib.import_module("graphiti_core.embedder.openai")
    embedder_cls = getattr(embedder_module, "OpenAIEmbedder")
    embedder_config_cls = getattr(embedder_module, "OpenAIEmbedderConfig")
    embedder_kwargs: dict[str, Any] = {
        "api_key": embedding_api_key or api_key,
        "base_url": embedding_base_url or base_url,
    }
    if embedding_model is not None:
        embedder_kwargs["embedding_model"] = embedding_model
    if embedding_dim is not None:
        embedder_kwargs["embedding_dim"] = int(embedding_dim)
    embedder_openai_client = _openai_async_client(
        api_key=embedder_kwargs.get("api_key"),
        base_url=embedder_kwargs.get("base_url"),
        timeout=embedding_timeout,
        connect_timeout=connect_timeout,
        max_retries=openai_max_retries,
    )
    embedder = embedder_cls(config=embedder_config_cls(**embedder_kwargs), client=embedder_openai_client)

    reranker_config = llm_config_cls(
        api_key=api_key,
        base_url=base_url,
        model=reranker_model,
        small_model=reranker_model,
        temperature=0,
        max_tokens=llm_max_tokens,
    )
    reranker_module = importlib.import_module("graphiti_core.cross_encoder.openai_reranker_client")
    reranker_cls = getattr(reranker_module, "OpenAIRerankerClient")
    reranker = reranker_cls(config=reranker_config, client=llm_openai_client)

    backend = graphiti_cls(
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=reranker,
        **backend_kwargs,
    )
    setattr(
        backend,
        "_amst_graphiti_options",
        {
            "driver_kind": driver_kind,
            "episode_only_fallback": episode_only_fallback,
            "bootstrap_indices": bootstrap_indices,
            "structured_response_format": structured_response_format,
        },
    )
    setattr(backend, "_amst_graphiti_bootstrapped_groups", set())
    if bootstrap_indices:
        _bootstrap_graphiti_indices(backend)
    return backend


def _json_object_structured_generic_client_cls(base_cls: Any) -> Any:
    """Graphiti generic client variant for OpenAI-compatible APIs without json_schema support."""

    class JsonObjectStructuredOpenAIGenericClient(base_cls):  # type: ignore[misc, valid-type]
        async def _generate_response(
            self,
            messages: list[Any],
            response_model: type[Any] | None = None,
            max_tokens: int | None = None,
            model_size: Any = None,
        ) -> dict[str, Any]:
            openai_module = importlib.import_module("openai")
            errors_module = importlib.import_module("graphiti_core.llm_client.errors")
            rate_limit_error_cls = getattr(errors_module, "RateLimitError")

            openai_messages = []
            for message in messages:
                message.content = self._clean_input(message.content)
                if message.role == "user":
                    openai_messages.append({"role": "user", "content": message.content})
                elif message.role == "system":
                    openai_messages.append({"role": "system", "content": message.content})

            if response_model is not None:
                schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
                schema_instruction = (
                    "\n\nRespond only with one valid JSON object matching this JSON schema. "
                    "Include every required field. If a required array has no entries, use an empty array. "
                    "Do not include markdown fences, explanations, or partial JSON.\n"
                    f"{schema}"
                )
                target_idx = next(
                    (idx for idx in range(len(openai_messages) - 1, -1, -1) if openai_messages[idx]["role"] == "user"),
                    len(openai_messages) - 1,
                )
                if target_idx >= 0:
                    openai_messages[target_idx]["content"] += schema_instruction

            try:
                attempts = 2 if response_model is not None else 1
                messages_for_call = list(openai_messages)
                last_validation_error: Exception | None = None
                for attempt in range(attempts):
                    response = await self.client.chat.completions.create(
                        model=self.model or "gpt-4.1-mini",
                        messages=messages_for_call,
                        temperature=self.temperature,
                        max_tokens=max_tokens or self.max_tokens,
                        response_format={"type": "json_object"},
                    )
                    raw_content = response.choices[0].message.content or "{}"
                    parsed = json.loads(_extract_json_object(raw_content))
                    if response_model is None:
                        return parsed
                    try:
                        return response_model.model_validate(parsed).model_dump()
                    except Exception as exc:  # noqa: BLE001 - pydantic versions expose different validation errors.
                        last_validation_error = exc
                        if attempt + 1 >= attempts:
                            raise
                        messages_for_call = list(openai_messages)
                        messages_for_call.append(
                            {
                                "role": "user",
                                "content": (
                                    "The previous JSON did not validate against the required schema. "
                                    f"Validation error: {exc}. Previous JSON: "
                                    f"{json.dumps(parsed, ensure_ascii=False)[:4000]}. "
                                    "Return only a corrected JSON object that satisfies the schema."
                                ),
                            }
                        )
                if last_validation_error is not None:
                    raise last_validation_error
                raise RuntimeError("structured response generation failed without a validation result")
            except openai_module.RateLimitError as exc:
                raise rate_limit_error_cls from exc
            except Exception as exc:
                logger = importlib.import_module("logging").getLogger(__name__)
                logger.error(f"Error in generating LLM response: {exc}")
                raise

    JsonObjectStructuredOpenAIGenericClient.__name__ = "JsonObjectStructuredOpenAIGenericClient"
    return JsonObjectStructuredOpenAIGenericClient


def _extract_json_object(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].rstrip().endswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if text.startswith("{") or text.startswith("["):
        return text
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return text[index : index + end]
    return text


def _openai_async_client(
    *,
    api_key: str | None,
    base_url: str | None,
    timeout: float,
    connect_timeout: float,
    max_retries: int,
) -> Any:
    openai_module = importlib.import_module("openai")
    httpx_module = importlib.import_module("httpx")
    async_openai_cls = getattr(openai_module, "AsyncOpenAI")
    return async_openai_cls(
        api_key=api_key,
        base_url=base_url,
        timeout=httpx_module.Timeout(timeout, connect=connect_timeout),
        max_retries=max_retries,
    )


def _prepare_graphiti_backend(backend: Any, user_id: str) -> None:
    driver = getattr(backend, "driver", None)
    if driver is None:
        return
    if getattr(driver, "_database", None) in (None, ""):
        setattr(driver, "_database", str(user_id))
    if getattr(type(driver), "__name__", "") == "KuzuDriver":
        driver.clone = lambda database_name: driver.with_database(str(database_name))
    options = getattr(backend, "_amst_graphiti_options", {})
    if bool(options.get("bootstrap_indices")):
        if getattr(type(driver), "__name__", "") == "KuzuDriver":
            if bool(getattr(driver, "_amst_graphiti_kuzu_indices_bootstrapped", False)):
                setattr(backend, "_amst_graphiti_kuzu_indices_bootstrapped", True)
                return
            if not bool(getattr(backend, "_amst_graphiti_kuzu_indices_bootstrapped", False)):
                _bootstrap_graphiti_indices(backend)
            return
        bootstrapped_groups = getattr(backend, "_amst_graphiti_bootstrapped_groups", None)
        if not isinstance(bootstrapped_groups, set):
            bootstrapped_groups = set()
            setattr(backend, "_amst_graphiti_bootstrapped_groups", bootstrapped_groups)
        group_key = str(user_id)
        if group_key not in bootstrapped_groups:
            _bootstrap_graphiti_indices(backend)
            bootstrapped_groups.add(group_key)


def _bootstrap_graphiti_indices(backend: Any) -> None:
    driver = getattr(backend, "driver", None)
    if driver is not None and getattr(type(driver), "__name__", "") == "KuzuDriver":
        setup_schema = getattr(driver, "setup_schema", None)
        if callable(setup_schema):
            setup_schema()
        _bootstrap_kuzu_fts_indices(driver)
        setattr(backend, "_amst_graphiti_kuzu_indices_bootstrapped", True)
        setattr(driver, "_amst_graphiti_kuzu_indices_bootstrapped", True)
        return
    _coerce_sync(backend.build_indices_and_constraints())


def _bootstrap_kuzu_fts_indices(driver: Any) -> None:
    graph_queries = importlib.import_module("graphiti_core.graph_queries")
    driver_module = importlib.import_module("graphiti_core.driver.driver")
    graph_provider = getattr(driver_module, "GraphProvider")
    get_fulltext_indices = getattr(graph_queries, "get_fulltext_indices")
    for query in get_fulltext_indices(graph_provider.KUZU):
        try:
            _coerce_sync(driver.execute_query(query))
        except Exception as exc:  # noqa: BLE001 - Kuzu exposes backend-specific exception classes.
            if not _is_kuzu_index_already_exists_error(exc):
                raise


def _is_kuzu_index_already_exists_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "already exists" in message
        or "already exist" in message
        or ("index" in message and "exists in table" in message)
    )


def _remove_kuzu_db_path(path: Path) -> None:
    for candidate in (path, Path(str(path) + ".wal")):
        if candidate.is_dir():
            shutil.rmtree(candidate)
        elif candidate.exists():
            candidate.unlink()


def _bool_env(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _should_use_graphiti_episode_only_fallback(backend: Any, exc: RuntimeError) -> bool:
    return _graphiti_episode_only_mode(backend) and _is_graphiti_missing_index_error(exc)


def _is_graphiti_missing_index_error(exc: RuntimeError) -> bool:
    return "doesn't have an index with name edge_name_and_fact" in str(exc)


def _graphiti_store_episode_only(
    backend: Any,
    *,
    payload_text: str,
    user_id: str,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    nodes_module = importlib.import_module("graphiti_core.nodes")
    episodic_node_cls = getattr(nodes_module, "EpisodicNode")
    episode_type = getattr(nodes_module, "EpisodeType").message
    now = datetime.now(UTC)
    episode = episodic_node_cls(
        name=_graphiti_episode_name(user_id, metadata),
        group_id=user_id,
        labels=[],
        source=episode_type,
        content=payload_text,
        source_description=_source_description(metadata),
        created_at=now,
        valid_at=_reference_time(metadata),
    )
    _coerce_sync(episode.save(getattr(backend, "driver")))
    return {
        "uuid": str(getattr(episode, "uuid", "")),
        "summary": payload_text,
        "metadata": {
            "group_id": user_id,
            "source_description": _source_description(metadata),
            "fallback_mode": "episode_only",
        },
    }


def _rank_episode_hits(query: str, episodes: list[dict[str, Any]] | dict[str, Any], limit: int) -> list[dict[str, Any]]:
    items = episodes if isinstance(episodes, list) else [episodes]
    ranked = []
    for item in items:
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary", ""))
        score = _text_match_score(query, summary)
        ranked.append(
            {
                "uuid": str(item.get("uuid", "")),
                "summary": summary,
                "metadata": dict(item.get("metadata") or {}),
                "score": score,
            }
        )
    ranked.sort(key=lambda item: (-float(item.get("score") or 0.0), item.get("uuid", "")))
    return ranked[:limit]


def _text_match_score(query: str, content: str) -> float:
    query_terms = {term for term in query.lower().split() if term}
    content_terms = {term for term in content.lower().split() if term}
    if not query_terms:
        return 0.0
    overlap = len(query_terms & content_terms) / float(len(query_terms))
    substring_bonus = 0.5 if query.lower() in content.lower() else 0.0
    return overlap + substring_bonus


def _graphiti_episode_only_mode(backend: Any) -> bool:
    options = getattr(backend, "_amst_graphiti_options", {})
    return bool(options.get("episode_only_fallback"))
