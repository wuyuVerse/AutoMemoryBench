"""Official framework-SDK Letta/MemGPT client factory for AutoMemoryBench integration runs."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib
import inspect
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from amb.clients.core.common import effective_api_key as resolve_api_key, ensure_site_packages_from_venv, import_or_raise


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LETTA_VENV = PROJECT_ROOT / ".venv-letta"
_DEFAULT_MODULE_CANDIDATES = ("letta_client", "letta", "memgpt")
_DEFAULT_FACTORY_NAMES = ("create_client", "Client", "Letta", "LocalClient")


class LettaRealClient:
    """Thin adapter exposing Letta-like memory methods to AMST."""

    def __init__(
        self,
        backend: Any,
        *,
        default_limit: int = 5,
        environment_overrides: dict[str, str | None] | None = None,
    ) -> None:
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        self.backend = backend
        self.default_limit = default_limit
        self._agent_ids: dict[str, Any] = {}
        self._environment_overrides = {
            key: value for key, value in dict(environment_overrides or {}).items() if key
        }

    def reset(self, **_: Any) -> dict[str, Any]:
        return {"ok": True}

    def clear(self) -> dict[str, Any]:
        return {"ok": True}

    def add_memory(
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
            raise ValueError("user_id is required for Letta add_memory")
        if _is_archival_memory_backend(self.backend):
            agent_id = self._ensure_agent_id(str(user_id))
            payload_text = _payload_text(payload)
            result = self._call_with_environment(
                lambda: _call_variants(
                    (
                        lambda: self.backend.insert_archival_memory(agent_id=agent_id, memory=payload_text),
                        lambda: self.backend.insert_archival_memory(agent_id, payload_text),
                    )
                )
            )
            return _normalize_archival_insert_result(result, payload_text)
        if _is_letta_client_resource_backend(self.backend):
            agent_id = self._ensure_agent_id(str(user_id))
            payload_text = _payload_text(payload)
            result = self._call_with_environment(
                lambda: self.backend.agents.passages.create(
                    agent_id=agent_id,
                    text=payload_text,
                )
            )
            return _normalize_archival_insert_result(result, payload_text)
        method = _resolve_method(self.backend, "add_memory", "add", "insert_memory", "save")
        return self._call_with_environment(
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

    def search_memory(
        self,
        query: str | None = None,
        *,
        user_id: str | None = None,
        limit: int | None = None,
        top_k: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not query:
            raise ValueError("query is required for Letta search_memory")
        if not user_id:
            raise ValueError("user_id is required for Letta search_memory")
        k = int(limit or top_k or self.default_limit)
        if _is_archival_memory_backend(self.backend):
            agent_id = self._ensure_agent_id(str(user_id))
            if _supports_memgpt_local_archival_access(self.backend):
                return self._call_with_environment(
                    lambda: _search_memgpt_local_archival_memory(
                        self.backend,
                        agent_id=agent_id,
                        query=str(query),
                        limit=k,
                    )
                )
            items = self._call_with_environment(
                lambda: _normalize_archival_memories(
                    _call_variants(
                        (
                            lambda: self.backend.get_agent_archival_memory(agent_id=agent_id, limit=max(100, k * 10)),
                            lambda: self.backend.get_agent_archival_memory(agent_id, limit=max(100, k * 10)),
                            lambda: self.backend.get_agent_archival_memory(agent_id),
                        )
                    )
                )
            )
            return _rank_archival_hits(str(query), items, k)
        if _is_letta_client_resource_backend(self.backend):
            agent_id = self._ensure_agent_id(str(user_id))
            items = self._call_with_environment(
                lambda: _normalize_archival_memories(
                    self.backend.agents.passages.search(
                        agent_id=agent_id,
                        query=str(query),
                        top_k=k,
                    )
                )
            )
            return _rank_archival_hits(str(query), items, k)
        method = _resolve_method(self.backend, "search_memory", "search", "retrieve", "query")
        return self._call_with_environment(
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

    def list_memories(self, *, user_id: str | None = None, limit: int = 100, **_: Any) -> list[dict[str, Any]] | dict[str, Any]:
        if not user_id:
            raise ValueError("user_id is required for Letta list_memories")
        if _is_archival_memory_backend(self.backend):
            agent_id = self._ensure_agent_id(str(user_id))
            if _supports_memgpt_local_archival_access(self.backend):
                return self._call_with_environment(
                    lambda: _list_memgpt_local_archival_memory(
                        self.backend,
                        agent_id=agent_id,
                        limit=int(limit),
                    )
                )
            return self._call_with_environment(
                lambda: _normalize_archival_memories(
                    _call_variants(
                        (
                            lambda: self.backend.get_agent_archival_memory(agent_id=agent_id, limit=int(limit)),
                            lambda: self.backend.get_agent_archival_memory(agent_id, limit=int(limit)),
                            lambda: self.backend.get_agent_archival_memory(agent_id),
                        )
                    )
                )[: int(limit)]
            )
        if _is_letta_client_resource_backend(self.backend):
            agent_id = self._ensure_agent_id(str(user_id))
            return self._call_with_environment(
                lambda: _normalize_archival_memories(
                    self.backend.agents.passages.list(
                        agent_id=agent_id,
                        limit=int(limit),
                    )
                )
            )[: int(limit)]
        method = _resolve_method(self.backend, "list_memories", "get_all", "export_memory")
        return self._call_with_environment(
            lambda: _call_variants(
                (
                    lambda: method(user_id=str(user_id), limit=int(limit)),
                    lambda: method(user_id=str(user_id)),
                    lambda: method(str(user_id)),
                    lambda: method(),
                )
            )
        )

    def delete_memory(self, memory_id: str | None = None, *, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        if not memory_id:
            raise ValueError("memory_id is required for Letta delete_memory")
        if _is_archival_memory_backend(self.backend):
            agent_id = self._resolve_delete_agent_id(user_id)
            result = self._call_with_environment(
                lambda: _call_variants(
                    (
                        lambda: self.backend.delete_archival_memory(agent_id=agent_id, memory_id=str(memory_id)),
                        lambda: self.backend.delete_archival_memory(agent_id, str(memory_id)),
                    )
                )
            )
            return {"ok": True, "memory_id": str(memory_id), "raw_result": result}
        if _is_letta_client_resource_backend(self.backend):
            agent_id = self._resolve_delete_agent_id(user_id)
            result = self._call_with_environment(
                lambda: self.backend.agents.passages.delete(
                    str(memory_id),
                    agent_id=agent_id,
                )
            )
            return {"ok": True, "memory_id": str(memory_id), "raw_result": result}
        method = _resolve_method(self.backend, "delete_memory", "delete", "remove_memory")
        return self._call_with_environment(
            lambda: _call_variants(
                (
                    lambda: method(memory_id=str(memory_id), user_id=str(user_id)) if user_id else method(memory_id=str(memory_id)),
                    lambda: method(str(memory_id)),
                )
            )
        )

    def _ensure_agent_id(self, user_id: str) -> Any:
        def resolve():
            cached = self._agent_ids.get(user_id)
            if cached:
                return cached
            agent = _lookup_agent(self.backend, user_id)
            if agent is None:
                if _is_letta_client_resource_backend(self.backend):
                    agent = self.backend.agents.create(name=user_id)
                else:
                    create_method = _resolve_method(self.backend, "create_agent")
                    agent = _call_variants(
                        (
                            lambda: create_method(name=user_id),
                            lambda: create_method(agent_name=user_id),
                            lambda: create_method(user_id),
                            lambda: create_method(),
                        )
                    )
            agent_id = _agent_id(agent)
            self._agent_ids[user_id] = agent_id
            return agent_id

        return self._call_with_environment(resolve)

    def _resolve_delete_agent_id(self, user_id: str | None) -> Any:
        if user_id:
            return self._ensure_agent_id(str(user_id))
        if len(self._agent_ids) == 1:
            return next(iter(self._agent_ids.values()))
        raise ValueError("user_id is required for Letta delete_memory when multiple agents are active")

    def _call_with_environment(self, fn):
        if not self._environment_overrides:
            return fn()
        with _temporary_environment(**self._environment_overrides):
            return fn()


def create_client(
    *,
    backend: Any | None = None,
    backend_factory: str | None = None,
    module_name: str | None = None,
    venv_root: str | None = None,
    default_limit: int = 5,
    api_key: str | None = None,
    api_key_env: str = "LETTA_API_KEY",
    base_url: str | None = None,
    bootstrap_memgpt_local: bool = False,
    memgpt_runtime_root: str | None = None,
    memgpt_config_path: str | None = None,
    memgpt_credentials_path: str | None = None,
    memgpt_llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
    memgpt_llm_endpoint_type: str = "openai",
    memgpt_llm_endpoint: str | None = None,
    memgpt_llm_wrapper: str = "chatml",
    memgpt_llm_context_window: int = 32768,
    memgpt_embedding_endpoint_type: str = "openai",
    memgpt_embedding_endpoint: str | None = None,
    memgpt_embedding_model: str = "BAAI/bge-m3",
    memgpt_embedding_dim: int = 1024,
    memgpt_embedding_omit_dimensions: bool = False,
    memgpt_embedding_chunk_size: int = 300,
    embedding_model: str | None = None,         # alias for memgpt_embedding_model (matrix-injected)
    embedding_base_url: str | None = None,      # default: same endpoint as the LLM
    embedding_api_key_env: str | None = None,   # default: same key as the LLM
    disable_chroma_telemetry: bool = True,
    **factory_kwargs: Any,
) -> LettaRealClient:
    """Create a real Letta-backed client or wrap a supplied backend."""

    # The systems x models matrix injects the embedder model as `embedding_model`
    # (alongside `embedding_base_url`/`embedding_api_key_env`). Consume it here as an
    # alias for `memgpt_embedding_model` so it configures the EMBEDDER rather than
    # leaking into **factory_kwargs (which would forward it to the backend SDK
    # constructor that does not expect it).
    if embedding_model:
        memgpt_embedding_model = embedding_model

    runtime_root_path = _resolve_memgpt_runtime_root(
        memgpt_runtime_root=memgpt_runtime_root,
        memgpt_config_path=memgpt_config_path,
        memgpt_credentials_path=memgpt_credentials_path,
    )
    config_path = _resolve_memgpt_path(memgpt_config_path, default=runtime_root_path / "config")
    credentials_path = _resolve_memgpt_path(memgpt_credentials_path, default=runtime_root_path / "credentials")
    telemetry_env_value = "False" if disable_chroma_telemetry else None
    omit_embedding_dimensions_env = "1" if memgpt_embedding_omit_dimensions else None
    # The embedder may live on a different endpoint/key than the chat LLM (e.g.
    # chat and embeddings on your OpenAI-compatible endpoint(s) bge-m3). When unset, the embedder
    # keeps using the LLM's endpoint/key (backward compatible).
    resolved_embedding_key = (
        resolve_api_key(
            api_key=None,
            api_key_env=embedding_api_key_env,
            fallback_envs=(api_key_env, "OPENAI_API_KEY"),
        )
        if embedding_api_key_env
        else None
    )
    embedding_endpoint_value = embedding_base_url or memgpt_embedding_endpoint
    embedding_api_key_env_value = resolved_embedding_key
    if backend is None:
        if venv_root or not backend_factory:
            ensure_site_packages_from_venv(Path(venv_root) if venv_root else DEFAULT_LETTA_VENV)
            _patch_numpy_legacy_aliases()
        # Resolve the CHAT LLM key from the injected `api_key_env` (e.g. OPENAI_API_KEY),
        # falling back to MEMGPT_API_KEY then OPENAI_API_KEY. This key + `base_url`
        # configure the chat/LLM endpoint ONLY; the embedder uses its own
        # embedding_base_url/embedding_api_key_env (see AMST_MEMGPT_EMBEDDING_API_KEY).
        resolved_api_key = resolve_api_key(
            api_key=api_key,
            api_key_env=api_key_env,
            fallback_envs=("MEMGPT_API_KEY", "OPENAI_API_KEY"),
        )
        use_local_memgpt_backend = bool(bootstrap_memgpt_local)
        if bootstrap_memgpt_local:
            _bootstrap_memgpt_local_runtime(
                package_name=module_name or "memgpt",
                runtime_root=runtime_root_path,
                config_path=config_path,
                credentials_path=credentials_path,
                api_key=resolved_api_key,
                llm_model=memgpt_llm_model,
                llm_endpoint_type=memgpt_llm_endpoint_type,
                llm_endpoint=memgpt_llm_endpoint or base_url or "https://api.openai.com/v1",
                llm_wrapper=memgpt_llm_wrapper,
                llm_context_window=memgpt_llm_context_window,
                embedding_endpoint_type=memgpt_embedding_endpoint_type,
                embedding_endpoint=embedding_endpoint_value or base_url or "https://api.openai.com/v1",
                embedding_model=memgpt_embedding_model,
                embedding_dim=memgpt_embedding_dim,
                embedding_chunk_size=memgpt_embedding_chunk_size,
            )
        if resolved_api_key and not use_local_memgpt_backend:
            factory_kwargs.setdefault("token", resolved_api_key)
            factory_kwargs.setdefault("api_key", resolved_api_key)
        if base_url and "base_url" not in factory_kwargs and not use_local_memgpt_backend:
            factory_kwargs["base_url"] = base_url
        with _temporary_environment(
            MEMGPT_CONFIG_PATH=str(config_path),
            MEMGPT_CREDENTIALS_PATH=str(credentials_path),
            ANONYMIZED_TELEMETRY=telemetry_env_value,
            AMST_MEMGPT_OMIT_EMBEDDING_DIMENSIONS=omit_embedding_dimensions_env,
            AMST_MEMGPT_EMBEDDING_API_KEY=embedding_api_key_env_value,
        ):
            if backend_factory:
                factory_module_name, _, attr_name = str(backend_factory).partition(":")
                module = import_or_raise(factory_module_name)
                factory = getattr(module, attr_name)
                backend = _invoke_factory(factory, factory_kwargs)
            else:
                backend = _create_default_backend(module_name=module_name, factory_kwargs=factory_kwargs)
    env_overrides = {
        "HOME": str(runtime_root_path),
        "MEMGPT_CONFIG_PATH": str(config_path),
        "MEMGPT_CREDENTIALS_PATH": str(credentials_path),
        "ANONYMIZED_TELEMETRY": telemetry_env_value,
        "AMST_MEMGPT_OMIT_EMBEDDING_DIMENSIONS": omit_embedding_dimensions_env,
        "AMST_MEMGPT_EMBEDDING_API_KEY": embedding_api_key_env_value,
    }
    return LettaRealClient(
        backend,
        default_limit=default_limit,
        environment_overrides=env_overrides,
    )


def _normalize_payload(*, content: Any, messages: Any) -> Any:
    if messages is not None:
        return messages
    if isinstance(content, (str, list, dict)):
        return content
    if content is None:
        raise ValueError("content or messages is required")
    return str(content)


def _patch_numpy_legacy_aliases() -> None:
    """Keep older MemGPT/llama-index/scipy stacks working under NumPy 2.x."""

    try:
        numpy = importlib.import_module("numpy")
    except Exception:
        return
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "long": int,
        "ulong": int,
    }
    for name, value in aliases.items():
        if not hasattr(numpy, name):
            setattr(numpy, name, value)


def _resolve_memgpt_runtime_root(
    *,
    memgpt_runtime_root: str | None,
    memgpt_config_path: str | None,
    memgpt_credentials_path: str | None,
) -> Path:
    env_runtime_root = os.getenv("AMB_LETTA_RUNTIME_ROOT")
    if env_runtime_root:
        return _resolve_memgpt_path(env_runtime_root)
    if memgpt_runtime_root:
        return _resolve_memgpt_path(memgpt_runtime_root)
    if memgpt_config_path:
        return _resolve_memgpt_path(memgpt_config_path).parent
    if memgpt_credentials_path:
        return _resolve_memgpt_path(memgpt_credentials_path).parent
    return PROJECT_ROOT / ".cache" / "letta_memgpt_local_runtime"


def _resolve_memgpt_path(raw_path: str | Path | None, *, default: Path | None = None) -> Path:
    source = default if raw_path is None else Path(raw_path)
    if source is None:
        raise ValueError("memgpt path resolution requires a source path or default")
    return source if source.is_absolute() else PROJECT_ROOT / source


def _bootstrap_memgpt_local_runtime(
    *,
    package_name: str,
    runtime_root: Path,
    config_path: Path,
    credentials_path: Path,
    api_key: str | None,
    llm_model: str,
    llm_endpoint_type: str,
    llm_endpoint: str,
    llm_wrapper: str,
    llm_context_window: int,
    embedding_endpoint_type: str,
    embedding_endpoint: str,
    embedding_model: str,
    embedding_dim: int,
    embedding_chunk_size: int,
) -> None:
    config_parent = config_path.parent
    credentials_parent = credentials_path.parent
    for path in (runtime_root, config_parent, credentials_parent):
        path.mkdir(parents=True, exist_ok=True)
    metadata_dir = runtime_root / "metadata"
    recall_dir = runtime_root / "recall"
    archival_dir = runtime_root / "chroma"
    for path in (metadata_dir, recall_dir, archival_dir):
        path.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "sqlite.db").touch(exist_ok=True)
    (recall_dir / "sqlite.db").touch(exist_ok=True)

    with _temporary_environment(
        HOME=str(runtime_root),
        MEMGPT_CONFIG_PATH=str(config_path),
        MEMGPT_CREDENTIALS_PATH=str(credentials_path),
    ):
        config_mod = importlib.import_module(f"{package_name}.config")
        credentials_mod = importlib.import_module(f"{package_name}.credentials")
        data_types_mod = importlib.import_module(f"{package_name}.data_types")
        llm_config = data_types_mod.LLMConfig(
            model=llm_model,
            model_endpoint_type=llm_endpoint_type,
            model_endpoint=llm_endpoint,
            model_wrapper=llm_wrapper,
            context_window=int(llm_context_window),
        )
        embedding_config = data_types_mod.EmbeddingConfig(
            embedding_endpoint_type=embedding_endpoint_type,
            embedding_endpoint=embedding_endpoint,
            embedding_model=embedding_model,
            embedding_dim=int(embedding_dim),
            embedding_chunk_size=int(embedding_chunk_size),
        )
        config = config_mod.MemGPTConfig(
            config_path=str(config_path),
            default_llm_config=llm_config,
            default_embedding_config=embedding_config,
            archival_storage_type="chroma",
            archival_storage_path=str(archival_dir),
            recall_storage_type="sqlite",
            recall_storage_path=str(recall_dir),
            metadata_storage_type="sqlite",
            metadata_storage_path=str(metadata_dir),
        )
        config.save()
        credentials = credentials_mod.MemGPTCredentials(
            credentials_path=str(credentials_path),
            openai_key=api_key,
        )
        credentials.save()


@contextlib.contextmanager
def _temporary_environment(**updates: str | None):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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


def _create_default_backend(*, module_name: str | None, factory_kwargs: dict[str, Any]) -> Any:
    module_candidates = [candidate for candidate in (module_name, *_DEFAULT_MODULE_CANDIDATES) if candidate]
    last_error: Exception | None = None
    for candidate in module_candidates:
        try:
            module = importlib.import_module(candidate)
        except ModuleNotFoundError as exc:
            last_error = exc
            continue
        if candidate == "memgpt":
            _patch_memgpt_embedding_model(module)
            _patch_memgpt_sync_server_create_agent(module)
            _patch_memgpt_chroma_telemetry(module)
        for attr_name in _DEFAULT_FACTORY_NAMES:
            factory = getattr(module, attr_name, None)
            if factory is None:
                continue
            if callable(factory):
                try:
                    return _invoke_factory(factory, factory_kwargs)
                except TypeError as exc:
                    last_error = exc
                    continue
            return factory
    if last_error is not None:
        raise ModuleNotFoundError(
            "required Letta/MemGPT backend is not importable; install one of "
            f"{', '.join(_DEFAULT_MODULE_CANDIDATES)} into the configured venv first"
        ) from last_error
    raise AttributeError(
        "no supported Letta/MemGPT backend factory found; expected one of "
        f"{', '.join(_DEFAULT_FACTORY_NAMES)} on modules {', '.join(module_candidates)}"
    )


def _is_archival_memory_backend(backend: Any) -> bool:
    return all(
        callable(getattr(backend, name, None))
        for name in ("insert_archival_memory", "get_agent_archival_memory", "delete_archival_memory")
    )


def _supports_memgpt_local_archival_access(backend: Any) -> bool:
    server = getattr(backend, "server", None)
    user_id = getattr(backend, "user_id", None)
    return (
        server is not None
        and user_id is not None
        and callable(getattr(server, "_get_or_load_agent", None))
        and backend.__class__.__module__.startswith("memgpt")
    )


def _is_letta_client_resource_backend(backend: Any) -> bool:
    agents = getattr(backend, "agents", None)
    passages = getattr(agents, "passages", None)
    return (
        agents is not None
        and passages is not None
        and callable(getattr(agents, "create", None))
        and callable(getattr(agents, "list", None))
        and callable(getattr(passages, "create", None))
        and callable(getattr(passages, "list", None))
        and callable(getattr(passages, "search", None))
        and callable(getattr(passages, "delete", None))
    )


def _lookup_agent(backend: Any, user_id: str) -> Any | None:
    if _is_letta_client_resource_backend(backend):
        list_agents = getattr(getattr(backend, "agents", None), "list", None)
        if callable(list_agents):
            try:
                candidates = list_agents(search=user_id, limit=10)
            except TypeError:
                try:
                    candidates = list_agents(limit=10)
                except Exception:
                    candidates = []
            except Exception:
                candidates = []
            for candidate in _iter_collection_items(candidates):
                name = getattr(candidate, "name", None)
                if name == user_id:
                    return candidate
    get_agent = getattr(backend, "get_agent", None)
    if callable(get_agent):
        try:
            agent = get_agent(agent_name=user_id)
            if agent not in (None, False):
                return agent
        except Exception:
            pass
        try:
            agent = get_agent(user_id)
            if agent not in (None, False):
                return agent
        except Exception:
            pass
    agent_exists = getattr(backend, "agent_exists", None)
    if callable(agent_exists):
        try:
            exists = bool(agent_exists(agent_name=user_id))
        except Exception:
            exists = False
        if exists and callable(get_agent):
            try:
                agent = get_agent(agent_name=user_id)
                if agent not in (None, False):
                    return agent
            except Exception:
                return None
    return None


def _agent_id(agent: Any) -> Any:
    if isinstance(agent, dict):
        for key in ("agent_id", "id"):
            value = agent.get(key)
            if value is not None:
                return value
    for name in ("agent_id", "id"):
        value = getattr(agent, name, None)
        if value is not None:
            return value
    raise ValueError("cannot determine Letta agent id from backend response")


def _normalize_archival_insert_result(result: Any, payload_text: str) -> dict[str, Any]:
    ids = None
    if isinstance(result, dict):
        ids = result.get("ids") or result.get("id")
    else:
        ids = getattr(result, "ids", None) or getattr(result, "id", None)
    memory_id = None
    if isinstance(ids, list) and ids:
        memory_id = ids[0]
    elif ids is not None:
        memory_id = ids
    return {"memory_id": str(memory_id or f"letta:{abs(hash(payload_text))}"), "content": payload_text}


def _normalize_archival_memories(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        items = (
            result.get("archival_memory")
            or result.get("items")
            or result.get("results")
            or result.get("data")
            or result.get("passages")
            or []
        )
    else:
        items = (
            getattr(result, "archival_memory", None)
            or getattr(result, "items", None)
            or getattr(result, "results", None)
            or getattr(result, "data", None)
            or getattr(result, "passages", None)
            or result
        )
    items = _iter_collection_items(items)
    normalized: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            memory_id = item.get("memory_id") or item.get("id") or item.get("passage_id")
            content = item.get("content") or item.get("contents") or item.get("text")
        else:
            memory_id = (
                getattr(item, "memory_id", None)
                or getattr(item, "id", None)
                or getattr(item, "passage_id", None)
            )
            content = (
                getattr(item, "content", None)
                or getattr(item, "contents", None)
                or getattr(item, "text", None)
            )
        normalized.append({"memory_id": str(memory_id), "content": str(content or "")})
    return normalized


def _rank_archival_hits(query: str, items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ranked = []
    for item in items:
        score = _text_match_score(query, str(item.get("content", "")))
        ranked.append({**item, "score": score})
    ranked.sort(key=lambda item: (-float(item.get("score") or 0.0), str(item.get("memory_id", ""))))
    return ranked[:limit]


def _ordered_archival_hits(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if not items:
        return []
    capped = items[:limit]
    total = max(len(capped), 1)
    return [
        {
            **item,
            "score": float(total - index) / float(total),
        }
        for index, item in enumerate(capped)
    ]


def _text_match_score(query: str, content: str) -> float:
    query_terms = {term for term in query.lower().split() if term}
    content_terms = {term for term in content.lower().split() if term}
    if not query_terms:
        return 0.0
    overlap = len(query_terms & content_terms) / float(len(query_terms))
    substring_bonus = 0.5 if query.lower() in content.lower() else 0.0
    return overlap + substring_bonus


def _payload_text(payload: Any) -> str:
    return payload if isinstance(payload, str) else str(payload)


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


def _iter_collection_items(items: Any) -> list[Any]:
    if items is None:
        return []
    if isinstance(items, list):
        return items
    if isinstance(items, tuple):
        return list(items)
    data = getattr(items, "data", None)
    if isinstance(data, list):
        return data
    return [items]


def _memgpt_local_archival_memory(backend: Any, agent_id: Any):
    server = getattr(backend, "server", None)
    user_id = getattr(backend, "user_id", None)
    if server is None or user_id is None:
        raise AttributeError("backend does not expose MemGPT local archival access")
    agent = server._get_or_load_agent(user_id=user_id, agent_id=agent_id)
    persistence_manager = getattr(agent, "persistence_manager", None)
    archival_memory = getattr(persistence_manager, "archival_memory", None)
    if archival_memory is None:
        raise AttributeError("MemGPT local agent has no archival_memory")
    return archival_memory


def _search_memgpt_local_archival_memory(
    backend: Any,
    *,
    agent_id: Any,
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    archival_memory = _memgpt_local_archival_memory(backend, agent_id)
    package_root = backend.__class__.__module__.split(".", 1)[0] or "memgpt"
    embeddings_mod = importlib.import_module(f"{package_root}.embeddings")
    query_embedding = getattr(embeddings_mod, "query_embedding")
    query_vector = query_embedding(archival_memory.embed_model, str(query))
    try:
        records = _query_memgpt_local_storage(archival_memory.storage, str(query), query_vector, int(limit))
        return _ordered_archival_hits(_normalize_archival_memories(records), int(limit))
    except Exception as exc:
        if not _memgpt_query_requires_full_scan_fallback(exc):
            raise
        items = _normalize_archival_memories(_get_all_memgpt_local_storage(archival_memory.storage, None))
        return _rank_archival_hits(str(query), items, int(limit))


def _list_memgpt_local_archival_memory(
    backend: Any,
    *,
    agent_id: Any,
    limit: int,
) -> list[dict[str, Any]]:
    archival_memory = _memgpt_local_archival_memory(backend, agent_id)
    return _normalize_archival_memories(_get_all_memgpt_local_storage(archival_memory.storage, int(limit)))[: int(limit)]


def _query_memgpt_local_storage(storage: Any, query: str, query_vector: Any, limit: int):
    try:
        return storage.query(query, query_vector, top_k=limit)
    except AssertionError:
        if not all(hasattr(storage, name) for name in ("collection", "results_to_records", "include", "get_filters")):
            raise
        _ids, filters = storage.get_filters({})
        raw = storage.collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            include=storage.include,
            where=filters,
        )
        flattened: dict[str, Any] = {}
        for key, value in raw.items():
            if key == "included":
                continue
            if value:
                flattened[key] = _flatten_chroma_query_value(key, value[0])
            else:
                flattened[key] = value
        return storage.results_to_records(flattened)


def _get_all_memgpt_local_storage(storage: Any, limit: int | None):
    try:
        return storage.get_all(limit=limit)
    except ValueError as exc:
        if "Expected IDs to be a non-empty list" not in str(exc):
            raise
        if not all(hasattr(storage, name) for name in ("collection", "results_to_records", "include", "get_filters")):
            raise
        ids, filters = storage.get_filters({})
        request: dict[str, Any] = {
            "include": storage.include,
            "where": filters,
        }
        if limit is not None:
            request["limit"] = limit
        if ids:
            request["ids"] = ids
        raw = storage.collection.get(**request)
        if isinstance(raw, dict) and "included" in raw:
            raw = {key: value for key, value in raw.items() if key != "included"}
        if isinstance(raw, dict):
            raw = {key: _normalize_chroma_get_value(key, value) for key, value in raw.items()}
        return storage.results_to_records(raw)


def _flatten_chroma_query_value(key: str, value: Any) -> Any:
    flattened = value.tolist() if hasattr(value, "tolist") else value
    if key == "embeddings" and isinstance(flattened, list):
        if flattened and not isinstance(flattened[0], (list, tuple)):
            return [flattened]
    return flattened


def _normalize_chroma_get_value(key: str, value: Any) -> Any:
    normalized = value.tolist() if hasattr(value, "tolist") else value
    if key == "embeddings" and isinstance(normalized, list):
        if normalized and not isinstance(normalized[0], (list, tuple)):
            return [normalized]
    return normalized


def _memgpt_query_requires_full_scan_fallback(exc: Exception) -> bool:
    message = str(exc)
    return "contigious 2D array" in message or "ef or M is too small" in message


class _MemGPTOpenAICompatibleEmbeddingEndpoint:
    """Small embedding client for MemGPT local runs against OpenAI-compatible APIs.

    MemGPT's upstream `openai` embedding path currently routes through
    `llama_index.OpenAIEmbedding`, which only accepts OpenAI's hard-coded
    embedding model enum. Real providers such as SiliconFlow expose
    OpenAI-compatible `/embeddings` APIs but with non-OpenAI model ids such as
    `BAAI/bge-m3`. This adapter preserves MemGPT's `openai` config semantics
    while allowing arbitrary model identifiers.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        user: Any = None,
        timeout: float = 60.0,
        dimensions: int | None = None,
        max_retries: int | None = None,
        retry_base_sleep: float | None = None,
        retry_max_sleep: float | None = None,
        retry_jitter: float | None = None,
        min_interval: float | None = None,
        throttle_file: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        if not model:
            raise ValueError("embedding model is required for MemGPT openai-compatible embeddings")
        if not base_url:
            raise ValueError("embedding base_url is required for MemGPT openai-compatible embeddings")
        if not api_key:
            raise ValueError("embedding api_key is required for MemGPT openai-compatible embeddings")
        self.model_name = str(model)
        self.base_url = str(base_url).rstrip("/")
        self.api_key = str(api_key)
        self.user = None if user is None else str(user)
        self.timeout = float(timeout)
        self.dimensions = int(dimensions) if dimensions not in (None, 0, "") else None
        self.max_retries = _int_env("AMST_MEMGPT_EMBEDDING_MAX_RETRIES", max_retries, default=8)
        self.retry_base_sleep = _float_env("AMST_MEMGPT_EMBEDDING_RETRY_BASE_SLEEP", retry_base_sleep, default=2.0)
        self.retry_max_sleep = _float_env("AMST_MEMGPT_EMBEDDING_RETRY_MAX_SLEEP", retry_max_sleep, default=60.0)
        self.retry_jitter = _float_env("AMST_MEMGPT_EMBEDDING_RETRY_JITTER", retry_jitter, default=0.0)
        self.min_interval = _float_env("AMST_MEMGPT_EMBEDDING_MIN_INTERVAL", min_interval, default=0.0)
        self.log_every = _int_env("AMST_MEMGPT_EMBEDDING_LOG_EVERY", None, default=0)
        self._embedding_request_count = 0
        self._embedding_cache_hit_count = 0
        self.throttle_file = str(
            throttle_file
            or os.getenv("AMST_MEMGPT_EMBEDDING_THROTTLE_FILE")
            or _default_embedding_throttle_file(self.base_url, self.model_name)
        )
        self.cache_dir = _embedding_cache_dir(cache_dir)

    def get_text_embedding(self, text: str) -> list[float]:
        cache_path = self._cache_path(str(text))
        if cache_path is not None:
            cached = _read_embedding_cache(cache_path)
            if cached is not None:
                self._embedding_cache_hit_count += 1
                self._log_embedding_progress("cache_hit")
                return cached
        payload: dict[str, Any] = {
            "input": str(text),
            "model": self.model_name,
        }
        if self.user is not None:
            payload["user"] = self.user
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        response = self._post_with_retry(payload)
        body = response.json()
        if isinstance(body, dict):
            try:
                embedding = body["data"][0]["embedding"]
            except (KeyError, IndexError, TypeError) as exc:
                raise TypeError(f"unexpected OpenAI-compatible embedding payload: {body!r}") from exc
        elif isinstance(body, list):
            embedding = body
        else:
            raise TypeError(f"unexpected OpenAI-compatible embedding payload type: {type(body)!r}")
        result = list(embedding)
        self._embedding_request_count += 1
        self._log_embedding_progress("request")
        if cache_path is not None:
            _write_embedding_cache(cache_path, result)
        return result

    def _log_embedding_progress(self, event: str) -> None:
        log_every = max(int(self.log_every), 0)
        if log_every <= 0:
            return
        total = self._embedding_request_count + self._embedding_cache_hit_count
        if total <= 0 or total % log_every != 0:
            return
        print(
            "AMST_MEMGPT_EMBEDDING_PROGRESS "
            f"event={event} total={total} requests={self._embedding_request_count} "
            f"cache_hits={self._embedding_cache_hit_count} model={self.model_name}",
            file=sys.stderr,
            flush=True,
        )

    def _cache_path(self, text: str) -> Path | None:
        if self.cache_dir is None:
            return None
        key_payload = {
            "base_url": self.base_url,
            "dimensions": self.dimensions,
            "model": self.model_name,
            "text": text,
            "version": 1,
        }
        digest = hashlib.sha256(json.dumps(key_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def _post_with_retry(self, payload: dict[str, Any]) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(max(0, self.max_retries) + 1):
            try:
                self._throttle_before_request()
                response = httpx.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code not in {408, 409, 425, 429, 500, 502, 503, 504}:
                    response.raise_for_status()
                    return response
                response.raise_for_status()
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                retry_after = _retry_after_seconds(exc)
                sleep_s = retry_after if retry_after is not None else self.retry_base_sleep * (2 ** attempt)
                if self.retry_jitter > 0:
                    sleep_s += random.uniform(0.0, self.retry_jitter)
                max_sleep = max(float(self.retry_max_sleep), 0.1)
                sleep_s = min(max(sleep_s, 0.1), max_sleep)
                status = getattr(getattr(exc, "response", None), "status_code", None)
                print(
                    "AMST_MEMGPT_EMBEDDING_RETRY "
                    f"status={status} attempt={attempt + 1}/{self.max_retries} sleep_s={sleep_s:.3f}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(sleep_s)
                continue
        if last_error is not None:
            raise last_error
        raise RuntimeError("embedding request retry loop exited without response")

    def _throttle_before_request(self) -> None:
        interval = max(float(self.min_interval), 0.0)
        if interval <= 0:
            return
        path = Path(self.throttle_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                raw = handle.read().strip()
                try:
                    last_ts = float(raw) if raw else 0.0
                except ValueError:
                    last_ts = 0.0
                wait_s = interval - (time.time() - last_ts)
                if wait_s > 0:
                    time.sleep(wait_s)
                handle.seek(0)
                handle.truncate()
                handle.write(str(time.time()))
                handle.flush()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _int_env(name: str, value: int | None, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw not in (None, ""):
        try:
            return int(raw)
        except ValueError:
            return default
    if value is None:
        return default
    return int(value)


def _float_env(name: str, value: float | None, *, default: float) -> float:
    raw = os.environ.get(name)
    if raw not in (None, ""):
        try:
            return float(raw)
        except ValueError:
            return default
    if value is None:
        return default
    return float(value)


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _default_embedding_throttle_file(base_url: str, model: str) -> Path:
    key = hashlib.sha256(f"{base_url}|{model}".encode("utf-8")).hexdigest()[:16]
    return PROJECT_ROOT / ".cache" / "amst_memgpt_embedding_throttle" / f"{key}.lock"


def _embedding_cache_dir(raw_dir: str | None) -> Path | None:
    enabled = os.getenv("AMST_MEMGPT_EMBEDDING_CACHE", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None
    selected = raw_dir or os.getenv("AMST_MEMGPT_EMBEDDING_CACHE_DIR")
    path = Path(selected) if selected else PROJECT_ROOT / ".cache" / "amst_memgpt_embedding_cache"
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_embedding_cache(path: Path) -> list[float] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    embedding = payload.get("embedding") if isinstance(payload, dict) else None
    if not isinstance(embedding, list):
        return None
    try:
        return [float(value) for value in embedding]
    except (TypeError, ValueError):
        return None


def _write_embedding_cache(path: Path, embedding: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    payload = {"embedding": [float(value) for value in embedding], "schema_version": "amb-memgpt-embedding-cache-v1"}
    try:
        tmp_path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()


def _patch_memgpt_embedding_model(module: Any) -> None:
    package_name = getattr(module, "__name__", "")
    if not package_name:
        return
    embeddings_mod = importlib.import_module(f"{package_name}.embeddings")
    if getattr(embeddings_mod, "__amst_openai_compatible_patch__", False):
        return
    credentials_mod = importlib.import_module(f"{package_name}.credentials")
    original = embeddings_mod.embedding_model

    def patched(config: Any, user_id: Any = None):
        endpoint_type = getattr(config, "embedding_endpoint_type", None)
        if endpoint_type == "openai":
            credentials = credentials_mod.MemGPTCredentials.load()
            omit_dimensions = os.getenv("AMST_MEMGPT_OMIT_EMBEDDING_DIMENSIONS") == "1"
            # The embedder may use a different key than the chat LLM. When a
            # dedicated embedding key is provided it takes precedence over the
            # shared MemGPT credentials key (which the LLM also uses).
            embedding_api_key = os.getenv("AMST_MEMGPT_EMBEDDING_API_KEY") or str(
                getattr(credentials, "openai_key", "") or ""
            )
            return _MemGPTOpenAICompatibleEmbeddingEndpoint(
                model=str(getattr(config, "embedding_model", None) or "text-embedding-3-small"),
                base_url=str(getattr(config, "embedding_endpoint", None) or "https://api.openai.com/v1"),
                api_key=embedding_api_key,
                user=user_id,
                dimensions=None if omit_dimensions else getattr(config, "embedding_dim", None),
            )
        return original(config, user_id=user_id)

    setattr(patched, "__wrapped__", original)
    embeddings_mod.embedding_model = patched
    embeddings_mod.__amst_openai_compatible_patch__ = True
    memory_mod = importlib.import_module(f"{package_name}.memory")
    memory_mod.embedding_model = patched


def _patch_memgpt_sync_server_create_agent(module: Any) -> None:
    package_name = getattr(module, "__name__", "")
    if not package_name:
        return
    server_mod = importlib.import_module(f"{package_name}.server.server")
    sync_server_cls = getattr(server_mod, "SyncServer", None)
    if sync_server_cls is None or getattr(sync_server_cls.create_agent, "__amst_cache_patch__", False):
        return

    def patched(
        self,
        user_id,
        tools,
        memory,
        system=None,
        metadata=None,
        name=None,
        llm_config=None,
        embedding_config=None,
        interface=None,
    ):
        if self.ms.get_user(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")

        if interface is None:
            interface = self.default_interface_factory()
        if system is None:
            system = server_mod.gpt_system.get_system_text(self.config.preset)
        if name is None:
            name = server_mod.create_random_username()

        server_mod.logger.debug(f"Attempting to find user: {user_id}")
        user = self.ms.get_user(user_id=user_id)
        if not user:
            raise ValueError(f"cannot find user with associated client id: {user_id}")

        agent = None
        try:
            llm_config = llm_config if llm_config else self.server_llm_config
            embedding_config = embedding_config if embedding_config else self.server_embedding_config

            tool_objs = []
            for tool_name in tools:
                tool_obj = self.ms.get_tool(tool_name, user_id=user_id)
                assert tool_obj, f"Tool {tool_name} does not exist"
                tool_objs.append(tool_obj)

            memory_functions = server_mod.get_memory_functions(memory)
            for func_name, func in memory_functions.items():
                if func_name in tools:
                    continue
                source_code = server_mod.parse_source_code(func)
                json_schema = server_mod.generate_schema(func, func_name)
                source_type = "python"
                tags = ["memory", "memgpt-base"]
                tool = self.create_tool(
                    user_id=user_id,
                    json_schema=json_schema,
                    source_code=source_code,
                    source_type=source_type,
                    tags=tags,
                    exists_ok=True,
                )
                tool_objs.append(tool)
                tools.append(tool.name)

            agent_state = server_mod.AgentState(
                name=name,
                user_id=user_id,
                tools=tools,
                llm_config=llm_config,
                embedding_config=embedding_config,
                system=system,
                state={"system": system, "messages": None, "memory": memory.to_dict()},
                _metadata=metadata or {},
            )
            agent = server_mod.Agent(
                interface=interface,
                agent_state=agent_state,
                tools=tool_objs,
                first_message_verify_mono=True if (llm_config.model is not None and "gpt-4" in llm_config.model) else False,
            )
            # Upstream LocalClient loses the live Agent object after creation and
            # immediately re-loads it on the first archival-memory operation.
            # Cache the fresh Agent here so archival insert/search can reuse the
            # in-memory object instead of entering the broken reload path.
            self._add_agent(user_id=user_id, agent_id=agent.agent_state.id, agent_obj=agent)
        except Exception as exc:
            server_mod.logger.exception(exc)
            try:
                if agent is not None:
                    self.ms.delete_agent(agent_id=agent.agent_state.id)
            except Exception as delete_exc:
                server_mod.logger.exception(f"Failed to delete_agent:\n{delete_exc}")
            raise exc

        server_mod.save_agent(agent, self.ms)
        server_mod.logger.info(f"Created new agent from config: {agent}")
        return agent.agent_state

    setattr(patched, "__wrapped__", sync_server_cls.create_agent)
    setattr(patched, "__amst_cache_patch__", True)
    sync_server_cls.create_agent = patched


def _patch_memgpt_chroma_telemetry(module: Any) -> None:
    package_name = getattr(module, "__name__", "")
    if not package_name:
        return
    chroma_store_mod = importlib.import_module(f"{package_name}.agent_store.chroma")
    if getattr(chroma_store_mod, "__amst_chroma_telemetry_patch__", False):
        return
    chromadb_mod = getattr(chroma_store_mod, "chromadb", None)
    if chromadb_mod is None:
        return

    config_mod = importlib.import_module("chromadb.config")
    settings_cls = getattr(config_mod, "Settings", None)
    if settings_cls is None:
        return
    try:
        posthog_mod = importlib.import_module("chromadb.telemetry.product.posthog")
    except ModuleNotFoundError:
        posthog_mod = None

    original_persistent_client = getattr(chromadb_mod, "PersistentClient", None)
    original_http_client = getattr(chromadb_mod, "HttpClient", None)
    if not callable(original_persistent_client) or not callable(original_http_client):
        return

    def disabled_settings(settings: Any = None):
        if settings is None:
            return settings_cls(anonymized_telemetry=False)
        try:
            setattr(settings, "anonymized_telemetry", False)
            return settings
        except Exception:
            pass
        values: dict[str, Any] = {}
        for attr_name in ("model_dump", "dict"):
            dump = getattr(settings, attr_name, None)
            if not callable(dump):
                continue
            try:
                dumped = dump()
            except TypeError:
                dumped = dump(exclude_none=False)
            if isinstance(dumped, dict):
                values.update(dumped)
                break
        values["anonymized_telemetry"] = False
        return settings_cls(**values)

    def persistent_client(path: str = "./chroma", settings: Any = None, tenant: str = "default_tenant", database: str = "default_database"):
        return original_persistent_client(
            path=path,
            settings=disabled_settings(settings),
            tenant=tenant,
            database=database,
        )

    def http_client(
        host: str = "localhost",
        port: int = 8000,
        ssl: bool = False,
        headers: dict[str, str] | None = None,
        settings: Any = None,
        tenant: str = "default_tenant",
        database: str = "default_database",
    ):
        return original_http_client(
            host=host,
            port=port,
            ssl=ssl,
            headers=headers,
            settings=disabled_settings(settings),
            tenant=tenant,
            database=database,
        )

    chromadb_mod.PersistentClient = persistent_client
    chromadb_mod.HttpClient = http_client
    if posthog_mod is not None:
        posthog_client_cls = getattr(posthog_mod, "Posthog", None)
        if posthog_client_cls is not None:
            original_direct_capture = getattr(posthog_client_cls, "_direct_capture", None)
            if callable(original_direct_capture) and not getattr(original_direct_capture, "__amst_disabled_noop_patch__", False):
                def direct_capture(self, event: Any) -> None:
                    if getattr(getattr(posthog_mod, "posthog", None), "disabled", False):
                        return None
                    return original_direct_capture(self, event)

                setattr(direct_capture, "__wrapped__", original_direct_capture)
                setattr(direct_capture, "__amst_disabled_noop_patch__", True)
                posthog_client_cls._direct_capture = direct_capture
    chroma_store_mod.__amst_chroma_telemetry_patch__ = True
