"""Official framework-SDK Mem0 client factory for AutoMemoryBench integration runs."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

from amb.clients.core.common import effective_api_key as resolve_api_key, ensure_site_packages_from_venv, resolve_path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MEM0_VENV = PROJECT_ROOT / ".venv-mem0"
DEFAULT_STORAGE_ROOT = PROJECT_ROOT / ".cache" / "mem0_real"


class Mem0RealClient:
    """Thin adapter over ``mem0.Memory`` with AutoMemoryBench-friendly method signatures."""

    def __init__(
        self,
        memory: Any,
        *,
        default_limit: int = 5,
        transient_max_attempts: int = 8,
        transient_base_sleep: float = 2.0,
    ) -> None:
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        if transient_max_attempts <= 0:
            raise ValueError("transient_max_attempts must be positive")
        if transient_base_sleep <= 0:
            raise ValueError("transient_base_sleep must be positive")
        self.memory = memory
        self.default_limit = default_limit
        self.transient_max_attempts = transient_max_attempts
        self.transient_base_sleep = transient_base_sleep

    def reset(self, **_: Any) -> dict[str, Any]:
        # AMST scopes every add/search call by case_id -> user_id, so a no-op
        # reset is enough and avoids per-case delete-all overhead.
        return {"ok": True}

    def clear(self) -> dict[str, Any]:
        return {"ok": True}

    def add(
        self,
        content: Any = None,
        *,
        messages: Any = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        payload = _normalize_add_payload(content=content, messages=messages)
        if not user_id:
            raise ValueError("user_id is required for mem0 add")
        return self._call_with_transient_retry(
            lambda: self.memory.add(payload, user_id=str(user_id), metadata=dict(metadata or {}), infer=False)
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
            raise ValueError("query is required for mem0 search")
        if not user_id:
            raise ValueError("user_id is required for mem0 search")
        k = int(limit or top_k or self.default_limit)
        return self._call_with_transient_retry(lambda: self.memory.search(str(query), user_id=str(user_id), limit=k))

    def get_all(
        self,
        *,
        user_id: str | None = None,
        limit: int = 100,
        **_: Any,
    ) -> dict[str, Any]:
        if not user_id:
            raise ValueError("user_id is required for mem0 get_all")
        return self._call_with_transient_retry(lambda: self.memory.get_all(user_id=str(user_id), limit=int(limit)))

    def delete(self, memory_id: str | None = None, **_: Any) -> dict[str, Any]:
        if not memory_id:
            raise ValueError("memory_id is required for mem0 delete")
        return self._call_with_transient_retry(lambda: self.memory.delete(str(memory_id)))

    def _call_with_transient_retry(self, fn):
        return _call_with_transient_retry(
            fn,
            max_attempts=self.transient_max_attempts,
            base_sleep=self.transient_base_sleep,
        )


def create_client(
    *,
    venv_root: str | None = None,
    storage_root: str | None = None,
    mem0_dir: str | None = None,
    api_key: str | None = None,
    api_key_env: str = "SILICONFLOW_API_KEY",
    base_url: str = "https://api.siliconflow.cn/v1",
    default_limit: int = 5,
    llm_model: str = "deepseek-ai/DeepSeek-V3",
    llm_max_tokens: int = 64,
    embedding_model: str = "BAAI/bge-m3",
    embedding_dims: int = 1024,
    embedder_provider: str = "openai",
    embedding_base_url: str | None = None,      # default: same endpoint as the LLM
    embedding_api_key_env: str | None = None,   # default: same key as the LLM
    vector_store_provider: str = "qdrant",
    collection_name: str = "amst_mem0_real",
    omit_embedding_dimensions: bool = False,
    transient_max_attempts: int | None = None,
    transient_base_sleep: float | None = None,
    openai_timeout: float | None = None,
    openai_connect_timeout: float | None = None,
    openai_max_retries: int | None = None,
) -> Mem0RealClient:
    """Create a real Mem0-backed client without installing mem0 in the main env."""

    resolved_venv = Path(venv_root) if venv_root else DEFAULT_MEM0_VENV
    ensure_site_packages_from_venv(resolved_venv)

    resolved_storage, resolved_mem0_dir = _resolve_mem0_runtime_roots(
        storage_root=storage_root,
        mem0_dir=mem0_dir,
    )
    resolved_storage.mkdir(parents=True, exist_ok=True)
    resolved_mem0_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MEM0_DIR"] = str(resolved_mem0_dir)

    from mem0 import Memory  # type: ignore

    resolved_api_key = resolve_api_key(
        api_key=api_key,
        api_key_env=api_key_env,
        fallback_envs=("OPENAI_API_KEY",),
    )
    # Embedder may live on a different endpoint/key than the chat LLM (e.g. chat on
    # your provider, embeddings on your embedding endpoint). Default to the LLM's endpoint/key.
    resolved_embedding_key = (
        resolve_api_key(api_key=None, api_key_env=embedding_api_key_env,
                        fallback_envs=(api_key_env, "OPENAI_API_KEY"))
        if embedding_api_key_env else resolved_api_key
    )
    config = _build_mem0_config(
        storage_root=resolved_storage,
        base_url=base_url,
        api_key=resolved_api_key,
        llm_model=llm_model,
        llm_max_tokens=llm_max_tokens,
        embedding_model=embedding_model,
        embedding_dims=embedding_dims,
        embedder_provider=embedder_provider,
        embedding_base_url=embedding_base_url or base_url,
        embedding_api_key=resolved_embedding_key,
        vector_store_provider=vector_store_provider,
        collection_name=collection_name,
    )
    memory = Memory.from_config(config)
    _patch_mem0_openai_clients(
        memory,
        timeout=_float_env("AMB_MEM0_OPENAI_TIMEOUT", openai_timeout, default=120.0),
        connect_timeout=_float_env("AMB_MEM0_OPENAI_CONNECT_TIMEOUT", openai_connect_timeout, default=10.0),
        max_retries=_int_env("AMB_MEM0_OPENAI_MAX_RETRIES", openai_max_retries, default=4),
    )
    if omit_embedding_dimensions:
        _patch_openai_embedder_omit_dimensions(memory)
    return Mem0RealClient(
        memory,
        default_limit=default_limit,
        transient_max_attempts=_int_env("AMB_MEM0_TRANSIENT_MAX_ATTEMPTS", transient_max_attempts, default=8),
        transient_base_sleep=_float_env("AMB_MEM0_TRANSIENT_BASE_SLEEP", transient_base_sleep, default=2.0),
    )


def _resolve_mem0_dir(raw: str | None, storage_root: Path) -> Path:
    if raw:
        return resolve_path(raw, storage_root / "_mem0_home")
    return storage_root / "_mem0_home"


def _resolve_mem0_runtime_roots(*, storage_root: str | None, mem0_dir: str | None) -> tuple[Path, Path]:
    """Resolve shard-local Mem0 storage roots, honoring runner env overrides."""

    # Parallel shards for the same Mem0 config can run concurrently. The official
    # Mem0/Qdrant local backend mutates its storage path during construction, so
    # every shard needs an isolated runtime root to avoid rmtree races.
    storage_root = os.getenv("AMB_MEM0_STORAGE_ROOT") or storage_root
    mem0_dir = os.getenv("AMB_MEM0_DIR") or mem0_dir
    resolved_storage = resolve_path(storage_root, DEFAULT_STORAGE_ROOT)
    return resolved_storage, _resolve_mem0_dir(mem0_dir, resolved_storage)


def _build_mem0_config(
    *,
    storage_root: Path,
    base_url: str,
    api_key: str | None,
    llm_model: str,
    llm_max_tokens: int,
    embedding_model: str,
    embedding_dims: int,
    embedder_provider: str,
    vector_store_provider: str,
    collection_name: str,
    embedding_base_url: str | None = None,
    embedding_api_key: str | None = None,
) -> dict[str, Any]:
    vector_store_config: dict[str, Any] = {
        "provider": vector_store_provider,
        "config": {
            "collection_name": collection_name,
            "path": str(storage_root / "qdrant"),
            "embedding_model_dims": embedding_dims,
        },
    }
    llm_config: dict[str, Any] = {
        "provider": "openai",
        "config": {
            "model": llm_model,
            "api_key": api_key or "unused",
            "openai_base_url": base_url,
            "temperature": 0.0,
            "max_tokens": llm_max_tokens,
        },
    }
    if embedder_provider == "openai":
        emb_key = embedding_api_key or api_key
        if not emb_key:
            raise ValueError("api_key or api_key_env is required when embedder_provider=openai")
        embedder_config: dict[str, Any] = {
            "provider": "openai",
            "config": {
                "model": embedding_model,
                "api_key": emb_key,
                "openai_base_url": embedding_base_url or base_url,
                "embedding_dims": embedding_dims,
            },
        }
    elif embedder_provider == "huggingface":
        embedder_config = {
            "provider": "huggingface",
            "config": {
                "model": embedding_model,
                "embedding_dims": embedding_dims,
            },
        }
    else:
        raise ValueError(f"unsupported embedder_provider {embedder_provider!r}")
    return {
        "vector_store": vector_store_config,
        "llm": llm_config,
        "embedder": embedder_config,
        "history_db_path": str(storage_root / "history.db"),
        "version": "v1.1",
    }


def _patch_mem0_openai_clients(
    memory: Any,
    *,
    timeout: float,
    connect_timeout: float,
    max_retries: int,
) -> None:
    """Add bounded network timeouts to Mem0's official OpenAI-compatible clients."""

    import httpx
    from openai import OpenAI

    timeout_obj = httpx.Timeout(float(timeout), connect=float(connect_timeout))
    for component in (getattr(memory, "llm", None), getattr(memory, "embedding_model", None)):
        client = getattr(component, "client", None)
        if client is None:
            continue
        api_key = getattr(client, "api_key", None)
        base_url = getattr(client, "base_url", None)
        if base_url is not None:
            base_url = str(base_url).rstrip("/")
        try:
            setattr(
                component,
                "client",
                OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout_obj,
                    max_retries=int(max_retries),
                ),
            )
        except Exception:
            # Some Mem0 backends do not use OpenAI clients; leave them untouched.
            continue


def _normalize_add_payload(*, content: Any, messages: Any) -> Any:
    if messages is not None:
        return messages
    if isinstance(content, (str, list, dict)):
        return content
    if content is None:
        raise ValueError("content or messages is required")
    return str(content)


def _patch_openai_embedder_omit_dimensions(memory: Any) -> None:
    """Adapt official Mem0's OpenAI embedder for compatible APIs that reject dimensions."""

    embedder = getattr(memory, "embedding_model", None)
    client = getattr(embedder, "client", None)
    config = getattr(embedder, "config", None)
    if client is None or config is None or not callable(getattr(getattr(client, "embeddings", None), "create", None)):
        raise TypeError("omit_embedding_dimensions requires Mem0's OpenAI embedding backend")

    max_attempts = _int_env("AMB_MEM0_TRANSIENT_MAX_ATTEMPTS", None, default=8)
    base_sleep = _float_env("AMB_MEM0_TRANSIENT_BASE_SLEEP", None, default=2.0)
    max_chars = _int_env("AMB_MEM0_EMBEDDING_MAX_CHARS", None, default=6000)

    def embed_without_dimensions(text: str, memory_action: str | None = None) -> list[float]:
        del memory_action
        normalized = str(text).replace("\n", " ")
        response = _create_embedding_with_compatible_fallbacks(
            client,
            model=config.model,
            text=normalized,
            max_chars=max_chars,
            max_attempts=max_attempts,
            base_sleep=base_sleep,
        )
        return response.data[0].embedding

    embedder.embed = embed_without_dimensions


def _create_embedding_with_compatible_fallbacks(
    client: Any,
    *,
    model: str,
    text: str,
    max_chars: int,
    max_attempts: int,
    base_sleep: float,
) -> Any:
    normalized = str(text)
    truncated = normalized[:max(1, int(max_chars))]
    variants: list[Any] = [[normalized], normalized]
    if truncated != normalized:
        variants.extend(([truncated], truncated))
    last_invalid: Exception | None = None
    seen: set[str] = set()
    for input_payload in variants:
        marker = repr(input_payload)
        if marker in seen:
            continue
        seen.add(marker)
        try:
            return _call_with_transient_retry(
                lambda input_payload=input_payload: client.embeddings.create(input=input_payload, model=model),
                max_attempts=max_attempts,
                base_sleep=base_sleep,
            )
        except Exception as exc:  # pragma: no cover - provider-SDK dependent.
            if _is_invalid_parameter_provider_error(exc):
                last_invalid = exc
                continue
            raise
    if last_invalid is not None:
        raise last_invalid
    raise RuntimeError("Mem0 embedding fallback loop exited without result")


def _call_with_transient_retry(fn, *, max_attempts: int, base_sleep: float):
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - depends on provider SDK exception types.
            if not _is_transient_provider_error(exc) or attempt + 1 >= max_attempts:
                raise
            last_exc = exc
            retry_after = _retry_after_seconds(exc)
            sleep_s = retry_after if retry_after is not None else base_sleep * (2 ** attempt)
            time.sleep(min(max(sleep_s, 0.1), 60.0))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Mem0 transient retry loop exited without result")


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
            "ratelimit",
            "rate limit",
            "too many requests",
            "temporarily unavailable",
            "timeout",
            "connection reset",
            "service unavailable",
            "api connection",
            "connection error",
            "connecterror",
            "unexpected_eof",
            "eof occurred",
            "ssl",
        )
    )


def _is_invalid_parameter_provider_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    message = str(exc).lower()
    return (status_code == 400 or response_status == 400) and (
        "20015" in message or "parameter is invalid" in message
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
