"""Local official-source A-MEM client factory for AutoMemoryBench integration runs."""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from amb.clients.core.common import effective_api_key, ensure_site_packages_from_venv, ensure_source_path


class AMemOfficialSourceClient:
    """Thin adapter over the official ``AgenticMemorySystem`` API."""

    def __init__(self, memory_system: Any, *, default_limit: int = 5) -> None:
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        self.memory_system = memory_system
        self.default_limit = default_limit
        self.case_id: str | None = None
        self._written_ids: list[str] = []

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        self._written_ids = []
        memories = getattr(self.memory_system, "memories", None)
        if isinstance(memories, dict):
            memories.clear()
        retriever = getattr(self.memory_system, "retriever", None)
        client = getattr(retriever, "client", None)
        if client is not None:
            client.reset()
            retriever_cls = type(retriever)
            self.memory_system.retriever = retriever_cls(
                collection_name="memories",
                model_name=getattr(self.memory_system, "model_name", "all-MiniLM-L6-v2"),
            )
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
        text = _content_from_payload(content=content, messages=messages)
        timestamp = str((metadata or {}).get("timestamp") or "") or None
        memory_id = self.memory_system.add_note(text, time=timestamp)
        self._written_ids.append(str(memory_id))
        return {"id": str(memory_id), "memory": text, "user_id": user_id or self.case_id}

    def search(
        self,
        query: str | None = None,
        *,
        limit: int | None = None,
        top_k: int | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required for A-MEM search")
        k = int(limit or top_k or self.default_limit)
        return list(self.memory_system.search(str(query), k=k))

    def get_all(self, **_: Any) -> list[dict[str, Any]]:
        memories = getattr(self.memory_system, "memories", {})
        rows = []
        for memory_id, memory in memories.items():
            rows.append(
                {
                    "id": str(memory_id),
                    "content": str(getattr(memory, "content", "")),
                    "metadata": {
                        "keywords": list(getattr(memory, "keywords", []) or []),
                        "context": getattr(memory, "context", None),
                        "tags": list(getattr(memory, "tags", []) or []),
                        "timestamp": getattr(memory, "timestamp", None),
                    },
                }
            )
        return rows

    def delete(self, memory_id: str | None = None, **_: Any) -> bool:
        if not memory_id:
            raise ValueError("memory_id is required for A-MEM delete")
        return bool(self.memory_system.delete(str(memory_id)))


def create_client(
    *,
    source_root: str = "related_work/repos/A-mem-sys",
    venv_root: str | None = None,
    model_name: str = "all-MiniLM-L6-v2",
    llm_backend: str = "ollama",
    llm_model: str = "llama2",
    api_key: str | None = None,
    api_key_env: str = "A_MEM_OPENAI_API_KEY",
    fallback_envs: list[str] | tuple[str, ...] = ("OPENAI_API_KEY", "AMST_OPENAI_COMPAT_API_KEY"),
    base_url: str | None = None,
    base_url_env: str = "A_MEM_OPENAI_BASE_URL",
    embedding_model: str | None = None,
    embedding_base_url: str | None = None,
    embedding_api_key_env: str | None = None,
    evo_threshold: int = 100,
    default_limit: int = 5,
    require_llm_healthcheck: bool = False,
    llm_temperature: float | None = None,
    llm_timeout_s: float | None = None,
    llm_transient_max_attempts: int = 1,
    llm_transient_base_sleep_s: float = 2.0,
    **kwargs: Any,
) -> AMemOfficialSourceClient:
    """Create a client from the locally cloned official A-MEM source tree."""

    site_packages: Path | None = None
    if venv_root:
        site_packages = ensure_site_packages_from_venv(Path(venv_root))
        _purge_non_venv_dependency_modules(
            site_packages,
            module_roots=("attr", "attrs", "jsonschema", "chromadb"),
        )
    ensure_source_path(source_root)
    from agentic_memory.memory_system import AgenticMemorySystem  # type: ignore

    resolved_api_key = api_key
    if llm_backend in {"openai", "openrouter"}:
        resolved_api_key = effective_api_key(
            api_key=api_key,
            api_key_env=api_key_env,
            fallback_envs=tuple(fallback_envs),
        )
        if not resolved_api_key:
            raise RuntimeError(
                f"{api_key_env}, OPENAI_API_KEY, or AMST_OPENAI_COMPAT_API_KEY is required for A-MEM {llm_backend} runs"
            )
    resolved_base_url = base_url or os.getenv(base_url_env)

    # The systems x models matrix injects an embedder spec (embedding_model /
    # embedding_base_url / embedding_api_key_env). A-MEM's official embedder is a
    # local SentenceTransformer wired through ChromaRetriever(model_name=...) — it
    # has no remote-embedder hook. So map the injected embedder *name* onto the
    # only knob AgenticMemorySystem exposes (model_name) and consume the remote
    # endpoint/key params without forwarding them; they are accepted here purely
    # so they never leak into AgenticMemorySystem(**kwargs).
    resolved_embedder_model_name = embedding_model or model_name
    _ = (embedding_base_url, embedding_api_key_env)  # consumed: A-MEM has no remote embedder.

    with _temporary_env("OPENAI_BASE_URL", resolved_base_url if llm_backend == "openai" else None):
        memory_system = AgenticMemorySystem(
            model_name=resolved_embedder_model_name,
            llm_backend=llm_backend,
            llm_model=llm_model,
            api_key=resolved_api_key,
            evo_threshold=evo_threshold,
            **kwargs,
        )
    if llm_backend == "openai":
        if llm_temperature is not None and not 0.0 <= float(llm_temperature) <= 2.0:
            raise ValueError("llm_temperature must be between 0 and 2")
        _configure_openai_controller_runtime(
            memory_system,
            timeout_s=llm_timeout_s,
            max_attempts=llm_transient_max_attempts,
            base_sleep_s=llm_transient_base_sleep_s,
            model=llm_model,
            temperature=llm_temperature,
        )
    if require_llm_healthcheck:
        _validate_llm_healthcheck(memory_system, backend=llm_backend, model=llm_model)
    return AMemOfficialSourceClient(memory_system, default_limit=default_limit)


@contextmanager
def _temporary_env(name: str, value: str | None):
    if value is None:
        yield
        return
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _needs_json_object_downgrade(model: str) -> bool:
    """Models that reject OpenAI `json_schema` response_format but accept `json_object`.

    A-MEM hardcodes `response_format={"type":"json_schema",...}` in every LLM call
    (healthcheck + runtime metadata/evolution). Some OpenAI-compatible providers only
    implement the older `json_object` mode (e.g. some providers return
    HTTP 400 "does not support 'json_schema'"). For those we transparently downgrade the
    request to `{"type":"json_object"}` (the prompt already instructs the exact JSON), so
    the official A-MEM algorithm runs unchanged instead of failing the whole cell.
    """
    m = (model or "").lower()
    return (
        "qwen3.6-35b" in m
        or "qwen/qwen3.6" in m
        or "deepseek-v4-pro-ppio" in m
        or "deepseek-v4-flash-ppio" in m
    )


def _downgrade_json_schema(args: tuple, kwargs: dict, *, force: bool = False):
    """Rewrite a json_schema response_format arg to json_object in-place-safe."""
    def _fix(rf):
        if isinstance(rf, dict) and rf.get("type") == "json_schema":
            return {"type": "json_object"}
        return rf
    if "response_format" in kwargs:
        kwargs = {**kwargs, "response_format": _fix(kwargs["response_format"])}
    elif len(args) >= 2 and isinstance(args[1], dict) and args[1].get("type") == "json_schema":
        args = (args[0], _fix(args[1])) + tuple(args[2:])
    return args, kwargs


def _validate_llm_healthcheck(memory_system: Any, *, backend: str, model: str) -> None:
    """Reject A-MEM runs where the official LLM controller silently degrades."""

    controller = getattr(memory_system, "llm_controller", None)
    if controller is None or not callable(getattr(controller, "get_completion", None)):
        raise RuntimeError("A-MEM LLM healthcheck failed: llm_controller is not available")

    if _needs_json_object_downgrade(model):
        response_format = {"type": "json_object"}
    else:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "amst_a_mem_healthcheck",
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            },
        }
    try:
        raw = controller.get_completion(
            'Return exactly this JSON object: {"ok": true}',
            response_format=response_format,
            temperature=0.0,
        )
        payload = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 - expose provider-specific failure as a run blocker.
        raise RuntimeError(
            f"A-MEM LLM healthcheck failed for backend={backend!r} model={model!r}: {exc}"
        ) from exc
    if payload.get("ok") is not True:
        raise RuntimeError(
            "A-MEM LLM healthcheck failed for "
            f"backend={backend!r} model={model!r}: expected ok=true, got {payload!r}"
        )


def _configure_openai_controller_runtime(
    memory_system: Any,
    *,
    timeout_s: float | None,
    max_attempts: int,
    base_sleep_s: float,
    model: str = "",
    temperature: float | None = None,
) -> None:
    """Add transport reliability around A-MEM's official OpenAI-compatible controller.

    The official source does not set an OpenAI client timeout and lets transient
    provider errors bubble into A-MEM's metadata/evolution fallbacks. Retrying
    transient transport/rate-limit failures preserves the official algorithm
    while preventing partial degraded memories from entering benchmark scores.
    """

    controller = getattr(memory_system, "llm_controller", None)
    llm = getattr(controller, "llm", None)
    if llm is None:
        return

    if timeout_s is not None:
        if timeout_s <= 0:
            raise ValueError("llm_timeout_s must be positive")
        client = getattr(llm, "client", None)
        with_options = getattr(client, "with_options", None)
        if callable(with_options):
            llm.client = with_options(timeout=float(timeout_s))

    downgrade = _needs_json_object_downgrade(model)
    retry = max_attempts > 1
    if retry and base_sleep_s <= 0:
        raise ValueError("llm_transient_base_sleep_s must be positive")

    original = getattr(llm, "get_completion", None)
    if not callable(original):
        return

    def _wrapped(*args: Any, **kwargs: Any) -> str:
        try:
            response_schema = _response_schema(args, kwargs)
            required_fields = _required_response_fields(args, kwargs)
            if downgrade:
                args, kwargs = _downgrade_json_schema(args, kwargs)
            if temperature is not None:
                args, kwargs = _set_temperature(args, kwargs, temperature=float(temperature))
            if retry:
                response = _call_with_transient_retry(
                    original, args=args, kwargs=kwargs,
                    max_attempts=max_attempts, base_sleep_s=base_sleep_s,
                    require_json_object=True,
                    required_fields=required_fields,
                    response_schema=response_schema,
                )
            else:
                response = original(*args, **kwargs)
            _require_json_object_response(
                response,
                required_fields=required_fields,
                response_schema=response_schema,
            )
            return response
        except Exception as exc:
            # Official A-MEM catches Exception and silently substitutes empty metadata.
            # Escalate transport/output failures outside that catch boundary so a failed
            # shard cannot be mistaken for a completed benchmark result.
            raise _FatalLLMError(
                "A-MEM LLM call failed; aborting shard to prevent silent empty-memory fallback: "
                f"{exc}"
            ) from exc

    llm.get_completion = _wrapped


def _call_with_transient_retry(
    fn: Callable[..., str],
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    max_attempts: int,
    base_sleep_s: float,
    require_json_object: bool = False,
    required_fields: tuple[str, ...] = (),
    response_schema: dict[str, Any] | None = None,
) -> str:
    last_exc: Exception | None = None
    last_transient = False
    for attempt in range(max_attempts):
        try:
            result = fn(*args, **kwargs)
            if isinstance(result, str) and result.strip():
                if require_json_object:
                    try:
                        _require_json_object_response(
                            result,
                            required_fields=required_fields,
                            response_schema=response_schema,
                        )
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise _TransientMalformedCompletionError(
                            f"A-MEM LLM returned malformed JSON: {exc}"
                        ) from exc
                return result
            raise _TransientEmptyCompletionError(
                f"A-MEM LLM returned empty completion: {type(result).__name__}"
            )
        except Exception as exc:  # noqa: BLE001 - provider SDKs expose heterogeneous errors.
            last_exc = exc
            is_transient = isinstance(exc, _TransientMalformedCompletionError) or _is_transient_llm_error(exc)
            last_transient = is_transient
            if attempt >= max_attempts - 1 or not is_transient:
                if is_transient:
                    raise _FatalTransientLLMError(
                        "A-MEM LLM transient failure persisted after "
                        f"{max_attempts} attempts; aborting shard to avoid official fallback pollution: {exc}"
                    ) from exc
                raise
            retry_after = _retry_after_seconds(exc)
            sleep_s = retry_after if retry_after is not None else base_sleep_s * (2**attempt)
            time.sleep(min(max(float(sleep_s), 0.1), 60.0))
    if last_exc is not None:
        if last_transient:
            raise _FatalTransientLLMError(
                "A-MEM LLM transient retry loop exhausted; aborting shard to avoid official fallback pollution"
            ) from last_exc
        raise last_exc
    raise RuntimeError("A-MEM transient retry loop exited without result")


def _is_transient_llm_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504, 529}:
        return True
    text = str(exc).lower()
    transient_markers = (
        "a-mem llm returned empty completion",
        "rate limit",
        "rate limiting",
        "rpm limit",
        "tpm limit",
        "too many requests",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "connection",
        "server error",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "nonetype object is not subscriptable",
    )
    return any(marker in text for marker in transient_markers)


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class _TransientEmptyCompletionError(RuntimeError):
    """Provider returned no text for a completion that A-MEM must parse."""


class _TransientMalformedCompletionError(RuntimeError):
    """Provider returned non-JSON text despite A-MEM's JSON-object contract."""


class _FatalTransientLLMError(BaseException):
    """Abort A-MEM before official Exception-based fallback can write degraded memory."""


class _FatalLLMError(BaseException):
    """Abort A-MEM on any malformed or non-retriable LLM response."""


def _require_json_object_response(
    response: Any,
    *,
    required_fields: tuple[str, ...] = (),
    response_schema: dict[str, Any] | None = None,
) -> None:
    """Enforce the JSON-object contract required by every official A-MEM prompt."""

    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"empty A-MEM LLM completion: {type(response).__name__}")
    parsed = json.loads(response)
    if not isinstance(parsed, dict):
        raise ValueError(f"A-MEM LLM completion must be a JSON object, got {type(parsed).__name__}")
    missing = [field for field in required_fields if field not in parsed]
    if missing:
        raise ValueError(f"A-MEM LLM completion is missing required fields: {','.join(missing)}")
    if response_schema is not None:
        _validate_json_schema_value(parsed, response_schema, path="$")


def _validate_json_schema_value(value: Any, schema: dict[str, Any], *, path: str) -> None:
    """Validate the JSON-schema subset A-MEM declares before using json_object mode."""

    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object, got {type(value).__name__}")
        required = schema.get("required", [])
        if isinstance(required, list):
            missing = [field for field in required if isinstance(field, str) and field not in value]
            if missing:
                raise ValueError(f"{path} is missing required fields: {','.join(missing)}")
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, property_schema in properties.items():
                if key in value and isinstance(property_schema, dict):
                    _validate_json_schema_value(value[key], property_schema, path=f"{path}.{key}")
        return
    if expected_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array, got {type(value).__name__}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_json_schema_value(item, item_schema, path=f"{path}[{index}]")
        return
    if expected_type == "string" and not isinstance(value, str):
        raise ValueError(f"{path} must be a string, got {type(value).__name__}")
    if expected_type == "boolean" and type(value) is not bool:
        raise ValueError(f"{path} must be a boolean, got {type(value).__name__}")
    if expected_type == "integer" and (type(value) is not int):
        raise ValueError(f"{path} must be an integer, got {type(value).__name__}")
    if expected_type == "number" and (not isinstance(value, (int, float)) or type(value) is bool):
        raise ValueError(f"{path} must be a number, got {type(value).__name__}")


def _required_response_fields(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str, ...]:
    """Extract the official JSON-schema required keys before compatibility downgrade."""

    schema = _response_schema(args, kwargs)
    required = schema.get("required", []) if isinstance(schema, dict) else []
    return tuple(str(field) for field in required if isinstance(field, str))


def _response_schema(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the official schema before compatibility downgrade removes it."""

    response_format = kwargs.get("response_format")
    if response_format is None and len(args) >= 2:
        response_format = args[1]
    if not isinstance(response_format, dict):
        return None
    json_schema = response_format.get("json_schema", {})
    schema = json_schema.get("schema") if isinstance(json_schema, dict) else None
    return schema if isinstance(schema, dict) else None


def _set_temperature(args: tuple[Any, ...], kwargs: dict[str, Any], *, temperature: float) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Set a recorded decoding temperature without changing A-MEM prompts or logic."""

    if "temperature" in kwargs or len(args) < 3:
        return args, {**kwargs, "temperature": temperature}
    return args[:2] + (temperature,) + args[3:], kwargs


def _purge_non_venv_dependency_modules(site_packages: Path | None, *, module_roots: tuple[str, ...]) -> None:
    """Avoid mixing A-MEM's venv dependencies with system site packages."""

    if site_packages is None:
        return
    site_root = site_packages.resolve()
    for name, module in list(sys.modules.items()):
        root = name.split(".", 1)[0]
        if root not in module_roots:
            continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            Path(module_file).resolve().relative_to(site_root)
        except ValueError:
            sys.modules.pop(name, None)


def _content_from_payload(*, content: Any, messages: Any) -> str:
    if messages is not None:
        if isinstance(messages, list):
            return "\n".join(str(item.get("content", item)) if isinstance(item, dict) else str(item) for item in messages)
        if isinstance(messages, dict):
            return str(messages.get("content", messages))
        return str(messages)
    if content is None:
        raise ValueError("content or messages is required")
    return str(content)
