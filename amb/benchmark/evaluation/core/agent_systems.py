"""Config-driven loading for arbitrary black-box agent systems."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from amb.benchmark.evaluation.adapters import BlackBoxAgent
from amb.benchmark.evaluation.framework_adapters.optional_dependency import OptionalFrameworkDependencyError
from amb.benchmark.schemas.io import read_json
from amb.benchmark.security.secret_hygiene import secret_like_paths


GENERIC_AGENT_SYSTEM_SCHEMA = "amst-agent-system-config-v1"


@dataclass(frozen=True)
class AgentSystemSpec:
    """Resolved metadata for one benchmarked agent system."""

    system_id: str
    system_version: str
    loader: str
    provider: str
    config_type: str
    schema_version: str
    execution_mode: str
    config: dict[str, Any]
    dependencies: dict[str, Any]


def load_agent_system(config_path: str | Path) -> tuple[BlackBoxAgent, AgentSystemSpec]:
    """Load a black-box agent from either a provider integration or generic config.

    Existing integration configs remain supported:

    ```json
    {"provider": "mem0", "client_factory": "my_module:create_client"}
    ```

    Generic configs let users benchmark new agent systems without editing the
    AutoMemoryBench provider registry:

    ```json
    {
      "schema_version": "amst-agent-system-config-v1",
      "system_id": "my_agent",
      "loader": "my_module:create_agent",
      "loader_kwargs": {}
    }
    ```
    """

    config = read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"agent system config must be a JSON object: {config_path}")
    if _is_generic_agent_system_config(config):
        return _load_generic_agent_system(config)
    return _load_provider_integration(config_path, config)


def _is_generic_agent_system_config(config: dict[str, Any]) -> bool:
    return (
        str(config.get("schema_version") or "") == GENERIC_AGENT_SYSTEM_SCHEMA
        or "loader" in config
        or "agent_factory" in config
    )


def _load_provider_integration(config_path: str | Path, config: dict[str, Any]) -> tuple[BlackBoxAgent, AgentSystemSpec]:
    from amb.benchmark.integrations.factory import canonical_provider, load_integration_agent

    agent = load_integration_agent(config_path)
    provider = canonical_provider(str(config.get("provider", "")))
    system_id = str(config.get("system_id") or agent.config.system_id)
    loader = str(config.get("client_factory", ""))
    dependencies = {
        "integration_provider": provider or str(config.get("provider") or system_id),
        "integration_config": str(config_path),
        "agent_memory_benchmark": "local",
    }
    if loader:
        dependencies["agent_loader"] = loader
    return agent, AgentSystemSpec(
        system_id=system_id,
        system_version=str(config.get("system_version") or "unspecified"),
        loader=loader,
        provider=provider,
        config_type="provider_integration",
        schema_version=str(config.get("schema_version") or "legacy-provider-integration"),
        execution_mode=str(config.get("execution_mode") or "integration_smoke"),
        config=config,
        dependencies=dependencies,
    )


def _load_generic_agent_system(config: dict[str, Any]) -> tuple[BlackBoxAgent, AgentSystemSpec]:
    schema_version = str(config.get("schema_version") or GENERIC_AGENT_SYSTEM_SCHEMA)
    if schema_version != GENERIC_AGENT_SYSTEM_SCHEMA:
        raise ValueError(
            f"unknown agent system schema_version {schema_version!r}; "
            f"expected {GENERIC_AGENT_SYSTEM_SCHEMA!r}"
        )
    execution_mode = str(config.get("execution_mode") or "integration_smoke")
    if execution_mode == "dependency_preflight":
        raise OptionalFrameworkDependencyError(
            "dependency_preflight agent-system configs are validation-only and cannot be loaded for scoring"
        )
    loader = str(config.get("loader") or config.get("agent_factory") or "")
    if not loader:
        raise ValueError("generic agent system config must provide loader as 'module:callable'")
    loader_kwargs = config.get("loader_kwargs", {})
    if not isinstance(loader_kwargs, dict):
        raise ValueError("generic agent system loader_kwargs must be an object")
    secret_paths = secret_like_paths(loader_kwargs, path="$.loader_kwargs")
    if secret_paths:
        raise ValueError(
            "generic agent system loader_kwargs must not contain materialized secrets; "
            f"offending paths: {', '.join(secret_paths)}"
        )
    agent = _call_loader(loader, dict(loader_kwargs))
    if not isinstance(agent, BlackBoxAgent):
        raise ValueError(
            f"agent loader {loader!r} returned {type(agent).__name__}, "
            "which does not implement reset/observe/answer_or_act"
        )
    system_id = str(config.get("system_id") or _system_id_from_agent(agent) or loader.split(":", 1)[0])
    provider = str(config.get("provider") or "generic")
    dependencies = {
        "agent_system_provider": provider,
        "agent_loader": loader,
        "agent_memory_benchmark": "local",
    }
    return agent, AgentSystemSpec(
        system_id=system_id,
        system_version=str(config.get("system_version") or "unspecified"),
        loader=loader,
        provider=provider,
        config_type="generic_agent_system",
        schema_version=schema_version,
        execution_mode=execution_mode,
        config=config,
        dependencies=dependencies,
    )


def agent_system_metadata(spec: AgentSystemSpec, config_path: str | Path) -> dict[str, Any]:
    """Return explicit run metadata for interpreting agent-system results."""

    metadata = {
        "config_type": spec.config_type,
        "schema_version": spec.schema_version,
        "system_id": spec.system_id,
        "system_version": spec.system_version,
        "provider": spec.provider,
        "loader": spec.loader,
        "config_path": str(config_path),
        "execution_mode": spec.execution_mode,
    }
    for key in (
        "agent_framework",
        "agent_runtime",
        "orchestration_mode",
        "memory_backend",
        "memory_backend_version",
        "model_backend",
        "tool_runtime_id",
    ):
        if spec.config.get(key) not in (None, ""):
            metadata[key] = spec.config[key]
    return metadata


def bind_agent_system_metadata(
    agent_system: dict[str, Any] | None,
    *,
    system_id: str,
    system_version: str,
    execution_mode: str,
) -> dict[str, Any] | None:
    """Bind agent-system metadata to the concrete run identity.

    CLI callers may override ``system_id`` or ``system_version`` relative to the
    source config. The run metadata and framework trace attribution must follow
    the actual output identity, not the stale config default.
    """

    if agent_system is None:
        return None
    metadata = dict(agent_system)
    metadata["system_id"] = system_id
    metadata["system_version"] = system_version
    metadata["execution_mode"] = execution_mode
    return metadata


def _call_loader(reference: str, kwargs: dict[str, Any]) -> Any:
    module_name, separator, attr_name = reference.partition(":")
    if not separator or not module_name or not attr_name:
        raise ValueError("agent system loader must use 'module:callable' format")
    module = import_module(module_name)
    factory = getattr(module, attr_name, None)
    if not callable(factory):
        raise ValueError(f"agent system loader {reference!r} is not callable")
    return factory(**kwargs)


def _system_id_from_agent(agent: BlackBoxAgent) -> str | None:
    config = getattr(agent, "config", None)
    system_id = getattr(config, "system_id", None)
    if system_id:
        return str(system_id)
    return None
