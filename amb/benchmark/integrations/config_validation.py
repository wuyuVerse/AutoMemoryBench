"""Static validation for AutoMemoryBench integration config files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from amb.benchmark.artifact_contract import localize_report_contract
from amb.benchmark.evaluation.agent_systems import GENERIC_AGENT_SYSTEM_SCHEMA
from amb.benchmark.evaluation.framework_contracts import (
    FrameworkAdapterContract,
    load_framework_adapter_contracts,
)
from amb.benchmark.integrations.factory import PROVIDERS, canonical_provider
from amb.benchmark.quality.run_metadata import (
    REQUIRED_GENERIC_AGENT_SYSTEM_INTERPRETABILITY_FIELDS,
    build_real_system_attestation,
    validate_real_system_run_metadata,
)
from amb.benchmark.schemas.io import read_json, write_json
from amb.benchmark.security.secret_hygiene import secret_like_paths

INTEGRATION_CONFIG_VALIDATION_SCHEMA_VERSION = "amst-integration-config-validation-v1"


def validate_integration_config(
    path: str | Path,
    *,
    framework_contract_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Validate one integration config without importing its client factory."""

    source = Path(path)
    errors: list[str] = []
    framework_contracts = _load_framework_contracts(framework_contract_paths, errors)
    try:
        config = read_json(source)
    except Exception as exc:
        return _config_report(source, None, {}, (f"cannot read integration config: {exc}",))
    if not isinstance(config, dict):
        return _config_report(source, None, {}, ("integration config must be a JSON object",))

    config_type = "generic_agent_system" if _is_generic_agent_system_config(config) else "provider_integration"
    provider = str(config.get("provider", "generic")) if config_type == "generic_agent_system" else canonical_provider(str(config.get("provider", "")))
    client_factory = str(config.get("client_factory", ""))
    loader = str(config.get("loader") or config.get("agent_factory") or "")
    if config_type == "generic_agent_system":
        schema_version = str(config.get("schema_version") or GENERIC_AGENT_SYSTEM_SCHEMA)
        if schema_version != GENERIC_AGENT_SYSTEM_SCHEMA:
            errors.append(f"unknown schema_version {schema_version!r}; expected {GENERIC_AGENT_SYSTEM_SCHEMA!r}")
        if not _valid_factory_reference(loader):
            errors.append("loader must use 'module:callable' format")
        if "loader_kwargs" in config and not isinstance(config.get("loader_kwargs"), dict):
            errors.append("loader_kwargs must be an object")
        if isinstance(config.get("loader_kwargs"), dict):
            secret_paths = secret_like_paths(config.get("loader_kwargs"), path="$.loader_kwargs")
            errors.extend(f"loader_kwargs must not contain materialized secrets: {path}" for path in secret_paths)
        for field in REQUIRED_GENERIC_AGENT_SYSTEM_INTERPRETABILITY_FIELDS:
            value = config.get(field)
            if value is None or value == "" or value == {} or value == []:
                errors.append(f"{field} is required for generic_agent_system")
        if framework_contracts:
            errors.extend(_validate_against_framework_contract(config, loader, framework_contracts))
    else:
        if provider not in PROVIDERS:
            errors.append(f"unknown provider {config.get('provider')!r}; expected one of {', '.join(sorted(PROVIDERS))}")
        if not _valid_factory_reference(client_factory):
            errors.append("client_factory must use 'module:callable' format")
    execution_mode = str(config.get("execution_mode") or "integration_smoke")
    artifact = _artifact_info(source)
    attestation: dict[str, Any] | None = None
    attestation_factory = loader or client_factory
    if "real_system_attestation" in config:
        try:
            attestation = build_real_system_attestation(
                config.get("real_system_attestation"),
                artifact,
                config_provider=provider,
                config_client_factory=attestation_factory,
            )
        except ValueError as exc:
            errors.append(str(exc))
    if execution_mode == "real_system" and attestation is None:
        errors.append("real_system execution_mode requires real_system_attestation")
    if execution_mode != "real_system" and attestation is not None:
        errors.append("real_system_attestation requires execution_mode=real_system")
    if attestation is not None:
        errors.extend(
            validate_real_system_run_metadata(
                {
                    "execution_mode": execution_mode,
                    "integration_config_artifact": artifact,
                    "real_system_attestation": attestation,
                }
            )
        )

    return _config_report(
        source,
        provider,
        config,
        tuple(errors),
        execution_mode=execution_mode,
        client_factory=client_factory,
        loader=loader,
        config_type=config_type,
        integration_config_artifact=artifact,
        real_system_attestation=attestation,
        framework_contract_id=_matched_framework_contract_id(config, framework_contracts),
    )


def validate_integration_config_files(
    paths: Iterable[str | Path],
    *,
    framework_contract_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    framework_contract_paths = tuple(framework_contract_paths)
    reports = [
        validate_integration_config(path, framework_contract_paths=framework_contract_paths)
        for path in paths
    ]
    errors = [error for report in reports for error in report["errors"]]
    return {
        "schema_version": INTEGRATION_CONFIG_VALIDATION_SCHEMA_VERSION,
        "status": "passed" if not errors else "failed",
        "num_configs": len(reports),
        "num_framework_contracts": len(framework_contract_paths),
        "configs": reports,
        "errors": errors,
    }


def write_integration_config_validation(
    paths: Iterable[str | Path],
    output: str | Path,
    *,
    framework_contract_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    framework_contract_paths = tuple(framework_contract_paths)
    report = validate_integration_config_files(paths, framework_contract_paths=framework_contract_paths)
    report = localize_report_contract(
        report,
        output_path=output,
        project_root_hints=(*tuple(paths), *framework_contract_paths),
    )
    write_json(output, report)
    return report


def _config_report(
    source: Path,
    provider: str | None,
    config: dict[str, Any],
    errors: tuple[str, ...],
    *,
    execution_mode: str | None = None,
    client_factory: str | None = None,
    loader: str | None = None,
    config_type: str | None = None,
    integration_config_artifact: dict[str, Any] | None = None,
    real_system_attestation: dict[str, Any] | None = None,
    framework_contract_id: str | None = None,
) -> dict[str, Any]:
    return {
        "path": str(source),
        "status": "passed" if not errors else "failed",
        "config_type": config_type,
        "provider": provider,
        "system_id": config.get("system_id"),
        "execution_mode": execution_mode,
        "client_factory": client_factory,
        "loader": loader,
        "agent_framework": config.get("agent_framework"),
        "orchestration_mode": config.get("orchestration_mode"),
        "memory_backend": config.get("memory_backend"),
        "tool_runtime_id": config.get("tool_runtime_id"),
        "framework_contract_id": framework_contract_id,
        "integration_config_artifact": integration_config_artifact,
        "real_system_attestation": real_system_attestation,
        "errors": list(errors),
    }


def _is_generic_agent_system_config(config: dict[str, Any]) -> bool:
    return (
        str(config.get("schema_version") or "") == GENERIC_AGENT_SYSTEM_SCHEMA
        or "loader" in config
        or "agent_factory" in config
    )


def _valid_factory_reference(reference: str) -> bool:
    module_name, separator, attr_name = reference.partition(":")
    return bool(separator and module_name and attr_name)


def _load_framework_contracts(
    paths: Iterable[str | Path],
    errors: list[str],
) -> dict[str, FrameworkAdapterContract]:
    paths = tuple(paths)
    if not paths:
        return {}
    try:
        contracts = load_framework_adapter_contracts(list(paths))
    except ValueError as exc:
        errors.append(f"framework contract validation failed: {exc}")
        return {}
    by_id: dict[str, FrameworkAdapterContract] = {}
    for contract in contracts:
        if contract.framework_id in by_id:
            errors.append(f"duplicate framework contract id: {contract.framework_id}")
            continue
        by_id[contract.framework_id] = contract
    return by_id


def _validate_against_framework_contract(
    config: dict[str, Any],
    loader: str,
    contracts: dict[str, FrameworkAdapterContract],
) -> list[str]:
    errors: list[str] = []
    framework_id = str(config.get("agent_framework") or "")
    contract = contracts.get(framework_id)
    if contract is None:
        return [f"agent_framework {framework_id!r} is not declared by supplied framework contracts"]
    if str(config.get("orchestration_mode") or "") not in contract.orchestration_modes:
        errors.append(
            f"orchestration_mode {config.get('orchestration_mode')!r} is not allowed for framework {framework_id}"
        )
    if str(config.get("memory_backend") or "") not in contract.memory_backend_ids:
        errors.append(
            f"memory_backend {config.get('memory_backend')!r} is not allowed for framework {framework_id}"
        )
    if str(config.get("tool_runtime_id") or "") != contract.tool_runtime_id:
        errors.append(
            f"tool_runtime_id {config.get('tool_runtime_id')!r} does not match framework contract {contract.tool_runtime_id!r}"
        )
    if loader != contract.adapter_entrypoint:
        errors.append(
            f"loader {loader!r} must match framework adapter entrypoint {contract.adapter_entrypoint!r}"
        )
    if str(config.get("execution_mode") or "") == "dependency_preflight" and loader != contract.adapter_entrypoint:
        errors.append(
            f"dependency_preflight loader {loader!r} must match framework adapter entrypoint {contract.adapter_entrypoint!r}"
        )
    if contract.is_implemented and loader != contract.adapter_entrypoint:
        errors.append(
            f"loader {loader!r} does not match implemented framework adapter entrypoint {contract.adapter_entrypoint!r}"
        )
    return errors


def _matched_framework_contract_id(
    config: dict[str, Any],
    contracts: dict[str, FrameworkAdapterContract],
) -> str | None:
    framework_id = str(config.get("agent_framework") or "")
    return framework_id if framework_id in contracts else None


def _artifact_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
