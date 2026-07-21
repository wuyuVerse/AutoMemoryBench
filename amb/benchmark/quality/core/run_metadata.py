"""Run metadata validation for reproducible AutoMemoryBench submissions."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import platform
from pathlib import Path
import re
from typing import Any

from amb.benchmark.schemas.io import read_json


REQUIRED_RUN_METADATA_FIELDS = (
    "system_id",
    "system_version",
    "benchmark_id",
    "release_split",
    "command",
    "environment",
    "dependencies",
    "timestamp",
)

REQUIRED_REAL_SYSTEM_ATTESTATION_FIELDS = (
    "provider",
    "package_name",
    "package_version",
    "client_factory",
    "credential_source",
    "config_sha256",
)

REQUIRED_AGENT_SYSTEM_METADATA_FIELDS = (
    "config_type",
    "schema_version",
    "system_id",
    "system_version",
    "provider",
    "loader",
    "config_path",
    "execution_mode",
)

REQUIRED_GENERIC_AGENT_SYSTEM_INTERPRETABILITY_FIELDS = (
    "agent_framework",
    "orchestration_mode",
    "memory_backend",
    "tool_runtime_id",
)

VALID_AGENT_SYSTEM_CONFIG_TYPES = ("generic_agent_system", "provider_integration")

FORBIDDEN_REAL_SYSTEM_FACTORY_PATTERNS = ("tests.", "integration_fixtures", "fake", "mock", "stub")
RELEASE_CONTRACT_FINGERPRINT_SCHEMA_VERSION = "amst-release-contract-fingerprint-v1"
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def load_run_metadata(path: str | Path) -> dict[str, Any]:
    metadata = read_json(path)
    if not isinstance(metadata, dict):
        raise ValueError("run metadata must be a JSON object")
    return metadata


def build_run_metadata(
    *,
    system_id: str,
    benchmark_id: str,
    release_split: str,
    command: str,
    system_version: str = "unspecified",
    environment: dict[str, Any] | None = None,
    dependencies: dict[str, Any] | None = None,
    execution_mode: str = "integration_smoke",
    integration_config_artifact: dict[str, Any] | None = None,
    real_system_attestation: dict[str, Any] | None = None,
    release_contract_fingerprint: dict[str, Any] | None = None,
    agent_system: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    metadata = {
        "system_id": system_id,
        "system_version": system_version,
        "benchmark_id": benchmark_id,
        "release_split": release_split,
        "command": command,
        "execution_mode": execution_mode,
        "environment": environment
        or {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "dependencies": dependencies or {"agent_memory_benchmark": "local"},
        "timestamp": timestamp or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    if integration_config_artifact is not None:
        metadata["integration_config_artifact"] = integration_config_artifact
    if real_system_attestation is not None:
        metadata["real_system_attestation"] = real_system_attestation
    if release_contract_fingerprint is not None:
        metadata["release_contract_fingerprint"] = release_contract_fingerprint
    if agent_system is not None:
        metadata["agent_system"] = agent_system
    return metadata


def build_real_system_attestation(
    raw_attestation: dict[str, Any] | None,
    integration_config_artifact: dict[str, Any],
    *,
    config_provider: str | None = None,
    config_client_factory: str | None = None,
) -> dict[str, Any] | None:
    """Attach the actual config artifact hash to a real-system attestation.

    The hash is stamped at runtime instead of being required inside the source
    config file; otherwise users would need to solve a self-referential SHA-256
    value for the config that contains the hash.
    """

    if raw_attestation is None:
        return None
    if not isinstance(raw_attestation, dict):
        raise ValueError("real_system_attestation must be an object")
    config_sha = integration_config_artifact.get("sha256")
    if not config_sha:
        raise ValueError("integration_config_artifact.sha256 is required for real_system_attestation")
    attestation = dict(raw_attestation)
    supplied_sha = attestation.get("config_sha256")
    if supplied_sha and supplied_sha != config_sha:
        raise ValueError("real_system_attestation.config_sha256 must match the integration config artifact; omit it to auto-stamp")
    if config_provider is not None and attestation.get("provider") and str(attestation["provider"]) != config_provider:
        raise ValueError("real_system_attestation.provider must match integration config provider")
    if (
        config_client_factory is not None
        and attestation.get("client_factory")
        and str(attestation["client_factory"]) != config_client_factory
    ):
        raise ValueError("real_system_attestation.client_factory must match integration config client_factory")
    attestation["config_sha256"] = config_sha
    return attestation


def validate_real_system_run_metadata(metadata: dict[str, Any]) -> tuple[str, ...]:
    """Validate the extra evidence required for real external memory-system runs."""

    errors: list[str] = []
    if metadata.get("execution_mode") != "real_system":
        errors.append("run_metadata.execution_mode must be real_system")
    attestation = metadata.get("real_system_attestation")
    if not isinstance(attestation, dict):
        errors.append("run_metadata.real_system_attestation is required")
        return tuple(errors)
    for field in REQUIRED_REAL_SYSTEM_ATTESTATION_FIELDS:
        if not attestation.get(field):
            errors.append(f"run_metadata.real_system_attestation.{field} is required")
    config_artifact = metadata.get("integration_config_artifact")
    if not isinstance(config_artifact, dict):
        errors.append("run_metadata.integration_config_artifact is required")
    elif config_artifact.get("sha256") != attestation.get("config_sha256"):
        errors.append("run_metadata.real_system_attestation.config_sha256 must match integration_config_artifact.sha256")
    client_factory = str(attestation.get("client_factory", "")).lower()
    if any(pattern in client_factory for pattern in FORBIDDEN_REAL_SYSTEM_FACTORY_PATTERNS):
        errors.append("run_metadata.real_system_attestation.client_factory must not reference test, fake, mock, or stub code")
    return tuple(errors)


def run_metadata_artifact(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": _sha256(source),
    }


def validate_run_metadata(
    metadata: dict[str, Any],
    *,
    system_id: str | None = None,
    benchmark_id: str | None = None,
    release_split: str | None = None,
) -> tuple[str, ...]:
    errors: list[str] = []
    for field in REQUIRED_RUN_METADATA_FIELDS:
        value = metadata.get(field)
        if value is None or value == "" or value == {} or value == []:
            errors.append(f"run_metadata.{field} is required")
    if system_id is not None and metadata.get("system_id") != system_id:
        errors.append(f"run_metadata.system_id must match report system_id {system_id!r}")
    if benchmark_id is not None and metadata.get("benchmark_id") != benchmark_id:
        errors.append(f"run_metadata.benchmark_id must match report benchmark_id {benchmark_id!r}")
    if release_split is not None and metadata.get("release_split") != release_split:
        errors.append(f"run_metadata.release_split must match report release_split {release_split!r}")
    if metadata.get("timestamp") is not None and not _valid_timestamp(str(metadata["timestamp"])):
        errors.append("run_metadata.timestamp must be ISO-like")
    if metadata.get("environment") is not None and not isinstance(metadata["environment"], dict):
        errors.append("run_metadata.environment must be an object")
    if metadata.get("dependencies") is not None and not isinstance(metadata["dependencies"], dict):
        errors.append("run_metadata.dependencies must be an object")
    errors.extend(_validate_release_contract_fingerprint(metadata.get("release_contract_fingerprint"), metadata=metadata))
    errors.extend(_validate_agent_system_metadata(metadata.get("agent_system"), metadata=metadata))
    return tuple(errors)


def _validate_release_contract_fingerprint(value: Any, *, metadata: dict[str, Any]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, dict):
        return ["run_metadata.release_contract_fingerprint must be an object"]
    errors: list[str] = []
    if value.get("schema_version") != RELEASE_CONTRACT_FINGERPRINT_SCHEMA_VERSION:
        errors.append(
            "run_metadata.release_contract_fingerprint.schema_version must be "
            f"{RELEASE_CONTRACT_FINGERPRINT_SCHEMA_VERSION}"
        )
    split = value.get("split")
    if not split:
        errors.append("run_metadata.release_contract_fingerprint.split is required")
    elif split != metadata.get("release_split"):
        errors.append("run_metadata.release_contract_fingerprint.split must match run_metadata.release_split")
    num_queries = value.get("num_queries")
    if not isinstance(num_queries, int) or num_queries < 0:
        errors.append("run_metadata.release_contract_fingerprint.num_queries must be a non-negative integer")
    for field in ("query_id_sha256", "query_contract_sha256"):
        digest = value.get(field)
        if not isinstance(digest, str) or not SHA256_HEX_RE.fullmatch(digest):
            errors.append(f"run_metadata.release_contract_fingerprint.{field} must be a 64-character sha256 hex digest")
    return errors


def _validate_agent_system_metadata(agent_system: Any, *, metadata: dict[str, Any]) -> list[str]:
    if agent_system is None:
        return []
    errors: list[str] = []
    if not isinstance(agent_system, dict):
        return ["run_metadata.agent_system must be an object"]
    for field in REQUIRED_AGENT_SYSTEM_METADATA_FIELDS:
        value = agent_system.get(field)
        if value is None or value == "" or value == {} or value == []:
            errors.append(f"run_metadata.agent_system.{field} is required")
    config_type = agent_system.get("config_type")
    if config_type not in VALID_AGENT_SYSTEM_CONFIG_TYPES:
        errors.append(
            "run_metadata.agent_system.config_type must be one of "
            f"{', '.join(VALID_AGENT_SYSTEM_CONFIG_TYPES)}"
        )
    if config_type == "generic_agent_system":
        if agent_system.get("schema_version") != "amst-agent-system-config-v1":
            errors.append("run_metadata.agent_system.schema_version must be amst-agent-system-config-v1")
        for field in REQUIRED_GENERIC_AGENT_SYSTEM_INTERPRETABILITY_FIELDS:
            value = agent_system.get(field)
            if value is None or value == "" or value == {} or value == []:
                errors.append(f"run_metadata.agent_system.{field} is required for generic_agent_system")
    for field in ("system_id", "system_version", "execution_mode"):
        if agent_system.get(field) != metadata.get(field):
            errors.append(f"run_metadata.agent_system.{field} must match run_metadata.{field}")
    return errors


def _valid_timestamp(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
