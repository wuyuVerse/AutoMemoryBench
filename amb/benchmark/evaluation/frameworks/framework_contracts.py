"""Contracts for agent-framework adapter readiness and claim boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from amb.benchmark.schemas.io import read_json
from amb.benchmark.artifact_contract import localize_report_contract
from amb.benchmark.schemas.io import write_json


FRAMEWORK_ADAPTER_CONTRACT_SCHEMA = "amst-agent-framework-adapter-contract-v1"
IMPLEMENTED_STATUSES = {"t0_smoke", "t1_ready", "t2_ready", "t3_ready", "t4_ready", "t5_ready"}
TIER_ORDER = ("T0", "T1", "T2", "T3", "T4", "T5")
REQUIRED_FIELDS = (
    "schema_version",
    "framework_id",
    "framework_label",
    "adapter_status",
    "adapter_entrypoint",
    "framework_runtime",
    "orchestration_modes",
    "memory_backend_ids",
    "tool_runtime_id",
    "supported_tiers",
    "dependency_status",
    "claim_boundary",
)


@dataclass(frozen=True)
class FrameworkAdapterContract:
    """Normalized contract for one framework adapter row."""

    path: str
    framework_id: str
    framework_label: str
    adapter_status: str
    adapter_entrypoint: str
    framework_runtime: str
    orchestration_modes: tuple[str, ...]
    memory_backend_ids: tuple[str, ...]
    tool_runtime_id: str
    supported_tiers: tuple[str, ...]
    dependency_status: str
    claim_boundary: str
    raw: dict[str, Any]

    @property
    def max_tier(self) -> str:
        ordered = [tier for tier in TIER_ORDER if tier in self.supported_tiers]
        return ordered[-1] if ordered else "none"

    @property
    def is_implemented(self) -> bool:
        return self.adapter_status in IMPLEMENTED_STATUSES


def load_framework_adapter_contract(path: str | Path) -> FrameworkAdapterContract:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"framework adapter contract must be a JSON object: {path}")
    errors = validate_framework_adapter_contract(payload, require_importable=Path(path).exists())
    if errors:
        raise ValueError("; ".join(errors))
    return _normalize_contract(payload, path=str(path))


def validate_framework_adapter_contract(
    payload: dict[str, Any],
    *,
    require_importable: bool = False,
) -> tuple[str, ...]:
    """Validate one framework adapter contract.

    Planned contracts are allowed to name a future adapter entrypoint. Import
    checks only apply once ``adapter_status`` marks the contract implemented.
    """

    errors: list[str] = []
    if not isinstance(payload, dict):
        return ("contract must be an object",)
    for field in REQUIRED_FIELDS:
        value = payload.get(field)
        if value in (None, "", []):
            errors.append(f"{field} is required")
    if payload.get("schema_version") != FRAMEWORK_ADAPTER_CONTRACT_SCHEMA:
        errors.append(f"schema_version must be {FRAMEWORK_ADAPTER_CONTRACT_SCHEMA}")
    for field in ("orchestration_modes", "memory_backend_ids", "supported_tiers"):
        if field in payload and not _non_empty_str_list(payload[field]):
            errors.append(f"{field} must be a non-empty list of strings")
    tiers = tuple(str(tier) for tier in payload.get("supported_tiers", []) if isinstance(tier, str))
    invalid_tiers = [tier for tier in tiers if tier not in TIER_ORDER]
    if invalid_tiers:
        errors.append(f"unsupported tiers: {invalid_tiers}")
    if tiers and tuple(tier for tier in TIER_ORDER if tier in tiers) != tiers:
        errors.append("supported_tiers must be sorted by maturity")
    claim_boundary = str(payload.get("claim_boundary", "")).casefold()
    if "not" not in claim_boundary and "不能" not in claim_boundary:
        errors.append("claim_boundary must state what is not claimable")
    adapter_status = str(payload.get("adapter_status", ""))
    dependency_status = str(payload.get("dependency_status", ""))
    if adapter_status in IMPLEMENTED_STATUSES and dependency_status not in {"packaged", "installed", "available"}:
        errors.append("implemented adapter contracts require packaged/installed/available dependency_status")
    entrypoint = str(payload.get("adapter_entrypoint", ""))
    if adapter_status in IMPLEMENTED_STATUSES and require_importable:
        errors.extend(_validate_entrypoint(entrypoint))
    return tuple(errors)


def load_framework_adapter_contracts(paths: list[str | Path]) -> list[FrameworkAdapterContract]:
    return [load_framework_adapter_contract(path) for path in paths]


def validate_framework_adapter_contract_files(paths: list[str | Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    valid_contracts: list[FrameworkAdapterContract] = []
    for path in paths:
        source = Path(path)
        try:
            payload = read_json(source)
        except Exception as exc:
            rows.append(
                {
                    "path": str(source),
                    "status": "failed",
                    "framework_id": None,
                    "adapter_status": None,
                    "dependency_status": None,
                    "errors": [f"cannot read framework adapter contract: {exc}"],
                }
            )
            continue
        if not isinstance(payload, dict):
            rows.append(
                {
                    "path": str(source),
                    "status": "failed",
                    "framework_id": None,
                    "adapter_status": None,
                    "dependency_status": None,
                    "errors": ["framework adapter contract must be a JSON object"],
                }
            )
            continue
        errors = list(validate_framework_adapter_contract(payload, require_importable=source.exists()))
        rows.append(
            {
                "path": str(source),
                "status": "passed" if not errors else "failed",
                "framework_id": payload.get("framework_id"),
                "adapter_status": payload.get("adapter_status"),
                "dependency_status": payload.get("dependency_status"),
                "supported_tiers": payload.get("supported_tiers"),
                "adapter_entrypoint": payload.get("adapter_entrypoint"),
                "errors": errors,
            }
        )
        if not errors:
            valid_contracts.append(_normalize_contract(payload, path=str(source)))
    errors = [error for row in rows for error in row["errors"]]
    summary = summarize_framework_adapter_contracts(valid_contracts)
    return {
        "schema_version": "amst-agent-framework-adapter-contract-validation-v1",
        "status": "passed" if not errors else "failed",
        **summary,
        "num_contract_validation_errors": len(errors),
        "contracts": rows,
        "errors": errors,
    }


def write_framework_adapter_contract_validation(paths: list[str | Path], output: str | Path) -> dict[str, Any]:
    report = validate_framework_adapter_contract_files(paths)
    report = localize_report_contract(
        report,
        output_path=output,
        project_root_hints=(Path.cwd() / "reports", *tuple(paths)),
    )
    write_json(output, report)
    return report


def summarize_framework_adapter_contracts(contracts: list[FrameworkAdapterContract]) -> dict[str, Any]:
    return {
        "num_contracts": len(contracts),
        "num_implemented_contracts": sum(contract.is_implemented for contract in contracts),
        "num_t0_or_higher_contracts": sum("T0" in contract.supported_tiers for contract in contracts),
        "num_t1_or_higher_contracts": sum(_tier_at_least(contract.max_tier, "T1") for contract in contracts),
        "framework_ids": [contract.framework_id for contract in contracts],
        "implemented_framework_ids": [contract.framework_id for contract in contracts if contract.is_implemented],
        "max_tiers": {contract.framework_id: contract.max_tier for contract in contracts},
    }


def _normalize_contract(payload: dict[str, Any], *, path: str) -> FrameworkAdapterContract:
    return FrameworkAdapterContract(
        path=path,
        framework_id=str(payload["framework_id"]),
        framework_label=str(payload["framework_label"]),
        adapter_status=str(payload["adapter_status"]),
        adapter_entrypoint=str(payload["adapter_entrypoint"]),
        framework_runtime=str(payload["framework_runtime"]),
        orchestration_modes=tuple(str(row) for row in payload["orchestration_modes"]),
        memory_backend_ids=tuple(str(row) for row in payload["memory_backend_ids"]),
        tool_runtime_id=str(payload["tool_runtime_id"]),
        supported_tiers=tuple(str(row) for row in payload["supported_tiers"]),
        dependency_status=str(payload["dependency_status"]),
        claim_boundary=str(payload["claim_boundary"]),
        raw=dict(payload),
    )


def _non_empty_str_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(row, str) and row for row in value)


def _validate_entrypoint(entrypoint: str) -> list[str]:
    module_name, separator, attr_name = entrypoint.partition(":")
    if not separator or not module_name or not attr_name:
        return ["adapter_entrypoint must use 'module:callable' format"]
    try:
        module = import_module(module_name)
    except Exception as exc:  # pragma: no cover - exact dependency errors vary by environment.
        return [f"adapter_entrypoint module is not importable: {exc}"]
    factory = getattr(module, attr_name, None)
    if not callable(factory):
        return [f"adapter_entrypoint callable is missing: {entrypoint}"]
    return []


def _tier_at_least(actual: str, minimum: str) -> bool:
    if actual not in TIER_ORDER or minimum not in TIER_ORDER:
        return False
    return TIER_ORDER.index(actual) >= TIER_ORDER.index(minimum)
