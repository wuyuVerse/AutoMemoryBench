"""Tool-runtime contracts for framework-comparative agent evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from amb.benchmark.artifact_contract import localize_report_contract
from amb.benchmark.schemas.io import read_json, write_json


TOOL_RUNTIME_CONTRACT_SCHEMA = "amst-tool-runtime-contract-v1"
TOOL_RUNTIME_VALIDATION_SCHEMA = "amst-tool-runtime-contract-validation-v1"
REQUIRED_CONTRACT_FIELDS = (
    "schema_version",
    "tool_runtime_id",
    "version",
    "execution_mode",
    "side_effect_policy",
    "approval_states",
    "tool_call_schema",
    "required_trace_fields",
    "standard_tools",
    "claim_boundary",
)
REQUIRED_TOOL_FIELDS = ("tool_name", "description", "parameters", "side_effect")
REQUIRED_TOOL_CALL_FIELDS = ("tool_name", "arguments", "approval_state", "result_summary")


@dataclass(frozen=True)
class ToolRuntimeContract:
    path: str
    tool_runtime_id: str
    version: str
    execution_mode: str
    side_effect_policy: str
    approval_states: tuple[str, ...]
    standard_tools: tuple[str, ...]
    raw: dict[str, Any]


def load_tool_runtime_contract(path: str | Path) -> ToolRuntimeContract:
    payload = read_json(path)
    errors = validate_tool_runtime_contract(payload)
    if errors:
        raise ValueError("; ".join(errors))
    return _normalize_contract(payload, path=str(path))


def validate_tool_runtime_contract(payload: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ("tool runtime contract must be an object",)
    for field in REQUIRED_CONTRACT_FIELDS:
        value = payload.get(field)
        if value in (None, "", [], {}):
            errors.append(f"{field} is required")
    if payload.get("schema_version") != TOOL_RUNTIME_CONTRACT_SCHEMA:
        errors.append(f"schema_version must be {TOOL_RUNTIME_CONTRACT_SCHEMA}")
    approval_states = payload.get("approval_states", [])
    if not _non_empty_str_list(approval_states):
        errors.append("approval_states must be a non-empty list of strings")
    for required_state in ("not_required", "required", "approved", "denied"):
        if required_state not in approval_states:
            errors.append(f"approval_states must include {required_state}")
    required_trace_fields = payload.get("required_trace_fields", [])
    if not _non_empty_str_list(required_trace_fields):
        errors.append("required_trace_fields must be a non-empty list of strings")
    for field in REQUIRED_TOOL_CALL_FIELDS:
        if field not in required_trace_fields:
            errors.append(f"required_trace_fields must include {field}")
    tool_call_schema = payload.get("tool_call_schema", {})
    if not isinstance(tool_call_schema, dict):
        errors.append("tool_call_schema must be an object")
    else:
        schema_required = tool_call_schema.get("required", [])
        if not isinstance(schema_required, list):
            errors.append("tool_call_schema.required must be a list")
        else:
            for field in REQUIRED_TOOL_CALL_FIELDS:
                if field not in schema_required:
                    errors.append(f"tool_call_schema.required must include {field}")
    standard_tools = payload.get("standard_tools", [])
    if not isinstance(standard_tools, list) or not standard_tools:
        errors.append("standard_tools must be a non-empty list")
    else:
        tool_names: list[str] = []
        for index, tool in enumerate(standard_tools):
            if not isinstance(tool, dict):
                errors.append(f"standard_tools[{index}] must be an object")
                continue
            for field in REQUIRED_TOOL_FIELDS:
                if tool.get(field) in (None, "", [], {}):
                    errors.append(f"standard_tools[{index}].{field} is required")
            tool_name = tool.get("tool_name")
            if isinstance(tool_name, str):
                tool_names.append(tool_name)
        duplicates = sorted({name for name in tool_names if tool_names.count(name) > 1})
        if duplicates:
            errors.append(f"standard_tools contain duplicate tool_name values: {duplicates}")
    claim_boundary = str(payload.get("claim_boundary", "")).casefold()
    if "not" not in claim_boundary and "不能" not in claim_boundary:
        errors.append("claim_boundary must state what is not claimable")
    return tuple(errors)


def validate_tool_runtime_contract_file(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = read_json(source)
    except Exception as exc:
        return _validation_report(source, None, (f"cannot read tool runtime contract: {exc}",))
    errors = validate_tool_runtime_contract(payload)
    contract = _normalize_contract(payload, path=str(source)) if not errors else None
    return _validation_report(source, contract, errors)


def write_tool_runtime_contract_validation(path: str | Path, output: str | Path) -> dict[str, Any]:
    report = validate_tool_runtime_contract_file(path)
    report = localize_report_contract(
        report,
        output_path=output,
        project_root_hints=(Path.cwd() / "reports", path),
    )
    write_json(output, report)
    return report


def validate_tool_call_against_runtime(tool_call: dict[str, Any], contract: ToolRuntimeContract) -> tuple[str, ...]:
    errors: list[str] = []
    if not isinstance(tool_call, dict):
        return ("tool call must be an object",)
    for field in REQUIRED_TOOL_CALL_FIELDS:
        if tool_call.get(field) in (None, ""):
            errors.append(f"tool_call.{field} is required")
    if tool_call.get("tool_name") not in contract.standard_tools:
        errors.append(f"tool_call.tool_name is not in runtime standard tools: {tool_call.get('tool_name')}")
    if tool_call.get("approval_state") not in contract.approval_states:
        errors.append(f"tool_call.approval_state is not allowed: {tool_call.get('approval_state')}")
    if "arguments" in tool_call and not isinstance(tool_call["arguments"], dict):
        errors.append("tool_call.arguments must be an object")
    return tuple(errors)


def _validation_report(source: Path, contract: ToolRuntimeContract | None, errors: tuple[str, ...]) -> dict[str, Any]:
    return {
        "schema_version": TOOL_RUNTIME_VALIDATION_SCHEMA,
        "status": "passed" if not errors else "failed",
        "path": str(source),
        "tool_runtime_id": contract.tool_runtime_id if contract else None,
        "version": contract.version if contract else None,
        "execution_mode": contract.execution_mode if contract else None,
        "side_effect_policy": contract.side_effect_policy if contract else None,
        "num_standard_tools": len(contract.standard_tools) if contract else 0,
        "approval_states": list(contract.approval_states) if contract else [],
        "errors": list(errors),
    }


def _normalize_contract(payload: dict[str, Any], *, path: str) -> ToolRuntimeContract:
    return ToolRuntimeContract(
        path=path,
        tool_runtime_id=str(payload["tool_runtime_id"]),
        version=str(payload["version"]),
        execution_mode=str(payload["execution_mode"]),
        side_effect_policy=str(payload["side_effect_policy"]),
        approval_states=tuple(str(row) for row in payload["approval_states"]),
        standard_tools=tuple(str(row["tool_name"]) for row in payload["standard_tools"]),
        raw=dict(payload),
    )


def _non_empty_str_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(row, str) and row for row in value)
