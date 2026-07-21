"""Dependency-gate contracts for planned framework adapters."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from importlib import metadata as importlib_metadata
import importlib.util
from pathlib import Path
from typing import Any

from amb.benchmark.artifact_contract import localize_report_contract
from amb.benchmark.evaluation.framework_contracts import load_framework_adapter_contract
from amb.benchmark.schemas.io import read_json, write_json


FRAMEWORK_DEPENDENCY_GATE_SCHEMA = "amst-agent-framework-dependency-gate-contract-v1"
FRAMEWORK_DEPENDENCY_GATE_VALIDATION_SCHEMA = "amst-agent-framework-dependency-gate-validation-v1"
FRAMEWORK_DEPENDENCY_LOCK_PLAN_SCHEMA = "amst-agent-framework-dependency-lock-plan-v1"
DEPENDENCY_LOCK_STATUSES = {"not_locked", "locked"}
PACKAGE_PIN_STATUSES = {"unlocked", "pinned"}
LOCK_PLAN_STATUSES = {"not_locked", "locked"}
LOCK_PLAN_PIN_STATUSES = {"missing", "pinned"}
LOCK_PLAN_ACCEPTANCE_STATUSES = {"pending", "passed"}
REQUIRED_LOCK_PLAN_ACCEPTANCE_STAGES = (
    "resolve_version_pin",
    "materialize_lock_artifact",
    "update_dependency_gate_to_locked",
    "installed_runtime_probe_passes",
    "runnable_scoring_config_passes",
    "fresh_t1_plus_rerun_bound",
)


@dataclass(frozen=True)
class FrameworkDependencyGate:
    """One dependency gate row for a planned adapter."""

    framework_id: str
    adapter_entrypoint: str
    framework_contract_path: str
    dependency_preflight_config_path: str
    dependency_status: str
    dependency_lock_status: str
    required_modules: tuple[str, ...]
    package_specs: tuple[dict[str, Any], ...]
    install_hint: str
    raw: dict[str, Any]


def validate_dependency_gate_contract(payload: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ("dependency gate contract must be an object",)
    if payload.get("schema_version") != FRAMEWORK_DEPENDENCY_GATE_SCHEMA:
        errors.append(f"schema_version must be {FRAMEWORK_DEPENDENCY_GATE_SCHEMA}")
    if payload.get("dependency_lock_status") not in DEPENDENCY_LOCK_STATUSES:
        errors.append("dependency_lock_status must be not_locked or locked")
    claim_boundary = str(payload.get("claim_boundary") or "").casefold()
    if "not" not in claim_boundary and "不能" not in claim_boundary:
        errors.append("claim_boundary must state what is not claimable")
    frameworks = payload.get("frameworks")
    if not isinstance(frameworks, list) or not frameworks:
        errors.append("frameworks must be a non-empty list")
        return tuple(errors)
    seen: set[str] = set()
    for index, row in enumerate(frameworks):
        prefix = f"frameworks[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{prefix} must be an object")
            continue
        framework_id = str(row.get("framework_id") or "")
        if not framework_id:
            errors.append(f"{prefix}.framework_id is required")
        elif framework_id in seen:
            errors.append(f"duplicate framework_id: {framework_id}")
        seen.add(framework_id)
        for field in (
            "adapter_entrypoint",
            "framework_contract_path",
            "dependency_preflight_config_path",
            "dependency_status",
            "dependency_lock_status",
            "install_hint",
        ):
            if row.get(field) in (None, "", [], {}):
                errors.append(f"{prefix}.{field} is required")
        if row.get("dependency_lock_status") not in DEPENDENCY_LOCK_STATUSES:
            errors.append(f"{prefix}.dependency_lock_status must be not_locked or locked")
        required_modules = row.get("required_modules")
        if not _non_empty_str_list(required_modules):
            errors.append(f"{prefix}.required_modules must be a non-empty list of strings")
        package_specs = row.get("package_specs")
        if not isinstance(package_specs, list) or not package_specs:
            errors.append(f"{prefix}.package_specs must be a non-empty list")
            continue
        import_modules = set(required_modules or [])
        package_import_modules: set[str] = set()
        for package_index, package in enumerate(package_specs):
            package_prefix = f"{prefix}.package_specs[{package_index}]"
            if not isinstance(package, dict):
                errors.append(f"{package_prefix} must be an object")
                continue
            for field in ("package", "import_module", "pin_status"):
                if package.get(field) in (None, ""):
                    errors.append(f"{package_prefix}.{field} is required")
            if package.get("pin_status") not in PACKAGE_PIN_STATUSES:
                errors.append(f"{package_prefix}.pin_status must be unlocked or pinned")
            if row.get("dependency_lock_status") == "locked" and package.get("pin_status") != "pinned":
                errors.append(f"{package_prefix}.pin_status must be pinned when dependency_lock_status=locked")
            if package.get("pin_status") == "pinned" and not package.get("version_pin"):
                errors.append(f"{package_prefix}.version_pin is required when pin_status=pinned")
            if isinstance(package.get("import_module"), str):
                package_import_modules.add(package["import_module"])
        missing_package_specs = sorted(import_modules - package_import_modules)
        if missing_package_specs:
            errors.append(f"{prefix}.package_specs missing import modules: {missing_package_specs}")
    return tuple(errors)


def load_dependency_gate_contract(path: str | Path) -> list[FrameworkDependencyGate]:
    payload = read_json(path)
    errors = validate_dependency_gate_contract(payload)
    if errors:
        raise ValueError("; ".join(errors))
    return [_normalize_gate(row) for row in payload["frameworks"]]


def validate_dependency_gate_contract_file(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = read_json(source)
    except Exception as exc:
        return _validation_report(source, (), (f"cannot read dependency gate contract: {exc}",))
    errors = validate_dependency_gate_contract(payload)
    gates = () if errors else tuple(_normalize_gate(row) for row in payload["frameworks"])
    return _validation_report(source, gates, errors)


def write_dependency_gate_contract_validation(path: str | Path, output: str | Path) -> dict[str, Any]:
    report = validate_dependency_gate_contract_file(path)
    report = localize_report_contract(
        report,
        output_path=output,
        project_root_hints=(Path.cwd() / "reports", path),
    )
    write_json(output, report)
    return report


def validate_dependency_lock_plan(payload: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ("dependency lock plan must be an object",)
    if payload.get("schema_version") != FRAMEWORK_DEPENDENCY_LOCK_PLAN_SCHEMA:
        errors.append(f"schema_version must be {FRAMEWORK_DEPENDENCY_LOCK_PLAN_SCHEMA}")
    claim_boundary = str(payload.get("claim_boundary") or "").casefold()
    if "not" not in claim_boundary and "不能" not in claim_boundary:
        errors.append("claim_boundary must state what is not claimable")
    required_stages = payload.get("required_acceptance_stages")
    if tuple(required_stages or ()) != REQUIRED_LOCK_PLAN_ACCEPTANCE_STAGES:
        errors.append("required_acceptance_stages must match the canonical lock-plan stage order")
    frameworks = payload.get("frameworks")
    if not isinstance(frameworks, list) or not frameworks:
        errors.append("frameworks must be a non-empty list")
        return tuple(errors)
    seen: set[str] = set()
    for index, row in enumerate(frameworks):
        prefix = f"frameworks[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{prefix} must be an object")
            continue
        framework_id = str(row.get("framework_id") or "")
        if not framework_id:
            errors.append(f"{prefix}.framework_id is required")
        elif framework_id in seen:
            errors.append(f"duplicate framework_id: {framework_id}")
        seen.add(framework_id)
        for field in (
            "dependency_gate_contract_path",
            "target_lock_artifact_path",
            "target_runnable_config_path",
            "target_prediction_artifact_path",
            "target_run_metadata_artifact_path",
            "target_trace_artifact_path",
            "target_t1_plus_report_path",
            "lock_status",
        ):
            if row.get(field) in (None, "", [], {}):
                errors.append(f"{prefix}.{field} is required")
        if row.get("lock_status") not in LOCK_PLAN_STATUSES:
            errors.append(f"{prefix}.lock_status must be not_locked or locked")
        package_pins = row.get("package_pins")
        if not isinstance(package_pins, list) or not package_pins:
            errors.append(f"{prefix}.package_pins must be a non-empty list")
        else:
            for package_index, package in enumerate(package_pins):
                package_prefix = f"{prefix}.package_pins[{package_index}]"
                if not isinstance(package, dict):
                    errors.append(f"{package_prefix} must be an object")
                    continue
                for field in ("package", "import_module", "pin_status"):
                    if package.get(field) in (None, ""):
                        errors.append(f"{package_prefix}.{field} is required")
                if package.get("pin_status") not in LOCK_PLAN_PIN_STATUSES:
                    errors.append(f"{package_prefix}.pin_status must be missing or pinned")
                if package.get("pin_status") == "pinned" and not package.get("version_pin"):
                    errors.append(f"{package_prefix}.version_pin is required when pin_status=pinned")
                if row.get("lock_status") == "locked" and package.get("pin_status") != "pinned":
                    errors.append(f"{package_prefix}.pin_status must be pinned when lock_status=locked")
        stages = row.get("acceptance_stages")
        if not isinstance(stages, list) or not stages:
            errors.append(f"{prefix}.acceptance_stages must be a non-empty list")
            continue
        stage_ids = [stage.get("stage_id") for stage in stages if isinstance(stage, dict)]
        if tuple(stage_ids) != REQUIRED_LOCK_PLAN_ACCEPTANCE_STAGES:
            errors.append(f"{prefix}.acceptance_stages must match the canonical lock-plan stage order")
        for stage_index, stage in enumerate(stages):
            stage_prefix = f"{prefix}.acceptance_stages[{stage_index}]"
            if not isinstance(stage, dict):
                errors.append(f"{stage_prefix} must be an object")
                continue
            if stage.get("status") not in LOCK_PLAN_ACCEPTANCE_STATUSES:
                errors.append(f"{stage_prefix}.status must be pending or passed")
            if stage.get("required_evidence") in (None, ""):
                errors.append(f"{stage_prefix}.required_evidence is required")
        if row.get("lock_status") == "locked" and any(
            isinstance(stage, dict) and stage.get("status") != "passed" for stage in stages
        ):
            errors.append(f"{prefix}.acceptance_stages must all pass when lock_status=locked")
    return tuple(errors)


def build_dependency_lock_plan_rows(
    *,
    root: Path,
    gates: list[FrameworkDependencyGate],
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    del root
    gate_by_framework = {gate.framework_id: gate for gate in gates}
    rows: list[dict[str, Any]] = []
    for plan_row in plan.get("frameworks", []):
        if not isinstance(plan_row, dict):
            continue
        framework_id = str(plan_row.get("framework_id") or "")
        gate = gate_by_framework.get(framework_id)
        package_pins = list(plan_row.get("package_pins") or [])
        acceptance_stages = list(plan_row.get("acceptance_stages") or [])
        gate_packages = {package.get("package") for package in gate.package_specs} if gate else set()
        gate_import_modules = {package.get("import_module") for package in gate.package_specs} if gate else set()
        plan_packages = {package.get("package") for package in package_pins if isinstance(package, dict)}
        plan_import_modules = {
            package.get("import_module") for package in package_pins if isinstance(package, dict)
        }
        stage_ids = [stage.get("stage_id") for stage in acceptance_stages if isinstance(stage, dict)]
        num_missing_version_pins = sum(
            not package.get("version_pin") for package in package_pins if isinstance(package, dict)
        )
        num_passed_stages = sum(
            stage.get("status") == "passed" for stage in acceptance_stages if isinstance(stage, dict)
        )
        all_pins_resolved = bool(package_pins) and num_missing_version_pins == 0 and all(
            package.get("pin_status") == "pinned" for package in package_pins if isinstance(package, dict)
        )
        all_stages_passed = (
            tuple(stage_ids) == REQUIRED_LOCK_PLAN_ACCEPTANCE_STAGES
            and num_passed_stages == len(REQUIRED_LOCK_PLAN_ACCEPTANCE_STAGES)
        )
        rows.append(
            {
                "framework_id": framework_id,
                "dependency_gate_contract_path": plan_row.get("dependency_gate_contract_path"),
                "target_lock_artifact_path": plan_row.get("target_lock_artifact_path"),
                "target_runnable_config_path": plan_row.get("target_runnable_config_path"),
                "target_prediction_artifact_path": plan_row.get("target_prediction_artifact_path"),
                "target_run_metadata_artifact_path": plan_row.get("target_run_metadata_artifact_path"),
                "target_trace_artifact_path": plan_row.get("target_trace_artifact_path"),
                "target_t1_plus_report_path": plan_row.get("target_t1_plus_report_path"),
                "lock_status": plan_row.get("lock_status"),
                "package_pins": package_pins,
                "acceptance_stages": acceptance_stages,
                "gate_row_present": gate is not None,
                "gate_packages_covered": gate is not None and gate_packages <= plan_packages,
                "gate_import_modules_covered": gate is not None and gate_import_modules <= plan_import_modules,
                "canonical_acceptance_stages_present": tuple(stage_ids) == REQUIRED_LOCK_PLAN_ACCEPTANCE_STAGES,
                "num_package_pin_requirements": len(package_pins),
                "num_missing_version_pins": num_missing_version_pins,
                "num_acceptance_stages": len(acceptance_stages),
                "num_passed_acceptance_stages": num_passed_stages,
                "lock_ready": bool(plan_row.get("lock_status") == "locked" and all_pins_resolved and all_stages_passed),
            }
        )
    return rows


def summarize_dependency_lock_plan_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_dependency_lock_plan_rows": len(rows),
        "num_gate_covered_lock_plan_rows": sum(row["gate_row_present"] for row in rows),
        "num_gate_package_covered_lock_plan_rows": sum(row["gate_packages_covered"] for row in rows),
        "num_gate_import_covered_lock_plan_rows": sum(row["gate_import_modules_covered"] for row in rows),
        "num_package_pin_requirements": sum(row["num_package_pin_requirements"] for row in rows),
        "num_missing_version_pins": sum(row["num_missing_version_pins"] for row in rows),
        "num_required_acceptance_stages": len(rows) * len(REQUIRED_LOCK_PLAN_ACCEPTANCE_STAGES),
        "num_pending_acceptance_stages": sum(
            row["num_acceptance_stages"] - row["num_passed_acceptance_stages"] for row in rows
        ),
        "num_passed_acceptance_stages": sum(row["num_passed_acceptance_stages"] for row in rows),
        "num_lock_ready_frameworks": sum(row["lock_ready"] for row in rows),
    }


def build_dependency_gate_rows(*, root: Path, gates: list[FrameworkDependencyGate]) -> list[dict[str, Any]]:
    return [_dependency_row(root=root, gate=gate) for gate in gates]


def summarize_dependency_gate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_dependency_gate_rows": len(rows),
        "num_spec_bound_dependency_gates": sum(row["adapter_spec_matches_dependency_gate"] for row in rows),
        "num_contract_bound_dependency_gates": sum(row["framework_contract_matches_dependency_gate"] for row in rows),
        "num_config_bound_dependency_gates": sum(row["dependency_preflight_config_matches_dependency_gate"] for row in rows),
        "num_unlocked_dependency_gates": sum(row["dependency_lock_status"] == "not_locked" for row in rows),
        "num_import_modules_declared": sum(len(row["required_modules"]) for row in rows),
        "num_currently_importable_modules": sum(len(row["currently_importable_modules"]) for row in rows),
    }


def probe_dependency_gate_runtime(*, root: Path, gates: list[FrameworkDependencyGate]) -> list[dict[str, Any]]:
    del root  # Reserved for future environment/lockfile-relative probes.
    rows: list[dict[str, Any]] = []
    for gate in gates:
        for package in gate.package_specs:
            import_module = str(package.get("import_module") or "")
            package_name = str(package.get("package") or "")
            pin_status = str(package.get("pin_status") or "")
            version_pin = package.get("version_pin")
            installed_version = package_version(package_name)
            module_importable = module_available(import_module)
            version_matches_pin = bool(
                pin_status == "pinned"
                and version_pin
                and installed_version
                and installed_version == str(version_pin)
            )
            installed_runtime_claimable = bool(
                gate.dependency_lock_status == "locked"
                and pin_status == "pinned"
                and module_importable
                and version_matches_pin
            )
            rows.append(
                {
                    "framework_id": gate.framework_id,
                    "adapter_entrypoint": gate.adapter_entrypoint,
                    "dependency_lock_status": gate.dependency_lock_status,
                    "package": package_name,
                    "import_module": import_module,
                    "pin_status": pin_status,
                    "version_pin": version_pin,
                    "module_importable": module_importable,
                    "installed_version": installed_version,
                    "version_matches_pin": version_matches_pin,
                    "installed_runtime_claimable": installed_runtime_claimable,
                }
            )
    return rows


def summarize_dependency_runtime_probe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    locked_frameworks = {
        row["framework_id"] for row in rows if row["dependency_lock_status"] == "locked"
    }
    return {
        "num_runtime_probe_rows": len(rows),
        "num_importable_modules": sum(row["module_importable"] for row in rows),
        "num_installed_packages_detected": sum(row["installed_version"] is not None for row in rows),
        "num_locked_dependency_rows": len(locked_frameworks),
        "num_pinned_package_specs": sum(row["pin_status"] == "pinned" for row in rows),
        "num_installed_runtime_claimable_rows": sum(row["installed_runtime_claimable"] for row in rows),
    }


def load_adapter_dependency_spec(entrypoint: str) -> dict[str, Any]:
    module_name, separator, _attr_name = entrypoint.partition(":")
    if not separator:
        return {}
    module = importlib.import_module(module_name)
    spec = getattr(module, "SPEC", None)
    return {
        "framework_id": getattr(spec, "framework_id", None),
        "framework_label": getattr(spec, "framework_label", None),
        "required_modules": tuple(getattr(spec, "required_modules", ())),
        "install_hint": getattr(spec, "install_hint", None),
        "contract_path": getattr(spec, "contract_path", None),
    }


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _normalize_gate(row: dict[str, Any]) -> FrameworkDependencyGate:
    return FrameworkDependencyGate(
        framework_id=str(row["framework_id"]),
        adapter_entrypoint=str(row["adapter_entrypoint"]),
        framework_contract_path=str(row["framework_contract_path"]),
        dependency_preflight_config_path=str(row["dependency_preflight_config_path"]),
        dependency_status=str(row["dependency_status"]),
        dependency_lock_status=str(row["dependency_lock_status"]),
        required_modules=tuple(str(module) for module in row["required_modules"]),
        package_specs=tuple(dict(package) for package in row["package_specs"]),
        install_hint=str(row["install_hint"]),
        raw=dict(row),
    )


def _validation_report(
    source: Path,
    gates: tuple[FrameworkDependencyGate, ...],
    errors: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": FRAMEWORK_DEPENDENCY_GATE_VALIDATION_SCHEMA,
        "status": "passed" if not errors else "failed",
        "path": str(source),
        "num_dependency_gate_rows": len(gates),
        "num_unlocked_dependency_gates": sum(gate.dependency_lock_status == "not_locked" for gate in gates),
        "num_import_modules_declared": sum(len(gate.required_modules) for gate in gates),
        "framework_ids": [gate.framework_id for gate in gates],
        "dependency_lock_statuses": {
            gate.framework_id: gate.dependency_lock_status for gate in gates
        },
        "errors": list(errors),
    }


def _dependency_row(*, root: Path, gate: FrameworkDependencyGate) -> dict[str, Any]:
    spec = load_adapter_dependency_spec(gate.adapter_entrypoint)
    framework_contract = load_framework_adapter_contract(root / gate.framework_contract_path)
    preflight_config = read_json(root / gate.dependency_preflight_config_path)
    return {
        "framework_id": gate.framework_id,
        "adapter_entrypoint": gate.adapter_entrypoint,
        "framework_contract_path": gate.framework_contract_path,
        "dependency_preflight_config_path": gate.dependency_preflight_config_path,
        "dependency_status": gate.dependency_status,
        "dependency_lock_status": gate.dependency_lock_status,
        "required_modules": list(gate.required_modules),
        "package_specs": list(gate.package_specs),
        "install_hint": gate.install_hint,
        "currently_importable_modules": [module for module in gate.required_modules if module_available(module)],
        "adapter_spec_matches_dependency_gate": (
            spec.get("framework_id") == gate.framework_id
            and spec.get("contract_path") == gate.framework_contract_path
            and tuple(spec.get("required_modules", ())) == gate.required_modules
            and spec.get("install_hint") == gate.install_hint
        ),
        "framework_contract_matches_dependency_gate": (
            framework_contract.framework_id == gate.framework_id
            and framework_contract.adapter_entrypoint == gate.adapter_entrypoint
            and framework_contract.dependency_status == gate.dependency_status
        ),
        "dependency_preflight_config_matches_dependency_gate": (
            preflight_config.get("agent_framework") == gate.framework_id
            and preflight_config.get("loader") == gate.adapter_entrypoint
            and preflight_config.get("execution_mode") == "dependency_preflight"
        ),
    }


def _non_empty_str_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(row, str) and row for row in value)
