"""Counterfactual edits and axis metadata for AutoMemoryBench scenario variants."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

from amb.benchmark.generation.types import DomainSpec


RECOMMENDED_COUNTERFACTUAL_AXES = (
    "current_value",
    "deletion_state",
    "authorization_state",
    "tool_result",
    "role_project_boundary",
)

COUNTERFACTUAL_EDIT_BY_AXIS = {
    "current_value": "update_value",
    "deletion_state": "retain_deleted_memory",
    "authorization_state": "authorize_sensitive_memory",
    "tool_result": "tool_result",
    "role_project_boundary": "role_project_boundary",
}

COUNTERFACTUAL_RULE_FIELDS = {
    "current_value": "mutable_item.current_value",
    "deletion_state": "deletion_state",
    "authorization_state": "authorization_state",
    "tool_result": "tool_result.result",
    "role_project_boundary": "plan_constraint.project_boundary",
}

COUNTERFACTUAL_PROBE_FLIPS = {
    "current_value": ("update_probe", "stale_guard_probe"),
    "deletion_state": ("forget_probe",),
    "authorization_state": ("governance_probe",),
    "tool_result": ("tool_probe",),
    "role_project_boundary": (
        "planning_probe",
        "governed_transfer_probe",
        "scope_contrast_probe",
        "conflict_resolution_probe",
        "cross_session_synthesis_probe",
        "adversarial_state_synthesis_probe",
        "temporal_causal_reconciliation_probe",
        "policy_temporal_state_probe",
    ),
}


def counterfactual_axis_for_variant(cf_index: int) -> str:
    if cf_index <= 0:
        return "base"
    if cf_index == 1:
        return "current_value"
    if cf_index == 2:
        return "deletion_state"
    if cf_index == 3:
        return "authorization_state"
    if cf_index == 4:
        return "tool_result"
    if cf_index == 5:
        return "role_project_boundary"
    return "current_value"


def counterfactual_axes_for_variants(variants_per_case: int) -> frozenset[str]:
    return frozenset(counterfactual_axis_for_variant(index) for index in range(1, variants_per_case + 1))


def counterfactual_axis_coverage(variants_per_case: int, *, base_scenarios: int) -> dict[str, Any]:
    per_base = Counter(counterfactual_axis_for_variant(index) for index in range(1, variants_per_case + 1))
    per_base.pop("base", None)
    covered = tuple(axis for axis in RECOMMENDED_COUNTERFACTUAL_AXES if per_base.get(axis, 0) > 0)
    missing = tuple(axis for axis in RECOMMENDED_COUNTERFACTUAL_AXES if axis not in covered)
    return {
        "recommended_axes": list(RECOMMENDED_COUNTERFACTUAL_AXES),
        "covered_axes": list(covered),
        "missing_recommended_axes": list(missing),
        "variants_per_base_by_axis": dict(sorted(per_base.items())),
        "total_variants_by_axis": {
            axis: count * base_scenarios for axis, count in sorted(per_base.items())
        },
        "covers_all_recommended_axes": not missing,
    }


def counterfactual_rule_for_axis(spec: DomainSpec, axis: str) -> dict[str, Any]:
    if axis not in COUNTERFACTUAL_RULE_FIELDS:
        raise ValueError(f"unknown counterfactual axis {axis!r}")
    if axis == "current_value":
        base_value = spec.new_value
        counterfactual_value = spec.counterfactual_new_value
    elif axis == "deletion_state":
        base_value = "deleted"
        counterfactual_value = "retained"
    elif axis == "authorization_state":
        base_value = "restricted"
        counterfactual_value = "authorized"
    elif axis == "tool_result":
        base_value = spec.tool_result
        counterfactual_value = _mark_counterfactual_tool_result(spec.tool_result)
    else:
        base_value = spec.plan_constraint
        counterfactual_value = (
            f"{spec.plan_constraint}; do not transfer details across the new project boundary"
        )
    return {
        "axis": axis,
        "field": COUNTERFACTUAL_RULE_FIELDS[axis],
        "base_value": base_value,
        "counterfactual_value": counterfactual_value,
        "expected_probe_flip": list(COUNTERFACTUAL_PROBE_FLIPS[axis]),
        "expected_counterfactual_edit": COUNTERFACTUAL_EDIT_BY_AXIS[axis],
    }


def apply_counterfactual_edit(spec: DomainSpec, cf_index: int) -> DomainSpec:
    """Return a deterministic counterfactual variant of a domain spec.

    Variant 1 edits the current value of an updated fact. Variant 2 flips a
    deletion request into a retained-memory state. Variant 3 authorizes a
    sensitive memory that is otherwise forbidden. Variant 4 edits a tool
    observation. Additional variants cycle through value edits with unique
    values so large generated slices remain deterministic.
    """

    if cf_index == 1:
        return replace(spec, new_value=spec.counterfactual_new_value, counterfactual_edit="update_value")
    if cf_index == 2:
        return replace(spec, counterfactual_edit="retain_deleted_memory")
    if cf_index == 3:
        return replace(spec, counterfactual_edit="authorize_sensitive_memory")
    if cf_index == 4:
        return replace(
            spec,
            tool_result=_mark_counterfactual_tool_result(spec.tool_result),
            counterfactual_edit="tool_result",
        )
    if cf_index == 5:
        return replace(
            spec,
            actor=f"{spec.actor} in a different project boundary",
            plan_constraint=f"{spec.plan_constraint}; do not transfer details across the new project boundary",
            governance_rule=f"{spec.governance_rule} New project boundary: details from the original project are not authorized in the new project.",
            counterfactual_edit="role_project_boundary",
        )
    return replace(
        spec,
        new_value=f"{spec.counterfactual_new_value} variant {cf_index}",
        counterfactual_edit="update_value",
    )


def _mark_counterfactual_tool_result(value: str) -> str:
    text = str(value)
    if ";" not in text:
        return f"{text} after the follow-up check"
    head, tail = text.split(";", 1)
    return f"{head} after the follow-up check;{tail}"
