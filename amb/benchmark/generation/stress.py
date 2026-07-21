"""Deterministic scenario stress profiles for harder AutoMemoryBench groups."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

from amb.benchmark.generation.types import DomainSpec


@dataclass(frozen=True)
class StressProfile:
    family: str
    tags: tuple[str, ...]
    hidden_priority: int


def stress_profile_for_group_id(group_id: str) -> StressProfile:
    """Return a deterministic stress profile for one counterfactual group."""

    canonical_group_id = str(group_id).split(":", 1)[0]
    bucket = _stable_bucket(canonical_group_id)
    if bucket < 10:
        return StressProfile(
            family="boundary_governance_counterfactual",
            tags=("governance", "counterfactual", "cross_subject"),
            hidden_priority=4,
        )
    if bucket < 25:
        return StressProfile(
            family="cross_scope_boundary",
            tags=("governance", "cross_subject"),
            hidden_priority=3,
        )
    if bucket < 40:
        return StressProfile(
            family="governance_scope",
            tags=("governance",),
            hidden_priority=2,
        )
    if bucket < 55:
        return StressProfile(
            family="counterfactual_confusor",
            tags=("counterfactual",),
            hidden_priority=1,
        )
    return StressProfile(family="routine", tags=(), hidden_priority=0)


def apply_stress_profile(spec: DomainSpec, group_id: str) -> tuple[DomainSpec, StressProfile]:
    """Apply a deterministic stress rewrite to a domain spec."""

    profile = stress_profile_for_group_id(group_id)
    if profile.family == "routine":
        return spec, profile
    if profile.family == "counterfactual_confusor":
        return _apply_counterfactual_confusor(spec), profile
    if profile.family == "governance_scope":
        return _apply_governance_scope(spec), profile
    if profile.family == "cross_scope_boundary":
        return _apply_cross_scope_boundary(spec), profile
    if profile.family == "boundary_governance_counterfactual":
        stressed = _apply_counterfactual_confusor(spec)
        stressed = _apply_cross_scope_boundary(stressed)
        stressed = _apply_governance_scope(stressed)
        return stressed, profile
    raise ValueError(f"unknown stress family {profile.family!r}")


def _apply_counterfactual_confusor(spec: DomainSpec) -> DomainSpec:
    old_value = f"{spec.old_value} as the previous authoritative value"
    new_value = f"{spec.new_value} as the current authoritative value"
    counterfactual_new_value = f"{spec.counterfactual_new_value} as the alternate authoritative value"
    return replace(
        spec,
        mutable_item=f"authoritative {spec.mutable_item}",
        old_value=old_value,
        new_value=new_value,
        counterfactual_new_value=counterfactual_new_value,
        deleted_value=f"{spec.deleted_value} as the deprecated fallback",
        tool_result=f"{spec.tool_result}; stale notes still echo {old_value}",
        plan_constraint=f"{spec.plan_constraint}; prefer the current authoritative state over repeated stale notes",
        feedback=f"{spec.feedback}; stale previous-state notes should not override the current authoritative state",
        distractor=f"An outdated but repeated note still claims {old_value}.",
    )


def _apply_governance_scope(spec: DomainSpec) -> DomainSpec:
    deleted_item = _project_item(f"expired {spec.deletion_item}")
    deleted_value = f"{spec.deleted_value} as an expired project-only value"
    sensitive_item = _project_item(f"restricted {spec.sensitive_item}")
    sensitive_value = f"{spec.sensitive_value} as a restricted project-only value"
    return replace(
        spec,
        deletion_item=deleted_item,
        deleted_value=deleted_value,
        sensitive_item=sensitive_item,
        sensitive_value=sensitive_value,
        plan_constraint=f"{spec.plan_constraint}; do not reuse expired fallbacks or restricted project-only values",
        governance_rule=(
            f"{spec.governance_rule} Scope rule: expired fallbacks and restricted "
            "project-only values must not be reused outside the current project."
        ),
        feedback=f"{spec.feedback}; avoid expired fallbacks and restricted project-only details",
        distractor=f"A stale project-only exception suggested {deleted_value}, but it is no longer allowed.",
    )


def _apply_cross_scope_boundary(spec: DomainSpec) -> DomainSpec:
    return replace(
        spec,
        actor=f"{spec.actor} on the current project",
        stable_item=f"{spec.stable_item} for the current project",
        stable_value=f"{spec.stable_value} for the current project",
        mutable_item=f"{spec.mutable_item} for the current project",
        old_value=f"{spec.old_value} from older workspace notes",
        new_value=f"{spec.new_value} for the current project",
        counterfactual_new_value=f"{spec.counterfactual_new_value} from separate workspace notes",
        deletion_item=f"{spec.deletion_item} from an older workspace",
        deleted_value=f"{spec.deleted_value} from an older workspace",
        sensitive_item=f"{spec.sensitive_item} from an older workspace",
        sensitive_value=f"{spec.sensitive_value} from an older workspace",
        tool_result=f"{spec.tool_result}; older workspace notes contain a similar but unauthorized mention",
        plan_goal=f"{spec.plan_goal} for the current project",
        plan_constraint=f"{spec.plan_constraint}; keep current-workspace and older-workspace memory separate",
        governance_rule=(
            f"{spec.governance_rule} Boundary rule: older-workspace information is not "
            "authorized for the current project unless it is explicitly re-authorized."
        ),
        distractor=(
            f"An older-workspace note mentions a similar item for {spec.plan_goal}, "
            "but it is not authoritative for the current project."
        ),
    )


def _project_item(value: str) -> str:
    text = str(value)
    if "current project" in text or "older workspace" in text:
        return text
    return f"{text} for the current project"


def _stable_bucket(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100
