"""Assign planned generation case groups to release splits."""

from __future__ import annotations

import random
from typing import TypeAlias

from amb.benchmark.generation.stress import stress_profile_for_group_id
from amb.benchmark.generation.types import CaseGroupPlan
from amb.benchmark.release.splits import RELEASE_SPLITS, ReleaseConfig, split_count_mapping


PlanSplitAssignment: TypeAlias = dict[str, dict[str, tuple[CaseGroupPlan, ...]]]


def assign_case_group_plans(
    plans: tuple[CaseGroupPlan, ...],
    *,
    strategy: str,
    release_config: ReleaseConfig,
) -> PlanSplitAssignment:
    """Assign planned case groups to release splits without generating cases."""

    if strategy == "domain_stratified_group_preserving":
        return _assign_domain_stratified(plans, release_config)
    if strategy == "global_group_preserving":
        return _assign_global(plans, release_config)
    raise ValueError(f"unknown release split strategy: {strategy}")


def group_assignment_ids(split_plan: PlanSplitAssignment) -> dict[str, dict[str, list[str]]]:
    return {
        split: {
            domain: [plan.counterfactual_group_id for plan in plans]
            for domain, plans in sorted(domain_plans.items())
            if plans
        }
        for split, domain_plans in split_plan.items()
    }


def _assign_global(plans: tuple[CaseGroupPlan, ...], release_config: ReleaseConfig) -> PlanSplitAssignment:
    ordered = sorted(plans, key=lambda plan: plan.counterfactual_group_id)
    rng = random.Random(release_config.seed)
    rng.shuffle(ordered)
    split_counts = split_count_mapping(len(ordered), release_config)
    split_plan = _empty_assignment()
    cursor = 0
    for split in _ASSIGNMENT_ORDER:
        count = split_counts[split]
        selected = ordered[cursor : cursor + count]
        cursor += count
        _add_plans(split_plan, split, selected)
    return split_plan


def _assign_domain_stratified(plans: tuple[CaseGroupPlan, ...], release_config: ReleaseConfig) -> PlanSplitAssignment:
    rng = random.Random(release_config.seed)
    split_plan = _empty_assignment()
    for domain, domain_plans in sorted(_group_by_domain(plans).items()):
        ordered = sorted(domain_plans, key=lambda plan: plan.counterfactual_group_id)
        rng.shuffle(ordered)
        ordered = _prioritize_hidden_plans(ordered)
        split_counts = split_count_mapping(len(ordered), release_config)
        cursor = 0
        for split in _ASSIGNMENT_ORDER:
            count = split_counts[split]
            selected = ordered[cursor : cursor + count]
            cursor += count
            _add_plans(split_plan, split, selected)
    return split_plan


def _empty_assignment() -> PlanSplitAssignment:
    return {split: {} for split in RELEASE_SPLITS}


def _add_plans(split_plan: PlanSplitAssignment, split: str, plans: list[CaseGroupPlan]) -> None:
    for plan in plans:
        existing = split_plan[split].setdefault(plan.domain, ())
        split_plan[split][plan.domain] = (*existing, plan)


def _group_by_domain(plans: tuple[CaseGroupPlan, ...]) -> dict[str, tuple[CaseGroupPlan, ...]]:
    grouped: dict[str, list[CaseGroupPlan]] = {}
    for plan in plans:
        grouped.setdefault(plan.domain, []).append(plan)
    return {domain: tuple(items) for domain, items in sorted(grouped.items())}


def _prioritize_hidden_plans(plans: list[CaseGroupPlan]) -> list[CaseGroupPlan]:
    return sorted(plans, key=_hidden_plan_priority, reverse=True)


def _hidden_plan_priority(plan: CaseGroupPlan) -> tuple[int, int, str]:
    profile = stress_profile_for_group_id(plan.counterfactual_group_id)
    return profile.hidden_priority, len(profile.tags), plan.counterfactual_group_id


_ASSIGNMENT_ORDER = ("hidden_test", "audit_subset", "public_dev", "public_test")
