"""Shared types for AutoMemoryBench generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


TASK_TYPES = ("answer", "update", "forget", "governance", "tool", "planning")


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for AutoMemoryBench main dataset generation."""

    case_count_per_domain: int = 10
    seed: int = 13
    benchmark_id: str = "amst-main-generated"
    name: str = "AutoMemoryBench Main Generated"
    domains: tuple[str, ...] | None = None
    counterfactual_variants_per_case: int = 2


@dataclass(frozen=True)
class CaseGroupPlan:
    domain: str
    index: int
    case_seed: int
    counterfactual_group_id: str


@dataclass(frozen=True)
class DomainSpec:
    domain: str
    actor: str
    stable_item: str
    stable_value: str
    mutable_item: str
    old_value: str
    new_value: str
    counterfactual_new_value: str
    deletion_item: str
    deleted_value: str
    sensitive_item: str
    sensitive_value: str
    tool_name: str
    tool_result: str
    plan_goal: str
    plan_constraint: str
    procedure: str
    feedback: str
    task_result: str
    governance_rule: str
    distractor: str
    counterfactual_edit: str = "base"


@dataclass(frozen=True)
class GraphEvent:
    event_id: str
    event_type: str
    timestamp: datetime
    actor: str
    subject: str
    value: str
    memory_type: str
    privacy_level: str = "normal"
    should_store: bool = True
    should_delete: bool = False
    supersedes: str | None = None
