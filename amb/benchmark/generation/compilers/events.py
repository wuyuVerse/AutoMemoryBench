"""Event-graph construction for AutoMemoryBench scenarios."""

from __future__ import annotations

from datetime import datetime, timedelta

from amb.benchmark.generation.types import DomainSpec, GraphEvent


def build_event_graph(spec: DomainSpec, start: datetime, variant: int) -> tuple[GraphEvent, ...]:
    """Build the canonical event graph before rendering or probing."""

    suffix = f"v{variant:03d}"
    return (
        GraphEvent(
            f"e_stable_{suffix}",
            "fact_introduction",
            start,
            spec.actor,
            spec.stable_item,
            spec.stable_value,
            "preference",
        ),
        GraphEvent(
            f"e_stable_reinforce_{suffix}",
            "fact_reinforcement",
            start + timedelta(days=1),
            spec.actor,
            spec.stable_item,
            spec.stable_value,
            "semantic_memory",
        ),
        GraphEvent(
            f"e_old_{suffix}",
            "fact_introduction",
            start + timedelta(days=3),
            spec.actor,
            spec.mutable_item,
            spec.old_value,
            "user_fact",
        ),
        GraphEvent(
            f"e_old_reinforce_{suffix}",
            "fact_reinforcement",
            start + timedelta(days=4),
            spec.actor,
            spec.mutable_item,
            spec.old_value,
            "semantic_memory",
        ),
        GraphEvent(
            f"e_conflict_{suffix}",
            "conflict_event",
            start + timedelta(days=6),
            spec.actor,
            spec.mutable_item,
            f"unverified alternative: {spec.counterfactual_new_value}",
            "working_state_memory",
            should_store=False,
        ),
        GraphEvent(
            f"e_expiry_{suffix}",
            "expiry_event",
            start + timedelta(days=8),
            spec.actor,
            "temporary context",
            f"temporary context for {spec.plan_goal} expires before final evaluation",
            "working_state_memory",
            should_store=False,
            should_delete=True,
        ),
        GraphEvent(
            f"e_update_{suffix}",
            "fact_update",
            start + timedelta(days=10),
            spec.actor,
            spec.mutable_item,
            spec.new_value,
            "user_fact",
            supersedes=f"e_old_{suffix}",
        ),
        GraphEvent(
            f"e_near_miss_update_{suffix}",
            "near_miss_context",
            start + timedelta(days=10, hours=6),
            spec.actor,
            spec.mutable_item,
            (
                f"similar note from another workspace: {spec.old_value} was discussed elsewhere, "
                f"but it must not override {spec.new_value}"
            ),
            "context_note",
            should_store=False,
        ),
        GraphEvent(
            f"e_deleted_intro_{suffix}",
            "fact_introduction",
            start + timedelta(days=11),
            spec.actor,
            spec.deletion_item,
            spec.deleted_value,
            "ephemeral_context",
        ),
        _deletion_or_retention_event(spec, suffix, start),
        GraphEvent(
            f"e_authorization_{suffix}",
            "authorization_event",
            start + timedelta(days=12, hours=12),
            spec.actor,
            "authorization scope",
            f"{spec.sensitive_item} may be used only after the current requester explicitly approves it",
            "governance_memory",
        ),
        _sensitive_or_authorized_event(spec, suffix, start),
        GraphEvent(
            f"e_tool_{suffix}",
            "tool_result",
            start + timedelta(days=15),
            spec.tool_name,
            spec.tool_name,
            spec.tool_result,
            "tool_observation",
        ),
        GraphEvent(
            f"e_tool_outcome_{suffix}",
            "tool_outcome_event",
            start + timedelta(days=15, hours=12),
            spec.tool_name,
            spec.tool_name,
            f"confirmed outcome: {spec.tool_result}",
            "episodic_memory",
        ),
        GraphEvent(
            f"e_near_miss_tool_{suffix}",
            "near_miss_context",
            start + timedelta(days=15, hours=18),
            spec.tool_name,
            spec.tool_name,
            (
                f"similar tool note from another workspace mentioned {spec.old_value}; "
                f"the current tool result remains {spec.tool_result}"
            ),
            "context_note",
            should_store=False,
        ),
        GraphEvent(
            f"e_plan_{suffix}",
            "planning_constraint",
            start + timedelta(days=16),
            spec.actor,
            spec.plan_goal,
            spec.plan_constraint,
            "planning_constraint",
        ),
        GraphEvent(
            f"e_document_{suffix}",
            "document_note",
            start + timedelta(days=16, hours=12),
            spec.actor,
            spec.plan_goal,
            f"documented constraint: {spec.plan_constraint}",
            "semantic_memory",
        ),
        GraphEvent(
            f"e_procedure_{suffix}",
            "procedural_event",
            start + timedelta(days=17),
            spec.actor,
            "procedure",
            spec.procedure,
            "procedural_memory",
        ),
        GraphEvent(
            f"e_feedback_{suffix}",
            "feedback_event",
            start + timedelta(days=18),
            spec.actor,
            "feedback",
            spec.feedback,
            "feedback_memory",
        ),
        GraphEvent(
            f"e_near_miss_feedback_{suffix}",
            "near_miss_context",
            start + timedelta(days=18, hours=6),
            spec.actor,
            "feedback",
            (
                f"similar feedback from another workspace: ignore the old path using {spec.old_value}; "
                f"the reusable lesson is still {spec.feedback}"
            ),
            "context_note",
            should_store=False,
        ),
        GraphEvent(
            f"e_reflection_{suffix}",
            "reflective_event",
            start + timedelta(days=18, hours=12),
            spec.actor,
            "reflected lesson",
            f"future {spec.plan_goal} tasks should apply feedback before finalizing actions",
            "reflective_memory",
        ),
        GraphEvent(
            f"e_task_result_{suffix}",
            "task_result_event",
            start + timedelta(days=19),
            spec.actor,
            "task result",
            spec.task_result,
            "task_outcome",
        ),
        GraphEvent(
            f"e_governance_{suffix}",
            "governance_rule",
            start + timedelta(days=20),
            "policy",
            "memory governance",
            spec.governance_rule,
            "governance_rule",
            privacy_level="restricted",
        ),
        GraphEvent(
            f"e_platform_{suffix}",
            "platform_message",
            start + timedelta(days=20, hours=12),
            spec.actor,
            spec.plan_goal,
            f"platform follow-up confirmed that {spec.task_result}",
            "episodic_memory",
        ),
        GraphEvent(
            f"e_distractor_{suffix}",
            "distractor",
            start + timedelta(days=21),
            spec.actor,
            "distractor",
            spec.distractor,
            "context_note",
            should_store=False,
        ),
    )


def _deletion_or_retention_event(spec: DomainSpec, suffix: str, start: datetime) -> GraphEvent:
    if spec.counterfactual_edit == "retain_deleted_memory":
        return GraphEvent(
            f"e_retain_{suffix}",
            "retention_confirmation",
            start + timedelta(days=12),
            spec.actor,
            spec.deletion_item,
            spec.deleted_value,
            "ephemeral_context",
            should_store=True,
            should_delete=False,
            supersedes=f"e_deleted_intro_{suffix}",
        )
    return GraphEvent(
        f"e_delete_{suffix}",
        "deletion_request",
        start + timedelta(days=12),
        spec.actor,
        spec.deletion_item,
        spec.deleted_value,
        "ephemeral_context",
        should_store=False,
        should_delete=True,
        supersedes=f"e_deleted_intro_{suffix}",
    )


def _sensitive_or_authorized_event(spec: DomainSpec, suffix: str, start: datetime) -> GraphEvent:
    if spec.counterfactual_edit == "authorize_sensitive_memory":
        return GraphEvent(
            f"e_authorized_sensitive_{suffix}",
            "authorized_sensitive_memory",
            start + timedelta(days=13),
            spec.actor,
            spec.sensitive_item,
            spec.sensitive_value,
            "security_boundary",
            privacy_level="normal",
            should_store=True,
            should_delete=False,
        )
    return GraphEvent(
        f"e_sensitive_{suffix}",
        "sensitive_disclosure",
        start + timedelta(days=13),
        spec.actor,
        spec.sensitive_item,
        spec.sensitive_value,
        "security_boundary",
        privacy_level="sensitive",
        should_store=False,
        should_delete=True,
    )
