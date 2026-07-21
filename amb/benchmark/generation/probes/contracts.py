"""Shared probe-contract metadata for AutoMemoryBench generation and audits."""

from __future__ import annotations


PROBE_SCORING_RULES = {
    "answer_probe": "must_include_and_must_not_include",
    "update_probe": "current_value_without_stale_value",
    "forget_probe": "deletion_state_response",
    "governance_probe": "authorization_state_response",
    "tool_probe": "tool_exact_and_parameter_f1",
    "planning_probe": "plan_constraint_satisfaction",
    "write_probe": "write_set_f1_and_required_content",
    "retrieval_probe": "retrieval_evidence_complete_without_stale",
    "compression_probe": "summary_preserves_required_and_excludes_forbidden",
    "evolution_probe": "procedural_feedback_reuse",
    "governed_transfer_probe": "governed_stateful_transfer",
    "scope_contrast_probe": "governed_scope_contrast",
    "conflict_resolution_probe": "governed_conflict_resolution",
    "cross_session_synthesis_probe": "cross_session_governed_synthesis",
    "adversarial_state_synthesis_probe": "instruction_resistant_governed_synthesis",
    "temporal_causal_reconciliation_probe": "temporal_causal_state_reconciliation",
    "policy_temporal_state_probe": "policy_temporal_state_reconciliation",
    "policy_exception_probe": "policy_exception_state_reconciliation",
    "state_transition_audit_probe": "state_transition_audit_reconciliation",
    "no_memory_probe": "no_memory_answer",
}


PROBE_TASK_TYPES = {
    "answer_probe": "answer",
    "update_probe": "update",
    "forget_probe": "forget",
    "governance_probe": "governance",
    "tool_probe": "tool",
    "planning_probe": "planning",
    "write_probe": "write",
    "retrieval_probe": "retrieval",
    "compression_probe": "compression",
    "evolution_probe": "evolution",
    "governed_transfer_probe": "planning",
    "scope_contrast_probe": "planning",
    "conflict_resolution_probe": "planning",
    "cross_session_synthesis_probe": "planning",
    "adversarial_state_synthesis_probe": "planning",
    "temporal_causal_reconciliation_probe": "planning",
    "policy_temporal_state_probe": "planning",
    "policy_exception_probe": "planning",
    "state_transition_audit_probe": "planning",
    "no_memory_probe": "no_memory",
}


PROBE_BEHAVIOR_TYPES = {
    "answer_probe": "answer",
    "update_probe": "answer",
    "tool_probe": "tool_call",
    "planning_probe": "plan",
    "write_probe": "memory_write",
    "retrieval_probe": "memory_retrieval",
    "compression_probe": "memory_compression",
    "evolution_probe": "policy_reuse",
    "governed_transfer_probe": "plan",
    "scope_contrast_probe": "plan",
    "conflict_resolution_probe": "plan",
    "cross_session_synthesis_probe": "plan",
    "adversarial_state_synthesis_probe": "plan",
    "temporal_causal_reconciliation_probe": "plan",
    "policy_temporal_state_probe": "plan",
    "policy_exception_probe": "plan",
    "state_transition_audit_probe": "plan",
    "no_memory_probe": "no_memory",
}


def expected_behavior_type(probe_type: str, *, should_refuse: bool) -> str | None:
    if probe_type == "forget_probe":
        return "refusal" if should_refuse else "answer"
    if probe_type == "governance_probe":
        return "refusal" if should_refuse else "answer"
    return PROBE_BEHAVIOR_TYPES.get(probe_type)


def expected_requires_memory(probe_type: str, *, should_refuse: bool) -> bool | None:
    if probe_type == "no_memory_probe":
        return False
    if probe_type in {"forget_probe", "governance_probe"}:
        return not should_refuse
    if probe_type in PROBE_SCORING_RULES:
        return True
    return None
