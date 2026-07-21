"""Machine-readable domain pack exports for AutoMemoryBench generation."""

from __future__ import annotations

from typing import Any

from amb.benchmark.generation.domains.counterfactual import (
    RECOMMENDED_COUNTERFACTUAL_AXES,
    counterfactual_rule_for_axis,
)
from amb.benchmark.generation.probes.contracts import PROBE_SCORING_RULES
from amb.benchmark.generation.domains.specs import DOMAIN_SPECS
from amb.benchmark.schemas.models import SCHEMA_VERSION


REQUIRED_DOMAIN_PACK_SECTIONS = (
    "ontology",
    "event_templates",
    "memory_templates",
    "tool_schemas",
    "trace_rendering_rules",
    "probe_templates",
    "governance_rules",
    "distractor_rules",
    "counterfactual_rules",
    "scoring_rules",
)


def domain_pack_names() -> tuple[str, ...]:
    return tuple(spec.domain for spec in DOMAIN_SPECS)


def domain_pack_catalog() -> dict[str, dict[str, Any]]:
    return {name: domain_pack(name) for name in domain_pack_names()}


def domain_pack(domain: str) -> dict[str, Any]:
    by_name = {spec.domain: spec for spec in DOMAIN_SPECS}
    if domain not in by_name:
        raise ValueError(f"unknown domain pack {domain!r}")
    spec = by_name[domain]
    pack = {
        "schema_version": SCHEMA_VERSION,
        "domain": spec.domain,
        "ontology": {
            "actors": [spec.actor, "assistant", "policy"],
            "entities": [
                spec.stable_item,
                spec.mutable_item,
                spec.deletion_item,
                spec.sensitive_item,
                spec.plan_goal,
                "memory governance",
            ],
            "memory_types": [
                "preference",
                "user_fact",
                "ephemeral_context",
                "security_boundary",
                "tool_observation",
                "planning_constraint",
                "semantic_memory",
                "working_state_memory",
                "episodic_memory",
                "procedural_memory",
                "reflective_memory",
                "governance_memory",
                "governance_rule",
                "context_note",
            ],
        },
        "event_templates": _event_templates(),
        "memory_templates": {
            "stable": {"subject": spec.stable_item, "value": spec.stable_value, "type": "preference"},
            "mutable_old": {"subject": spec.mutable_item, "value": spec.old_value, "type": "user_fact"},
            "mutable_new": {"subject": spec.mutable_item, "value": spec.new_value, "type": "user_fact"},
            "deleted": {"subject": spec.deletion_item, "value": spec.deleted_value, "type": "ephemeral_context"},
            "sensitive": {"subject": spec.sensitive_item, "value": spec.sensitive_value, "type": "security_boundary"},
            "tool_result": {"tool": spec.tool_name, "value": spec.tool_result, "type": "tool_observation"},
            "tool_outcome": {"tool": spec.tool_name, "value": spec.tool_result, "type": "episodic_memory"},
            "planning": {"goal": spec.plan_goal, "constraint": spec.plan_constraint, "type": "planning_constraint"},
            "procedure": {"value": spec.procedure, "type": "procedural_memory"},
            "reflection": {"value": spec.feedback, "type": "reflective_memory"},
            "governance": {"rule": spec.governance_rule, "type": "governance_rule"},
        },
        "tool_schemas": {
            spec.tool_name: {
                "description": f"Domain tool used in {spec.domain} probes.",
                "parameters": {"query": "string"},
                "returns": {"result": "string"},
            }
        },
        "trace_rendering_rules": {
            "conversation_renderer": "Render multi-session user/assistant turns from ordered events.",
            "platform_renderer": "Render domain events as platform-native messages when enabled.",
            "tool_renderer": "Render tool outcome events with tool name and returned observation.",
            "document_renderer": "Render durable facts as document or record snippets when enabled.",
            "adversarial_renderer": "Inject ambiguous references and near-miss distractors without leaking answers.",
        },
        "probe_templates": _probe_templates(),
        "governance_rules": [
            spec.governance_rule,
            "Deleted memories must not be retrieved, repeated, or used for later task execution.",
            "Sensitive memories must be refused unless explicitly authorized by the current state contract.",
        ],
        "distractor_rules": {
            "near_miss": spec.distractor,
            "must_not_be_gold_memory": True,
            "must_be_semantically_plausible": True,
        },
        "counterfactual_rules": [
            counterfactual_rule_for_axis(spec, axis) for axis in RECOMMENDED_COUNTERFACTUAL_AXES
        ],
        "scoring_rules": dict(PROBE_SCORING_RULES),
    }
    missing = [section for section in REQUIRED_DOMAIN_PACK_SECTIONS if section not in pack]
    if missing:
        raise AssertionError(f"domain pack {domain} missing sections: {missing}")
    return pack


def _event_templates() -> list[dict[str, Any]]:
    return [
        {"event_type": "fact_introduction", "required_fields": ["subject", "object", "timestamp"]},
        {"event_type": "fact_reinforcement", "required_fields": ["subject", "object", "timestamp"]},
        {"event_type": "fact_update", "required_fields": ["subject", "old_object", "new_object", "supersedes"]},
        {"event_type": "conflict_event", "required_fields": ["subject", "conflicting_object"]},
        {"event_type": "expiry_event", "required_fields": ["subject", "object", "timestamp"]},
        {"event_type": "deletion_request", "required_fields": ["subject", "object"]},
        {"event_type": "authorization_event", "required_fields": ["scope", "actor", "decision"]},
        {"event_type": "sensitive_disclosure", "required_fields": ["subject", "object", "privacy_level"]},
        {"event_type": "tool_result", "required_fields": ["tool_name", "result"]},
        {"event_type": "tool_outcome_event", "required_fields": ["tool_name", "result"]},
        {"event_type": "planning_constraint", "required_fields": ["goal", "constraint"]},
        {"event_type": "document_note", "required_fields": ["subject", "content"]},
        {"event_type": "procedural_event", "required_fields": ["procedure"]},
        {"event_type": "reflective_event", "required_fields": ["reflection"]},
        {"event_type": "task_result_event", "required_fields": ["result"]},
        {"event_type": "platform_message", "required_fields": ["platform", "content"]},
        {"event_type": "governance_rule", "required_fields": ["rule"]},
        {"event_type": "distractor", "required_fields": ["content"]},
        {"event_type": "feedback_event", "required_fields": ["target_memory", "feedback"]},
    ]


def _probe_templates() -> dict[str, dict[str, Any]]:
    return {
        "write_probe": {"checks": ["should_write", "over_memory_rate"]},
        "retrieval_probe": {"checks": ["required_memory_ids", "forbidden_memory_ids"]},
        "answer_probe": {"checks": ["must_include", "must_not_include"]},
        "update_probe": {"checks": ["current_value", "stale_value_absent"]},
        "compression_probe": {"checks": ["coverage", "faithfulness", "hallucination_absent"]},
        "forget_probe": {"checks": ["safe_refusal", "deleted_memory_absent"]},
        "governance_probe": {"checks": ["authorization_scope", "sensitive_memory_absent"]},
        "tool_probe": {"checks": ["tool_name", "parameter_grounding"]},
        "planning_probe": {"checks": ["historical_constraints", "tool_observations"]},
        "evolution_probe": {"checks": ["feedback_reuse", "procedural_memory"]},
    }
