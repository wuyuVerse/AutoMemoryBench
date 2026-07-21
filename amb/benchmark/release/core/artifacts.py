"""Shared release artifact helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from amb.benchmark.quality.annotation import AUDIT_CHECK_FIELDS
from amb.benchmark.schemas.models import Case


def artifact_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_audit_template(path: Path, cases: tuple[Case, ...]) -> None:
    state_contracts = _state_contract_index(cases)
    counterfactual_groups = _counterfactual_group_index(cases)
    evidence_by_case = _evidence_index(cases)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for case in cases:
            for query in case.queries:
                group = counterfactual_groups.get(query.counterfactual_group_id) if query.counterfactual_group_id else None
                record = {
                    "case_id": case.case_id,
                    "query_id": query.query_id,
                    "domain": case.domain,
                    "probe_type": query.probe_type,
                    "counterfactual_group_id": query.counterfactual_group_id,
                    "counterfactual_axis": _counterfactual_axis(group),
                    "counterfactual_edit": str(case.difficulty.values.get("counterfactual_edit", "base")),
                    "prompt": query.prompt,
                    "task_type": query.task_type,
                    "scoring_rule": query.scoring_rule,
                    "memory_dependency": query.memory_dependency,
                    "memory_requirement": "requires_memory" if query.requires_memory else "no_memory_required",
                    "difficulty_level": str(query.difficulty.get("level", "")),
                    "state_contract_id": query.state_contract_id,
                    "gold_memory_ids": list(query.gold_memory_ids),
                    "forbidden_memory_ids": list(query.forbidden_memory_ids),
                    "applicable_checks": _applicable_checks(query),
                    "checks": {field: None for field in AUDIT_CHECK_FIELDS},
                    "state_contract_summary": _state_contract_summary(state_contracts.get((case.case_id, query.state_contract_id))),
                    "expected_behavior": asdict(query.expected_behavior),
                    "gold_memory_evidence": _memory_evidence(case, query.gold_memory_ids),
                    "forbidden_memory_evidence": _memory_evidence(case, query.forbidden_memory_ids),
                    "relevant_events": _event_evidence(case, query, evidence_by_case.get(case.case_id, {})),
                    "counterfactual_context": _counterfactual_context(case.case_id, query.query_id, group, state_contracts),
                    "audit_reference": {
                        "requires_memory": query.requires_memory,
                        "counterfactual_target_state_only": bool(group and len(group) >= 2) if query.counterfactual_group_id else None,
                    },
                    "annotator_id": None,
                    "notes": None,
                }
                fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                fh.write("\n")


def _state_contract_index(cases: tuple[Case, ...]) -> dict[tuple[str, str | None], dict[str, Any]]:
    index: dict[tuple[str, str | None], dict[str, Any]] = {}
    for case in cases:
        for contract in case.state_contracts:
            index[(case.case_id, contract.state_contract_id)] = {
                "timestamp": contract.timestamp,
                "active_memory_ids": tuple(contract.active_memory_ids),
                "deleted_memory_ids": tuple(contract.deleted_memory_ids),
                "forbidden_memory_ids": tuple(contract.forbidden_memory_ids),
                "restricted_memory_ids": tuple(contract.restricted_memory_ids),
                "required_governance_rules": tuple(contract.required_governance_rules),
                "transition_types": tuple(sorted({transition.transition_type for transition in contract.transitions})),
            }
    return index


def _counterfactual_group_index(cases: tuple[Case, ...]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        case_edit = str(case.difficulty.values.get("counterfactual_edit", "base"))
        for query in case.queries:
            group_id = query.counterfactual_group_id
            if not group_id:
                continue
            groups.setdefault(group_id, []).append(
                {
                    "case_id": case.case_id,
                    "query_id": query.query_id,
                    "group_id": group_id,
                    "prompt": query.prompt,
                    "probe_type": query.probe_type,
                    "task_type": query.task_type,
                    "state_contract_id": query.state_contract_id,
                    "gold_memory_ids": tuple(query.gold_memory_ids),
                    "forbidden_memory_ids": tuple(query.forbidden_memory_ids),
                    "counterfactual_edit": case_edit,
                }
            )
    return groups


def _evidence_index(cases: tuple[Case, ...]) -> dict[str, dict[str, Any]]:
    return {
        case.case_id: {event.event_id: event for event in case.events}
        for case in cases
    }


def _memory_evidence(case: Case, memory_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units}
    evidence = []
    for memory_id in memory_ids:
        memory = memory_by_id.get(memory_id)
        if memory is None:
            continue
        evidence.append(
            {
                "memory_id": memory.memory_id,
                "content": memory.content,
                "status": memory.status,
                "privacy_level": memory.privacy_level,
                "authorization_scope": memory.authorization_scope,
                "source_event_ids": list(memory.source_event_ids),
                "expected_use": memory.expected_use,
            }
        )
    return evidence


def _event_evidence(case: Case, query: Any, event_by_id: dict[str, Any]) -> list[dict[str, Any]]:
    memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units}
    event_ids: set[str] = set()
    for memory_id in (*query.gold_memory_ids, *query.forbidden_memory_ids):
        memory = memory_by_id.get(memory_id)
        if memory is not None:
            event_ids.update(memory.source_event_ids)
    events = []
    for event_id in sorted(event_ids):
        event = event_by_id.get(event_id)
        if event is None:
            continue
        events.append(
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "timestamp": event.timestamp,
                "subject": event.subject,
                "predicate": event.predicate,
                "object": event.object,
            }
        )
    return events


def _applicable_checks(query: Any) -> list[str]:
    checks = [field for field in AUDIT_CHECK_FIELDS if field != "counterfactual_target_state_only"]
    if query.counterfactual_group_id:
        checks.append("counterfactual_target_state_only")
    return checks


def _state_contract_summary(state_contract: dict[str, Any] | None) -> dict[str, Any] | None:
    if state_contract is None:
        return None
    return {
        "timestamp": state_contract.get("timestamp"),
        "active_memory_count": len(state_contract.get("active_memory_ids", ())),
        "deleted_memory_count": len(state_contract.get("deleted_memory_ids", ())),
        "forbidden_memory_count": len(state_contract.get("forbidden_memory_ids", ())),
        "restricted_memory_count": len(state_contract.get("restricted_memory_ids", ())),
        "required_governance_rules": list(state_contract.get("required_governance_rules", ())),
        "transition_types": list(state_contract.get("transition_types", ())),
    }


def _counterfactual_context(
    case_id: str,
    query_id: str,
    group: list[dict[str, Any]] | None,
    state_contracts: dict[tuple[str, str | None], dict[str, Any]],
) -> dict[str, Any] | None:
    if not group or len(group) < 2:
        return None
    current = next((item for item in group if item["case_id"] == case_id and item["query_id"] == query_id), None)
    if current is None:
        return None
    peers = [item for item in group if item is not current]
    pair_state_deltas = [
        _state_delta_summary(
            state_contracts.get((current["case_id"], current["state_contract_id"])),
            state_contracts.get((peer["case_id"], peer["state_contract_id"])),
        )
        for peer in peers
    ]
    return {
        "group_id": current["group_id"],
        "num_group_items": len(group),
        "axis": _counterfactual_axis(group),
        "shared_prompt_across_group": len({item["prompt"] for item in group}) == 1,
        "shared_probe_type_across_group": len({item["probe_type"] for item in group}) == 1,
        "shared_task_type_across_group": len({item["task_type"] for item in group}) == 1,
        "paired_items": [
            {
                "case_id": peer["case_id"],
                "query_id": peer["query_id"],
                "state_contract_id": peer["state_contract_id"],
                "counterfactual_edit": peer["counterfactual_edit"],
            }
            for peer in peers
        ],
        "gold_memory_delta_pairs": [
            {
                "paired_query_id": peer["query_id"],
                "gold_added": sorted(set(peer["gold_memory_ids"]) - set(current["gold_memory_ids"])),
                "gold_removed": sorted(set(current["gold_memory_ids"]) - set(peer["gold_memory_ids"])),
                "forbidden_added": sorted(set(peer["forbidden_memory_ids"]) - set(current["forbidden_memory_ids"])),
                "forbidden_removed": sorted(set(current["forbidden_memory_ids"]) - set(peer["forbidden_memory_ids"])),
            }
            for peer in peers
        ],
        "state_delta_pairs": pair_state_deltas,
    }


def _counterfactual_axis(group: list[dict[str, Any]]) -> str | None:
    if not group:
        return None
    sample = str(group[0].get("group_id", ""))
    parts = sample.split(":")
    if len(parts) < 3:
        return None
    return parts[1]


def _state_delta_summary(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any] | None:
    if left is None or right is None:
        return None
    return {
        "active_added": sorted(set(right.get("active_memory_ids", ())) - set(left.get("active_memory_ids", ()))),
        "active_removed": sorted(set(left.get("active_memory_ids", ())) - set(right.get("active_memory_ids", ()))),
        "deleted_added": sorted(set(right.get("deleted_memory_ids", ())) - set(left.get("deleted_memory_ids", ()))),
        "deleted_removed": sorted(set(left.get("deleted_memory_ids", ())) - set(right.get("deleted_memory_ids", ()))),
        "forbidden_added": sorted(set(right.get("forbidden_memory_ids", ())) - set(left.get("forbidden_memory_ids", ()))),
        "forbidden_removed": sorted(set(left.get("forbidden_memory_ids", ())) - set(right.get("forbidden_memory_ids", ()))),
        "restricted_added": sorted(set(right.get("restricted_memory_ids", ())) - set(left.get("restricted_memory_ids", ()))),
        "restricted_removed": sorted(set(left.get("restricted_memory_ids", ())) - set(right.get("restricted_memory_ids", ()))),
        "transition_type_pairs": sorted(set(left.get("transition_types", ())) | set(right.get("transition_types", ()))),
    }
