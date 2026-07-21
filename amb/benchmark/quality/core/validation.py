"""Validation utilities for benchmark and prediction artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from amb.benchmark.schemas.models import Benchmark, PredictionSet
from amb.benchmark.schemas.state import state_contract_differences


@dataclass(frozen=True)
class ValidationResult:
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_benchmark(benchmark: Benchmark) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    case_ids: set[str] = set()
    query_ids: set[str] = set()

    if not benchmark.cases:
        warnings.append("benchmark has no cases")

    for case in benchmark.cases:
        _check_unique(case.case_id, "case_id", case_ids, errors)
        if case.scenario_id and case.scenario_id != case.case_id:
            errors.append(f"{case.case_id} scenario_id {case.scenario_id} does not match case_id")
        if case.scenario is not None:
            if case.scenario.scenario_id != case.case_id:
                errors.append(f"{case.case_id} scenario.scenario_id {case.scenario.scenario_id} does not match case_id")
            if case.scenario.domain != case.domain:
                errors.append(f"{case.case_id} scenario.domain {case.scenario.domain} does not match case domain {case.domain}")
            if case.scenario.time_span is None:
                warnings.append(f"{case.case_id} scenario metadata has no time_span")
        turn_ids = {turn.turn_id for session in case.sessions for turn in session.turns}
        event_ids: set[str] = set()
        memory_ids: set[str] = set()
        state_contract_ids: set[str] = set()

        for session in case.sessions:
            session_turns: set[str] = set()
            for turn in session.turns:
                _check_unique(turn.turn_id, f"turn_id in case {case.case_id}", session_turns, errors)

        for event in case.events:
            _check_unique(event.event_id, f"event_id in case {case.case_id}", event_ids, errors)
            if not event.source_turn_ids:
                warnings.append(f"{event.event_id} has no source_turn_ids")
            for source_id in event.source_turn_ids:
                if source_id not in turn_ids:
                    errors.append(f"{event.event_id} references missing source turn {source_id}")

        for event in case.events:
            supersedes = event.attributes.get("supersedes")
            if supersedes and supersedes not in event_ids:
                errors.append(f"{event.event_id} supersedes missing event {supersedes}")

        for edge in case.event_edges:
            if edge.source_event_id not in event_ids:
                errors.append(f"edge references missing source event {edge.source_event_id}")
            if edge.target_event_id not in event_ids:
                errors.append(f"edge references missing target event {edge.target_event_id}")

        for memory in case.gold_memory_units:
            _check_unique(memory.memory_id, f"memory_id in case {case.case_id}", memory_ids, errors)
            if memory.scenario_id and memory.scenario_id != case.case_id:
                errors.append(f"{memory.memory_id} scenario_id {memory.scenario_id} does not match case {case.case_id}")
            if memory.memory_type and memory.memory_type != memory.type:
                errors.append(f"{memory.memory_id} memory_type {memory.memory_type} does not match type {memory.type}")
            if not memory.canonical_form:
                warnings.append(f"{memory.memory_id} has no canonical_form")
            else:
                for field in ("subject", "predicate", "object"):
                    if field not in memory.canonical_form:
                        errors.append(f"{memory.memory_id} canonical_form missing {field}")
            if not memory.source_turn_ids:
                warnings.append(f"{memory.memory_id} has no source_turn_ids")
            for source_id in memory.source_turn_ids:
                if source_id not in turn_ids:
                    errors.append(f"{memory.memory_id} references missing source turn {source_id}")
            for source_event_id in memory.source_event_ids:
                if source_event_id not in event_ids:
                    errors.append(f"{memory.memory_id} references missing source event {source_event_id}")
            if memory.confidence is not None and not 0.0 <= memory.confidence <= 1.0:
                errors.append(f"{memory.memory_id} confidence must be in [0.0, 1.0]")
            if memory.importance is not None and not 1 <= memory.importance <= 5:
                errors.append(f"{memory.memory_id} importance must be in [1, 5]")
            if memory.update_of and memory.update_of == memory.memory_id:
                errors.append(f"{memory.memory_id} cannot update itself")
            for invalidated_id in memory.invalidates:
                if invalidated_id == memory.memory_id:
                    errors.append(f"{memory.memory_id} cannot invalidate itself")

        for memory in case.gold_memory_units:
            if memory.update_of and memory.update_of not in memory_ids:
                errors.append(f"{memory.memory_id} update_of references missing memory {memory.update_of}")
            for invalidated_id in memory.invalidates:
                if invalidated_id not in memory_ids:
                    errors.append(f"{memory.memory_id} invalidates missing memory {invalidated_id}")

        for contract in case.state_contracts:
            _check_unique(contract.state_contract_id, f"state_contract_id in case {case.case_id}", state_contract_ids, errors)
            if contract.scenario_id and contract.scenario_id != case.case_id:
                errors.append(
                    f"{contract.state_contract_id} scenario_id {contract.scenario_id} does not match case {case.case_id}"
                )
            for label, ids in {
                "active_memory_ids": contract.active_memory_ids,
                "inactive_memory_ids": contract.inactive_memory_ids,
                "deleted_memory_ids": contract.deleted_memory_ids,
                "forbidden_memory_ids": contract.forbidden_memory_ids,
                "superseded_memory_ids": contract.superseded_memory_ids,
                "restricted_memory_ids": contract.restricted_memory_ids,
            }.items():
                for memory_id in ids:
                    if memory_id not in memory_ids:
                        errors.append(f"{contract.state_contract_id} {label} references missing memory {memory_id}")
            active = set(contract.active_memory_ids)
            blocked = set(contract.deleted_memory_ids) | set(contract.forbidden_memory_ids) | set(contract.superseded_memory_ids)
            overlap = sorted(active & blocked)
            if overlap:
                errors.append(f"{contract.state_contract_id} has active memories also blocked: {overlap}")
            governance_rules = set(contract.required_governance_rules)
            if "same_user_only" not in governance_rules:
                errors.append(f"{contract.state_contract_id} required_governance_rules missing same_user_only")
            if contract.deleted_memory_ids and "do_not_recall_deleted" not in governance_rules:
                errors.append(f"{contract.state_contract_id} deleted memories require do_not_recall_deleted governance rule")
            if (contract.forbidden_memory_ids or contract.restricted_memory_ids) and "respect_authorization_scope" not in governance_rules:
                errors.append(
                    f"{contract.state_contract_id} forbidden/restricted memories require respect_authorization_scope governance rule"
                )
            for transition in contract.transitions:
                if transition.from_memory_id not in memory_ids:
                    errors.append(f"{contract.state_contract_id} transition references missing from memory {transition.from_memory_id}")
                if transition.to_memory_id not in memory_ids:
                    errors.append(f"{contract.state_contract_id} transition references missing to memory {transition.to_memory_id}")
                if transition.trigger_event_id not in event_ids:
                    errors.append(f"{contract.state_contract_id} transition references missing event {transition.trigger_event_id}")
            errors.extend(state_contract_differences(case, contract))

        for query in case.queries:
            _check_unique(query.query_id, "query_id", query_ids, errors)
            for memory_id in query.gold_memory_ids:
                if memory_id not in memory_ids:
                    errors.append(f"{query.query_id} references missing gold memory {memory_id}")
            for memory_id in query.forbidden_memory_ids:
                if memory_id not in memory_ids:
                    errors.append(f"{query.query_id} references missing forbidden memory {memory_id}")
            overlap = sorted(set(query.gold_memory_ids) & set(query.forbidden_memory_ids))
            if overlap:
                errors.append(f"{query.query_id} has memories both required and forbidden: {overlap}")
            if query.state_contract_id and query.state_contract_id not in state_contract_ids:
                errors.append(f"{query.query_id} references missing state contract {query.state_contract_id}")
            if query.requires_memory and not query.gold_memory_ids:
                warnings.append(f"{query.query_id} requires memory but has no gold_memory_ids")
            if not query.requires_memory and query.gold_memory_ids:
                warnings.append(f"{query.query_id} has gold memories but requires_memory is false")
            if not query.difficulty:
                warnings.append(f"{query.query_id} has no query difficulty metadata")
            else:
                level = query.difficulty.get("level")
                if level is not None and str(level) not in {"easy", "medium", "hard"}:
                    errors.append(f"{query.query_id} difficulty level must be easy/medium/hard")
                factors = query.difficulty.get("factors")
                if factors is not None and not isinstance(factors, (list, tuple)):
                    errors.append(f"{query.query_id} difficulty factors must be a list-like value")
                for field in ("score", "num_required_memories", "num_forbidden_memories"):
                    value = query.difficulty.get(field)
                    if value is not None:
                        try:
                            numeric = int(value)
                        except (TypeError, ValueError):
                            errors.append(f"{query.query_id} difficulty {field} must be an integer")
                            continue
                        if numeric < 0:
                            errors.append(f"{query.query_id} difficulty {field} must be non-negative")

        for key, value in case.difficulty.values.items():
            if isinstance(value, (int, float)) and value < 0:
                errors.append(f"{case.case_id} difficulty {key} must be non-negative")

    return ValidationResult(tuple(errors), tuple(warnings))


def validate_predictions(predictions: PredictionSet, benchmark: Benchmark) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    query_by_id = {
        query.query_id: query
        for case in benchmark.cases
        for query in case.queries
    }
    known_queries = set(query_by_id)
    seen: set[str] = set()

    for prediction in predictions.predictions:
        _check_unique(prediction.query_id, "prediction query_id", seen, errors)
        if prediction.query_id not in known_queries:
            warnings.append(f"prediction for unknown query {prediction.query_id}")
            continue
        query = query_by_id[prediction.query_id]
        if query.probe_type == "compression_probe" and not str(prediction.compression_summary or "").strip():
            warnings.append(f"{prediction.query_id} compression_probe prediction missing compression_summary")
        for op in prediction.memory_operations:
            if op.operation not in {"write", "update", "delete", "skip"}:
                errors.append(f"{prediction.query_id} has unknown operation {op.operation}")

    missing = sorted(known_queries - seen)
    for query_id in missing:
        warnings.append(f"missing prediction for {query_id}")

    return ValidationResult(tuple(errors), tuple(warnings))


def _check_unique(value: str, label: str, seen: set[str], errors: list[str]) -> None:
    if value in seen:
        errors.append(f"duplicate {label}: {value}")
    seen.add(value)
