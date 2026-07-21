"""Deterministic data-quality gates for benchmark artifacts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import re
from typing import Any, Iterable, Mapping

from amb.benchmark.quality.contamination import contamination_report
from amb.benchmark.schemas.models import Benchmark, Case, ExpectedBehavior, MemoryUnit, Query
from amb.benchmark.schemas.state import event_graph_state_contract_differences


QUALITY_GATES = (
    "answer_query_leakage",
    "prompt_naturalness",
    "evidence_sufficiency",
    "evidence_necessity",
    "event_state_contract_closure",
    "answer_uniqueness",
    "distractor_validity",
    "governance_closure",
    "counterfactual_consistency",
    "query_construction",
    "oracle_solvability",
    "no_memory_unsolvability",
    "contamination_check",
)


def quality_checks(
    benchmark: Benchmark,
    *,
    reference_texts: Iterable[str] = (),
    reference_records: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Run deterministic quality gates over an AMST-style benchmark.

    The checks intentionally avoid model calls. They enforce that prompts do not
    leak answer material, required answers have explicit gold-memory support,
    memory-dependent answers are not answerable from the prompt alone, and
    counterfactual groups actually differ in expected behavior.
    """

    issues: list[dict[str, Any]] = []
    counterfactual_groups: dict[str, list[tuple[Case, Query]]] = {}
    num_queries = 0
    num_requires_memory = 0
    num_refusal_queries = 0
    all_queries: list[tuple[Case, Query]] = []

    for case in benchmark.cases:
        memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units}
        contract_blocks = _contract_blocked_memory_ids(case)
        issues.extend(_event_state_contract_closure_issues(case))
        issues.extend(_distractor_validity_issues(case))
        for query in case.queries:
            all_queries.append((case, query))
            num_queries += 1
            num_requires_memory += int(query.requires_memory)
            num_refusal_queries += int(query.expected_behavior.should_refuse)
            if query.counterfactual_group_id:
                counterfactual_groups.setdefault(query.counterfactual_group_id, []).append((case, query))

            issues.extend(_leakage_issues(case, query, memory_by_id))
            issues.extend(_prompt_naturalness_issues(case, query))
            issues.extend(_evidence_sufficiency_issues(case, query, memory_by_id, contract_blocks))
            issues.extend(_evidence_necessity_issues(case, query))
            issues.extend(_answer_uniqueness_issues(case, query))
            issues.extend(_governance_closure_issues(case, query, contract_blocks))

    issues.extend(_counterfactual_issues(counterfactual_groups))
    issues.extend(_query_construction_issues(all_queries))
    issues.extend(_baseline_sanity_issues(benchmark, bool(counterfactual_groups)))
    contamination = contamination_report(
        benchmark,
        reference_texts=reference_texts,
        reference_records=reference_records,
        include_internal=False,
    )
    issues.extend(_contamination_issues(contamination))

    issue_counts = {gate: 0 for gate in QUALITY_GATES}
    for issue in issues:
        issue_counts[issue["gate"]] += 1

    gates = {gate: issue_counts[gate] == 0 for gate in QUALITY_GATES}
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "issues": issues,
        "summary": {
            "num_cases": len(benchmark.cases),
            "num_queries": num_queries,
            "num_requires_memory_queries": num_requires_memory,
            "num_refusal_queries": num_refusal_queries,
            "num_counterfactual_groups": len(counterfactual_groups),
            "num_issues": len(issues),
            "issue_counts_by_gate": issue_counts,
            "contamination": contamination["summary"],
        },
    }


def _leakage_issues(case: Case, query: Query, memory_by_id: dict[str, MemoryUnit]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    checked_memory_ids = tuple(dict.fromkeys((*query.gold_memory_ids, *query.forbidden_memory_ids)))

    for memory_id in checked_memory_ids:
        if _contains_phrase(query.prompt, memory_id):
            issues.append(
                _issue(
                    "answer_query_leakage",
                    "prompt_contains_memory_id",
                    case,
                    query,
                    f"Prompt contains memory_id {memory_id}.",
                    memory_id=memory_id,
                )
            )
        memory = memory_by_id.get(memory_id)
        if memory and _contains_phrase(query.prompt, memory.content):
            issues.append(
                _issue(
                    "answer_query_leakage",
                    "prompt_contains_memory_content",
                    case,
                    query,
                    f"Prompt contains content from {memory_id}.",
                    memory_id=memory_id,
                )
            )

    if query.requires_memory and not _is_write_probe(query):
        for fragment in query.expected_behavior.must_include:
            if _contains_phrase(query.prompt, fragment):
                issues.append(
                    _issue(
                        "answer_query_leakage",
                        "prompt_contains_must_include",
                        case,
                        query,
                        "Prompt contains an expected answer fragment.",
                        fragment=fragment,
                    )
                )

    for fragment in query.expected_behavior.must_not_include:
        if _contains_phrase(query.prompt, fragment) and not _contains_forbidden_phrase_as_negated_decoy(
            query.prompt,
            fragment,
        ):
            issues.append(
                _issue(
                    "answer_query_leakage",
                    "prompt_contains_forbidden_fragment",
                    case,
                    query,
                    "Prompt contains a forbidden answer fragment.",
                    fragment=fragment,
                )
            )

    return issues


def _evidence_sufficiency_issues(
    case: Case,
    query: Query,
    memory_by_id: dict[str, MemoryUnit],
    contract_blocks: set[str],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    expected = query.expected_behavior

    if query.requires_memory and not expected.should_refuse:
        if not expected.must_include:
            issues.append(
                _issue(
                    "evidence_sufficiency",
                    "missing_must_include",
                    case,
                    query,
                    "Memory-dependent non-refusal query has no must_include evidence target.",
                )
            )
            return issues

        gold_text = "\n".join(
            memory_by_id[memory_id].content for memory_id in query.gold_memory_ids if memory_id in memory_by_id
        )
        expected_meta = _expected_metadata_text(expected)
        for fragment in expected.must_include:
            if not (
                _fragment_supported_by_text(gold_text, fragment)
                or _contains_phrase(expected_meta, fragment)
            ):
                issues.append(
                    _issue(
                        "evidence_sufficiency",
                        "unsupported_must_include",
                        case,
                        query,
                        "must_include fragment is not supported by gold memory content or expected metadata.",
                        fragment=fragment,
                    )
                )

    if expected.should_refuse:
        supporting_memories = tuple(
            memory
            for memory_id in query.forbidden_memory_ids
            if (memory := memory_by_id.get(memory_id)) and _is_blocking_memory(memory, contract_blocks)
        )
        if not supporting_memories:
            issues.append(
                _issue(
                    "evidence_sufficiency",
                    "unsupported_refusal_or_governance",
                    case,
                    query,
                    "Refusal/governance query is not supported by forbidden, deleted, or sensitive memory.",
                )
            )
        blocked_text = "\n".join(memory.content for memory in supporting_memories)
        for fragment in expected.must_not_include:
            if not _contains_phrase(blocked_text, fragment):
                issues.append(
                    _issue(
                        "evidence_sufficiency",
                        "unsupported_forbidden_fragment",
                        case,
                        query,
                        "must_not_include fragment is not supported by blocked memory content.",
                        fragment=fragment,
                    )
                )

    return issues


def _evidence_necessity_issues(case: Case, query: Query) -> list[dict[str, Any]]:
    if not query.requires_memory or _is_write_probe(query):
        return []

    issues: list[dict[str, Any]] = []
    for fragment in query.expected_behavior.must_include:
        if _contains_phrase(query.prompt, fragment):
            issues.append(
                _issue(
                    "evidence_necessity",
                    "prompt_answers_memory_dependent_query",
                    case,
                    query,
                    "A memory-dependent must_include fragment appears directly in the prompt.",
                    fragment=fragment,
                )
            )
    return issues


_SYNTHETIC_PROMPT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("scenario_marker", re.compile(r"\(\s*scenario\s+\d+\s*\)", re.IGNORECASE)),
    ("scope_alpha_beta", re.compile(r"\bscope\s+(alpha|beta)\b", re.IGNORECASE)),
    ("active_scope", re.compile(r"\bactive\s+scope\b", re.IGNORECASE)),
    ("scoped_tool_suffix", re.compile(r"\b[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*_s\d+\b", re.IGNORECASE)),
    ("case_or_query_id", re.compile(r"\b(case_[a-z0-9_]+|q_[a-z0-9_]+)\b", re.IGNORECASE)),
)


def _prompt_naturalness_issues(case: Case, query: Query) -> list[dict[str, Any]]:
    prompt = str(query.prompt or "")
    issues: list[dict[str, Any]] = []
    for code, pattern in _SYNTHETIC_PROMPT_PATTERNS:
        match = pattern.search(prompt)
        if match is None:
            continue
        issues.append(
            _issue(
                "prompt_naturalness",
                f"synthetic_{code}_in_prompt",
                case,
                query,
                "Prompt exposes a synthetic benchmark marker instead of natural task language.",
                matched_text=match.group(0),
            )
        )
    return issues


def _answer_uniqueness_issues(case: Case, query: Query) -> list[dict[str, Any]]:
    expected = query.expected_behavior
    if expected.should_refuse:
        return []
    if expected.must_include:
        return []
    if expected.tool_name or expected.parameters:
        return []
    if query.memory_dependency == "none" and query.task_type == "no_memory":
        return []
    return [
        _issue(
            "answer_uniqueness",
            "missing_deterministic_expected_behavior",
            case,
            query,
            "Non-refusal probe has no must_include, tool target, or parameter target.",
        )
    ]


def _distractor_validity_issues(case: Case) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    distractor_event_ids = {event.event_id for event in case.events if event.event_type == "distractor"}
    if not distractor_event_ids:
        return [
            {
                "gate": "distractor_validity",
                "code": "missing_distractor_event",
                "case_id": case.case_id,
                "query_id": None,
                "detail": "Case has no distractor event in the event graph.",
            }
        ]

    for memory in case.gold_memory_units:
        overlap = sorted(distractor_event_ids & set(memory.source_event_ids))
        if overlap:
            issues.append(
                {
                    "gate": "distractor_validity",
                    "code": "distractor_became_gold_memory",
                    "case_id": case.case_id,
                    "query_id": None,
                    "detail": "Distractor event is referenced by a gold memory.",
                    "memory_id": memory.memory_id,
                    "source_event_ids": overlap,
                }
            )

    memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units}
    for query in case.queries:
        referenced = tuple(dict.fromkeys((*query.gold_memory_ids, *query.forbidden_memory_ids)))
        for memory_id in referenced:
            memory = memory_by_id.get(memory_id)
            if memory and distractor_event_ids & set(memory.source_event_ids):
                issues.append(
                    _issue(
                        "distractor_validity",
                        "query_references_distractor_memory",
                        case,
                        query,
                        "Query references a memory sourced from a distractor event.",
                        memory_id=memory_id,
                    )
                )
    return issues


def _event_state_contract_closure_issues(case: Case) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for contract in case.state_contracts:
        for difference in event_graph_state_contract_differences(case, contract):
            issues.append(
                {
                    "gate": "event_state_contract_closure",
                    "code": "event_graph_state_contract_mismatch",
                    "case_id": case.case_id,
                    "query_id": None,
                    "state_contract_id": contract.state_contract_id,
                    "detail": difference,
                }
            )
    return issues


def _governance_closure_issues(case: Case, query: Query, contract_blocks: set[str]) -> list[dict[str, Any]]:
    if not query.expected_behavior.should_refuse:
        return []
    issues: list[dict[str, Any]] = []
    if not query.forbidden_memory_ids:
        issues.append(
            _issue(
                "governance_closure",
                "missing_forbidden_memory_reference",
                case,
                query,
                "Refusal/governance query has no forbidden memory reference.",
            )
        )
    for memory_id in query.forbidden_memory_ids:
        if memory_id not in contract_blocks:
            issues.append(
                _issue(
                    "governance_closure",
                    "forbidden_memory_not_blocked_by_contract",
                    case,
                    query,
                    "Forbidden memory is not deleted, forbidden, or restricted in any state contract.",
                    memory_id=memory_id,
                )
            )
    return issues


def _counterfactual_issues(groups: dict[str, list[tuple[Case, Query]]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for group_id, members in sorted(groups.items()):
        if len(members) < 2:
            case, query = members[0]
            issues.append(
                _issue(
                    "counterfactual_consistency",
                    "singleton_counterfactual_group",
                    case,
                    query,
                    "counterfactual_group_id appears on fewer than two queries.",
                    counterfactual_group_id=group_id,
                )
            )
            continue

        signatures = {_counterfactual_signature(case, query) for case, query in members}
        if len(signatures) == 1:
            case, query = members[0]
            issues.append(
                _issue(
                    "counterfactual_consistency",
                    "counterfactual_group_has_no_expected_difference",
                    case,
                    query,
                    "Counterfactual group has no expected must_include or forbidden/refusal behavior difference.",
                    counterfactual_group_id=group_id,
                    group_size=len(members),
                )
            )
            continue

        prompts = {_counterfactual_prompt_signature(query.prompt) for _, query in members}
        task_types = {query.task_type for _, query in members}
        probe_types = {query.probe_type for _, query in members}
        scoring_rules = {query.scoring_rule for _, query in members}
        if len(task_types) != 1 or len(probe_types) != 1 or len(scoring_rules) != 1:
            case, query = members[0]
            issues.append(
                _issue(
                    "counterfactual_consistency",
                    "counterfactual_group_not_comparable",
                    case,
                    query,
                    "Counterfactual group changes task type, probe type, or scoring rule across states.",
                    counterfactual_group_id=group_id,
                    prompt_variants=len(prompts),
                    task_type_variants=len(task_types),
                    probe_type_variants=len(probe_types),
                    scoring_rule_variants=len(scoring_rules),
                )
            )
            continue

        shared_slots, changing_slots = _counterfactual_target_slots(members)
        if not shared_slots:
            case, query = members[0]
            issues.append(
                _issue(
                    "counterfactual_consistency",
                    "counterfactual_group_missing_shared_target_slot",
                    case,
                    query,
                    "Counterfactual group changes expected behavior, but referenced memories do not align to any shared canonical target slot.",
                    counterfactual_group_id=group_id,
                    group_size=len(members),
                )
            )
            continue

        if not changing_slots:
            case, query = members[0]
            issues.append(
                _issue(
                    "counterfactual_consistency",
                    "counterfactual_group_target_slot_state_static",
                    case,
                    query,
                    "Counterfactual group shares a canonical target slot, but the referenced target-slot state does not change across members.",
                    counterfactual_group_id=group_id,
                    group_size=len(members),
                    shared_target_slots=sorted(shared_slots),
                )
            )
    return issues


def _query_construction_issues(pairs: list[tuple[Case, Query]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    skeleton_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for case, query in pairs:
        memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units}
        contract = next(
            (item for item in case.state_contracts if item.state_contract_id == query.state_contract_id),
            None,
        )
        active_ids = set(contract.active_memory_ids) if contract is not None else set(memory_by_id)
        gold_ids = tuple(memory_id for memory_id in query.gold_memory_ids if memory_id in memory_by_id)
        non_gold_active_ids = sorted(active_ids - set(gold_ids))

        issues.extend(_gold_minimality_issues(case, query, memory_by_id, gold_ids, non_gold_active_ids))
        issues.extend(_adversarial_competitor_issues(case, query, memory_by_id, gold_ids))
        issues.extend(_hard_query_reasoning_issues(case, query, memory_by_id, gold_ids))

        skeleton = _prompt_skeleton(query.prompt, case, query, memory_by_id)
        probe_type = query.probe_type or query.task_type
        skeleton_counts[probe_type][skeleton] += 1

    for probe_type, counts in sorted(skeleton_counts.items()):
        if probe_type == "no_memory_probe":
            continue
        total = sum(counts.values())
        dominant_skeleton, dominant_count = max(counts.items(), key=lambda item: item[1])
        if total >= 64 and dominant_count / total > 0.95:
            issues.append(
                {
                    "gate": "query_construction",
                    "code": "low_prompt_skeleton_diversity",
                    "case_id": None,
                    "query_id": None,
                    "detail": "A probe family is overly dominated by one prompt skeleton.",
                    "probe_type": probe_type,
                    "dominant_skeleton_share": dominant_count / total,
                    "dominant_skeleton_count": dominant_count,
                    "num_queries": total,
                    "dominant_skeleton": dominant_skeleton,
                }
            )

    return issues


def _gold_minimality_issues(
    case: Case,
    query: Query,
    memory_by_id: dict[str, MemoryUnit],
    gold_ids: tuple[str, ...],
    non_gold_active_ids: list[str],
) -> list[dict[str, Any]]:
    if not query.requires_memory or query.expected_behavior.should_refuse:
        return []

    issues: list[dict[str, Any]] = []
    fragments = tuple(
        _normalize_text(fragment)
        for fragment in _gold_support_targets(query)
        if _normalize_text(fragment)
    )
    if not fragments:
        return issues

    gold_fragment_support = {
        memory_id: {
            fragment
            for fragment in fragments
            if _fragment_supported_by_text(memory_by_id[memory_id].content, fragment)
        }
        for memory_id in gold_ids
    }
    supported_fragments = sorted({fragment for values in gold_fragment_support.values() for fragment in values})
    missing_support = [fragment for fragment in fragments if fragment not in supported_fragments]
    if missing_support:
        issues.append(
            _issue(
                "query_construction",
                "gold_memory_missing_required_fragment",
                case,
                query,
                "Declared gold memory set does not cover all required answer fragments.",
                missing_fragments=missing_support,
            )
        )

    if not supported_fragments:
        issues.append(
            _issue(
                "query_construction",
                "gold_memory_has_no_direct_support",
                case,
                query,
                "None of the declared gold memories directly supports the expected answer fragments.",
                gold_memory_ids=list(gold_ids),
            )
        )
        return issues

    redundant_gold = sorted(memory_id for memory_id, values in gold_fragment_support.items() if not values)
    if redundant_gold:
        issues.append(
            _issue(
                "query_construction",
                "overdeclared_gold_memory",
                case,
                query,
                "Some declared gold memories do not directly support the expected answer fragments.",
                redundant_gold_memory_ids=redundant_gold,
            )
        )

    return issues


def _gold_support_targets(query: Query) -> tuple[str, ...]:
    targets = list(query.expected_behavior.must_include)
    if query.expected_behavior.parameters:
        targets.extend(
            str(value)
            for key, value in query.expected_behavior.parameters.items()
            if key != "sensitive_output_policy"
        )
    return tuple(targets)


def _adversarial_competitor_issues(
    case: Case,
    query: Query,
    memory_by_id: dict[str, MemoryUnit],
    gold_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    needs_competitor = bool(query.forbidden_memory_ids)
    if not needs_competitor:
        return []
    if not gold_ids and not query.expected_behavior.should_refuse:
        return []

    gold_text = " ".join(memory_by_id[memory_id].content for memory_id in gold_ids if memory_id in memory_by_id)
    competitors = []
    for memory_id in query.forbidden_memory_ids:
        memory = memory_by_id.get(memory_id)
        if memory is None:
            continue
        if _token_overlap_ratio(gold_text, memory.content) >= 0.2 or any(
            _contains_phrase(memory.content, fragment)
            for fragment in query.expected_behavior.must_not_include
        ):
            competitors.append(memory_id)
    if competitors:
        if gold_ids:
            grounded = _state_grounded_competitor_ids(memory_by_id, gold_ids, tuple(competitors))
            if not grounded:
                return [
                    _issue(
                        "query_construction",
                        "missing_state_grounded_competitor",
                        case,
                        query,
                        "Competing memory exists lexically, but is not grounded in the same state slot or transition lineage.",
                        forbidden_memory_ids=list(query.forbidden_memory_ids),
                    )
                ]
        return []
    return [
        _issue(
            "query_construction",
            "missing_adversarial_competitor",
            case,
            query,
            "Forbidden/deprecated evidence exists, but no sufficiently similar competing memory was found.",
            forbidden_memory_ids=list(query.forbidden_memory_ids),
        )
    ]


def _hard_query_reasoning_issues(
    case: Case,
    query: Query,
    memory_by_id: dict[str, MemoryUnit],
    gold_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not query.requires_memory:
        return []
    difficulty_level = str((query.difficulty or {}).get("level", "")).lower()
    if difficulty_level != "hard":
        return []

    fragments = tuple(
        dict.fromkeys(
            _normalize_text(fragment)
            for fragment in _gold_support_targets(query)
            if _normalize_text(fragment)
        )
    )
    if not fragments:
        return []

    gold_fragment_support = {
        memory_id: {
            fragment
            for fragment in fragments
            if memory_id in memory_by_id and _contains_phrase(memory_by_id[memory_id].content, fragment)
        }
        for memory_id in gold_ids
    }
    covered_fragments = {
        fragment
        for values in gold_fragment_support.values()
        for fragment in values
    }
    if covered_fragments != set(fragments):
        return []

    single_solver_ids = sorted(
        memory_id
        for memory_id, values in gold_fragment_support.items()
        if values == set(fragments)
    )
    contributing_gold_ids = sorted(memory_id for memory_id, values in gold_fragment_support.items() if values)
    grounded_competitors = _state_grounded_competitor_ids(memory_by_id, gold_ids, query.forbidden_memory_ids)
    compositional = (
        len(fragments) >= 2
        and len(contributing_gold_ids) >= 2
        and not single_solver_ids
    )

    issues: list[dict[str, Any]] = []
    if (query.probe_type or query.task_type) == "update_probe" and not grounded_competitors:
        issues.append(
            _issue(
                "query_construction",
                "hard_query_missing_compositional_support",
                case,
                query,
                "Hard update query must retain a same-slot stale competitor even when it is compositionally supported.",
                gold_memory_ids=list(gold_ids),
                forbidden_memory_ids=list(query.forbidden_memory_ids),
                target_fragments=list(fragments),
            )
        )
    if not compositional and not grounded_competitors:
        issues.append(
            _issue(
                "query_construction",
                "hard_query_missing_compositional_support",
                case,
                query,
                "Hard query is neither compositionally supported by multiple gold memories nor anchored by a same-slot competitor.",
                gold_memory_ids=list(gold_ids),
                forbidden_memory_ids=list(query.forbidden_memory_ids),
                target_fragments=list(fragments),
            )
        )
    if single_solver_ids and not grounded_competitors:
        issues.append(
            _issue(
                "query_construction",
                "hard_query_literal_sloting",
                case,
                query,
                "Hard query can be solved by copying one gold memory without resolving a competing state.",
                single_solver_memory_ids=single_solver_ids,
                target_fragments=list(fragments),
            )
        )
    return issues


def _state_grounded_competitor_ids(
    memory_by_id: dict[str, MemoryUnit],
    gold_ids: tuple[str, ...],
    forbidden_ids: tuple[str, ...],
) -> list[str]:
    grounded: list[str] = []
    gold_memories = [memory_by_id[memory_id] for memory_id in gold_ids if memory_id in memory_by_id]
    for forbidden_id in forbidden_ids:
        forbidden = memory_by_id.get(forbidden_id)
        if forbidden is None:
            continue
        if any(_same_state_slot_or_transition(gold, forbidden) for gold in gold_memories):
            grounded.append(forbidden_id)
    return grounded


def _same_state_slot_or_transition(gold: MemoryUnit, competitor: MemoryUnit) -> bool:
    gold_subject = str(gold.canonical_form.get("subject", ""))
    competitor_subject = str(competitor.canonical_form.get("subject", ""))
    if not gold_subject or gold_subject != competitor_subject:
        return False

    gold_predicate = str(gold.canonical_form.get("predicate", ""))
    competitor_predicate = str(competitor.canonical_form.get("predicate", ""))
    same_predicate = bool(gold_predicate) and gold_predicate == competitor_predicate
    transition_linked = (
        competitor.memory_id in set(gold.invalidates)
        or gold.memory_id in set(competitor.invalidates)
        or competitor.update_of == gold.memory_id
        or gold.update_of == competitor.memory_id
    )
    different_object = str(gold.canonical_form.get("object", "")) != str(competitor.canonical_form.get("object", ""))
    governed_difference = (
        gold.authorization_scope != competitor.authorization_scope
        or gold.privacy_level != competitor.privacy_level
        or gold.status != competitor.status
    )
    competitor_blocked = competitor.status.lower() in {"superseded", "deleted", "forbidden", "restricted"} or competitor.should_delete
    return transition_linked or (same_predicate and different_object and (governed_difference or competitor_blocked))


def _baseline_sanity_issues(benchmark: Benchmark, has_counterfactual_groups: bool) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not benchmark.cases:
        return issues

    from amb.benchmark.evaluation.baselines import make_baseline
    from amb.benchmark.evaluation.scoring import Scorer

    retrieval_k = max(
        5,
        max((len(query.gold_memory_ids) for case in benchmark.cases for query in case.queries), default=0),
    )
    scorer = Scorer(retrieval_k=retrieval_k)
    oracle_report = scorer.score(benchmark, make_baseline(benchmark, "oracle_memory"))
    no_memory_report = scorer.score(benchmark, make_baseline(benchmark, "no_memory"))
    oracle_aggregate = oracle_report["aggregate"]
    no_memory_aggregate = no_memory_report["aggregate"]

    for metric in ("task.task_success", "retrieval.recall_at_k", "safety.safety_pass"):
        if float(oracle_aggregate.get(metric, 0.0)) < 0.99:
            issues.append(
                {
                    "gate": "oracle_solvability",
                    "code": "oracle_metric_below_threshold",
                    "case_id": None,
                    "query_id": None,
                    "detail": "Oracle baseline should solve the generated benchmark.",
                    "metric": metric,
                    "value": oracle_aggregate.get(metric),
                }
            )

    if has_counterfactual_groups:
        md_proxy = oracle_report["counterfactual"].get("memory_dependence_proxy")
        if md_proxy is None or float(md_proxy) < 0.99:
            issues.append(
                {
                    "gate": "oracle_solvability",
                    "code": "oracle_counterfactual_dependence_below_threshold",
                    "case_id": None,
                    "query_id": None,
                    "detail": "Oracle baseline should flip correctly across counterfactual memory states.",
                    "value": md_proxy,
                }
            )

    requires_memory = no_memory_report.get("by_memory_requirement", {}).get("requires_memory")
    if requires_memory and float(requires_memory.get("retrieval.recall_at_k", 0.0)) > 0.0:
        issues.append(
            {
                "gate": "no_memory_unsolvability",
                "code": "no_memory_retrieves_required_memory",
                "case_id": None,
                "query_id": None,
                "detail": "No-memory baseline should have zero retrieval recall on memory-required queries.",
                "value": requires_memory.get("retrieval.recall_at_k"),
            }
        )

    if float(no_memory_aggregate.get("task.task_success", 0.0)) >= float(oracle_aggregate.get("task.task_success", 0.0)):
        issues.append(
            {
                "gate": "no_memory_unsolvability",
                "code": "no_memory_not_below_oracle",
                "case_id": None,
                "query_id": None,
                "detail": "No-memory task success should remain below oracle task success.",
                "no_memory_task_success": no_memory_aggregate.get("task.task_success"),
                "oracle_task_success": oracle_aggregate.get("task.task_success"),
            }
        )

    return issues


def _contamination_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for issue in report["issues"]:
        converted = dict(issue)
        converted["gate"] = "contamination_check"
        converted["detail"] = converted.pop("message", "Potential benchmark contamination or near duplicate.")
        converted.setdefault("case_id", None)
        converted.setdefault("query_id", None)
        issues.append(converted)
    return issues


def _counterfactual_signature(case: Case, query: Query) -> tuple[Any, ...]:
    memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units}
    blocked_categories = tuple(
        sorted(_blocking_category(memory_by_id[memory_id]) for memory_id in query.forbidden_memory_ids if memory_id in memory_by_id)
    )
    expected = query.expected_behavior
    return (
        tuple(_normalize_text(fragment) for fragment in expected.must_include),
        tuple(_normalize_text(fragment) for fragment in expected.must_not_include),
        expected.should_refuse,
        expected.behavior_type,
        blocked_categories,
    )


def _counterfactual_target_slots(
    members: list[tuple[Case, Query]],
) -> tuple[set[str], set[str]]:
    slot_maps = [_referenced_slot_state_map(case, query) for case, query in members]
    present_slot_sets = [set(slot_map) for slot_map in slot_maps if slot_map]
    if not present_slot_sets:
        return set(), set()

    target_slots = set.union(*present_slot_sets)
    changing_slots: set[str] = set()
    for slot in target_slots:
        slot_states = {
            tuple(sorted(slot_map.get(slot, [])))
            for slot_map in slot_maps
        }
        if len(slot_states) > 1:
            changing_slots.add(slot)
    return target_slots, changing_slots


def _referenced_slot_state_map(case: Case, query: Query) -> dict[tuple[str, str], list[tuple[Any, ...]]]:
    memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units}
    slot_map: dict[str, list[tuple[Any, ...]]] = defaultdict(list)

    for role, memory_ids in (("gold", query.gold_memory_ids), ("forbidden", query.forbidden_memory_ids)):
        for memory_id in memory_ids:
            memory = memory_by_id.get(memory_id)
            if memory is None:
                continue
            slot = _counterfactual_slot_signature(memory)
            if slot is None:
                continue
            slot_map[slot].append((role, *_counterfactual_slot_state_signature(memory)))
    return dict(slot_map)


def _counterfactual_slot_signature(memory: MemoryUnit) -> str | None:
    subject = _normalize_text(str(memory.canonical_form.get("subject", "")))
    if not subject:
        return None
    return subject


def _counterfactual_slot_state_signature(memory: MemoryUnit) -> tuple[Any, ...]:
    return (
        _normalize_text(str(memory.canonical_form.get("object", ""))),
        _normalize_text(memory.status),
        _normalize_text(memory.authorization_scope),
        _normalize_text(memory.privacy_level),
        bool(memory.should_store),
        bool(memory.should_delete),
    )
    expected = query.expected_behavior
    return (
        tuple(_normalize_text(fragment) for fragment in expected.must_include),
        tuple(_normalize_text(fragment) for fragment in expected.must_not_include),
        expected.should_refuse,
        expected.behavior_type,
        blocked_categories,
    )


def _contract_blocked_memory_ids(case: Case) -> set[str]:
    blocked: set[str] = set()
    for contract in case.state_contracts:
        blocked.update(contract.deleted_memory_ids)
        blocked.update(contract.forbidden_memory_ids)
        blocked.update(contract.restricted_memory_ids)
    return blocked


def _is_blocking_memory(memory: MemoryUnit, contract_blocks: set[str]) -> bool:
    return (
        memory.memory_id in contract_blocks
        or memory.status.lower() in {"deleted", "forbidden", "restricted"}
        or memory.should_delete
        or memory.is_sensitive
        or not memory.should_store
    )


def _blocking_category(memory: MemoryUnit) -> str:
    if memory.should_delete or memory.status.lower() == "deleted":
        return "deleted"
    if memory.is_sensitive:
        return "sensitive"
    if memory.status.lower() in {"forbidden", "restricted"} or not memory.should_store:
        return "forbidden"
    return memory.status.lower()


def _expected_metadata_text(expected: ExpectedBehavior) -> str:
    metadata = asdict(expected)
    metadata.pop("must_include", None)
    return repr(metadata)


def _issue(gate: str, code: str, case: Case, query: Query, detail: str, **extra: Any) -> dict[str, Any]:
    return {
        "gate": gate,
        "code": code,
        "case_id": case.case_id,
        "query_id": query.query_id,
        "detail": detail,
        **extra,
    }


def _contains_phrase(haystack: str, needle: str) -> bool:
    normalized_haystack = _normalize_text(haystack)
    normalized_needle = _normalize_text(needle)
    if not normalized_needle:
        return False
    return re.search(rf"(?<!\w){re.escape(normalized_needle)}(?!\w)", normalized_haystack) is not None


def _fragment_supported_by_text(haystack: str, needle: str) -> bool:
    return _contains_phrase(haystack, needle) or _authorization_boundary_semantic_match(haystack, needle)


def _authorization_boundary_semantic_match(haystack: str, needle: str) -> bool:
    normalized_haystack = _normalize_text(haystack)
    normalized_needle = _normalize_text(needle)
    marker = " may be used only "
    if marker not in normalized_needle:
        return False
    if "explicitly authoriz" not in normalized_needle and "current scope" not in normalized_needle:
        return False
    subject = normalized_needle.split(marker, 1)[0]
    if not subject or subject not in normalized_haystack:
        return False
    return "may be used only after" in normalized_haystack and "explicitly approves" in normalized_haystack


def _is_write_probe(query: Query) -> bool:
    return query.task_type == "write" or query.probe_type == "write_probe"


_NEGATED_DECOY_CUES = (
    "do not include",
    "do not use",
    "don't include",
    "don't use",
    "exclude",
    "omit",
    "without",
    "reject",
)


def _contains_forbidden_phrase_as_negated_decoy(prompt: str, fragment: str) -> bool:
    """Allow prompts to name stale/forbidden decoys only when explicitly negated."""

    normalized_prompt = _normalize_text(prompt)
    normalized_fragment = _normalize_text(fragment)
    if not normalized_fragment:
        return False
    for match in re.finditer(rf"(?<!\w){re.escape(normalized_fragment)}(?!\w)", normalized_prompt):
        prefix = normalized_prompt[max(0, match.start() - 96) : match.start()]
        if any(cue in prefix for cue in _NEGATED_DECOY_CUES):
            return True
    return False


_COUNTERFACTUAL_PROMPT_EQUIVALENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bdo not use superseded details\.?"),
    re.compile(r"\buse only still-valid team context\.?"),
    re.compile(r"\buse a practical next-step format\.?"),
    re.compile(r"\buse a direct execution-note format\.?"),
    re.compile(r"\brespect current permissions and scope\.?"),
    re.compile(r"\bseparate current facts from older context\.?"),
    re.compile(r"\bmake it a compact operational note\.?"),
    re.compile(r"\bkeep the response concise and current\.?"),
    re.compile(r"\bdo not rely on retired assumptions\.?"),
    re.compile(r"\buse a concise handoff style\.?"),
    re.compile(r"\bignore similar but inactive prior notes\.?"),
    re.compile(r"\bkeep obsolete channels out\.?"),
    re.compile(r"\bwrite it as a scoped task answer\.?"),
    re.compile(r"\bapply the latest valid instruction\.?"),
    re.compile(r"\bkeep the wording natural and specific\.?"),
    re.compile(r"\buse current evidence before older context\.?"),
    re.compile(r"\bmake the response audit-friendly\.?"),
    re.compile(r"\bdo not carry over outdated alternatives\.?"),
    re.compile(r"\buse the verified current state\.?"),
    re.compile(r"\banswer as a live-context reminder\.?"),
    re.compile(r"\buse the current workspace facts only\.?"),
    re.compile(r"\bretain only active evidence\.?"),
    re.compile(r"\bframe it as a current-state check\.?"),
    re.compile(r"\bprefer the latest valid information\.?"),
    re.compile(r"\bkeep it as a short decision aid\.?"),
    re.compile(r"\bavoid near-match memories from other contexts\.?"),
    re.compile(r"\bground the response in the active task state\.?"),
    re.compile(r"\bkeep the result aligned with current constraints\.?"),
    re.compile(r"\bignore removed or restricted options\.?"),
    re.compile(r"\banswer from the surviving state\.?"),
    re.compile(r"\bwrite it as a short working note\.?"),
    re.compile(r"\buse the active memory evidence\.?"),
    re.compile(r"\bkeep stale or restricted notes out\.?"),
    re.compile(r"\buse live state, not older drafts\.?"),
    re.compile(r"\buse only the current valid context\.?"),
    re.compile(r"\breturn it as a brief action note\.?"),
    re.compile(r"\bkeep the answer tied to the live request\.?"),
    re.compile(r"\bdrop replaced or unauthorized details\.?"),
    re.compile(r"\bkeep the answer scoped to this task\.?"),
    re.compile(r"\banswer as a current-context summary\.?"),
    re.compile(r"\bexclude stale cross-session echoes\.?"),
    re.compile(r"\bkeep it suitable for a teammate\.?"),
)
_STATE_RECONCILIATION_PROMPT_PREFIX = (
    "use the remembered project facts for the next step, but reject any recalled instruction that conflicts with "
    "deletion, authorization, current value, or project-boundary rules."
)


def _counterfactual_prompt_signature(prompt: str) -> str:
    """Normalize prompt wording variants that preserve the same counterfactual task."""

    normalized = _normalize_text(prompt)
    if normalized.startswith(_STATE_RECONCILIATION_PROMPT_PREFIX):
        return _STATE_RECONCILIATION_PROMPT_PREFIX
    for pattern in _COUNTERFACTUAL_PROMPT_EQUIVALENCE_PATTERNS:
        normalized = pattern.sub(" ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _prompt_skeleton(prompt: str, case: Case, query: Query, memory_by_id: dict[str, MemoryUnit]) -> str:
    normalized = _normalize_text(prompt)
    replacements = {
        _normalize_text(case.case_id),
        *(_normalize_text(fragment) for fragment in query.expected_behavior.must_include),
        *(_normalize_text(fragment) for fragment in query.expected_behavior.must_not_include),
    }
    for memory_id in (*query.gold_memory_ids, *query.forbidden_memory_ids):
        memory = memory_by_id.get(memory_id)
        if memory is not None:
            replacements.add(_normalize_text(memory.content))
    for token in sorted(replacements, key=len, reverse=True):
        if token:
            normalized = normalized.replace(token, "<slot>")
    normalized = re.sub(r"scenario\s+\d+", "<scenario>", normalized)
    normalized = re.sub(r"\b[a-z_]+\.search(_s\d+)?\b", "<tool>", normalized)
    normalized = re.sub(r"\b[a-z_]+\.lookup(_s\d+)?\b", "<tool>", normalized)
    normalized = re.sub(r"\b[a-z_]+\.query(_s\d+)?\b", "<tool>", normalized)
    normalized = re.sub(r"\b[a-z_]+\.log(_s\d+)?\b", "<tool>", normalized)
    return normalized


def _token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9_]+", _normalize_text(left)))
    right_tokens = set(re.findall(r"[a-z0-9_]+", _normalize_text(right)))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return overlap / min(len(left_tokens), len(right_tokens))


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()
