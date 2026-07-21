"""Contamination and near-duplicate checks for benchmark text artifacts."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from amb.benchmark.schemas.models import Benchmark


TextRecord = dict[str, Any]


def extract_benchmark_text_records(benchmark: Benchmark) -> list[TextRecord]:
    """Extract JSON-serializable text records from a Benchmark.

    The record shape is intentionally simple so quality, audit, release, or CLI
    layers can reuse it without importing benchmark internals.
    """

    records: list[TextRecord] = []
    for case in benchmark.cases:
        counterfactual_group_id = case.difficulty.values.get("counterfactual_group_id")
        for session in case.sessions:
            for turn in session.turns:
                records.append(
                    _record(
                        record_id=f"{case.case_id}/session/{session.session_id}/turn/{turn.turn_id}",
                        source_type="turn",
                        text=turn.content,
                        case_id=case.case_id,
                        domain=case.domain,
                        counterfactual_group_id=counterfactual_group_id,
                        session_id=session.session_id,
                        turn_id=turn.turn_id,
                        role=turn.role,
                    )
                )

        for memory in case.gold_memory_units:
            records.append(
                _record(
                    record_id=f"{case.case_id}/memory/{memory.memory_id}",
                    source_type="memory",
                    text=memory.content,
                    case_id=case.case_id,
                    domain=case.domain,
                    counterfactual_group_id=counterfactual_group_id,
                    memory_id=memory.memory_id,
                    memory_type=memory.type,
                )
            )

        for event in case.events:
            event_text = " ".join((event.subject, event.predicate, event.object))
            records.append(
                _record(
                    record_id=f"{case.case_id}/event/{event.event_id}",
                    source_type="event",
                    text=event_text,
                    case_id=case.case_id,
                    domain=case.domain,
                    counterfactual_group_id=counterfactual_group_id,
                    event_id=event.event_id,
                    event_type=event.event_type,
                )
            )

        for query in case.queries:
            records.append(
                _record(
                    record_id=f"{case.case_id}/query/{query.query_id}/prompt",
                    source_type="query_prompt",
                    text=query.prompt,
                    case_id=case.case_id,
                    domain=case.domain,
                    counterfactual_group_id=counterfactual_group_id,
                    query_id=query.query_id,
                    task_type=query.task_type,
                )
            )
            for index, fragment in enumerate(query.expected_behavior.must_include):
                records.append(
                    _record(
                        record_id=f"{case.case_id}/query/{query.query_id}/must_include/{index}",
                        source_type="expected_must_include",
                        text=fragment,
                        case_id=case.case_id,
                        domain=case.domain,
                        counterfactual_group_id=counterfactual_group_id,
                        query_id=query.query_id,
                    )
                )
            for index, fragment in enumerate(query.expected_behavior.must_not_include):
                records.append(
                    _record(
                        record_id=f"{case.case_id}/query/{query.query_id}/must_not_include/{index}",
                        source_type="expected_must_not_include",
                        text=fragment,
                        case_id=case.case_id,
                        domain=case.domain,
                        counterfactual_group_id=counterfactual_group_id,
                        query_id=query.query_id,
                    )
                )

    return [record for record in records if record["text"]]


def normalize_reference_texts(reference_texts: Iterable[str]) -> list[TextRecord]:
    """Convert external reference strings into comparison records."""

    return [
        _record(record_id=f"reference_text/{index}", source_type="reference_text", text=text)
        for index, text in enumerate(reference_texts)
        if str(text).strip()
    ]


def normalize_reference_records(reference_records: Iterable[Mapping[str, Any]]) -> list[TextRecord]:
    """Normalize external reference records.

    Each input record must contain a ``text`` field. ``record_id`` or ``id`` is
    preserved when present; other JSON-serializable metadata is copied through.
    """

    records: list[TextRecord] = []
    for index, reference in enumerate(reference_records):
        text = str(reference.get("text", "")).strip()
        if not text:
            continue
        record_id = str(reference.get("record_id") or reference.get("id") or f"reference_record/{index}")
        normalized = dict(reference)
        normalized.update({"record_id": record_id, "source_type": str(reference.get("source_type", "reference_record")), "text": text})
        records.append(normalized)
    return records


def contamination_report(
    benchmark_or_records: Benchmark | Sequence[Mapping[str, Any]],
    *,
    reference_texts: Iterable[str] = (),
    reference_records: Iterable[Mapping[str, Any]] = (),
    shingle_size: int = 5,
    jaccard_threshold: float = 0.82,
    min_tokens: int = 5,
    include_internal: bool = False,
) -> dict[str, Any]:
    """Check exact duplicates and shingle-Jaccard near duplicates.

    ``benchmark_or_records`` may be a Benchmark or pre-extracted records. The
    returned report contains only JSON-serializable values.
    """

    if shingle_size < 1:
        raise ValueError("shingle_size must be >= 1")
    if not 0.0 <= jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must be between 0 and 1")
    if min_tokens < 1:
        raise ValueError("min_tokens must be >= 1")

    target_records = (
        extract_benchmark_text_records(benchmark_or_records)  # type: ignore[arg-type]
        if _looks_like_benchmark(benchmark_or_records)
        else normalize_reference_records(benchmark_or_records)  # type: ignore[arg-type]
    )
    references = [*normalize_reference_texts(reference_texts), *normalize_reference_records(reference_records)]

    issues: list[dict[str, Any]] = []
    pairs = _comparison_pairs(target_records, references, include_internal=include_internal)
    for target, reference, scope in pairs:
        target_tokens = _tokens(target["text"])
        reference_tokens = _tokens(reference["text"])
        if len(target_tokens) < min_tokens or len(reference_tokens) < min_tokens:
            continue

        if _normalized_text(target["text"]) == _normalized_text(reference["text"]):
            issues.append(_issue("exact_duplicate", target, reference, 1.0, scope))
            continue

        score = shingle_jaccard(target_tokens, reference_tokens, shingle_size)
        if score >= jaccard_threshold:
            issues.append(_issue("near_duplicate", target, reference, score, scope))

    issues.sort(key=lambda item: (-item["similarity"], item["target_record_id"], item["reference_record_id"]))
    passed = not issues
    return {
        "passed": passed,
        "gate": {
            "name": "contamination",
            "passed": passed,
            "checks": {
                "exact_duplicate": not any(issue["code"] == "exact_duplicate" for issue in issues),
                "near_duplicate": not any(issue["code"] == "near_duplicate" for issue in issues),
            },
        },
        "issues": issues,
        "summary": {
            "num_target_records": len(target_records),
            "num_reference_records": len(references),
            "num_issues": len(issues),
            "num_exact_duplicates": sum(1 for issue in issues if issue["code"] == "exact_duplicate"),
            "num_near_duplicates": sum(1 for issue in issues if issue["code"] == "near_duplicate"),
            "jaccard_threshold": jaccard_threshold,
            "shingle_size": shingle_size,
            "min_tokens": min_tokens,
            "include_internal": include_internal,
        },
    }


def shingle_jaccard(left_tokens: Sequence[str], right_tokens: Sequence[str], shingle_size: int = 5) -> float:
    """Return Jaccard similarity over token shingles."""

    left = _shingles(left_tokens, shingle_size)
    right = _shingles(right_tokens, shingle_size)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _looks_like_benchmark(value: Any) -> bool:
    """Accept both canonical and legacy-imported Benchmark instances.

    The legacy ``agent_memory_benchmark`` package is a compatibility import path
    over ``amb.benchmark``. Python can load dataclasses from the same source file
    under both module names, so class identity is not a safe compatibility test.
    """

    return isinstance(value, Benchmark) or (
        hasattr(value, "cases")
        and hasattr(value, "benchmark_id")
        and not isinstance(value, (str, bytes, Mapping))
    )


def _record(record_id: str, source_type: str, text: Any, **metadata: Any) -> TextRecord:
    return {"record_id": record_id, "source_type": source_type, "text": str(text).strip(), **metadata}


def _comparison_pairs(
    targets: Sequence[TextRecord],
    references: Sequence[TextRecord],
    *,
    include_internal: bool,
) -> list[tuple[TextRecord, TextRecord, str]]:
    pairs: list[tuple[TextRecord, TextRecord, str]] = []
    for target in targets:
        for reference in references:
            pairs.append((target, reference, "external"))
    if include_internal:
        for index, target in enumerate(targets):
            for reference in targets[index + 1 :]:
                if _same_internal_scope(target, reference):
                    continue
                pairs.append((target, reference, "internal"))
    return pairs


def _same_internal_scope(left: TextRecord, right: TextRecord) -> bool:
    if left.get("case_id") and left.get("case_id") == right.get("case_id"):
        return True
    if left.get("counterfactual_group_id") and left.get("counterfactual_group_id") == right.get("counterfactual_group_id"):
        return True
    return False


def _issue(code: str, target: TextRecord, reference: TextRecord, similarity: float, scope: str) -> dict[str, Any]:
    return {
        "gate": "contamination",
        "code": code,
        "scope": scope,
        "target_record_id": target["record_id"],
        "target_source_type": target.get("source_type"),
        "reference_record_id": reference["record_id"],
        "reference_source_type": reference.get("source_type"),
        "similarity": round(float(similarity), 6),
        "message": f"{code} between {target['record_id']} and {reference['record_id']}.",
    }


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalized_text(text: str) -> str:
    return " ".join(_tokens(text))


def _shingles(tokens: Sequence[str], shingle_size: int) -> set[tuple[str, ...]]:
    if len(tokens) < shingle_size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + shingle_size]) for index in range(len(tokens) - shingle_size + 1)}
