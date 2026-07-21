"""Strong self-audit gates for AMST audit-task packages and release rows.

The report produced here is an AI/machine audit artifact. It is deliberately
not named or treated as completed human annotation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from amb.benchmark.quality.annotation import AUDIT_CHECK_FIELDS
from amb.benchmark.schemas.io import read_json, write_json


STRONG_AI_AUDIT_SCHEMA_VERSION = "amst-strong-ai-audit-v1"
CLAIM_BOUNDARY = (
    "This is a Codex/AI self-audit over the same items that require human audit. "
    "It closes machine usability gates but is not independent double-human annotation."
)

SECRET_PATTERNS = {
    "api_key_like": re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9_]{8,}|xoxb-[A-Za-z0-9-]{8,}|AKIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{20,})\b"
    ),
    "credit_card_like": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "password_literal": re.compile(r"\bpassword\b[^\n]{0,30}[:=][^\n]{4,}", re.IGNORECASE),
}

PROMPT_TEMPLATE_PATTERNS = {
    "case_id_leak": re.compile(r"\bcase_[a-z_]+_\d{4}(?:_cf\d{2})?\b"),
    "memory_id_leak": re.compile(r"\bm_[a-z]_[a-z_]+_v\d+\b"),
    "scenario_index": re.compile(r"\bscenario\s+\d+\b", re.IGNORECASE),
    "synthetic_next_the": re.compile(r"\bnext the\b", re.IGNORECASE),
}

NO_MEMORY_BEHAVIORS = {"no_memory"}
NON_ANSWER_BEHAVIORS = {"refusal", "no_memory"}
GOVERNANCE_RULES = {"same_user_only", "do_not_recall_deleted", "respect_authorization_scope"}
STATE_DECISION_TARGETS = {
    "not available for the current task",
    "not authorized for the current scope",
}


def write_strong_ai_audit(
    output_path: str | Path,
    *,
    task_manifest_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    """Write a strong AI audit report for a task manifest or release manifest."""

    report = build_strong_ai_audit(
        task_manifest_path=task_manifest_path,
        manifest_path=manifest_path,
        expected_rows=expected_rows,
    )
    write_json(output_path, report)
    return report


def write_strong_ai_audit_ledger(
    output_path: str | Path,
    *,
    task_manifest_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    """Write one JSONL audit record per row and return a compact summary."""

    summary = build_strong_ai_audit_ledger(
        task_manifest_path=task_manifest_path,
        manifest_path=manifest_path,
        expected_rows=expected_rows,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in summary.pop("_entries"):
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    summary["ledger_path"] = str(path)
    summary["ledger_sha256"] = _sha256(path)
    return summary


def build_strong_ai_audit_ledger(
    *,
    task_manifest_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    """Build row-level audit entries without collapsing them into a summary-only report."""

    if bool(task_manifest_path) == bool(manifest_path):
        raise ValueError("exactly one of task_manifest_path or manifest_path is required")

    source_type = "task_manifest" if task_manifest_path is not None else "release_manifest"
    source_path = Path(task_manifest_path or manifest_path or "")
    if task_manifest_path is not None:
        rows = _rows_from_task_manifest(Path(task_manifest_path))
    else:
        rows = _rows_from_release_manifest(Path(manifest_path or ""))

    duplicate_query_ids = _duplicate_query_ids(rows, source_type=source_type)
    entries: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    check_counts: dict[str, Counter[str]] = {field: Counter() for field in AUDIT_CHECK_FIELDS}
    digest = hashlib.sha256()

    for row_index, row in enumerate(rows, start=1):
        row_result = _audit_row_with_global_context(row, duplicate_query_ids=duplicate_query_ids)
        entry = _row_ledger_entry(
            row,
            row_result,
            row_index=row_index,
            source_type=source_type,
            source_path=source_path,
        )
        entries.append(entry)
        issue_counts.update(issue["code"] for issue in row_result["issues"])
        for check_id, passed in row_result["checks"].items():
            check_counts[check_id]["passed" if passed else "failed"] += 1
        digest.update(json.dumps(_row_audit_digest_payload(row, row_result), ensure_ascii=False, sort_keys=True).encode())
        digest.update(b"\n")

    global_issues: list[dict[str, Any]] = []
    if expected_rows is not None and len(rows) != expected_rows:
        global_issues.append(
            {
                "check_id": "expected_row_count",
                "code": "expected_row_count_mismatch",
                "detail": {"expected": expected_rows, "actual": len(rows)},
            }
        )
        issue_counts["expected_row_count_mismatch"] += 1

    failed_checks = sorted(check_id for check_id, counts in check_counts.items() if counts["failed"] > 0)
    status = "passed" if not issue_counts and not failed_checks else "failed"
    return {
        "schema_version": "amst-strong-ai-audit-ledger-v1",
        "source_type": source_type,
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "status": status,
        "claim_boundary": CLAIM_BOUNDARY,
        "num_rows": len(rows),
        "expected_rows": expected_rows,
        "num_issues": sum(issue_counts.values()),
        "num_failed_checks": len(failed_checks),
        "failed_check_ids": failed_checks,
        "issue_counts": dict(sorted(issue_counts.items())),
        "row_audit_digest": digest.hexdigest(),
        "global_issues": global_issues,
        "checks": {
            check_id: {
                "passed": counts["failed"] == 0,
                "num_passed_rows": counts["passed"],
                "num_failed_rows": counts["failed"],
            }
            for check_id, counts in check_counts.items()
        },
        "_entries": entries,
    }


def build_strong_ai_audit(
    *,
    task_manifest_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    """Run deterministic AI/self-audit checks over every row in the input."""

    if bool(task_manifest_path) == bool(manifest_path):
        raise ValueError("exactly one of task_manifest_path or manifest_path is required")

    source_type = "task_manifest" if task_manifest_path is not None else "release_manifest"
    source_path = Path(task_manifest_path or manifest_path or "")
    if task_manifest_path is not None:
        rows = _rows_from_task_manifest(Path(task_manifest_path))
    else:
        rows = _rows_from_release_manifest(Path(manifest_path or ""))

    issues: list[dict[str, Any]] = []
    check_counts: dict[str, Counter[str]] = {field: Counter() for field in AUDIT_CHECK_FIELDS}
    split_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    probe_counts: Counter[str] = Counter()
    behavior_counts: Counter[str] = Counter()
    row_digest = hashlib.sha256()
    seen_query_ids: set[str] = set()
    duplicate_query_ids = _duplicate_query_ids(rows, source_type=source_type)

    for row in rows:
        query_id = str(row.get("query_id") or "")
        if query_id:
            seen_query_ids.add(query_id)
        split_counts[str(row.get("release_split") or "audit_task")] += 1
        domain_counts[str(row.get("domain") or "unknown")] += 1
        probe_counts[str(row.get("probe_type") or "unknown")] += 1
        behavior_counts[str(_expected(row).get("behavior_type") or "unknown")] += 1

        row_result = _audit_row_with_global_context(row, duplicate_query_ids=duplicate_query_ids)
        issues.extend(row_result["issues"])
        for check_id, passed in row_result["checks"].items():
            check_counts[check_id]["passed" if passed else "failed"] += 1
        row_digest.update(json.dumps(_row_audit_digest_payload(row, row_result), ensure_ascii=False, sort_keys=True).encode("utf-8"))
        row_digest.update(b"\n")

    if expected_rows is not None and len(rows) != expected_rows:
        issues.append(
            {
                "query_id": "",
                "check_id": "expected_row_count",
                "code": "expected_row_count_mismatch",
                "detail": {"expected": expected_rows, "actual": len(rows)},
            }
        )
    issue_counts = Counter(issue["code"] for issue in issues)
    check_summary = {
        check_id: {
            "passed": counts["failed"] == 0,
            "num_passed_rows": counts["passed"],
            "num_failed_rows": counts["failed"],
        }
        for check_id, counts in check_counts.items()
    }
    failed_checks = sorted(check_id for check_id, item in check_summary.items() if not item["passed"])
    status = "passed" if not issues and not failed_checks else "failed"
    return {
        "schema_version": STRONG_AI_AUDIT_SCHEMA_VERSION,
        "source_type": source_type,
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "status": status,
        "claim_boundary": CLAIM_BOUNDARY,
        "summary": {
            "num_rows": len(rows),
            "expected_rows": expected_rows,
            "num_unique_query_ids": len(seen_query_ids),
            "num_duplicate_query_ids": len(duplicate_query_ids),
            "num_issues": len(issues),
            "num_failed_checks": len(failed_checks),
            "failed_check_ids": failed_checks,
            "issue_counts": dict(sorted(issue_counts.items())),
            "split_counts": dict(sorted(split_counts.items())),
            "domain_counts": dict(sorted(domain_counts.items())),
            "probe_type_counts": dict(sorted(probe_counts.items())),
            "behavior_type_counts": dict(sorted(behavior_counts.items())),
            "row_audit_digest": row_digest.hexdigest(),
        },
        "checks": check_summary,
        "sample_issues": issues[:1000],
    }


def _rows_from_task_manifest(task_manifest_path: Path) -> list[dict[str, Any]]:
    manifest = read_json(task_manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("task manifest must be a JSON object")
    template_files = manifest.get("template_files")
    if not isinstance(template_files, list) or not template_files:
        raise ValueError("task manifest template_files must be a non-empty list")

    rows: list[dict[str, Any]] = []
    for raw_path in template_files:
        template_path = _resolve_path(task_manifest_path.parent, str(raw_path))
        with template_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{template_path}:{line_no}: template row must be an object")
                row = dict(row)
                row["_source_template"] = str(template_path)
                row["_source_template_line"] = line_no
                rows.append(row)
    return rows


def _rows_from_release_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("release manifest must be a JSON object")
    root = manifest_path.parent
    rows: list[dict[str, Any]] = []
    for split, by_domain in sorted((manifest.get("split_files") or {}).items()):
        if not isinstance(by_domain, dict):
            continue
        for domain, raw_path in sorted(by_domain.items()):
            if not raw_path:
                continue
            shard_path = _resolve_path(root, str(raw_path))
            shard = read_json(shard_path)
            rows.extend(_rows_from_shard(shard, shard_path=shard_path, split=str(split), domain=str(domain)))
    return rows


def _rows_from_shard(shard: dict[str, Any], *, shard_path: Path, split: str, domain: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in shard.get("cases") or []:
        if not isinstance(case, dict):
            continue
        memory_by_id = {
            str(memory.get("memory_id")): memory
            for memory in case.get("gold_memory_units") or []
            if isinstance(memory, dict) and memory.get("memory_id")
        }
        event_by_id = {
            str(event.get("event_id")): event
            for event in case.get("events") or []
            if isinstance(event, dict) and event.get("event_id")
        }
        contract_by_id = {
            str(contract.get("state_contract_id")): contract
            for contract in case.get("state_contracts") or []
            if isinstance(contract, dict) and contract.get("state_contract_id")
        }
        for query_index, query in enumerate(case.get("queries") or [], start=1):
            if not isinstance(query, dict):
                continue
            gold_ids = [str(value) for value in query.get("gold_memory_ids") or []]
            forbidden_ids = [str(value) for value in query.get("forbidden_memory_ids") or []]
            evidence_ids = [*gold_ids, *forbidden_ids]
            event_ids = {
                str(event_id)
                for memory_id in evidence_ids
                for event_id in (memory_by_id.get(memory_id, {}).get("source_event_ids") or [])
                if event_id
            }
            contract = contract_by_id.get(str(query.get("state_contract_id") or ""), {})
            rows.append(
                {
                    "_source_shard": str(shard_path),
                    "_source_query_index": query_index,
                    "case_id": case.get("case_id"),
                    "domain": case.get("domain") or domain,
                    "release_split": split,
                    "query_id": query.get("query_id"),
                    "probe_type": query.get("probe_type"),
                    "task_type": query.get("task_type"),
                    "scoring_rule": query.get("scoring_rule"),
                    "difficulty_level": (query.get("difficulty") or {}).get("level"),
                    "memory_requirement": "requires_memory" if query.get("requires_memory") else "no_memory_required",
                    "memory_dependency": query.get("memory_dependency"),
                    "prompt": query.get("prompt"),
                    "expected_behavior": query.get("expected_behavior"),
                    "gold_memory_ids": gold_ids,
                    "gold_memory_evidence": [memory_by_id[memory_id] for memory_id in gold_ids if memory_id in memory_by_id],
                    "forbidden_memory_ids": forbidden_ids,
                    "forbidden_memory_evidence": [
                        memory_by_id[memory_id] for memory_id in forbidden_ids if memory_id in memory_by_id
                    ],
                    "relevant_events": [event_by_id[event_id] for event_id in sorted(event_ids) if event_id in event_by_id],
                    "state_contract_id": query.get("state_contract_id"),
                    "state_contract_summary": _state_contract_summary(contract),
                    "_state_contract": contract,
                    "counterfactual_group_id": query.get("counterfactual_group_id"),
                    "applicable_checks": list(AUDIT_CHECK_FIELDS),
                }
            )
    return rows


def _duplicate_query_ids(rows: list[dict[str, Any]], *, source_type: str) -> set[str]:
    if source_type != "release_manifest":
        return set()
    counts = Counter(str(row.get("query_id") or "") for row in rows if row.get("query_id"))
    return {query_id for query_id, count in counts.items() if count > 1}


def _audit_row_with_global_context(
    row: dict[str, Any],
    *,
    duplicate_query_ids: set[str],
) -> dict[str, Any]:
    row_result = _audit_row(row)
    query_id = str(row.get("query_id") or "")
    if not query_id or query_id not in duplicate_query_ids:
        return row_result
    checks = dict(row_result["checks"])
    checks["answer_unique"] = False
    issues = list(row_result["issues"])
    issues.append(
        {
            "query_id": query_id,
            "case_id": str(row.get("case_id") or ""),
            "domain": str(row.get("domain") or ""),
            "probe_type": str(row.get("probe_type") or ""),
            "check_id": "answer_unique",
            "code": "duplicate_query_id",
            "detail": "query_id must be unique within a release manifest",
            "source": row.get("_source_template") or row.get("_source_shard"),
            "line": row.get("_source_template_line") or row.get("_source_query_index"),
        }
    )
    return {"checks": checks, "issues": issues}


def _row_audit_digest_payload(row: dict[str, Any], row_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": row.get("release_split") or "",
        "case_id": row.get("case_id") or "",
        "query_id": str(row.get("query_id") or ""),
        "checks": row_result["checks"],
        "issue_codes": [issue["code"] for issue in row_result["issues"]],
    }


def _row_ledger_entry(
    row: dict[str, Any],
    row_result: dict[str, Any],
    *,
    row_index: int,
    source_type: str,
    source_path: Path,
) -> dict[str, Any]:
    digest_payload = _row_audit_digest_payload(row, row_result)
    row_hash = hashlib.sha256(
        json.dumps(digest_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    expected = _expected(row)
    issues = row_result["issues"]
    checks = row_result["checks"]
    return {
        "schema_version": "amst-strong-ai-audit-row-v1",
        "row_index": row_index,
        "source_type": source_type,
        "source": str(source_path),
        "source_row_path": row.get("_source_template") or row.get("_source_shard"),
        "source_row_index": row.get("_source_template_line") or row.get("_source_query_index"),
        "release_split": row.get("release_split") or "audit_task",
        "domain": row.get("domain") or "unknown",
        "case_id": row.get("case_id") or "",
        "query_id": row.get("query_id") or "",
        "probe_type": row.get("probe_type") or "unknown",
        "behavior_type": expected.get("behavior_type") or "unknown",
        "memory_requirement": row.get("memory_requirement") or "",
        "memory_dependency": row.get("memory_dependency") or "",
        "passed": all(checks.values()) and not issues,
        "num_issues": len(issues),
        "issue_codes": [issue["code"] for issue in issues],
        "checks": checks,
        "issues": issues,
        "evidence_counts": {
            "gold_memory_ids": len(_ids(row.get("gold_memory_ids"))),
            "forbidden_memory_ids": len(_ids(row.get("forbidden_memory_ids"))),
            "gold_memory_evidence": len(row.get("gold_memory_evidence") or []),
            "forbidden_memory_evidence": len(row.get("forbidden_memory_evidence") or []),
            "relevant_events": len(row.get("relevant_events") or []),
        },
        "row_audit_hash": row_hash,
    }


def _audit_row(row: dict[str, Any]) -> dict[str, Any]:
    checks = {field: True for field in AUDIT_CHECK_FIELDS}
    issues: list[dict[str, Any]] = []

    def fail(check_id: str, code: str, detail: Any) -> None:
        checks[check_id] = False
        issues.append(
            {
                "query_id": str(row.get("query_id") or ""),
                "case_id": str(row.get("case_id") or ""),
                "domain": str(row.get("domain") or ""),
                "probe_type": str(row.get("probe_type") or ""),
                "check_id": check_id,
                "code": code,
                "detail": detail,
                "source": row.get("_source_template") or row.get("_source_shard"),
                "line": row.get("_source_template_line") or row.get("_source_query_index"),
            }
        )

    expected = _expected(row)
    prompt = str(row.get("prompt") or "")
    gold_ids = _ids(row.get("gold_memory_ids"))
    forbidden_ids = _ids(row.get("forbidden_memory_ids"))
    gold_evidence = row.get("gold_memory_evidence") if isinstance(row.get("gold_memory_evidence"), list) else []
    forbidden_evidence = (
        row.get("forbidden_memory_evidence") if isinstance(row.get("forbidden_memory_evidence"), list) else []
    )
    events = row.get("relevant_events") if isinstance(row.get("relevant_events"), list) else []
    behavior_type = str(expected.get("behavior_type") or "")
    must_include = [str(value) for value in expected.get("must_include") or []]
    must_not = [str(value) for value in expected.get("must_not_include") or []]
    parameters = expected.get("parameters") if isinstance(expected.get("parameters"), dict) else {}
    state_summary = row.get("state_contract_summary") if isinstance(row.get("state_contract_summary"), dict) else {}

    if not row.get("query_id"):
        fail("evidence_sufficient", "missing_query_id", None)
    if not prompt.strip():
        fail("trace_natural", "empty_prompt", None)
    if not expected or not behavior_type:
        fail("evidence_sufficient", "missing_expected_behavior_type", expected)
    if _evidence_ids(gold_evidence) != gold_ids:
        fail(
            "evidence_sufficient",
            "gold_ids_mismatch_evidence",
            {"ids": sorted(gold_ids), "evidence_ids": sorted(_evidence_ids(gold_evidence))},
        )
    if _evidence_ids(forbidden_evidence) != forbidden_ids:
        fail(
            "evidence_sufficient",
            "forbidden_ids_mismatch_evidence",
            {"ids": sorted(forbidden_ids), "evidence_ids": sorted(_evidence_ids(forbidden_evidence))},
        )

    if row.get("memory_requirement") == "requires_memory" and behavior_type not in NON_ANSWER_BEHAVIORS and not gold_ids:
        fail("scenario_memory_required", "requires_memory_without_gold_ids", None)
    if row.get("memory_requirement") == "no_memory_required" and gold_ids and behavior_type in NO_MEMORY_BEHAVIORS:
        fail("scenario_memory_required", "no_memory_control_has_gold_answer_ids", sorted(gold_ids))

    if behavior_type not in NON_ANSWER_BEHAVIORS and not (must_include or expected.get("tool_name") or parameters):
        fail("answer_unique", "missing_positive_answer_targets", behavior_type)
    if len(set(must_include)) != len(must_include):
        fail("answer_unique", "duplicate_must_include_targets", _duplicates(must_include))
    if len(set(must_not)) != len(must_not):
        fail("answer_unique", "duplicate_must_not_targets", _duplicates(must_not))
    overlap = set(must_include) & set(must_not)
    if overlap:
        fail("answer_unique", "must_include_must_not_overlap", sorted(overlap))

    support_text = _text([gold_evidence, events, expected.get("tool_name"), parameters])
    invalid_text = _text([forbidden_evidence, events])
    if behavior_type not in NO_MEMORY_BEHAVIORS:
        for phrase in must_include:
            if _state_decision_target(phrase, behavior_type=behavior_type, expected=expected):
                continue
            if _raw_restricted_value(phrase) and _sensitive_output_policy(parameters):
                # For governed non-refusal tasks the answer must mention the
                # boundary, not the raw restricted value.
                fail("governance_boundary_clear", "raw_restricted_value_in_must_include", phrase)
            elif not _phrase_supported(phrase, support_text):
                fail("evidence_sufficient", "must_include_not_supported", phrase)
    for phrase in must_not:
        if not _phrase_supported(phrase, invalid_text):
            fail("evidence_sufficient", "must_not_not_supported", phrase)

    if expected.get("should_refuse") or behavior_type in {"refusal", "governance"} or forbidden_ids:
        rules = set(str(rule) for rule in state_summary.get("required_governance_rules") or [])
        if state_summary and not (rules & GOVERNANCE_RULES):
            fail("governance_boundary_clear", "missing_required_governance_rules", sorted(rules))
    if expected.get("should_refuse") or behavior_type in {"refusal", "governance"}:
        if not forbidden_ids:
            fail("governance_boundary_clear", "refusal_without_forbidden_ids", None)

    prompt_and_expected = prompt + "\n" + _text(expected)
    for name, pattern in PROMPT_TEMPLATE_PATTERNS.items():
        match = pattern.search(prompt)
        if match:
            fail("trace_natural", f"prompt_template_artifact_{name}", match.group(0))
    for name, pattern in SECRET_PATTERNS.items():
        match = pattern.search(prompt_and_expected)
        if match and not str(match.group(0)).startswith("SYNTHETIC_REDACTED_"):
            fail("trace_natural", f"unredacted_secret_like_{name}", match.group(0))

    if row.get("counterfactual_group_id"):
        context = row.get("counterfactual_context") if isinstance(row.get("counterfactual_context"), dict) else {}
        if context:
            for key in ("shared_prompt_across_group", "shared_task_type_across_group", "shared_probe_type_across_group"):
                if context.get(key) is not True:
                    fail("counterfactual_target_state_only", f"counterfactual_{key}_false", context.get(key))
            if not context.get("state_delta_pairs") and not context.get("gold_memory_delta_pairs"):
                fail("counterfactual_target_state_only", "counterfactual_missing_delta_pairs", None)

    applicable_checks = set(row.get("applicable_checks") or AUDIT_CHECK_FIELDS)
    for check_id in set(checks) - applicable_checks:
        checks[check_id] = True
        issues = [issue for issue in issues if issue["check_id"] != check_id]
    return {"checks": checks, "issues": issues}


def _expected(row: dict[str, Any]) -> dict[str, Any]:
    expected = row.get("expected_behavior")
    return expected if isinstance(expected, dict) else {}


def _state_contract_summary(contract: dict[str, Any]) -> dict[str, Any]:
    if not contract:
        return {}
    return {
        "active_memory_count": len(contract.get("active_memory_ids") or []),
        "deleted_memory_count": len(contract.get("deleted_memory_ids") or []),
        "forbidden_memory_count": len(contract.get("forbidden_memory_ids") or []),
        "restricted_memory_count": len(contract.get("restricted_memory_ids") or []),
        "required_governance_rules": contract.get("governance_rules") or contract.get("required_governance_rules") or [],
        "timestamp": contract.get("timestamp"),
        "transition_types": contract.get("transition_types") or [],
    }


def _resolve_path(root: Path, raw_path: str) -> Path:
    root_relative = root / raw_path
    if root_relative.exists():
        return root_relative
    repo_relative = Path(raw_path)
    if repo_relative.exists():
        return repo_relative
    return root_relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values if isinstance(value, str)}


def _evidence_ids(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(item.get("memory_id")) for item in values if isinstance(item, dict) and item.get("memory_id")}


def _strings(payload: Any) -> list[str]:
    if payload is None:
        return []
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, dict):
        out: list[str] = []
        for value in payload.values():
            out.extend(_strings(value))
        return out
    if isinstance(payload, list):
        out = []
        for value in payload:
            out.extend(_strings(value))
        return out
    return [str(payload)]


def _text(payload: Any) -> str:
    return "\n".join(_strings(payload))


def _normalize(value: str) -> str:
    return " ".join(str(value).lower().split())


def _phrase_supported(phrase: str, text: str) -> bool:
    normalized_phrase = _normalize(phrase)
    haystack = _normalize(text)
    if not normalized_phrase:
        return True
    if normalized_phrase in haystack:
        return True
    if _raw_restricted_value(normalized_phrase):
        token = _raw_restricted_value(normalized_phrase)
        return bool(token and token.lower() in haystack)
    tokens = [token for token in re.findall(r"[a-z0-9_@./:#-]+", normalized_phrase) if len(token) >= 4]
    if not tokens:
        return False
    return sum(1 for token in tokens if token in haystack) >= max(1, min(3, len(tokens)))


def _raw_restricted_value(value: str) -> str | None:
    match = re.search(r"\bSYNTHETIC_REDACTED_[A-Z0-9_-]+", value)
    return match.group(0) if match else None


def _sensitive_output_policy(parameters: dict[str, Any]) -> bool:
    return "do not reveal raw restricted values" in str(parameters.get("sensitive_output_policy") or "").lower()


def _state_decision_target(phrase: str, *, behavior_type: str, expected: dict[str, Any]) -> bool:
    normalized = _normalize(phrase)
    return normalized in STATE_DECISION_TARGETS and (behavior_type in {"refusal", "governance"} or expected.get("should_refuse"))


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)
