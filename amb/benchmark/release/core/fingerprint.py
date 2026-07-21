"""Release contract fingerprints for score-artifact currentness checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from amb.benchmark.schemas.io import read_json


FINGERPRINT_SCHEMA_VERSION = "amst-release-contract-fingerprint-v1"


def release_split_contract_fingerprint(manifest_path: str | Path, split: str) -> dict[str, Any]:
    """Hash the query contract for a materialized release split.

    The digest intentionally covers prompts, expected behavior, required and forbidden
    evidence, scoring rules, and difficulty metadata. Query IDs alone are insufficient
    because hardening passes can preserve IDs while changing the actual task contract.
    """

    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    split_files = manifest.get("split_files") if isinstance(manifest.get("split_files"), dict) else {}
    by_domain = split_files.get(split) if isinstance(split_files.get(split), dict) else {}
    rows: list[dict[str, Any]] = []
    for domain, raw_path in sorted(by_domain.items()):
        shard_path = _resolve_manifest_path(manifest_file.parent, str(raw_path))
        shard = read_json(shard_path)
        for case in shard.get("cases", []):
            if not isinstance(case, dict):
                continue
            for query in case.get("queries", []):
                if not isinstance(query, dict):
                    continue
                rows.append(_query_contract_row(domain=str(domain), case=case, query=query))
    rows.sort(key=lambda row: row["query_id"])
    contract_payload = {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "benchmark_id": manifest.get("benchmark_id"),
        "profile_id": manifest.get("profile_id"),
        "package_type": manifest.get("package_type"),
        "split": split,
        "num_queries": len(rows),
        "rows": rows,
    }
    query_contract_sha256 = _sha256_json(contract_payload)
    query_id_sha256 = hashlib.sha256("\n".join(row["query_id"] for row in rows).encode("utf-8")).hexdigest()
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "benchmark_id": manifest.get("benchmark_id"),
        "profile_id": manifest.get("profile_id"),
        "package_type": manifest.get("package_type"),
        "split": split,
        "num_queries": len(rows),
        "query_id_sha256": query_id_sha256,
        "query_contract_sha256": query_contract_sha256,
    }


def _query_contract_row(*, domain: str, case: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    expected = query.get("expected_behavior") if isinstance(query.get("expected_behavior"), dict) else {}
    return {
        "domain": query.get("domain") or domain,
        "case_id": case.get("case_id"),
        "query_id": query.get("query_id"),
        "timestamp": query.get("timestamp"),
        "prompt": query.get("prompt"),
        "task_type": query.get("task_type"),
        "probe_type": query.get("probe_type"),
        "requires_memory": query.get("requires_memory"),
        "memory_dependency": query.get("memory_dependency"),
        "scoring_rule": query.get("scoring_rule"),
        "state_contract_id": query.get("state_contract_id"),
        "counterfactual_group_id": query.get("counterfactual_group_id"),
        "gold_memory_ids": query.get("gold_memory_ids") or [],
        "forbidden_memory_ids": query.get("forbidden_memory_ids") or [],
        "expected_behavior": {
            "behavior_type": expected.get("behavior_type"),
            "must_include": expected.get("must_include") or [],
            "must_not_include": expected.get("must_not_include") or [],
            "should_refuse": expected.get("should_refuse"),
            "tool_name": expected.get("tool_name"),
            "parameters": expected.get("parameters") or {},
        },
        "difficulty": query.get("difficulty") or {},
    }


def _resolve_manifest_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    root_relative = root / path
    if root_relative.exists():
        return root_relative
    if path.exists():
        return path
    return root_relative


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
