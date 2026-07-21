"""Materialize release splits from an in-memory benchmark."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from amb.benchmark.quality.audit import audit_benchmark
from amb.benchmark.release.artifacts import artifact_info, write_audit_template
from amb.benchmark.release.splits import (
    RELEASE_SPLITS,
    ReleaseConfig,
    assign_release_splits,
    case_group_id,
    split_strategy_for_benchmark,
)
from amb.benchmark.schemas.io import write_json
from amb.benchmark.schemas.models import Benchmark, Case


def build_release(
    benchmark: Benchmark,
    output_dir: str | Path,
    config: ReleaseConfig | None = None,
    *,
    source_benchmark_path: str | Path | None = None,
    build_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or ReleaseConfig()
    split_groups = assign_release_splits(benchmark, cfg)
    split_cases = {split: tuple(case for group in groups for case in group) for split, groups in split_groups.items()}
    output = Path(output_dir)
    written: dict[str, str] = {}
    split_reports: dict[str, Any] = {}
    split_artifacts: dict[str, Any] = {}

    for split in RELEASE_SPLITS:
        split_benchmark = Benchmark(
            schema_version=benchmark.schema_version,
            benchmark_id=f"{benchmark.benchmark_id}-{split}",
            name=f"{benchmark.name} {split}",
            cases=split_cases[split],
        )
        split_path = output / "data" / split / "benchmark.json"
        write_json(split_path, asdict(split_benchmark))
        written[split] = str(split_path)
        split_artifacts[split] = artifact_info(split_path)
        split_reports[split] = _split_summary(split_benchmark)

    template_path = output / "data" / "audit_subset" / "annotation_template.jsonl"
    write_audit_template(template_path, split_cases["audit_subset"])
    written["audit_annotation_template"] = str(template_path)
    split_artifacts["audit_annotation_template"] = artifact_info(template_path)

    manifest = {
        "schema_version": benchmark.schema_version,
        "benchmark_id": benchmark.benchmark_id,
        "build_timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "builder": {
            "name": "agent_memory_benchmark",
            "schema_version": benchmark.schema_version,
        },
        "release_status": "generated_unreviewed",
        "split_strategy": split_strategy_for_benchmark(benchmark, cfg),
        "release_config": asdict(cfg),
        "build_metadata": build_metadata or {},
        "source_benchmark": _source_benchmark_info(source_benchmark_path),
        "visibility": {
            "public_dev": "public",
            "public_test": "public",
            "audit_subset": "controlled_public",
            "hidden_test": "private_leaderboard_only",
        },
        "audit_plan": {
            "audit_required": True,
            "audit_fraction_target": cfg.audit_fraction,
            "human_audit_status": "template_generated",
            "audit_template_file": written["audit_annotation_template"],
            "audit_annotations_file": None,
            "agreement_metrics": None,
        },
        "split_files": written,
        "split_artifacts": split_artifacts,
        "split_reports": split_reports,
        "group_assignments": {split: [case_group_id(group[0]) for group in groups] for split, groups in split_groups.items()},
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _split_summary(benchmark: Benchmark) -> dict[str, Any]:
    audit = audit_benchmark(benchmark)
    return {
        "num_cases": audit["num_cases"],
        "num_queries": audit["num_queries"],
        "num_memories": audit["num_memories"],
        "num_events": audit["num_events"],
        "num_state_contracts": audit["num_state_contracts"],
        "domains": audit["domains"],
        "task_types": audit["task_types"],
        "probe_types": audit["probe_types"],
        "query_difficulty_levels": audit.get("query_difficulty_levels", {}),
        "query_difficulty_keys": audit.get("query_difficulty_keys", []),
        "counterfactual_edits": audit["counterfactual_edits"],
        "stress_families": audit.get("stress_families", {}),
        "stress_tags": audit.get("stress_tags", {}),
        "renderer_coverage": audit["renderer_coverage"],
        "coverage": audit["coverage"],
        "quality_gates_passed": audit["quality_gates_passed"],
        "construction_gates_passed": audit["construction_gates_passed"],
        "data_quality_gates_passed": audit["data_quality_gates_passed"],
    }


def _source_benchmark_info(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    source = Path(path)
    if not source.exists():
        return {"path": str(source), "exists": False}
    info = artifact_info(source)
    info["exists"] = True
    return info
