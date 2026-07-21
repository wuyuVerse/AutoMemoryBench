"""Sampling helpers for release benchmarks while preserving benchmark structure."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import random
from typing import Any

from amb.benchmark.artifact_contract import localize_report_contract
from amb.benchmark.schemas.io import load_benchmark, read_json, write_json
from amb.benchmark.schemas.models import Benchmark, Case


def sample_release_split(
    manifest_path: str | Path,
    *,
    split: str,
    output_path: str | Path,
    seed: int = 13,
    groups_per_domain: int = 2,
    domains: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build a small benchmark sample from a release split.

    Sampling is group-preserving by `counterfactual_group_id`, stratified by
    domain, and retains all case variants within each selected group.
    """

    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    split_files = manifest.get("split_files", {}).get(split)
    if not isinstance(split_files, dict) or not split_files:
        raise ValueError(f"release split {split!r} must exist and expose domain shard files")
    if groups_per_domain <= 0:
        raise ValueError("groups_per_domain must be positive")
    selected_domains = tuple(sorted({str(domain) for domain in domains})) if domains else None

    rng = random.Random(seed)
    sampled_cases: list[Case] = []
    sample_summary: dict[str, Any] = {
        "source_manifest": str(manifest_file),
        "source_benchmark_id": str(manifest.get("benchmark_id", "release")),
        "source_split": split,
        "seed": seed,
        "groups_per_domain": groups_per_domain,
        "domains": {},
    }

    for domain, raw_path in sorted(split_files.items()):
        if selected_domains is not None and domain not in selected_domains:
            continue
        benchmark_path = _resolve_path(str(raw_path), manifest_file.parent)
        benchmark = load_benchmark(benchmark_path)
        groups = _case_groups(benchmark.cases)
        ordered_groups = list(groups.items())
        rng.shuffle(ordered_groups)
        chosen = ordered_groups[: min(groups_per_domain, len(ordered_groups))]
        domain_cases = [case for _, items in sorted(chosen) for case in items]
        sampled_cases.extend(domain_cases)
        sample_summary["domains"][domain] = {
            "available_groups": len(groups),
            "sampled_groups": len(chosen),
            "sampled_case_variants": len(domain_cases),
            "sampled_group_ids": [group_id for group_id, _ in sorted(chosen)],
        }

    benchmark_id = f"{manifest.get('benchmark_id', 'release')}-{split}-sample-g{groups_per_domain}-s{seed}"
    sample = Benchmark(
        schema_version="1.0.0",
        benchmark_id=benchmark_id,
        name=f"{manifest.get('benchmark_id', 'release')} {split} sample",
        cases=tuple(sampled_cases),
    )
    output = Path(output_path)
    payload = localize_report_contract(
        asdict(sample),
        output_path=output,
        project_root_hints=(manifest_path,),
    )
    write_json(output, payload)
    return {
        "benchmark_path": str(output),
        "benchmark_id": benchmark_id,
        "num_cases": len(sample.cases),
        "num_queries": sum(len(case.queries) for case in sample.cases),
        "num_domains": len(sample_summary["domains"]),
        "sample_summary": sample_summary,
    }


def _case_groups(cases: tuple[Case, ...]) -> dict[str, tuple[Case, ...]]:
    grouped: dict[str, list[Case]] = defaultdict(list)
    for case in cases:
        group_id = str(case.difficulty.values.get("counterfactual_group_id") or case.case_id)
        grouped[group_id].append(case)
    return {group_id: tuple(items) for group_id, items in sorted(grouped.items())}


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return base / path
