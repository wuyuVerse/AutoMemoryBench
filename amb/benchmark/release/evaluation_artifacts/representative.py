"""Representative baseline artifact generation for public release packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from amb.benchmark.analysis import analyze_report_files
from amb.benchmark.evaluation.scoring import DEFAULT_RETRIEVAL_K
from amb.benchmark.leaderboard import write_leaderboard_summary
from amb.benchmark.artifact_contract import localize_report_contract
from amb.benchmark.release.evaluation import evaluate_release_split_baselines
from amb.benchmark.schemas.io import read_json, write_json


REPRESENTATIVE_BASELINES = (
    "no_memory",
    "full_history",
    "graph_memory",
    "oracle_memory",
)


def build_representative_baseline_artifacts(
    manifest_path: str | Path,
    *,
    split: str = "public_dev",
    output_dir: str | Path = "reports/examples",
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    bootstrap_samples: int = 200,
    seed: int = 13,
) -> dict[str, Any]:
    """Generate representative baseline reports, analysis, and leaderboard."""

    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    manifest_benchmark_id = str(manifest.get("benchmark_id", "release"))
    benchmark_id = f"{manifest_benchmark_id}-{split}"
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    prefix = _artifact_prefix(manifest_benchmark_id, split)
    report_paths: list[str] = []
    report_artifacts: dict[str, str] = {}

    reports = evaluate_release_split_baselines(
        manifest_file,
        split=split,
        baseline_kinds=REPRESENTATIVE_BASELINES,
        retrieval_k=retrieval_k,
    )

    for kind in REPRESENTATIVE_BASELINES:
        report = reports[kind]
        path = output_root / f"{prefix}_{kind}_report.json"
        report = localize_report_contract(
            report,
            output_path=path,
            project_root_hints=(manifest_path,),
        )
        write_json(path, report)
        report_paths.append(str(path))
        report_artifacts[kind] = str(path)

    analysis = analyze_report_files(report_paths, seed=seed, bootstrap_samples=bootstrap_samples)
    analysis_path = output_root / f"{prefix}_representative_baselines_analysis.json"
    analysis = localize_report_contract(
        analysis,
        output_path=analysis_path,
        project_root_hints=tuple(report_paths),
    )
    write_json(analysis_path, analysis)

    leaderboard_path = output_root / f"{prefix}_leaderboard.json"
    leaderboard = write_leaderboard_summary(report_paths, leaderboard_path)

    return {
        "benchmark_id": benchmark_id,
        "release_split": split,
        "representative_baselines": list(REPRESENTATIVE_BASELINES),
        "report_paths": report_artifacts,
        "analysis_path": str(analysis_path),
        "leaderboard_path": str(leaderboard_path),
        "top_system_id": leaderboard["rows"][0]["system_id"] if leaderboard["rows"] else None,
        "num_reports": len(report_paths),
    }


def _artifact_prefix(benchmark_id: str, split: str) -> str:
    safe_benchmark = benchmark_id.replace("-", "_")
    return f"{safe_benchmark}_{split}"
