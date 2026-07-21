"""Rank-correlation analysis between AMST reports and external benchmarks."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

from amb.benchmark.analysis.statistics import kendall_tau_b, numeric_or_none, quantile, spearman_correlation
from amb.benchmark.schemas.io import read_json, write_json


DEFAULT_RANK_BOOTSTRAP_SAMPLES = 2000
DEFAULT_RANK_BOOTSTRAP_SEED = 13013
DEFAULT_RANK_CI_CONFIDENCE = 0.95


def analyze_external_correlations(
    amst_report_paths: Iterable[str | Path],
    external_score_paths: Iterable[str | Path],
    *,
    amst_metric: str = "lifecycle.amq",
    external_metric: str = "score",
    bootstrap_samples: int = DEFAULT_RANK_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_RANK_BOOTSTRAP_SEED,
    bootstrap_confidence: float = DEFAULT_RANK_CI_CONFIDENCE,
) -> dict[str, Any]:
    amst_scores = _load_amst_scores(amst_report_paths, amst_metric)
    external_results = []
    rng = random.Random(bootstrap_seed)
    for path in external_score_paths:
        score_path = Path(path)
        external_artifact = _load_external_artifact(score_path, external_metric)
        external_scores = external_artifact["scores"]
        common_systems = sorted(set(amst_scores) & set(external_scores))
        amst_values = [amst_scores[system] for system in common_systems]
        external_values = [external_scores[system] for system in common_systems]
        spearman = spearman_correlation(amst_values, external_values)
        kendall = kendall_tau_b(amst_values, external_values)
        rank_bootstrap = rank_correlation_bootstrap_ci(
            amst_values,
            external_values,
            rng=rng,
            samples=bootstrap_samples,
            confidence=bootstrap_confidence,
        )
        external_results.append(
            {
                "external_score_path": str(score_path),
                "external_score_schema_version": external_artifact.get("score_schema_version"),
                "external_benchmark_id": external_artifact.get("benchmark_id"),
                "external_source_artifact": external_artifact.get("source_artifact"),
                "external_run_config": external_artifact.get("run_config"),
                "external_metric": external_metric,
                "num_common_systems": len(common_systems),
                "spearman": spearman,
                "spearman_ci95_low": rank_bootstrap["spearman"]["lower"],
                "spearman_ci95_high": rank_bootstrap["spearman"]["upper"],
                "kendall_tau_b": kendall,
                "kendall_tau_b_ci95_low": rank_bootstrap["kendall_tau_b"]["lower"],
                "kendall_tau_b_ci95_high": rank_bootstrap["kendall_tau_b"]["upper"],
                "rank_bootstrap": {
                    **rank_bootstrap,
                    "seed": bootstrap_seed,
                },
                "systems": [
                    {
                        "system_id": system,
                        "amst_score": amst_scores[system],
                        "external_score": external_scores[system],
                    }
                    for system in common_systems
                ],
                "missing_in_amst": sorted(set(external_scores) - set(amst_scores)),
                "missing_in_external": sorted(set(amst_scores) - set(external_scores)),
            }
        )
    return {
        "analysis_schema_version": "amst-external-correlation-v1",
        "amst_metric": amst_metric,
        "num_amst_systems": len(amst_scores),
        "rank_bootstrap": {
            "method": "paired percentile bootstrap over shared systems",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence": bootstrap_confidence,
        },
        "external_results": external_results,
    }


def write_external_correlations(
    amst_report_paths: Iterable[str | Path],
    external_score_paths: Iterable[str | Path],
    output: str | Path,
    *,
    amst_metric: str = "lifecycle.amq",
    external_metric: str = "score",
    bootstrap_samples: int = DEFAULT_RANK_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_RANK_BOOTSTRAP_SEED,
    bootstrap_confidence: float = DEFAULT_RANK_CI_CONFIDENCE,
) -> dict[str, Any]:
    report = analyze_external_correlations(
        amst_report_paths,
        external_score_paths,
        amst_metric=amst_metric,
        external_metric=external_metric,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_confidence=bootstrap_confidence,
    )
    report = _localize_written_external_correlation_report(report, Path(output))
    write_json(output, report)
    return report


def rank_correlation_bootstrap_ci(
    first: list[float],
    second: list[float],
    *,
    rng: random.Random,
    samples: int,
    confidence: float = DEFAULT_RANK_CI_CONFIDENCE,
) -> dict[str, Any]:
    if len(first) != len(second):
        raise ValueError("correlation bootstrap inputs must have the same length")
    alpha = (1.0 - confidence) / 2.0
    if len(first) < 2 or samples <= 0:
        return {
            "method": "paired percentile bootstrap over shared systems",
            "confidence": confidence,
            "requested_samples": samples,
            "valid_samples": 0,
            "num_observations": len(first),
            "spearman": {"lower": None, "upper": None},
            "kendall_tau_b": {"lower": None, "upper": None},
        }

    n = len(first)
    spearman_values: list[float] = []
    kendall_values: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        sampled_first = [first[index] for index in indices]
        sampled_second = [second[index] for index in indices]
        spearman = spearman_correlation(sampled_first, sampled_second)
        kendall = kendall_tau_b(sampled_first, sampled_second)
        if spearman is not None:
            spearman_values.append(spearman)
        if kendall is not None:
            kendall_values.append(kendall)

    return {
        "method": "paired percentile bootstrap over shared systems",
        "confidence": confidence,
        "requested_samples": samples,
        "valid_samples": min(len(spearman_values), len(kendall_values)),
        "num_observations": n,
        "spearman": _percentile_interval(spearman_values, alpha=alpha),
        "kendall_tau_b": _percentile_interval(kendall_values, alpha=alpha),
    }


def _percentile_interval(values: list[float], *, alpha: float) -> dict[str, float | None]:
    if not values:
        return {"lower": None, "upper": None}
    ordered = sorted(values)
    return {
        "lower": quantile(ordered, alpha),
        "upper": quantile(ordered, 1.0 - alpha),
    }


def _load_amst_scores(paths: Iterable[str | Path], metric: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for path in paths:
        report = read_json(path)
        system_id = str(report.get("system_id", Path(path).stem))
        value = numeric_or_none(report.get("aggregate", {}).get(metric))
        if value is None:
            raise ValueError(f"{path} missing numeric aggregate metric {metric!r}")
        scores[system_id] = value
    return scores


def _load_external_scores(path: Path, metric: str) -> dict[str, float]:
    return _load_external_artifact(path, metric)["scores"]


def _load_external_artifact(path: Path, metric: str) -> dict[str, Any]:
    if path.suffix.lower() == ".csv":
        return {"scores": _load_external_csv(path, metric), "source_format": "csv"}
    data = read_json(path)
    artifact = {
        "scores": _load_external_json(data, metric, path),
        "source_format": path.suffix.lower().lstrip(".") or "json",
    }
    if isinstance(data, dict) and data.get("score_schema_version") == "amst-external-scores-v1":
        artifact.update(
            {
                "score_schema_version": data.get("score_schema_version"),
                "benchmark_id": data.get("benchmark_id"),
                "source_artifact": data.get("source_artifact"),
                "run_config": data.get("run_config"),
            }
        )
    return artifact


def _load_external_csv(path: Path, metric: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if "system_id" not in (reader.fieldnames or ()):
            raise ValueError(f"{path} must contain a system_id column")
        for row in reader:
            value = numeric_or_none(_floatish(row.get(metric)))
            if value is None:
                continue
            scores[str(row["system_id"])] = value
    return scores


def _load_external_json(data: Any, metric: str, path: Path) -> dict[str, float]:
    if isinstance(data, dict) and "systems" in data and isinstance(data["systems"], list):
        return _rows_to_scores(data["systems"], metric)
    if isinstance(data, dict) and "rows" in data and isinstance(data["rows"], list):
        return _rows_to_scores(data["rows"], metric)
    if isinstance(data, list):
        return _rows_to_scores(data, metric)
    if isinstance(data, dict) and "system_id" in data and "aggregate" in data:
        value = numeric_or_none(data.get("aggregate", {}).get(metric))
        if value is None:
            raise ValueError(f"{path} missing numeric aggregate metric {metric!r}")
        return {str(data["system_id"]): value}
    if isinstance(data, dict):
        scores = {}
        for system_id, raw_value in data.items():
            value = numeric_or_none(raw_value)
            if value is not None:
                scores[str(system_id)] = value
        if scores:
            return scores
    raise ValueError(f"{path} is not a recognized external score artifact")


def _rows_to_scores(rows: list[Any], metric: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict) or "system_id" not in row:
            continue
        value = numeric_or_none(row.get(metric))
        if value is None:
            value = numeric_or_none(row.get("aggregate", {}).get(metric)) if isinstance(row.get("aggregate"), dict) else None
        if value is not None:
            scores[str(row["system_id"])] = value
    return scores


def _floatish(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _localize_written_external_correlation_report(
    report: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    project_root = _external_contract_project_root(output_path.parent)
    if project_root is None:
        return report
    normalized = json.loads(json.dumps(report))
    root_ref = _project_root_ref(output_path.parent, project_root=project_root)
    if root_ref is not None:
        normalized["root"] = root_ref
    results = normalized.get("external_results")
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            for field in ("external_score_path", "external_source_artifact"):
                raw_path = item.get(field)
                if isinstance(raw_path, str) and raw_path:
                    item[field] = _normalize_project_relative_path_string(
                        raw_path,
                        project_root=project_root,
                        base_dir=output_path.parent,
                    )
    return normalized


def _normalize_project_relative_path_string(
    raw_path: str,
    *,
    project_root: Path,
    base_dir: Path,
) -> str:
    path = Path(raw_path)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend((project_root / path, base_dir / path, Path.cwd() / path))
    for candidate in candidates:
        if candidate.exists():
            return _project_relative_path_or_absolute(candidate.resolve(), project_root)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return _project_relative_path_or_absolute(path.resolve(), project_root)
    except OSError:
        return raw_path


def _external_contract_project_root(base_dir: Path) -> Path | None:
    resolved_base_dir = base_dir.resolve()
    for candidate in (resolved_base_dir, *resolved_base_dir.parents):
        if candidate.name != "external":
            continue
        if candidate.parent.name != "reports":
            continue
        return candidate.parent.parent
    return None


def _project_relative_path_or_absolute(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def _project_root_ref(base_dir: Path, *, project_root: Path | None) -> str | None:
    if project_root is None:
        return None
    try:
        return Path(os.path.relpath(project_root.resolve(), base_dir.resolve())).as_posix()
    except ValueError:
        return str(project_root.resolve())
