"""Normalize external benchmark score artifacts for AMST correlation analysis."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from amb.benchmark.analysis.statistics import numeric_or_none
from amb.benchmark.schemas.io import read_json, write_json

EXTERNAL_SCORE_SCHEMA_VERSION = "amst-external-scores-v1"
LONGMEMEVAL_SCORE_MANIFEST_SCHEMA_VERSION = "amst-longmemeval-score-manifest-v1"
MERGED_EXTERNAL_SCORE_MANIFEST_SCHEMA_VERSION = "amst-merged-external-score-manifest-v1"


def normalize_external_scores(
    input_path: str | Path,
    *,
    benchmark_id: str,
    metric: str = "score",
    system_id_field: str = "system_id",
    score_field: str = "score",
    run_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Normalize a JSON/CSV benchmark result into a provenance-preserving score artifact."""

    source = Path(input_path)
    run_config = read_json(run_config_path) if run_config_path is not None else None
    rows = _load_rows(source)
    systems = _rows_to_system_scores(
        rows,
        system_id_field=system_id_field,
        score_field=score_field,
        metric=metric,
        source_artifact=str(source),
    )
    if not systems:
        raise ValueError(f"{source} did not contain any numeric external scores")
    return {
        "score_schema_version": EXTERNAL_SCORE_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "metric": metric,
        "source_artifact": str(source),
        "source_format": source.suffix.lower().lstrip(".") or "unknown",
        "system_id_field": system_id_field,
        "score_field": score_field,
        "num_systems": len(systems),
        "run_config": run_config,
        "systems": systems,
    }


def write_normalized_external_scores(
    input_path: str | Path,
    output_path: str | Path,
    *,
    benchmark_id: str,
    metric: str = "score",
    system_id_field: str = "system_id",
    score_field: str = "score",
    run_config_path: str | Path | None = None,
) -> dict[str, Any]:
    report = normalize_external_scores(
        input_path,
        benchmark_id=benchmark_id,
        metric=metric,
        system_id_field=system_id_field,
        score_field=score_field,
        run_config_path=run_config_path,
    )
    report = _localize_written_external_score_report(report, Path(output_path))
    write_json(output_path, report)
    return report


def normalize_longmemeval_scores(
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Normalize LongMemEval official outputs into a correlation-ready score artifact.

    The input manifest lists one result artifact per system. Each artifact may be
    either:

    - an official/system-level LongMemEval result JSON with `metrics_by_cutoff`;
    - an official judged QA log (`jsonl` or JSON list) with `question_id` and
      `autoeval_label.label`.
    """

    source = Path(manifest_path)
    manifest = read_json(source)
    if not isinstance(manifest, dict):
        raise ValueError("LongMemEval score manifest must be a JSON object")
    schema_version = str(manifest.get("schema_version") or LONGMEMEVAL_SCORE_MANIFEST_SCHEMA_VERSION)
    if schema_version != LONGMEMEVAL_SCORE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"LongMemEval score manifest schema_version must be {LONGMEMEVAL_SCORE_MANIFEST_SCHEMA_VERSION}")
    benchmark_id = str(manifest.get("benchmark_id") or "longmemeval")
    if benchmark_id != "longmemeval":
        raise ValueError("LongMemEval score manifest benchmark_id must be 'longmemeval'")
    metric = str(manifest.get("metric") or "accuracy")
    run_config = manifest.get("run_config")
    systems = manifest.get("systems")
    if not isinstance(systems, list) or not systems:
        raise ValueError("LongMemEval score manifest must contain a non-empty systems list")

    normalized_systems: list[dict[str, Any]] = []
    normalization_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    default_reference = manifest.get("reference_path")
    default_cutoff = manifest.get("cutoff")

    for index, item in enumerate(systems, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"systems[{index}] must be an object")
        system_id = str(item.get("system_id") or "").strip()
        if not system_id:
            raise ValueError(f"systems[{index}].system_id is required")
        if system_id in seen_ids:
            raise ValueError(f"duplicate system_id in LongMemEval score manifest: {system_id}")
        seen_ids.add(system_id)
        raw_result_path = item.get("result_path")
        if not raw_result_path:
            raise ValueError(f"systems[{index}].result_path is required")
        result_path = Path(raw_result_path)
        if not result_path.is_absolute():
            result_path = source.parent / result_path
        reference_path = item.get("reference_path", default_reference)
        if reference_path is not None:
            reference = Path(reference_path)
            if not reference.is_absolute():
                reference = source.parent / reference
        else:
            reference = None
        cutoff = item.get("cutoff", default_cutoff)
        score, detail = _longmemeval_score_from_result(
            result_path,
            reference_path=reference,
            cutoff=cutoff,
        )
        normalized_systems.append(
            {
                "system_id": system_id,
                "score": score,
                "metric": metric,
                "source_artifact": str(result_path),
                "source_row": index,
            }
        )
        normalization_rows.append({"system_id": system_id, **detail})

    return {
        "score_schema_version": EXTERNAL_SCORE_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "metric": metric,
        "source_artifact": str(source),
        "source_format": "longmemeval_manifest",
        "system_id_field": "systems[].system_id",
        "score_field": "derived.accuracy",
        "num_systems": len(normalized_systems),
        "run_config": run_config,
        "systems": normalized_systems,
        "normalization_details": {
            "input_schema_version": schema_version,
            "rows": normalization_rows,
        },
    }


def write_normalized_longmemeval_scores(
    manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    report = normalize_longmemeval_scores(manifest_path)
    report = _localize_written_external_score_report(report, Path(output_path))
    write_json(output_path, report)
    return report


def merge_normalized_external_scores(
    score_paths: list[str | Path],
    *,
    benchmark_id: str | None = None,
    system_cohort: list[str] | None = None,
    source_artifact: str | None = None,
    replace_system_ids: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge compatible normalized score artifacts into one same-cohort artifact.

    Provider return packets need the refreshed score artifact to contain the
    full target cohort, e.g. control anchors plus one real memory system. This
    helper merges already-normalized score artifacts while preserving component
    provenance instead of pretending that all rows came from a single raw file.
    """

    if len(score_paths) < 2:
        raise ValueError("at least two normalized score artifacts are required")
    artifacts: list[tuple[Path, dict[str, Any]]] = []
    for raw_path in score_paths:
        path = Path(raw_path)
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must be a JSON object")
        if payload.get("score_schema_version") != EXTERNAL_SCORE_SCHEMA_VERSION:
            raise ValueError(f"{path} score_schema_version must be {EXTERNAL_SCORE_SCHEMA_VERSION}")
        artifacts.append((path, payload))

    benchmark_ids = {str(payload.get("benchmark_id") or "") for _, payload in artifacts}
    benchmark_ids.discard("")
    if benchmark_id is None:
        if len(benchmark_ids) != 1:
            raise ValueError("merged score artifacts must share exactly one benchmark_id")
        benchmark_id = next(iter(benchmark_ids))
    elif benchmark_ids != {benchmark_id}:
        raise ValueError(f"all score artifacts must have benchmark_id {benchmark_id!r}")

    metrics = {str(payload.get("metric") or "score") for _, payload in artifacts}
    if len(metrics) != 1:
        raise ValueError("merged score artifacts must use the same metric")
    metric = next(iter(metrics))

    replace_ids = {str(item) for item in (replace_system_ids or [])}
    systems_by_id: dict[str, dict[str, Any]] = {}
    artifact_rows: list[dict[str, Any]] = []
    for artifact_index, (path, payload) in enumerate(artifacts, start=1):
        systems = payload.get("systems")
        if not isinstance(systems, list) or not systems:
            raise ValueError(f"{path} must contain a non-empty systems list")
        component_ids: list[str] = []
        for row_index, row in enumerate(systems, start=1):
            if not isinstance(row, dict):
                raise ValueError(f"{path} systems[{row_index}] must be an object")
            system_id = str(row.get("system_id") or "").strip()
            if not system_id:
                raise ValueError(f"{path} systems[{row_index}].system_id is required")
            if system_id in systems_by_id and system_id not in replace_ids:
                raise ValueError(f"duplicate system_id across merged artifacts: {system_id}")
            score = numeric_or_none(row.get("score"))
            if score is None:
                raise ValueError(f"{path} systems[{row_index}].score must be numeric")
            merged_row = dict(row)
            merged_row["system_id"] = system_id
            merged_row["score"] = score
            merged_row.setdefault("metric", metric)
            merged_row["source_score_artifact"] = str(path)
            merged_row["source_score_row"] = row_index
            systems_by_id[system_id] = merged_row
            component_ids.append(system_id)
        artifact_rows.append(
            {
                "source_score_artifact": str(path),
                "benchmark_id": payload.get("benchmark_id"),
                "metric": payload.get("metric"),
                "source_artifact": payload.get("source_artifact"),
                "run_config": payload.get("run_config"),
                "system_ids": component_ids,
                "artifact_index": artifact_index,
            }
        )

    target_cohort = [str(item) for item in (system_cohort or list(systems_by_id))]
    if sorted(target_cohort) != sorted(systems_by_id):
        raise ValueError("system_cohort must match the merged scored systems")
    systems = [systems_by_id[system_id] for system_id in target_cohort]
    run_config = _merged_external_run_config(benchmark_id, artifacts, target_cohort)
    manifest = {
        "schema_version": MERGED_EXTERNAL_SCORE_MANIFEST_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "metric": metric,
        "target_system_cohort": target_cohort,
        "input_score_artifacts": artifact_rows,
    }
    report = {
        "score_schema_version": EXTERNAL_SCORE_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "metric": metric,
        "source_artifact": source_artifact or "",
        "source_format": "merged_normalized_score_artifacts",
        "system_id_field": "systems[].system_id",
        "score_field": "systems[].score",
        "num_systems": len(systems),
        "run_config": run_config,
        "systems": systems,
        "merge_details": manifest,
    }
    return report, manifest


def write_merged_normalized_external_scores(
    score_paths: list[str | Path],
    output_path: str | Path,
    *,
    benchmark_id: str | None = None,
    system_cohort: list[str] | None = None,
    source_manifest_path: str | Path | None = None,
    replace_system_ids: list[str] | None = None,
) -> dict[str, Any]:
    output = Path(output_path)
    manifest_path = (
        Path(source_manifest_path)
        if source_manifest_path is not None
        else output.with_name(f"{output.stem}_source_manifest.json")
    )
    report, manifest = merge_normalized_external_scores(
        score_paths,
        benchmark_id=benchmark_id,
        system_cohort=system_cohort,
        source_artifact=str(manifest_path),
        replace_system_ids=replace_system_ids,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, _localize_external_score_merge_manifest(manifest, manifest_path))
    report = _localize_written_external_score_report(report, output)
    write_json(output, report)
    return report


def _merged_external_run_config(
    benchmark_id: str,
    artifacts: list[tuple[Path, dict[str, Any]]],
    target_cohort: list[str],
) -> dict[str, Any]:
    component_configs = [
        {
            "source_score_artifact": str(path),
            "source_artifact": payload.get("source_artifact"),
            "run_config": payload.get("run_config"),
        }
        for path, payload in artifacts
    ]
    merged: dict[str, Any] = {}
    for field, fallback in (
        ("benchmark_version", f"{benchmark_id}_merged_external_cohort"),
        ("split", "merged_external_cohort"),
        ("execution_protocol", "merged_normalized_external_scores_v1"),
    ):
        values = [
            payload.get("run_config", {}).get(field)
            for _, payload in artifacts
            if isinstance(payload.get("run_config"), dict) and payload.get("run_config", {}).get(field)
        ]
        unique_values = {json.dumps(value, sort_keys=True) for value in values}
        merged[field] = values[0] if values and len(unique_values) == 1 else fallback
    merged["system_cohort"] = target_cohort
    merged["merge_protocol"] = "control_anchor_plus_provider_normalized_scores_v1"
    merged["component_run_configs"] = component_configs
    return merged


def _localize_external_score_merge_manifest(manifest: dict[str, Any], output_path: Path) -> dict[str, Any]:
    project_root = _external_contract_project_root(output_path.parent)
    if project_root is None:
        return manifest
    normalized = json.loads(json.dumps(manifest))
    root_ref = _project_root_ref(output_path.parent, project_root=project_root)
    if root_ref is not None:
        normalized["root"] = root_ref
    _normalize_named_path_fields(
        normalized,
        field_names={"source_artifact", "source_score_artifact", "reference_artifact"},
        project_root=project_root,
        base_dir=output_path.parent,
    )
    return normalized


def _localize_written_external_score_report(
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
    _normalize_named_path_fields(
        normalized,
        field_names={"source_artifact", "reference_artifact"},
        project_root=project_root,
        base_dir=output_path.parent,
    )
    return normalized


def _load_rows(path: Path) -> list[Any]:
    if path.suffix.lower() == ".csv":
        return _load_csv_rows(path)
    data = read_json(path)
    if isinstance(data, dict) and isinstance(data.get("systems"), list):
        return data["systems"]
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return data["rows"]
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "system_id" in data:
        return [data]
    if isinstance(data, dict):
        return [{"system_id": system_id, "score": score} for system_id, score in data.items()]
    raise ValueError(f"{path} is not a recognized external score artifact")


def _longmemeval_score_from_result(
    result_path: Path,
    *,
    reference_path: Path | None,
    cutoff: str | None,
) -> tuple[float, dict[str, Any]]:
    if result_path.suffix.lower() == ".jsonl" or ".eval-results-" in result_path.name:
        rows = _load_jsonl_rows(result_path)
        score, detail = _longmemeval_score_from_judged_rows(rows, reference_path=reference_path)
        detail["source_format"] = "official_judged_log_jsonl"
        return score, detail

    data = read_json(result_path)
    if isinstance(data, dict) and isinstance(data.get("metrics_by_cutoff"), dict):
        score, detail = _longmemeval_score_from_metrics_payload(data, cutoff=cutoff)
        detail["source_format"] = "longmemeval_metrics_json"
        return score, detail
    if isinstance(data, list):
        score, detail = _longmemeval_score_from_judged_rows(data, reference_path=reference_path)
        detail["source_format"] = "official_judged_log_json"
        return score, detail
    if isinstance(data, dict) and isinstance(data.get("evaluations"), list) and isinstance(data.get("metrics_by_cutoff"), dict):
        score, detail = _longmemeval_score_from_metrics_payload(data, cutoff=cutoff)
        detail["source_format"] = "longmemeval_metrics_json"
        return score, detail
    raise ValueError(f"{result_path} is not a recognized LongMemEval result artifact")


def _longmemeval_score_from_metrics_payload(
    data: dict[str, Any],
    *,
    cutoff: str | None,
) -> tuple[float, dict[str, Any]]:
    metrics_by_cutoff = data.get("metrics_by_cutoff")
    if not isinstance(metrics_by_cutoff, dict) or not metrics_by_cutoff:
        raise ValueError("LongMemEval metrics payload must contain non-empty metrics_by_cutoff")
    selected_cutoff = str(cutoff) if cutoff is not None else next(reversed(metrics_by_cutoff))
    selected = metrics_by_cutoff.get(selected_cutoff)
    if not isinstance(selected, dict):
        raise ValueError(f"LongMemEval metrics payload does not contain cutoff {selected_cutoff!r}")
    overall = selected.get("overall")
    if not isinstance(overall, dict):
        raise ValueError(f"LongMemEval metrics payload cutoff {selected_cutoff!r} must contain overall metrics")
    accuracy = numeric_or_none(overall.get("accuracy"))
    if accuracy is None:
        pass_rate = numeric_or_none(overall.get("pass_rate"))
        if pass_rate is None:
            passed = numeric_or_none(overall.get("passed"))
            total = numeric_or_none(overall.get("total"))
            if passed is not None and total not in (None, 0):
                accuracy = float(passed) / float(total)
        elif pass_rate is not None:
            accuracy = float(pass_rate) / 100.0 if float(pass_rate) > 1.0 else float(pass_rate)
    elif accuracy is not None and float(accuracy) > 1.0:
        accuracy = float(accuracy) / 100.0
    if accuracy is None:
        raise ValueError(f"LongMemEval metrics payload cutoff {selected_cutoff!r} does not expose accuracy/pass_rate")
    total = numeric_or_none(overall.get("total"))
    return float(accuracy), {
        "selected_cutoff": selected_cutoff,
        "num_items": int(total) if total is not None else None,
    }


def _longmemeval_score_from_judged_rows(
    rows: list[Any],
    *,
    reference_path: Path | None,
) -> tuple[float, dict[str, Any]]:
    if not rows:
        raise ValueError("LongMemEval judged log must contain at least one row")
    labels: list[float] = []
    seen_qids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"LongMemEval judged log row {index} must be an object")
        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            raise ValueError(f"LongMemEval judged log row {index} is missing question_id")
        if question_id in seen_qids:
            raise ValueError(f"duplicate LongMemEval question_id in judged log: {question_id}")
        seen_qids.add(question_id)
        autoeval = row.get("autoeval_label")
        label = None
        if isinstance(autoeval, dict):
            raw = autoeval.get("label")
            if isinstance(raw, bool):
                label = raw
            elif isinstance(raw, str):
                lowered = raw.strip().lower()
                if lowered in {"yes", "true", "1"}:
                    label = True
                elif lowered in {"no", "false", "0"}:
                    label = False
        if label is None:
            raise ValueError(f"LongMemEval judged log row {index} is missing boolean autoeval_label.label")
        labels.append(1.0 if label else 0.0)
    if reference_path is not None:
        reference_rows = read_json(reference_path)
        if not isinstance(reference_rows, list):
            raise ValueError(f"{reference_path} must contain a JSON list of LongMemEval references")
        reference_ids = {
            str(item.get("question_id"))
            for item in reference_rows
            if isinstance(item, dict) and item.get("question_id") is not None
        }
        missing = sorted(seen_qids - reference_ids)
        if missing:
            raise ValueError(f"{reference_path} does not contain question_ids present in judged log: {missing[:3]}")
    return sum(labels) / len(labels), {
        "selected_cutoff": None,
        "num_items": len(labels),
        "reference_artifact": None if reference_path is None else str(reference_path),
    }


def _load_jsonl_rows(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSONL row at line {line_number}: {exc}") from exc
    return rows


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _rows_to_system_scores(
    rows: list[Any],
    *,
    system_id_field: str,
    score_field: str,
    metric: str,
    source_artifact: str,
) -> list[dict[str, Any]]:
    systems: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        system_id = _lookup(row, system_id_field)
        score = _numeric_score(_lookup(row, score_field))
        if system_id is None or score is None:
            continue
        system_id = str(system_id)
        if system_id in seen:
            raise ValueError(f"duplicate system_id in external score artifact: {system_id}")
        seen.add(system_id)
        systems.append(
            {
                "system_id": system_id,
                "score": score,
                "metric": metric,
                "source_artifact": source_artifact,
                "source_row": row_index,
            }
        )
    return systems


def _lookup(row: dict[str, Any], field: str) -> Any:
    current: Any = row
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _numeric_score(value: Any) -> float | None:
    numeric = numeric_or_none(value)
    if numeric is not None:
        return numeric
    if isinstance(value, str):
        try:
            return numeric_or_none(float(value))
        except ValueError:
            return None
    return None


def _normalize_named_path_fields(
    value: Any,
    *,
    field_names: set[str],
    project_root: Path,
    base_dir: Path,
) -> None:
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if key in field_names and isinstance(item, str) and item:
                value[key] = _normalize_project_relative_path_string(item, project_root=project_root, base_dir=base_dir)
            else:
                _normalize_named_path_fields(item, field_names=field_names, project_root=project_root, base_dir=base_dir)
    elif isinstance(value, list):
        for item in value:
            _normalize_named_path_fields(item, field_names=field_names, project_root=project_root, base_dir=base_dir)


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
        project_candidate = project_root / path
        try:
            return _project_relative_path_or_absolute(project_candidate.resolve(), project_root)
        except OSError:
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
