"""Portable path/root contract helpers for generated artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def localize_report_contract(
    report: dict[str, Any],
    *,
    output_path: str | Path,
    project_root_hints: Iterable[str | Path | None] = (),
) -> dict[str, Any]:
    """Attach a root contract and localize absolute path strings when possible."""

    normalized = json.loads(json.dumps(report))
    resolved_output = Path(output_path).resolve()
    project_root = _infer_project_root(resolved_output, project_root_hints)
    if project_root is None:
        return normalized
    normalized["root"] = _artifact_root_ref(resolved_output.parent, project_root)
    _normalize_absolute_paths(normalized, project_root)
    return normalized


def _infer_project_root(output_path: Path, hints: Iterable[str | Path | None]) -> Path | None:
    named_candidates: list[Path] = []
    anchors: list[Path] = [_contract_anchor(output_path)]
    for raw_hint in hints:
        if raw_hint is None:
            continue
        hint = Path(raw_hint).resolve()
        anchors.append(_contract_anchor(hint))
        for anchor_name in ("data", "reports"):
            candidate = _project_root_from_named_ancestor(hint, anchor_name)
            if candidate is not None:
                named_candidates.append(candidate.resolve())
    if named_candidates:
        return Path(os.path.commonpath([str(path) for path in named_candidates]))
    if anchors:
        return Path(os.path.commonpath([str(path) for path in anchors]))
    return None


def _contract_anchor(path: Path) -> Path:
    return path if path.exists() and path.is_dir() else path.parent


def _project_root_from_named_ancestor(path: Path, anchor: str) -> Path | None:
    current = path if path.is_dir() else path.parent
    for parent in (current, *current.parents):
        if parent.name == anchor:
            return parent.parent
    return None


def _artifact_root_ref(base_dir: Path, project_root: Path) -> str:
    try:
        return Path(os.path.relpath(project_root.resolve(), base_dir.resolve())).as_posix()
    except ValueError:
        return str(project_root.resolve())


def _normalize_absolute_paths(value: Any, project_root: Path) -> Any:
    if isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _normalize_absolute_paths(item, project_root)
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _normalize_absolute_paths(item, project_root)
        return value
    if isinstance(value, str):
        return _project_relative_or_original(value, project_root)
    return value


def _project_relative_or_original(raw_value: str, project_root: Path) -> str:
    path = Path(raw_value)
    if not path.is_absolute():
        return raw_value
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return raw_value
