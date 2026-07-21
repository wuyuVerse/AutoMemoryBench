"""Release package export helpers."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
import shutil
from typing import Any

from amb.benchmark.release.artifacts import artifact_info
from amb.benchmark.release.public_docs import generate_public_release_docs
from amb.benchmark.release.splits import RELEASE_SPLITS
from amb.benchmark.schemas.io import read_json, write_json


PUBLIC_RELEASE_SPLITS = ("public_dev", "public_test", "audit_subset")
WITHHELD_PUBLIC_SPLITS = ("hidden_test",)


def export_public_release_package(
    source_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Export a public release package without hidden-test trace artifacts."""

    source_manifest = Path(source_manifest_path)
    source_root = source_manifest.parent
    output = Path(output_dir)
    source = read_json(source_manifest)
    package = deepcopy(source)

    package["build_timestamp"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    package["package_type"] = "public_release_export"
    package["included_splits"] = list(PUBLIC_RELEASE_SPLITS)
    source_manifest_artifact = artifact_info(source_manifest)
    package["exported_from_manifest"] = {
        "benchmark_id": source.get("benchmark_id"),
        "path": str(source_manifest),
        "sha256": source_manifest_artifact["sha256"],
        "size_bytes": source_manifest_artifact["size_bytes"],
        "build_timestamp": source.get("build_timestamp"),
    }
    package["builder"] = {
        "name": "amb.benchmark.public_release_exporter",
        "schema_version": str(source.get("schema_version", "1.0.0")),
    }
    package["build_metadata"] = {
        **dict(source.get("build_metadata", {})),
        "package_visibility": "public",
        "source_release_manifest": str(source_manifest),
        "hidden_test_artifacts_withheld": True,
    }
    package["source_release_manifest_artifact"] = _relative_artifact_info(source_manifest, source_manifest)

    package["split_files"] = {split: {} for split in RELEASE_SPLITS}
    package["split_artifacts"] = {split: {} for split in RELEASE_SPLITS}
    package["group_assignments"] = {
        split: deepcopy(source.get("group_assignments", {}).get(split, {}))
        for split in PUBLIC_RELEASE_SPLITS
    }
    package["group_assignments"].update({split: {} for split in WITHHELD_PUBLIC_SPLITS})

    for split in PUBLIC_RELEASE_SPLITS:
        files = source.get("split_files", {}).get(split, {})
        package["split_files"][split], package["split_artifacts"][split] = _copy_split_files(
            split,
            files,
            source_root=source_root,
            output=output,
        )

    package["withheld_splits"] = {
        split: {
            "reason": "private_leaderboard_only",
            "visibility": source.get("visibility", {}).get(split),
            "artifact_status": "withheld",
            "report_source": f"source_manifest.split_reports.{split}",
            "split_report": deepcopy(source.get("split_reports", {}).get(split, {})),
            "release_plan": deepcopy(source.get("release_plan", {}).get("split_reports", {}).get(split, {})),
            "num_group_assignments": _count_group_assignments(source.get("group_assignments", {}).get(split, {})),
            "group_counts_by_domain": _group_assignment_counts_by_domain(
                source.get("group_assignments", {}).get(split, {})
            ),
        }
        for split in WITHHELD_PUBLIC_SPLITS
    }

    package["audit_plan"] = deepcopy(source.get("audit_plan", {}))
    package["audit_plan"]["audit_template_file"] = None
    package["audit_plan"]["audit_template_files"] = {}
    package["audit_template_artifacts"] = {}
    template_files = source.get("audit_plan", {}).get("audit_template_files", {})
    if isinstance(template_files, dict):
        for label, raw_path in sorted(template_files.items()):
            source_path = _resolve_source_path(str(raw_path), source_root)
            rel_path = Path("data") / "audit_subset" / "annotation_templates" / f"{label}.jsonl"
            target = output / rel_path
            _copy_file(source_path, target)
            package["audit_plan"]["audit_template_files"][label] = rel_path.as_posix()
            package["audit_template_artifacts"][label] = _relative_artifact_info(target, rel_path)
    _scrub_private_human_audit_evidence(package)

    manifest_path = output / "manifest.json"
    write_json(manifest_path, package)
    generate_public_release_docs(package, output, source_manifest_path=source_manifest)
    package["manifest_path"] = str(manifest_path)
    return package


def _copy_split_files(
    split: str,
    files: Any,
    *,
    source_root: Path,
    output: Path,
) -> tuple[dict[str, str] | str, dict[str, Any]]:
    if isinstance(files, str):
        source_path = _resolve_source_path(files, source_root)
        rel_path = Path("data") / split / "benchmark.json"
        target = output / rel_path
        _copy_file(source_path, target)
        return rel_path.as_posix(), _relative_artifact_info(target, rel_path)
    if isinstance(files, dict):
        copied_files: dict[str, str] = {}
        copied_artifacts: dict[str, Any] = {}
        for label, raw_path in sorted(files.items()):
            source_path = _resolve_source_path(str(raw_path), source_root)
            rel_path = Path("data") / split / "shards" / f"{label}.json"
            target = output / rel_path
            _copy_file(source_path, target)
            copied_files[str(label)] = rel_path.as_posix()
            copied_artifacts[str(label)] = _relative_artifact_info(target, rel_path)
        return copied_files, copied_artifacts
    return {}, {}


def _resolve_source_path(value: str, source_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return source_root / path


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _relative_artifact_info(path: Path, relative_path: Path) -> dict[str, Any]:
    info = artifact_info(path)
    info["path"] = relative_path.as_posix()
    return info


def _scrub_private_human_audit_evidence(package: dict[str, Any]) -> None:
    audit_plan = package.get("audit_plan")
    if not isinstance(audit_plan, dict):
        return
    if audit_plan.get("human_audit_status") != "completed":
        return
    audit_plan["human_audit_status"] = "completed_private_source"
    for key in (
        "audit_annotations_file",
        "audit_task_manifest_file",
        "annotator_attestation_file",
        "agreement_metrics_file",
    ):
        audit_plan[key] = None


def _count_group_assignments(value: Any) -> int:
    if isinstance(value, dict):
        return sum(len(groups) for groups in value.values())
    if isinstance(value, list):
        return len(value)
    return 0


def _group_assignment_counts_by_domain(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        return {str(domain): len(groups) for domain, groups in sorted(value.items())}
    return {}
