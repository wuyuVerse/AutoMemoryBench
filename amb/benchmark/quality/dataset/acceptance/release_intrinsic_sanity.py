"""Release-level intrinsic sanity checks derived from shard audit summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from amb.benchmark.quality.artifact_contract import localize_report_contract
from amb.benchmark.release.splits import RELEASE_SPLITS
from amb.benchmark.schemas.io import read_json, write_json

RELEASE_INTRINSIC_SANITY_SCHEMA_VERSION = "amst-release-intrinsic-sanity-v1"
INTRINSIC_GATES = ("oracle_solvability", "no_memory_unsolvability")


def validate_release_intrinsic_sanity(manifest_path: str | Path) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    split_reports = manifest.get("split_reports", {})
    domain_reports = manifest.get("domain_reports", {})

    split_gate_summary: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    warnings: list[str] = []

    for split in RELEASE_SPLITS:
        report = split_reports.get(split, {})
        if not isinstance(report, dict):
            errors.append(f"split_reports.{split} is missing")
            continue
        data_quality_gates = report.get("data_quality_gates", {})
        gate_values = {gate: bool(data_quality_gates.get(gate)) for gate in INTRINSIC_GATES}
        split_gate_summary[split] = {
            "present": True,
            "num_cases": int(report.get("num_cases", 0)),
            "num_queries": int(report.get("num_queries", 0)),
            "gates": gate_values,
            "passed": all(gate_values.values()),
        }
        if split_gate_summary[split]["num_cases"] == 0:
            warnings.append(f"{split} has no visible cases for intrinsic sanity validation")
        elif not split_gate_summary[split]["passed"]:
            errors.append(
                f"{split} intrinsic sanity failed: "
                + ", ".join(gate for gate, passed in gate_values.items() if not passed)
            )

    per_domain_summary: dict[str, dict[str, Any]] = {}
    for domain, report in sorted(domain_reports.items()):
        if not isinstance(report, dict):
            continue
        split_domain = {}
        for split in RELEASE_SPLITS:
            split_report = report.get("split_reports", {}).get(split, {})
            if not isinstance(split_report, dict) or int(split_report.get("num_cases", 0)) == 0:
                continue
            data_quality_gates = split_report.get("data_quality_gates", {})
            split_domain[split] = {
                "num_cases": int(split_report.get("num_cases", 0)),
                "gates": {gate: bool(data_quality_gates.get(gate)) for gate in INTRINSIC_GATES},
            }
        if split_domain:
            per_domain_summary[str(domain)] = split_domain

    checks = []
    for split, summary in sorted(split_gate_summary.items()):
        if summary["num_cases"] == 0:
            continue
        for gate, passed in summary["gates"].items():
            checks.append(
                {
                    "check_id": f"{split}.{gate}",
                    "status": "passed" if passed else "failed",
                    "detail": {
                        "num_cases": summary["num_cases"],
                        "num_queries": summary["num_queries"],
                    },
                }
            )

    return {
        "schema_version": RELEASE_INTRINSIC_SANITY_SCHEMA_VERSION,
        "manifest_path": str(manifest_file),
        "benchmark_id": manifest.get("benchmark_id"),
        "status": "passed" if not errors else "failed",
        "summary": {
            "num_splits_checked": sum(1 for item in split_gate_summary.values() if item["num_cases"] > 0),
            "num_domains_checked": len(per_domain_summary),
            "failed_checks": sum(1 for check in checks if check["status"] == "failed"),
            "warnings": len(warnings),
        },
        "checks": checks,
        "split_intrinsic_sanity": split_gate_summary,
        "domain_intrinsic_sanity": per_domain_summary,
        "warnings": warnings,
        "errors": errors,
    }


def write_release_intrinsic_sanity(
    manifest_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    report = validate_release_intrinsic_sanity(manifest_path)
    report = localize_report_contract(
        report,
        output_path=output,
        project_root_hints=(manifest_path,),
    )
    write_json(output, report)
    return report
