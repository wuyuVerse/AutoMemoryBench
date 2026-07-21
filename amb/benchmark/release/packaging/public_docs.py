"""Generate human-facing documentation for public release packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from amb.benchmark.schemas.io import read_json


BASELINE_KINDS = (
    "no_memory",
    "sliding_window",
    "recency_memory",
    "keyword_memory",
    "bm25_memory",
    "dense_memory",
    "hybrid_memory",
    "entity_memory",
    "graph_memory",
    "rolling_summary",
    "hierarchical_summary",
    "full_history",
    "oracle_retrieval",
    "oracle_memory",
)

AXIS_LABELS = {
    "current_value": "current-value",
    "deletion_state": "deletion-state",
    "authorization_state": "authorization-state",
    "tool_result": "tool-result",
    "role_project_boundary": "role/project-boundary",
}


def generate_public_release_docs(
    package_manifest: dict[str, Any],
    output_dir: str | Path,
    *,
    source_manifest_path: str | Path,
) -> None:
    """Write README and docs/* markdown files for a public release package."""

    output = Path(output_dir)
    docs_dir = output / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = Path(source_manifest_path)
    project_root = _infer_project_root(source_manifest)
    context = _build_context(package_manifest, output, source_manifest, project_root)

    (output / "README.md").write_text(_render_package_readme(context), encoding="utf-8")
    (docs_dir / "README.md").write_text(_render_docs_index(context), encoding="utf-8")
    (docs_dir / "dataset_card.md").write_text(_render_dataset_card(context), encoding="utf-8")
    (docs_dir / "benchmark_card.md").write_text(_render_benchmark_card(context), encoding="utf-8")
    (docs_dir / "schema.md").write_text(_render_schema_guide(context), encoding="utf-8")
    (docs_dir / "evaluation.md").write_text(_render_evaluation_guide(context), encoding="utf-8")
    (docs_dir / "baselines.md").write_text(_render_baselines_guide(context), encoding="utf-8")
    (docs_dir / "submission_guide.md").write_text(_render_submission_guide(context), encoding="utf-8")
    (docs_dir / "memory_system_reporting_template.md").write_text(
        _render_reporting_template(context),
        encoding="utf-8",
    )
    (docs_dir / "annotation_guideline.md").write_text(_render_annotation_guideline(context), encoding="utf-8")
    (docs_dir / "validation.md").write_text(_render_validation_guide(context), encoding="utf-8")
    (docs_dir / "leaderboard_policy.md").write_text(_render_leaderboard_policy(context), encoding="utf-8")
    (docs_dir / "governance_privacy_statement.md").write_text(
        _render_governance_statement(context),
        encoding="utf-8",
    )
    (docs_dir / "reproducibility_checklist.md").write_text(
        _render_reproducibility_checklist(context),
        encoding="utf-8",
    )


def _build_context(
    manifest: dict[str, Any],
    output_dir: Path,
    source_manifest: Path,
    project_root: Path | None,
) -> dict[str, Any]:
    profile_id = str(manifest.get("profile_id", manifest.get("benchmark_id", "release")))
    benchmark_id = str(manifest.get("benchmark_id", profile_id))
    package_name = output_dir.name
    package_display_path = output_dir.as_posix()
    validation_report_path = f"reports/examples/{package_name}_release_validation.json"
    analysis_report_path = f"reports/examples/{package_name}_dev_representative_baselines_analysis.json"
    leaderboard_path = f"reports/examples/{package_name}_dev_leaderboard.json"
    split_reports = manifest.get("split_reports", {})
    expected = manifest.get("expected_generation_summary", {})
    axis_coverage = expected.get("counterfactual_axis_coverage", {})
    covered_axes = [str(axis) for axis in axis_coverage.get("covered_axes", [])]
    recommended_axes = [str(axis) for axis in axis_coverage.get("recommended_axes", [])]
    baseline_reports = _discover_baseline_reports(project_root, package_name)
    validation_report = _load_optional_json(project_root, validation_report_path)

    return {
        "profile_id": profile_id,
        "benchmark_id": benchmark_id,
        "package_name": package_name,
        "package_display_path": package_display_path,
        "source_manifest_path": source_manifest.as_posix(),
        "source_release_dir": source_manifest.parent.as_posix(),
        "public_manifest_path": f"{package_display_path}/manifest.json",
        "validation_report_path": validation_report_path,
        "validation_report": validation_report,
        "analysis_report_path": analysis_report_path,
        "analysis_report_exists": project_root is not None and (project_root / analysis_report_path).exists(),
        "leaderboard_path": leaderboard_path,
        "leaderboard_exists": project_root is not None and (project_root / leaderboard_path).exists(),
        "split_reports": split_reports,
        "expected": expected,
        "axis_coverage": axis_coverage,
        "covered_axes": covered_axes,
        "covered_axis_labels": [_axis_label(axis) for axis in covered_axes],
        "recommended_axes": recommended_axes,
        "domains": list(expected.get("domains", [])),
        "probe_count": expected.get("total_queries_with_counterfactuals"),
        "base_scenarios": expected.get("base_scenarios"),
        "counterfactual_scenarios": expected.get("counterfactual_scenarios"),
        "total_cases": expected.get("num_cases"),
        "total_memories": expected.get("total_memories_with_counterfactuals"),
        "total_events": expected.get("total_events_with_counterfactuals"),
        "variants_per_base": expected.get("counterfactual_variants_per_base"),
        "all_axes": bool(axis_coverage.get("covers_all_recommended_axes")),
        "baseline_reports": baseline_reports,
        "audit_status": str(manifest.get("audit_plan", {}).get("human_audit_status", "unknown")),
    }


def _infer_project_root(start: Path) -> Path | None:
    for parent in (start, *start.parents):
        if (parent / "agent_memory_benchmark").exists() and (parent / "reports").exists():
            return parent
    return None


def _discover_baseline_reports(project_root: Path | None, package_name: str) -> list[dict[str, Any]]:
    if project_root is None:
        return []
    rows: list[dict[str, Any]] = []
    for kind in BASELINE_KINDS:
        relative = Path("reports/examples") / f"{package_name}_dev_{kind}_report.json"
        path = project_root / relative
        if not path.exists():
            continue
        report = read_json(path)
        aggregate = report.get("aggregate", {})
        counterfactual = report.get("counterfactual", {})
        rows.append(
            {
                "kind": kind,
                "path": relative.as_posix(),
                "amq": aggregate.get("lifecycle.amq"),
                "task": aggregate.get("task.task_success"),
                "recall": aggregate.get("retrieval.recall_at_k"),
                "evidence": aggregate.get("retrieval.evidence_complete"),
                "safety": aggregate.get("safety.safety_pass"),
                "memory_dependence_proxy": counterfactual.get("memory_dependence_proxy"),
            }
        )
    return rows


def _load_optional_json(project_root: Path | None, relative_path: str) -> dict[str, Any] | None:
    if project_root is None:
        return None
    path = project_root / relative_path
    if not path.exists():
        return None
    return read_json(path)


def _axis_label(axis: str) -> str:
    return AXIS_LABELS.get(axis, axis.replace("_", "-"))


def _format_metric(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _package_title(context: dict[str, Any]) -> str:
    return f"AutoMemoryBench {context['profile_id']}"


def _render_package_readme(context: dict[str, Any]) -> str:
    split_reports = context["split_reports"]
    covered_axes = ", ".join(context["covered_axis_labels"]) or "none reported"
    hidden = split_reports.get("hidden_test", {})
    return f"""# {_package_title(context)} Public Release Package

Generated from `{context["source_manifest_path"]}`.

This package is the public distribution form. It includes `public_dev`,
`public_test`, and `audit_subset`. It does not include hidden-test trace or
shard artifacts.

## Included Data

- `public_dev`: {split_reports.get("public_dev", {}).get("num_cases", 0)} case variants, {split_reports.get("public_dev", {}).get("num_queries", 0)} queries
- `public_test`: {split_reports.get("public_test", {}).get("num_cases", 0)} case variants, {split_reports.get("public_test", {}).get("num_queries", 0)} queries
- `audit_subset`: {split_reports.get("audit_subset", {}).get("num_cases", 0)} case variants, {split_reports.get("audit_subset", {}).get("num_queries", 0)} queries
- `audit_subset/annotation_templates`: 8 JSONL files, one per domain
- `docs/`: dataset card, benchmark card, submission guide, reporting template,
  annotation guideline, governance statement, and reproducibility checklist

The paper taxonomy uses 11 core probe families; the JSON shards expose 20
concrete `probe_type` values. StrictCore is the headline metric but is derived
from namespaced evaluator components, while `lifecycle.amq` remains diagnostic.

## Counterfactual Coverage

- Variants per base scenario: {context["variants_per_base"]}
- Covered axes: {covered_axes}
- Full recommended axis coverage: {"yes" if context["all_axes"] else "no"}

## Withheld Data

`hidden_test` is declared in `manifest.json` as `private_leaderboard_only` and
`withheld`. Its aggregate split report is retained for reproducibility and
release accounting:

- hidden case variants: {hidden.get("num_cases", 0)}
- hidden queries: {hidden.get("num_queries", 0)}

No `data/hidden_test/**` artifacts are present in this public package.

## Verification

```bash
python -m agent_memory_benchmark validate-release \\
  --manifest {context["public_manifest_path"]} \\
  --output {context["validation_report_path"]}
```

Expected result: `ok=True`, `errors=0`, `warnings=0`.
"""


def _render_docs_index(context: dict[str, Any]) -> str:
    return f"""# {_package_title(context)} Public Release Documentation

This directory contains the human-facing documentation required to use,
evaluate, audit, and cite the {_package_title(context)} public release.

## Documents

- `dataset_card.md`: dataset scope, composition, splits, provenance, limitations.
- `benchmark_card.md`: evaluated capability dimensions, metrics, baselines, interpretation.
- `schema.md`: benchmark, case, memory, query, state-contract, prediction, and report schemas.
- `evaluation.md`: scoring workflow, metrics, CLI commands, and report fields.
- `baselines.md`: deterministic baseline definitions and public-dev sanity table.
- `submission_guide.md`: system interface, prediction format, validation, scoring workflow.
- `memory_system_reporting_template.md`: required metadata for reporting memory systems.
- `annotation_guideline.md`: audit-subset human review instructions.
- `validation.md`: release validation scope and expected checks.
- `leaderboard_policy.md`: public, audit, and hidden-test use policy.
- `governance_privacy_statement.md`: hidden split, sensitive memory, deletion, and privacy policy.
- `reproducibility_checklist.md`: exact commands and artifacts for reproducing the release checks.

## Canonical Artifacts

- Public manifest: `{context["public_manifest_path"]}`
- Public release validation: `{context["validation_report_path"]}`
- Baseline analysis: `{context["analysis_report_path"]}`

## Verification

```bash
python -m agent_memory_benchmark validate-release \\
  --manifest {context["public_manifest_path"]} \\
  --output {context["validation_report_path"]}
```

Expected result: `ok=True`, `errors=0`, `warnings=0`.
"""


def _render_dataset_card(context: dict[str, Any]) -> str:
    split_reports = context["split_reports"]
    validation = context["validation_report"]
    validation_lines = (
        [
            f"- Validation report: `{context['validation_report_path']}`",
            f"- `ok`: {validation.get('ok')}",
            f"- `errors`: {len(validation.get('errors', []))}",
            f"- `warnings`: {len(validation.get('warnings', []))}",
        ]
        if validation is not None
        else [
            f"- Validation report path: `{context['validation_report_path']}`",
            "- Validation status: not bundled in the current workspace export.",
        ]
    )
    return f"""# Dataset Card: {_package_title(context)}

## Dataset Summary

{_package_title(context)} is a public release of the Agent Memory State
Transition Benchmark. It evaluates whether an AI agent can form, update,
retrieve, use, compress, and govern memory across sessions, tasks, time,
tools, and subjects.

The benchmark is event-graph-first: each case is generated from lifecycle
events, compiled into gold memory units and memory state contracts, rendered
as interaction traces, and probed at query time.

## Intended Use

Use this release to evaluate agent memory systems under the public splits:

- `public_dev`: debugging and development.
- `public_test`: public reproducibility and paper reporting.
- `audit_subset`: human audit and data-quality review.

Do not use this package to evaluate hidden-test leaderboard performance.
`hidden_test` is withheld from the public package.

## Dataset Composition

The complete source release contains:

- {len(context["domains"])} domains.
- {context["base_scenarios"]} base scenarios.
- {context["counterfactual_scenarios"]} counterfactual scenario variants.
- {context["total_cases"]} case variants.
- {context["probe_count"]} probes.
- {context["total_memories"]} gold memory units.
- {context["total_events"]} events.

Probe coverage is reported at two levels: 11 paper-level core probe families and
20 concrete JSON `probe_type` values. The 20 concrete values include the core
families plus state-contract stress probes used for conflict resolution,
counterfactual scope, governed transfer, policy exceptions, and temporal-causal
state checks.

The public package includes:

- `public_dev`: {split_reports.get("public_dev", {}).get("num_cases", 0)} case variants and {split_reports.get("public_dev", {}).get("num_queries", 0)} probes.
- `public_test`: {split_reports.get("public_test", {}).get("num_cases", 0)} case variants and {split_reports.get("public_test", {}).get("num_queries", 0)} probes.
- `audit_subset`: {split_reports.get("audit_subset", {}).get("num_cases", 0)} case variants and {split_reports.get("audit_subset", {}).get("num_queries", 0)} probes.

The withheld split retains only aggregate metadata:

- `hidden_test`: {split_reports.get("hidden_test", {}).get("num_cases", 0)} case variants and {split_reports.get("hidden_test", {}).get("num_queries", 0)} probes.

## Domains

The release covers:

{_bullet_lines(context["domains"])}

## Generation Method

The source release was generated by:

```bash
python -m agent_memory_benchmark release \\
  --profile {context["profile_id"]} \\
  --materialize-shards \\
  --output-dir {context["source_release_dir"]}
```

The public package was exported by:

```bash
python -m agent_memory_benchmark export-public-release \\
  --manifest {context["source_manifest_path"]} \\
  --output-dir {context["package_display_path"]}
```

## Counterfactual Coverage

- Variants per base scenario: {context["variants_per_base"]}
- Covered axes: {", ".join(context["covered_axis_labels"]) or "none reported"}
- Full recommended axis coverage: {"yes" if context["all_axes"] else "no"}

## Validation Status

{chr(10).join(validation_lines)}
"""


def _render_benchmark_card(context: dict[str, Any]) -> str:
    baseline_lines = _bullet_lines(BASELINE_KINDS)
    return f"""# Benchmark Card: {_package_title(context)}

## Benchmark Scope

AutoMemoryBench evaluates agent memory state transition capability rather than
static long-context question answering. The primary object of evaluation is
whether the system maintains and uses the correct query-time memory state.

## Probe Types

The paper-level taxonomy has 11 core probe families:

- `write_probe`
- `retrieval_probe`
- `answer_probe`
- `update_probe`
- `compression_probe`
- `forget_probe`
- `governance_probe`
- `tool_probe`
- `planning_probe`
- `evolution_probe`
- `no_memory_probe`

The JSON release uses 20 concrete `probe_type` values: the 11 family labels
above plus state-contract stress subtypes such as
`adversarial_state_synthesis_probe`, `conflict_resolution_probe`,
`cross_session_synthesis_probe`, `governed_transfer_probe`,
`policy_exception_probe`, `policy_temporal_state_probe`, `scope_contrast_probe`,
`state_transition_audit_probe`, and
`temporal_causal_reconciliation_probe`.

## Main Metrics

The primary score is StrictCore. A query passes StrictCore only if task success,
current-state correctness, and safety all pass at query time. This makes stale,
deleted, forbidden, or unauthorized memory use a failure even when retrieval or
answer accuracy appears high. StrictCore is derived for paper/reporting
artifacts from namespaced evaluator components; it is not a native aggregate
column guaranteed in raw evaluator `report.json` files.

Headline dimensions reported by the scorer include:

- `task.task_success`
- `retrieval.recall_at_k`
- `retrieval.evidence_complete`
- `update.temporal_validity`
- `safety.safety_pass`
- `counterfactual.memory_dependence_proxy`
- `efficiency.*`

## Required Baselines

The public-dev release uses deterministic baselines:

{baseline_lines}

## Interpretation

Do not interpret `lifecycle.amq` in isolation. AMQ is diagnostic only; headline
claims must use StrictCore and its component vector. Report every memory-system
score together with:

- `requires_memory` slice performance;
- `no_memory_required` control performance;
- counterfactual dependence metrics;
- governance and safety outcomes.
"""


def _render_schema_guide(context: dict[str, Any]) -> str:
    return f"""# Schema Guide

## Schema Files

Machine-readable schemas are stored under:

```text
amb/benchmark/schemas/json/
  benchmark.schema.json
  prediction.schema.json
  report.schema.json
  release_manifest.schema.json
```

## Case

Each case contains:

- `case_id`
- `domain`
- `sessions`
- `events`
- `event_edges`
- `gold_memory_units`
- `state_contracts`
- `queries`
- `difficulty`

## Query

Each query contains:

- `query_id`
- `prompt`
- `task_type`
- `probe_type`
- `counterfactual_group_id`
- `requires_memory`
- `gold_memory_ids`
- `forbidden_memory_ids`
- `state_contract_id`
- `expected_behavior`

`counterfactual_group_id` may be `null`; when present, it links paired or grouped
queries used by counterfactual state-intervention analyses. `probe_type` is one
of the 20 concrete code-level probe types in the release, which map to 11
paper-level core probe families.

## State Contract

State contracts contain:

- `state_contract_id`
- `timestamp`
- `active_memory_ids`
- `inactive_memory_ids`
- `deleted_memory_ids`
- `forbidden_memory_ids`
- `restricted_memory_ids`
- `superseded_memory_ids`
- `transitions`
"""


def _render_evaluation_guide(context: dict[str, Any]) -> str:
    return f"""# Evaluation Guide

## Evaluation Object

AutoMemoryBench scores a prediction set against query-time memory requirements,
forbidden memory constraints, expected behavior, and state contracts. It is not
only an answer-accuracy benchmark.

## Main CLI

For a single benchmark file:

```bash
python -m agent_memory_benchmark evaluate \\
  --benchmark <benchmark.json> \\
  --predictions <predictions.json> \\
  --output <report.json>
```

For deterministic baselines over a release split:

```bash
python -m agent_memory_benchmark evaluate-release-baseline \\
  --manifest {context["public_manifest_path"]} \\
  --split public_dev \\
  --kind graph_memory \\
  --output reports/examples/{context["package_name"]}_dev_graph_memory_report.json
```

## Headline Metrics

The primary headline metric is `strict_core_success`: a query passes StrictCore
only when the answer is task-correct, uses the current required memory state,
and does not use stale, deleted, forbidden, or out-of-scope memory. This is a
derived paper/reporting metric. Native evaluator reports expose namespaced
component fields, so analysis code must derive StrictCore from those components
on the `requires_memory` slice rather than reading `strict_core_success`
directly from a raw evaluator `report.json`.

Report StrictCore with the component vector below; do not collapse memory
ability to retrieval alone:

- `task.task_success`
- `retrieval.recall_at_k`
- `retrieval.evidence_complete`
- `update.temporal_validity`
- `safety.safety_pass`
- `counterfactual.memory_dependence_proxy`

`lifecycle.amq` is retained as a diagnostic composite for error analysis and
ablation only. It must not be used alone to claim memory ability.

## Probe Types

The paper describes 11 core probe families: write, retrieval, answer, update,
compression, forget, governance, tool, planning, evolution, and no-memory. The
released JSON shards materialize these as 20 concrete `probe_type` values:

- `adversarial_state_synthesis_probe`
- `answer_probe`
- `compression_probe`
- `conflict_resolution_probe`
- `cross_session_synthesis_probe`
- `evolution_probe`
- `forget_probe`
- `governance_probe`
- `governed_transfer_probe`
- `no_memory_probe`
- `planning_probe`
- `policy_exception_probe`
- `policy_temporal_state_probe`
- `retrieval_probe`
- `scope_contrast_probe`
- `state_transition_audit_probe`
- `temporal_causal_reconciliation_probe`
- `tool_probe`
- `update_probe`
- `write_probe`

## Analysis

Use the analyzer for bootstrap intervals, paired comparisons, and
quality-cost frontier analysis:

```bash
python -m agent_memory_benchmark analyze \\
  --reports <report_a.json> <report_b.json> \\
  --output <analysis.json>
```
"""


def _render_baselines_guide(context: dict[str, Any]) -> str:
    baseline_rows = context["baseline_reports"]
    table = ""
    if baseline_rows:
        lines = [
            "| Baseline | AMQ | Task | Recall | Evidence | Safety | MD Proxy |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in baseline_rows:
            lines.append(
                "| "
                f"{row['kind']} | {_format_metric(row['amq'])} | {_format_metric(row['task'])} | "
                f"{_format_metric(row['recall'])} | {_format_metric(row['evidence'])} | "
                f"{_format_metric(row['safety'])} | {_format_metric(row['memory_dependence_proxy'])} |"
            )
        table = "\n".join(lines)
    else:
        table = (
            "Representative public-dev baseline reports are not yet bundled in the "
            "current workspace export. Use the commands below to generate them."
        )

    return f"""# Baselines

## Baseline List

The current release includes deterministic baselines:

{_bullet_lines(BASELINE_KINDS)}

## Public-Dev Results

{table}

## Run Command

```bash
for kind in bm25_memory dense_memory hybrid_memory entity_memory graph_memory \\
            recency_memory rolling_summary hierarchical_summary sliding_window \\
            oracle_retrieval oracle_memory no_memory full_history keyword_memory; do
  python -m agent_memory_benchmark evaluate-release-baseline \\
    --manifest {context["public_manifest_path"]} \\
    --split public_dev \\
    --kind "$kind" \\
    --output "reports/examples/{context["package_name"]}_dev_${{kind}}_report.json" \\
    --quiet
done
```

## Representative Sanity Command

```bash
python -m agent_memory_benchmark build-release-representative-baselines \\
  --manifest {context["public_manifest_path"]} \\
  --split public_dev \\
  --output-dir reports/examples \\
  --bootstrap-samples 200 \\
  --seed 13
```

## Analyze Command

```bash
python -m agent_memory_benchmark analyze \\
  --reports reports/examples/{context["package_name"]}_dev_no_memory_report.json \\
            reports/examples/{context["package_name"]}_dev_graph_memory_report.json \\
            reports/examples/{context["package_name"]}_dev_full_history_report.json \\
            reports/examples/{context["package_name"]}_dev_oracle_memory_report.json \\
  --output {context["analysis_report_path"]} \\
  --bootstrap-samples 200
```
"""


def _render_submission_guide(context: dict[str, Any]) -> str:
    return f"""# System Submission Guide

## Prediction Format

Systems must output an AMST prediction JSON object:

```json
{{
  "schema_version": "1.0.0",
  "system_id": "my_memory_agent",
  "predictions": [
    {{
      "query_id": "case_x:q_answer",
      "memory_needed": true,
      "activated_memory_ids": ["m_e_x"],
      "response": "answer text",
      "tool_name": null,
      "parameters": {{}},
      "memory_operations": [],
      "cost": {{
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0,
        "retrieval_latency_ms": 0,
        "storage_bytes": 0
      }}
    }}
  ]
}}
```

## Validation

Validate the public release package:

```bash
python -m agent_memory_benchmark validate-release \\
  --manifest {context["public_manifest_path"]}
```

## Scoring

```bash
python -m agent_memory_benchmark evaluate-release-predictions \\
  --manifest {context["public_manifest_path"]} \\
  --split public_test \\
  --predictions <submission.json> \\
  --output <report.json>
```

## Hidden Test Policy

`hidden_test` is withheld from the public package. Public users should not
expect to score `hidden_test` locally. Hidden evaluation belongs to a private
leaderboard environment.
"""


def _render_reporting_template(context: dict[str, Any]) -> str:
    return """# Memory System Reporting Template

## System Identity

- System name:
- Version or commit:
- Organization:
- Contact:
- Date:

## Model and Runtime

- Base model:
- Model provider:
- Context window:
- Decoding settings:
- Hardware:
- Runtime environment:

## Memory Architecture

- Memory type: none / full-history / summary / RAG / graph / hybrid / learned / other.
- Write policy:
- Retrieval policy:
- Update policy:
- Forget/delete policy:
- Compression policy:
- Governance and authorization policy:
- Cross-user or cross-project isolation policy:

## Tools and Actions

- Tool interface:
- Tool parameter grounding method:
- Action validation:
- Error handling:

## Metrics To Report

- `strict_core_success` (derived from component fields on the `requires_memory` slice)
- `task.task_success`
- `retrieval.recall_at_k`
- `retrieval.evidence_complete`
- `update.temporal_validity`
- `safety.safety_pass`
- `counterfactual.memory_dependence_proxy`
- `lifecycle.amq` (diagnostic only, not a standalone headline score)
- Cost fields: input tokens, output tokens, latency, retrieval latency, storage bytes.
"""


def _render_annotation_guideline(context: dict[str, Any]) -> str:
    return f"""# Annotation Guideline for the Audit Subset

## Purpose

The audit subset is used to verify that AutoMemoryBench cases are answerable,
evidence-grounded, governance-compliant, and naturally rendered.

## Unit of Annotation

Annotate one query record at a time from:

```text
{context["package_display_path"]}/data/audit_subset/annotation_templates/{{domain}}.jsonl
```

## Required Checks

Annotators should fill the boolean checks listed in each record's
`applicable_checks` field.

Core checks:

- `evidence_sufficient`
- `answer_unique`
- `governance_boundary_clear`
- `trace_natural`
- `scenario_memory_required`

Counterfactual-only check:

- `counterfactual_target_state_only`

## Double Annotation

Each audit item should be labeled independently by two annotators. Disagreements
should be adjudicated after agreement metrics are computed.

## Known Current Status

The public package includes annotation templates. Completed human double
annotations are not yet included in this workspace release.
"""


def _render_validation_guide(context: dict[str, Any]) -> str:
    validation = context["validation_report"]
    if validation is not None:
        status_text = (
            "Expected output:\n\n```text\n"
            f"ok={validation.get('ok')}\n"
            f"errors={len(validation.get('errors', []))}\n"
            f"warnings={len(validation.get('warnings', []))}\n"
            "```"
        )
    else:
        status_text = (
            "Expected output will be written to the selected report path after "
            "running the validation command."
        )
    return f"""# Validation Guide

## Release Validation Command

```bash
python -m agent_memory_benchmark validate-release \\
  --manifest {context["public_manifest_path"]} \\
  --output {context["validation_report_path"]}
```

{status_text}

## What Is Checked

Release validation checks:

- Manifest shape and required fields.
- Referenced shard paths.
- Artifact `sha256` and `size_bytes`.
- Benchmark loading and semantic validation.
- Duplicate case ids.
- Counterfactual group preservation.
- Split report consistency.
- Release plan consistency.
- Expected generation summary totals.
- Audit template alignment with audit-subset queries.
- Public-package hidden split withholding.

## Withheld Hidden Split

In the public package, `hidden_test` is expected to have no shard artifacts.
The manifest must instead declare it as withheld and `private_leaderboard_only`.
"""


def _render_leaderboard_policy(context: dict[str, Any]) -> str:
    return """# Leaderboard Policy

## Split Roles

- `public_dev`: development, debugging, and ablation.
- `public_test`: public reproducibility and paper reporting.
- `audit_subset`: human data-quality audit.
- `hidden_test`: private leaderboard only.

## Public Package Boundary

The public package does not include hidden-test trace, shard, memory, query, or
state-contract artifacts. It only includes hidden aggregate statistics for
release accounting.

## Hidden Evaluation

Hidden-test evaluation should be executed only by a private evaluator with
access to non-public artifacts. The public package verifier will refuse to
evaluate `hidden_test` because the split is withheld.
"""


def _render_governance_statement(context: dict[str, Any]) -> str:
    return """# Governance and Privacy Statement

## Release Boundary

This is a public release package. It includes `public_dev`, `public_test`, and
`audit_subset`. It does not include hidden-test trace artifacts.

## Sensitive Memory

The benchmark intentionally includes synthetic sensitive-memory situations to
evaluate privacy and governance behavior. Sensitive values are generated
benchmark artifacts, not real user secrets.

## Deletion and Forgetting

Some cases include deletion requests, retention confirmations, stale memories,
and forbidden memories. Systems are expected to stop using deleted memory and
avoid activating forbidden memory.

## Hidden Test Use

Hidden-test data must not be redistributed or inspected by submitted systems.
Hidden leaderboard results should only be generated by an evaluator with access
to the private hidden package.
"""


def _render_reproducibility_checklist(context: dict[str, Any]) -> str:
    return f"""# Reproducibility Checklist

## Release Artifacts

- Public manifest exists: `{context["public_manifest_path"]}`
- Public docs exist: `{context["package_display_path"]}/docs/`
- Public package excludes `data/hidden_test/**`
- Audit templates exist for all 8 domains
- Public release validation path: `{context["validation_report_path"]}`

## Regenerate Source Release

```bash
python -m agent_memory_benchmark release \\
  --profile {context["profile_id"]} \\
  --materialize-shards \\
  --output-dir {context["source_release_dir"]}
```

## Export Public Package

```bash
python -m agent_memory_benchmark export-public-release \\
  --manifest {context["source_manifest_path"]} \\
  --output-dir {context["package_display_path"]}
```

## Validate Public Package

```bash
python -m agent_memory_benchmark validate-release \\
  --manifest {context["public_manifest_path"]} \\
  --output {context["validation_report_path"]}
```

## Run Tests

```bash
python -m pytest tests/test_release.py tests/test_release_docs.py -q
```

## Analyze Baselines

```bash
python -m agent_memory_benchmark build-release-representative-baselines \\
  --manifest {context["public_manifest_path"]} \\
  --split public_dev \\
  --output-dir reports/examples \\
  --bootstrap-samples 200
```
"""


def _bullet_lines(items: list[str] | tuple[str, ...]) -> str:
    return "\n".join(f"- `{item}`" for item in items)
