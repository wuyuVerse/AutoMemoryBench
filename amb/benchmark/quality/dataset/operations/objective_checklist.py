"""Prompt-to-artifact checklist for the AutoMemoryBench implementation objective."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from amb.benchmark.quality.completion_audit import _artifact_root_ref
from amb.benchmark.quality.completion_audit import build_completion_audit
from amb.benchmark.schemas.io import read_json
from amb.benchmark.schemas.io import write_json

OBJECTIVE_CHECKLIST_SCHEMA_VERSION = "amst-objective-checklist-v1"


@dataclass(frozen=True)
class ChecklistSpec:
    item_id: str
    source: str
    requirement: str
    evidence_paths: tuple[str, ...]
    verification_commands: tuple[str, ...] = ()
    completion_check_id: str | None = None
    notes: str | None = None


OBJECTIVE_RESTATEMENT = (
    "Implement the AutoMemoryBench final plan as a reproducible agent-memory benchmark: "
    "event-graph-first main dataset construction, auditable release artifacts, "
    "quality gates, runnable baselines/evaluation, tests, and explicit evidence "
    "for the remaining human/external-system validation gates."
)

SUCCESS_CRITERIA: tuple[str, ...] = (
    "The final design document exists and is traceable to implementation artifacts.",
    "Schema, domain packs, compiler, renderer, probe factory, validator, runner, scorer, analyzer, and leaderboard are implemented in separate layers.",
    "The main dataset exists as sharded releases with public and hidden splits plus validation reports, using main-v1-strict as the canonical final main release and main-v1 as a lightweight compatibility release.",
    "Quality gates, sanity baselines, release validation, and public documentation are present and reproducible by CLI.",
    "No completion claim is made until human audit, external benchmark correlation, and real memory-system runs contain real evidence.",
)


CHECKLIST_SPECS: tuple[ChecklistSpec, ...] = (
    ChecklistSpec(
        "objective_source",
        "user objective + docs/design/AutoMemoryBench_final_design.md",
        "Use the final AutoMemoryBench design as the implementation target.",
        ("docs/design/AutoMemoryBench_final_design.md",),
    ),
    ChecklistSpec(
        "architecture_and_code_structure",
        "user objective: keep code structure reasonable, decoupled, layered",
        "Maintain explicit module boundaries for schema, generation, quality, evaluation, release, integration, analysis, and interfaces.",
        ("amb/benchmark/ARCHITECTURE.md", "tests/test_architecture.py"),
        ("python -m pytest tests/test_architecture.py -q",),
    ),
    ChecklistSpec(
        "schema_standard_layer",
        "final plan 3.1 and 10.1",
        "Provide versioned schemas for benchmark, prediction, release manifest, and report artifacts.",
        (
            "amb/benchmark/schemas/models.py",
            "amb/benchmark/schemas/state.py",
            "amb/benchmark/schemas/json/benchmark.schema.json",
            "amb/benchmark/schemas/json/prediction.schema.json",
            "amb/benchmark/schemas/json/release_manifest.schema.json",
            "amb/benchmark/schemas/json/report.schema.json",
        ),
        ("python -m agent_memory_benchmark validate --benchmark data/samples/mini_benchmark.json",),
        "schema",
    ),
    ChecklistSpec(
        "domain_packs",
        "final plan 3.2 and 10.5",
        "Provide eight application-domain packs for the main benchmark.",
        ("amb/benchmark/generation/domains/specs.py", "data/domain_packs"),
        ("python -m agent_memory_benchmark domain-packs --output-dir data/domain_packs",),
        "domain_packs",
    ),
    ChecklistSpec(
        "compiler_validator",
        "final plan 3.3, 3.6, and 10.2",
        "Compile event graphs into memory units and state contracts, then validate traceability and quality gates.",
        (
            "amb/benchmark/generation/compilers",
            "amb/benchmark/quality/validation.py",
            "amb/benchmark/quality/gates.py",
            "amb/benchmark/release/public_test_sanity.py",
            "tests/test_generation.py",
            "tests/test_quality.py",
            "tests/test_public_test_sanity.py",
        ),
        ("python -m pytest tests/test_generation.py tests/test_quality.py tests/test_public_test_sanity.py -q",),
        "compiler_validator",
    ),
    ChecklistSpec(
        "renderer_probe_factory",
        "final plan 3.4, 3.5, and 10.3",
        "Render multi-source traces and generate probes with required/forbidden memories and scoring rules.",
        (
            "amb/benchmark/generation/renderers",
            "amb/benchmark/generation/probes/factory.py",
            "tests/test_renderers.py",
            "tests/test_generation.py",
        ),
        ("python -m pytest tests/test_renderers.py tests/test_generation.py -q",),
        "renderer_probe_factory",
    ),
    ChecklistSpec(
        "runner_scorer_analyzer",
        "final plan 3.7, 3.8, 10.4, and 21",
        "Run agents/baselines, compute lifecycle metrics, produce diagnostics, statistics, and leaderboard artifacts.",
        (
            "amb/benchmark/evaluation",
            "amb/benchmark/metrics",
            "amb/benchmark/analysis",
            "amb/benchmark/leaderboard",
            "tests/test_scoring.py",
            "tests/test_analyzer.py",
        ),
        ("python -m pytest tests/test_scoring.py tests/test_analyzer.py -q",),
        "runner_scorer_analyzer",
    ),
    ChecklistSpec(
        "main_dataset_release",
        "final plan 5.1 and 10.5",
        "Build a sharded main dataset release with public-dev, public-test, hidden-test, and audit-subset splits; treat main-v1-strict as the canonical final main dataset and keep main-v1 as a lightweight compatibility release.",
        (
            "amb/benchmark/generation/profiles.py",
            "amb/benchmark/release/sharded.py",
            "data/releases/amst_main_v1/manifest.json",
            "data/releases/amst_main_v1_public/manifest.json",
            "reports/examples/amst_main_v1_release_validation.json",
            "reports/examples/amst_main_v1_public_release_validation.json",
            "reports/examples/amst_main_v1_acceptance.json",
            "reports/examples/amst_main_v1_public_test_required_slices.json",
            "reports/examples/amst_main_v1_hidden_test_sanity.json",
            "reports/examples/amst_main_v1_question_craftsmanship_audit.json",
            "reports/examples/amst_main_v1_query_construction_audit.json",
            "reports/examples/amst_main_v1_probe_discriminativeness_audit.json",
            "reports/examples/amst_main_v1_difficulty_calibration_audit.json",
            "reports/examples/amst_main_v1_domain_construct_validity_audit.json",
            "data/releases/amst_main_v1_strict/manifest.json",
            "data/releases/amst_main_v1_strict_public/manifest.json",
            "reports/examples/amst_main_v1_strict_release_validation.json",
            "reports/examples/amst_main_v1_strict_public_release_validation.json",
            "reports/examples/amst_main_v1_strict_acceptance.json",
            "reports/examples/amst_main_v1_strict_lineage_audit.json",
            "reports/examples/amst_main_v1_strict_public_test_required_slices.json",
            "reports/examples/amst_main_v1_strict_hidden_test_sanity.json",
            "reports/examples/amst_main_v1_strict_foundation_validation_audit.json",
            "reports/examples/amst_main_v1_strict_question_craftsmanship_audit.json",
            "reports/examples/amst_main_v1_strict_query_construction_audit.json",
            "reports/examples/amst_main_v1_strict_probe_discriminativeness_audit.json",
            "reports/examples/amst_main_v1_strict_difficulty_calibration_audit.json",
            "reports/examples/amst_main_v1_strict_domain_construct_validity_audit.json",
            "amb/benchmark/quality/main_dataset_acceptance.py",
            "amb/benchmark/quality/lineage.py",
            "amb/benchmark/quality/foundation_validation.py",
            "amb/benchmark/quality/query_construction.py",
            "amb/benchmark/quality/question_craftsmanship.py",
            "amb/benchmark/quality/probe_discriminativeness.py",
            "amb/benchmark/quality/difficulty_calibration.py",
            "amb/benchmark/quality/domain_construct_validity.py",
        ),
        (
            "python -m agent_memory_benchmark release --profile main-v1-strict --materialize-shards --output-dir data/releases/amst_main_v1_strict",
            "python -m agent_memory_benchmark export-public-release --manifest data/releases/amst_main_v1_strict/manifest.json --output-dir data/releases/amst_main_v1_strict_public",
            "python -m agent_memory_benchmark foundation-validation-audit --reports reports/siliconflow/deepseek_v32_main_v1_strict_public_dev_sample_g1_d2_query_only_report.json reports/siliconflow/deepseek_v32_main_v1_strict_public_dev_sample_g1_d2_full_history_report.json reports/siliconflow/deepseek_v32_main_v1_strict_public_dev_sample_g1_d2_oracle_state_report.json --expected-benchmark-id amst-main-v1-strict-public_dev-sample-g1-s13 --cohort-id deepseek_v32_main_v1_strict_public_dev_sample_g1_d2 --require-full-history --output reports/examples/amst_main_v1_strict_foundation_validation_audit.json",
            "python -m agent_memory_benchmark question-craftsmanship-audit --manifest data/releases/amst_main_v1_strict_public/manifest.json --output reports/examples/amst_main_v1_strict_question_craftsmanship_audit.json",
            "python -m agent_memory_benchmark query-construction-audit --manifest data/releases/amst_main_v1_strict_public/manifest.json --output reports/examples/amst_main_v1_strict_query_construction_audit.json",
            "python -m agent_memory_benchmark probe-discriminativeness-audit --manifest data/releases/amst_main_v1_strict_public/manifest.json --split public_dev --reports-dir reports/examples --output reports/examples/amst_main_v1_strict_probe_discriminativeness_audit.json",
            "python -m agent_memory_benchmark difficulty-calibration-audit --manifest data/releases/amst_main_v1_strict_public/manifest.json --split public_dev --reports-dir reports/examples --output reports/examples/amst_main_v1_strict_difficulty_calibration_audit.json",
            "python -m agent_memory_benchmark domain-construct-validity-audit --manifest data/releases/amst_main_v1_strict_public/manifest.json --split public_dev --reports-dir reports/examples --output reports/examples/amst_main_v1_strict_domain_construct_validity_audit.json",
            "python -m agent_memory_benchmark main-dataset-acceptance --manifest data/releases/amst_main_v1_strict/manifest.json --expected-benchmark-id amst-main-v1-strict --expected-profile-id main-v1-strict --run-release-validation --release-validation-report reports/examples/amst_main_v1_strict_release_validation.json --public-release-manifest data/releases/amst_main_v1_strict_public/manifest.json --public-release-validation-report reports/examples/amst_main_v1_strict_public_release_validation.json --require-public-release-validation --representative-reports-dir reports/examples --representative-split public_dev --require-representative-baselines --require-all-counterfactual-axes --public-test-sanity-report reports/examples/amst_main_v1_strict_public_test_sanity.json --require-public-test-sanity --public-result-slices-report reports/examples/amst_main_v1_strict_public_test_required_slices.json --require-public-result-slices --hidden-test-sanity-report reports/examples/amst_main_v1_strict_hidden_test_sanity.json --require-hidden-test-sanity --question-craftsmanship-report reports/examples/amst_main_v1_strict_question_craftsmanship_audit.json --require-question-craftsmanship --query-construction-report reports/examples/amst_main_v1_strict_query_construction_audit.json --require-query-construction --probe-discriminativeness-report reports/examples/amst_main_v1_strict_probe_discriminativeness_audit.json --require-probe-discriminativeness --difficulty-calibration-report reports/examples/amst_main_v1_strict_difficulty_calibration_audit.json --require-difficulty-calibration --domain-construct-validity-report reports/examples/amst_main_v1_strict_domain_construct_validity_audit.json --require-domain-construct-validity --foundation-validation-report reports/examples/amst_main_v1_strict_foundation_validation_audit.json --require-foundation-validation --output reports/examples/amst_main_v1_strict_acceptance.json",
            "python -m agent_memory_benchmark lineage-audit --manifest data/releases/amst_main_v1_strict/manifest.json --output reports/examples/amst_main_v1_strict_lineage_audit.json",
            "python -m agent_memory_benchmark validate-release --manifest data/releases/amst_main_v1_strict_public/manifest.json",
            "python -m agent_memory_benchmark release --profile main-v1 --dry-run-summary",
            "python -m agent_memory_benchmark question-craftsmanship-audit --manifest data/releases/amst_main_v1_public/manifest.json --output reports/examples/amst_main_v1_question_craftsmanship_audit.json",
            "python -m agent_memory_benchmark query-construction-audit --manifest data/releases/amst_main_v1_public/manifest.json --output reports/examples/amst_main_v1_query_construction_audit.json",
            "python -m agent_memory_benchmark probe-discriminativeness-audit --manifest data/releases/amst_main_v1_public/manifest.json --split public_dev --reports-dir reports/examples --output reports/examples/amst_main_v1_probe_discriminativeness_audit.json",
            "python -m agent_memory_benchmark difficulty-calibration-audit --manifest data/releases/amst_main_v1_public/manifest.json --split public_dev --reports-dir reports/examples --output reports/examples/amst_main_v1_difficulty_calibration_audit.json",
            "python -m agent_memory_benchmark domain-construct-validity-audit --manifest data/releases/amst_main_v1_public/manifest.json --split public_dev --reports-dir reports/examples --output reports/examples/amst_main_v1_domain_construct_validity_audit.json",
            "python -m agent_memory_benchmark main-dataset-acceptance --manifest data/releases/amst_main_v1/manifest.json --expected-benchmark-id amst-main-v1 --expected-profile-id main-v1 --run-release-validation --release-validation-report reports/examples/amst_main_v1_release_validation.json --public-release-manifest data/releases/amst_main_v1_public/manifest.json --public-release-validation-report reports/examples/amst_main_v1_public_release_validation.json --require-public-release-validation --representative-reports-dir reports/examples --representative-split public_dev --require-representative-baselines --question-craftsmanship-report reports/examples/amst_main_v1_question_craftsmanship_audit.json --require-question-craftsmanship --query-construction-report reports/examples/amst_main_v1_query_construction_audit.json --require-query-construction --probe-discriminativeness-report reports/examples/amst_main_v1_probe_discriminativeness_audit.json --require-probe-discriminativeness --difficulty-calibration-report reports/examples/amst_main_v1_difficulty_calibration_audit.json --require-difficulty-calibration --domain-construct-validity-report reports/examples/amst_main_v1_domain_construct_validity_audit.json --require-domain-construct-validity --output reports/examples/amst_main_v1_acceptance.json",
            "python -m agent_memory_benchmark lineage-audit --manifest data/releases/amst_main_v1/manifest.json --output reports/examples/amst_main_v1_lineage_audit.json",
            "python -m agent_memory_benchmark validate-release --manifest data/releases/amst_main_v1_public/manifest.json",
        ),
        "main_release",
    ),
    ChecklistSpec(
        "challenge_release",
        "final plan 5.2, 5.3, and 10.5",
        "Build challenge/hidden/counterfactual release artifacts without leaking hidden-test data into public packages.",
        (
            "data/releases/amst_challenge_v1/manifest.json",
            "data/releases/amst_challenge_v1_public/manifest.json",
            "reports/examples/amst_challenge_v1_release_validation.json",
            "reports/examples/amst_challenge_v1_public_release_validation.json",
            "reports/examples/amst_challenge_v1_acceptance.json",
            "reports/examples/amst_challenge_v1_question_craftsmanship_audit.json",
            "reports/examples/amst_challenge_v1_query_construction_audit.json",
            "reports/examples/amst_challenge_v1_probe_discriminativeness_audit.json",
            "reports/examples/amst_challenge_v1_difficulty_calibration_audit.json",
            "reports/examples/amst_challenge_v1_domain_construct_validity_audit.json",
        ),
        (
            "python -m agent_memory_benchmark validate-release --manifest data/releases/amst_challenge_v1_public/manifest.json",
            "python -m agent_memory_benchmark question-craftsmanship-audit --manifest data/releases/amst_challenge_v1_public/manifest.json --output reports/examples/amst_challenge_v1_question_craftsmanship_audit.json",
            "python -m agent_memory_benchmark query-construction-audit --manifest data/releases/amst_challenge_v1/manifest.json --output reports/examples/amst_challenge_v1_query_construction_audit.json",
            "python -m agent_memory_benchmark build-release-representative-baselines --manifest data/releases/amst_challenge_v1_public/manifest.json --split public_dev --output-dir reports/examples",
            "python -m agent_memory_benchmark probe-discriminativeness-audit --manifest data/releases/amst_challenge_v1_public/manifest.json --split public_dev --reports-dir reports/examples --output reports/examples/amst_challenge_v1_probe_discriminativeness_audit.json",
            "python -m agent_memory_benchmark difficulty-calibration-audit --manifest data/releases/amst_challenge_v1_public/manifest.json --split public_dev --reports-dir reports/examples --output reports/examples/amst_challenge_v1_difficulty_calibration_audit.json",
            "python -m agent_memory_benchmark domain-construct-validity-audit --manifest data/releases/amst_challenge_v1_public/manifest.json --split public_dev --reports-dir reports/examples --output reports/examples/amst_challenge_v1_domain_construct_validity_audit.json",
            "python -m agent_memory_benchmark challenge-release-acceptance --manifest data/releases/amst_challenge_v1/manifest.json --public-release-manifest data/releases/amst_challenge_v1_public/manifest.json --release-validation-report reports/examples/amst_challenge_v1_release_validation.json --public-release-validation-report reports/examples/amst_challenge_v1_public_release_validation.json --representative-reports-dir reports/examples --question-craftsmanship-report reports/examples/amst_challenge_v1_question_craftsmanship_audit.json --query-construction-report reports/examples/amst_challenge_v1_query_construction_audit.json --probe-discriminativeness-report reports/examples/amst_challenge_v1_probe_discriminativeness_audit.json --difficulty-calibration-report reports/examples/amst_challenge_v1_difficulty_calibration_audit.json --domain-construct-validity-report reports/examples/amst_challenge_v1_domain_construct_validity_audit.json --output reports/examples/amst_challenge_v1_acceptance.json",
        ),
        "challenge_release",
    ),
    ChecklistSpec(
        "quarterly_hidden_refresh",
        "final plan 5.3 and 10.5",
        "Build a quarterly private hidden refresh package with hidden-only quality validation and no public leakage.",
        (
            "data/releases/amst_hidden_quarterly_v1/manifest.json",
            "reports/examples/amst_hidden_quarterly_v1_release_validation.json",
            "reports/examples/amst_hidden_quarterly_v1_release_intrinsic_sanity.json",
            "reports/examples/amst_hidden_quarterly_v1_acceptance.json",
            "reports/examples/amst_hidden_quarterly_v1_question_craftsmanship_audit.json",
            "reports/examples/amst_hidden_quarterly_v1_query_construction_audit.json",
        ),
        (
            "python -m agent_memory_benchmark build-private-leaderboard-package --output-dir data/releases/amst_hidden_quarterly_v1 --source-profile main-v1-strict --refresh-id 2026Q2 --num-hidden-scenarios 300",
            "python -m agent_memory_benchmark validate-release --manifest data/releases/amst_hidden_quarterly_v1/manifest.json --output reports/examples/amst_hidden_quarterly_v1_release_validation.json",
            "python -m agent_memory_benchmark release-intrinsic-sanity --manifest data/releases/amst_hidden_quarterly_v1/manifest.json --output reports/examples/amst_hidden_quarterly_v1_release_intrinsic_sanity.json",
            "python -m agent_memory_benchmark question-craftsmanship-audit --manifest data/releases/amst_hidden_quarterly_v1/manifest.json --output reports/examples/amst_hidden_quarterly_v1_question_craftsmanship_audit.json",
            "python -m agent_memory_benchmark query-construction-audit --manifest data/releases/amst_hidden_quarterly_v1/manifest.json --output reports/examples/amst_hidden_quarterly_v1_query_construction_audit.json",
            "python -m agent_memory_benchmark hidden-quarterly-acceptance --manifest data/releases/amst_hidden_quarterly_v1/manifest.json --release-validation-report reports/examples/amst_hidden_quarterly_v1_release_validation.json --intrinsic-sanity-report reports/examples/amst_hidden_quarterly_v1_release_intrinsic_sanity.json --question-craftsmanship-report reports/examples/amst_hidden_quarterly_v1_question_craftsmanship_audit.json --query-construction-report reports/examples/amst_hidden_quarterly_v1_query_construction_audit.json --output reports/examples/amst_hidden_quarterly_v1_acceptance.json",
        ),
        "quarterly_hidden_refresh",
    ),
    ChecklistSpec(
        "effectiveness_and_sanity_reports",
        "final plan 10.5, 11, 21, and 23",
        "Provide baseline effectiveness evidence, including oracle/no-memory/full-history sanity checks and leaderboard reports.",
        (
            "reports/examples/amst_main_v1_strict_public_dev_oracle_memory_report.json",
            "reports/examples/amst_main_v1_strict_public_dev_no_memory_report.json",
            "reports/examples/amst_main_v1_strict_public_dev_full_history_report.json",
            "reports/examples/amst_main_v1_strict_public_dev_dense_memory_report.json",
            "reports/examples/amst_main_v1_strict_public_dev_hybrid_memory_report.json",
            "reports/examples/amst_main_v1_strict_public_dev_graph_memory_report.json",
            "reports/examples/amst_main_v1_strict_public_dev_representative_baselines_analysis.json",
            "reports/examples/amst_main_v1_strict_public_dev_leaderboard.json",
            "reports/examples/amst_main_v1_strict_public_dev_failure_mode_diagnostics.json",
            "reports/examples/amst_main_v1_strict_public_test_sanity.json",
            "reports/examples/amst_main_v1_strict_public_test_failure_mode_diagnostics.json",
        ),
        (
            "python -m agent_memory_benchmark analyze --reports reports/examples/amst_main_v1_strict_public_dev_no_memory_report.json reports/examples/amst_main_v1_strict_public_dev_graph_memory_report.json reports/examples/amst_main_v1_strict_public_dev_full_history_report.json reports/examples/amst_main_v1_strict_public_dev_oracle_memory_report.json --output reports/examples/amst_main_v1_strict_public_dev_representative_baselines_analysis.json --bootstrap-samples 200",
        ),
        "effectiveness_reports",
    ),
    ChecklistSpec(
        "public_release_docs",
        "final plan 10.6 and 13",
        "Publish dataset card, benchmark card, evaluation guide, validation guide, and reproducibility checklist.",
        (
            "data/releases/amst_main_v1_public/docs/dataset_card.md",
            "data/releases/amst_main_v1_public/docs/benchmark_card.md",
            "data/releases/amst_main_v1_public/docs/evaluation.md",
            "data/releases/amst_main_v1_public/docs/validation.md",
            "data/releases/amst_main_v1_public/docs/reproducibility_checklist.md",
            "data/releases/amst_main_v1_strict_public/docs/dataset_card.md",
            "data/releases/amst_main_v1_strict_public/docs/benchmark_card.md",
            "data/releases/amst_main_v1_strict_public/docs/evaluation.md",
            "data/releases/amst_main_v1_strict_public/docs/validation.md",
            "data/releases/amst_main_v1_strict_public/docs/reproducibility_checklist.md",
        ),
        (),
        "public_docs",
    ),
    ChecklistSpec(
        "test_suite",
        "user objective: do testing and verify whether it is effective",
        "Maintain project tests that cover generation, validation, release, scoring, analysis, integrations, and CLI flows.",
        ("tests",),
        ("python -m pytest tests -q",),
        None,
        "This checklist verifies that the test suite exists; the command output must still be inspected separately.",
    ),
    ChecklistSpec(
        "human_audit_evidence",
        "final plan 10.5 and 21.2",
        "Complete double annotation and agreement metrics for the audit subset using raw annotation evidence.",
        (
            "amb/benchmark/quality/annotation.py",
            "amb/benchmark/quality/human_audit.py",
            "data/releases/amst_main_v1_strict_public/manifest.json",
            "reports/human_audit_bundle/current/bundle_manifest.json",
            "reports/human_audit_bundle/current/handoff_manifest.json",
            "reports/human_audit_bundle/current/bundle_verification.json",
            "reports/human_audit_bundle/current/return_inbox_sync.json",
            "reports/human_audit_bundle/current/return_inbox_watch.json",
        ),
        (
            "PYTHONPATH=. python -m agent_memory_benchmark build-human-audit-evidence-bundle --manifest data/releases/amst_main_v1_strict_public/manifest.json --task-manifest reports/examples/human_audit_tasks/main_v1_strict_public/task_manifest.json --output-dir reports/human_audit_bundle/current",
            "PYTHONPATH=. python -m agent_memory_benchmark verify-human-audit-evidence-bundle --bundle-dir reports/human_audit_bundle/current --output reports/human_audit_bundle/current/bundle_verification.json",
            "PYTHONPATH=. python -m agent_memory_benchmark watch-human-audit-return-inbox --bundle-dir reports/human_audit_bundle/current --interval-s 120 --max-iterations 0 --stop-when-ready --stop-when-rejected --output reports/human_audit_bundle/current/return_inbox_watch.json",
            "PYTHONPATH=. python -m agent_memory_benchmark sync-human-audit-return-inbox --bundle-dir reports/human_audit_bundle/current --reconcile-when-ready --signed-at 2026-05-13T00:00:00Z --annotation-guideline reports/human_audit_bundle/current/docs/annotation_guideline.md --adjudication-policy 'Disagreements are adjudicated after double annotation.' --output reports/human_audit_bundle/current/return_inbox_sync.json",
            "python -m agent_memory_benchmark prepare-human-audit --manifest data/releases/amst_main_v1_strict_public/manifest.json --annotator-id ann_a --annotator-id ann_b --output-dir HUMAN_AUDIT_TASKS",
            "python -m agent_memory_benchmark merge-human-audit-annotations --task-manifest HUMAN_AUDIT_TASKS/task_manifest.json --output COMPLETED.jsonl",
            "python -m agent_memory_benchmark generate-human-audit-attestation --task-manifest HUMAN_AUDIT_TASKS/task_manifest.json --annotations COMPLETED.jsonl --output ATTESTATION.json --signed-at 2026-05-13T00:00:00Z --annotation-guideline data/releases/amst_main_v1_strict_public/docs/annotation_guideline.md --adjudication-policy 'Disagreements are adjudicated after double annotation.'",
            "python -m agent_memory_benchmark finalize-human-audit --manifest data/releases/amst_main_v1_strict_public/manifest.json --annotations COMPLETED.jsonl --task-manifest HUMAN_AUDIT_TASKS/task_manifest.json --annotator-attestation ATTESTATION.json --agreement-output agreement.json",
        ),
        "human_audit",
    ),
    ChecklistSpec(
        "external_benchmark_correlation",
        "final plan 20.4, 21.4, 23, and 25.4",
        "Run the same systems on AutoMemoryBench and external memory benchmarks, then validate rank-correlation evidence.",
        (
            "amb/benchmark/analysis/external_scores.py",
            "amb/benchmark/analysis/external_correlation.py",
            "amb/benchmark/analysis/external_protocol.py",
            "reports/external/locomo_correlation.json",
            "reports/external/longmemeval_correlation.json",
            "reports/external/mem2actbench_correlation.json",
            "reports/external/memoryagentbench_correlation.json",
            "reports/external/evidence_validation.json",
            "reports/external/evidence_gap_report.json",
            "reports/external/cohort_expansion_plan.json",
            "reports/external/cohort_expansion_validation.json",
            "reports/external/handoff_manifest.json",
            "reports/external/return_inbox_sync.json",
            "reports/external/return_inbox_watch.json",
        ),
        (
            "python -m agent_memory_benchmark normalize-external-scores --input RAW --benchmark-id BENCH --output reports/external/BENCH_scores.json",
            "python -m agent_memory_benchmark external-correlation-batch --amst-reports AMST_REPORTS --external-scores reports/external/*_scores.json --output-dir reports/external",
            "python -m agent_memory_benchmark summarize-external-evidence-gaps --correlations reports/external/*_correlation.json --real-system-validation reports/real_system_runs/canonical_public_dev_refresh_current_matrix_validation.json --output reports/external/evidence_gap_report.json --min-shared-systems 3 --min-control-shared-systems 1 --min-real-memory-shared-systems 1",
            "python -m agent_memory_benchmark build-external-cohort-expansion-plan --correlations reports/external/*_correlation.json --real-system-validation reports/real_system_runs/canonical_public_dev_refresh_current_matrix_validation.json --output reports/external/cohort_expansion_plan.json --min-shared-systems 3 --min-control-shared-systems 1 --min-real-memory-shared-systems 1",
            "PYTHONPATH=. python -m agent_memory_benchmark verify-external-cohort-expansion-plan --expansion reports/external/cohort_expansion_plan.json --output reports/external/cohort_expansion_validation.json",
            "PYTHONPATH=. python -m agent_memory_benchmark watch-external-cohort-return-inbox --expansion reports/external/cohort_expansion_plan.json --interval-s 120 --max-iterations 0 --stop-when-ready --stop-when-rejected --output reports/external/return_inbox_watch.json",
            "PYTHONPATH=. python -m agent_memory_benchmark sync-external-cohort-return-inbox --expansion reports/external/cohort_expansion_plan.json --output reports/external/return_inbox_sync.json --real-system-validation reports/real_system_runs/canonical_public_dev_refresh_current_matrix_validation.json",
            "PYTHONPATH=. python -m agent_memory_benchmark refresh-external-canonical --root . --output-dir reports/external --real-system-validation reports/real_system_runs/canonical_public_dev_refresh_current_matrix_validation.json --min-shared-systems 3 --min-control-shared-systems 1 --min-real-memory-shared-systems 1",
            "python -m agent_memory_benchmark validate-external-evidence --correlations reports/external/*_correlation.json --output reports/external/evidence_validation.json --min-shared-systems 3 --min-control-shared-systems 1 --min-real-memory-shared-systems 1",
        ),
        "external_benchmark_correlation",
    ),
    ChecklistSpec(
        "real_memory_system_integrations",
        "final plan 21.3, 21.7, and 25.5",
        "Run representative real memory systems on AMST release splits with real-system attestation, cost, and latency evidence.",
        (
            "amb/benchmark/integrations",
            "amb/benchmark/quality/real_system.py",
            "configs/real_system/canonical_public_dev_refresh.json",
            "reports/real_system_runs/canonical_public_dev_refresh_config_validation.json",
            "reports/real_system_runs/canonical_public_dev_refresh_current_analysis.json",
            "reports/real_system_runs/canonical_public_dev_refresh_current_matrix_summary.json",
            "reports/real_system_runs/canonical_public_dev_refresh_current_matrix_validation.json",
        ),
        (
            "python -m agent_memory_benchmark validate-integration-configs --configs configs/real_system/mem0_siliconflow_real.json configs/real_system/letta_memgpt_local_real.json configs/real_system/langmem_real.json configs/real_system/zep_graphiti_graphiti_core_local.json --output reports/real_system_runs/canonical_public_dev_refresh_config_validation.json",
            "python -m agent_memory_benchmark refresh-real-system-canonical --spec configs/real_system/canonical_public_dev_refresh.json",
            "python -m agent_memory_benchmark watch-real-system-canonical --spec configs/real_system/canonical_public_dev_refresh.json --root . --interval-s 120 --max-iterations 0",
            "python -m agent_memory_benchmark validate-real-system-matrix --summary reports/real_system_runs/canonical_public_dev_refresh_current_matrix_summary.json --output reports/real_system_runs/canonical_public_dev_refresh_current_matrix_validation.json --expected-benchmark-id amst-main-v1-public_dev --expected-release-split public_dev",
        ),
        "real_memory_system_integrations",
    ),
    ChecklistSpec(
        "git_management",
        "user objective: keep good git management",
        "Keep repository metadata and ignore rules in place, and ensure critical source/config/test assets are tracked or intentionally ignored.",
        (".git", ".gitignore"),
        (
            "git status --short",
            "git ls-files amb agent_memory_benchmark amst_real_clients configs tests",
            "git log --oneline -5",
        ),
        "git_hygiene",
    ),
)


def build_objective_checklist(root: str | Path = ".") -> dict[str, Any]:
    project = Path(root)
    completion = build_completion_audit(project)
    completion_checks = {check.check_id: check for check in completion.checks}
    items = [_build_item(project, spec, completion_checks) for spec in CHECKLIST_SPECS]
    summary = {"passed": 0, "partial": 0, "missing": 0}
    for item in items:
        summary[item["status"]] = summary.get(item["status"], 0) + 1
    completed_item_ids = [item["item_id"] for item in items if item["status"] == "passed"]
    pending_item_ids = [item["item_id"] for item in items if item["status"] != "passed"]
    partial_item_ids = [item["item_id"] for item in items if item["status"] == "partial"]
    missing_item_ids = [item["item_id"] for item in items if item["status"] == "missing"]
    blockers = [asdict(check) for check in completion.checks if check.status != "passed"]
    return {
        "schema_version": OBJECTIVE_CHECKLIST_SCHEMA_VERSION,
        "root": str(project.resolve()),
        "objective_restatement": OBJECTIVE_RESTATEMENT,
        "success_criteria": list(SUCCESS_CRITERIA),
        "status": "complete" if all(item["status"] == "passed" for item in items) else "incomplete",
        "summary": summary,
        "completed_item_ids": completed_item_ids,
        "pending_item_ids": pending_item_ids,
        "partial_item_ids": partial_item_ids,
        "missing_item_ids": missing_item_ids,
        "items": items,
        "completion_audit": {
            "status": completion.status,
            "summary": completion.summary,
            "blocker_check_ids": [check["check_id"] for check in blockers],
            "blockers": blockers,
        },
    }


def write_objective_checklist(root: str | Path, output: str | Path) -> dict[str, Any]:
    project = Path(root)
    report = build_objective_checklist(root)
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = project / output_path
    report["root"] = _artifact_root_ref(output_path.parent, project)
    write_json(output, report)
    return report


def _build_item(project: Path, spec: ChecklistSpec, completion_checks: dict[str, Any]) -> dict[str, Any]:
    evidence_paths = spec.evidence_paths
    if spec.item_id == "effectiveness_and_sanity_reports":
        evidence_paths = _resolve_current_evidence_paths(project, spec.evidence_paths)
    elif spec.item_id in {"main_dataset_release", "challenge_release", "quarterly_hidden_refresh"}:
        evidence_paths = _resolve_current_evidence_paths(project, spec.evidence_paths)
    check = completion_checks.get(spec.completion_check_id or "")
    existing_paths = tuple(path for path in evidence_paths if (project / path).exists())
    missing_paths = tuple(path for path in evidence_paths if not (project / path).exists())
    if check is not None:
        status = check.status
        evidence = tuple(dict.fromkeys((*existing_paths, *check.evidence)))
        missing = (
            ()
            if status == "passed"
            else tuple(dict.fromkeys((*missing_paths, *check.missing)))
        )
        notes = check.notes or spec.notes
    elif missing_paths:
        status = "missing"
        evidence = existing_paths
        missing = missing_paths
        notes = spec.notes
    else:
        status = "passed"
        evidence = existing_paths
        missing = ()
        notes = spec.notes
    verification_commands = spec.verification_commands
    if spec.item_id == "human_audit_evidence":
        verification_commands = _build_human_audit_verification_commands(project, spec.verification_commands)
    elif spec.item_id == "external_benchmark_correlation":
        verification_commands = _build_external_verification_commands(project, spec.verification_commands)
    elif spec.item_id == "effectiveness_and_sanity_reports":
        verification_commands = _build_effectiveness_verification_commands(
            project,
            spec.verification_commands,
            spec.evidence_paths,
        )
    elif spec.item_id in {"main_dataset_release", "challenge_release", "quarterly_hidden_refresh"}:
        verification_commands = _build_main_release_verification_commands(
            project,
            spec.verification_commands,
            spec.evidence_paths,
        )
    return {
        "item_id": spec.item_id,
        "source": spec.source,
        "requirement": spec.requirement,
        "status": status,
        "evidence": list(evidence),
        "missing": list(missing),
        "verification_commands": list(verification_commands),
        "notes": notes,
    }


def _prefer_current_example_artifact(path: Path) -> Path:
    if path.suffix != ".json" or path.name.endswith("_current.json"):
        return path
    current_path = path.with_name(f"{path.stem}_current{path.suffix}")
    return current_path if current_path.exists() else path


def _artifact_ref(path: Path, project: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), project.resolve())).as_posix()
    except ValueError:
        return str(path)


def _resolve_current_evidence_paths(project: Path, evidence_paths: tuple[str, ...]) -> tuple[str, ...]:
    resolved: list[str] = []
    for relative_path in evidence_paths:
        path = project / relative_path
        resolved.append(_artifact_ref(_prefer_current_example_artifact(path), project))
    return tuple(dict.fromkeys(resolved))


def _build_main_release_verification_commands(
    project: Path,
    fallback: tuple[str, ...],
    evidence_paths: tuple[str, ...],
) -> tuple[str, ...]:
    replacements: dict[str, str] = {}
    for relative_path in evidence_paths:
        path = project / relative_path
        preferred = _prefer_current_example_artifact(path)
        if preferred != path:
            replacements[relative_path] = _artifact_ref(preferred, project)

    commands: list[str] = []
    for command in fallback:
        updated = command
        for source, target in replacements.items():
            updated = updated.replace(source, target)
        updated = re.sub(
            r"reports/examples/[A-Za-z0-9_./-]+\.json",
            lambda match: _artifact_ref(
                _prefer_current_example_artifact(project / match.group(0)),
                project,
            ),
            updated,
        )
        commands.append(updated)
    return tuple(dict.fromkeys(commands))


def _build_effectiveness_verification_commands(
    project: Path,
    fallback: tuple[str, ...],
    evidence_paths: tuple[str, ...],
) -> tuple[str, ...]:
    replacements: dict[str, str] = {}
    for relative_path in evidence_paths:
        path = project / relative_path
        preferred = _prefer_current_example_artifact(path)
        if preferred != path:
            replacements[relative_path] = _artifact_ref(preferred, project)

    commands: list[str] = []
    for command in fallback:
        updated = command
        for source, target in replacements.items():
            updated = updated.replace(source, target)
        commands.append(updated)
    return tuple(dict.fromkeys(commands))


def _build_external_verification_commands(project: Path, fallback: tuple[str, ...]) -> tuple[str, ...]:
    handoff_path = project / "reports/external/handoff_manifest.json"
    expansion_path = project / "reports/external/cohort_expansion_plan.json"
    contract: dict[str, Any] | None = None
    for path in (handoff_path, expansion_path):
        if path.exists():
            contract = read_json(path)
            break
    if not contract:
        return fallback

    commands: list[str] = []
    recommended_next = contract.get("recommended_next_command")
    if isinstance(recommended_next, str) and recommended_next:
        commands.append(recommended_next)

    operator_commands = contract.get("operator_commands", {})
    if isinstance(operator_commands, dict):
        for key in (
            "build_cohort_expansion_plan",
            "watch_return_inbox",
            "sync_return_inbox",
            "review_rejected_returns",
            "refresh_canonical",
        ):
            command = operator_commands.get(key)
            if isinstance(command, str) and command:
                commands.append(command)

    for command in fallback:
        if any(token in command for token in ("RAW", "BENCH", "AMST_REPORTS")):
            continue
        commands.append(command)

    return tuple(dict.fromkeys(commands))


def _build_human_audit_verification_commands(project: Path, fallback: tuple[str, ...]) -> tuple[str, ...]:
    contract: dict[str, Any] | None = None
    for path in (
        project / "reports/human_audit_bundle/current/handoff_manifest.json",
        project / "reports/human_audit_bundle/current/progress.json",
        project / "reports/human_audit_bundle/current/bundle_manifest.json",
    ):
        if path.exists():
            contract = read_json(path)
            break
    if not contract:
        return fallback

    commands: list[str] = []
    recommended_script = contract.get("recommended_next_script_file")
    if isinstance(recommended_script, str) and recommended_script:
        commands.append(recommended_script)
    recommended_command = contract.get("recommended_next_command")
    if isinstance(recommended_command, str) and recommended_command:
        commands.append(recommended_command)

    operator_commands = contract.get("operator_commands", {})
    if isinstance(operator_commands, dict):
        for key in (
            "verify_bundle",
            "watch_return_inbox",
            "sync_return_inbox",
            "review_rejected_returns",
            "reconcile_when_ready",
            "progress",
        ):
            command = operator_commands.get(key)
            if isinstance(command, str) and command:
                commands.append(command)

    for command in fallback:
        if any(token in command for token in ("HUMAN_AUDIT_TASKS", "COMPLETED.jsonl", "ATTESTATION.json")):
            continue
        commands.append(command)

    return tuple(dict.fromkeys(commands))
