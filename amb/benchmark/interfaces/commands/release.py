"""Release CLI commands."""

from __future__ import annotations

import argparse

from amb.benchmark.evaluation.scoring import DEFAULT_RETRIEVAL_K
from amb.benchmark.evaluation.report import print_validation
from amb.benchmark.generation import expected_generation_summary, generation_profile, profile_names
from amb.benchmark.interfaces.commands.common import print_release_plan_summary
from amb.benchmark.quality.validation import validate_benchmark
from amb.benchmark.release import (
    MAIN_RELEASE_PROFILES,
    ReleaseConfig,
    build_hidden_test_sanity_artifact,
    build_main_release_workflow,
    build_quarterly_hidden_refresh_package,
    build_public_test_baseline_artifacts,
    build_public_test_sanity_summary,
    build_profile_release_shards,
    build_release_failure_mode_diagnostics,
    build_representative_baseline_artifacts,
    build_release,
    export_public_release_package,
    planned_release_summary,
    sample_release_split,
)
from amb.benchmark.schemas.io import load_benchmark


def register_release_commands(subparsers: argparse._SubParsersAction) -> None:
    release = subparsers.add_parser("release", help="Build group-preserving release splits")
    release.add_argument("--benchmark")
    release.add_argument("--profile", choices=profile_names(), help="Plan release splits from a named generation profile.")
    release.add_argument("--dry-run-summary", action="store_true", help="Print planned release split summary without writing split files.")
    release.add_argument("--materialize-shards", action="store_true", help="Generate split/domain shards directly from a profile.")
    release.add_argument("--output-dir")
    release.add_argument("--seed", type=int, default=13)
    release.add_argument("--dev-fraction", type=float, default=0.20)
    release.add_argument("--audit-fraction", type=float, default=0.10)
    release.add_argument("--hidden-fraction", type=float, default=0.20)
    release.set_defaults(handler=cmd_release)

    export_public = subparsers.add_parser("export-public-release", help="Export a public release package without hidden-test artifacts")
    export_public.add_argument("--manifest", required=True)
    export_public.add_argument("--output-dir", required=True)
    export_public.set_defaults(handler=cmd_export_public_release)

    private_hidden = subparsers.add_parser(
        "build-private-leaderboard-package",
        help="Build a quarterly private hidden refresh package with 300 hidden scenarios",
    )
    private_hidden.add_argument("--output-dir", required=True)
    private_hidden.add_argument("--source-profile", default="main-v1-strict", choices=profile_names())
    private_hidden.add_argument("--refresh-id", default="2026Q2")
    private_hidden.add_argument("--seed", type=int, default=13)
    private_hidden.add_argument("--num-hidden-scenarios", type=int, default=300)
    private_hidden.set_defaults(handler=cmd_build_private_leaderboard_package)

    sample_release = subparsers.add_parser("sample-release-split", help="Sample a small benchmark from a release split while preserving counterfactual groups")
    sample_release.add_argument("--manifest", required=True)
    sample_release.add_argument("--split", required=True, choices=("public_dev", "public_test", "audit_subset", "hidden_test"))
    sample_release.add_argument("--groups-per-domain", type=int, default=2)
    sample_release.add_argument("--domains", nargs="+")
    sample_release.add_argument("--seed", type=int, default=13)
    sample_release.add_argument("--output", required=True)
    sample_release.set_defaults(handler=cmd_sample_release_split)

    representative = subparsers.add_parser(
        "build-release-representative-baselines",
        help="Build representative baseline reports, analysis, and leaderboard for a public release split",
    )
    representative.add_argument("--manifest", required=True)
    representative.add_argument("--split", default="public_dev", choices=("public_dev", "public_test", "audit_subset"))
    representative.add_argument("--output-dir", default="reports/examples")
    representative.add_argument("--retrieval-k", type=int, default=DEFAULT_RETRIEVAL_K)
    representative.add_argument("--bootstrap-samples", type=int, default=200)
    representative.add_argument("--seed", type=int, default=13)
    representative.set_defaults(handler=cmd_build_release_representative_baselines)

    public_test_baselines = subparsers.add_parser(
        "build-public-test-baselines",
        help="Build canonical public-test baseline reports for release-level sanity and reporting",
    )
    public_test_baselines.add_argument("--manifest", required=True)
    public_test_baselines.add_argument("--split", default="public_test", choices=("public_test",))
    public_test_baselines.add_argument("--output-dir", default="reports/examples")
    public_test_baselines.add_argument("--retrieval-k", type=int, default=DEFAULT_RETRIEVAL_K)
    public_test_baselines.set_defaults(handler=cmd_build_public_test_baselines)

    public_test_sanity = subparsers.add_parser(
        "build-public-test-sanity",
        help="Build machine-readable public-test sanity verdict from canonical baseline summaries and failure-mode diagnostics",
    )
    public_test_sanity.add_argument("--baselines", required=True)
    public_test_sanity.add_argument("--failure-modes", required=True)
    public_test_sanity.add_argument("--output", required=True)
    public_test_sanity.set_defaults(handler=cmd_build_public_test_sanity)

    hidden_test_sanity = subparsers.add_parser(
        "build-hidden-test-sanity",
        help="Build a private hidden-test sanity artifact from a private release manifest",
    )
    hidden_test_sanity.add_argument("--manifest", required=True)
    hidden_test_sanity.add_argument("--output-dir", default="reports/examples")
    hidden_test_sanity.add_argument("--retrieval-k", type=int, default=DEFAULT_RETRIEVAL_K)
    hidden_test_sanity.set_defaults(handler=cmd_build_hidden_test_sanity)

    failure_modes = subparsers.add_parser(
        "build-release-failure-modes",
        help="Build release-level failure-mode diagnostics for RAG/full-history/graph/oracle baselines",
    )
    failure_modes.add_argument("--manifest", required=True)
    failure_modes.add_argument("--split", default="public_dev", choices=("public_dev", "public_test", "audit_subset"))
    failure_modes.add_argument("--output-dir", default="reports/examples")
    failure_modes.add_argument("--retrieval-k", type=int, default=DEFAULT_RETRIEVAL_K)
    failure_modes.set_defaults(handler=cmd_build_release_failure_modes)

    workflow = subparsers.add_parser(
        "build-main-release-workflow",
        help="Build the full main-release workflow: sharded release, public package, validation, acceptance, lineage, and representative baselines",
    )
    workflow.add_argument("--profile", required=True, choices=MAIN_RELEASE_PROFILES)
    workflow.add_argument("--output-dir", required=True)
    workflow.add_argument("--existing-release-manifest")
    workflow.add_argument("--public-output-dir")
    workflow.add_argument("--reports-dir", default="reports/examples")
    workflow.add_argument("--seed", type=int, default=13)
    workflow.add_argument("--dev-fraction", type=float, default=0.20)
    workflow.add_argument("--audit-fraction", type=float, default=0.10)
    workflow.add_argument("--hidden-fraction", type=float, default=0.20)
    workflow.add_argument("--retrieval-k", type=int, default=DEFAULT_RETRIEVAL_K)
    workflow.add_argument("--representative-split", default="public_dev", choices=("public_dev", "public_test", "audit_subset"))
    workflow.add_argument("--bootstrap-samples", type=int, default=200)
    workflow.add_argument("--skip-representative-baselines", action="store_true")
    workflow.add_argument("--require-completed-human-audit", action="store_true")
    workflow.add_argument("--foundation-report", action="append", dest="foundation_reports")
    workflow.add_argument("--foundation-expected-benchmark-id")
    workflow.add_argument("--foundation-cohort-id")
    workflow.add_argument("--foundation-require-full-history", action="store_true")
    workflow.add_argument("--require-foundation-validation", action="store_true")
    workflow.set_defaults(handler=cmd_build_main_release_workflow)


def cmd_release(args: argparse.Namespace) -> None:
    cfg = ReleaseConfig(
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        audit_fraction=args.audit_fraction,
        hidden_fraction=args.hidden_fraction,
    )
    if args.dry_run_summary:
        if not args.profile:
            raise SystemExit("--profile is required with release --dry-run-summary")
        generation_summary = expected_generation_summary(generation_profile(args.profile).config)
        release_summary = planned_release_summary(generation_summary, cfg)
        print_release_plan_summary(release_summary)
        return
    if args.materialize_shards:
        if not args.profile or not args.output_dir:
            raise SystemExit("--profile and --output-dir are required with --materialize-shards")
        manifest = build_profile_release_shards(args.profile, args.output_dir, cfg)
        print(f"manifest: {manifest['manifest_path']}")
        for split, report in manifest["split_reports"].items():
            print(f"{split}: cases={report['num_cases']} queries={report['num_queries']}")
        return
    if not args.benchmark or not args.output_dir:
        raise SystemExit("--benchmark and --output-dir are required unless --dry-run-summary is set")
    benchmark = load_benchmark(args.benchmark)
    result = validate_benchmark(benchmark)
    if result.errors:
        print(print_validation(result.errors, result.warnings))
        raise SystemExit(1)
    manifest = build_release(
        benchmark,
        args.output_dir,
        cfg,
        source_benchmark_path=args.benchmark,
    )
    print(f"manifest: {manifest['manifest_path']}")
    for split, report in manifest["split_reports"].items():
        print(f"{split}: cases={report['num_cases']} queries={report['num_queries']}")


def cmd_export_public_release(args: argparse.Namespace) -> None:
    manifest = export_public_release_package(args.manifest, args.output_dir)
    print(f"manifest: {manifest['manifest_path']}")
    print(f"package_type: {manifest['package_type']}")
    print(f"included_splits: {', '.join(manifest['included_splits'])}")
    print(f"withheld_splits: {', '.join(sorted(manifest['withheld_splits']))}")


def cmd_build_private_leaderboard_package(args: argparse.Namespace) -> None:
    manifest = build_quarterly_hidden_refresh_package(
        args.output_dir,
        source_profile_id=args.source_profile,
        refresh_id=args.refresh_id,
        seed=args.seed,
        num_hidden_scenarios=args.num_hidden_scenarios,
    )
    print(f"manifest: {manifest['manifest_path']}")
    print(f"package_type: {manifest['package_type']}")
    print(f"refresh_id: {manifest['refresh_id']}")
    print(f"hidden_scenarios: {manifest['quarterly_hidden_refresh']['num_hidden_scenarios']}")


def cmd_sample_release_split(args: argparse.Namespace) -> None:
    result = sample_release_split(
        args.manifest,
        split=args.split,
        output_path=args.output,
        seed=args.seed,
        groups_per_domain=args.groups_per_domain,
        domains=tuple(args.domains) if args.domains else None,
    )
    print(f"benchmark_id: {result['benchmark_id']}")
    print(f"cases: {result['num_cases']}")
    print(f"queries: {result['num_queries']}")
    print(f"output: {result['benchmark_path']}")


def cmd_build_release_representative_baselines(args: argparse.Namespace) -> None:
    result = build_representative_baseline_artifacts(
        args.manifest,
        split=args.split,
        output_dir=args.output_dir,
        retrieval_k=args.retrieval_k,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(f"benchmark_id: {result['benchmark_id']}")
    print(f"split: {result['release_split']}")
    print(f"num_reports: {result['num_reports']}")
    print(f"top_system: {result['top_system_id']}")
    print(f"analysis: {result['analysis_path']}")
    print(f"leaderboard: {result['leaderboard_path']}")


def cmd_build_public_test_baselines(args: argparse.Namespace) -> None:
    result = build_public_test_baseline_artifacts(
        args.manifest,
        split=args.split,
        output_dir=args.output_dir,
        retrieval_k=args.retrieval_k,
    )
    print(f"benchmark_id: {result['benchmark_id']}")
    print(f"split: {result['release_split']}")
    print(f"baselines: {len(result['baseline_kinds'])}")
    print(f"output_dir: {args.output_dir}")
    print(f"summary: {result['output_path']}")


def cmd_build_public_test_sanity(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path

    baselines = json.loads(Path(args.baselines).read_text(encoding="utf-8"))
    failure_modes = json.loads(Path(args.failure_modes).read_text(encoding="utf-8"))
    result = build_public_test_sanity_summary(
        baselines,
        failure_modes,
        output_path=args.output,
    )
    print(f"benchmark_id: {result['benchmark_id']}")
    print(f"split: {result['release_split']}")
    print(f"status: {result['status']}")
    print(f"output: {result['path']}")


def cmd_build_hidden_test_sanity(args: argparse.Namespace) -> None:
    result = build_hidden_test_sanity_artifact(
        args.manifest,
        output_dir=args.output_dir,
        retrieval_k=args.retrieval_k,
    )
    print(f"benchmark_id: {result['benchmark_id']}")
    print(f"split: {result['release_split']}")
    print(f"status: {result['status']}")
    print(f"output: {result['path']}")


def cmd_build_release_failure_modes(args: argparse.Namespace) -> None:
    result = build_release_failure_mode_diagnostics(
        args.manifest,
        split=args.split,
        output_dir=args.output_dir,
        retrieval_k=args.retrieval_k,
    )
    print(f"benchmark_id: {result['benchmark_id']}")
    print(f"split: {result['release_split']}")
    print(f"status: {result['status']}")
    print(f"diagnostics: {result['diagnostics_path']}")


def cmd_build_main_release_workflow(args: argparse.Namespace) -> None:
    cfg = ReleaseConfig(
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        audit_fraction=args.audit_fraction,
        hidden_fraction=args.hidden_fraction,
    )
    result = build_main_release_workflow(
        args.profile,
        args.output_dir,
        existing_release_manifest_path=args.existing_release_manifest,
        public_output_dir=args.public_output_dir,
        reports_dir=args.reports_dir,
        release_config=cfg,
        retrieval_k=args.retrieval_k,
        representative_split=args.representative_split,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        skip_representative_baselines=args.skip_representative_baselines,
        require_completed_human_audit=args.require_completed_human_audit,
        foundation_report_paths=tuple(args.foundation_reports or ()),
        foundation_expected_benchmark_id=args.foundation_expected_benchmark_id,
        foundation_cohort_id=args.foundation_cohort_id,
        foundation_require_full_history=args.foundation_require_full_history,
        require_foundation_validation=args.require_foundation_validation,
    )
    print(f"profile: {result['profile_id']}")
    print(f"benchmark_id: {result['benchmark_id']}")
    print(f"release_manifest: {result['release_manifest_path']}")
    print(f"public_manifest: {result['public_manifest_path']}")
    print(f"release_validation_ok: {result['release_validation_ok']}")
    print(f"public_release_validation_ok: {result['public_release_validation_ok']}")
    print(f"intrinsic_sanity_status: {result['intrinsic_sanity_status']}")
    print(f"acceptance_status: {result['acceptance_status']}")
    print(f"public_result_slices_status: {result['public_result_slices_status']}")
    print(f"hidden_test_sanity_status: {result['hidden_test_sanity_status']}")
    print(f"question_craftsmanship_status: {result['question_craftsmanship_status']}")
    print(f"query_construction_status: {result['query_construction_status']}")
    print(f"probe_discriminativeness_status: {result['probe_discriminativeness_status']}")
    print(f"difficulty_calibration_status: {result['difficulty_calibration_status']}")
    print(f"domain_construct_validity_status: {result['domain_construct_validity_status']}")
    if result["foundation_validation_status"] is not None:
        print(f"foundation_validation_status: {result['foundation_validation_status']}")
    print(f"lineage_status: {result['lineage_status']}")
    if result["representative_baselines"] is not None:
        print(f"representative_top_system: {result['representative_baselines']['top_system_id']}")
        print(f"representative_leaderboard: {result['representative_baselines']['leaderboard_path']}")
    human_bundle = result.get("human_audit_evidence_bundle")
    if isinstance(human_bundle, dict):
        print(f"human_audit_bundle: {human_bundle['bundle_dir']}")
    print(f"workflow_manifest: {result['workflow_manifest_path']}")
    if not result.get("workflow_ok", False):
        raise SystemExit(1)
