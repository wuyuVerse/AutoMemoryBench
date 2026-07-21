"""Analysis and leaderboard CLI commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from amb.benchmark.analysis import (
    EXTERNAL_COHORT_REJECTED_RETURNS_REPORT_SCHEMA_VERSION,
    apply_external_cohort_return_packet,
    analyze_report_files,
    build_external_cohort_return_packet,
    build_external_evidence_gap_report,
    build_protocol_strength_report,
    write_protocol_strength_report,
    build_external_cohort_expansion_validation,
    summarize_external_cohort_rejected_returns,
    sync_external_cohort_return_inbox,
    watch_external_cohort_return_inbox,
    write_external_canonical_refresh,
    write_external_correlation_batch,
    write_external_correlation_batch_summary,
    write_external_cohort_expansion_plan,
    write_external_correlations,
    write_external_evidence_validation,
    write_external_evidence_plan,
    write_merged_normalized_external_scores,
    write_normalized_external_scores,
    write_normalized_longmemeval_scores,
)
from amb.benchmark.artifact_contract import localize_report_contract
from amb.benchmark.interfaces.commands.common import format_metric
from amb.benchmark.leaderboard import write_leaderboard_summary
from amb.benchmark.schemas.io import write_json


def register_analysis_commands(subparsers: argparse._SubParsersAction) -> None:
    analyze = subparsers.add_parser("analyze", help="Analyze one or more evaluation reports")
    analyze.add_argument("--reports", nargs="+", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--seed", type=int, default=13)
    analyze.add_argument("--bootstrap-samples", type=int, default=200)
    analyze.set_defaults(handler=cmd_analyze)

    protocol_strength = subparsers.add_parser(
        "protocol-strength",
        help="Summarize protocol-sensitive and cross-model differences from AMST evaluation reports",
    )
    protocol_strength.add_argument("--reports", nargs="+", required=True)
    protocol_strength.add_argument("--output", required=True)
    protocol_strength.set_defaults(handler=cmd_protocol_strength)

    leaderboard = subparsers.add_parser("leaderboard", help="Build a local leaderboard summary from evaluation reports")
    leaderboard.add_argument("--reports", nargs="+", required=True)
    leaderboard.add_argument("--output", required=True)
    leaderboard.add_argument("--csv-output")
    leaderboard.set_defaults(handler=cmd_leaderboard)

    external = subparsers.add_parser("external-correlation", help="Correlate AMST scores with external benchmark scores")
    external.add_argument("--amst-reports", nargs="+", required=True)
    external.add_argument("--external-scores", nargs="+", required=True)
    external.add_argument("--output", required=True)
    external.add_argument("--amst-metric", default="lifecycle.amq")
    external.add_argument("--external-metric", default="score")
    external.add_argument("--bootstrap-samples", type=int, default=2000)
    external.add_argument("--bootstrap-seed", type=int, default=13013)
    external.set_defaults(handler=cmd_external_correlation)

    normalize_external = subparsers.add_parser("normalize-external-scores", help="Normalize external benchmark scores for correlation")
    normalize_external.add_argument("--input", required=True)
    normalize_external.add_argument("--output", required=True)
    normalize_external.add_argument("--benchmark-id", required=True)
    normalize_external.add_argument("--metric", default="score")
    normalize_external.add_argument("--system-id-field", default="system_id")
    normalize_external.add_argument("--score-field", default="score")
    normalize_external.add_argument("--run-config")
    normalize_external.set_defaults(handler=cmd_normalize_external_scores)

    normalize_longmemeval = subparsers.add_parser(
        "normalize-longmemeval-scores",
        help="Normalize LongMemEval official logs or system-level result JSONs into a correlation-ready score artifact",
    )
    normalize_longmemeval.add_argument("--manifest", required=True)
    normalize_longmemeval.add_argument("--output", required=True)
    normalize_longmemeval.set_defaults(handler=cmd_normalize_longmemeval_scores)

    merge_external_scores = subparsers.add_parser(
        "merge-external-scores",
        help="Merge normalized external score artifacts into one target same-cohort artifact",
    )
    merge_external_scores.add_argument("--input", nargs="+", required=True)
    merge_external_scores.add_argument("--output", required=True)
    merge_external_scores.add_argument("--benchmark-id")
    merge_external_scores.add_argument("--system-cohort", nargs="+", required=True)
    merge_external_scores.add_argument("--source-manifest")
    merge_external_scores.add_argument("--replace-system-id", action="append", default=[])
    merge_external_scores.set_defaults(handler=cmd_merge_external_scores)

    external_batch = subparsers.add_parser(
        "external-correlation-batch",
        help="Generate one external correlation report per normalized benchmark score artifact",
    )
    external_batch.add_argument("--amst-reports", nargs="+", required=True)
    external_batch.add_argument("--external-scores", nargs="+", required=True)
    external_batch.add_argument("--output-dir", required=True)
    external_batch.add_argument("--summary-output")
    external_batch.add_argument("--amst-metric", default="lifecycle.amq")
    external_batch.add_argument("--external-metric", default="score")
    external_batch.add_argument("--bootstrap-samples", type=int, default=2000)
    external_batch.add_argument("--bootstrap-seed", type=int, default=13013)
    external_batch.add_argument("--allow-invalid", action="store_true")
    external_batch.set_defaults(handler=cmd_external_correlation_batch)

    external_plan = subparsers.add_parser(
        "external-evidence-plan",
        help="Write the required external benchmark evidence checklist",
    )
    external_plan.add_argument("--root", default=".")
    external_plan.add_argument("--output", required=True)
    external_plan.add_argument("--output-dir", default="reports/external")
    external_plan.add_argument("--benchmark-id", action="append")
    external_plan.set_defaults(handler=cmd_external_evidence_plan)

    external_validate = subparsers.add_parser(
        "validate-external-evidence",
        help="Validate external correlation reports as one same-cohort evidence set",
    )
    external_validate.add_argument("--correlations", nargs="+", required=True)
    external_validate.add_argument("--output", required=True)
    external_validate.add_argument("--benchmark-id", action="append")
    external_validate.add_argument("--min-shared-systems", type=int, default=3)
    external_validate.add_argument("--min-control-shared-systems", type=int, default=1)
    external_validate.add_argument("--min-real-memory-shared-systems", type=int, default=1)
    external_validate.add_argument("--allow-different-cohorts", action="store_true")
    external_validate.set_defaults(handler=cmd_validate_external_evidence)

    external_gap = subparsers.add_parser(
        "summarize-external-evidence-gaps",
        help="Summarize why the external evidence set still fails the completion gate",
    )
    external_gap.add_argument("--correlations", nargs="+", required=True)
    external_gap.add_argument("--output", required=True)
    external_gap.add_argument("--benchmark-id", action="append")
    external_gap.add_argument("--real-system-validation")
    external_gap.add_argument("--min-shared-systems", type=int, default=3)
    external_gap.add_argument("--min-control-shared-systems", type=int, default=1)
    external_gap.add_argument("--min-real-memory-shared-systems", type=int, default=1)
    external_gap.add_argument("--allow-different-cohorts", action="store_true")
    external_gap.set_defaults(handler=cmd_summarize_external_evidence_gaps)

    external_expand = subparsers.add_parser(
        "build-external-cohort-expansion-plan",
        help="Build a concrete provider/benchmark plan for expanding the external same-system cohort",
    )
    external_expand.add_argument("--correlations", nargs="+", required=True)
    external_expand.add_argument("--output", required=True)
    external_expand.add_argument("--benchmark-id", action="append")
    external_expand.add_argument("--real-system-validation")
    external_expand.add_argument("--min-shared-systems", type=int, default=3)
    external_expand.add_argument("--min-control-shared-systems", type=int, default=1)
    external_expand.add_argument("--min-real-memory-shared-systems", type=int, default=1)
    external_expand.add_argument("--allow-different-cohorts", action="store_true")
    external_expand.set_defaults(handler=cmd_build_external_cohort_expansion_plan)

    external_expand_verify = subparsers.add_parser(
        "verify-external-cohort-expansion-plan",
        help="Verify that the external cohort expansion plan includes runnable scripts and a complete recommended candidate contract",
    )
    external_expand_verify.add_argument("--expansion", required=True)
    external_expand_verify.add_argument("--output", required=True)
    external_expand_verify.set_defaults(handler=cmd_verify_external_cohort_expansion_plan)

    external_return_build = subparsers.add_parser(
        "build-external-cohort-return-packet",
        help="Package refreshed same-cohort external score artifacts into a provider return packet",
    )
    external_return_build.add_argument("--expansion", required=True)
    external_return_build.add_argument("--provider", required=True)
    external_return_build.add_argument("--score", action="append", required=True, help="benchmark_id=/abs/path/to/score.json")
    external_return_build.add_argument("--output", required=True)
    external_return_build.set_defaults(handler=cmd_build_external_cohort_return_packet)

    external_return_apply = subparsers.add_parser(
        "apply-external-cohort-return-packet",
        help="Apply a provider return packet back into canonical external evidence and refresh correlations",
    )
    external_return_apply.add_argument("--expansion", required=True)
    external_return_apply.add_argument("--packet", required=True)
    external_return_apply.add_argument("--output")
    external_return_apply.add_argument("--real-system-validation")
    external_return_apply.set_defaults(handler=cmd_apply_external_cohort_return_packet)

    external_return_sync = subparsers.add_parser(
        "sync-external-cohort-return-inbox",
        help="Apply all provider return packets waiting in reports/external/returns/inbox",
    )
    external_return_sync.add_argument("--expansion", required=True)
    external_return_sync.add_argument("--output", required=True)
    external_return_sync.add_argument("--real-system-validation")
    external_return_sync.set_defaults(handler=cmd_sync_external_cohort_return_inbox)

    external_return_watch = subparsers.add_parser(
        "watch-external-cohort-return-inbox",
        help="Repeatedly sync provider return packets from reports/external/returns/inbox and keep one watch summary current",
    )
    external_return_watch.add_argument("--expansion", required=True)
    external_return_watch.add_argument("--output", required=True)
    external_return_watch.add_argument("--real-system-validation")
    external_return_watch.add_argument("--interval-s", type=float, default=60.0)
    external_return_watch.add_argument("--max-iterations", type=int, default=1)
    external_return_watch.add_argument("--stop-when-ready", action="store_true")
    external_return_watch.add_argument("--stop-when-rejected", action="store_true")
    external_return_watch.set_defaults(handler=cmd_watch_external_cohort_return_inbox)

    external_rejected = subparsers.add_parser(
        "summarize-external-cohort-rejected-returns",
        help="Summarize invalid provider return packets quarantined under reports/external/returns/rejected",
    )
    external_rejected.add_argument("--external-dir", required=True)
    external_rejected.add_argument("--output", required=True)
    external_rejected.set_defaults(handler=cmd_summarize_external_cohort_rejected_returns)

    external_refresh = subparsers.add_parser(
        "refresh-external-canonical",
        help="Refresh canonical external evidence plan, validation, and gap artifacts together",
    )
    external_refresh.add_argument("--root", default=".")
    external_refresh.add_argument("--output-dir", default="reports/external")
    external_refresh.add_argument("--benchmark-id", action="append")
    external_refresh.add_argument("--plan-output")
    external_refresh.add_argument("--validation-output")
    external_refresh.add_argument("--gap-output")
    external_refresh.add_argument("--expansion-output")
    external_refresh.add_argument("--expansion-validation-output")
    external_refresh.add_argument("--real-system-validation")
    external_refresh.add_argument("--min-shared-systems", type=int, default=3)
    external_refresh.add_argument("--min-control-shared-systems", type=int, default=1)
    external_refresh.add_argument("--min-real-memory-shared-systems", type=int, default=1)
    external_refresh.add_argument("--allow-different-cohorts", action="store_true")
    external_refresh.set_defaults(handler=cmd_refresh_external_canonical)


def cmd_analyze(args: argparse.Namespace) -> None:
    analysis = analyze_report_files(
        args.reports,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    analysis = localize_report_contract(
        analysis,
        output_path=args.output,
        project_root_hints=tuple(args.reports),
    )
    write_json(args.output, analysis)
    print(f"reports: {analysis['num_reports']}")
    print(f"metrics: {', '.join(analysis['metrics'])}")
    print(f"comparisons: {len(analysis['comparisons'])}")
    print(f"frontier_points: {len(analysis['quality_cost_frontier']['frontier'])}")
    print(f"weight_profiles: {len(analysis.get('weight_sensitivity', {}).get('profiles', []))}")


def cmd_protocol_strength(args: argparse.Namespace) -> None:
    report = write_protocol_strength_report(args.reports, args.output)
    print(f"reports: {report['num_reports']}")
    print(f"pairs: {len(report['pairwise'])}")
    print(f"metrics: {', '.join(report['metrics'])}")


def cmd_leaderboard(args: argparse.Namespace) -> None:
    summary = write_leaderboard_summary(args.reports, args.output, output_csv=args.csv_output)
    print(f"systems: {summary['num_systems']}")
    if summary["rows"]:
        top = summary["rows"][0]
        print(f"top_system: {top['system_id']}")
        print(f"top_amq: {format_metric(top['amq'])}")
    if args.csv_output:
        print(f"csv: {args.csv_output}")


def cmd_external_correlation(args: argparse.Namespace) -> None:
    report = write_external_correlations(
        args.amst_reports,
        args.external_scores,
        args.output,
        amst_metric=args.amst_metric,
        external_metric=args.external_metric,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(f"amst_systems: {report['num_amst_systems']}")
    for result in report["external_results"]:
        print(
            f"{result['external_score_path']}: "
            f"systems={result['num_common_systems']} "
            f"spearman={format_metric(result['spearman'])} "
            f"kendall_tau_b={format_metric(result['kendall_tau_b'])}"
        )


def cmd_normalize_external_scores(args: argparse.Namespace) -> None:
    report = write_normalized_external_scores(
        args.input,
        args.output,
        benchmark_id=args.benchmark_id,
        metric=args.metric,
        system_id_field=args.system_id_field,
        score_field=args.score_field,
        run_config_path=args.run_config,
    )
    print(f"benchmark_id: {report['benchmark_id']}")
    print(f"metric: {report['metric']}")
    print(f"systems: {report['num_systems']}")
    print(f"output: {args.output}")


def cmd_normalize_longmemeval_scores(args: argparse.Namespace) -> None:
    report = write_normalized_longmemeval_scores(args.manifest, args.output)
    print(f"benchmark_id: {report['benchmark_id']}")
    print(f"metric: {report['metric']}")
    print(f"systems: {report['num_systems']}")
    print(f"output: {args.output}")


def cmd_merge_external_scores(args: argparse.Namespace) -> None:
    report = write_merged_normalized_external_scores(
        args.input,
        args.output,
        benchmark_id=args.benchmark_id,
        system_cohort=args.system_cohort,
        source_manifest_path=args.source_manifest,
        replace_system_ids=args.replace_system_id,
    )
    print(f"benchmark_id: {report['benchmark_id']}")
    print(f"metric: {report['metric']}")
    print(f"systems: {report['num_systems']}")
    print(f"output: {args.output}")


def cmd_external_correlation_batch(args: argparse.Namespace) -> None:
    summary = write_external_correlation_batch(
        args.amst_reports,
        args.external_scores,
        args.output_dir,
        amst_metric=args.amst_metric,
        external_metric=args.external_metric,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        strict=not args.allow_invalid,
    )
    if args.summary_output:
        write_external_correlation_batch_summary(summary, args.summary_output)
    print(f"status: {summary['status']}")
    print(f"external_benchmarks: {summary['num_external_benchmarks']}")
    for row in summary["outputs"]:
        print(f"{row['benchmark_id']}: {row['status']} -> {row['correlation_report']}")
    if args.summary_output:
        print(f"summary: {args.summary_output}")


def cmd_external_evidence_plan(args: argparse.Namespace) -> None:
    plan = write_external_evidence_plan(
        args.output,
        root=args.root,
        output_dir=args.output_dir,
        benchmark_ids=args.benchmark_id,
    )
    print(f"status: {plan['status']}")
    print(
        "summary: "
        f"passed={plan['summary']['passed']} "
        f"ready={plan['summary']['ready']} "
        f"missing={plan['summary']['missing']} "
        f"invalid={plan['summary']['invalid']}"
    )
    print(f"output: {args.output}")


def cmd_validate_external_evidence(args: argparse.Namespace) -> None:
    report = write_external_evidence_validation(
        args.correlations,
        args.output,
        required_benchmark_ids=args.benchmark_id,
        min_shared_systems=args.min_shared_systems,
        min_shared_control_systems=args.min_control_shared_systems,
        min_shared_real_memory_systems=args.min_real_memory_shared_systems,
        require_identical_systems=not args.allow_different_cohorts,
    )
    print(f"status: {report['status']}")
    print(f"covered_benchmarks: {len(report['covered_benchmark_ids'])}")
    print(f"shared_systems: {report['num_shared_systems']}")
    print(f"shared_controls: {report['num_shared_control_systems']}")
    print(f"shared_real_memory_systems: {report['num_shared_real_memory_systems']}")
    print(f"errors: {len(report['errors'])}")
    if report["errors"]:
        raise SystemExit(1)


def cmd_summarize_external_evidence_gaps(args: argparse.Namespace) -> None:
    report = build_external_evidence_gap_report(
        args.correlations,
        real_system_validation_path=args.real_system_validation,
        required_benchmark_ids=args.benchmark_id,
        min_shared_systems=args.min_shared_systems,
        min_shared_control_systems=args.min_control_shared_systems,
        min_shared_real_memory_systems=args.min_real_memory_shared_systems,
        require_identical_systems=not args.allow_different_cohorts,
    )
    write_json(args.output, report)
    validation = report["validation"]
    print(f"status: {report['status']}")
    print(f"shared_systems: {validation['num_shared_systems']}")
    print(f"shared_controls: {validation['num_shared_control_systems']}")
    print(f"shared_real_memory_systems: {validation['num_shared_real_memory_systems']}")
    print(f"providers_missing_everywhere: {len(report['providers_missing_from_all_external_benchmarks'])}")
    print(f"output: {args.output}")


def cmd_build_external_cohort_expansion_plan(args: argparse.Namespace) -> None:
    report = write_external_cohort_expansion_plan(
        args.correlations,
        args.output,
        root=".",
        real_system_validation_path=args.real_system_validation,
        required_benchmark_ids=args.benchmark_id,
        min_shared_systems=args.min_shared_systems,
        min_shared_control_systems=args.min_control_shared_systems,
        min_shared_real_memory_systems=args.min_real_memory_shared_systems,
        require_identical_systems=not args.allow_different_cohorts,
    )
    print(f"status: {report['status']}")
    print(f"minimum_completion_candidates: {len(report['minimum_completion_candidates'])}")
    print(f"available_real_memory_candidates: {len(report['available_real_memory_candidates'])}")
    if report.get("handoff_manifest_file"):
        print(f"handoff_manifest: {report['handoff_manifest_file']}")
    print(f"output: {args.output}")


def cmd_verify_external_cohort_expansion_plan(args: argparse.Namespace) -> None:
    expansion_path = args.expansion
    from amb.benchmark.schemas.io import read_json  # local import keeps top-level CLI imports minimal

    payload = read_json(expansion_path)
    report = build_external_cohort_expansion_validation(payload, root=Path(expansion_path).resolve().parent)
    write_json(args.output, report)
    print(f"status: {report['status']}")
    print(f"errors: {len(report['errors'])}")
    print(f"output: {args.output}")
    if report["errors"]:
        raise SystemExit(1)


def cmd_build_external_cohort_return_packet(args: argparse.Namespace) -> None:
    report = build_external_cohort_return_packet(
        args.expansion,
        provider=args.provider,
        score_paths=_parse_external_score_pairs(args.score),
        output=args.output,
    )
    print(f"status: {report['status']}")
    print(f"provider: {report['provider']}")
    print(f"packet_file: {report['packet_file']}")


def cmd_apply_external_cohort_return_packet(args: argparse.Namespace) -> None:
    report = apply_external_cohort_return_packet(
        args.packet,
        expansion_path=args.expansion,
        root=".",
        real_system_validation_path=args.real_system_validation,
    )
    if args.output:
        write_json(args.output, report)
    print(f"status: {report['status']}")
    print(f"provider: {report['provider']}")
    print(f"refresh_status: {report['refresh_status']}")
    print(f"completion_validation_status: {report['completion_validation_status']}")
    if report.get("completion_validation_status") != "passed":
        raise SystemExit(1)


def cmd_sync_external_cohort_return_inbox(args: argparse.Namespace) -> None:
    report = sync_external_cohort_return_inbox(
        args.expansion,
        root=".",
        real_system_validation_path=args.real_system_validation,
    )
    write_json(args.output, report)
    print(f"status: {report['status']}")
    print(f"processed_packets: {report['num_processed_packets']}")
    print(f"rejected_packets: {report['num_rejected_packets']}")
    print(f"output: {args.output}")
    if report["rejected_packets"]:
        raise SystemExit(1)


def cmd_watch_external_cohort_return_inbox(args: argparse.Namespace) -> None:
    if args.interval_s < 0:
        raise SystemExit("--interval-s must be non-negative")
    if args.max_iterations < 0:
        raise SystemExit("--max-iterations must be non-negative")
    try:
        summary = watch_external_cohort_return_inbox(
            args.expansion,
            root=".",
            real_system_validation_path=args.real_system_validation,
            interval_s=args.interval_s,
            max_iterations=args.max_iterations,
            stop_when_ready=args.stop_when_ready,
            stop_when_rejected=args.stop_when_rejected,
            output_path=args.output,
        )
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    print(f"iterations: {summary['iteration_count']}")
    print(f"expansion_status: {summary['expansion_status']}")
    print(f"ready_for_completion: {summary['ready_for_completion']}")
    print(f"stop_reason: {summary['stop_reason']}")
    print(f"stop_exit_code: {summary.get('stop_exit_code', 0)}")
    if summary.get("stop_reason") == "rejected_returns":
        print(f"rejected_returns_report: {summary.get('rejected_returns_report_file')}")
    if summary.get("stop_reason") in {"ready", "rejected_returns"}:
        if isinstance(summary.get("next_command"), str) and summary["next_command"].strip():
            print(f"next_command: {summary['next_command']}")
        if isinstance(summary.get("next_script"), str) and summary["next_script"].strip():
            print(f"next_script: {summary['next_script']}")
        if isinstance(summary.get("next_script_file"), str) and summary["next_script_file"].strip():
            print(f"next_script_file: {summary['next_script_file']}")
    print(f"output: {args.output}")
    stop_exit_code = int(summary.get("stop_exit_code", 0) or 0)
    if stop_exit_code:
        raise SystemExit(stop_exit_code)


def cmd_summarize_external_cohort_rejected_returns(args: argparse.Namespace) -> None:
    report = summarize_external_cohort_rejected_returns(args.external_dir)
    payload = {
        "schema_version": EXTERNAL_COHORT_REJECTED_RETURNS_REPORT_SCHEMA_VERSION,
        "status": "passed" if report["num_rejected_candidate_packets"] == 0 else "incomplete",
        "external_dir": str(Path(args.external_dir).resolve()),
        "rejected_return_summary": report,
        "num_rejected_candidate_packets": report["num_rejected_candidate_packets"],
        "rejected_candidate_packets": report["rejected_candidate_packets"],
    }
    write_json(args.output, payload)
    print(f"status: {payload['status']}")
    print(f"rejected_packets: {payload['num_rejected_candidate_packets']}")
    print(f"output: {args.output}")
    if payload["num_rejected_candidate_packets"]:
        raise SystemExit(1)


def cmd_refresh_external_canonical(args: argparse.Namespace) -> None:
    report = write_external_canonical_refresh(
        root=args.root,
        output_dir=args.output_dir,
        benchmark_ids=args.benchmark_id,
        plan_output=args.plan_output,
        validation_output=args.validation_output,
        gap_output=args.gap_output,
        expansion_output=args.expansion_output,
        expansion_validation_output=args.expansion_validation_output,
        real_system_validation_path=args.real_system_validation,
        min_shared_systems=args.min_shared_systems,
        min_shared_control_systems=args.min_control_shared_systems,
        min_shared_real_memory_systems=args.min_real_memory_shared_systems,
        require_identical_systems=not args.allow_different_cohorts,
    )
    print(f"status: {report['status']}")
    print(f"plan_status: {report['plan_status']}")
    print(f"validation_status: {report['validation_status']}")
    print(f"gap_status: {report['gap_status']}")
    print(f"expansion_status: {report['expansion_status']}")
    print(f"plan_output: {report['plan_output']}")
    print(f"validation_output: {report['validation_output']}")
    print(f"gap_output: {report['gap_output']}")
    print(f"expansion_output: {report['expansion_output']}")
    print(f"expansion_validation_output: {report['expansion_validation_output']}")
    if report.get("expansion_handoff_output"):
        print(f"expansion_handoff_output: {report['expansion_handoff_output']}")


def _parse_external_score_pairs(values: list[str]) -> dict[str, str]:
    score_paths: dict[str, str] = {}
    for value in values:
        benchmark_id, separator, path = value.partition("=")
        if not separator or not benchmark_id or not path:
            raise SystemExit(f"--score must be BENCHMARK_ID=/path/to/score.json, got: {value!r}")
        score_paths[benchmark_id] = path
    return score_paths
