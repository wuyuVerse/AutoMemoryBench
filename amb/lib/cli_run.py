"""The single `amb run` command (plus `amb list-systems`).

This collapses the legacy verbs (evaluate-release-baseline, run-release-agent,
run-release-agent-matrix, run-foundation-model) into one:

    amb run --system mem0 --split public_test
    amb run --system oracle_memory --benchmark data/samples/amst_generated_slice.json
    amb run --system codex_memory_controller --split public_dev --domains coding_agent
    amb list-systems
"""
from __future__ import annotations

import argparse
import json


def register_run_commands(subparsers: argparse._SubParsersAction) -> None:
    run = subparsers.add_parser("run", help="Run one system on a split (or benchmark file) and score it")
    run.add_argument("--system", required=True, help="system name (see `amb list-systems`)")
    src = run.add_mutually_exclusive_group(required=True)
    src.add_argument("--split", choices=("public_dev", "public_test", "audit_subset", "hidden_test"))
    src.add_argument("--benchmark", help="path to a benchmark JSON file (e.g. a sample slice)")
    run.add_argument("--manifest", help="release manifest (default: amst_main_v1_strict_public)")
    run.add_argument("--config", help="explicit system config path (overrides registry lookup)")
    run.add_argument("--kind", choices=("baseline", "memory", "agent"), help="force system kind")
    run.add_argument("--model", help="backbone model id from the registry (see `amb list-models`)")
    run.add_argument("--limit", type=int, help="only the first N cases (benchmark-file path; for smoke)")
    run.add_argument("--domains", help="comma-separated domain filter (split path)")
    run.add_argument("--retrieval-k", type=int, default=8)
    run.add_argument("--task-judge-plugin", default=None)
    run.add_argument("--in-process", action="store_true",
                     help="load memory/agent systems in-process (default: isolated venv worker)")
    run.add_argument("--out", help="write the full report JSON here")
    run.set_defaults(handler=_cmd_run)

    lst = subparsers.add_parser("list-systems", help="List all runnable systems by kind")
    lst.add_argument("--kind", choices=("baseline", "memory", "agent"), help="filter by kind")
    lst.set_defaults(handler=_cmd_list)

    lm = subparsers.add_parser("list-models", help="List backbone models on the matrix model axis")
    lm.set_defaults(handler=_cmd_list_models)


def _cmd_list_models(args: argparse.Namespace) -> None:
    from amb.lib.models import MODEL_REGISTRY

    for mid in sorted(MODEL_REGISTRY):
        s = MODEL_REGISTRY[mid]
        print(f"  {mid:18s} -> {s.model_name:34s} [{s.base_url}] {s.notes}")


def _cmd_run(args: argparse.Namespace) -> None:
    from amb.lib.run import run_system

    domains = {d.strip() for d in args.domains.split(",")} if args.domains else None
    report = run_system(
        args.system,
        split=args.split,
        manifest=args.manifest,
        benchmark=args.benchmark,
        config=args.config,
        kind=args.kind,
        limit=args.limit,
        domains=domains,
        retrieval_k=args.retrieval_k,
        task_judge_plugin=args.task_judge_plugin,
        isolate=not args.in_process,
        model=args.model,
        out=args.out,
    )
    agg = report.get("aggregate", {}) if isinstance(report, dict) else {}
    validity = report.get("validity") if isinstance(report, dict) else None
    summary = {
        "system": args.system,
        "target": args.split or args.benchmark,
        "num_scored_queries": agg.get("num_scored_queries"),
        "amq": agg.get("lifecycle.amq"),
        "task_success": agg.get("task.task_success"),
        "safety_pass": agg.get("safety.safety_pass"),
        "recall_at_k": agg.get("retrieval.recall_at_k"),
    }
    if validity is not None:
        summary["validity"] = validity
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if validity is not None and not validity.get("valid"):
        print("# ⚠️ INVALID RUN — score is an empty-output artifact, not a real measurement:")
        print(f"#    {validity.get('reason')}")
    if args.out:
        print(f"# full report -> {args.out}")


def _cmd_list(args: argparse.Namespace) -> None:
    from amb.lib.systems import list_systems

    systems = list_systems()
    kinds = [args.kind] if args.kind else ["baseline", "memory", "agent"]
    for kind in kinds:
        names = systems.get(kind, [])
        print(f"\n[{kind}]  ({len(names)})")
        for n in names:
            print(f"  {n}")
