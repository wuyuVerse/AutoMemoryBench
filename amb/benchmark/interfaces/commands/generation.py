"""Generation and domain-pack CLI commands."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace

from amb.benchmark.generation import (
    GenerationConfig,
    benchmark_construction_summary,
    expected_generation_summary,
    generate_benchmark,
    generation_profile,
    profile_names,
)
from amb.benchmark.generation.domains.packs import domain_pack_catalog, domain_pack_names
from amb.benchmark.interfaces.commands.common import print_generation_summary
from amb.benchmark.quality.validation import validate_benchmark
from amb.benchmark.schemas.io import write_json
from amb.benchmark.evaluation.report import print_validation


def register_generation_commands(subparsers: argparse._SubParsersAction) -> None:
    packs = subparsers.add_parser("domain-packs", help="Export machine-readable domain pack definitions")
    packs.add_argument("--output-dir", required=True)
    packs.add_argument(
        "--domains",
        help="Optional comma-separated domain names. Omit to export all built-in domain packs.",
    )
    packs.set_defaults(handler=cmd_domain_packs)

    generate = subparsers.add_parser("generate", help="Generate an event-graph-first AutoMemoryBench dataset")
    generate.add_argument("--output")
    generate.add_argument("--profile", choices=profile_names(), help="Named generation profile.")
    generate.add_argument("--dry-run-summary", action="store_true", help="Print the planned construction summary without writing data.")
    generate.add_argument("--case-count-per-domain", type=int)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--benchmark-id")
    generate.add_argument("--name")
    generate.add_argument(
        "--counterfactual-variants-per-case",
        type=int,
        help="Number of counterfactual variants generated for each base case.",
    )
    generate.add_argument(
        "--domains",
        help="Optional comma-separated domain names. Omit to generate all built-in domains.",
    )
    generate.set_defaults(handler=cmd_generate)


def cmd_domain_packs(args: argparse.Namespace) -> None:
    requested = None
    if args.domains:
        requested = tuple(item.strip() for item in args.domains.split(",") if item.strip())
    known = set(domain_pack_names())
    if requested:
        missing = sorted(set(requested) - known)
        if missing:
            raise SystemExit(f"Unknown domain pack(s): {', '.join(missing)}")
    catalog = domain_pack_catalog()
    exported = []
    for domain, pack in sorted(catalog.items()):
        if requested and domain not in requested:
            continue
        path = f"{args.output_dir}/{domain}.json"
        write_json(path, pack)
        exported.append(path)
    print(f"exported_domain_packs: {len(exported)}")
    for path in exported:
        print(path)


def cmd_generate(args: argparse.Namespace) -> None:
    domains = None
    if args.domains:
        domains = tuple(item.strip() for item in args.domains.split(",") if item.strip())
    config = generation_profile(args.profile).config if args.profile else GenerationConfig()
    if args.case_count_per_domain is not None:
        config = replace(config, case_count_per_domain=args.case_count_per_domain)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.benchmark_id is not None:
        config = replace(config, benchmark_id=args.benchmark_id)
    if args.name is not None:
        config = replace(config, name=args.name)
    if domains is not None:
        config = replace(config, domains=domains)
    if args.counterfactual_variants_per_case is not None:
        config = replace(config, counterfactual_variants_per_case=args.counterfactual_variants_per_case)
    if args.dry_run_summary:
        summary = expected_generation_summary(config)
        write_json(args.output, summary) if args.output else None
        print_generation_summary(summary)
        return
    if not args.output:
        raise SystemExit("--output is required unless --dry-run-summary is set")
    benchmark = generate_benchmark(config)
    result = validate_benchmark(benchmark)
    if result.errors:
        print(print_validation(result.errors, result.warnings))
        raise SystemExit(1)
    write_json(args.output, asdict(benchmark))
    summary = benchmark_construction_summary(benchmark)
    print(f"Wrote benchmark to {args.output}")
    print_generation_summary(summary)
