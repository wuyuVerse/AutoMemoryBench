"""Command-line interface adapter.

Open-source evaluation build: the full benchmark logic is retained unchanged,
but the user-facing CLI exposes only the evaluation-relevant command surface.
Internal dataset-authoring, human-audit, external-cohort, and release-packaging
commands are registered by the underlying modules but hidden from ``--help`` and
rejected at parse time via a public-command allow-list. No scoring, metric, or
evaluation logic is modified by this filtering.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from amb.benchmark.interfaces.commands.analysis import register_analysis_commands
from amb.benchmark.interfaces.commands.evaluation import register_evaluation_commands
from amb.benchmark.interfaces.commands.generation import register_generation_commands
from amb.benchmark.interfaces.commands.quality import register_quality_commands
from amb.benchmark.interfaces.commands.release import register_release_commands
from amb.lib.cli_run import register_run_commands

# Commands exposed in the open-source evaluation build. Everything else is still
# importable/registerable internally but is not offered on the public CLI.
PUBLIC_COMMANDS = frozenset(
    {
        # generation
        "generate",
        "domain-packs",
        # evaluation (the core reviewer surface)
        "baseline",
        "evaluate",
        "evaluate-release-baseline",
        "evaluate-release-predictions",
        "run-agent",
        "run-foundation-model",
        "run-release-agent",
        "run-release-agent-matrix",
        "scaffold-agent-system",
        "validate-integration-configs",
        "validate-agent-framework-contracts",
        "validate-tool-runtime-contract",
        "validate-agent-framework-dependencies",
        # quality (validation of an external submission only)
        "validate",
        "validate-release",
        # discovery
        "run",
        "list-systems",
        "list-models",
    }
)


class _FilteringSubParsers:
    """Proxy over an ``argparse`` subparsers action.

    Delegates ``add_parser`` to the real action only for allow-listed command
    names; hidden commands get a throwaway parser so registration code runs
    without error but the command never appears in ``--help`` or the public
    namespace.
    """

    def __init__(self, real: argparse._SubParsersAction) -> None:
        self._real = real

    def add_parser(self, name: str, *args: object, **kwargs: object):
        if name in PUBLIC_COMMANDS:
            return self._real.add_parser(name, *args, **kwargs)
        # Hidden command: return a standalone, detached parser so the command
        # module's set_defaults()/add_argument() calls succeed, but the command
        # is never attached to the public subparser and cannot be invoked.
        kwargs.pop("help", None)
        return argparse.ArgumentParser(prog=f"amb {name}", add_help=False)

    def __getattr__(self, item: str) -> object:
        return getattr(self._real, item)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="amb",
        description="AutoMemoryBench (AMB) evaluation CLI",
        allow_abbrev=False,
    )
    real_subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=_NoAbbrevArgumentParser
    )
    subparsers = _FilteringSubParsers(real_subparsers)

    register_quality_commands(subparsers)
    register_generation_commands(subparsers)
    register_evaluation_commands(subparsers)
    register_release_commands(subparsers)
    register_analysis_commands(subparsers)
    register_run_commands(subparsers)

    args = parser.parse_args(list(argv) if argv is not None else None)
    args.handler(args)


class _NoAbbrevArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    main()
