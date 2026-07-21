"""Evaluation CLI commands."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import os
from pathlib import Path
import sys
import uuid

from amb.benchmark.artifact_contract import localize_report_contract
from amb.benchmark.evaluation.baselines import available_baselines, make_baseline
from amb.benchmark.evaluation.foundation_runner import (
    FOUNDATION_PROTOCOLS,
    prediction_set_to_dict,
    run_foundation_model,
    run_foundation_model_release_split,
)
from amb.benchmark.evaluation.agent_systems import (
    AgentSystemSpec,
    agent_system_metadata,
    bind_agent_system_metadata,
    load_agent_system,
)
from amb.benchmark.evaluation.framework_dependencies import write_dependency_gate_contract_validation
from amb.benchmark.evaluation.framework_contracts import write_framework_adapter_contract_validation
from amb.benchmark.evaluation.framework_trace import (
    framework_trace_artifact_payload,
    load_default_tool_runtime_contracts,
    validate_framework_trace_artifact_payload,
)
from amb.benchmark.evaluation.openai_compatible import OpenAICompatibleChatClient
from amb.benchmark.evaluation.report import compact_summary, print_validation
from amb.benchmark.evaluation.runner import run_black_box_agent
from amb.benchmark.evaluation.scoring import DEFAULT_RETRIEVAL_K, Scorer
from amb.benchmark.evaluation.tool_runtime import write_tool_runtime_contract_validation
from amb.benchmark.metrics.task_judges import (
    DEFAULT_TASK_JUDGE_PLUGIN_ID,
    available_task_judge_plugins,
    load_task_judge_plugin,
)
from amb.benchmark.integrations.config_validation import write_integration_config_validation
from amb.benchmark.quality.run_metadata import (
    build_real_system_attestation,
    build_run_metadata,
    validate_real_system_run_metadata,
    run_metadata_artifact,
)
from amb.benchmark.quality.validation import validate_benchmark, validate_predictions
from amb.benchmark.release.evaluation import (
    evaluate_release_split_baseline,
    evaluate_release_split_predictions,
    release_split_benchmark_id,
    run_release_split_agent,
    run_release_split_agent_experiment,
    run_release_split_agent_experiment_with_retries,
)
from amb.benchmark.schemas.io import read_json
from amb.benchmark.release.artifacts import artifact_info
from amb.benchmark.release.fingerprint import release_split_contract_fingerprint
from amb.benchmark.schemas.io import load_benchmark, load_predictions, read_json, write_json
from amb.benchmark.security.env_vars import require_env_var_name


@dataclass(frozen=True)
class LoadedCliAgent:
    agent: object
    spec: AgentSystemSpec
    config_artifact: dict[str, object]
    execution_mode: str
    real_system_attestation: dict[str, object] | None
    dependencies: dict[str, object]
    agent_system: dict[str, object]


def register_evaluation_commands(subparsers: argparse._SubParsersAction) -> None:
    baseline = subparsers.add_parser("baseline", help="Generate deterministic baseline predictions")
    baseline.add_argument("--benchmark", required=True)
    baseline.add_argument("--kind", required=True, choices=available_baselines())
    baseline.add_argument("--output", required=True)
    baseline.set_defaults(handler=cmd_baseline)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate predictions")
    evaluate.add_argument("--benchmark", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--retrieval-k", type=int, default=DEFAULT_RETRIEVAL_K)
    evaluate.add_argument(
        "--task-judge-plugin",
        choices=available_task_judge_plugins(),
        default=DEFAULT_TASK_JUDGE_PLUGIN_ID,
    )
    evaluate.add_argument("--quiet", action="store_true")
    evaluate.set_defaults(handler=cmd_evaluate)

    evaluate_release = subparsers.add_parser("evaluate-release-baseline", help="Evaluate a deterministic baseline over release split shards")
    evaluate_release.add_argument("--manifest", required=True)
    evaluate_release.add_argument("--split", required=True, choices=("public_dev", "public_test", "audit_subset", "hidden_test"))
    evaluate_release.add_argument("--kind", required=True, choices=available_baselines())
    evaluate_release.add_argument("--output", required=True)
    evaluate_release.add_argument("--retrieval-k", type=int, default=DEFAULT_RETRIEVAL_K)
    evaluate_release.add_argument(
        "--baseline-top-k",
        type=int,
        help="Optional predictor retrieval top-k for tunable deterministic baselines.",
    )
    evaluate_release.add_argument(
        "--task-judge-plugin",
        choices=available_task_judge_plugins(),
        default=DEFAULT_TASK_JUDGE_PLUGIN_ID,
    )
    evaluate_release.add_argument("--quiet", action="store_true")
    evaluate_release.set_defaults(handler=cmd_evaluate_release_baseline)

    evaluate_release_predictions = subparsers.add_parser("evaluate-release-predictions", help="Evaluate submitted predictions over release split shards")
    evaluate_release_predictions.add_argument("--manifest", required=True)
    evaluate_release_predictions.add_argument("--split", required=True, choices=("public_dev", "public_test", "audit_subset", "hidden_test"))
    evaluate_release_predictions.add_argument("--predictions", required=True)
    evaluate_release_predictions.add_argument("--output", required=True)
    evaluate_release_predictions.add_argument("--retrieval-k", type=int, default=DEFAULT_RETRIEVAL_K)
    evaluate_release_predictions.add_argument(
        "--task-judge-plugin",
        choices=available_task_judge_plugins(),
        default=DEFAULT_TASK_JUDGE_PLUGIN_ID,
    )
    evaluate_release_predictions.add_argument("--run-metadata")
    evaluate_release_predictions.add_argument("--framework-trace")
    evaluate_release_predictions.add_argument("--case-ids-file")
    evaluate_release_predictions.add_argument("--domains", nargs="+")
    evaluate_release_predictions.add_argument("--quiet", action="store_true")
    evaluate_release_predictions.set_defaults(handler=cmd_evaluate_release_predictions)

    run_agent = subparsers.add_parser("run-agent", help="Run an external memory-agent integration over a benchmark")
    run_agent.add_argument("--benchmark", required=True)
    run_agent.add_argument("--config", required=True)
    run_agent.add_argument("--output", required=True)
    run_agent.add_argument("--system-id")
    run_agent.add_argument("--framework-trace-output")
    run_agent.set_defaults(handler=cmd_run_agent)

    scaffold_agent_system = subparsers.add_parser(
        "scaffold-agent-system",
        help="Create a generic agent-system config and optional Python adapter template",
    )
    scaffold_agent_system.add_argument("--system-id", required=True)
    scaffold_agent_system.add_argument("--loader", required=True, help="Adapter factory reference in module:callable format")
    scaffold_agent_system.add_argument("--framework", required=True, help="Agent framework/runtime name")
    scaffold_agent_system.add_argument("--output", required=True, help="Path for the generated agent-system JSON config")
    scaffold_agent_system.add_argument("--adapter-output", help="Optional path for a starter Python adapter module")
    scaffold_agent_system.add_argument("--provider", help="Provider/runtime label; defaults to --framework")
    scaffold_agent_system.add_argument("--system-version", default="0.1.0")
    scaffold_agent_system.add_argument("--execution-mode", default="integration_smoke")
    scaffold_agent_system.add_argument("--agent-runtime", default="python_in_process")
    scaffold_agent_system.add_argument("--orchestration-mode", default="single_agent")
    scaffold_agent_system.add_argument("--memory-backend", default="custom_memory")
    scaffold_agent_system.add_argument("--model-backend", default="custom_model")
    scaffold_agent_system.add_argument("--tool-runtime-id", default="automemorybench_tool_runtime_v1")
    scaffold_agent_system.add_argument("--loader-kwarg", action="append", help="Repeatable KEY=VALUE loader kwarg")
    scaffold_agent_system.add_argument("--force", action="store_true", help="Overwrite existing output files")
    scaffold_agent_system.set_defaults(handler=cmd_scaffold_agent_system)

    run_agent_experiment = subparsers.add_parser(
        "run-agent-experiment",
        help="Run, evaluate, and package one external memory-agent experiment over a benchmark sample",
    )
    run_agent_experiment.add_argument("--benchmark", required=True)
    run_agent_experiment.add_argument("--config", required=True)
    run_agent_experiment.add_argument("--output-dir", required=True)
    run_agent_experiment.add_argument("--system-id")
    run_agent_experiment.add_argument("--system-version", default="unspecified")
    run_agent_experiment.add_argument("--retrieval-k", type=int, default=DEFAULT_RETRIEVAL_K)
    run_agent_experiment.add_argument(
        "--task-judge-plugin",
        choices=available_task_judge_plugins(),
        default=DEFAULT_TASK_JUDGE_PLUGIN_ID,
    )
    run_agent_experiment.add_argument("--benchmark-id")
    run_agent_experiment.add_argument("--release-split", default="sample")
    run_agent_experiment.add_argument("--framework-trace-output")
    run_agent_experiment.set_defaults(handler=cmd_run_agent_experiment)

    run_foundation = subparsers.add_parser("run-foundation-model", help="Run an OpenAI-compatible foundation model over a benchmark or release split")
    run_foundation.add_argument("--benchmark")
    run_foundation.add_argument("--manifest")
    run_foundation.add_argument("--split", choices=("public_dev", "public_test", "audit_subset", "hidden_test"))
    run_foundation.add_argument("--base-url", required=True)
    run_foundation.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run_foundation.add_argument("--model", required=True)
    run_foundation.add_argument("--protocol", required=True, choices=FOUNDATION_PROTOCOLS)
    run_foundation.add_argument("--output", required=True)
    run_foundation.add_argument("--system-id")
    run_foundation.add_argument("--temperature", type=float, default=0.0)
    run_foundation.add_argument("--max-tokens", type=int, default=512)
    run_foundation.add_argument("--timeout-s", type=float, default=60.0)
    run_foundation.set_defaults(handler=cmd_run_foundation_model)

    run_release_agent = subparsers.add_parser("run-release-agent", help="Run an external memory-agent integration over a release split")
    run_release_agent.add_argument("--manifest", required=True)
    run_release_agent.add_argument("--split", required=True, choices=("public_dev", "public_test", "audit_subset", "hidden_test"))
    run_release_agent.add_argument("--config", required=True)
    run_release_agent.add_argument("--output", required=True)
    run_release_agent.add_argument("--system-id")
    run_release_agent.add_argument("--system-version", default="unspecified")
    run_release_agent.add_argument("--execution-mode", default="integration_smoke")
    run_release_agent.add_argument("--run-metadata-output")
    run_release_agent.add_argument("--framework-trace-output")
    run_release_agent.add_argument("--resume", action="store_true")
    run_release_agent.add_argument(
        "--case-ids-file",
        help="Optional newline-delimited case_id file for safe sharded release-agent runs.",
    )
    run_release_agent.add_argument(
        "--domains",
        nargs="+",
        help="Optional domain filter for safe sharded release-agent runs. Accepts space-separated values or comma lists.",
    )
    run_release_agent.add_argument(
        "--failure-predictions",
        action="store_true",
        help="Record failed queries as empty strict-failure predictions instead of aborting the run.",
    )
    run_release_agent.set_defaults(handler=cmd_run_release_agent)

    matrix = subparsers.add_parser("run-release-agent-matrix", help="Run and evaluate multiple integration configs over a release split")
    matrix.add_argument("--manifest", required=True)
    matrix.add_argument("--split", required=True, choices=("public_dev", "public_test", "audit_subset", "hidden_test"))
    matrix.add_argument("--configs", nargs="+", required=True)
    matrix.add_argument("--output-dir", required=True)
    matrix.add_argument("--system-version", default="unspecified")
    matrix.add_argument("--retrieval-k", type=int, default=DEFAULT_RETRIEVAL_K)
    matrix.add_argument(
        "--task-judge-plugin",
        choices=available_task_judge_plugins(),
        default=DEFAULT_TASK_JUDGE_PLUGIN_ID,
    )
    matrix.add_argument("--resume", action="store_true")
    matrix.add_argument("--framework-traces", action="store_true", help="Write per-system framework_trace.json artifacts")
    matrix.add_argument("--max-run-attempts", type=int, default=3)
    matrix.add_argument("--retry-backoff-s", type=float, default=1.0)
    matrix.set_defaults(handler=cmd_run_release_agent_matrix)

    validate_configs = subparsers.add_parser(
        "validate-integration-configs",
        help="Statically validate integration configs before running agents",
    )
    validate_configs.add_argument("--configs", nargs="+", required=True)
    validate_configs.add_argument("--framework-contracts", nargs="*", default=())
    validate_configs.add_argument("--output", required=True)
    validate_configs.set_defaults(handler=cmd_validate_integration_configs)

    validate_framework_contracts = subparsers.add_parser(
        "validate-agent-framework-contracts",
        help="Validate agent-framework adapter contracts before running framework-comparative experiments",
    )
    validate_framework_contracts.add_argument("--contracts", nargs="+", required=True)
    validate_framework_contracts.add_argument("--output", required=True)
    validate_framework_contracts.set_defaults(handler=cmd_validate_agent_framework_contracts)

    validate_tool_runtime = subparsers.add_parser(
        "validate-tool-runtime-contract",
        help="Validate the standard tool runtime contract used by framework-comparative runs",
    )
    validate_tool_runtime.add_argument("--contract", required=True)
    validate_tool_runtime.add_argument("--output", required=True)
    validate_tool_runtime.set_defaults(handler=cmd_validate_tool_runtime_contract)

    validate_framework_dependencies = subparsers.add_parser(
        "validate-agent-framework-dependencies",
        help="Validate dependency-gate contracts for planned framework adapters",
    )
    validate_framework_dependencies.add_argument("--contract", required=True)
    validate_framework_dependencies.add_argument("--output", required=True)
    validate_framework_dependencies.set_defaults(handler=cmd_validate_agent_framework_dependencies)


def cmd_baseline(args: argparse.Namespace) -> None:
    benchmark = load_benchmark(args.benchmark)
    result = validate_benchmark(benchmark)
    if result.errors:
        print(print_validation(result.errors, result.warnings))
        raise SystemExit(1)
    predictions = make_baseline(benchmark, args.kind)
    payload = localize_report_contract(
        asdict(predictions),
        output_path=args.output,
        project_root_hints=(args.benchmark,),
    )
    write_json(args.output, payload)
    print(f"Wrote {args.kind} predictions to {args.output}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    benchmark = load_benchmark(args.benchmark)
    predictions = load_predictions(args.predictions)
    bench_result = validate_benchmark(benchmark)
    pred_result = validate_predictions(predictions, benchmark)
    if bench_result.errors or pred_result.errors:
        print(print_validation(bench_result.errors, bench_result.warnings))
        print(print_validation(pred_result.errors, pred_result.warnings))
        raise SystemExit(1)
    report = Scorer(
        retrieval_k=args.retrieval_k,
        task_judge_plugin=load_task_judge_plugin(args.task_judge_plugin),
    ).score(benchmark, predictions)
    report = localize_report_contract(
        report,
        output_path=args.output,
        project_root_hints=(args.benchmark, args.predictions),
    )
    write_json(args.output, report)
    if not args.quiet:
        print(compact_summary(report))


def cmd_evaluate_release_baseline(args: argparse.Namespace) -> None:
    report = evaluate_release_split_baseline(
        args.manifest,
        split=args.split,
        baseline_kind=args.kind,
        retrieval_k=args.retrieval_k,
        baseline_top_k=args.baseline_top_k,
        task_judge_plugin=args.task_judge_plugin,
    )
    report = localize_report_contract(
        report,
        output_path=args.output,
        project_root_hints=(args.manifest,),
    )
    write_json(args.output, report)
    if not args.quiet:
        print(compact_summary(report))


def cmd_evaluate_release_predictions(args: argparse.Namespace) -> None:
    report = evaluate_release_split_predictions(
        args.manifest,
        args.predictions,
        split=args.split,
        retrieval_k=args.retrieval_k,
        run_metadata_path=args.run_metadata,
        framework_trace_path=args.framework_trace,
        task_judge_plugin=args.task_judge_plugin,
        case_ids=_read_case_ids_file(args.case_ids_file),
        domains=_parse_domains(args.domains),
    )
    report = localize_report_contract(
        report,
        output_path=args.output,
        project_root_hints=(args.manifest, args.predictions, args.run_metadata, args.framework_trace),
    )
    write_json(args.output, report)
    if not args.quiet:
        print(compact_summary(report))


def _load_cli_agent(config_path: str | Path, *, default_execution_mode: str = "integration_smoke") -> LoadedCliAgent:
    from amb.benchmark.integrations.factory import canonical_provider

    preflight_config = read_json(config_path)
    if isinstance(preflight_config, dict) and str(preflight_config.get("execution_mode") or "") == "dependency_preflight":
        raise SystemExit(
            "dependency_preflight configs are validation-only and cannot be used for scoring runs; "
            "run validate-integration-configs with --framework-contracts or a preflight audit instead"
        )
    try:
        agent, spec = load_agent_system(config_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config = spec.config
    config_artifact = artifact_info(Path(config_path))
    execution_mode = str(config.get("execution_mode") or default_execution_mode or spec.execution_mode)
    client_factory = str(config.get("client_factory") or config.get("loader") or "")
    provider = canonical_provider(str(config.get("provider", ""))) if "provider" in config else None
    try:
        real_system_attestation = build_real_system_attestation(
            config.get("real_system_attestation") if "real_system_attestation" in config else None,
            config_artifact,
            config_provider=provider,
            config_client_factory=client_factory or None,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if execution_mode == "real_system" and real_system_attestation is None:
        raise SystemExit("real_system execution_mode requires real_system_attestation")
    if execution_mode != "real_system" and real_system_attestation is not None:
        raise SystemExit("real_system_attestation requires execution_mode=real_system")
    if real_system_attestation is not None:
        metadata_errors = validate_real_system_run_metadata(
            {
                "execution_mode": execution_mode,
                "integration_config_artifact": config_artifact,
                "real_system_attestation": real_system_attestation,
            }
        )
        if metadata_errors:
            raise SystemExit(f"invalid real_system_attestation: {metadata_errors}")
    dependencies = dict(spec.dependencies)
    dependencies.setdefault("agent_system_config", str(config_path))
    return LoadedCliAgent(
        agent=agent,
        spec=spec,
        config_artifact=config_artifact,
        execution_mode=execution_mode,
        real_system_attestation=real_system_attestation,
        dependencies=dependencies,
        agent_system=agent_system_metadata(spec, config_path),
    )


def _write_json_atomic(path: str | Path, data: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        write_json(temp_path, data)
        temp_path.replace(target)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def cmd_scaffold_agent_system(args: argparse.Namespace) -> None:
    loader = str(args.loader)
    if ":" not in loader or not all(loader.split(":", 1)):
        raise SystemExit("loader must use 'module:callable' format")
    output = Path(args.output)
    adapter_output = Path(args.adapter_output) if args.adapter_output else None
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing config without --force: {output}")
    if adapter_output is not None and adapter_output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing adapter without --force: {adapter_output}")

    loader_kwargs = _parse_loader_kwargs(args.loader_kwarg or ())
    config = {
        "schema_version": "amst-agent-system-config-v1",
        "provider": str(args.provider or args.framework),
        "system_id": str(args.system_id),
        "system_version": str(args.system_version),
        "execution_mode": str(args.execution_mode),
        "agent_framework": str(args.framework),
        "agent_runtime": str(args.agent_runtime),
        "orchestration_mode": str(args.orchestration_mode),
        "memory_backend": str(args.memory_backend),
        "model_backend": str(args.model_backend),
        "tool_runtime_id": str(args.tool_runtime_id),
        "loader": loader,
        "loader_kwargs": loader_kwargs,
    }
    _write_json_atomic(output, config)
    if adapter_output is not None:
        adapter_output.parent.mkdir(parents=True, exist_ok=True)
        adapter_output.write_text(
            _agent_adapter_template(
                system_id=str(args.system_id),
                framework=str(args.framework),
                memory_backend=str(args.memory_backend),
                model_backend=str(args.model_backend),
                tool_runtime_id=str(args.tool_runtime_id),
                agent_runtime=str(args.agent_runtime),
                orchestration_mode=str(args.orchestration_mode),
            ),
            encoding="utf-8",
        )

    validation_report = write_integration_config_validation((output,), output.with_suffix(".validation.json"))
    summary = {
        "status": validation_report["status"],
        "config_output": str(output),
        "validation_output": str(output.with_suffix(".validation.json")),
        "adapter_output": str(adapter_output) if adapter_output is not None else None,
        "system_id": str(args.system_id),
        "loader": loader,
        "agent_framework": str(args.framework),
        "errors": validation_report["errors"],
        "next_commands": [
            (
                "python -m agent_memory_benchmark run-agent-experiment "
                f"--benchmark data/samples/mini_benchmark.json --config {output} "
                "--output-dir /tmp/amst_agent_smoke"
            ),
            (
                "python -m agent_memory_benchmark run-release-agent-matrix "
                "--manifest <release_manifest.json> --split public_dev "
                f"--configs {output} --output-dir /tmp/amst_agent_matrix --framework-traces"
            ),
        ],
    }
    write_json("-", summary) if False else print_validation_summary(summary)
    if validation_report["errors"]:
        raise SystemExit(1)


def _parse_loader_kwargs(items: tuple[str, ...] | list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        key, separator, value = str(item).partition("=")
        if not separator or not key:
            raise SystemExit("--loader-kwarg must use KEY=VALUE format")
        parsed[key] = value
    return parsed


def print_validation_summary(summary: dict[str, object]) -> None:
    for key in (
        "status",
        "config_output",
        "validation_output",
        "adapter_output",
        "system_id",
        "loader",
        "agent_framework",
        "errors",
        "next_commands",
    ):
        print(f"{key}: {summary[key]}")


def _agent_adapter_template(
    *,
    system_id: str,
    framework: str,
    memory_backend: str,
    model_backend: str,
    tool_runtime_id: str,
    agent_runtime: str,
    orchestration_mode: str,
) -> str:
    return f'''"""Starter AutoMemoryBench black-box adapter.

Replace the TODO sections with calls into your agent framework. Keep credentials
in environment variables; do not store API keys in this file or in loader_kwargs.
"""

from __future__ import annotations

from typing import Any


class BenchAgent:
    def __init__(
        self,
        *,
        system_id: str = "{system_id}",
        framework_id: str = "{framework}",
        memory_backend_id: str = "{memory_backend}",
        model_id: str = "{model_backend}",
        tool_runtime_id: str = "{tool_runtime_id}",
        framework_runtime: str = "{agent_runtime}",
        orchestration_mode: str = "{orchestration_mode}",
    ) -> None:
        self.system_id = system_id
        self.framework_id = framework_id
        self.memory_backend_id = memory_backend_id
        self.model_id = model_id
        self.tool_runtime_id = tool_runtime_id
        self.framework_runtime = framework_runtime
        self.orchestration_mode = orchestration_mode
        self._case_id: str | None = None
        self._namespace: str | None = None
        self._observations: list[dict[str, Any]] = []
        self._last_trace: dict[str, Any] = self._base_trace()

    def reset(self, case_id: str, namespace: str | None = None) -> None:
        self._case_id = case_id
        self._namespace = namespace
        self._observations = []
        self._last_trace = self._base_trace(namespace=namespace)
        # TODO: reset or namespace your framework memory for this case.

    def observe(self, observation: dict[str, Any]) -> None:
        self._observations.append(dict(observation))
        # TODO: ingest the chronological turn/event into your agent memory.

    def ingest_turn(self, turn: dict[str, Any]) -> None:
        self.observe(turn)

    def answer_or_act(self, probe: dict[str, Any]) -> dict[str, Any]:
        return self.run_probe(probe)

    def run_probe(self, probe: dict[str, Any], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        # TODO: query your agent and return benchmark-compatible prediction fields.
        query_id = str(probe.get("query_id", ""))
        prompt = str(probe.get("prompt", ""))
        buffer_memory_id = str(self._case_id or self.system_id) + ":observed_turn_buffer"
        memory_ops = [
            {{
                "operation": "read",
                "memory_id": buffer_memory_id,
                "source_memory_ids": [],
            }}
        ] if self._observations else []
        planner_trace = [
            {{
                "step": "adapter_template_placeholder",
                "decision": "replace_with_framework_call",
                "executor": self.framework_id,
                "input_memory_ids": [buffer_memory_id] if self._observations else [],
                "output_memory_ids": [],
                "tool_call_ids": [],
            }}
        ]
        cost = {{
            "input_tokens": len(prompt.split()),
            "output_tokens": 0,
            "latency_ms": 0.0,
            "memory_op_count": len(memory_ops),
            "tool_call_count": 0,
        }}
        self._last_trace = dict(
            self._base_trace(),
            memory_ops=memory_ops,
            planner_trace=planner_trace,
            cost=cost,
            framework_state={{
                "adapter_status": "template_unimplemented",
                "case_id": self._case_id,
                "num_observations": len(self._observations),
            }},
        )
        return {{
            "query_id": query_id,
            "memory_needed": None,
            "activated_memory_ids": [],
            "response": "",
            "compression_summary": None,
            "tool_name": None,
            "parameters": {{
                "adapter_status": "template_unimplemented",
                "case_id": self._case_id,
                "num_observations": len(self._observations),
            }},
            "memory_operations": [],
            "retrieval_hits": [],
            "tool_calls": [],
            "planner_trace": planner_trace,
            "cost": {{}},
        }}

    def export_memory(self) -> list[dict[str, Any]]:
        return [
            {{
                "memory_id": str(self._case_id or self.system_id) + ":observed_turn_buffer",
                "namespace": [self.framework_id, self._namespace or self._case_id or self.system_id],
                "content": str(len(self._observations)) + " observations buffered",
                "metadata": {{"adapter_status": "template_unimplemented"}},
            }}
        ]

    def export_trace(self) -> dict[str, Any]:
        return dict(self._last_trace)

    def export_tool_calls(self) -> list[dict[str, Any]]:
        return list(self._last_trace.get("tool_calls", []))

    def export_framework_state(self) -> dict[str, Any]:
        return dict(self._last_trace.get("framework_state", {{}}))

    def _base_trace(self, *, namespace: str | None = None) -> dict[str, Any]:
        return {{
            "framework_id": self.framework_id,
            "framework_version": "adapter_template_v1",
            "framework_runtime": self.framework_runtime,
            "orchestration_mode": self.orchestration_mode,
            "model_id": self.model_id,
            "memory_backend_id": self.memory_backend_id,
            "tool_runtime_id": self.tool_runtime_id,
            "session_id": self._case_id or "",
            "namespace": [self.framework_id, namespace or self._namespace or self._case_id or self.system_id],
            "message_history_policy": "adapter_template_placeholder",
            "memory_ops": [],
            "retrieval_hits": [],
            "tool_calls": [],
            "planner_trace": [],
            "handoff_trace": [],
            "cost": {{}},
            "framework_state": {{}},
        }}


def create_adapter(**kwargs: Any) -> BenchAgent:
    return BenchAgent(**kwargs)
'''


def cmd_run_agent(args: argparse.Namespace) -> None:
    benchmark = load_benchmark(args.benchmark)
    result = validate_benchmark(benchmark)
    if result.errors:
        print(print_validation(result.errors, result.warnings))
        raise SystemExit(1)
    loaded = _load_cli_agent(args.config)
    agent = loaded.agent
    system_id = args.system_id or loaded.spec.system_id
    system_version = str(loaded.spec.config.get("system_version") or loaded.spec.system_version)
    agent_system = bind_agent_system_metadata(
        loaded.agent_system,
        system_id=system_id,
        system_version=system_version,
        execution_mode=loaded.execution_mode,
    )
    framework_traces: list[dict[str, object]] = []
    run_id = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    predictions = run_black_box_agent(
        benchmark,
        agent,
        system_id,
        framework_trace_callback=framework_traces.append if args.framework_trace_output else None,
        agent_system=agent_system,
    )
    pred_result = validate_predictions(predictions, benchmark)
    if pred_result.errors:
        print(print_validation(pred_result.errors, pred_result.warnings))
        raise SystemExit(1)
    payload = localize_report_contract(
        asdict(predictions),
        output_path=args.output,
        project_root_hints=(args.benchmark, args.config),
    )
    write_json(args.output, payload)
    if args.framework_trace_output:
        trace_payload = framework_trace_artifact_payload(
            framework_traces=framework_traces,
            system_id=system_id,
            benchmark_id=benchmark.benchmark_id,
            release_split="ad_hoc",
            run_id=run_id,
            release_contract_fingerprint=None,
        )
        trace_errors = validate_framework_trace_artifact_payload(
            trace_payload,
            expected_records=len(predictions.predictions),
            require_envelope=True,
            tool_runtime_contracts=load_default_tool_runtime_contracts(),
        )
        if trace_errors:
            print(print_validation(trace_errors, ()))
            raise SystemExit(1)
        write_json(args.framework_trace_output, trace_payload)
    print(f"system_id: {predictions.system_id}")
    print(f"predictions: {len(predictions.predictions)}")
    print(f"output: {args.output}")
    if args.framework_trace_output:
        print(f"framework_trace: {args.framework_trace_output}")


def cmd_run_agent_experiment(args: argparse.Namespace) -> None:
    benchmark = load_benchmark(args.benchmark)
    result = validate_benchmark(benchmark)
    if result.errors:
        print(print_validation(result.errors, result.warnings))
        raise SystemExit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = _load_cli_agent(args.config)
    config = loaded.spec.config
    agent = loaded.agent
    system_id = args.system_id or loaded.spec.system_id
    system_version = str(config.get("system_version") or args.system_version)
    agent_system = bind_agent_system_metadata(
        loaded.agent_system,
        system_id=system_id,
        system_version=system_version,
        execution_mode=loaded.execution_mode,
    )
    framework_traces: list[dict[str, object]] = []
    run_id = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    predictions = run_black_box_agent(
        benchmark,
        agent,
        system_id,
        framework_trace_callback=framework_traces.append if args.framework_trace_output else None,
        agent_system=agent_system,
    )
    pred_result = validate_predictions(predictions, benchmark)
    if pred_result.errors:
        print(print_validation(pred_result.errors, pred_result.warnings))
        raise SystemExit(1)

    predictions_path = output_dir / "predictions.json"
    predictions_payload = localize_report_contract(
        asdict(predictions),
        output_path=predictions_path,
        project_root_hints=(args.benchmark, args.config),
    )
    write_json(predictions_path, predictions_payload)
    if args.framework_trace_output:
        benchmark_id = args.benchmark_id or benchmark.benchmark_id
        trace_payload = framework_trace_artifact_payload(
            framework_traces,
            system_id=system_id,
            benchmark_id=benchmark_id,
            release_split=args.release_split,
            run_id=run_id,
            release_contract_fingerprint=None,
        )
        trace_errors = validate_framework_trace_artifact_payload(
            trace_payload,
            expected_records=len(predictions.predictions),
            require_envelope=True,
            tool_runtime_contracts=load_default_tool_runtime_contracts(),
        )
        if trace_errors:
            print(print_validation(trace_errors, ()))
            raise SystemExit(1)
        trace_payload = localize_report_contract(
            trace_payload,
            output_path=args.framework_trace_output,
            project_root_hints=(args.benchmark, args.config, predictions_path),
        )
        write_json(args.framework_trace_output, trace_payload)

    benchmark_id = args.benchmark_id or benchmark.benchmark_id
    metadata = build_run_metadata(
        system_id=system_id,
        system_version=system_version,
        benchmark_id=benchmark_id,
        release_split=args.release_split,
        command=(
            "python -m agent_memory_benchmark run-agent-experiment "
            f"--benchmark {args.benchmark} --config {args.config} --output-dir {args.output_dir} "
            f"--retrieval-k {args.retrieval_k} --task-judge-plugin {args.task_judge_plugin}"
        ),
        execution_mode=loaded.execution_mode,
        integration_config_artifact=loaded.config_artifact,
        real_system_attestation=loaded.real_system_attestation,
        dependencies=loaded.dependencies,
        agent_system=agent_system,
        timestamp=run_id,
    )
    metadata_path = output_dir / "run_metadata.json"
    metadata_payload = localize_report_contract(
        metadata,
        output_path=metadata_path,
        project_root_hints=(args.benchmark, args.config, predictions_path),
    )
    write_json(metadata_path, metadata_payload)

    report = Scorer(
        retrieval_k=args.retrieval_k,
        task_judge_plugin=load_task_judge_plugin(args.task_judge_plugin),
    ).score(benchmark, predictions)
    report["benchmark_id"] = benchmark_id
    report["release_split"] = args.release_split
    report["submission"] = {
        "prediction_file": str(predictions_path),
        "prediction_artifact": artifact_info(predictions_path),
        "manifest_artifact": artifact_info(Path(args.benchmark)),
        "run_metadata_file": str(metadata_path),
        "run_metadata_artifact": run_metadata_artifact(metadata_path),
        "scoring_config": report["scoring_config"],
    }
    if args.framework_trace_output:
        framework_trace_path = Path(args.framework_trace_output)
        report["submission"]["framework_trace_file"] = str(framework_trace_path)
        report["submission"]["framework_trace_artifact"] = artifact_info(framework_trace_path)
    report["run_metadata"] = metadata
    report_path = output_dir / "report.json"
    report = localize_report_contract(
        report,
        output_path=report_path,
        project_root_hints=(
            args.benchmark,
            args.config,
            predictions_path,
            metadata_path,
            args.framework_trace_output,
        ),
    )
    write_json(report_path, report)

    print(f"system_id: {system_id}")
    print(f"benchmark_id: {benchmark_id}")
    print(f"release_split: {args.release_split}")
    print(f"predictions: {len(predictions.predictions)}")
    print(f"predictions_output: {predictions_path}")
    print(f"run_metadata_output: {metadata_path}")
    print(f"report_output: {report_path}")
    if args.framework_trace_output:
        print(f"framework_trace_output: {args.framework_trace_output}")


def cmd_run_foundation_model(args: argparse.Namespace) -> None:
    if bool(args.benchmark) == bool(args.manifest):
        raise SystemExit("exactly one of --benchmark or --manifest must be provided")
    if args.manifest and not args.split:
        raise SystemExit("--split is required when using --manifest")
    require_env_var_name(args.api_key_env)
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key: set {args.api_key_env} or pass --api-key-env with an environment variable name")
    client = OpenAICompatibleChatClient(
        base_url=args.base_url,
        api_key=api_key,
        timeout_s=args.timeout_s,
    )
    system_id = args.system_id or _default_foundation_system_id(args.model, args.protocol)
    if args.benchmark:
        benchmark = load_benchmark(args.benchmark)
        result = validate_benchmark(benchmark)
        if result.errors:
            print(print_validation(result.errors, result.warnings))
            raise SystemExit(1)
        predictions = run_foundation_model(
            benchmark,
            client=client,
            model=args.model,
            protocol=args.protocol,
            system_id=system_id,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        pred_result = validate_predictions(predictions, benchmark)
        if pred_result.errors:
            print(print_validation(pred_result.errors, pred_result.warnings))
            raise SystemExit(1)
    else:
        predictions = run_foundation_model_release_split(
            args.manifest,
            split=args.split,
            client=client,
            model=args.model,
            protocol=args.protocol,
            system_id=system_id,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    payload = localize_report_contract(
        prediction_set_to_dict(predictions),
        output_path=args.output,
        project_root_hints=(args.benchmark, args.manifest),
    )
    write_json(args.output, payload)
    print(f"system_id: {predictions.system_id}")
    if args.benchmark:
        print(f"benchmark: {args.benchmark}")
    else:
        print(f"split: {args.split}")
        print(f"manifest: {args.manifest}")
    print(f"protocol: {args.protocol}")
    print(f"model: {args.model}")
    print(f"predictions: {len(predictions.predictions)}")
    print(f"output: {args.output}")


def cmd_run_release_agent(args: argparse.Namespace) -> None:
    loaded = _load_cli_agent(args.config, default_execution_mode=args.execution_mode)
    config = loaded.spec.config
    agent = loaded.agent
    system_id = args.system_id or loaded.spec.system_id
    system_version = str(config.get("system_version") or args.system_version)
    agent_system = bind_agent_system_metadata(
        loaded.agent_system,
        system_id=system_id,
        system_version=system_version,
        execution_mode=loaded.execution_mode,
    )
    framework_traces: list[dict[str, object]] = []
    run_id = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    release_contract = release_split_contract_fingerprint(args.manifest, args.split)
    benchmark_id = release_split_benchmark_id(args.manifest, args.split)
    loaded_framework_trace_checkpoint = False
    selected_case_ids = _read_case_ids_file(args.case_ids_file)
    selected_domains = _parse_domains(args.domains)

    def _write_framework_trace_checkpoint() -> None:
        if not args.framework_trace_output:
            return
        trace_payload = framework_trace_artifact_payload(
            framework_traces,
            system_id=system_id,
            benchmark_id=benchmark_id,
            release_split=args.split,
            run_id=run_id,
            release_contract_fingerprint=release_contract,
        )
        trace_payload = localize_report_contract(
            trace_payload,
            output_path=args.framework_trace_output,
            project_root_hints=(args.manifest, args.config, args.output),
        )
        _write_json_atomic(args.framework_trace_output, trace_payload)

    if args.framework_trace_output and args.resume and Path(args.output).exists():
        resume_predictions = load_predictions(args.output)
        resume_prediction_count = len(resume_predictions.predictions)
        trace_path = Path(args.framework_trace_output)
        if resume_prediction_count and not trace_path.exists():
            raise SystemExit(
                "cannot resume with framework traces: predictions checkpoint exists "
                f"({resume_prediction_count} rows) but framework trace checkpoint is missing; "
                "remove the partial predictions or rerun without --resume"
            )
        if trace_path.exists():
            trace_payload = read_json(trace_path)
            existing_traces = trace_payload.get("framework_traces")
            if not isinstance(existing_traces, list):
                raise SystemExit("cannot resume with framework traces: existing framework_trace.json is invalid")
            expected_query_ids = [item.query_id for item in resume_predictions.predictions]
            existing_query_ids = [str(item.get("query_id") or "") for item in existing_traces]
            try:
                safe_trace_count = _safe_resume_framework_trace_count(
                    args.manifest,
                    split=args.split,
                    prediction_query_ids=expected_query_ids,
                    trace_query_ids=existing_query_ids,
                    case_ids=selected_case_ids,
                    domains=selected_domains,
                )
            except ValueError as exc:
                raise SystemExit(f"cannot resume with framework traces: {exc}") from exc
            framework_traces.extend(existing_traces[:safe_trace_count])
            loaded_framework_trace_checkpoint = bool(safe_trace_count)
            if len(existing_traces) != safe_trace_count:
                _write_framework_trace_checkpoint()

    def _record_framework_trace(record: dict[str, object]) -> None:
        framework_traces.append(record)
        # Keep trace checkpoints close to prediction checkpoints. Chunked
        # release runs set this to 1 so best-available merging never consumes
        # predictions whose corresponding trace rows are still only in memory.
        checkpoint_every = max(1, int(os.environ.get("AMB_FRAMEWORK_TRACE_CHECKPOINT_EVERY", "12")))
        if args.resume and len(framework_traces) % checkpoint_every == 0:
            _write_framework_trace_checkpoint()

    def _progress(item: dict[str, object]) -> None:
        status = "resume-skip" if item.get("resumed") else "completed"
        print(
            f"[progress] {system_id} {status} case={item.get('case_id')} "
            f"cases={item.get('completed_cases')}/{item.get('total_cases')} "
            f"queries={item.get('num_predictions')}/{item.get('total_queries')}",
            flush=True,
        )
    predictions = run_release_split_agent(
        args.manifest,
        split=args.split,
        agent=agent,
        system_id=system_id,
        resume_path=args.output if args.resume else None,
        checkpoint_path=args.output if args.resume else None,
        progress_callback=_progress if args.resume else None,
        framework_trace_callback=_record_framework_trace if args.framework_trace_output else None,
        emit_resume_traces=not loaded_framework_trace_checkpoint,
        agent_system=agent_system,
        case_ids=selected_case_ids,
        domains=selected_domains,
        failure_predictions=args.failure_predictions,
    )
    predictions_payload = localize_report_contract(
        asdict(predictions),
        output_path=args.output,
        project_root_hints=(args.manifest, args.config),
    )
    write_json(args.output, predictions_payload)
    if args.framework_trace_output:
        trace_payload = framework_trace_artifact_payload(
            framework_traces,
            system_id=system_id,
            benchmark_id=benchmark_id,
            release_split=args.split,
            run_id=run_id,
            release_contract_fingerprint=release_contract,
        )
        trace_errors = validate_framework_trace_artifact_payload(
            trace_payload,
            expected_records=len(predictions.predictions),
            require_envelope=True,
            tool_runtime_contracts=load_default_tool_runtime_contracts(),
        )
        if trace_errors:
            print(print_validation(trace_errors, ()))
            raise SystemExit(1)
        trace_payload = localize_report_contract(
            trace_payload,
            output_path=args.framework_trace_output,
            project_root_hints=(args.manifest, args.config, args.output),
        )
        write_json(args.framework_trace_output, trace_payload)
    if args.run_metadata_output:
        metadata = build_run_metadata(
            system_id=system_id,
            system_version=system_version,
            benchmark_id=benchmark_id,
            release_split=args.split,
            command=_release_agent_command(args),
            execution_mode=loaded.execution_mode,
            integration_config_artifact=loaded.config_artifact,
            real_system_attestation=loaded.real_system_attestation,
            release_contract_fingerprint=release_contract,
            dependencies=loaded.dependencies,
            agent_system=agent_system,
            timestamp=run_id,
        )
        metadata_payload = localize_report_contract(
            metadata,
            output_path=args.run_metadata_output,
            project_root_hints=(args.manifest, args.config, args.output),
        )
        write_json(args.run_metadata_output, metadata_payload)
    print(f"system_id: {predictions.system_id}")
    print(f"split: {args.split}")
    print(f"predictions: {len(predictions.predictions)}")
    print(f"output: {args.output}")
    if args.run_metadata_output:
        print(f"run_metadata: {args.run_metadata_output}")
    if args.framework_trace_output:
        print(f"framework_trace: {args.framework_trace_output}")
    _force_exit_after_release_agent_if_requested()


def _force_exit_after_release_agent_if_requested() -> None:
    """Terminate non-daemon provider threads after all run-release-agent artifacts are durable."""

    raw = os.environ.get("AMB_FORCE_EXIT_AFTER_RUN_RELEASE_AGENT", "")
    if raw.strip().lower() not in {"1", "true", "yes"}:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def cmd_run_release_agent_matrix(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    rows = []
    seen_system_ids: set[str] = set()
    for config_path in args.configs:
        loaded = _load_cli_agent(config_path)
        config = loaded.spec.config
        system_id = loaded.spec.system_id
        system_version = str(config.get("system_version") or args.system_version)
        agent_system = bind_agent_system_metadata(
            loaded.agent_system,
            system_id=system_id,
            system_version=system_version,
            execution_mode=loaded.execution_mode,
        )
        if system_id in seen_system_ids:
            raise SystemExit(f"duplicate system_id in integration matrix: {system_id}")
        seen_system_ids.add(system_id)

        def _progress(item: dict[str, object]) -> None:
            status = "resume-skip" if item.get("resumed") else "completed"
            print(
                f"[progress] {system_id} {status} case={item.get('case_id')} "
                f"cases={item.get('completed_cases')}/{item.get('total_cases')} "
                f"queries={item.get('num_predictions')}/{item.get('total_queries')}",
                flush=True,
            )

        def _retry(item: dict[str, object]) -> None:
            attempt = item.get("attempt")
            max_attempts = item.get("max_run_attempts")
            error_type = item.get("error_type")
            error = item.get("error")
            if item.get("will_retry"):
                print(
                    f"[retry] {system_id} attempt={attempt}/{max_attempts} "
                    f"error_type={error_type} error={error}",
                    flush=True,
                )
            else:
                print(
                    f"[retry] {system_id} exhausted attempt={attempt}/{max_attempts} "
                    f"error_type={error_type} error={error}",
                    flush=True,
                )

        row = run_release_split_agent_experiment_with_retries(
            args.manifest,
            split=args.split,
            agent_factory=lambda config_path=config_path: load_agent_system(config_path)[0],
            system_id=system_id,
            output_dir=output_dir,
            system_version=system_version,
            retrieval_k=args.retrieval_k,
            task_judge_plugin=args.task_judge_plugin,
            command=(
                "python -m agent_memory_benchmark run-release-agent-matrix "
                f"--manifest {args.manifest} --split {args.split} --configs {' '.join(args.configs)} "
                f"--output-dir {args.output_dir} --system-version {args.system_version} "
                f"--retrieval-k {args.retrieval_k} --task-judge-plugin {args.task_judge_plugin}"
                f"{' --resume' if args.resume else ''} "
                f"{' --framework-traces' if args.framework_traces else ''} "
                f"--max-run-attempts {args.max_run_attempts} --retry-backoff-s {args.retry_backoff_s}"
            ),
            dependencies=loaded.dependencies,
            execution_mode=loaded.execution_mode,
            integration_config_artifact=loaded.config_artifact,
            real_system_attestation=loaded.real_system_attestation,
            agent_system=agent_system,
            framework_trace_path=(output_dir / system_id / "framework_trace.json") if args.framework_traces else None,
            resume=args.resume,
            progress_callback=_progress if (args.resume or args.max_run_attempts > 1) else None,
            retry_callback=_retry if args.max_run_attempts > 1 else None,
            max_run_attempts=args.max_run_attempts,
            retry_backoff_s=args.retry_backoff_s,
        )
        rows.append(row)
        print(f"{system_id}: report={row['report_artifact']['path']} attempts={row.get('execution_attempts', 1)}")

    summary = {
        "schema_version": "amst-release-agent-matrix-v1",
        "release_manifest": str(args.manifest),
        "release_manifest_artifact": artifact_info(Path(args.manifest)),
        "release_split": args.split,
        "benchmark_id": release_split_benchmark_id(args.manifest, args.split),
        "num_systems": len(rows),
        "systems": rows,
    }
    summary_path = output_dir / "matrix_summary.json"
    summary = localize_report_contract(
        summary,
        output_path=summary_path,
        project_root_hints=(args.manifest, *args.configs, output_dir),
    )
    write_json(summary_path, summary)
    print(f"systems: {len(rows)}")
    print(f"summary: {summary_path}")


def cmd_validate_integration_configs(args: argparse.Namespace) -> None:
    report = write_integration_config_validation(
        args.configs,
        args.output,
        framework_contract_paths=args.framework_contracts,
    )
    print(f"status: {report['status']}")
    print(f"configs: {report['num_configs']}")
    print(f"framework_contracts: {report.get('num_framework_contracts', 0)}")
    print(f"errors: {len(report['errors'])}")
    if report["errors"]:
        raise SystemExit(1)


def cmd_validate_agent_framework_contracts(args: argparse.Namespace) -> None:
    report = write_framework_adapter_contract_validation(args.contracts, args.output)
    print(f"status: {report['status']}")
    print(f"contracts: {report['num_contracts']}")
    print(f"errors: {len(report['errors'])}")
    if report["errors"]:
        raise SystemExit(1)


def cmd_validate_tool_runtime_contract(args: argparse.Namespace) -> None:
    report = write_tool_runtime_contract_validation(args.contract, args.output)
    print(f"status: {report['status']}")
    print(f"tool_runtime_id: {report.get('tool_runtime_id')}")
    print(f"errors: {len(report['errors'])}")
    if report["errors"]:
        raise SystemExit(1)


def cmd_validate_agent_framework_dependencies(args: argparse.Namespace) -> None:
    report = write_dependency_gate_contract_validation(args.contract, args.output)
    print(f"status: {report['status']}")
    print(f"dependency_gates: {report['num_dependency_gate_rows']}")
    print(f"errors: {len(report['errors'])}")
    if report["errors"]:
        raise SystemExit(1)


def _default_foundation_system_id(model: str, protocol: str) -> str:
    safe_model = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in model)
    return f"{safe_model}:{protocol}"


def _read_case_ids_file(path: str | None) -> set[str] | None:
    if not path:
        return None
    case_ids = {line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()}
    if not case_ids:
        raise SystemExit(f"--case-ids-file is empty: {path}")
    return case_ids


def _parse_domains(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    domains = {part.strip() for value in values for part in value.split(",") if part.strip()}
    if not domains:
        raise SystemExit("--domains did not contain any non-empty domain names")
    return domains


def _safe_resume_framework_trace_count(
    manifest_path: str | Path,
    *,
    split: str,
    prediction_query_ids: list[str],
    trace_query_ids: list[str],
    case_ids: set[str] | None,
    domains: set[str] | None,
) -> int:
    """Return the trace prefix that is safe to preload before resume.

    Release-agent resume skips whole cases only. If a checkpoint contains a
    partial case, the runner will rerun that entire case, so preloading trace
    rows from the partial case would duplicate query traces. Keep only the
    common manifest/prediction/trace prefix ending at a complete case boundary.
    """

    query_order, case_boundaries = _release_split_query_order_and_boundaries(
        manifest_path,
        split=split,
        case_ids=case_ids,
        domains=domains,
    )
    if not prediction_query_ids:
        return 0
    if prediction_query_ids != query_order[: len(prediction_query_ids)]:
        raise ValueError("existing predictions are not a prefix of the selected release split query order")
    completed_prediction_boundaries = [boundary for boundary in case_boundaries if boundary <= len(prediction_query_ids)]
    completed_prediction_count = max(completed_prediction_boundaries) if completed_prediction_boundaries else 0
    limit = min(len(prediction_query_ids), len(trace_query_ids), len(query_order))
    common_prefix = 0
    for index in range(limit):
        if prediction_query_ids[index] == trace_query_ids[index] == query_order[index]:
            common_prefix = index + 1
        else:
            break
    safe_boundaries = [boundary for boundary in case_boundaries if boundary <= common_prefix]
    safe_count = max(safe_boundaries) if safe_boundaries else 0
    if safe_count == completed_prediction_count:
        return safe_count
    # The existing trace does not cover every case that resume will skip.
    # Preloading a shorter prefix would suppress trace reconstruction for the
    # remaining skipped cases, so let the runner rebuild all skipped traces from
    # the prediction checkpoint instead.
    return 0


def _release_split_query_order_and_boundaries(
    manifest_path: str | Path,
    *,
    split: str,
    case_ids: set[str] | None,
    domains: set[str] | None,
) -> tuple[list[str], list[int]]:
    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    split_files = manifest.get("split_files") if isinstance(manifest.get("split_files"), dict) else {}
    entries = _manifest_split_entries(split_files.get(split))
    query_order: list[str] = []
    case_boundaries: list[int] = []
    for _, raw_path in entries:
        benchmark = load_benchmark(_resolve_manifest_path(str(raw_path), manifest_file.parent))
        for case in benchmark.cases:
            if case_ids is not None and case.case_id not in case_ids:
                continue
            if domains is not None and case.domain not in domains:
                continue
            query_order.extend(query.query_id for query in case.queries)
            case_boundaries.append(len(query_order))
    return query_order, case_boundaries


def _manifest_split_entries(value: object) -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [("benchmark", value)]
    if isinstance(value, dict):
        return [(str(label), str(path)) for label, path in sorted(value.items())]
    return []


def _resolve_manifest_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return manifest_dir / path


def _release_agent_command(args: argparse.Namespace) -> str:
    parts = [
        "python",
        "-m",
        "agent_memory_benchmark",
        "run-release-agent",
        "--manifest",
        str(args.manifest),
        "--split",
        str(args.split),
        "--config",
        str(args.config),
        "--output",
        str(args.output),
    ]
    if args.case_ids_file:
        parts.extend(["--case-ids-file", str(args.case_ids_file)])
    if args.domains:
        parts.extend(["--domains", *[str(value) for value in args.domains]])
    return " ".join(parts)
