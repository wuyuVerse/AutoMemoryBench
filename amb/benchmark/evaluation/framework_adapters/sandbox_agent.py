"""Sandbox-agent adapter for command/SDK/API based memory controllers.

The adapter materializes one auditable, gold-free workspace per benchmark query
and requires the external agent to return the normalized AMB JSON payload.
"""

from __future__ import annotations

import importlib
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
from typing import Any

from amb.benchmark.schemas.models import Case


SANDBOX_LAYOUT_VERSION = "amb-sandbox-agent-layout-v1"
SUPPORTED_OUTPUT_MODES = {"command_output_file", "command_stdout_json", "python_sdk", "remote_api"}


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["response", "activated_memory_ids"],
    "properties": {
        "response": {"type": "string"},
        "memory_needed": {"type": ["boolean", "null"]},
        "activated_memory_ids": {"type": "array", "items": {"type": "string"}},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "source_path": {"type": "string"},
                    "line": {"type": "integer"},
                },
            },
        },
        "memory_operations": {"type": "array"},
        "tool_calls": {"type": "array"},
        "cost": {"type": "object"},
    },
    "additionalProperties": True,
}


class SandboxAgentAdapter:
    """Reusable adapter for official sandbox/coding-agent surfaces."""

    def __init__(
        self,
        *,
        system_id: str,
        framework_id: str = "sandbox_agent",
        framework_label: str = "Sandbox Agent",
        agent_family: str = "sandbox_agent_memory_controller",
        official_surface: str = "",
        binary_or_sdk: str = "",
        output_mode: str = "command_output_file",
        command_template: str | list[str] | None = None,
        sdk_runner: str | None = None,
        remote_api_runner: str | None = None,
        workspace_root: str | Path = "reports/runtime_state/sandbox_agents/workspaces",
        output_file: str = "agent_output.json",
        timeout_s: float = 120.0,
        max_retries: int = 0,
        retain_workspaces: bool = True,
        version_command: str | list[str] | None = None,
        required_env_vars: list[str] | tuple[str, ...] = (),
        model_id: str = "unspecified",
        provider: str = "unspecified",
        sandbox_layout_version: str = SANDBOX_LAYOUT_VERSION,
        claim_boundary: str = "",
        extra_env: dict[str, str] | None = None,
        allow_nonzero_with_output: bool = False,
    ) -> None:
        if output_mode not in SUPPORTED_OUTPUT_MODES:
            raise ValueError(f"unsupported sandbox output_mode {output_mode!r}")
        if output_mode.startswith("command") and not command_template:
            raise ValueError(f"{output_mode} requires command_template")
        if output_mode == "python_sdk" and not sdk_runner:
            raise ValueError("python_sdk mode requires sdk_runner='module:callable'")
        if output_mode == "remote_api" and not remote_api_runner:
            raise ValueError("remote_api mode requires remote_api_runner='module:callable'")

        self.system_id = system_id
        self.framework_id = framework_id
        self.framework_label = framework_label
        self.agent_family = agent_family
        self.official_surface = official_surface
        self.binary_or_sdk = binary_or_sdk
        self.output_mode = output_mode
        self.command_template = command_template
        self.sdk_runner = sdk_runner
        self.remote_api_runner = remote_api_runner
        default_workspace_root = "reports/runtime_state/sandbox_agents/workspaces"
        env_workspace_root = os.environ.get("AMB_SANDBOX_AGENT_WORKSPACE_ROOT")
        if env_workspace_root and str(workspace_root) == default_workspace_root:
            workspace_root = env_workspace_root
        self.workspace_root = Path(workspace_root)
        self.output_file = output_file
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.retain_workspaces = bool(retain_workspaces)
        self.version_command = version_command
        self.required_env_vars = tuple(str(item) for item in required_env_vars)
        self.model_id = model_id
        self.provider = provider
        self.sandbox_layout_version = sandbox_layout_version
        self.claim_boundary = claim_boundary
        self.extra_env = dict(extra_env or {})
        self.allow_nonzero_with_output = bool(allow_nonzero_with_output)

        self.case_id = ""
        self._case: Case | None = None
        self._observations: list[dict[str, Any]] = []
        self._last_trace: dict[str, Any] = self._base_trace()

    def set_case_reference(self, case: Case) -> None:
        self._case = case

    def reset(self, case_id: str, namespace: str | None = None) -> None:
        self.case_id = case_id
        self._observations = []
        self._last_trace = self._base_trace(namespace=namespace)

    def observe(self, observation: dict[str, Any]) -> None:
        self._observations.append(dict(observation))

    def ingest_turn(self, turn: dict[str, Any]) -> None:
        self.observe(turn)

    def answer_or_act(self, probe: dict[str, Any]) -> dict[str, Any]:
        return self.run_probe(probe)

    def run_probe(self, probe: dict[str, Any], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        query_id = str(probe.get("query_id") or "query")
        sandbox_dir = self._sandbox_dir(query_id)
        if sandbox_dir.exists() and not self.retain_workspaces:
            shutil.rmtree(sandbox_dir)
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        visible_memory_ids = self._write_sandbox(sandbox_dir, probe)
        env_status = {name: bool(os.environ.get(name)) for name in self.required_env_vars}
        missing_env = [name for name, present in env_status.items() if not present]
        if missing_env:
            raise RuntimeError(
                f"sandbox agent {self.system_id} missing required environment variables: {', '.join(missing_env)}"
            )

        start = time.monotonic()
        attempts: list[dict[str, Any]] = []
        last_error: str | None = None
        last_attempt_trace: dict[str, Any] = {}
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._run_once(sandbox_dir=sandbox_dir, probe=probe, attempt=attempt)
                attempt_trace = dict(self._last_trace)
                normalized = self._validate_output(raw, visible_memory_ids=visible_memory_ids)
                latency_ms = (time.monotonic() - start) * 1000.0
                normalized.setdefault("memory_needed", bool(normalized["activated_memory_ids"]))
                normalized.setdefault("evidence", [])
                normalized.setdefault("memory_operations", [])
                normalized.setdefault("tool_calls", [])
                normalized.setdefault("cost", {})
                normalized["cost"] = dict(normalized["cost"] or {})
                normalized["cost"].setdefault("latency_ms", latency_ms)
                normalized["parameters"] = dict(normalized.get("parameters", {}))
                normalized["parameters"].setdefault("sandbox_dir", str(sandbox_dir))
                normalized["parameters"].setdefault("sandbox_layout_version", self.sandbox_layout_version)
                self._last_trace = {**self._base_trace(), **attempt_trace}
                self._last_trace.update(
                    {
                        "case_id": self.case_id,
                        "query_id": query_id,
                        "sandbox_dir": str(sandbox_dir),
                        "output_file": str(sandbox_dir / self.output_file),
                        "output_mode": self.output_mode,
                        "attempts": attempts + [{"attempt": attempt, "status": "passed"}],
                        "env_presence": env_status,
                        "version": self._safe_version(),
                        "cost": normalized["cost"],
                        "framework_state": {
                            "visible_memory_ids": sorted(visible_memory_ids),
                            "observations_seen": len(self._observations),
                            "schema_validation": "passed",
                        },
                    }
                )
                return normalized
            except Exception as exc:
                last_error = str(exc)
                last_attempt_trace = dict(self._last_trace)
                attempts.append({"attempt": attempt, "status": "failed", "error": last_error})
                if attempt >= self.max_retries:
                    break
        self._last_trace = {**self._base_trace(), **last_attempt_trace}
        self._last_trace.update(
            {
                "case_id": self.case_id,
                "query_id": query_id,
                "sandbox_dir": str(sandbox_dir),
                "output_file": str(sandbox_dir / self.output_file),
                "output_mode": self.output_mode,
                "attempts": attempts,
                "env_presence": env_status,
                "version": self._safe_version(),
                "framework_state": {"schema_validation": "failed", "last_error": last_error},
            }
        )
        raise RuntimeError(f"sandbox agent {self.system_id} failed query {query_id}: {last_error}")

    def export_memory(self) -> list[dict[str, Any]]:
        if self._case is None:
            return []
        return [self._public_memory(memory) for memory in self._case.gold_memory_units]

    def export_trace(self) -> dict[str, Any]:
        return dict(self._last_trace)

    def export_tool_calls(self) -> list[dict[str, Any]]:
        return list(self._last_trace.get("tool_calls", []))

    def export_framework_state(self) -> dict[str, Any]:
        return dict(self._last_trace.get("framework_state", {}))

    def _run_once(self, *, sandbox_dir: Path, probe: dict[str, Any], attempt: int) -> dict[str, Any]:
        if self.output_mode in {"command_output_file", "command_stdout_json"}:
            return self._run_command_once(sandbox_dir=sandbox_dir, probe=probe, attempt=attempt)
        runner_ref = self.sdk_runner if self.output_mode == "python_sdk" else self.remote_api_runner
        runner = _load_callable(str(runner_ref))
        started = time.monotonic()
        result = runner(
            sandbox_dir=str(sandbox_dir),
            output_file=str(sandbox_dir / self.output_file),
            instruction_file=str(sandbox_dir / "INSTRUCTION.md"),
            query_file=str(sandbox_dir / "query.json"),
            output_schema_file=str(sandbox_dir / "output_schema.json"),
            system_id=self.system_id,
            model_id=self.model_id,
            provider=self.provider,
            attempt=attempt,
        )
        self._last_trace = self._base_trace()
        self._last_trace.update(
            {
                "sandbox_dir": str(sandbox_dir),
                "runner": str(runner_ref),
                "runner_mode": self.output_mode,
                "runner_latency_ms": (time.monotonic() - started) * 1000.0,
            }
        )
        if isinstance(result, dict):
            return dict(result)
        output_path = sandbox_dir / self.output_file
        if output_path.exists():
            return _read_json_object(output_path)
        raise RuntimeError(f"{self.output_mode} runner returned {type(result).__name__} and did not write {output_path}")

    def _run_command_once(self, *, sandbox_dir: Path, probe: dict[str, Any], attempt: int) -> dict[str, Any]:
        sandbox_abs = sandbox_dir.resolve()
        command = self._render_command(sandbox_dir=sandbox_dir, probe=probe)
        output_path = sandbox_abs / self.output_file
        if self.output_mode == "command_output_file" and output_path.exists():
            output_path.unlink()
        logs_dir = sandbox_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        stdout_path = logs_dir / f"attempt_{attempt}.stdout"
        stderr_path = logs_dir / f"attempt_{attempt}.stderr"
        command_log_path = logs_dir / f"attempt_{attempt}.command.json"
        protected_hashes = _protected_input_hashes(sandbox_abs)
        env = os.environ.copy()
        env.update(self.extra_env)
        # openhands (--headless) writes files into its own workspace_base, not cwd, so
        # the model's agent_output.json never lands in the per-query sandbox the adapter
        # reads. Point openhands' workspace knobs at the sandbox dir. Scoped to openhands
        # (substring match) so other agents are unaffected.
        if command and "openhands" in str(command[0]).lower():
            for _k in ("WORKSPACE_BASE", "OPENHANDS_WORKSPACE_BASE", "WORKSPACE_MOUNT_PATH",
                       "WORKSPACE_MOUNT_PATH_IN_SANDBOX", "FILE_STORE_PATH"):
                env[_k] = str(sandbox_abs)
            env["RUNTIME"] = env.get("RUNTIME", "local")
        command_log = {
            "schema_version": "amb-sandbox-agent-command-attempt-v1",
            "system_id": self.system_id,
            "case_id": self.case_id,
            "query_id": str(probe.get("query_id") or ""),
            "attempt": attempt,
            "cwd": str(sandbox_abs),
            "command": _redacted_command(command),
            "binary_path": shutil.which(command[0], path=env.get("PATH")) if command else None,
            "output_file": str(sandbox_abs / self.output_file),
            "stdout_path": str(stdout_path.resolve()),
            "stderr_path": str(stderr_path.resolve()),
            "timeout_s": self.timeout_s,
            "env_presence": {name: bool(env.get(name)) for name in self.required_env_vars},
        }
        _write_json(command_log_path, command_log)
        self._last_trace = self._base_trace()
        self._last_trace.update(
            {
                "sandbox_dir": str(sandbox_dir),
                "command": command_log["command"],
                "binary_path": command_log["binary_path"],
                "command_log_path": str(command_log_path),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "returncode": None,
                "command_latency_ms": None,
            }
        )
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=sandbox_abs,
                env=env,
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
            )
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
        except subprocess.TimeoutExpired as exc:
            # subprocess.run(text=True) does NOT decode exc.stdout/stderr on
            # TimeoutExpired — they can come back as bytes, and write_text(bytes)
            # then raises "data must be str, not bytes", which masked the real
            # timeout with a confusing type error (and zeroed codex on 5 backbones).
            # Coerce to text defensively so the timeout surfaces as the true cause.
            stdout_path.write_text(_as_text(exc.stdout), encoding="utf-8")
            stderr_path.write_text(_as_text(exc.stderr), encoding="utf-8")
            raise RuntimeError(f"command timed out after {self.timeout_s}s: {_redacted_command(command)}") from exc
        latency_ms = (time.monotonic() - started) * 1000.0
        mutated_inputs = _mutated_inputs(sandbox_abs, protected_hashes)
        self._last_trace.update(
            {
                "sandbox_dir": str(sandbox_dir),
                "command": _redacted_command(command),
                "binary_path": command_log["binary_path"],
                "command_log_path": str(command_log_path),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "returncode": completed.returncode,
                "command_latency_ms": latency_ms,
                "protected_input_mutations": mutated_inputs,
            }
        )
        if mutated_inputs:
            raise RuntimeError(f"command modified protected sandbox input files: {mutated_inputs}")
        if completed.returncode != 0:
            if self.output_mode == "command_output_file" and self.allow_nonzero_with_output and output_path.exists():
                return _read_json_object(output_path)
            raise RuntimeError(f"command returned {completed.returncode}: {_redacted_command(command)}")
        if self.output_mode == "command_output_file":
            return _read_json_object(output_path)
        return _parse_stdout_json(completed.stdout)

    def _write_sandbox(self, sandbox_dir: Path, probe: dict[str, Any]) -> set[str]:
        case = self._case
        visible_memory_ids: set[str] = set()
        _write_json(sandbox_dir / "case_manifest.json", self._case_manifest(case=case, probe=probe))
        _write_jsonl(sandbox_dir / "sessions.jsonl", self._observations)
        _write_jsonl(sandbox_dir / "events.jsonl", [self._public_event(event) for event in (case.events if case else ())])
        memories = [self._public_memory(memory) for memory in (case.gold_memory_units if case else ())]
        for memory in memories:
            visible_memory_ids.add(str(memory["memory_id"]))
        _write_jsonl(sandbox_dir / "memories.jsonl", memories)
        _write_jsonl(
            sandbox_dir / "state_contracts.jsonl",
            [self._public_state_contract(contract) for contract in (case.state_contracts if case else ())],
        )
        _write_json(sandbox_dir / "query.json", self._public_probe(probe))
        _write_json(sandbox_dir / "output_schema.json", OUTPUT_SCHEMA)
        (sandbox_dir / "INSTRUCTION.md").write_text(self._instruction_text(), encoding="utf-8")
        return visible_memory_ids

    def _case_manifest(self, *, case: Case | None, probe: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": SANDBOX_LAYOUT_VERSION,
            "system_id": self.system_id,
            "case_id": self.case_id or str(probe.get("case_id") or ""),
            "domain": str(probe.get("domain") or (case.domain if case else "")),
            "query_id": str(probe.get("query_id") or ""),
            "files": [
                "INSTRUCTION.md",
                "sessions.jsonl",
                "events.jsonl",
                "memories.jsonl",
                "state_contracts.jsonl",
                "query.json",
                "output_schema.json",
            ],
            "counts": {
                "observations": len(self._observations),
                "events": len(case.events) if case else 0,
                "memories": len(case.gold_memory_units) if case else 0,
                "state_contracts": len(case.state_contracts) if case else 0,
            },
            "forbidden_fields": [
                "gold_memory_ids",
                "forbidden_memory_ids",
                "expected_behavior",
                "must_include",
                "must_not_include",
                "active_memory_ids",
                "inactive_memory_ids",
                "deleted_memory_ids",
                "scorer internals",
            ],
        }

    def _public_probe(self, probe: dict[str, Any]) -> dict[str, Any]:
        allowed = (
            "case_id",
            "domain",
            "query_id",
            "timestamp",
            "prompt",
            "task_type",
            "probe_type",
            "scoring_rule",
            "state_contract_id",
            "memory_dependency",
        )
        return {key: probe.get(key) for key in allowed if key in probe}

    def _public_event(self, event: Any) -> dict[str, Any]:
        return {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "timestamp": event.timestamp,
            "subject": event.subject,
            "predicate": event.predicate,
            "object": event.object,
            "source_turn_ids": list(event.source_turn_ids),
            "attributes": dict(event.attributes),
        }

    def _public_memory(self, memory: Any) -> dict[str, Any]:
        return {
            "memory_id": memory.memory_id,
            "type": memory.type,
            "content": memory.content,
            "source_turn_ids": list(memory.source_turn_ids),
            "scenario_id": memory.scenario_id,
            "memory_type": memory.memory_type,
            "canonical_form": dict(memory.canonical_form),
            "source_event_ids": list(memory.source_event_ids),
            "source_trace_ids": list(memory.source_trace_ids),
            "valid_from": memory.valid_from,
            "valid_until": memory.valid_until,
            "status": memory.status,
            "importance": memory.importance,
            "confidence": memory.confidence,
            "privacy_level": memory.privacy_level,
            "sensitivity": memory.sensitivity,
            "authorization_scope": memory.authorization_scope,
            "update_of": memory.update_of,
            "invalidates": list(memory.invalidates),
            "forget_policy": memory.forget_policy,
        }

    def _public_state_contract(self, contract: Any) -> dict[str, Any]:
        return {
            "state_contract_id": contract.state_contract_id,
            "timestamp": contract.timestamp,
            "scenario_id": contract.scenario_id,
            "required_governance_rules": list(contract.required_governance_rules),
        }

    def _instruction_text(self) -> str:
        return (
            "# AutoMemoryBench Sandbox Agent Task\n\n"
            "Read only the files in this directory. Do not assume access to hidden gold labels.\n"
            "Answer the query in query.json using sessions.jsonl, events.jsonl, memories.jsonl, "
            "and public state_contracts.jsonl.\n\n"
            f"Write exactly one JSON object to {self.output_file} unless the adapter mode requires stdout JSON. "
            "The JSON must match output_schema.json and include response plus activated_memory_ids. "
            "activated_memory_ids must name only memories that materially support the answer. "
            "Do not output free text outside JSON.\n"
        )

    def _validate_output(self, raw: dict[str, Any], *, visible_memory_ids: set[str]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise TypeError("sandbox output must be a JSON object")
        if "response" not in raw or not isinstance(raw.get("response"), str):
            raise ValueError("sandbox output must include string field response")
        if "activated_memory_ids" not in raw or not isinstance(raw.get("activated_memory_ids"), list):
            raise ValueError("sandbox output must include activated_memory_ids as a list")
        activated = [str(item) for item in raw["activated_memory_ids"]]
        unknown = sorted(set(activated) - visible_memory_ids)
        if unknown:
            raise ValueError(f"activated_memory_ids contains unknown memory ids: {unknown}")
        evidence = raw.get("evidence", [])
        if evidence is None:
            evidence = []
        if not isinstance(evidence, list):
            raise ValueError("evidence must be a list")
        evidence_ids = [
            str(item.get("memory_id"))
            for item in evidence
            if isinstance(item, dict) and item.get("memory_id") not in (None, "")
        ]
        unknown_evidence = sorted(set(evidence_ids) - visible_memory_ids)
        if unknown_evidence:
            raise ValueError(f"evidence contains unknown memory ids: {unknown_evidence}")
        memory_ops = raw.get("memory_operations", [])
        if memory_ops is None:
            memory_ops = []
        if not isinstance(memory_ops, list):
            raise ValueError("memory_operations must be a list")
        for item in memory_ops:
            if not isinstance(item, dict):
                raise ValueError("memory_operations entries must be objects")
            memory_id = item.get("memory_id")
            if memory_id not in (None, "") and str(memory_id) not in visible_memory_ids:
                raise ValueError(f"memory_operations contains unknown memory id: {memory_id}")
        tool_calls = raw.get("tool_calls", [])
        if tool_calls is None:
            tool_calls = []
        if not isinstance(tool_calls, list):
            raise ValueError("tool_calls must be a list")
        cost = raw.get("cost", {})
        if cost is None:
            cost = {}
        if not isinstance(cost, dict):
            raise ValueError("cost must be an object")
        normalized = dict(raw)
        normalized["activated_memory_ids"] = activated
        normalized["evidence"] = evidence
        normalized["memory_operations"] = memory_ops
        normalized["tool_calls"] = tool_calls
        normalized["cost"] = cost
        return normalized

    def _render_command(self, *, sandbox_dir: Path, probe: dict[str, Any]) -> list[str]:
        sandbox_abs = sandbox_dir.resolve()
        output_path = sandbox_abs / self.output_file
        replacements = {
            "sandbox_dir": str(sandbox_abs),
            "instruction_file": str(sandbox_abs / "INSTRUCTION.md"),
            "query_file": str(sandbox_abs / "query.json"),
            "output_schema_file": str(sandbox_abs / "output_schema.json"),
            "output_file": str(output_path),
            "case_id": self.case_id,
            "query_id": str(probe.get("query_id") or ""),
            "system_id": self.system_id,
            "model_id": self.model_id,
            "provider": self.provider,
            "home": os.path.expanduser("~"),
        }
        if isinstance(self.command_template, str):
            rendered = _format_template(self.command_template, replacements)
            return shlex.split(rendered)
        if isinstance(self.command_template, list):
            return [_format_template(str(part), replacements) for part in self.command_template]
        raise ValueError("command_template must be a string or list of strings")

    def _safe_version(self) -> dict[str, Any]:
        if not self.version_command:
            return {"status": "not_configured"}
        command = shlex.split(self.version_command) if isinstance(self.version_command, str) else list(self.version_command)
        try:
            completed = subprocess.run(command, text=True, capture_output=True, timeout=15.0, check=False)
        except Exception as exc:
            return {"status": "failed", "command": _redacted_command(command), "error": str(exc)}
        return {
            "status": "passed" if completed.returncode == 0 else "failed",
            "command": _redacted_command(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip()[:500],
            "stderr": completed.stderr.strip()[:500],
        }

    def _sandbox_dir(self, query_id: str) -> Path:
        return self.workspace_root / _safe_path_component(self.system_id) / _safe_path_component(self.case_id) / _safe_path_component(query_id)

    def _base_trace(self, *, namespace: str | None = None) -> dict[str, Any]:
        return {
            "framework_id": self.framework_id,
            "framework_label": self.framework_label,
            "framework_version": self.sandbox_layout_version,
            "framework_runtime": self.output_mode,
            "orchestration_mode": "single_agent",
            "agent_family": self.agent_family,
            "official_surface": self.official_surface,
            "binary_or_sdk": self.binary_or_sdk,
            "model_id": self.model_id,
            "provider": self.provider,
            "memory_backend_id": "case_sandbox_files",
            "tool_runtime_id": "automemorybench_tool_runtime_v1",
            "session_id": self.case_id,
            "namespace": ["sandbox_agent", namespace or self.case_id],
            "message_history_policy": "case_sandbox_gold_free",
            "memory_ops": [],
            "retrieval_hits": [],
            "tool_calls": [],
            "planner_trace": [],
            "handoff_trace": [],
            "cost": {},
            "framework_state": {},
            "claim_boundary": self.claim_boundary,
        }


def create_adapter(**kwargs: Any) -> SandboxAgentAdapter:
    return SandboxAgentAdapter(**kwargs)


def _load_callable(reference: str) -> Any:
    module_name, separator, attr_name = reference.partition(":")
    if not separator or not module_name or not attr_name:
        raise ValueError("runner reference must use 'module:callable' format")
    module = importlib.import_module(module_name)
    target = getattr(module, attr_name, None)
    if not callable(target):
        raise ValueError(f"runner reference {reference!r} is not callable")
    return target


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"expected sandbox JSON output does not exist: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise TypeError(f"sandbox JSON output must be an object: {path}")
    return payload


def _protected_input_hashes(sandbox_dir: Path) -> dict[str, str]:
    names = (
        "INSTRUCTION.md",
        "case_manifest.json",
        "sessions.jsonl",
        "events.jsonl",
        "memories.jsonl",
        "state_contracts.jsonl",
        "query.json",
        "output_schema.json",
    )
    hashes: dict[str, str] = {}
    for name in names:
        path = sandbox_dir / name
        if path.exists():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _mutated_inputs(sandbox_dir: Path, expected_hashes: dict[str, str]) -> list[str]:
    mutated: list[str] = []
    for name, expected_hash in expected_hashes.items():
        path = sandbox_dir / name
        if not path.exists():
            mutated.append(name)
            continue
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            mutated.append(name)
    return sorted(mutated)


def _as_text(value: Any) -> str:
    """Coerce subprocess stream output to str.

    subprocess.run(text=True) leaves TimeoutExpired.stdout/stderr undecoded (bytes)
    on some platforms/Python versions; passing that to write_text raises
    "data must be str, not bytes". Decode bytes leniently, pass str through, empty
    for None.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _parse_stdout_json(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("stdout JSON mode received empty stdout")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ValueError("stdout JSON mode requires the last non-empty stdout line to be a JSON object") from exc
    if not isinstance(payload, dict):
        raise TypeError("stdout JSON mode requires a JSON object")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _format_template(template: str, replacements: dict[str, str]) -> str:
    try:
        return template.format(**replacements)
    except KeyError as exc:
        raise ValueError(f"unknown command_template placeholder: {exc.args[0]}") from exc


def _safe_path_component(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text[:128] or "unknown"


def _redacted_command(command: list[str]) -> list[str]:
    redacted: list[str] = []
    skip_next = False
    secret_flags = {"--api-key", "--token", "--access-token", "--secret", "--password"}
    for part in command:
        if skip_next:
            redacted.append("<redacted>")
            skip_next = False
            continue
        lower = part.lower()
        if lower in secret_flags:
            redacted.append(part)
            skip_next = True
        elif any(marker in lower for marker in ("api_key=", "token=", "secret=", "password=")):
            redacted.append(part.split("=", 1)[0] + "=<redacted>")
        else:
            redacted.append(part)
    return redacted
