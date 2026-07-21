"""Deterministic raw-LLM-runner framework adapter smoke implementation."""

from __future__ import annotations

from typing import Any


class RawLLMRunnerAdapter:
    """Packaged T0 adapter for framework trace and tool-runtime smoke tests.

    This is intentionally deterministic and model-free. It proves the adapter
    interface, trace export, and standard tool-runtime envelope without claiming
    model performance.
    """

    def __init__(self, prefix: str = "raw_llm_runner", memory_backend_id: str = "full_history") -> None:
        self.prefix = prefix
        self.memory_backend_id = memory_backend_id
        self.case_id = ""
        self.observations: list[dict[str, Any]] = []
        self._last_trace: dict[str, Any] = {}

    def reset(self, case_id: str, namespace: str | None = None) -> None:
        self.case_id = case_id
        self.observations = []
        self._last_trace = self._base_trace(namespace=namespace)

    def observe(self, observation: dict[str, Any]) -> None:
        self.observations.append(dict(observation))

    def ingest_turn(self, turn: dict[str, Any]) -> None:
        self.observe(turn)

    def answer_or_act(self, probe: dict[str, Any]) -> dict[str, Any]:
        return self.run_probe(probe)

    def run_probe(self, probe: dict[str, Any], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        prompt = str(probe.get("prompt", ""))
        query_id = str(probe.get("query_id", ""))
        task_type = str(probe.get("task_type", ""))
        is_tool_task = task_type == "tool" or "tool" in str(probe.get("scoring_rule", ""))
        tool_call = self._standard_tool_call(prompt) if is_tool_task else None
        response = f"{self.prefix}: query={query_id} observations_seen={len(self.observations)}"
        self._last_trace = {
            **self._base_trace(),
            "message_history_policy": "full",
            "memory_ops": [
                {
                    "op": "inject",
                    "memory_id": "full_history_context",
                    "namespace": ["raw_llm_runner", self.case_id],
                    "metadata": {"observations_seen": len(self.observations)},
                }
            ],
            "tool_calls": [tool_call] if tool_call else [],
            "planner_trace": [
                {
                    "step": "deterministic_raw_runner_smoke",
                    "query_id": query_id,
                    "tool_task": is_tool_task,
                }
            ],
            "framework_state": {
                "observations_seen": len(self.observations),
                "memory_backend_id": self.memory_backend_id,
            },
        }
        payload: dict[str, Any] = {
            "memory_needed": bool(self.observations),
            "activated_memory_ids": [],
            "response": response,
            "memory_operations": [],
            "tool_calls": [tool_call] if tool_call else [],
            "cost": {
                "input_tokens": len(prompt.split()),
                "output_tokens": len(response.split()),
                "latency_ms": 0.0,
            },
        }
        if tool_call:
            payload["tool_name"] = tool_call["tool_name"]
            payload["parameters"] = dict(tool_call["arguments"])
        return payload

    def export_memory(self) -> list[dict[str, Any]]:
        return [
            {
                "memory_id": "full_history_context",
                "namespace": ["raw_llm_runner", self.case_id],
                "content": f"{len(self.observations)} observations seen",
            }
        ]

    def export_trace(self) -> dict[str, Any]:
        return dict(self._last_trace)

    def export_tool_calls(self) -> list[dict[str, Any]]:
        return list(self._last_trace.get("tool_calls", []))

    def export_framework_state(self) -> dict[str, Any]:
        return dict(self._last_trace.get("framework_state", {}))

    def _base_trace(self, *, namespace: str | None = None) -> dict[str, Any]:
        return {
            "framework_id": "raw_llm_runner",
            "framework_version": "packaged_t0",
            "framework_runtime": "python_in_process",
            "orchestration_mode": "single_agent",
            "model_id": "deterministic_no_model",
            "memory_backend_id": self.memory_backend_id,
            "tool_runtime_id": "automemorybench_tool_runtime_v1",
            "session_id": self.case_id,
            "namespace": ["raw_llm_runner", namespace or self.case_id],
            "message_history_policy": "full",
            "memory_ops": [],
            "retrieval_hits": [],
            "tool_calls": [],
            "planner_trace": [],
            "handoff_trace": [],
            "cost": {},
            "framework_state": {},
        }

    def _standard_tool_call(self, prompt: str) -> dict[str, Any]:
        return {
            "tool_name": "slack.search",
            "arguments": {
                "channel": "benchmark",
                "query": " ".join(prompt.split()[:8]) or "memory state",
            },
            "approval_state": "not_required",
            "result_summary": "deterministic mock search result",
            "source_memory_ids": [],
            "side_effect_id": None,
        }


def create_adapter(prefix: str = "raw_llm_runner", memory_backend_id: str = "full_history") -> RawLLMRunnerAdapter:
    return RawLLMRunnerAdapter(prefix=prefix, memory_backend_id=memory_backend_id)
