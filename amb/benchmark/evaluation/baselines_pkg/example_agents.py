"""Small black-box agents for integration smoke tests and examples."""

from __future__ import annotations

from typing import Any


class EchoBlackBoxAgent:
    """Deterministic protocol smoke agent.

    This agent is intentionally simple: it counts observations in process
    memory and returns a short deterministic response. It avoids echoing raw
    observations so the example does not model unsafe replay of sensitive input.
    It is useful for verifying that custom agent-system configs, CLI wiring,
    metadata, and matrix packaging work before replacing it with a real agent
    framework adapter.
    """

    def __init__(self, prefix: str = "echo_agent", max_observations: int = 3) -> None:
        self.prefix = prefix
        self.max_observations = max(1, int(max_observations))
        self.case_id = ""
        self.observations: list[dict[str, Any]] = []
        self._last_trace: dict[str, Any] = {}
        self._tool_calls: list[dict[str, Any]] = []

    def reset(self, case_id: str) -> None:
        self.case_id = case_id
        self.observations = []
        self._last_trace = {
            "framework_id": "example_blackbox",
            "framework_version": "smoke",
            "framework_runtime": "python_in_process",
            "orchestration_mode": "single_agent",
            "memory_backend_id": "in_process_observation_counter",
            "tool_runtime_id": "none",
            "session_id": case_id,
            "namespace": ["example_blackbox", case_id],
            "message_history_policy": "window",
            "memory_ops": [],
            "retrieval_hits": [],
            "tool_calls": [],
            "planner_trace": [],
            "handoff_trace": [],
            "framework_state": {"observations_seen": 0},
        }
        self._tool_calls = []

    def observe(self, observation: dict[str, Any]) -> None:
        self.observations.append(dict(observation))

    def answer_or_act(self, probe: dict[str, Any]) -> dict[str, Any]:
        prompt = str(probe.get("prompt", ""))
        retained = min(len(self.observations), self.max_observations)
        response = f"{self.prefix}: {prompt} | observations_seen: {len(self.observations)} | retained_window: {retained}"
        self._last_trace = {
            **self._last_trace,
            "memory_ops": [
                {
                    "op": "inject",
                    "memory_id": "observation_counter",
                    "namespace": ["example_blackbox", self.case_id],
                    "metadata": {"observations_seen": len(self.observations), "retained_window": retained},
                }
            ],
            "planner_trace": [
                {
                    "step": "count_observations",
                    "query_id": str(probe.get("query_id", "")),
                    "used_raw_observation_content": False,
                }
            ],
            "framework_state": {"observations_seen": len(self.observations), "retained_window": retained},
        }
        return {
            "memory_needed": bool(self.observations),
            "activated_memory_ids": [],
            "response": response,
            "memory_operations": [],
            "cost": {
                "input_tokens": len(prompt.split()),
                "output_tokens": len(response.split()),
                "latency_ms": 0.0,
            },
        }

    def ingest_turn(self, turn: dict[str, Any]) -> None:
        self.observe(turn)

    def run_probe(self, probe: dict[str, Any], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return self.answer_or_act(probe)

    def export_memory(self) -> list[dict[str, Any]]:
        return [
            {
                "memory_id": "observation_counter",
                "content": f"{len(self.observations)} observations seen",
                "namespace": ["example_blackbox", self.case_id],
            }
        ]

    def export_trace(self) -> dict[str, Any]:
        return dict(self._last_trace)

    def export_tool_calls(self) -> list[dict[str, Any]]:
        return list(self._tool_calls)

    def export_framework_state(self) -> dict[str, Any]:
        return dict(self._last_trace.get("framework_state", {}))


def create_echo_agent(prefix: str = "echo_agent", max_observations: int = 3) -> EchoBlackBoxAgent:
    return EchoBlackBoxAgent(prefix=prefix, max_observations=max_observations)
