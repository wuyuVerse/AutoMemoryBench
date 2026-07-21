"""Agent adapter interfaces for AutoMemoryBench runners."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BlackBoxAgent(Protocol):
    """Minimal black-box interface for memory agents under test."""

    def reset(self, case_id: str) -> None:
        """Reset agent state before a case starts."""

    def observe(self, observation: dict[str, Any]) -> None:
        """Feed one chronological observation to the agent."""

    def answer_or_act(self, probe: dict[str, Any]) -> dict[str, Any]:
        """Return a prediction-like dictionary for a benchmark query."""


@runtime_checkable
class WhiteBoxAgent(BlackBoxAgent, Protocol):
    """Optional white-box interface for systems that expose memory internals."""

    def export_memory(self) -> list[dict[str, Any]]:
        """Export current memory store after observations."""

    def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        """Return raw retrieval results for diagnostic evaluation."""

    def delete(self, deletion_request: dict[str, Any]) -> dict[str, Any]:
        """Execute or simulate a deletion request."""

    def export_trace(self) -> dict[str, Any]:
        """Export agent-side trace metadata for debugging and audit."""


@runtime_checkable
class AgentFrameworkAdapter(BlackBoxAgent, Protocol):
    """Optional richer protocol for comparing agent orchestration frameworks.

    The core benchmark remains black-box. Framework adapters can implement these
    hooks to expose planner, memory, tool, handoff, and runtime state traces for
    framework-comparative analysis without changing the prediction schema.
    """

    def reset(self, case_id: str, namespace: str | None = None) -> None:
        """Reset framework state before a case starts."""

    def ingest_turn(self, turn: dict[str, Any]) -> None:
        """Feed one chronological conversation turn into the framework runtime."""

    def run_probe(self, probe: dict[str, Any], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Run one benchmark probe through the framework runtime."""

    def export_memory(self) -> list[dict[str, Any]]:
        """Export framework-visible memory state."""

    def export_trace(self) -> dict[str, Any]:
        """Export planner, retrieval, tool, handoff, and runtime traces."""

    def export_tool_calls(self) -> list[dict[str, Any]]:
        """Export normalized tool calls made by the framework."""

    def export_framework_state(self) -> dict[str, Any]:
        """Export framework-specific state for audit and reproducibility."""
