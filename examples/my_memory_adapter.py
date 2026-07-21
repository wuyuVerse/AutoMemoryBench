"""Minimal worked example: bring your own memory system.

This implements the AutoMemoryBench black-box protocol with a trivial in-memory
store, so you can see the exact shape a real adapter needs. Replace the body of
`observe`/`answer_or_act` with calls into your own memory system.

Run it:

    amb scaffold-agent-system \
      --system-id my_memory \
      --loader examples.my_memory_adapter:create_client \
      --framework example \
      --output configs/agent_systems/my_memory.json

    amb run-release-agent \
      --manifest data/sample/manifest.json --split audit_subset \
      --config configs/agent_systems/my_memory.json \
      --output reports/my_memory.json --domains coding_agent

Then derive StrictCore from the report components (see docs/METRICS.md).
"""
from __future__ import annotations

from typing import Any


class MyMemory:
    """A toy adapter: stores every observation's text and returns the most
    recent one that shares a word with the query. Not a real memory system —
    just a protocol demonstration."""

    def __init__(self, max_items: int = 100) -> None:
        self.max_items = max_items
        self._store: list[str] = []

    # --- required black-box protocol -----------------------------------------
    def reset(self, case_id: str) -> None:
        self._store = []

    def observe(self, observation: dict[str, Any]) -> None:
        # A real system would extract/update/delete memory here. We just append
        # any text content we can find in the observation.
        text = observation.get("text") or observation.get("content") or ""
        if text:
            self._store.append(str(text))
            if len(self._store) > self.max_items:
                self._store = self._store[-self.max_items :]

    def answer_or_act(self, probe: dict[str, Any]) -> dict[str, Any]:
        query = str(probe.get("query") or probe.get("prompt") or "")
        q_words = set(query.lower().split())
        # naive "retrieval": most recent stored item overlapping the query
        best = ""
        for item in reversed(self._store):
            if q_words & set(item.lower().split()):
                best = item
                break
        return {"answer": best, "activated_memory_ids": []}


def create_client(**loader_kwargs: Any) -> MyMemory:
    """Factory referenced by the agent-system config's ``loader`` field.

    Read any endpoint/model settings from ``loader_kwargs`` here; read secrets
    from environment variables (never hardcode keys).
    """
    return MyMemory(max_items=int(loader_kwargs.get("max_items", 100)))
