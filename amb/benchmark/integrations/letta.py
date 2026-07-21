"""Letta/MemGPT integration wrapper."""

from __future__ import annotations

from typing import Any

from amb.benchmark.integrations.base import ExternalMemoryAgent, IntegrationConfig


class LettaAgent(ExternalMemoryAgent):
    """Adapter for Letta/MemGPT-like memory clients."""

    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(
            client,
            IntegrationConfig(
                system_id="letta",
                top_k=top_k,
                add_methods=("add_memory", "add", "insert_memory", "save"),
                search_methods=("search_memory", "search", "retrieve"),
                delete_methods=("delete_memory", "delete", "remove_memory"),
                export_methods=("export_memory", "list_memories", "get_all"),
            ),
        )
