"""Mem0 integration wrapper."""

from __future__ import annotations

from typing import Any

from amb.benchmark.integrations.base import ExternalMemoryAgent, IntegrationConfig


class Mem0Agent(ExternalMemoryAgent):
    """Adapter for Mem0-like clients with `add` and `search` methods."""

    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(
            client,
            IntegrationConfig(
                system_id="mem0",
                top_k=top_k,
                add_methods=("add", "add_memory"),
                search_methods=("search",),
                delete_methods=("delete", "delete_memory"),
                export_methods=("get_all", "get", "export_memory"),
            ),
        )
