"""Zep/Graphiti temporal graph memory integration wrapper."""

from __future__ import annotations

from typing import Any

from amb.benchmark.integrations.base import ExternalMemoryAgent, IntegrationConfig


class ZepGraphitiAgent(ExternalMemoryAgent):
    """Adapter for Zep/Graphiti-like temporal graph memory clients."""

    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(
            client,
            IntegrationConfig(
                system_id="zep_graphiti",
                top_k=top_k,
                add_methods=("add_episode", "add", "add_memory", "save"),
                search_methods=("search", "search_memory", "retrieve", "query"),
                delete_methods=("delete", "delete_episode", "remove"),
                export_methods=("get_episodes", "get_all", "export_memory", "list"),
            ),
        )
