"""LangMem integration wrapper."""

from __future__ import annotations

from typing import Any

from amb.benchmark.integrations.base import ExternalMemoryAgent, IntegrationConfig


class LangMemAgent(ExternalMemoryAgent):
    """Adapter for LangMem-like stores and managers."""

    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(
            client,
            IntegrationConfig(
                system_id="langmem",
                top_k=top_k,
                add_methods=("put", "add", "store", "save"),
                search_methods=("search", "retrieve", "query"),
                delete_methods=("delete", "remove"),
                export_methods=("list", "all", "get_all", "export_memory"),
            ),
        )
