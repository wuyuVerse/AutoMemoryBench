"""Integration wrappers for official/source-backed SOTA memory systems."""

from __future__ import annotations

from typing import Any

from amb.benchmark.integrations.base import ExternalMemoryAgent, IntegrationConfig


class MemOSAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(
            client,
            IntegrationConfig(
                system_id="memos",
                top_k=top_k,
                add_methods=("add",),
                search_methods=("search",),
                delete_methods=("delete",),
                export_methods=("get_all",),
            ),
        )


class AMemAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(
            client,
            IntegrationConfig(
                system_id="a_mem_agentic_memory",
                top_k=top_k,
                add_methods=("add",),
                search_methods=("search", "search_agentic"),
                delete_methods=("delete",),
                export_methods=("get_all",),
            ),
        )


class LightMemAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(
            client,
            IntegrationConfig(
                system_id="lightmem",
                top_k=top_k,
                add_methods=("add", "add_memory"),
                search_methods=("search", "retrieve"),
                delete_methods=("delete",),
                export_methods=("get_all",),
            ),
        )


class HindsightAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(
            client,
            IntegrationConfig(
                system_id="hindsight_or_prod_api",
                top_k=top_k,
                add_methods=("add", "retain"),
                search_methods=("search", "recall"),
                delete_methods=("delete",),
                export_methods=("get_all",),
            ),
        )


class CogneeAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(
            client,
            IntegrationConfig(
                system_id="cognee_or_hipporag",
                top_k=top_k,
                add_methods=("add",),
                search_methods=("search",),
                delete_methods=("delete",),
                export_methods=("get_all",),
            ),
        )


class MirixAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(
            client,
            IntegrationConfig(
                system_id="mirix",
                top_k=top_k,
                add_methods=("add",),
                search_methods=("search",),
                delete_methods=("delete",),
                export_methods=("get_all",),
            ),
        )


class SupermemoryAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(
            client,
            IntegrationConfig(
                system_id="supermemory",
                top_k=top_k,
                add_methods=("add",),
                search_methods=("search",),
                delete_methods=("delete",),
                export_methods=("get_all",),
            ),
        )


class MemobaseAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(
            client,
            IntegrationConfig(
                system_id="memobase",
                top_k=top_k,
                add_methods=("add",),
                search_methods=("search",),
                delete_methods=("delete",),
                export_methods=("get_all",),
            ),
        )


class MemoryOSAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(
            client,
            IntegrationConfig(
                system_id="memorybank_or_memoryos",
                top_k=top_k,
                add_methods=("add",),
                search_methods=("search",),
                delete_methods=("delete",),
                export_methods=("get_all",),
            ),
        )


class MemInsightAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(
            client,
            IntegrationConfig(
                system_id="meminsight",
                top_k=top_k,
                add_methods=("add",),
                search_methods=("search",),
                delete_methods=("delete",),
                export_methods=("get_all",),
            ),
        )


class SimpleMemAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(
            client,
            IntegrationConfig(
                system_id="simplemem",
                top_k=top_k,
                add_methods=("add",),
                search_methods=("search",),
                delete_methods=("delete",),
                export_methods=("get_all",),
            ),
        )


class EvermindAgent(ExternalMemoryAgent):
    def __init__(self, client, *, top_k: int = 5) -> None:
        super().__init__(client, IntegrationConfig(system_id="evermind", top_k=top_k,
            add_methods=("add",), search_methods=("search",), delete_methods=("delete",), export_methods=("get_all",)))


class SynapAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(
            client,
            IntegrationConfig(
                system_id="synap",
                top_k=top_k,
                add_methods=("add",),
                search_methods=("search",),
                delete_methods=("delete",),
                export_methods=("get_all",),
            ),
        )


class MirixCloudAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(
            client,
            IntegrationConfig(
                system_id="mirix_cloud",
                top_k=top_k,
                add_methods=("add",),
                search_methods=("search",),
                delete_methods=("delete",),
                export_methods=("get_all",),
            ),
        )


class RemeAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(client, IntegrationConfig(system_id="reme", top_k=top_k,
            add_methods=("add",), search_methods=("search",), delete_methods=("delete",), export_methods=("get_all",)))


class MemoripyAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(client, IntegrationConfig(system_id="memoripy", top_k=top_k,
            add_methods=("add",), search_methods=("search",), delete_methods=("delete",), export_methods=("get_all",)))


class LightRAGAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(client, IntegrationConfig(system_id="lightrag", top_k=top_k,
            add_methods=("add",), search_methods=("search",), delete_methods=("delete",), export_methods=("get_all",)))


class NanoGraphRAGAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(client, IntegrationConfig(system_id="nanographrag", top_k=top_k,
            add_methods=("add",), search_methods=("search",), delete_methods=("delete",), export_methods=("get_all",)))


class TxtaiAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(client, IntegrationConfig(system_id="txtai", top_k=top_k,
            add_methods=("add",), search_methods=("search",), delete_methods=("delete",), export_methods=("get_all",)))


class MemEngineAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(client, IntegrationConfig(system_id="memengine", top_k=top_k,
            add_methods=("add",), search_methods=("search",), delete_methods=("delete",), export_methods=("get_all",)))


class ZepCloudAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(
            client,
            IntegrationConfig(
                system_id="zep_cloud",
                top_k=top_k,
                add_methods=("add",),
                search_methods=("search",),
                delete_methods=("delete",),
                export_methods=("get_all",),
            ),
        )


class OpenAIMemoryAgent(ExternalMemoryAgent):
    def __init__(self, client: Any, *, top_k: int = 5) -> None:
        super().__init__(
            client,
            IntegrationConfig(
                system_id="openai_memory",
                top_k=top_k,
                add_methods=("add",),
                search_methods=("search",),
                delete_methods=("delete",),
                export_methods=("get_all",),
            ),
        )
