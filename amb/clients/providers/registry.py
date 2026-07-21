"""Registry of official-code-backed real-client adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSpec:
    """Stable identity for an AMST real-system provider adapter."""

    method_id: str
    module: str
    factory: str = "create_client"
    requires_credentials: bool = False
    notes: str = ""

    @property
    def factory_path(self) -> str:
        return f"{self.module}:{self.factory}"


PROVIDER_SPECS: tuple[ProviderSpec, ...] = (
    ProviderSpec("a_mem_agentic_memory", "amb.clients.a_mem_agentic_memory"),
    ProviderSpec("cognee_or_hipporag", "amb.clients.cognee", requires_credentials=True),
    ProviderSpec("hindsight_or_prod_api", "amb.clients.hindsight", requires_credentials=True),
    ProviderSpec("langmem", "amb.clients.langmem"),
    ProviderSpec("letta_memgpt", "amb.clients.letta", requires_credentials=True),
    ProviderSpec("lightmem", "amb.clients.lightmem"),
    ProviderSpec("mem0", "amb.clients.mem0", requires_credentials=True),
    ProviderSpec("meminsight", "amb.clients.meminsight", requires_credentials=True),
    ProviderSpec("memobase", "amb.clients.memobase", requires_credentials=True),
    ProviderSpec("memorybank_or_memoryos", "amb.clients.memorybank_or_memoryos", requires_credentials=True),
    ProviderSpec("memos", "amb.clients.memos"),
    ProviderSpec("mirix", "amb.clients.mirix", requires_credentials=True),
    ProviderSpec("openai_memory", "amb.clients.openai_memory"),
    ProviderSpec("simplemem", "amb.clients.simplemem"),
    ProviderSpec("memoripy", "amb.clients.memoripy"),
    ProviderSpec("synap", "amb.clients.synap"),
    ProviderSpec("evermind", "amb.clients.evermind"),
    ProviderSpec("supermemory", "amb.clients.supermemory", requires_credentials=True),
    ProviderSpec("zep_cloud", "amb.clients.zep_cloud", requires_credentials=True),
    ProviderSpec("mirix_cloud", "amb.clients.mirix_cloud", requires_credentials=True),
    ProviderSpec("zep_graphiti", "amb.clients.zep_graphiti", requires_credentials=True),
    ProviderSpec("reme", "amb.clients.reme"),
    ProviderSpec("lightrag", "amb.clients.lightrag"),
    ProviderSpec("nanographrag", "amb.clients.nanographrag"),
    ProviderSpec("txtai", "amb.clients.txtai"),
    ProviderSpec("memengine", "amb.clients.memengine"),
)

PROVIDER_BY_METHOD_ID: dict[str, ProviderSpec] = {spec.method_id: spec for spec in PROVIDER_SPECS}
FACTORY_BY_METHOD_ID: dict[str, str] = {spec.method_id: spec.factory_path for spec in PROVIDER_SPECS}

__all__ = [
    "FACTORY_BY_METHOD_ID",
    "PROVIDER_BY_METHOD_ID",
    "PROVIDER_SPECS",
    "ProviderSpec",
]
