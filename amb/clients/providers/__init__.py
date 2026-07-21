"""Provider-layer implementations for real memory-system clients."""

from __future__ import annotations

import importlib
import sys


_PROVIDER_TARGETS = {
    "a_mem_agentic_memory": "amb.clients.providers.local_sources.a_mem_agentic_memory",
    "cognee": "amb.clients.providers.local_sources.cognee",
    "hindsight": "amb.clients.providers.managed_services.hindsight",
    "langmem": "amb.clients.providers.framework_sdks.langmem",
    "letta": "amb.clients.providers.framework_sdks.letta",
    "lightmem": "amb.clients.providers.local_sources.lightmem",
    "mem0": "amb.clients.providers.framework_sdks.mem0",
    "meminsight": "amb.clients.providers.local_sources.meminsight",
    "memobase": "amb.clients.providers.managed_services.memobase",
    "memorybank_or_memoryos": "amb.clients.providers.aliases.memorybank_or_memoryos",
    "memoryos": "amb.clients.providers.local_sources.memoryos",
    "memos": "amb.clients.providers.local_sources.memos",
    "mirix": "amb.clients.providers.managed_services.mirix",
    "openai_memory": "amb.clients.providers.framework_sdks.openai_memory",
    "simplemem": "amb.clients.providers.local_sources.simplemem",
    "memoripy": "amb.clients.providers.local_sources.memoripy",
    "synap": "amb.clients.providers.local_sources.synap",
    "evermind": "amb.clients.providers.local_sources.evermind",
    "supermemory": "amb.clients.providers.managed_services.supermemory",
    "zep_cloud": "amb.clients.providers.managed_services.zep_cloud",
    "mirix_cloud": "amb.clients.providers.managed_services.mirix_cloud",
    "zep_graphiti": "amb.clients.providers.managed_services.zep_graphiti",
    "reme": "amb.clients.providers.managed_services.reme",
    "lightrag": "amb.clients.providers.local_sources.lightrag",
    "nanographrag": "amb.clients.providers.local_sources.nanographrag",
    "txtai": "amb.clients.providers.local_sources.txtai_mem",
    "memengine": "amb.clients.providers.local_sources.memengine_mem",
}


def _alias(local_name: str, target: str) -> None:
    module = importlib.import_module(target)
    sys.modules[f"{__name__}.{local_name}"] = module
    globals()[local_name] = module


for _local_name, _target in _PROVIDER_TARGETS.items():
    _alias(_local_name, _target)


__all__ = sorted(_PROVIDER_TARGETS)
