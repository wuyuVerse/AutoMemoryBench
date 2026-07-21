"""AutoMemoryBench real memory-system client factories."""

from __future__ import annotations

import importlib
import sys


_ROOT_PROVIDER_MODULES = (
    "a_mem_agentic_memory",
    "cognee",
    "hindsight",
    "langmem",
    "letta",
    "lightmem",
    "mem0",
    "meminsight",
    "memobase",
    "memorybank_or_memoryos",
    "memoryos",
    "memos",
    "mirix",
    "openai_memory",
    "simplemem",
    "memoripy",
    "synap",
    "evermind",
    "supermemory",
    "zep_cloud",
    "mirix_cloud",
    "zep_graphiti",
    "reme",
    "lightrag",
    "nanographrag",
    "txtai",
    "memengine",
)


def _alias(local_name: str, target: str) -> None:
    module = importlib.import_module(target)
    sys.modules[f"{__name__}.{local_name}"] = module
    globals()[local_name] = module


_alias("_common", "amb.clients.core.common")
for _module_name in _ROOT_PROVIDER_MODULES:
    _alias(_module_name, f"amb.clients.providers.{_module_name}")


__all__ = ["_common", *_ROOT_PROVIDER_MODULES]
