"""Factory utilities for optional external memory-system agents."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from amb.benchmark.integrations.langmem import LangMemAgent
from amb.benchmark.integrations.letta import LettaAgent
from amb.benchmark.integrations.mem0 import Mem0Agent
from amb.benchmark.integrations.sota_official import (
    AMemAgent,
    CogneeAgent,
    HindsightAgent,
    LightMemAgent,
    MemOSAgent,
    MemInsightAgent,
    SimpleMemAgent,
    SynapAgent,
    EvermindAgent,
    MemoryOSAgent,
    MemobaseAgent,
    MirixAgent,
    OpenAIMemoryAgent,
    SupermemoryAgent,
)
from amb.benchmark.integrations.sota_official import ZepCloudAgent, MirixCloudAgent, MemoripyAgent, RemeAgent, LightRAGAgent, NanoGraphRAGAgent, TxtaiAgent, MemEngineAgent
from amb.benchmark.integrations.zep_graphiti import ZepGraphitiAgent
from amb.benchmark.schemas.io import read_json


PROVIDERS = {
    "mem0": Mem0Agent,
    "letta": LettaAgent,
    "memgpt": LettaAgent,
    "langmem": LangMemAgent,
    "zep": ZepGraphitiAgent,
    "graphiti": ZepGraphitiAgent,
    "zep_graphiti": ZepGraphitiAgent,
    "memos": MemOSAgent,
    "a_mem_agentic_memory": AMemAgent,
    "a_mem": AMemAgent,
    "lightmem": LightMemAgent,
    "hindsight": HindsightAgent,
    "hindsight_or_prod_api": HindsightAgent,
    "cognee_or_hipporag": CogneeAgent,
    "cognee": CogneeAgent,
    "mirix": MirixAgent,
    "supermemory": SupermemoryAgent,
    "zep_cloud": ZepCloudAgent,
    "mirix_cloud": MirixCloudAgent,
    "memobase": MemobaseAgent,
    "meminsight": MemInsightAgent,
    "simplemem": SimpleMemAgent,
    "synap": SynapAgent,
    "memoripy": MemoripyAgent,
    "reme": RemeAgent,
    "lightrag": LightRAGAgent,
    "nanographrag": NanoGraphRAGAgent,
    "txtai": TxtaiAgent,
    "memengine": MemEngineAgent,
    "evermind": EvermindAgent,
    "openai_memory": OpenAIMemoryAgent,
    "openai_agents_session_memory": OpenAIMemoryAgent,
    "memorybank_or_memoryos": MemoryOSAgent,
    "memoryos": MemoryOSAgent,
}

PROVIDER_ALIASES = {
    "memgpt": "letta",
    "zep": "zep_graphiti",
    "graphiti": "zep_graphiti",
    "a_mem": "a_mem_agentic_memory",
    "hindsight": "hindsight_or_prod_api",
    "cognee": "cognee_or_hipporag",
    "memoryos": "memorybank_or_memoryos",
}


def canonical_provider(provider: str) -> str:
    normalized = str(provider).strip().lower()
    return PROVIDER_ALIASES.get(normalized, normalized)


def load_integration_agent(config_path: str | Path):
    """Load an external memory agent from a small JSON config.

    Expected config shape:

    ```json
    {
      "provider": "mem0",
      "client_factory": "my_module:create_client",
      "client_kwargs": {},
      "top_k": 5
    }
    ```
    """

    config = read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError("integration config must be a JSON object")
    provider = canonical_provider(str(config.get("provider", "")))
    agent_cls = PROVIDERS.get(provider)
    if agent_cls is None:
        raise ValueError(f"unknown integration provider {provider!r}; expected one of {', '.join(sorted(PROVIDERS))}")
    factory_ref = config.get("client_factory")
    if not factory_ref:
        raise ValueError("integration config must provide client_factory as 'module:callable'")
    client = _call_factory(str(factory_ref), dict(config.get("client_kwargs", {})))
    return agent_cls(client, top_k=int(config.get("top_k", 5)))


def _call_factory(reference: str, kwargs: dict[str, Any]) -> Any:
    module_name, separator, attr_name = reference.partition(":")
    if not separator or not module_name or not attr_name:
        raise ValueError("client_factory must use 'module:callable' format")
    module = import_module(module_name)
    factory = getattr(module, attr_name, None)
    if not callable(factory):
        raise ValueError(f"client_factory {reference!r} is not callable")
    return factory(**kwargs)
