# Integrating your memory system

AutoMemoryBench is designed so a third party can evaluate their own memory
system with minimal glue. The benchmark is **black-box by default**: it feeds
your system a chronological stream of observations, then asks queries.

## 1. The black-box protocol

Implement three methods (`amb/benchmark/evaluation/core/adapters.py`):

```python
from typing import Any

class MyMemory:
    def reset(self, case_id: str) -> None:
        """Clear all state before a new case starts."""

    def observe(self, observation: dict[str, Any]) -> None:
        """Ingest one chronological item (a conversation turn or lifecycle event).
        Store/update/delete memory however your system does."""

    def answer_or_act(self, probe: dict[str, Any]) -> dict[str, Any]:
        """Answer one query. Return a prediction dict, e.g.
        {"answer": "...", "activated_memory_ids": [...]}.
        activated_memory_ids is optional but enables white-box safety scoring."""
```

Optionally implement the white-box hooks (`export_memory`, `retrieve`, `delete`,
`export_trace`) for richer diagnostics — see the `WhiteBoxAgent` protocol.

## 2. A factory + config

`amb run-agent`/`run-release-agent` load your adapter through a **factory
reference** (`module:callable`) named in a JSON config. Generate a template:

```bash
amb scaffold-agent-system \
  --system-id my_memory \
  --loader my_pkg.my_adapter:create_client \
  --framework my_framework \
  --output configs/agent_systems/my_memory.json
```

Your `create_client(**loader_kwargs)` returns an object implementing the
protocol above. Put any endpoint/model settings in `loader_kwargs`; read secrets
from environment variables (see `.env.example`) — never hardcode keys.

## 3. Run and score

```bash
# Over the sample split (one command produces + scores predictions):
amb run-release-agent \
  --manifest data/sample/manifest.json --split audit_subset \
  --config configs/agent_systems/my_memory.json \
  --output reports/my_memory.json

# Limit to one domain while developing:
amb run-release-agent ... --domains coding_agent
```

Then derive StrictCore from the report components (see `docs/METRICS.md`).

## 4. Reference adapters

`amb/clients/providers/` contains working adapters for real systems you can copy
from:

- `framework_sdks/mem0.py` — Mem0 (vector store + LLM extraction)
- `framework_sdks/letta.py` — Letta / MemGPT
- `framework_sdks/langmem.py` — LangMem
- `managed_services/zep_graphiti.py` — Zep / Graphiti
- `local_sources/` — MemoryOS, A-MEM, LightRAG, Memoripy, txtai, and more

Each keeps the embedder constant (bge-m3 by default) and takes the chat backbone
from the model registry, so architecture effects separate from backbone effects.
Install a system's dependency with its extra, e.g. `pip install -e ".[mem0]"`.
