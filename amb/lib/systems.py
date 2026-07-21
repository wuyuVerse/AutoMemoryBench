"""Unified system registry — one place that knows every runnable system.

A "system" is anything that can produce predictions for the benchmark:
  - kind="baseline": a deterministic baseline (in-process, no venv, no creds).
  - kind="memory":   an external memory framework (mem0, langmem, ...), driven
                     through a config in configs/real_system/.
  - kind="agent":    an agent framework / CLI (codex, goose, langgraph, ...),
                     driven through a config in configs/agent_systems/.

Systems are auto-discovered so adding one needs no edit here:
  - every baseline from `available_baselines()`,
  - every configs/real_system/*.json  -> a memory system (name = file stem),
  - every configs/agent_systems/**/*.json -> an agent system (name = file stem).

Short, friendly aliases (mem0, langmem, codex, ...) map to a sensible default
config so the common case is `amb run --system mem0 --split public_test`.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_SYSTEM_DIR = REPO_ROOT / "configs" / "real_system"
AGENT_SYSTEM_DIR = REPO_ROOT / "configs" / "agent_systems"

# Friendly short name -> config file stem (the preferred default config).
MEMORY_ALIASES = {
    "mem0": "mem0_siliconflow_real",
    "langmem": "langmem_real",
    "letta": "letta_real",
    "zep_graphiti": "zep_graphiti_real",
    "memos": "memos_official",
    "memoryos": "memoryos_official",
    "lightmem": "lightmem_official",
    "a_mem": "a_mem_agentic_memory_official",
    "mirix": "mirix_official",
    "cognee": "cognee_or_hipporag_official",
    "hindsight": "hindsight_or_prod_api_official",
    "supermemory": "supermemory_official",
    "memobase": "memobase_official",
    "openai_memory": "openai_memory_official",
    "meminsight": "meminsight_official",
}


# Memory family -> its dependency-isolated venv directory (see adapter inventory).
# Managed REST services (supermemory/memobase/mirix/cognee/hindsight/meminsight)
# have no venv and run in the host env.
VENV_DIRS = {
    "mem0": ".venv-mem0",
    "langmem": ".venv-langmem",
    "letta": ".venv-letta",
    "zep_graphiti": ".venv-zep-graphiti",
    "memos": ".venv-memos",
    "memoryos": ".venv-memoryos",
    "lightmem": ".venv-lightmem",
    "a_mem": ".venv-amem",
    "openai_memory": ".venv-openai-memory",
}


@dataclass(frozen=True)
class SystemSpec:
    name: str
    kind: str  # "baseline" | "memory" | "agent"
    baseline_kind: str | None = None  # for kind == "baseline"
    config_path: str | None = None  # for kind in {"memory", "agent"}


def venv_python(spec: SystemSpec) -> str | None:
    """Path to the per-system venv interpreter, or None to use the host env."""
    if spec.kind != "memory":
        return None
    stem = (spec.config_path and Path(spec.config_path).stem) or spec.name
    for family, vdir in VENV_DIRS.items():
        if spec.name.startswith(family) or stem.startswith(family):
            python = REPO_ROOT / vdir / "bin" / "python"
            return str(python) if python.exists() else None
    return None


@lru_cache(maxsize=1)
def registry() -> dict[str, SystemSpec]:
    reg: dict[str, SystemSpec] = {}

    # Baselines (in-process, deterministic).
    from amb.benchmark.evaluation.baselines import available_baselines

    for kind in available_baselines():
        reg[kind] = SystemSpec(name=kind, kind="baseline", baseline_kind=kind)

    # Memory systems (configs/real_system/*.json).
    if REAL_SYSTEM_DIR.is_dir():
        for cfg in sorted(REAL_SYSTEM_DIR.glob("*.json")):
            reg.setdefault(cfg.stem, SystemSpec(cfg.stem, "memory", config_path=str(cfg)))

    # Agent systems (configs/agent_systems/**/*.json).
    if AGENT_SYSTEM_DIR.is_dir():
        for cfg in sorted(AGENT_SYSTEM_DIR.rglob("*.json")):
            reg.setdefault(cfg.stem, SystemSpec(cfg.stem, "agent", config_path=str(cfg)))

    # Friendly aliases -> default memory config.
    for alias, stem in MEMORY_ALIASES.items():
        if alias not in reg and stem in reg:
            base = reg[stem]
            reg[alias] = SystemSpec(alias, "memory", config_path=base.config_path)

    return reg


def resolve(name: str, *, config_path: str | None = None, kind: str | None = None) -> SystemSpec:
    """Resolve a system by name, or build an ad-hoc spec from an explicit config."""
    if config_path:
        inferred = kind or ("agent" if "agent_systems" in config_path else "memory")
        return SystemSpec(name=name or Path(config_path).stem, kind=inferred, config_path=config_path)
    reg = registry()
    if name not in reg:
        raise KeyError(
            f"unknown system {name!r}. Use `amb list-systems` to see options, "
            f"or pass --config PATH for an ad-hoc config."
        )
    return reg[name]


def list_systems() -> dict[str, list[str]]:
    by_kind: dict[str, list[str]] = {"baseline": [], "memory": [], "agent": []}
    for spec in registry().values():
        by_kind[spec.kind].append(spec.name)
    return {k: sorted(v) for k, v in by_kind.items()}
