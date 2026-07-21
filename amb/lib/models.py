"""Model axis registry + injection — the extensible knob for the systems×models matrix.

Add a new model = add ONE entry to MODEL_REGISTRY (id -> endpoint + key env + wire).
Inject a model into any system = `apply_model(spec, model_id)` returns a config
patch + env overrides mapped to that system's specific knob, so callers never need
to know each framework's idiosyncratic model field.

Design goals (per owner): extensible (one line per model / per system family) and
easy to use (`amb run --model <id>`; `amb list-models`). Embeddings for memory
systems are held CONSTANT (a single embedder, e.g. bge-m3) so the matrix varies
only the chat/LLM backbone. Endpoints may be given as ``${OPENAI_BASE_URL}``
placeholders in the registry and are resolved from the environment at load time.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _expand_env(value: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` placeholders from the environment.

    Lets ``configs/model_registry.json`` ship endpoint placeholders such as
    ``"${OPENAI_BASE_URL}"`` instead of hardcoded URLs; users set the variable in
    their shell or ``.env``. A literal string with no placeholder is returned
    unchanged, so existing concrete URLs keep working.
    """
    if not isinstance(value, str) or "${" not in value:
        return value

    def _sub(match: "re.Match[str]") -> str:
        name = match.group("name")
        default = match.group("default")
        resolved = os.getenv(name)
        if resolved is None or resolved == "":
            resolved = default if default is not None else ""
        return resolved

    # Innermost ${...} (no nested braces) first; iterate so nested defaults such
    # as ${A:-${B}} resolve fully. Bounded to avoid pathological loops.
    pattern = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^{}]*))?\}")
    for _ in range(10):
        if "${" not in value:
            break
        new_value = pattern.sub(_sub, value)
        if new_value == value:
            break
        value = new_value
    return value

# Model endpoints live in a CONFIG file (not in lib code) so new models/endpoints
# are added by editing JSON — no code change — and lib stays URL-free. API keys
# are read from environment variables named by each entry's key_env (see
# .env.example); nothing secret is stored in code or config.
REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "model_registry.json"


@dataclass(frozen=True)
class ModelSpec:
    id: str                      # canonical matrix id (column label)
    model_name: str              # the name to send to the provider API
    base_url: str
    api_key_env: str
    wire: str = "chat"           # "chat" | "responses" | "anthropic"
    notes: str = ""


@lru_cache(maxsize=1)
def _registry_doc() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


@lru_cache(maxsize=1)
def _registry() -> dict[str, ModelSpec]:
    doc = _registry_doc()
    out: dict[str, ModelSpec] = {}
    for mid, e in doc.get("models", {}).items():
        out[mid] = ModelSpec(mid, e["model_name"], _expand_env(e["base_url"]), e["key_env"],
                             e.get("wire", "chat"), e.get("notes", ""))
    return out


# Embedding endpoint held constant across the whole matrix (from config).
def _embedding() -> dict:
    emb = dict(_registry_doc().get("_embedding", {}))
    if "base_url" in emb:
        emb["base_url"] = _expand_env(emb["base_url"])
    return emb


class _RegistryView(dict):
    """Lazy dict view so `MODEL_REGISTRY[id]` and iteration work off the config."""
    def __getitem__(self, k):
        return _registry()[k]

    def __iter__(self):
        return iter(_registry())

    def __contains__(self, k):
        return k in _registry()

    def get(self, k, default=None):
        return _registry().get(k, default)

    def items(self):
        return _registry().items()


MODEL_REGISTRY = _RegistryView()


def get_model(model_id: str) -> ModelSpec:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_id]


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


# ---- per-system injection map -----------------------------------------------
# How each system family receives its backbone model. memory systems take the
# chat model via a config field (+ base_url/key) and keep the constant embedder;
# agents take it via model_id + env. Locked systems ignore the override.
# Truly locked: cannot take an arbitrary backbone from the model axis.
LOCKED = {
    "lightmem": "local Qwen2.5-0.5B (hardcoded)",
    "memos": "no LLM (pure retrieval) — model-irrelevant",
    "openai_memory": "OpenAI Agents SDK SQLite session store — no LLM in observe/retrieve, model-irrelevant (not N×N)",
    "raw_llm_runner": "deterministic stub (no model)",
}
# codex speaks the Responses API and claude_code speaks Anthropic Messages; a
# serves both (/v1/responses 200, /v1/messages present). They CAN take models from
# the axis on the matching protocol (codex: gpt-5.x via responses; claude_code:
# claude-* via anthropic). The run wrapper points OPENAI_BASE_URL/ANTHROPIC_BASE_URL
# at the model's endpoint; the CLI picks its own protocol.
# memory family -> the client_kwargs field that names the chat LLM
MEMORY_LLM_FIELD = {
    "mem0": "llm_model", "a_mem": "llm_model", "zep_graphiti": "llm_model",
    "memoryos": "llm_model", "letta": "memgpt_llm_model", "langmem": "model",
    "meminsight": "model",  # attribute annotator backbone; embedding_* absorbed/ignored
    "simplemem": "model",   # SimpleMem LLM via env (OPENAI_BASE_URL/LLM_MODEL); embedding_* absorbed
    "evermind": "model",    # Evermind langchain ChatOpenAI=matrix model, embed=bge-m3
    "synap": "model",
    "memoripy": "model",
    "reme": "model",       # Synap pluggable LLMProvider=matrix model, EmbeddingProvider=bge-m3
    "lightrag": "model",   # LightRAG llm_model_func=matrix model, embedding_func=bge-m3
    "nanographrag": "model",  # nano-graphrag best/cheap_model_func=matrix model, embedding_func=bge-m3
    "txtai": "model",  # txtai external embed=bge-m3; no LLM gen (retrieval-only memory)
    "memengine": "model",  # MemEngine LTMemory: APIEncoder embed=bge-m3; no LLM gen (retrieval-only)
}
# agent family -> extra env var carrying the model name (besides model_id)
AGENT_MODEL_ENV = {
    "goose": "GOOSE_MODEL", "openhands": "LLM_MODEL",
    "cline": "LLM_MODEL", "aider": "LLM_MODEL", "swe": "LLM_MODEL",
    "opencode": "OPENCODE_MODEL",  # bare model name; provider "amb" defined in opencode.json by the wrapper
    "continue": "CONTINUE_MODEL",  # model in ~/.continue/config.yaml written by the wrapper
    "crush": "CRUSH_MODEL",  # bare model name; provider "amb" defined in ~/.config/crush/crush.json by the wrapper
    "qwen_code": "QWEN_MODEL",  # passed via -m <model>; OpenAI endpoint+key via OPENAI_* env (wrapper)
    "pi": "PI_MODEL",  # earendil-works/pi; provider "amb" registered via ~/.pi/extensions/amb.js (wrapper)
}


@dataclass
class ModelInjection:
    locked: bool
    reason: str = ""
    config_patch: dict = field(default_factory=dict)   # merged into the system config
    env: dict = field(default_factory=dict)            # exported before the run


def _family(system_name: str) -> str:
    for fam in list(LOCKED) + list(MEMORY_LLM_FIELD) + list(AGENT_MODEL_ENV) + ["openai_compatible_custom"]:
        if system_name.startswith(fam) or fam in system_name:
            return fam
    return system_name


def apply_model(system_name: str, model_id: str) -> ModelInjection:
    """Return how to inject `model_id` into `system_name` (config patch + env)."""
    fam = _family(system_name)
    if fam in LOCKED:
        return ModelInjection(locked=True, reason=LOCKED[fam])

    spec = get_model(model_id)
    # common OpenAI-compatible routing for whatever calls the chat endpoint
    env = {
        "OPENAI_BASE_URL": spec.base_url,
        "OPENAI_API_BASE": spec.base_url,
        "AMB_MODEL_API_KEY_ENV": spec.api_key_env,  # which env holds the key
    }
    patch: dict = {}

    if fam in MEMORY_LLM_FIELD:
        emb = _embedding()
        # Chat LLM on the model's own provider via its key; the embedder stays
        # constant on the embedding endpoint with its own key.
        patch = {"client_kwargs": {
            MEMORY_LLM_FIELD[fam]: spec.model_name,
            "base_url": spec.base_url,
            "api_key_env": spec.api_key_env,             # chat model API key env var
            "embedding_model": emb.get("model_name", "BAAI/bge-m3"),
            "embedding_base_url": emb.get("base_url", "${OPENAI_BASE_URL}"),
            "embedding_api_key_env": emb.get("key_env", "OPENAI_API_KEY"),  # embedding key env
        }}
    elif fam in AGENT_MODEL_ENV:
        # litellm-based agents (openhands/swe/aider) need the "openai/" provider
        # prefix for an OpenAI-compatible endpoint; goose/cline take the bare name.
        litellm = fam in {"openhands", "swe", "aider"}
        mn = f"openai/{spec.model_name}" if litellm else spec.model_name
        patch = {"loader_kwargs": {"model_id": mn}}
        env[AGENT_MODEL_ENV[fam]] = mn
        env["LLM_MODEL"] = mn
        if fam == "goose":
            env["GOOSE_MODEL"] = spec.model_name  # goose wants the bare name
            env["GOOSE_PROVIDER__HOST"] = spec.base_url
    elif fam == "openai_compatible_custom":
        env["AMST_OPENAI_COMPATIBLE_MODEL"] = spec.model_name
        env["AMST_OPENAI_COMPATIBLE_BASE_URL"] = spec.base_url
    else:
        patch = {"loader_kwargs": {"model_id": spec.model_name}}

    return ModelInjection(locked=False, config_patch=patch, env=env)
