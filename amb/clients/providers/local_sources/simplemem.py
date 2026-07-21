"""Local official-source SimpleMem client factory for AutoMemoryBench integration.

SimpleMem (arXiv 2601.02553, github.com/aiming-lab/SimpleMem) is a lifelong-memory
framework that beats mem0 on LoCoMo. It reads its LLM config from env
(``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` / ``LLM_MODEL`` — see simplemem/core/settings.py),
so we inject our endpoint via env and run it against any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path
from typing import Any

from amb.clients.core.common import ensure_site_packages_from_venv, ensure_source_path, venv_site_packages

# SimpleMem's text path pulls openai/lancedb/sentence-transformers from this venv,
# not the pod base interpreter.
DEFAULT_SIMPLEMEM_VENV = Path(__file__).resolve().parents[4] / ".venv-simplemem"
DEFAULT_SIMPLEMEM_SOURCE = "related_work/repos/SimpleMem"
_OPENAI_FALLBACK_VENVS = (
    Path(__file__).resolve().parents[4] / ".venv-meminsight",
    Path(__file__).resolve().parents[4] / ".venv-langmem",
    Path(__file__).resolve().parents[4] / ".venv-mem0",
)
# sentence-transformers (pulled by SimpleMem's text path) needs torch, which lives in
# the NGC base image's system dist-packages — NOT inside .venv-simplemem (that venv only
# --system-site-packages *references* it). The cluster pod's eval interpreter may not have
# this dir on its path, so add it explicitly before importing SimpleMem/sentence-transformers.
_SYSTEM_DIST_PACKAGES = (
    "/usr/local/lib/python3.12/dist-packages",
    "/usr/lib/python3.12/dist-packages",
)


def _ensure_system_torch_on_path() -> None:
    simplemem_site = venv_site_packages(DEFAULT_SIMPLEMEM_VENV)
    anchor = str(simplemem_site) if simplemem_site is not None else None
    anchor_index = _resolved_sys_path_index(anchor) if anchor else None
    insert_at = (anchor_index + 1) if anchor_index is not None else len(sys.path)
    for p in _SYSTEM_DIST_PACKAGES:
        if not os.path.isdir(p):
            continue
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(insert_at, p)
        insert_at += 1


def _append_site_packages_from_venv(venv: Path) -> bool:
    site_packages = venv_site_packages(venv)
    if site_packages is None:
        return False
    raw_path = str(site_packages)
    if raw_path not in sys.path:
        sys.path.append(raw_path)
    return True


def _ensure_openai_sdk_on_path() -> None:
    """SimpleMem imports OpenAI lazily; some cluster pods lack user-site packages."""
    if importlib.util.find_spec("openai") is not None:
        return
    for venv in _OPENAI_FALLBACK_VENVS:
        if not _append_site_packages_from_venv(venv):
            continue
        if importlib.util.find_spec("openai") is not None:
            return
    raise ModuleNotFoundError(
        "SimpleMem requires the openai SDK, but it is not importable from "
        f"{DEFAULT_SIMPLEMEM_VENV} or fallback venvs: "
        f"{', '.join(str(p) for p in _OPENAI_FALLBACK_VENVS)}"
    )


def _ensure_sentence_transformer_deps_on_path() -> None:
    """SimpleMem's embedder imports HF deps lazily from the active process."""
    if importlib.util.find_spec("huggingface_hub") is not None:
        return
    for venv in (Path(__file__).resolve().parents[4] / ".venv-mem0",):
        if not _append_site_packages_from_venv(venv):
            continue
        if importlib.util.find_spec("huggingface_hub") is not None:
            return
    raise ModuleNotFoundError(
        "SimpleMem requires huggingface_hub for its SentenceTransformer embedder, "
        "but it is not importable from the base image or .venv-mem0"
    )


def _ensure_huggingface_hub_compat() -> None:
    """Patch older fallback huggingface_hub builds for SimpleMem's transformers."""
    import huggingface_hub  # type: ignore

    version = str(getattr(huggingface_hub, "__version__", "0"))
    if _version_tuple(version) < (1, 3, 0):
        for name in list(sys.modules):
            if name == "huggingface_hub" or name.startswith("huggingface_hub."):
                sys.modules.pop(name, None)
        import huggingface_hub  # type: ignore[no-redef]

    if not hasattr(huggingface_hub, "is_offline_mode"):
        def _is_offline_mode() -> bool:
            return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}

        huggingface_hub.is_offline_mode = _is_offline_mode  # type: ignore[attr-defined]


def _ensure_simplemem_query_analysis_compat() -> None:
    """Normalize SimpleMem planner output when the LLM returns JSON arrays.

    Official SimpleMem asks the LLM for a JSON object, but OpenAI-compatible
    backbones occasionally return a top-level list. The upstream retriever then
    calls `.get(...)` and crashes before scoring. Coercing that schema drift back
    to the documented object shape preserves the retrieval path instead of
    replacing it with an AMB-specific approximation.
    """
    try:
        from simplemem.core.hybrid_retriever import HybridRetriever  # type: ignore
    except Exception:
        return

    if getattr(HybridRetriever, "_amb_query_analysis_compat", False):
        return
    original = HybridRetriever._analyze_query

    def _analyze_query_compat(self: Any, query: str) -> dict[str, Any]:
        return _normalize_query_analysis(original(self, query), query)

    HybridRetriever._analyze_query = _analyze_query_compat  # type: ignore[assignment]
    HybridRetriever._amb_query_analysis_compat = True  # type: ignore[attr-defined]


def _ensure_simplemem_fts_compat() -> None:
    """Patch SimpleMem's LanceDB FTS init for LanceDB builds without Tantivy.

    Upstream SimpleMem calls ``create_fts_index(..., use_tantivy=True)`` for local
    stores. Current LanceDB rejects that mode and asks callers to use native FTS.
    Keep SimpleMem's official lexical layer, but make the index creation compatible
    with the installed LanceDB version.
    """
    try:
        from simplemem.core.database.vector_store import VectorStore  # type: ignore
    except Exception:
        return

    if getattr(VectorStore, "_amb_native_fts_compat", False):
        return
    VectorStore._init_fts_index = _native_fts_init  # type: ignore[assignment]
    VectorStore._amb_native_fts_compat = True  # type: ignore[attr-defined]


def _native_fts_init(self: Any) -> None:
    if getattr(self, "_fts_initialized", False):
        return

    try:
        try:
            self.table.create_fts_index(
                "lossless_restatement",
                use_tantivy=False,
                replace=True,
            )
        except TypeError:
            # Older LanceDB builds do not expose use_tantivy.
            self.table.create_fts_index(
                "lossless_restatement",
                replace=True,
            )
        print("FTS index created (native mode)")
        self._fts_initialized = True
    except Exception as exc:
        print(f"FTS index creation skipped: {exc}")


def _normalize_query_analysis(value: Any, query: str) -> dict[str, Any]:
    default: dict[str, Any] = {
        "keywords": [query],
        "persons": [],
        "time_expression": None,
        "location": None,
        "entities": [],
    }
    if isinstance(value, dict):
        out = dict(default)
        out.update(value)
        out["keywords"] = _string_list(out.get("keywords")) or [query]
        out["persons"] = _string_list(out.get("persons"))
        out["entities"] = _string_list(out.get("entities"))
        out["time_expression"] = _optional_string(out.get("time_expression"))
        out["location"] = _optional_string(out.get("location"))
        return out
    if isinstance(value, list):
        keywords: list[str] = []
        persons: list[str] = []
        entities: list[str] = []
        time_expression: str | None = None
        location: str | None = None
        for item in value:
            if isinstance(item, dict):
                keywords.extend(_string_list(item.get("keywords")))
                persons.extend(_string_list(item.get("persons")))
                entities.extend(_string_list(item.get("entities")))
                time_expression = time_expression or _optional_string(item.get("time_expression"))
                location = location or _optional_string(item.get("location"))
            elif isinstance(item, str) and item.strip():
                keywords.append(item.strip())
        return {
            "keywords": _dedupe_strings(keywords) or [query],
            "persons": _dedupe_strings(persons),
            "time_expression": time_expression,
            "location": location,
            "entities": _dedupe_strings(entities),
        }
    return default


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return _dedupe_strings(str(item).strip() for item in value if str(item).strip())
    return [str(value).strip()] if str(value).strip() else []


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if not text or text.lower() == "null" else text


def _dedupe_strings(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out


def _version_tuple(raw: str) -> tuple[int, ...]:
    out = []
    for part in raw.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out or [0])


def _resolved_sys_path_index(path: str) -> int | None:
    try:
        target = str(Path(path).resolve())
    except Exception:
        target = path
    for index, item in enumerate(sys.path):
        try:
            current = str(Path(item).resolve())
        except Exception:
            current = item
        if current == target:
            return index
    return None


class SimpleMemOfficialSourceClient:
    """Adapter over SimpleMem's ``add_dialogue`` / ``query`` surface.

    Maps AMB's reset/add/search contract onto a fresh ``SimpleMem`` instance per case.
    """

    def __init__(self, *, default_limit: int = 5) -> None:
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        self.default_limit = default_limit
        self.case_id: str | None = None
        self._mem: Any = None
        self._SimpleMem: Any = None
        self._finalized: bool = False

    def _new_mem(self) -> Any:
        if self._SimpleMem is None:
            from simplemem import SimpleMem  # type: ignore
            self._SimpleMem = SimpleMem
        # No mode arg: router auto-selects the text backend from the first
        # add_dialogue() call (passing mode= here collides with router.create(mode=...)).
        return self._SimpleMem()

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        # Per-case lancedb in pod-local /tmp: LANCEDB_PATH defaults to "./lancedb_data"
        # (CWD-relative). On the cluster every shard's CWD is the repo root on the SHARED
        # volume → concurrent shards contend on one lancedb and hang. Isolate per case.
        import tempfile
        os.environ["LANCEDB_PATH"] = tempfile.mkdtemp(prefix="smlance_")
        self._mem = self._new_mem()
        self._finalized = False
        return {"ok": True}

    def add(
        self,
        content: Any = None,
        *,
        messages: Any = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if self._mem is None:
            self._mem = self._new_mem()
        added = 0
        for speaker, text in _iter_turns(content=content, messages=messages):
            self._mem.add_dialogue(speaker=speaker, content=text)
            added += 1
        return {"added": added, "user_id": user_id or self.case_id}

    def search(
        self,
        query: str | None = None,
        *,
        limit: int | None = None,
        top_k: int | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required for SimpleMem search")
        if self._mem is None:
            return []
        k = int(limit or top_k or self.default_limit)
        backend = self._text_backend()
        if backend is None:
            return []
        if not self._finalized:
            try:
                self._mem.finalize()
            except Exception:
                pass
            self._finalized = True
        # Use SimpleMem's own hybrid retriever (intent-aware planning) — return the
        # retrieved contexts as memory items (AMB generates/scores the response).
        # enable_reflection=False: skip SimpleMem's iterative reflection LLM loop
        # (keeps intent-planning + retrieval; cuts ~1 LLM call/query for cluster feasibility).
        try:
            contexts = backend.hybrid_retriever.retrieve(str(query), enable_reflection=False)
        except TypeError:
            contexts = backend.hybrid_retriever.retrieve(str(query))
        out: list[dict[str, Any]] = []
        for it in list(contexts or [])[:k]:
            text, meta = _context_text(it)
            out.append({"id": meta.get("id", f"simplemem-{len(out)+1}"), "content": text, "metadata": meta})
        return out

    def _text_backend(self) -> Any:
        be = getattr(self._mem, "_backend", None)
        if be is not None and hasattr(be, "hybrid_retriever"):
            return be
        return self._mem if hasattr(self._mem, "hybrid_retriever") else None

    def get_all(self, **_: Any) -> list[dict[str, Any]]:
        return []

    def delete(self, memory_id: str | None = None, **_: Any) -> dict[str, Any]:
        # SimpleMem has no per-id delete in the public surface; reset clears per case.
        return {"deleted": False, "memory_id": memory_id}


def create_client(
    *,
    source_root: str = DEFAULT_SIMPLEMEM_SOURCE,
    default_limit: int = 5,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    api_key: str | None = None,
    **_: Any,  # absorb embedding_* injected by apply_model (text path uses bm25 keyword retrieval)
) -> SimpleMemOfficialSourceClient:
    """Create a SimpleMem client from the cloned official source, on our endpoint.

    Injects model/base_url/api_key via env (SimpleMem's settings layer reads them),
    then ensures the venv site-packages + source path before first import.
    """
    # SimpleMem's released default text config uses keyword retrieval by default
    # (k_sem=0). The embedding object is still constructed for LanceDB schema/write
    # paths, so use the official-source AMB no-op embedder hook to avoid loading a
    # non-ranking SentenceTransformers/HF stack inside cluster pods.
    os.environ.setdefault("SIMPLEMEM_NOOP_EMBED", "1")
    os.environ.setdefault("SIMPLEMEM_NOOP_DIM", "384")
    ensure_site_packages_from_venv(DEFAULT_SIMPLEMEM_VENV)
    _ensure_openai_sdk_on_path()
    _ensure_system_torch_on_path()
    if os.environ.get("SIMPLEMEM_NOOP_EMBED", "").strip().lower() not in {"1", "true", "yes", "on"}:
        _ensure_sentence_transformer_deps_on_path()
        _ensure_huggingface_hub_compat()
    ensure_source_path(source_root)

    if model:
        os.environ["LLM_MODEL"] = str(model)
    # SimpleMem instantiates an EmbeddingModel at init, but its default retrieval
    # config is keyword-only (k_sem=0) so the embedding is not used for ranking.
    # Pin a tiny ST model to avoid the 1.2GB Qwen3-Embedding download.
    os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # Cluster pods have no HF download access → point HF/ST at a volume-resident cache
    # (pre-populated locally) and run offline so the embedder loads from disk, not the network.
    _hf_cache = str(Path(__file__).resolve().parents[4] / "reports" / ".hf_cache")
    os.environ.setdefault("HF_HOME", _hf_cache)
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _hf_cache)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if base_url:
        os.environ["OPENAI_BASE_URL"] = str(base_url)
    key = api_key or (os.environ.get(api_key_env or "OPENAI_API_KEY") or "")
    if key:
        os.environ["OPENAI_API_KEY"] = key
    _ensure_simplemem_query_analysis_compat()
    _ensure_simplemem_fts_compat()
    return SimpleMemOfficialSourceClient(default_limit=default_limit)


def _context_text(it: Any) -> tuple[str, dict[str, Any]]:
    """Normalize a retrieved context (str / dict / MemoryEntry) to (text, metadata)."""
    if isinstance(it, str):
        return it, {}
    if isinstance(it, dict):
        text = it.get("summary") or it.get("content") or it.get("text") or ""
        meta = {k: v for k, v in it.items() if k not in ("summary", "content", "text")}
        return str(text), meta
    # object (e.g. MemoryEntry): SimpleMem stores the canonical text in
    # ``lossless_restatement``; fall back to other common attrs.
    for attr in ("lossless_restatement", "summary", "content", "text"):
        val = getattr(it, attr, None)
        if val:
            return str(val), {"id": getattr(it, "entry_id", None) or getattr(it, "id", None)}
    return str(it), {}


def _iter_turns(*, content: Any, messages: Any):
    """Yield (speaker, text) turns from AMB add() payloads."""
    if messages is not None:
        values = messages if isinstance(messages, list) else [messages]
        for v in values:
            if isinstance(v, dict):
                yield str(v.get("role") or "user"), str(v.get("content", ""))
            else:
                yield "user", str(v)
        return
    if content is None:
        raise ValueError("content or messages is required")
    yield "user", str(content)
