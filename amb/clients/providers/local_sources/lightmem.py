"""Local official-source LightMem client factory for AutoMemoryBench integration runs."""

from __future__ import annotations

import copy
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from amb.clients.core.common import ensure_site_packages_from_venv, ensure_source_path


class LightMemOfficialSourceClient:
    """Thin adapter over the official ``lightmem.memory.LightMemory`` API."""

    def __init__(self, memory_factory: Any, config: dict[str, Any] | None = None, *, default_limit: int = 5) -> None:
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        self._direct_memory_instance = config is None and not callable(memory_factory)
        self._memory_factory = memory_factory
        self._config_template = copy.deepcopy(config or {})
        self._extraction_prompt = self._config_template.pop("amst_extraction_prompt", None)
        self._active_config: dict[str, Any] | None = None
        self.memory: Any | None = memory_factory if self._direct_memory_instance else None
        self.default_limit = default_limit
        self.case_id: str | None = None
        self._raw_observations: list[dict[str, Any]] = []
        self._pending_messages: list[dict[str, Any]] = []

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        self._raw_observations = []
        self._pending_messages = []
        config = copy.deepcopy(self._config_template)
        _scope_lightmem_qdrant_config(config, self.case_id or "default")
        self._active_config = config
        if self._direct_memory_instance:
            return {"ok": True}
        if self.memory is None:
            self.memory = self._memory_factory(config)
        else:
            _reset_lightmem_buffers(self.memory)
            _replace_lightmem_embedding_retriever(self.memory, config)
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
        if self.memory is None:
            self.reset(case_id=user_id or self.case_id or "default")
        message = _message_from_payload(content=content, messages=messages, metadata=metadata)
        self._raw_observations.append(message)
        if self._direct_memory_instance:
            result = self.memory.add_memory(message, force_segment=True, force_extract=True)
        else:
            self._pending_messages.extend(_lightmem_turn_pair(message))
            result = {"buffered": True, "pending_messages": len(self._pending_messages)}
        return {"result": result, "user_id": user_id or self.case_id}

    def search(
        self,
        query: str | None = None,
        *,
        limit: int | None = None,
        top_k: int | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required for LightMem retrieve")
        if self.memory is None:
            self.reset(case_id=self.case_id or "default")
        self._flush_pending()
        k = int(limit or top_k or self.default_limit)
        raw = self.memory.retrieve(str(query), limit=k)
        return [{"id": f"lightmem-{idx}", "content": str(item), "score": None} for idx, item in enumerate(raw, start=1)]

    def get_all(self, **_: Any) -> list[dict[str, Any]]:
        return [{"id": f"observation-{idx}", "content": row.get("content", ""), "metadata": row} for idx, row in enumerate(self._raw_observations, start=1)]

    def delete(self, memory_id: str | None = None, **_: Any) -> dict[str, Any]:
        return {"deleted": False, "reason": "official LightMem source wrapper does not expose per-id delete in this AMST path", "memory_id": memory_id}

    def _flush_pending(self) -> None:
        if self._direct_memory_instance or not self._pending_messages:
            return
        kwargs: dict[str, Any] = {"force_segment": True, "force_extract": True}
        if self._extraction_prompt:
            kwargs["METADATA_GENERATE_PROMPT"] = self._extraction_prompt
        self.memory.add_memory(list(self._pending_messages), **kwargs)
        self._pending_messages = []


def create_client(
    *,
    source_root: str = "related_work/repos/LightMem",
    venv_root: str | None = None,
    config: dict[str, Any] | None = None,
    default_limit: int = 5,
) -> LightMemOfficialSourceClient:
    """Create a client from the locally cloned official LightMem source tree."""

    if venv_root:
        ensure_site_packages_from_venv(Path(venv_root))
    ensure_source_path(source_root)
    from lightmem.memory.lightmem import LightMemory  # type: ignore

    _patch_lightmem_source_id_correction()
    return LightMemOfficialSourceClient(LightMemory.from_config, config or {}, default_limit=default_limit)


def _message_from_payload(*, content: Any, messages: Any, metadata: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(messages, list) and messages:
        first = messages[0]
        if isinstance(first, dict):
            content = first.get("content", content)
    elif isinstance(messages, dict):
        content = messages.get("content", content)
    if content is None:
        raise ValueError("content or messages is required")
    meta = metadata or {}
    timestamp = _lightmem_session_timestamp(meta.get("timestamp"))
    source_role = str(meta.get("role") or "user")
    return {
        "role": "user",
        "speaker_id": source_role,
        "speaker_name": source_role,
        "content": str(content),
        "time_stamp": timestamp,
    }


def _lightmem_session_timestamp(raw: Any) -> str:
    """Normalize upstream benchmark timestamps to LightMem's documented session format."""

    if raw is None:
        return datetime.utcnow().strftime("%Y/%m/%d (%a) %H:%M")
    raw_text = str(raw).strip()
    if not raw_text:
        return datetime.utcnow().strftime("%Y/%m/%d (%a) %H:%M")
    if re.search(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}\s*\([^)]+\)\s*\d{1,2}:\d{2}", raw_text):
        return raw_text
    parsed = _parse_benchmark_timestamp(raw_text)
    return parsed.strftime("%Y/%m/%d (%a) %H:%M")


def _parse_benchmark_timestamp(raw_text: str) -> datetime:
    normalized = re.sub(r"\bon\b", " ", raw_text, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for fmt in (
        "%I:%M %p %d %B, %Y",
        "%I:%M%p %d %B, %Y",
        "%I:%M %p %d %b, %Y",
        "%I:%M%p %d %b, %Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
    ):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    try:
        from dateutil import parser as date_parser  # type: ignore

        return date_parser.parse(raw_text)
    except Exception as exc:
        raise ValueError(f"Could not parse benchmark timestamp for LightMem: {raw_text!r}") from exc


def _lightmem_turn_pair(message: dict[str, Any]) -> list[dict[str, Any]]:
    assistant_turn = dict(message)
    assistant_turn["role"] = "assistant"
    assistant_turn["speaker_id"] = "assistant"
    assistant_turn["speaker_name"] = "assistant"
    assistant_turn["content"] = ""
    return [message, assistant_turn]


def _scope_lightmem_qdrant_config(config: dict[str, Any], case_id: str) -> None:
    """Give each benchmark case an isolated official Qdrant collection/path."""

    retriever = config.get("embedding_retriever")
    if not isinstance(retriever, dict):
        return
    retriever_config = retriever.get("configs")
    if not isinstance(retriever_config, dict):
        return
    safe_case = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)[:120] or "default"
    base_collection = str(retriever_config.get("collection_name") or "lightmem_amst")
    base_path = str(retriever_config.get("path") or "reports/runtime_state/lightmem/qdrant/default")
    retriever_config["collection_name"] = f"{base_collection}_{safe_case}"
    retriever_config["path"] = str(Path(base_path).parent / safe_case)


def _reset_lightmem_buffers(memory: Any) -> None:
    for attr in ("senmem_buffer_manager", "shortmem_buffer_manager"):
        manager = getattr(memory, attr, None)
        if manager is None:
            continue
        if hasattr(manager, "buffer"):
            manager.buffer.clear()
        if hasattr(manager, "big_buffer"):
            manager.big_buffer.clear()
        if hasattr(manager, "token_count"):
            manager.token_count = 0


def _replace_lightmem_embedding_retriever(memory: Any, config: dict[str, Any]) -> None:
    retriever_config = config.get("embedding_retriever")
    if not isinstance(retriever_config, dict):
        return
    from lightmem.configs.retriever.embeddingretriever.base import EmbeddingRetrieverConfig  # type: ignore
    from lightmem.factory.retriever.embeddingretriever.factory import EmbeddingRetrieverFactory  # type: ignore

    parsed_config = EmbeddingRetrieverConfig(**retriever_config)
    memory.embedding_retriever = EmbeddingRetrieverFactory.from_config(parsed_config)
    if hasattr(memory, "config"):
        memory.config.embedding_retriever = parsed_config


def _coerce_lightmem_source_id(raw_source_id: Any) -> int:
    """Coerce LightMem LLM-extracted source IDs without letting non-finite values crash."""

    if raw_source_id is None:
        return 0
    if isinstance(raw_source_id, bool):
        return int(raw_source_id)
    if isinstance(raw_source_id, int):
        return max(raw_source_id, 0)
    try:
        numeric = float(raw_source_id)
    except (TypeError, ValueError, OverflowError):
        return 0
    if not math.isfinite(numeric):
        return 0
    return max(int(numeric), 0)


def _patch_lightmem_source_id_correction() -> None:
    """Apply the source_id clamp that official LightMem logs but does not pass through."""

    import lightmem.memory.lightmem as lightmem_module  # type: ignore
    import lightmem.memory.utils as utils_module  # type: ignore

    if getattr(utils_module, "_amst_source_id_patch", False):
        return
    create_entry = utils_module._create_memory_entry_from_fact

    def convert_extraction_results_to_memory_entries(
        extracted_results: list[Any],
        timestamps_list: list[Any],
        weekday_list: list[Any],
        speaker_list: list[Any] | None = None,
        topic_id_map: dict[int, int] | None = None,
        max_source_ids: list[int] | None = None,
        logger: Any = None,
    ) -> list[Any]:
        topic_id_map = topic_id_map or {}
        memory_entries = []
        extracted_memory_entry = [
            item["cleaned_result"]
            for item in extracted_results
            if item and item.get("cleaned_result")
        ]
        for batch_idx, topic_memory in enumerate(extracted_memory_entry):
            max_valid_sid = max_source_ids[batch_idx] if max_source_ids and batch_idx < len(max_source_ids) else None
            for fact_list in topic_memory:
                if not isinstance(fact_list, list):
                    fact_list = [fact_list]
                for fact_entry in fact_list:
                    corrected = dict(fact_entry)
                    sid = _coerce_lightmem_source_id(corrected.get("source_id", 0))
                    if max_valid_sid is not None and sid > max_valid_sid:
                        if logger:
                            logger.warning(
                                "LLM returned invalid source_id=%s (valid range: [0, %s]); using source_id=%s. Fact: %s...",
                                sid,
                                max_valid_sid,
                                max_valid_sid,
                                str(corrected.get("fact", ""))[:100],
                            )
                        sid = max_valid_sid
                    seq_candidate = sid * 2
                    if seq_candidate not in topic_id_map:
                        valid_sequences = sorted(topic_id_map)
                        if not valid_sequences:
                            continue
                        seq_candidate = min(valid_sequences, key=lambda value: abs(value - seq_candidate))
                        sid = seq_candidate // 2
                    corrected["source_id"] = sid
                    mem_obj = create_entry(
                        corrected,
                        timestamps_list,
                        weekday_list,
                        speaker_list,
                        topic_id=topic_id_map.get(seq_candidate),
                        topic_summary="",
                        logger=logger,
                    )
                    if mem_obj:
                        memory_entries.append(mem_obj)
        return memory_entries

    utils_module.convert_extraction_results_to_memory_entries = convert_extraction_results_to_memory_entries
    lightmem_module.convert_extraction_results_to_memory_entries = convert_extraction_results_to_memory_entries
    utils_module._amst_source_id_patch = True
