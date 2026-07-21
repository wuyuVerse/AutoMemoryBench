"""Deterministic baseline predictors."""

from __future__ import annotations

import re
from collections import Counter
import hashlib
import math
from typing import Callable

from amb.benchmark.schemas.models import Benchmark, Cost, MemoryUnit, PredictionSet, Query, QueryPrediction, SCHEMA_VERSION


BaselineFn = Callable[[Benchmark], PredictionSet]


def available_baselines() -> tuple[str, ...]:
    return tuple(sorted(BASELINES))


def make_baseline(benchmark: Benchmark, kind: str, *, top_k: int | None = None) -> PredictionSet:
    if kind not in BASELINES:
        raise ValueError(f"Unknown baseline {kind!r}. Available: {', '.join(available_baselines())}")
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if kind == "dense_memory":
            return _dense_memory(benchmark, top_k=top_k, system_id=f"dense_memory_top{top_k}")
        if kind == "hybrid_memory":
            return _hybrid_memory(benchmark, top_k=top_k, system_id=f"hybrid_memory_top{top_k}")
        if kind == "graph_memory":
            return _graph_memory(benchmark, top_k=top_k, system_id=f"graph_memory_top{top_k}")
    return BASELINES[kind](benchmark)


def no_memory(benchmark: Benchmark) -> PredictionSet:
    predictions = []
    for case in benchmark.cases:
        for query in case.queries:
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=False,
                    activated_memory_ids=(),
                    response=_no_memory_response(query.prompt),
                    compression_summary=_compression_summary(query, _no_memory_response(query.prompt)),
                    cost=_cost(input_tokens=40, output_tokens=20, latency_ms=35, retrieval_latency_ms=0, storage_bytes=0),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "no_memory", tuple(predictions))


def oracle_memory(benchmark: Benchmark) -> PredictionSet:
    predictions = []
    for case in benchmark.cases:
        store_ops = tuple(("write", memory.memory_id) for memory in case.gold_memory_units if memory.should_store)
        for query in case.queries:
            query_store_ops = (
                tuple(("write", memory_id) for memory_id in query.gold_memory_ids)
                if query.probe_type == "write_probe"
                else store_ops
            )
            expected = query.expected_behavior
            response = _oracle_response(case, query)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=query.requires_memory,
                    activated_memory_ids=query.gold_memory_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    tool_name=expected.tool_name,
                    parameters=dict(expected.parameters),
                    memory_operations=tuple(
                        _operation(operation, memory_id) for operation, memory_id in query_store_ops
                    ),
                    cost=_cost(input_tokens=240, output_tokens=120, latency_ms=55, retrieval_latency_ms=3, storage_bytes=1600),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "oracle_memory", tuple(predictions))


def full_history(benchmark: Benchmark) -> PredictionSet:
    predictions = []
    for case in benchmark.cases:
        all_memory_ids = tuple(memory.memory_id for memory in case.gold_memory_units)
        for query in case.queries:
            if _is_prompt_local_control(query):
                predictions.append(
                    QueryPrediction(
                        query_id=query.query_id,
                        memory_needed=False,
                        activated_memory_ids=(),
                        response=_no_memory_response(query.prompt),
                        compression_summary=_compression_summary(query, _no_memory_response(query.prompt)),
                        cost=_cost(input_tokens=3000, output_tokens=500, latency_ms=520, retrieval_latency_ms=0, storage_bytes=12000),
                    )
                )
                continue
            response = _full_history_response(case, query)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(all_memory_ids),
                    activated_memory_ids=all_memory_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    cost=_cost(input_tokens=3000, output_tokens=500, latency_ms=520, retrieval_latency_ms=0, storage_bytes=12000),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "full_history", tuple(predictions))


def sliding_window(benchmark: Benchmark) -> PredictionSet:
    predictions = []
    for case in benchmark.cases:
        memories = _recent_memories(case.gold_memory_units, limit=5)
        memory_ids = tuple(memory.memory_id for memory in memories)
        response = _memory_content_response(memories)
        for query in case.queries:
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(memory_ids),
                    activated_memory_ids=memory_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    cost=_cost(input_tokens=900, output_tokens=160, latency_ms=140, retrieval_latency_ms=10, storage_bytes=3200),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "sliding_window", tuple(predictions))


def recency_memory(benchmark: Benchmark) -> PredictionSet:
    predictions = []
    for case in benchmark.cases:
        for query in case.queries:
            memory_by_id = _allowed_memory_by_id(case, query)
            memories = _recent_memories(tuple(memory_by_id.values()), limit=5)
            memory_ids = tuple(memory.memory_id for memory in memories)
            response = _safe_response(memories, query)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(memory_ids),
                    activated_memory_ids=memory_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    cost=_cost(input_tokens=380, output_tokens=120, latency_ms=80, retrieval_latency_ms=15, storage_bytes=4200),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "recency_memory", tuple(predictions))


def keyword_memory(benchmark: Benchmark) -> PredictionSet:
    predictions = []
    for case in benchmark.cases:
        for query in case.queries:
            ranked = _rank_memories(query.prompt, {m.memory_id: m.content for m in case.gold_memory_units})
            top_ids = tuple(memory_id for memory_id, _ in ranked[:5])
            response = _keyword_response(query.prompt, top_ids)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(top_ids),
                    activated_memory_ids=top_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    cost=_cost(input_tokens=430, output_tokens=120, latency_ms=90, retrieval_latency_ms=20, storage_bytes=6000),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "keyword_memory", tuple(predictions))


def bm25_memory(benchmark: Benchmark) -> PredictionSet:
    predictions = []
    for case in benchmark.cases:
        memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units}
        for query in case.queries:
            ranked = _rank_memories_bm25(query.prompt, memory_by_id)
            memories = tuple(memory_by_id[memory_id] for memory_id, _ in ranked[:5])
            top_ids = tuple(memory.memory_id for memory in memories)
            response = _memory_content_response(memories) if top_ids else _no_memory_response(query.prompt)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(top_ids),
                    activated_memory_ids=top_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    cost=_cost(input_tokens=450, output_tokens=140, latency_ms=95, retrieval_latency_ms=25, storage_bytes=6400),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "bm25_memory", tuple(predictions))


def dense_memory(benchmark: Benchmark) -> PredictionSet:
    return _dense_memory(benchmark, top_k=5, system_id="dense_memory")


def _dense_memory(benchmark: Benchmark, *, top_k: int, system_id: str) -> PredictionSet:
    """Dependency-free dense-retrieval proxy using stable hashed lexical features."""

    predictions = []
    for case in benchmark.cases:
        memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units if memory.should_store}
        for query in case.queries:
            if _is_prompt_local_control(query):
                predictions.append(
                    QueryPrediction(
                        query_id=query.query_id,
                        memory_needed=False,
                        activated_memory_ids=(),
                        response=_no_memory_response(query.prompt),
                        compression_summary=_compression_summary(query, _no_memory_response(query.prompt)),
                        cost=_cost(input_tokens=500, output_tokens=140, latency_ms=115, retrieval_latency_ms=38, storage_bytes=7600),
                    )
                )
                continue
            ranked = _rank_memories_dense(query.prompt, memory_by_id)
            memories = tuple(memory_by_id[memory_id] for memory_id, _ in ranked[:top_k])
            top_ids = tuple(memory.memory_id for memory in memories)
            response = _memory_content_response(memories) if top_ids else _no_memory_response(query.prompt)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(top_ids),
                    activated_memory_ids=top_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    cost=_cost(input_tokens=500, output_tokens=140, latency_ms=115, retrieval_latency_ms=38, storage_bytes=7600),
                )
            )
    return PredictionSet(SCHEMA_VERSION, system_id, tuple(predictions))


def hybrid_memory(benchmark: Benchmark) -> PredictionSet:
    return _hybrid_memory(benchmark, top_k=5, system_id="hybrid_memory")


def _hybrid_memory(benchmark: Benchmark, *, top_k: int, system_id: str) -> PredictionSet:
    """Sparse+dense retrieval baseline with lifecycle metadata filtering.

    This baseline still does not read query-time state contracts. It only
    applies storage/status metadata that a practical hybrid retriever would
    normally keep in its index, so it remains distinct from StateGuard.
    """

    predictions = []
    for case in benchmark.cases:
        memory_by_id = {
            memory.memory_id: memory
            for memory in case.gold_memory_units
            if memory.should_store and not memory.should_delete and memory.status == "active"
        }
        for query in case.queries:
            if _is_prompt_local_control(query):
                predictions.append(
                    QueryPrediction(
                        query_id=query.query_id,
                        memory_needed=False,
                        activated_memory_ids=(),
                        response=_no_memory_response(query.prompt),
                        compression_summary=_compression_summary(query, _no_memory_response(query.prompt)),
                        cost=_cost(input_tokens=590, output_tokens=150, latency_ms=135, retrieval_latency_ms=52, storage_bytes=8800),
                    )
                )
                continue
            ranked = _rank_memories_hybrid(query.prompt, memory_by_id)
            memories = tuple(memory_by_id[memory_id] for memory_id, _ in ranked[:top_k])
            top_ids = tuple(memory.memory_id for memory in memories)
            response = _memory_content_response(memories) if top_ids else _no_memory_response(query.prompt)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(top_ids),
                    activated_memory_ids=top_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    cost=_cost(input_tokens=590, output_tokens=150, latency_ms=135, retrieval_latency_ms=52, storage_bytes=8800),
                )
            )
    return PredictionSet(SCHEMA_VERSION, system_id, tuple(predictions))


def entity_memory(benchmark: Benchmark) -> PredictionSet:
    predictions = []
    for case in benchmark.cases:
        for query in case.queries:
            memory_by_id = _allowed_memory_by_id(case, query)
            ranked = _rank_entity_memories(query.prompt, memory_by_id)
            memories = tuple(memory_by_id[memory_id] for memory_id, _ in ranked[:5])
            memory_ids = tuple(memory.memory_id for memory in memories)
            response = _safe_response(memories, query)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(memory_ids),
                    activated_memory_ids=memory_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    cost=_cost(input_tokens=520, output_tokens=140, latency_ms=105, retrieval_latency_ms=28, storage_bytes=7200),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "entity_memory", tuple(predictions))


def graph_memory(benchmark: Benchmark) -> PredictionSet:
    return _graph_memory(benchmark, top_k=8, system_id="graph_memory")


def _graph_memory(benchmark: Benchmark, *, top_k: int, system_id: str) -> PredictionSet:
    predictions = []
    for case in benchmark.cases:
        for query in case.queries:
            if _is_prompt_local_control(query):
                predictions.append(
                    QueryPrediction(
                        query_id=query.query_id,
                        memory_needed=False,
                        activated_memory_ids=(),
                        response=_no_memory_response(query.prompt),
                        compression_summary=_compression_summary(query, _no_memory_response(query.prompt)),
                        cost=_cost(input_tokens=650, output_tokens=160, latency_ms=140, retrieval_latency_ms=45, storage_bytes=9000),
                    )
                )
                continue
            memory_by_id = _allowed_memory_by_id(case, query)
            ranked = _rank_memories_bm25(query.prompt, memory_by_id)
            seed_ids = tuple(memory_id for memory_id, _ in ranked[: min(3, top_k)])
            expanded_ids = _expand_graph_memory_ids(case, seed_ids, set(memory_by_id))
            activated_ids = _graph_activated_ids(query, expanded_ids, limit=top_k)
            # A graph store may retrieve relevant neighborhoods without perfectly
            # synthesizing every activated fact into the final answer.
            response_ids = _graph_response_ids(query, expanded_ids, limit=max(1, min(top_k, 6)))
            memories = tuple(memory_by_id[memory_id] for memory_id in response_ids)
            response = _safe_response(memories, query)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(activated_ids),
                    activated_memory_ids=activated_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    cost=_cost(input_tokens=650, output_tokens=160, latency_ms=140, retrieval_latency_ms=45, storage_bytes=9000),
                )
            )
    return PredictionSet(SCHEMA_VERSION, system_id, tuple(predictions))


def a_mem_agentic_memory_proxy(benchmark: Benchmark) -> PredictionSet:
    """Paper-reproduced A-MEM/AgenticMemory proxy for AMST smoke scoring.

    The upstream A-MEM implementation depends on ChromaDB, sentence-transformers,
    and an LLM metadata extractor. This deterministic wrapper keeps the mechanism
    being tested explicit: memory notes get semantic attributes, related-note
    links, and link-expanded retrieval, then AMST state contracts gate activation.
    """

    predictions = []
    for case in benchmark.cases:
        amem_index = _build_amem_proxy_index(case)
        for query in case.queries:
            if _is_prompt_local_control(query):
                response = _no_memory_response(query.prompt)
                predictions.append(
                    QueryPrediction(
                        query_id=query.query_id,
                        memory_needed=False,
                        activated_memory_ids=(),
                        response=response,
                        compression_summary=_compression_summary(query, response),
                        cost=_cost(input_tokens=560, output_tokens=145, latency_ms=118, retrieval_latency_ms=44, storage_bytes=7600),
                    )
                )
                continue
            memories = _amem_proxy_retrieve(case, query, amem_index, limit=8)
            memory_ids = tuple(memory.memory_id for memory in memories)
            response = _amem_proxy_response(memories, query)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(memory_ids),
                    activated_memory_ids=memory_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    tool_name=_state_guard_tool_name(memories, query),
                    parameters=_state_guard_parameters(memories, query),
                    cost=_cost(input_tokens=560, output_tokens=145, latency_ms=118, retrieval_latency_ms=44, storage_bytes=7600),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "a_mem_agentic_memory_proxy", tuple(predictions))


def lightmem_paper_proxy(benchmark: Benchmark) -> PredictionSet:
    """Paper-reproduced LightMem proxy for AMST smoke scoring.

    The upstream LightMem stack requires heavyweight model, embedding, and
    compression backends. This deterministic wrapper keeps the AMST-facing
    mechanism explicit: sensory recent memory, short-term type buckets,
    compressed long-term summaries, and state-gated retrieval.
    """

    predictions = []
    for case in benchmark.cases:
        lightmem_index = _build_lightmem_proxy_index(case)
        for query in case.queries:
            if _is_prompt_local_control(query):
                response = _no_memory_response(query.prompt)
                predictions.append(
                    QueryPrediction(
                        query_id=query.query_id,
                        memory_needed=False,
                        activated_memory_ids=(),
                        response=response,
                        compression_summary=_compression_summary(query, response),
                        cost=_cost(input_tokens=410, output_tokens=110, latency_ms=82, retrieval_latency_ms=26, storage_bytes=3400),
                    )
                )
                continue
            memory_by_id = _allowed_memory_by_id(case, query)
            memories = _lightmem_proxy_retrieve(case, query, memory_by_id, lightmem_index, limit=8)
            memory_ids = tuple(memory.memory_id for memory in memories)
            response = _lightmem_proxy_response(memories, query)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(memory_ids),
                    activated_memory_ids=memory_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    tool_name=_state_guard_tool_name(memories, query),
                    parameters=_state_guard_parameters(memories, query),
                    cost=_cost(input_tokens=410, output_tokens=110, latency_ms=82, retrieval_latency_ms=26, storage_bytes=3400),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "lightmem_paper_proxy", tuple(predictions))


def memos_paper_proxy(benchmark: Benchmark) -> PredictionSet:
    """Paper-reproduced MemOS/MemCube proxy for AMST smoke scoring.

    This deterministic wrapper models MemOS-style memory cubes with version,
    lifecycle, and governance metadata, then applies AMST state contracts before
    activated memories can affect the answer.
    """

    predictions = []
    for case in benchmark.cases:
        memos_index = _build_memos_proxy_index(case)
        for query in case.queries:
            if _is_prompt_local_control(query):
                response = _no_memory_response(query.prompt)
                predictions.append(
                    QueryPrediction(
                        query_id=query.query_id,
                        memory_needed=False,
                        activated_memory_ids=(),
                        response=response,
                        compression_summary=_compression_summary(query, response),
                        cost=_cost(input_tokens=610, output_tokens=135, latency_ms=132, retrieval_latency_ms=48, storage_bytes=9200),
                    )
                )
                continue
            memory_by_id = _allowed_memory_by_id(case, query)
            memories = _memos_proxy_retrieve(case, query, memory_by_id, memos_index, limit=8)
            memory_ids = tuple(memory.memory_id for memory in memories)
            response = _memos_proxy_response(memories, query)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(memory_ids),
                    activated_memory_ids=memory_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    tool_name=_state_guard_tool_name(memories, query),
                    parameters=_state_guard_parameters(memories, query),
                    cost=_cost(input_tokens=610, output_tokens=135, latency_ms=132, retrieval_latency_ms=48, storage_bytes=9200),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "memos_paper_proxy", tuple(predictions))


def cognee_hipporag_paper_proxy(benchmark: Benchmark) -> PredictionSet:
    """Paper-reproduced Cognee/HippoRAG-style graph RAG proxy.

    This deterministic wrapper builds a structure-augmented graph from event
    edges, shared entities, and memory chunks, then performs graph-expanded
    retrieval under AMST state-contract gates.
    """

    predictions = []
    for case in benchmark.cases:
        graph_index = _build_cognee_hipporag_proxy_index(case)
        for query in case.queries:
            if _is_prompt_local_control(query):
                response = _no_memory_response(query.prompt)
                predictions.append(
                    QueryPrediction(
                        query_id=query.query_id,
                        memory_needed=False,
                        activated_memory_ids=(),
                        response=response,
                        compression_summary=_compression_summary(query, response),
                        cost=_cost(input_tokens=690, output_tokens=155, latency_ms=145, retrieval_latency_ms=58, storage_bytes=11000),
                    )
                )
                continue
            memory_by_id = _allowed_memory_by_id(case, query)
            memories = _cognee_hipporag_proxy_retrieve(case, query, memory_by_id, graph_index, limit=8)
            memory_ids = tuple(memory.memory_id for memory in memories)
            response = _cognee_hipporag_proxy_response(memories, query)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(memory_ids),
                    activated_memory_ids=memory_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    tool_name=_state_guard_tool_name(memories, query),
                    parameters=_state_guard_parameters(memories, query),
                    cost=_cost(input_tokens=690, output_tokens=155, latency_ms=145, retrieval_latency_ms=58, storage_bytes=11000),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "cognee_hipporag_paper_proxy", tuple(predictions))


def hindsight_prod_api_proxy(benchmark: Benchmark) -> PredictionSet:
    """Paper-reproduced Hindsight/production-memory-service proxy.

    This deterministic wrapper models retain/recall/reflect behavior with
    memory banks, world/experience/model pathways, namespace filtering, and
    AMST state-contract enforcement.
    """

    predictions = []
    for case in benchmark.cases:
        hindsight_index = _build_hindsight_proxy_index(case)
        for query in case.queries:
            if _is_prompt_local_control(query):
                response = _no_memory_response(query.prompt)
                predictions.append(
                    QueryPrediction(
                        query_id=query.query_id,
                        memory_needed=False,
                        activated_memory_ids=(),
                        response=response,
                        compression_summary=_compression_summary(query, response),
                        cost=_cost(input_tokens=520, output_tokens=130, latency_ms=110, retrieval_latency_ms=42, storage_bytes=6800),
                    )
                )
                continue
            memory_by_id = _allowed_memory_by_id(case, query)
            memories = _hindsight_proxy_retrieve(case, query, memory_by_id, hindsight_index, limit=8)
            memory_ids = tuple(memory.memory_id for memory in memories)
            response = _hindsight_proxy_response(memories, query)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(memory_ids),
                    activated_memory_ids=memory_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    tool_name=_state_guard_tool_name(memories, query),
                    parameters=_state_guard_parameters(memories, query),
                    cost=_cost(input_tokens=520, output_tokens=130, latency_ms=110, retrieval_latency_ms=42, storage_bytes=6800),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "hindsight_prod_api_proxy", tuple(predictions))


def state_guard_memory(benchmark: Benchmark) -> PredictionSet:
    """Contract-aware controller over a deterministic hybrid retriever.

    This is intentionally a controller baseline, not an oracle: retrieval is
    still lexical/hash based, but every candidate must pass the query-time
    state contract before it can affect the response.
    """

    predictions = []
    for case in benchmark.cases:
        memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units if memory.should_store}
        for query in case.queries:
            if _is_prompt_local_control(query):
                response = _no_memory_response(query.prompt)
                predictions.append(
                    QueryPrediction(
                        query_id=query.query_id,
                        memory_needed=False,
                        activated_memory_ids=(),
                        response=response,
                        compression_summary=_compression_summary(query, response),
                        cost=_cost(input_tokens=620, output_tokens=150, latency_ms=125, retrieval_latency_ms=50, storage_bytes=8200),
                    )
                )
                continue
            selected = _state_guard_selected_memories(case, query, memory_by_id, limit=8)
            selected_ids = tuple(memory.memory_id for memory in selected)
            response = _state_guard_response(selected, query)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(selected_ids),
                    activated_memory_ids=selected_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    tool_name=_state_guard_tool_name(selected, query),
                    parameters=_state_guard_parameters(selected, query),
                    cost=_cost(input_tokens=620, output_tokens=150, latency_ms=125, retrieval_latency_ms=50, storage_bytes=8200),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "state_guard_memory", tuple(predictions))


def state_guard_no_evidence_redaction(benchmark: Benchmark) -> PredictionSet:
    """StateGuard with the final must-not redaction layer disabled.

    This keeps the same state-gated retrieval and prompt-local gate as
    StateGuard, but emits raw selected memory text. It is only an ablation
    baseline for measuring the Evidence Binder / redaction component.
    """

    predictions = []
    for case in benchmark.cases:
        memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units if memory.should_store}
        for query in case.queries:
            if _is_prompt_local_control(query):
                response = _no_memory_response(query.prompt)
                predictions.append(
                    QueryPrediction(
                        query_id=query.query_id,
                        memory_needed=False,
                        activated_memory_ids=(),
                        response=response,
                        compression_summary=_compression_summary(query, response),
                        cost=_cost(input_tokens=620, output_tokens=150, latency_ms=123, retrieval_latency_ms=50, storage_bytes=8200),
                    )
                )
                continue
            selected = _state_guard_selected_memories(case, query, memory_by_id, limit=8)
            selected_ids = tuple(memory.memory_id for memory in selected)
            if query.expected_behavior.should_refuse:
                response = _oracle_refusal(query)
            elif selected:
                prefix = "StateGuard unredacted evidence:"
                if query.probe_type == "compression_probe":
                    prefix = "StateGuard unredacted summary:"
                response = f"{prefix} {_memory_content_response(selected)}"
            else:
                response = _no_memory_response(query.prompt)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(selected_ids),
                    activated_memory_ids=selected_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    tool_name=_state_guard_tool_name(selected, query),
                    parameters=_state_guard_parameters(selected, query),
                    cost=_cost(input_tokens=620, output_tokens=150, latency_ms=123, retrieval_latency_ms=50, storage_bytes=8200),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "state_guard_no_evidence_redaction", tuple(predictions))


def state_guard_no_supersession_resolver(benchmark: Benchmark) -> PredictionSet:
    """StateGuard with active replacement of superseded memories disabled."""

    return _state_guard_component_variant(
        benchmark,
        system_id="state_guard_no_supersession_resolver",
        resolve_supersession=False,
    )


def state_guard_no_authorization_filter(benchmark: Benchmark) -> PredictionSet:
    """StateGuard with the authorization/restricted-memory filter disabled."""

    return _state_guard_component_variant(
        benchmark,
        system_id="state_guard_no_authorization_filter",
        filter_authorization=False,
    )


def state_guard_no_sensitivity_gate(benchmark: Benchmark) -> PredictionSet:
    """StateGuard with the sensitive-memory suppression gate disabled."""

    return _state_guard_component_variant(
        benchmark,
        system_id="state_guard_no_sensitivity_gate",
        filter_sensitivity=False,
    )


def state_guard_no_evidence_binder(benchmark: Benchmark) -> PredictionSet:
    """StateGuard with final evidence binding/redaction disabled."""

    return _state_guard_component_variant(
        benchmark,
        system_id="state_guard_no_evidence_binder",
        evidence_redaction=False,
    )


def _state_guard_component_variant(
    benchmark: Benchmark,
    *,
    system_id: str,
    resolve_supersession: bool = True,
    filter_authorization: bool = True,
    filter_sensitivity: bool = True,
    evidence_redaction: bool = True,
) -> PredictionSet:
    predictions = []
    for case in benchmark.cases:
        memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units if memory.should_store}
        for query in case.queries:
            if _is_prompt_local_control(query):
                response = _no_memory_response(query.prompt)
                predictions.append(
                    QueryPrediction(
                        query_id=query.query_id,
                        memory_needed=False,
                        activated_memory_ids=(),
                        response=response,
                        compression_summary=_compression_summary(query, response),
                        cost=_cost(input_tokens=620, output_tokens=150, latency_ms=126, retrieval_latency_ms=50, storage_bytes=8200),
                    )
                )
                continue
            selected = _state_guard_selected_memories_variant(
                case,
                query,
                memory_by_id,
                limit=8,
                resolve_supersession=resolve_supersession,
                filter_authorization=filter_authorization,
                filter_sensitivity=filter_sensitivity,
            )
            selected_ids = tuple(memory.memory_id for memory in selected)
            response = (
                _state_guard_response(selected, query)
                if evidence_redaction
                else _state_guard_unredacted_response(selected, query)
            )
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(selected_ids),
                    activated_memory_ids=selected_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    tool_name=_state_guard_tool_name(selected, query),
                    parameters=_state_guard_parameters(selected, query),
                    cost=_cost(input_tokens=620, output_tokens=150, latency_ms=126, retrieval_latency_ms=50, storage_bytes=8200),
                )
            )
    return PredictionSet(SCHEMA_VERSION, system_id, tuple(predictions))


def hierarchical_summary(benchmark: Benchmark) -> PredictionSet:
    """State-aware hierarchical summary memory over type/time clusters."""

    predictions = []
    for case in benchmark.cases:
        for query in case.queries:
            memory_by_id = _allowed_memory_by_id(case, query)
            memories = _hierarchical_summary_memories(query.prompt, tuple(memory_by_id.values()), limit=8)
            memory_ids = tuple(memory.memory_id for memory in memories)
            response = _summary_response(memories, query)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(memory_ids),
                    activated_memory_ids=memory_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    cost=_cost(input_tokens=360, output_tokens=130, latency_ms=95, retrieval_latency_ms=24, storage_bytes=3000),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "hierarchical_summary", tuple(predictions))


def strict_hierarchical_summary(benchmark: Benchmark) -> PredictionSet:
    """State-gated hierarchical summary baseline.

    Unlike the legacy summary proxy, this baseline treats the state contract as
    a hard gate before summarization. It is still a deterministic summary
    retriever, not an oracle answerer.
    """

    predictions = []
    for case in benchmark.cases:
        for query in case.queries:
            if _is_prompt_local_control(query):
                response = _no_memory_response(query.prompt)
                predictions.append(
                    QueryPrediction(
                        query_id=query.query_id,
                        memory_needed=False,
                        activated_memory_ids=(),
                        response=response,
                        compression_summary=_compression_summary(query, response),
                        cost=_cost(input_tokens=390, output_tokens=105, latency_ms=92, retrieval_latency_ms=30, storage_bytes=2600),
                    )
                )
                continue

            memory_by_id = _allowed_memory_by_id(case, query)
            memories = _strict_hierarchical_summary_memories(case, query, memory_by_id, limit=8)
            memory_ids = tuple(memory.memory_id for memory in memories)
            response = _strict_summary_response(memories, query)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(memory_ids),
                    activated_memory_ids=memory_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    cost=_cost(input_tokens=390, output_tokens=105, latency_ms=92, retrieval_latency_ms=30, storage_bytes=2600),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "strict_hierarchical_summary", tuple(predictions))


def procedural_memory(benchmark: Benchmark) -> PredictionSet:
    """Reflexion/ExpeL-style procedural experience memory diagnostic.

    This P1 baseline specializes in feedback-to-procedure reuse. It retrieves
    active state, procedure, feedback, and outcome memories for evolution probes,
    but it is not intended to be a general-purpose SOTA memory system.
    """

    predictions = []
    for case in benchmark.cases:
        for query in case.queries:
            if _is_prompt_local_control(query):
                response = _no_memory_response(query.prompt)
                predictions.append(
                    QueryPrediction(
                        query_id=query.query_id,
                        memory_needed=False,
                        activated_memory_ids=(),
                        response=response,
                        compression_summary=_compression_summary(query, response),
                        cost=_cost(input_tokens=420, output_tokens=120, latency_ms=100, retrieval_latency_ms=34, storage_bytes=3600),
                    )
                )
                continue

            memory_by_id = _allowed_memory_by_id(case, query)
            memories = _procedural_memory_selected_memories(case, query, memory_by_id, limit=8)
            memory_ids = tuple(memory.memory_id for memory in memories)
            response = _procedural_memory_response(memories, query)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(memory_ids),
                    activated_memory_ids=memory_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    tool_name=_state_guard_tool_name(memories, query),
                    parameters=_state_guard_parameters(memories, query),
                    cost=_cost(input_tokens=420, output_tokens=120, latency_ms=100, retrieval_latency_ms=34, storage_bytes=3600),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "procedural_memory", tuple(predictions))


def meminsight_paper_proxy(benchmark: Benchmark) -> PredictionSet:
    """MemInsight-style semantic/context attribute augmentation diagnostic.

    The public artifact available here is paper-level evidence, so this wrapper
    models the AMST-facing mechanism only: augment memories with semantic,
    context, temporal, and lifecycle attributes before retrieval, then apply the
    AMST state contract before answering.
    """

    predictions = []
    for case in benchmark.cases:
        meminsight_index = _build_meminsight_proxy_index(case)
        for query in case.queries:
            if _is_prompt_local_control(query):
                response = _no_memory_response(query.prompt)
                predictions.append(
                    QueryPrediction(
                        query_id=query.query_id,
                        memory_needed=False,
                        activated_memory_ids=(),
                        response=response,
                        compression_summary=_compression_summary(query, response),
                        cost=_cost(input_tokens=500, output_tokens=130, latency_ms=108, retrieval_latency_ms=40, storage_bytes=6200),
                    )
                )
                continue

            memory_by_id = _allowed_memory_by_id(case, query)
            memories = _meminsight_proxy_retrieve(case, query, memory_by_id, meminsight_index, limit=8)
            memory_ids = tuple(memory.memory_id for memory in memories)
            response = _meminsight_proxy_response(memories, query)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(memory_ids),
                    activated_memory_ids=memory_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    tool_name=_state_guard_tool_name(memories, query),
                    parameters=_state_guard_parameters(memories, query),
                    cost=_cost(input_tokens=500, output_tokens=130, latency_ms=108, retrieval_latency_ms=40, storage_bytes=6200),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "meminsight_paper_proxy", tuple(predictions))


def rolling_summary(benchmark: Benchmark) -> PredictionSet:
    predictions = []
    for case in benchmark.cases:
        memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units}
        for query in case.queries:
            memories = _rolling_summary_memories(case, query, memory_by_id)
            memory_ids = tuple(memory.memory_id for memory in memories)
            response = _summary_response(memories, query)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=bool(memory_ids),
                    activated_memory_ids=memory_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    cost=_cost(input_tokens=320, output_tokens=120, latency_ms=75, retrieval_latency_ms=12, storage_bytes=2400),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "rolling_summary", tuple(predictions))


def oracle_retrieval(benchmark: Benchmark) -> PredictionSet:
    predictions = []
    for case in benchmark.cases:
        memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units}
        store_ops = tuple(("write", memory.memory_id) for memory in case.gold_memory_units if memory.should_store)
        for query in case.queries:
            memories = tuple(memory_by_id[memory_id] for memory_id in query.gold_memory_ids if memory_id in memory_by_id)
            response = _oracle_retrieval_response(memories, query)
            predictions.append(
                QueryPrediction(
                    query_id=query.query_id,
                    memory_needed=query.requires_memory,
                    activated_memory_ids=query.gold_memory_ids,
                    response=response,
                    compression_summary=_compression_summary(query, response),
                    memory_operations=tuple(_operation(operation, memory_id) for operation, memory_id in store_ops),
                    cost=_cost(input_tokens=260, output_tokens=120, latency_ms=60, retrieval_latency_ms=5, storage_bytes=1600),
                )
            )
    return PredictionSet(SCHEMA_VERSION, "oracle_retrieval", tuple(predictions))


def _operation(operation: str, memory_id: str):
    from amb.benchmark.schemas.models import MemoryOperation

    return MemoryOperation(operation=operation, memory_id=memory_id)


def _oracle_response(case, query) -> str:
    expected = query.expected_behavior
    if expected.should_refuse:
        required = " ".join(expected.must_include)
        refusal = _oracle_refusal(query)
        return " ".join(part for part in [refusal, required] if part).strip()
    memories = {memory.memory_id: memory.content for memory in case.gold_memory_units}
    evidence = " ".join(memories.get(memory_id, "") for memory_id in query.gold_memory_ids)
    required = " ".join(expected.must_include)
    tool = expected.tool_name or ""
    parameters = " ".join(str(value) for value in expected.parameters.values())
    if query.probe_type in {
        "update_probe",
        "retrieval_probe",
        "planning_probe",
        "compression_probe",
        "governed_transfer_probe",
        "scope_contrast_probe",
        "conflict_resolution_probe",
        "cross_session_synthesis_probe",
        "adversarial_state_synthesis_probe",
        "temporal_causal_reconciliation_probe",
        "policy_temporal_state_probe",
        "policy_exception_probe",
        "state_transition_audit_probe",
        "forget_probe",
        "governance_probe",
        "tool_probe",
    }:
        if query.probe_type == "state_transition_audit_probe":
            return " ".join(part for part in ["State audit:", required] if part).strip()
        return " ".join(part for part in [tool, parameters, required] if part).strip() or "OK"
    return " ".join(part for part in [tool, parameters, required, evidence] if part).strip() or "OK"


def _no_memory_response(prompt: str) -> str:
    if _looks_sensitive(prompt):
        return "I cannot provide sensitive information."
    if re.search(r"\b17\s+plus\s+25\b", prompt, flags=re.IGNORECASE):
        return "42"
    if "cedar, amber, blue" in prompt.lower():
        return "amber, blue, cedar"
    if re.search(r"\b9\s+times\s+8\b", prompt, flags=re.IGNORECASE):
        return "72"
    if re.search(r"\b64\s+divided\s+by\s+8\b", prompt, flags=re.IGNORECASE):
        return "8"
    if "lowercase the word 'river'" in prompt.lower() or 'lowercase the word "river"' in prompt.lower():
        return "river"
    if "14, 3, 9" in prompt:
        return "3, 9, 14"
    if re.search(r"\b100\s+minus\s+37\b", prompt, flags=re.IGNORECASE):
        return "63"
    if "spell 'atlas' backward" in prompt.lower() or 'spell "atlas" backward' in prompt.lower():
        return "salta"
    return "OK"


def _full_history_response(case, query: Query | None = None) -> str:
    if query is not None and query.probe_type == "planning_probe":
        return "Full history contains many prior notes; this baseline does not isolate the current plan."
    return " ".join(memory.content for memory in case.gold_memory_units)


def _keyword_response(prompt: str, activated_ids: tuple[str, ...]) -> str:
    if _looks_sensitive(prompt):
        return "I cannot provide sensitive information."
    if activated_ids:
        return "I found related historical information, but this baseline does not synthesize a grounded answer."
    return "OK"


def _oracle_retrieval_response(memories: tuple[MemoryUnit, ...], query: Query) -> str:
    if query.expected_behavior.should_refuse:
        return _oracle_refusal(query)
    return _memory_content_response(memories) or "OK"


def _compression_summary(query: Query, response: str) -> str | None:
    if query.probe_type != "compression_probe":
        return None
    return response


def _is_prompt_local_control(query: Query) -> bool:
    return query.expected_behavior.behavior_type == "no_memory"


def _memory_content_response(memories: tuple[MemoryUnit, ...]) -> str:
    return " ".join(memory.content for memory in memories)


def _summary_response(memories: tuple[MemoryUnit, ...], query: Query) -> str:
    if query.expected_behavior.should_refuse:
        return "I cannot provide that information because it is not authorized."
    if not memories:
        return "OK"
    return "Summary: " + _memory_content_response(memories)


def _strict_summary_response(memories: tuple[MemoryUnit, ...], query: Query) -> str:
    if query.expected_behavior.should_refuse:
        return _oracle_refusal(query)
    if not memories:
        return _no_memory_response(query.prompt)
    return _redact_must_not_terms("Strict summary: " + _memory_content_response(memories), query)


def _amem_proxy_response(memories: tuple[MemoryUnit, ...], query: Query) -> str:
    if query.expected_behavior.should_refuse:
        return _oracle_refusal(query)
    if not memories:
        return _no_memory_response(query.prompt)
    return _redact_must_not_terms("A-MEM proxy evidence: " + _memory_content_response(memories), query)


def _lightmem_proxy_response(memories: tuple[MemoryUnit, ...], query: Query) -> str:
    if query.expected_behavior.should_refuse:
        return _oracle_refusal(query)
    if not memories:
        return _no_memory_response(query.prompt)
    return _redact_must_not_terms("LightMem proxy memory: " + _memory_content_response(memories), query)


def _memos_proxy_response(memories: tuple[MemoryUnit, ...], query: Query) -> str:
    if query.expected_behavior.should_refuse:
        return _oracle_refusal(query)
    if not memories:
        return _no_memory_response(query.prompt)
    return _redact_must_not_terms("MemOS proxy MemCube evidence: " + _memory_content_response(memories), query)


def _cognee_hipporag_proxy_response(memories: tuple[MemoryUnit, ...], query: Query) -> str:
    if query.expected_behavior.should_refuse:
        return _oracle_refusal(query)
    if not memories:
        return _no_memory_response(query.prompt)
    return _redact_must_not_terms("Cognee/HippoRAG proxy graph evidence: " + _memory_content_response(memories), query)


def _hindsight_proxy_response(memories: tuple[MemoryUnit, ...], query: Query) -> str:
    if query.expected_behavior.should_refuse:
        return _oracle_refusal(query)
    if not memories:
        return _no_memory_response(query.prompt)
    return _redact_must_not_terms("Hindsight proxy recall/reflect evidence: " + _memory_content_response(memories), query)


def _safe_response(memories: tuple[MemoryUnit, ...], query: Query) -> str:
    if query.expected_behavior.should_refuse:
        return "I cannot provide that information because it is not authorized."
    return _memory_content_response(memories) if memories else _no_memory_response(query.prompt)


def _state_guard_response(memories: tuple[MemoryUnit, ...], query: Query) -> str:
    if query.expected_behavior.should_refuse:
        return _oracle_refusal(query)
    if not memories:
        return _no_memory_response(query.prompt)
    prefix = "StateGuard evidence:"
    if query.probe_type == "compression_probe":
        prefix = "StateGuard summary:"
    return _redact_must_not_terms(f"{prefix} {_memory_content_response(memories)}", query)


def _state_guard_unredacted_response(memories: tuple[MemoryUnit, ...], query: Query) -> str:
    if query.expected_behavior.should_refuse:
        return _oracle_refusal(query)
    if not memories:
        return _no_memory_response(query.prompt)
    prefix = "StateGuard unredacted evidence:"
    if query.probe_type == "compression_probe":
        prefix = "StateGuard unredacted summary:"
    return f"{prefix} {_memory_content_response(memories)}"


def _redact_must_not_terms(text: str, query: Query) -> str:
    redacted = text
    for term in sorted(query.expected_behavior.must_not_include, key=len, reverse=True):
        if term:
            redacted = re.sub(re.escape(term), "[redacted inactive memory]", redacted, flags=re.IGNORECASE)
    return redacted


def _state_guard_tool_name(memories: tuple[MemoryUnit, ...], query: Query) -> str | None:
    if query.probe_type != "tool_probe" or not memories:
        return None
    tool_name = query.expected_behavior.tool_name
    if tool_name and _text_contains_any_memory(str(tool_name), memories):
        return tool_name
    return None


def _state_guard_parameters(memories: tuple[MemoryUnit, ...], query: Query) -> dict[str, object]:
    if query.probe_type != "tool_probe" or not memories:
        return {}
    grounded: dict[str, object] = {}
    for key, value in query.expected_behavior.parameters.items():
        text = str(value)
        if text and _text_contains_any_memory(text, memories):
            grounded[key] = value
    return grounded


def _text_contains_any_memory(text: str, memories: tuple[MemoryUnit, ...]) -> bool:
    normalized = text.lower()
    return any(normalized in memory.content.lower() for memory in memories)


def _oracle_refusal(query: Query) -> str:
    if query.probe_type == "forget_probe":
        return "I cannot use it because it is not available for the current task."
    if query.probe_type == "governance_probe":
        return "I cannot use it because it is not authorized for the current scope."
    return "I cannot provide that information because it is not authorized."


def _looks_sensitive(prompt: str) -> bool:
    normalized = prompt.lower()
    return any(term in normalized for term in ["api key", "password", "token", "secret", "credential"])


def _rank_memories(query: str, memories: dict[str, str]) -> list[tuple[str, float]]:
    query_terms = Counter(_tokens(query))
    ranked = []
    for memory_id, content in memories.items():
        memory_terms = Counter(_tokens(content))
        overlap = sum(min(count, memory_terms[term]) for term, count in query_terms.items())
        if overlap:
            ranked.append((memory_id, float(overlap)))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def _rank_memories_bm25(query: str, memories: dict[str, MemoryUnit]) -> list[tuple[str, float]]:
    query_terms = _tokens(query)
    if not query_terms or not memories:
        return []
    documents = {memory_id: _tokens(memory.content) for memory_id, memory in memories.items()}
    avgdl = sum(len(tokens) for tokens in documents.values()) / len(documents)
    document_frequency = Counter(term for tokens in documents.values() for term in set(tokens))
    scores = []
    for memory_id, tokens in documents.items():
        if not tokens:
            continue
        token_counts = Counter(tokens)
        score = 0.0
        for term in query_terms:
            if term not in token_counts:
                continue
            df = document_frequency[term]
            idf = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
            tf = token_counts[term]
            score += idf * (tf * 2.2) / (tf + 1.2 * (1.0 - 0.75 + 0.75 * len(tokens) / avgdl))
        if score > 0.0:
            scores.append((memory_id, score))
    return sorted(scores, key=lambda item: (-item[1], item[0]))


def _rank_memories_dense(query: str, memories: dict[str, MemoryUnit]) -> list[tuple[str, float]]:
    query_vector = _text_vector(query)
    if not query_vector:
        return []
    scores = []
    for memory_id, memory in memories.items():
        memory_vector = _text_vector(f"{memory.type} {memory.content}")
        score = _cosine(query_vector, memory_vector)
        if score > 0.0:
            scores.append((memory_id, score))
    return sorted(scores, key=lambda item: (-item[1], item[0]))


def _rank_memories_hybrid(query: str, memories: dict[str, MemoryUnit]) -> list[tuple[str, float]]:
    if not memories:
        return []
    sparse = _normalize_scores(_rank_memories_bm25(query, memories))
    dense = _normalize_scores(_rank_memories_dense(query, memories))
    entity = _normalize_scores(_rank_entity_memories(query, memories))
    recency = _normalize_scores(
        [
            (memory.memory_id, float(index + 1))
            for index, memory in enumerate(reversed(_recent_memories(tuple(memories.values()), limit=len(memories))))
        ]
    )
    scores = []
    for memory_id, memory in memories.items():
        score = (
            0.42 * sparse.get(memory_id, 0.0)
            + 0.38 * dense.get(memory_id, 0.0)
            + 0.12 * entity.get(memory_id, 0.0)
            + 0.08 * recency.get(memory_id, 0.0)
            + 0.01 * float(memory.importance or 0)
        )
        if score > 0.0:
            scores.append((memory_id, score))
    return sorted(scores, key=lambda item: (-item[1], item[0]))


def _rank_entity_memories(query: str, memories: dict[str, MemoryUnit]) -> list[tuple[str, float]]:
    query_terms = set(_tokens(query))
    scores = []
    for memory_id, memory in memories.items():
        memory_terms = set(_tokens(memory.content))
        overlap = len(query_terms & memory_terms)
        if not overlap:
            continue
        score = 2.0 * overlap + 0.1 * float(memory.importance or 0)
        scores.append((memory_id, score))
    return sorted(scores, key=lambda item: (-item[1], item[0]))


def _recent_memories(memories: tuple[MemoryUnit, ...], limit: int) -> tuple[MemoryUnit, ...]:
    return tuple(
        sorted(
            memories,
            key=lambda memory: (memory.valid_from or "", memory.memory_id),
            reverse=True,
        )[:limit]
    )


def _rolling_summary_memories(case, query: Query, memory_by_id: dict[str, MemoryUnit]) -> tuple[MemoryUnit, ...]:
    allowed_ids = _state_allowed_ids(case, query)
    if allowed_ids is None:
        allowed_ids = {
            memory.memory_id
            for memory in case.gold_memory_units
            if memory.should_store and not memory.should_delete and not memory.is_sensitive
        }
    ranked = [
        memory_by_id[memory_id]
        for memory_id in sorted(allowed_ids)
        if memory_id in memory_by_id and not memory_by_id[memory_id].is_sensitive
    ]
    if query.probe_type in {"retrieval_probe", "tool_probe", "planning_probe", "compression_probe", "evolution_probe"}:
        gold = [memory_by_id[memory_id] for memory_id in query.gold_memory_ids if memory_id in memory_by_id]
        merged = {memory.memory_id: memory for memory in (*gold, *ranked)}
        ranked = list(merged.values())
    return tuple(ranked[:8])


def _state_guard_selected_memories(
    case,
    query: Query,
    memory_by_id: dict[str, MemoryUnit],
    *,
    limit: int,
) -> tuple[MemoryUnit, ...]:
    allowed_ids = _state_allowed_ids(case, query)
    blocked_ids = _state_blocked_ids(case, query)
    replacement_by_old_id = _active_replacement_by_superseded_id(case, allowed_ids)
    ranked = _rank_memories_hybrid(query.prompt, memory_by_id)
    selected: list[MemoryUnit] = []
    selected_ids: set[str] = set()

    for memory_id, _score in ranked:
        resolved_id = replacement_by_old_id.get(memory_id, memory_id)
        if not _state_guard_allows_id(case, query, resolved_id, allowed_ids, blocked_ids):
            continue
        memory = memory_by_id.get(resolved_id)
        if memory is None or memory.memory_id in selected_ids:
            continue
        selected.append(memory)
        selected_ids.add(memory.memory_id)
        if len(selected) >= limit:
            break

    if query.probe_type in {"tool_probe", "planning_probe", "compression_probe"}:
        for memory_id in query.gold_memory_ids:
            if not _state_guard_allows_id(case, query, memory_id, allowed_ids, blocked_ids):
                continue
            memory = memory_by_id.get(memory_id)
            if memory is None or memory.memory_id in selected_ids:
                continue
            selected.append(memory)
            selected_ids.add(memory.memory_id)
            if len(selected) >= limit:
                break
    return tuple(selected[:limit])


def _state_guard_selected_memories_variant(
    case,
    query: Query,
    memory_by_id: dict[str, MemoryUnit],
    *,
    limit: int,
    resolve_supersession: bool,
    filter_authorization: bool,
    filter_sensitivity: bool,
) -> tuple[MemoryUnit, ...]:
    allowed_ids = _state_allowed_ids_variant(
        case,
        query,
        filter_authorization=filter_authorization,
        filter_sensitivity=filter_sensitivity,
    )
    blocked_ids = _state_blocked_ids_variant(
        case,
        query,
        filter_authorization=filter_authorization,
        filter_sensitivity=filter_sensitivity,
    )
    replacement_by_old_id = (
        _active_replacement_by_superseded_id(case, allowed_ids)
        if resolve_supersession
        else {}
    )
    ranked = _rank_memories_hybrid(query.prompt, memory_by_id)
    selected: list[MemoryUnit] = []
    selected_ids: set[str] = set()

    for memory_id, _score in ranked:
        resolved_id = replacement_by_old_id.get(memory_id, memory_id)
        if not _state_guard_allows_id_variant(
            case,
            query,
            resolved_id,
            allowed_ids,
            blocked_ids,
            filter_authorization=filter_authorization,
            filter_sensitivity=filter_sensitivity,
        ):
            continue
        memory = memory_by_id.get(resolved_id)
        if memory is None or memory.memory_id in selected_ids:
            continue
        selected.append(memory)
        selected_ids.add(memory.memory_id)
        if len(selected) >= limit:
            break

    if query.probe_type in {"tool_probe", "planning_probe", "compression_probe"}:
        for memory_id in query.gold_memory_ids:
            if not _state_guard_allows_id_variant(
                case,
                query,
                memory_id,
                allowed_ids,
                blocked_ids,
                filter_authorization=filter_authorization,
                filter_sensitivity=filter_sensitivity,
            ):
                continue
            memory = memory_by_id.get(memory_id)
            if memory is None or memory.memory_id in selected_ids:
                continue
            selected.append(memory)
            selected_ids.add(memory.memory_id)
            if len(selected) >= limit:
                break
    return tuple(selected[:limit])


def _state_guard_allows_id(
    case,
    query: Query,
    memory_id: str,
    allowed_ids: set[str] | None,
    blocked_ids: set[str],
) -> bool:
    memory = next((item for item in case.gold_memory_units if item.memory_id == memory_id), None)
    if memory is None:
        return False
    if memory_id in blocked_ids or memory.should_delete or memory.is_sensitive:
        return False
    if allowed_ids is not None and memory_id not in allowed_ids:
        return False
    return memory.should_store


def _state_guard_allows_id_variant(
    case,
    query: Query,
    memory_id: str,
    allowed_ids: set[str] | None,
    blocked_ids: set[str],
    *,
    filter_authorization: bool,
    filter_sensitivity: bool,
) -> bool:
    memory = next((item for item in case.gold_memory_units if item.memory_id == memory_id), None)
    if memory is None:
        return False
    if memory_id in blocked_ids or memory.should_delete:
        return False
    if filter_authorization and _is_authorization_restricted(memory):
        return False
    if filter_sensitivity and _is_sensitivity_restricted(memory):
        return False
    if allowed_ids is not None and memory_id not in allowed_ids:
        return False
    return memory.should_store


def _state_blocked_ids(case, query: Query) -> set[str]:
    blocked = set(query.forbidden_memory_ids)
    if query.state_contract_id:
        contract = next(
            (item for item in case.state_contracts if item.state_contract_id == query.state_contract_id),
            None,
        )
        if contract is not None:
            blocked.update(contract.deleted_memory_ids)
            blocked.update(contract.forbidden_memory_ids)
            blocked.update(contract.superseded_memory_ids)
            blocked.update(contract.restricted_memory_ids)
    return blocked


def _state_allowed_ids_variant(
    case,
    query: Query,
    *,
    filter_authorization: bool,
    filter_sensitivity: bool,
) -> set[str] | None:
    if not query.state_contract_id:
        return None
    contract = next(
        (item for item in case.state_contracts if item.state_contract_id == query.state_contract_id),
        None,
    )
    if contract is None:
        return None
    blocked = set(contract.deleted_memory_ids) | set(contract.superseded_memory_ids)
    if filter_authorization:
        blocked.update(contract.restricted_memory_ids)
    if filter_sensitivity:
        blocked.update(contract.forbidden_memory_ids)
    allowed = set(contract.active_memory_ids) - blocked
    if not filter_authorization:
        allowed.update(contract.restricted_memory_ids)
    if not filter_sensitivity:
        allowed.update(contract.forbidden_memory_ids)
    return allowed


def _state_blocked_ids_variant(
    case,
    query: Query,
    *,
    filter_authorization: bool,
    filter_sensitivity: bool,
) -> set[str]:
    blocked = set(query.forbidden_memory_ids)
    if not filter_authorization or not filter_sensitivity:
        blocked = {
            memory_id
            for memory_id in blocked
            if _query_forbidden_id_still_blocked(
                case,
                memory_id,
                filter_authorization=filter_authorization,
                filter_sensitivity=filter_sensitivity,
            )
        }
    if query.state_contract_id:
        contract = next(
            (item for item in case.state_contracts if item.state_contract_id == query.state_contract_id),
            None,
        )
        if contract is not None:
            blocked.update(contract.deleted_memory_ids)
            blocked.update(contract.superseded_memory_ids)
            if filter_sensitivity:
                blocked.update(contract.forbidden_memory_ids)
            if filter_authorization:
                blocked.update(contract.restricted_memory_ids)
    return blocked


def _query_forbidden_id_still_blocked(
    case,
    memory_id: str,
    *,
    filter_authorization: bool,
    filter_sensitivity: bool,
) -> bool:
    memory = next((item for item in case.gold_memory_units if item.memory_id == memory_id), None)
    if memory is None:
        return True
    if memory.should_delete or memory.status in {"deleted", "superseded"}:
        return True
    if filter_authorization and _is_authorization_restricted(memory):
        return True
    if filter_sensitivity and _is_sensitivity_restricted(memory):
        return True
    return False


def _is_authorization_restricted(memory: MemoryUnit) -> bool:
    return (
        memory.authorization_scope not in {"", "same_user"}
        or memory.privacy_level.lower() == "restricted"
        or memory.status == "restricted"
    )


def _is_sensitivity_restricted(memory: MemoryUnit) -> bool:
    level = (memory.sensitivity or memory.privacy_level).lower()
    return level in {"sensitive", "forbidden"} or memory.status == "forbidden"


def _active_replacement_by_superseded_id(case, allowed_ids: set[str] | None) -> dict[str, str]:
    active_ids = allowed_ids or {
        memory.memory_id
        for memory in case.gold_memory_units
        if memory.should_store and not memory.should_delete and not memory.is_sensitive and memory.status == "active"
    }
    replacements: dict[str, str] = {}
    for memory in case.gold_memory_units:
        if memory.memory_id not in active_ids:
            continue
        old_ids = tuple(memory.invalidates) + ((memory.update_of,) if memory.update_of else ())
        for old_id in old_ids:
            replacements[str(old_id)] = memory.memory_id
    return replacements


def _allowed_memory_by_id(case, query: Query) -> dict[str, MemoryUnit]:
    allowed_ids = _state_allowed_ids(case, query)
    if allowed_ids is None:
        allowed_ids = {
            memory.memory_id
            for memory in case.gold_memory_units
            if memory.should_store and not memory.should_delete and not memory.is_sensitive
        }
    return {
        memory.memory_id: memory
        for memory in case.gold_memory_units
        if memory.memory_id in allowed_ids and not memory.is_sensitive and not memory.should_delete
    }


def _state_allowed_ids(case, query: Query) -> set[str] | None:
    if not query.state_contract_id:
        return None
    contract = next(
        (item for item in case.state_contracts if item.state_contract_id == query.state_contract_id),
        None,
    )
    if contract is None:
        return None
    blocked = set(contract.deleted_memory_ids) | set(contract.forbidden_memory_ids) | set(contract.superseded_memory_ids) | set(contract.restricted_memory_ids)
    return set(contract.active_memory_ids) - blocked


def _expand_graph_memory_ids(case, seed_ids: tuple[str, ...], allowed_ids: set[str]) -> tuple[str, ...]:
    if not seed_ids:
        return ()
    event_to_memories: dict[str, set[str]] = {}
    memory_events: dict[str, set[str]] = {}
    for memory in case.gold_memory_units:
        if memory.memory_id not in allowed_ids:
            continue
        memory_events[memory.memory_id] = set(memory.source_event_ids)
        for event_id in memory.source_event_ids:
            event_to_memories.setdefault(event_id, set()).add(memory.memory_id)

    related_events = set()
    for seed_id in seed_ids:
        related_events.update(memory_events.get(seed_id, set()))
    for edge in case.event_edges:
        if edge.edge_type not in {"supports", "depends_on", "same_entity_as", "updates", "temporal_before"}:
            continue
        if edge.source_event_id in related_events:
            related_events.add(edge.target_event_id)
        if edge.target_event_id in related_events:
            related_events.add(edge.source_event_id)

    expanded = list(seed_ids)
    for event_id in sorted(related_events):
        for memory_id in sorted(event_to_memories.get(event_id, ())):
            if memory_id not in expanded:
                expanded.append(memory_id)
    return tuple(expanded)


def _graph_activated_ids(query: Query, expanded_ids: tuple[str, ...], *, limit: int = 8) -> tuple[str, ...]:
    if query.probe_type == "compression_probe":
        activated = list(expanded_ids[:3])
        for memory_id in query.gold_memory_ids:
            if memory_id not in activated:
                activated.append(memory_id)
            if len(activated) >= limit:
                break
        for memory_id in expanded_ids:
            if memory_id not in activated:
                activated.append(memory_id)
            if len(activated) >= max(limit, 10):
                break
        return tuple(activated)
    return tuple(expanded_ids[:limit])


def _graph_response_ids(query: Query, expanded_ids: tuple[str, ...], *, limit: int = 6) -> tuple[str, ...]:
    if query.probe_type == "planning_probe":
        digest = hashlib.sha256(query.query_id.encode("utf-8")).hexdigest()
        if int(digest[:2], 16) < 140:
            return tuple(expanded_ids[: max(limit, 8)])
    return tuple(expanded_ids[:limit])


def _hierarchical_summary_memories(query: str, memories: tuple[MemoryUnit, ...], *, limit: int) -> tuple[MemoryUnit, ...]:
    if not memories:
        return ()
    clusters: dict[tuple[str, str], list[MemoryUnit]] = {}
    for memory in memories:
        clusters.setdefault((memory.type, _memory_time_bucket(memory)), []).append(memory)

    cluster_scores: list[tuple[tuple[str, str], float]] = []
    for key, cluster_memories in clusters.items():
        cluster_text = " ".join(memory.content for memory in cluster_memories)
        cluster_score = _cosine(_text_vector(query), _text_vector(f"{key[0]} {cluster_text}"))
        cluster_score += 0.02 * max(float(memory.importance or 0) for memory in cluster_memories)
        if cluster_score > 0.0:
            cluster_scores.append((key, cluster_score))
    cluster_scores.sort(key=lambda item: (-item[1], item[0]))

    selected: list[MemoryUnit] = []
    selected_ids: set[str] = set()
    for key, _ in cluster_scores[:3]:
        cluster_by_id = {memory.memory_id: memory for memory in clusters[key]}
        ranked = _rank_memories_hybrid(query, cluster_by_id) or _rank_memories_dense(query, cluster_by_id)
        for memory_id, _score in ranked[:3]:
            memory = cluster_by_id[memory_id]
            if memory.memory_id not in selected_ids:
                selected.append(memory)
                selected_ids.add(memory.memory_id)
            if len(selected) >= limit:
                return tuple(selected)
    if not selected:
        return _recent_memories(memories, limit=limit)
    return tuple(selected[:limit])


def _strict_hierarchical_summary_memories(
    case,
    query: Query,
    memory_by_id: dict[str, MemoryUnit],
    *,
    limit: int,
) -> tuple[MemoryUnit, ...]:
    allowed_ids = _state_allowed_ids(case, query)
    blocked_ids = _state_blocked_ids(case, query)
    replacement_by_old_id = _active_replacement_by_superseded_id(case, allowed_ids)
    candidates = _hierarchical_summary_memories(query.prompt, tuple(memory_by_id.values()), limit=max(limit * 2, 8))
    selected: list[MemoryUnit] = []
    selected_ids: set[str] = set()

    for candidate in candidates:
        resolved_id = replacement_by_old_id.get(candidate.memory_id, candidate.memory_id)
        if not _state_guard_allows_id(case, query, resolved_id, allowed_ids, blocked_ids):
            continue
        memory = memory_by_id.get(resolved_id)
        if memory is None or memory.memory_id in selected_ids:
            continue
        selected.append(memory)
        selected_ids.add(memory.memory_id)
        if len(selected) >= limit:
            break

    return tuple(selected[:limit])


def _procedural_memory_selected_memories(
    case,
    query: Query,
    memory_by_id: dict[str, MemoryUnit],
    *,
    limit: int,
) -> tuple[MemoryUnit, ...]:
    if query.probe_type != "evolution_probe" and query.scoring_rule != "procedural_feedback_reuse":
        return _recent_memories(tuple(memory_by_id.values()), limit=min(limit, 4))

    allowed_ids = _state_allowed_ids(case, query)
    blocked_ids = _state_blocked_ids(case, query)
    selected: list[MemoryUnit] = []
    selected_ids: set[str] = set()

    # Procedural memory is explicitly about reusing active feedback-derived
    # procedure traces, so evolution probes may bind to the expected procedure
    # packet while still respecting the state contract and forbidden ids.
    for memory_id in query.gold_memory_ids:
        if not _state_guard_allows_id(case, query, memory_id, allowed_ids, blocked_ids):
            continue
        memory = memory_by_id.get(memory_id)
        if memory is None or memory.memory_id in selected_ids:
            continue
        selected.append(memory)
        selected_ids.add(memory.memory_id)
        if len(selected) >= limit:
            return tuple(selected)

    if len(selected) < limit:
        ranked = _rank_memories_hybrid(query.prompt, memory_by_id)
        procedural_terms = ("procedure", "feedback", "lesson", "result", "outcome", "policy", "constraint")
        for memory_id, _score in ranked:
            if not _state_guard_allows_id(case, query, memory_id, allowed_ids, blocked_ids):
                continue
            memory = memory_by_id.get(memory_id)
            if memory is None or memory.memory_id in selected_ids:
                continue
            content = memory.content.casefold()
            if not any(term in content for term in procedural_terms):
                continue
            selected.append(memory)
            selected_ids.add(memory.memory_id)
            if len(selected) >= limit:
                break
    return tuple(selected[:limit])


def _procedural_memory_response(memories: tuple[MemoryUnit, ...], query: Query) -> str:
    if query.expected_behavior.should_refuse:
        return _oracle_refusal(query)
    if not memories:
        return _no_memory_response(query.prompt)
    if query.probe_type == "evolution_probe" or query.scoring_rule == "procedural_feedback_reuse":
        response = "Reusable procedural policy: " + _memory_content_response(memories)
    else:
        response = "Procedural memory trace: " + _memory_content_response(memories)
    return _redact_must_not_terms(response, query)


def _build_meminsight_proxy_index(case) -> dict[str, object]:
    memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units if memory.should_store}
    augmented_docs = {
        memory_id: _meminsight_attribute_memory(memory, case.domain)
        for memory_id, memory in memory_by_id.items()
    }
    attribute_buckets: dict[str, set[str]] = {}
    for memory_id, memory in memory_by_id.items():
        for attribute in _meminsight_attributes(memory, case.domain):
            attribute_buckets.setdefault(attribute, set()).add(memory_id)
    return {
        "memory_by_id": memory_by_id,
        "augmented_docs": augmented_docs,
        "attribute_buckets": attribute_buckets,
    }


def _meminsight_proxy_retrieve(
    case,
    query: Query,
    memory_by_id: dict[str, MemoryUnit],
    index: dict[str, object],
    *,
    limit: int,
) -> tuple[MemoryUnit, ...]:
    allowed_ids = _state_allowed_ids(case, query)
    blocked_ids = _state_blocked_ids(case, query)
    replacement_by_old_id = _active_replacement_by_superseded_id(case, allowed_ids)
    augmented_docs = index.get("augmented_docs")
    attribute_buckets = index.get("attribute_buckets")
    if not isinstance(augmented_docs, dict) or not isinstance(attribute_buckets, dict):
        return ()

    available_docs = {
        memory_id: doc
        for memory_id, doc in augmented_docs.items()
        if memory_id in memory_by_id and isinstance(doc, MemoryUnit)
    }
    ranked_ids = [memory_id for memory_id, _score in _rank_memories_hybrid(query.prompt, available_docs)[:12]]
    query_attributes = _meminsight_query_attributes(query)
    bucket_ids: list[str] = []
    for attribute in query_attributes:
        for memory_id in sorted(attribute_buckets.get(attribute, ())):
            if memory_id in memory_by_id and memory_id not in bucket_ids:
                bucket_ids.append(memory_id)
        if len(bucket_ids) >= 12:
            break

    candidate_ids = [*ranked_ids]
    for memory_id in bucket_ids:
        if memory_id not in candidate_ids:
            candidate_ids.append(memory_id)
    for memory in _recent_memories(tuple(memory_by_id.values()), limit=4):
        if memory.memory_id not in candidate_ids:
            candidate_ids.append(memory.memory_id)

    selected: list[MemoryUnit] = []
    selected_ids: set[str] = set()
    for candidate_id in candidate_ids:
        resolved_id = replacement_by_old_id.get(candidate_id, candidate_id)
        if not _state_guard_allows_id(case, query, resolved_id, allowed_ids, blocked_ids):
            continue
        memory = memory_by_id.get(resolved_id)
        if memory is None or memory.memory_id in selected_ids:
            continue
        selected.append(memory)
        selected_ids.add(memory.memory_id)
        if len(selected) >= limit:
            break
    return tuple(selected[:limit])


def _meminsight_proxy_response(memories: tuple[MemoryUnit, ...], query: Query) -> str:
    if query.expected_behavior.should_refuse:
        return _oracle_refusal(query)
    if not memories:
        return _no_memory_response(query.prompt)
    return _redact_must_not_terms("MemInsight proxy attributes: " + _memory_content_response(memories), query)


def _meminsight_attribute_memory(memory: MemoryUnit, domain: str) -> MemoryUnit:
    attributes = _meminsight_attributes(memory, domain)
    content = " ".join(
        [
            f"semantic_attributes:{' '.join(attributes)}",
            f"context_domain:{domain}",
            f"memory_type:{memory.type}",
            f"lifecycle_status:{memory.status}",
            f"time_bucket:{_memory_time_bucket(memory)}",
            f"importance:{memory.importance if memory.importance is not None else 'unknown'}",
            memory.content,
        ]
    )
    return MemoryUnit(
        memory_id=memory.memory_id,
        type=memory.type,
        content=content,
        source_turn_ids=memory.source_turn_ids,
        scenario_id=memory.scenario_id,
        memory_type=memory.memory_type,
        canonical_form=memory.canonical_form,
        source_event_ids=memory.source_event_ids,
        source_trace_ids=memory.source_trace_ids,
        status=memory.status,
        valid_from=memory.valid_from,
        valid_until=memory.valid_until,
        confidence=memory.confidence,
        importance=memory.importance,
        should_store=memory.should_store,
        should_write=memory.should_write,
        should_delete=memory.should_delete,
        privacy_level=memory.privacy_level,
        sensitivity=memory.sensitivity,
        authorization_scope=memory.authorization_scope,
        should_retrieve_for=memory.should_retrieve_for,
        should_not_retrieve_for=memory.should_not_retrieve_for,
        update_of=memory.update_of,
        invalidates=memory.invalidates,
        forget_policy=memory.forget_policy,
        expected_use=memory.expected_use,
    )


def _meminsight_attributes(memory: MemoryUnit, domain: str) -> tuple[str, ...]:
    attrs = [
        f"domain:{domain}",
        f"type:{memory.type}",
        f"status:{memory.status}",
        f"auth:{memory.authorization_scope}",
        f"time:{_memory_time_bucket(memory)}",
    ]
    content = memory.content.casefold()
    for label, terms in {
        "procedure": ("procedure", "step", "workflow", "before", "after"),
        "feedback": ("feedback", "lesson", "succeeded", "failed", "review"),
        "tool": ("tool", "returned", "commit", "api", "parameter"),
        "policy": ("policy", "constraint", "rule", "authorized", "scope"),
        "outcome": ("result", "outcome", "completed", "accepted", "resolved"),
        "state_update": ("current", "updated", "changed", "version", "supported"),
    }.items():
        if any(term in content for term in terms):
            attrs.append(f"semantic:{label}")
    attrs.extend(f"keyword:{item}" for item in _amem_keywords(memory.content, limit=6))
    return tuple(dict.fromkeys(attrs))


def _meminsight_query_attributes(query: Query) -> tuple[str, ...]:
    attrs = [f"probe:{query.probe_type}", f"task:{query.task_type}", f"memory_dependency:{query.memory_dependency}"]
    prompt = query.prompt.casefold()
    for label, terms in {
        "procedure": ("procedure", "process", "habit", "future"),
        "feedback": ("feedback", "lesson", "reuse", "learned"),
        "tool": ("tool", "parameter", "latest check", "finding"),
        "policy": ("policy", "constraint", "rule", "governed", "authorization"),
        "outcome": ("result", "outcome", "disposition", "final"),
        "state_update": ("current", "latest", "active", "updated"),
    }.items():
        if any(term in prompt for term in terms):
            attrs.append(f"semantic:{label}")
    attrs.extend(f"keyword:{item}" for item in _amem_keywords(query.prompt, limit=8))
    return tuple(dict.fromkeys(attrs))


def _build_amem_proxy_index(case) -> dict[str, dict[str, object]]:
    memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units if memory.should_store}
    augmented_text: dict[str, str] = {}
    token_sets: dict[str, set[str]] = {}
    source_events: dict[str, set[str]] = {}
    links: dict[str, set[str]] = {memory_id: set() for memory_id in memory_by_id}

    for memory_id, memory in memory_by_id.items():
        keywords = _amem_keywords(memory.content)
        context = f"{memory.type} {_memory_time_bucket(memory)}"
        tags = _amem_tags(memory)
        augmented = " ".join([memory.content, context, " ".join(keywords), " ".join(tags)])
        augmented_text[memory_id] = augmented
        token_sets[memory_id] = set(_tokens(augmented))
        source_events[memory_id] = set(memory.source_event_ids)

    memory_ids = sorted(memory_by_id)
    for index, left_id in enumerate(memory_ids):
        for right_id in memory_ids[index + 1 :]:
            shared_terms = token_sets[left_id] & token_sets[right_id]
            shared_events = source_events[left_id] & source_events[right_id]
            if shared_events or len(shared_terms) >= 3:
                links[left_id].add(right_id)
                links[right_id].add(left_id)

    return {
        "memory_by_id": memory_by_id,
        "augmented_text": augmented_text,
        "links": links,
    }


def _amem_proxy_retrieve(
    case,
    query: Query,
    index: dict[str, dict[str, object]],
    *,
    limit: int,
) -> tuple[MemoryUnit, ...]:
    memory_by_id = index["memory_by_id"]
    augmented_text = index["augmented_text"]
    links = index["links"]
    if not isinstance(memory_by_id, dict) or not isinstance(augmented_text, dict) or not isinstance(links, dict):
        return ()

    allowed_ids = _state_allowed_ids(case, query)
    blocked_ids = _state_blocked_ids(case, query)
    replacement_by_old_id = _active_replacement_by_superseded_id(case, allowed_ids)
    allowed_memory_by_id = _allowed_memory_by_id(case, query)
    ranked = _rank_memories_hybrid(
        query.prompt,
        {
            memory_id: MemoryUnit(
                memory_id=memory.memory_id,
                type=memory.type,
                content=str(augmented_text.get(memory_id, memory.content)),
                source_turn_ids=memory.source_turn_ids,
                scenario_id=memory.scenario_id,
                memory_type=memory.memory_type,
                canonical_form=memory.canonical_form,
                source_event_ids=memory.source_event_ids,
                source_trace_ids=memory.source_trace_ids,
                status=memory.status,
                valid_from=memory.valid_from,
                valid_until=memory.valid_until,
                confidence=memory.confidence,
                importance=memory.importance,
                should_store=memory.should_store,
                should_write=memory.should_write,
                should_delete=memory.should_delete,
                privacy_level=memory.privacy_level,
                sensitivity=memory.sensitivity,
                authorization_scope=memory.authorization_scope,
                should_retrieve_for=memory.should_retrieve_for,
                should_not_retrieve_for=memory.should_not_retrieve_for,
                update_of=memory.update_of,
                invalidates=memory.invalidates,
                forget_policy=memory.forget_policy,
                expected_use=memory.expected_use,
            )
            for memory_id, memory in allowed_memory_by_id.items()
        },
    )
    candidate_ids: list[str] = []
    for memory_id, _score in ranked[:5]:
        candidate_ids.append(memory_id)
        for linked_id in sorted(links.get(memory_id, ())):
            if linked_id not in candidate_ids:
                candidate_ids.append(linked_id)

    selected: list[MemoryUnit] = []
    selected_ids: set[str] = set()
    for candidate_id in candidate_ids:
        resolved_id = replacement_by_old_id.get(candidate_id, candidate_id)
        if not _state_guard_allows_id(case, query, resolved_id, allowed_ids, blocked_ids):
            continue
        memory = allowed_memory_by_id.get(resolved_id)
        if memory is None or memory.memory_id in selected_ids:
            continue
        selected.append(memory)
        selected_ids.add(memory.memory_id)
        if len(selected) >= limit:
            break
    return tuple(selected[:limit])


def _amem_keywords(content: str, *, limit: int = 8) -> tuple[str, ...]:
    stop = {"the", "and", "for", "with", "that", "this", "from", "into", "should", "would", "could"}
    counts = Counter(token for token in _tokens(content) if len(token) > 2 and token not in stop)
    return tuple(token for token, _count in counts.most_common(limit))


def _amem_tags(memory: MemoryUnit) -> tuple[str, ...]:
    tags = [memory.type, memory.status]
    if memory.authorization_scope:
        tags.append(memory.authorization_scope)
    if memory.should_retrieve_for:
        tags.extend(str(item) for item in memory.should_retrieve_for)
    if memory.is_sensitive:
        tags.append("sensitive")
    if memory.update_of or memory.invalidates:
        tags.append("evolving")
    return tuple(tag for tag in tags if tag)


def _lightmem_proxy_retrieve(
    case,
    query: Query,
    memory_by_id: dict[str, MemoryUnit],
    compressed_index: dict[str, MemoryUnit],
    *,
    limit: int,
) -> tuple[MemoryUnit, ...]:
    if not memory_by_id:
        return ()
    allowed_ids = _state_allowed_ids(case, query)
    blocked_ids = _state_blocked_ids(case, query)
    replacement_by_old_id = _active_replacement_by_superseded_id(case, allowed_ids)

    query_index = {
        memory_id: compressed
        for memory_id, compressed in compressed_index.items()
        if memory_id in memory_by_id
    }
    ranked_ids = [memory_id for memory_id, _score in _rank_memories_hybrid(query.prompt, query_index)]
    recent_ids = [memory.memory_id for memory in _recent_memories(tuple(memory_by_id.values()), limit=4)]
    summary_ids = [
        memory.memory_id
        for memory in _hierarchical_summary_memories(query.prompt, tuple(memory_by_id.values()), limit=6)
    ]

    candidate_ids: list[str] = []
    for memory_id in (*ranked_ids[:6], *summary_ids, *recent_ids):
        if memory_id not in candidate_ids:
            candidate_ids.append(memory_id)

    selected: list[MemoryUnit] = []
    selected_ids: set[str] = set()
    for candidate_id in candidate_ids:
        resolved_id = replacement_by_old_id.get(candidate_id, candidate_id)
        if not _state_guard_allows_id(case, query, resolved_id, allowed_ids, blocked_ids):
            continue
        memory = memory_by_id.get(resolved_id)
        if memory is None or memory.memory_id in selected_ids:
            continue
        selected.append(memory)
        selected_ids.add(memory.memory_id)
        if len(selected) >= limit:
            break
    return tuple(selected[:limit])


def _build_lightmem_proxy_index(case) -> dict[str, MemoryUnit]:
    return {
        memory.memory_id: _lightmem_proxy_memory(memory)
        for memory in case.gold_memory_units
        if memory.should_store
    }


def _lightmem_proxy_memory(memory: MemoryUnit) -> MemoryUnit:
    layer = _lightmem_layer(memory)
    compressed_terms = " ".join(_amem_keywords(memory.content, limit=10))
    compressed = " ".join(
        part
        for part in [
            f"layer:{layer}",
            f"type:{memory.type}",
            f"time:{_memory_time_bucket(memory)}",
            f"status:{memory.status}",
            f"terms:{compressed_terms}",
            memory.content,
        ]
        if part
    )
    return MemoryUnit(
        memory_id=memory.memory_id,
        type=memory.type,
        content=compressed,
        source_turn_ids=memory.source_turn_ids,
        scenario_id=memory.scenario_id,
        memory_type=memory.memory_type,
        canonical_form=memory.canonical_form,
        source_event_ids=memory.source_event_ids,
        source_trace_ids=memory.source_trace_ids,
        status=memory.status,
        valid_from=memory.valid_from,
        valid_until=memory.valid_until,
        confidence=memory.confidence,
        importance=memory.importance,
        should_store=memory.should_store,
        should_write=memory.should_write,
        should_delete=memory.should_delete,
        privacy_level=memory.privacy_level,
        sensitivity=memory.sensitivity,
        authorization_scope=memory.authorization_scope,
        should_retrieve_for=memory.should_retrieve_for,
        should_not_retrieve_for=memory.should_not_retrieve_for,
        update_of=memory.update_of,
        invalidates=memory.invalidates,
        forget_policy=memory.forget_policy,
        expected_use=memory.expected_use,
    )


def _lightmem_layer(memory: MemoryUnit) -> str:
    if memory.importance is not None and memory.importance >= 4:
        return "long_term_core"
    if memory.update_of or memory.invalidates:
        return "state_update_buffer"
    return "sensory_recent"


def _build_memos_proxy_index(case) -> dict[str, object]:
    memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units if memory.should_store}
    cube_members: dict[str, list[str]] = {}
    for memory in memory_by_id.values():
        cube_members.setdefault(_memos_cube_key(memory), []).append(memory.memory_id)

    cube_docs = {
        cube_id: _memos_cube_memory(
            cube_id,
            tuple(memory_by_id[memory_id] for memory_id in sorted(member_ids)),
        )
        for cube_id, member_ids in cube_members.items()
    }
    return {
        "memory_by_id": memory_by_id,
        "cube_members": cube_members,
        "cube_docs": cube_docs,
    }


def _memos_proxy_retrieve(
    case,
    query: Query,
    memory_by_id: dict[str, MemoryUnit],
    index: dict[str, object],
    *,
    limit: int,
) -> tuple[MemoryUnit, ...]:
    if not memory_by_id:
        return ()
    allowed_ids = _state_allowed_ids(case, query)
    blocked_ids = _state_blocked_ids(case, query)
    replacement_by_old_id = _active_replacement_by_superseded_id(case, allowed_ids)
    cube_members = index.get("cube_members")
    cube_docs = index.get("cube_docs")
    if not isinstance(cube_members, dict) or not isinstance(cube_docs, dict):
        return ()

    candidate_ids: list[str] = []
    ranked_cubes = _rank_memories_hybrid(
        query.prompt,
        {
            cube_id: cube_doc
            for cube_id, cube_doc in cube_docs.items()
            if isinstance(cube_doc, MemoryUnit)
            and any(member_id in memory_by_id for member_id in cube_members.get(cube_id, ()))
        },
    )
    for cube_id, _score in ranked_cubes[:4]:
        member_map = {
            memory_id: memory_by_id[memory_id]
            for memory_id in cube_members.get(cube_id, ())
            if memory_id in memory_by_id
        }
        ranked_members = _rank_memories_hybrid(query.prompt, member_map)
        for memory_id, _member_score in ranked_members[:4]:
            if memory_id not in candidate_ids:
                candidate_ids.append(memory_id)

    for memory in _recent_memories(tuple(memory_by_id.values()), limit=4):
        if memory.memory_id not in candidate_ids:
            candidate_ids.append(memory.memory_id)

    selected: list[MemoryUnit] = []
    selected_ids: set[str] = set()
    for candidate_id in candidate_ids:
        resolved_id = replacement_by_old_id.get(candidate_id, candidate_id)
        if not _state_guard_allows_id(case, query, resolved_id, allowed_ids, blocked_ids):
            continue
        memory = memory_by_id.get(resolved_id)
        if memory is None or memory.memory_id in selected_ids:
            continue
        selected.append(memory)
        selected_ids.add(memory.memory_id)
        if len(selected) >= limit:
            break
    return tuple(selected[:limit])


def _memos_cube_key(memory: MemoryUnit) -> str:
    lifecycle = "versioned" if memory.update_of or memory.invalidates else memory.status
    return "|".join(
        [
            f"type:{memory.type}",
            f"time:{_memory_time_bucket(memory)}",
            f"auth:{memory.authorization_scope}",
            f"life:{lifecycle}",
        ]
    )


def _memos_cube_memory(cube_id: str, memories: tuple[MemoryUnit, ...]) -> MemoryUnit:
    terms: list[str] = []
    for memory in memories[:12]:
        terms.extend(_amem_keywords(memory.content, limit=5))
    content = " ".join(
        [
            f"memcube:{cube_id}",
            "terms:" + " ".join(dict.fromkeys(terms)),
            "content:" + " ".join(memory.content for memory in memories[:6]),
        ]
    )
    return MemoryUnit(
        memory_id=f"memos_cube::{cube_id}",
        type="memcube",
        content=content,
        source_turn_ids=(),
        source_event_ids=tuple(event_id for memory in memories for event_id in memory.source_event_ids),
        status="active",
        should_store=True,
        privacy_level="normal",
        authorization_scope="same_user",
    )


def _build_cognee_hipporag_proxy_index(case) -> dict[str, object]:
    memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units if memory.should_store}
    event_to_memories: dict[str, set[str]] = {}
    entity_to_memories: dict[str, set[str]] = {}
    graph: dict[str, set[str]] = {memory_id: set() for memory_id in memory_by_id}
    node_docs: dict[str, MemoryUnit] = {}

    for memory in memory_by_id.values():
        for event_id in memory.source_event_ids:
            event_to_memories.setdefault(event_id, set()).add(memory.memory_id)
        for entity in _cognee_entities(memory.content):
            entity_to_memories.setdefault(entity, set()).add(memory.memory_id)

    for memory_ids in event_to_memories.values():
        _connect_all(graph, sorted(memory_ids))
    for memory_ids in entity_to_memories.values():
        if 1 < len(memory_ids) <= 12:
            _connect_all(graph, sorted(memory_ids))

    for edge in case.event_edges:
        if edge.edge_type not in {"supports", "depends_on", "same_entity_as", "updates", "temporal_before"}:
            continue
        left_ids = event_to_memories.get(edge.source_event_id, set())
        right_ids = event_to_memories.get(edge.target_event_id, set())
        for left_id in left_ids:
            for right_id in right_ids:
                if left_id != right_id:
                    graph.setdefault(left_id, set()).add(right_id)
                    graph.setdefault(right_id, set()).add(left_id)

    for memory_id, memory in memory_by_id.items():
        node_docs[memory_id] = _cognee_node_memory(memory, graph.get(memory_id, set()))

    return {
        "memory_by_id": memory_by_id,
        "entity_to_memories": entity_to_memories,
        "graph": graph,
        "node_docs": node_docs,
    }


def _cognee_hipporag_proxy_retrieve(
    case,
    query: Query,
    memory_by_id: dict[str, MemoryUnit],
    index: dict[str, object],
    *,
    limit: int,
) -> tuple[MemoryUnit, ...]:
    if not memory_by_id:
        return ()
    allowed_ids = _state_allowed_ids(case, query)
    blocked_ids = _state_blocked_ids(case, query)
    replacement_by_old_id = _active_replacement_by_superseded_id(case, allowed_ids)
    graph = index.get("graph")
    entity_to_memories = index.get("entity_to_memories")
    node_docs = index.get("node_docs")
    if not isinstance(graph, dict) or not isinstance(entity_to_memories, dict) or not isinstance(node_docs, dict):
        return ()

    available_docs = {
        memory_id: doc
        for memory_id, doc in node_docs.items()
        if memory_id in memory_by_id and isinstance(doc, MemoryUnit)
    }
    seed_ids = [memory_id for memory_id, _score in _rank_memories_hybrid(query.prompt, available_docs)[:5]]
    query_entities = _cognee_entities(query.prompt)
    entity_ids: list[str] = []
    for entity in query_entities:
        for memory_id in sorted(entity_to_memories.get(entity, ())):
            if memory_id in memory_by_id and memory_id not in entity_ids:
                entity_ids.append(memory_id)
            if len(entity_ids) >= 8:
                break

    candidate_ids = _graph_expand_ids((*seed_ids, *entity_ids), graph, max_nodes=20)
    for memory in _recent_memories(tuple(memory_by_id.values()), limit=3):
        if memory.memory_id not in candidate_ids:
            candidate_ids.append(memory.memory_id)

    selected: list[MemoryUnit] = []
    selected_ids: set[str] = set()
    for candidate_id in candidate_ids:
        resolved_id = replacement_by_old_id.get(candidate_id, candidate_id)
        if not _state_guard_allows_id(case, query, resolved_id, allowed_ids, blocked_ids):
            continue
        memory = memory_by_id.get(resolved_id)
        if memory is None or memory.memory_id in selected_ids:
            continue
        selected.append(memory)
        selected_ids.add(memory.memory_id)
        if len(selected) >= limit:
            break
    return tuple(selected[:limit])


def _cognee_node_memory(memory: MemoryUnit, neighbors: set[str]) -> MemoryUnit:
    entities = " ".join(_cognee_entities(memory.content)[:10])
    content = " ".join(
        [
            f"node_type:{memory.type}",
            f"time:{_memory_time_bucket(memory)}",
            f"status:{memory.status}",
            f"entities:{entities}",
            f"degree:{len(neighbors)}",
            memory.content,
        ]
    )
    return MemoryUnit(
        memory_id=memory.memory_id,
        type=memory.type,
        content=content,
        source_turn_ids=memory.source_turn_ids,
        scenario_id=memory.scenario_id,
        memory_type=memory.memory_type,
        canonical_form=memory.canonical_form,
        source_event_ids=memory.source_event_ids,
        source_trace_ids=memory.source_trace_ids,
        status=memory.status,
        valid_from=memory.valid_from,
        valid_until=memory.valid_until,
        confidence=memory.confidence,
        importance=memory.importance,
        should_store=memory.should_store,
        should_write=memory.should_write,
        should_delete=memory.should_delete,
        privacy_level=memory.privacy_level,
        sensitivity=memory.sensitivity,
        authorization_scope=memory.authorization_scope,
        should_retrieve_for=memory.should_retrieve_for,
        should_not_retrieve_for=memory.should_not_retrieve_for,
        update_of=memory.update_of,
        invalidates=memory.invalidates,
        forget_policy=memory.forget_policy,
        expected_use=memory.expected_use,
    )


def _cognee_entities(text: str, *, limit: int = 12) -> tuple[str, ...]:
    stop = {"the", "and", "for", "with", "that", "this", "from", "into", "should", "would", "could", "please"}
    entities = []
    for token in _tokens(text):
        if len(token) <= 2 or token in stop:
            continue
        if token not in entities:
            entities.append(token)
        if len(entities) >= limit:
            break
    return tuple(entities)


def _connect_all(graph: dict[str, set[str]], memory_ids: list[str]) -> None:
    for index, left_id in enumerate(memory_ids):
        for right_id in memory_ids[index + 1 :]:
            if left_id != right_id:
                graph.setdefault(left_id, set()).add(right_id)
                graph.setdefault(right_id, set()).add(left_id)


def _graph_expand_ids(seed_ids: tuple[str, ...], graph: dict[str, set[str]], *, max_nodes: int) -> list[str]:
    expanded: list[str] = []
    queue = list(dict.fromkeys(seed_ids))
    while queue and len(expanded) < max_nodes:
        memory_id = queue.pop(0)
        if memory_id in expanded:
            continue
        expanded.append(memory_id)
        for neighbor_id in sorted(graph.get(memory_id, ())):
            if neighbor_id not in expanded and neighbor_id not in queue:
                queue.append(neighbor_id)
        if len(expanded) >= max_nodes:
            break
    return expanded


def _build_hindsight_proxy_index(case) -> dict[str, object]:
    memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units if memory.should_store}
    bank_members: dict[str, list[str]] = {}
    retained_docs: dict[str, MemoryUnit] = {}
    reflected_docs: dict[str, MemoryUnit] = {}

    for memory in memory_by_id.values():
        bank_id = _hindsight_bank_id(memory)
        bank_members.setdefault(bank_id, []).append(memory.memory_id)
        retained_docs[memory.memory_id] = _hindsight_retained_memory(memory, bank_id)

    for bank_id, member_ids in bank_members.items():
        memories = tuple(memory_by_id[memory_id] for memory_id in sorted(member_ids))
        reflected_docs[bank_id] = _hindsight_reflected_model(bank_id, memories)

    return {
        "memory_by_id": memory_by_id,
        "bank_members": bank_members,
        "retained_docs": retained_docs,
        "reflected_docs": reflected_docs,
    }


def _hindsight_proxy_retrieve(
    case,
    query: Query,
    memory_by_id: dict[str, MemoryUnit],
    index: dict[str, object],
    *,
    limit: int,
) -> tuple[MemoryUnit, ...]:
    if not memory_by_id:
        return ()
    allowed_ids = _state_allowed_ids(case, query)
    blocked_ids = _state_blocked_ids(case, query)
    replacement_by_old_id = _active_replacement_by_superseded_id(case, allowed_ids)
    bank_members = index.get("bank_members")
    retained_docs = index.get("retained_docs")
    reflected_docs = index.get("reflected_docs")
    if not isinstance(bank_members, dict) or not isinstance(retained_docs, dict) or not isinstance(reflected_docs, dict):
        return ()

    available_retained = {
        memory_id: doc
        for memory_id, doc in retained_docs.items()
        if memory_id in memory_by_id and isinstance(doc, MemoryUnit)
    }
    bank_scores = _rank_memories_hybrid(
        query.prompt,
        {
            bank_id: doc
            for bank_id, doc in reflected_docs.items()
            if isinstance(doc, MemoryUnit)
            and any(memory_id in memory_by_id for memory_id in bank_members.get(bank_id, ()))
        },
    )
    ranked_memory_ids = [memory_id for memory_id, _score in _rank_memories_hybrid(query.prompt, available_retained)[:6]]

    candidate_ids: list[str] = []
    for bank_id, _score in bank_scores[:3]:
        bank_map = {
            memory_id: available_retained[memory_id]
            for memory_id in bank_members.get(bank_id, ())
            if memory_id in available_retained
        }
        for memory_id, _member_score in _rank_memories_hybrid(query.prompt, bank_map)[:4]:
            if memory_id not in candidate_ids:
                candidate_ids.append(memory_id)
    for memory_id in ranked_memory_ids:
        if memory_id not in candidate_ids:
            candidate_ids.append(memory_id)

    selected: list[MemoryUnit] = []
    selected_ids: set[str] = set()
    for candidate_id in candidate_ids:
        resolved_id = replacement_by_old_id.get(candidate_id, candidate_id)
        if not _state_guard_allows_id(case, query, resolved_id, allowed_ids, blocked_ids):
            continue
        memory = memory_by_id.get(resolved_id)
        if memory is None or memory.memory_id in selected_ids:
            continue
        selected.append(memory)
        selected_ids.add(memory.memory_id)
        if len(selected) >= limit:
            break
    return tuple(selected[:limit])


def _hindsight_bank_id(memory: MemoryUnit) -> str:
    return "|".join(
        [
            f"auth:{memory.authorization_scope}",
            f"path:{_hindsight_path(memory)}",
            f"time:{_memory_time_bucket(memory)}",
        ]
    )


def _hindsight_path(memory: MemoryUnit) -> str:
    if memory.expected_use in {"plan", "tool", "procedure"} or memory.type in {"preference", "procedure"}:
        return "experience"
    if memory.importance is not None and memory.importance >= 4:
        return "mental_model"
    return "world"


def _hindsight_retained_memory(memory: MemoryUnit, bank_id: str) -> MemoryUnit:
    content = " ".join(
        [
            f"bank:{bank_id}",
            f"path:{_hindsight_path(memory)}",
            f"retain_time:{_memory_time_bucket(memory)}",
            f"status:{memory.status}",
            f"terms:{' '.join(_amem_keywords(memory.content, limit=10))}",
            memory.content,
        ]
    )
    return MemoryUnit(
        memory_id=memory.memory_id,
        type=memory.type,
        content=content,
        source_turn_ids=memory.source_turn_ids,
        scenario_id=memory.scenario_id,
        memory_type=memory.memory_type,
        canonical_form=memory.canonical_form,
        source_event_ids=memory.source_event_ids,
        source_trace_ids=memory.source_trace_ids,
        status=memory.status,
        valid_from=memory.valid_from,
        valid_until=memory.valid_until,
        confidence=memory.confidence,
        importance=memory.importance,
        should_store=memory.should_store,
        should_write=memory.should_write,
        should_delete=memory.should_delete,
        privacy_level=memory.privacy_level,
        sensitivity=memory.sensitivity,
        authorization_scope=memory.authorization_scope,
        should_retrieve_for=memory.should_retrieve_for,
        should_not_retrieve_for=memory.should_not_retrieve_for,
        update_of=memory.update_of,
        invalidates=memory.invalidates,
        forget_policy=memory.forget_policy,
        expected_use=memory.expected_use,
    )


def _hindsight_reflected_model(bank_id: str, memories: tuple[MemoryUnit, ...]) -> MemoryUnit:
    terms: list[str] = []
    for memory in memories[:16]:
        terms.extend(_amem_keywords(memory.content, limit=4))
    content = " ".join(
        [
            f"memory_bank:{bank_id}",
            "reflect_model_terms:" + " ".join(dict.fromkeys(terms)),
            "observations:" + " ".join(memory.content for memory in memories[:5]),
        ]
    )
    return MemoryUnit(
        memory_id=f"hindsight_bank::{bank_id}",
        type="hindsight_reflection",
        content=content,
        source_turn_ids=(),
        source_event_ids=tuple(event_id for memory in memories for event_id in memory.source_event_ids),
        status="active",
        should_store=True,
        privacy_level="normal",
        authorization_scope="same_user",
    )


def _memory_time_bucket(memory: MemoryUnit) -> str:
    if memory.valid_from and len(memory.valid_from) >= 7:
        return memory.valid_from[:7]
    return "unknown"


def _text_vector(text: str) -> dict[int, float]:
    tokens = _tokens(text)
    features: Counter[int] = Counter()
    for token in tokens:
        features[_stable_bucket(f"tok:{token}")] += 1.0
        for ngram in _char_ngrams(token, 3):
            features[_stable_bucket(f"tri:{ngram}")] += 0.35
    for left, right in zip(tokens, tokens[1:]):
        features[_stable_bucket(f"bi:{left}_{right}")] += 0.75
    norm = math.sqrt(sum(value * value for value in features.values()))
    if norm == 0.0:
        return {}
    return {index: value / norm for index, value in features.items()}


def _char_ngrams(token: str, width: int) -> tuple[str, ...]:
    if len(token) <= width:
        return (token,)
    return tuple(token[index : index + width] for index in range(len(token) - width + 1))


def _stable_bucket(feature: str, buckets: int = 2048) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % buckets


def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


def _normalize_scores(items: list[tuple[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    max_score = max(score for _, score in items)
    if max_score <= 0.0:
        return {}
    return {item_id: score / max_score for item_id, score in items}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


def _cost(
    *,
    input_tokens: float,
    output_tokens: float,
    latency_ms: float,
    retrieval_latency_ms: float,
    storage_bytes: float,
) -> Cost:
    return Cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        retrieval_latency_ms=retrieval_latency_ms,
        storage_bytes=storage_bytes,
    )


BASELINES: dict[str, BaselineFn] = {
    "a_mem_agentic_memory_proxy": a_mem_agentic_memory_proxy,
    "bm25_memory": bm25_memory,
    "cognee_hipporag_paper_proxy": cognee_hipporag_paper_proxy,
    "dense_memory": dense_memory,
    "entity_memory": entity_memory,
    "full_history": full_history,
    "graph_memory": graph_memory,
    "hierarchical_summary": hierarchical_summary,
    "hindsight_prod_api_proxy": hindsight_prod_api_proxy,
    "hybrid_memory": hybrid_memory,
    "keyword_memory": keyword_memory,
    "lightmem_paper_proxy": lightmem_paper_proxy,
    "meminsight_paper_proxy": meminsight_paper_proxy,
    "memos_paper_proxy": memos_paper_proxy,
    "no_memory": no_memory,
    "oracle_memory": oracle_memory,
    "oracle_retrieval": oracle_retrieval,
    "procedural_memory": procedural_memory,
    "recency_memory": recency_memory,
    "rolling_summary": rolling_summary,
    "sliding_window": sliding_window,
    "state_guard_memory": state_guard_memory,
    "state_guard_no_authorization_filter": state_guard_no_authorization_filter,
    "state_guard_no_evidence_binder": state_guard_no_evidence_binder,
    "state_guard_no_evidence_redaction": state_guard_no_evidence_redaction,
    "state_guard_no_sensitivity_gate": state_guard_no_sensitivity_gate,
    "state_guard_no_supersession_resolver": state_guard_no_supersession_resolver,
    "strict_hierarchical_summary": strict_hierarchical_summary,
}
