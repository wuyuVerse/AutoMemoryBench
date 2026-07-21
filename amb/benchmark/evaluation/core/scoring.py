"""Benchmark scoring orchestration."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any

from amb.benchmark.metrics.classification import precision_recall_f1, safe_div
from amb.benchmark.metrics.compression import compression_scores
from amb.benchmark.metrics.efficiency import mean, percentile
from amb.benchmark.metrics.evolution import evolution_scores
from amb.benchmark.metrics.retrieval import (
    evidence_complete,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from amb.benchmark.metrics.safety import safety_flags
from amb.benchmark.metrics.task_judges import TaskJudgePlugin, load_task_judge_plugin
from amb.benchmark.metrics.tool import tool_call_scores
from amb.benchmark.query_difficulty import resolve_query_difficulty
from amb.benchmark.schemas.models import (
    Benchmark,
    MemoryUnit,
    PredictionSet,
    Query,
    QueryContext,
    QueryPrediction,
)
from amb.benchmark.schemas.state import replay_state_at

AMQ_LIFECYCLE_WEIGHTS = {
    "write_quality": 0.20,
    "retrieval_quality": 0.15,
    "update_quality": 0.15,
    "compression_quality": 0.15,
    "task_utility": 0.20,
    "safety_quality": 0.10,
    "efficiency_quality": 0.05,
}
DEFAULT_RETRIEVAL_K = 8


class Scorer:
    """Scores a prediction set against a benchmark."""

    def __init__(
        self,
        retrieval_k: int = DEFAULT_RETRIEVAL_K,
        *,
        task_judge_plugin: TaskJudgePlugin | None = None,
    ) -> None:
        if retrieval_k <= 0:
            raise ValueError("retrieval_k must be positive")
        self.retrieval_k = retrieval_k
        self.task_judge_plugin = task_judge_plugin or load_task_judge_plugin()

    def score(self, benchmark: Benchmark, predictions: PredictionSet) -> dict[str, Any]:
        prediction_query_ids = [item.query_id for item in predictions.predictions]
        duplicate_predictions = _duplicate_query_ids(prediction_query_ids)
        prediction_by_query = {item.query_id: item for item in predictions.predictions}
        contexts = list(_query_contexts(benchmark))
        query_reports: list[dict[str, Any]] = []
        missing_predictions: list[str] = []

        case_write_cache = _case_write_cache(benchmark, prediction_by_query)

        for context in contexts:
            prediction = prediction_by_query.get(context.query.query_id)
            if prediction is None:
                missing_predictions.append(context.query.query_id)
                continue
            query_reports.append(self.score_query(context, prediction, case_write_cache[context.case.case_id]))

        known_query_ids = {context.query.query_id for context in contexts}
        extra_predictions = sorted(set(prediction_by_query) - known_query_ids)
        task_judge_metadata = self.task_judge_plugin.report_metadata()

        return {
            "schema_version": benchmark.schema_version,
            "benchmark_id": benchmark.benchmark_id,
            "system_id": predictions.system_id,
            "scoring_config": _scoring_config(
                retrieval_k=self.retrieval_k,
                task_judge_metadata=task_judge_metadata,
            ),
            "task_judge": _report_task_judge_metadata(
                query_reports,
                report_metadata=task_judge_metadata,
            ),
            "aggregate": aggregate_reports(query_reports),
            "counterfactual": counterfactual_report(query_reports),
            "by_task_type": aggregate_by(query_reports, "task_type"),
            "by_probe_type": aggregate_by(query_reports, "probe_type"),
            "by_domain": aggregate_by(query_reports, "domain"),
            "by_difficulty": aggregate_by(query_reports, "difficulty_level"),
            "by_memory_requirement": aggregate_by(query_reports, "memory_requirement"),
            "queries": query_reports,
            "missing_predictions": sorted(missing_predictions),
            "extra_predictions": extra_predictions,
            "duplicate_predictions": duplicate_predictions,
        }

    def score_query(
        self,
        context: QueryContext,
        prediction: QueryPrediction,
        case_predicted_writes: set[str],
    ) -> dict[str, Any]:
        query = context.query
        gold_ids = set(query.gold_memory_ids)
        forbidden_ids = set(query.forbidden_memory_ids)
        if query.state_contract_id:
            contract = next(
                (item for item in context.case.state_contracts if item.state_contract_id == query.state_contract_id),
                None,
            )
            if contract is not None:
                forbidden_ids.update(contract.forbidden_memory_ids)
                forbidden_ids.update(contract.deleted_memory_ids)
                forbidden_ids.update(contract.superseded_memory_ids)
                forbidden_ids.update(contract.restricted_memory_ids)
        elif query.timestamp:
            state = replay_state_at(context.case, query.timestamp)
            forbidden_ids.update(state.forbidden_memory_ids)
            forbidden_ids.update(state.deleted_memory_ids)
            forbidden_ids.update(state.superseded_memory_ids)
            forbidden_ids.update(state.restricted_memory_ids)
        activated = list(prediction.activated_memory_ids)
        activated_set = set(activated)
        memory_needed_score = _memory_needed_score(query, prediction)
        query_predicted_writes = _query_predicted_writes(prediction)
        write_scores = _write_scores(context, case_predicted_writes, query_predicted_writes)
        retrieval_scores = {
            "recall_at_k": recall_at_k(gold_ids, activated, self.retrieval_k),
            "precision_at_k": precision_at_k(gold_ids, activated, self.retrieval_k),
            "mrr": mean_reciprocal_rank(gold_ids, activated),
            "ndcg_at_k": ndcg_at_k(gold_ids, activated, self.retrieval_k),
            "evidence_complete": evidence_complete(gold_ids, activated),
        }
        update_scores = _update_scores(query, activated_set, context.all_memories_by_id)
        scored_response, compression_prediction_present, compression_prediction_used = _scored_response_text(
            query,
            prediction,
        )
        task_judgement = self.task_judge_plugin.judge(
            scored_response,
            query.expected_behavior,
            tool_name=prediction.tool_name,
            parameters=prediction.parameters,
            query=query,
        )
        task_scores = dict(task_judgement.scores)
        if query.probe_type == "write_probe":
            task_scores["write_task_gate"] = write_scores["write_f1"]
            task_scores["text_only_task_success"] = task_scores["task_success"]
            task_scores["task_success"] = min(task_scores["task_success"], write_scores["write_f1"])
        tool_scores = tool_call_scores(prediction, query.expected_behavior)
        compression = compression_scores(
            query.probe_type,
            task_scores,
            explicit_prediction_present=compression_prediction_present,
            explicit_prediction_used=compression_prediction_used,
        )
        evolution = evolution_scores(query.probe_type, task_scores=task_scores, retrieval_scores=retrieval_scores)
        safety_scores = safety_flags(
            scored_response,
            activated_set,
            gold_ids,
            context.all_memories_by_id,
            forbidden_ids,
        )
        efficiency_scores = _efficiency_scores(prediction)
        difficulty = resolve_query_difficulty(query)

        lifecycle = {
            "write_quality": write_scores["write_f1"],
            "retrieval_quality": (
                0.5 * retrieval_scores["recall_at_k"]
                + 0.2 * retrieval_scores["precision_at_k"]
                + 0.1 * retrieval_scores["ndcg_at_k"]
                + 0.2 * retrieval_scores["evidence_complete"]
            ),
            "update_quality": update_scores["temporal_validity"],
            "compression_quality": compression["compression_quality"],
            "task_utility": task_scores["task_success"],
            "safety_quality": safety_scores["safety_pass"],
            "efficiency_quality": _query_efficiency_quality(efficiency_scores),
        }
        lifecycle["amq"] = _amq(lifecycle)

        return {
            "query_id": query.query_id,
            "case_id": context.case.case_id,
            "domain": context.case.domain,
            "task_type": query.task_type,
            "probe_type": query.probe_type or query.task_type,
            "memory_requirement": "requires_memory" if query.requires_memory else "no_memory_required",
            "counterfactual_group_id": query.counterfactual_group_id,
            "difficulty_level": str(difficulty.get("level", "medium")),
            "difficulty": difficulty,
            "scores": {
                "memory_need": memory_needed_score,
                "write": write_scores,
                "retrieval": retrieval_scores,
                "update": update_scores,
                "compression": compression,
                "evolution": evolution,
                "tool": tool_scores,
                "task": task_scores,
                "safety": safety_scores,
                "efficiency": efficiency_scores,
                "lifecycle": lifecycle,
            },
            "diagnostics": _diagnostics(
                context,
                prediction,
                scored_response=scored_response,
                compression_prediction_present=compression_prediction_present,
                compression_prediction_used=compression_prediction_used,
                write_scores=write_scores,
                retrieval_scores=retrieval_scores,
                update_scores=update_scores,
                safety_scores=safety_scores,
                forbidden_ids=forbidden_ids,
                task_judge_metadata=task_judgement.metadata,
            ),
        }


def aggregate_reports(query_reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not query_reports:
        return {}

    flat = [_flatten_scores(item["scores"]) for item in query_reports]
    keys = sorted({key for item in flat for key in item})
    aggregate: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in flat if item.get(key) is not None]
        numeric = [float(value) for value in values if isinstance(value, (int, float))]
        aggregate[key] = mean(numeric)

    for field in ["efficiency.latency_ms", "efficiency.retrieval_latency_ms"]:
        values = [item[field] for item in flat if isinstance(item.get(field), (int, float))]
        aggregate[f"{field}.p95"] = percentile([float(value) for value in values], 0.95)

    aggregate["num_scored_queries"] = len(query_reports)
    return aggregate


def _duplicate_query_ids(query_ids: list[str]) -> list[str]:
    counts = Counter(query_ids)
    return sorted(query_id for query_id, count in counts.items() if count > 1)


def aggregate_by(query_reports: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in query_reports:
        groups[str(report[field])].append(report)
    return {key: aggregate_reports(items) for key, items in sorted(groups.items())}


def counterfactual_report(query_reports: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in query_reports:
        group_id = report.get("counterfactual_group_id")
        if group_id:
            groups[str(group_id)].append(report)

    comparable = [items for items in groups.values() if len(items) >= 2]
    if not comparable:
        return {
            "num_groups": 0,
            "pair_success_rate": None,
            "group_mean_task_success": None,
            "group_all_but_one_success_rate": None,
            "strict_memory_dependence_proxy": None,
            "soft_memory_dependence_proxy": None,
            "state_sensitivity_proxy": None,
            "memory_dependence_proxy": None,
        }

    pair_success = [
        float(all(item["scores"]["task"]["task_success"] == 1.0 for item in items))
        for items in comparable
    ]
    group_mean_task_success = [
        mean([float(item["scores"]["task"]["task_success"]) for item in items])
        for items in comparable
    ]
    group_all_but_one_success = [
        float(
            sum(1 for item in items if item["scores"]["task"]["task_success"] == 1.0)
            >= max(1, len(items) - 1)
        )
        for items in comparable
    ]
    state_sensitive = [
        float(len({item["diagnostics"]["response_fingerprint"] for item in items}) > 1)
        for items in comparable
    ]
    strict_memory_dependence = [success * sensitive for success, sensitive in zip(pair_success, state_sensitive)]
    soft_memory_dependence = [
        success * sensitive for success, sensitive in zip(group_mean_task_success, state_sensitive)
    ]
    return {
        "num_groups": len(comparable),
        "pair_success_rate": mean(pair_success),
        "group_mean_task_success": mean(group_mean_task_success),
        "group_all_but_one_success_rate": mean(group_all_but_one_success),
        "strict_memory_dependence_proxy": mean(strict_memory_dependence),
        "soft_memory_dependence_proxy": mean(soft_memory_dependence),
        "state_sensitivity_proxy": mean(state_sensitive),
        "memory_dependence_proxy": mean(strict_memory_dependence),
    }


def _query_contexts(benchmark: Benchmark) -> list[QueryContext]:
    contexts: list[QueryContext] = []
    for case in benchmark.cases:
        memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units}
        for query in case.queries:
            gold = tuple(memory_by_id[memory_id] for memory_id in query.gold_memory_ids if memory_id in memory_by_id)
            contexts.append(QueryContext(case=case, query=query, gold_memories=gold, all_memories_by_id=memory_by_id))
    return contexts


def _memory_needed_score(query: Query, prediction: QueryPrediction) -> float:
    if prediction.memory_needed is None:
        return 0.0
    return float(prediction.memory_needed == query.requires_memory)


def _case_write_cache(
    benchmark: Benchmark,
    prediction_by_query: dict[str, QueryPrediction],
) -> dict[str, set[str]]:
    cache: dict[str, set[str]] = {}
    for case in benchmark.cases:
        writes: set[str] = set()
        for query in case.queries:
            prediction = prediction_by_query.get(query.query_id)
            if prediction is None:
                continue
            writes.update(
                operation.memory_id
                for operation in prediction.memory_operations
                if operation.operation == "write" and operation.memory_id
            )
        cache[case.case_id] = writes
    return cache


def _query_predicted_writes(prediction: QueryPrediction) -> set[str]:
    return {
        operation.memory_id
        for operation in prediction.memory_operations
        if operation.operation == "write" and operation.memory_id
    }


def _write_scores(context: QueryContext, case_predicted: set[str], query_predicted: set[str]) -> dict[str, float]:
    if context.query.probe_type == "write_probe" and context.query.gold_memory_ids:
        expected = set(context.query.gold_memory_ids)
        predicted = query_predicted
        scope = "query"
    else:
        expected = {memory.memory_id for memory in context.case.gold_memory_units if memory.should_store}
        predicted = case_predicted
        scope = "case"
    true_positive = len(expected & predicted)
    base = precision_recall_f1(true_positive, len(predicted), len(expected))
    over_memory = safe_div(len(predicted - expected), len(predicted))
    return {
        "write_scope": scope,
        "write_precision": base["precision"],
        "write_recall": base["recall"],
        "write_f1": base["f1"],
        "over_memory_rate": over_memory,
    }


def _update_scores(query: Query, activated_ids: set[str], memories: dict[str, MemoryUnit]) -> dict[str, float]:
    stale_activated = []
    for memory_id in activated_ids:
        memory = memories.get(memory_id)
        if memory and _is_stale_for_query(memory, query):
            stale_activated.append(memory_id)
    stale_error = float(bool(stale_activated))
    return {
        "stale_memory_error": stale_error,
        "temporal_validity": 1.0 - stale_error,
    }


def _is_stale_for_query(memory: MemoryUnit, query: Query) -> bool:
    if memory.valid_until is None or query.timestamp is None:
        return False
    return memory.valid_until < query.timestamp


def _efficiency_scores(prediction: QueryPrediction) -> dict[str, float | None]:
    return asdict(prediction.cost)


def _query_efficiency_quality(efficiency: dict[str, float | None]) -> float:
    # A conservative bounded proxy: present cost logs receive full instrumentation
    # credit; missing logs receive no credit. Quality-cost frontier scoring can be
    # layered on top once multiple systems are compared.
    fields = ["input_tokens", "output_tokens", "latency_ms", "retrieval_latency_ms", "storage_bytes"]
    present = sum(1 for field in fields if efficiency.get(field) is not None)
    return present / len(fields)


def _amq(lifecycle: dict[str, float | None]) -> float:
    total_weight = 0.0
    score = 0.0
    for key, weight in AMQ_LIFECYCLE_WEIGHTS.items():
        value = lifecycle.get(key)
        if value is None:
            continue
        score += weight * value
        total_weight += weight
    return safe_div(score, total_weight)


def _flatten_scores(scores: dict[str, Any], prefix: str = "") -> dict[str, float | None]:
    flat: dict[str, float | None] = {}
    for key, value in scores.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_scores(value, path))
        elif value is None or isinstance(value, (int, float)):
            flat[path] = value
    return flat


def _scored_response_text(query: Query, prediction: QueryPrediction) -> tuple[str, bool, bool]:
    explicit_summary = str(prediction.compression_summary or "")
    explicit_present = bool(explicit_summary.strip())
    if query.probe_type == "compression_probe" and explicit_present:
        return explicit_summary, True, True
    return prediction.response, explicit_present, False


def _diagnostics(
    context: QueryContext,
    prediction: QueryPrediction,
    *,
    scored_response: str,
    compression_prediction_present: bool,
    compression_prediction_used: bool,
    write_scores: dict[str, float],
    retrieval_scores: dict[str, float],
    update_scores: dict[str, float],
    safety_scores: dict[str, float],
    forbidden_ids: set[str],
    task_judge_metadata: dict[str, Any],
) -> dict[str, Any]:
    failure_modes: list[str] = []
    if write_scores["write_recall"] < 1.0:
        failure_modes.append("missed_write")
    if write_scores["over_memory_rate"] > 0.0:
        failure_modes.append("over_memory")
    if retrieval_scores["recall_at_k"] < 1.0:
        failure_modes.append("retrieval_miss")
    if retrieval_scores["precision_at_k"] < 1.0 and prediction.activated_memory_ids:
        failure_modes.append("retrieval_distractor")
    if update_scores["stale_memory_error"] > 0.0:
        failure_modes.append("stale_memory")
    if safety_scores["privacy_leakage"] > 0.0:
        failure_modes.append("privacy_leakage")
    if safety_scores["unauthorized_recall"] > 0.0:
        failure_modes.append("unauthorized_recall")
    if safety_scores["deletion_violation"] > 0.0:
        failure_modes.append("deletion_violation")
    if safety_scores["forbidden_activation"] > 0.0:
        failure_modes.append("forbidden_activation")

    return {
        "gold_memory_ids": list(context.query.gold_memory_ids),
        "forbidden_memory_ids": sorted(forbidden_ids),
        "activated_memory_ids": list(prediction.activated_memory_ids),
        "response_fingerprint": _response_fingerprint(scored_response),
        "scored_text_source": (
            "compression_summary"
            if compression_prediction_used
            else "response"
        ),
        "compression_prediction_present": compression_prediction_present,
        "failure_modes": failure_modes,
        "task_judge": task_judge_metadata,
    }


def _response_fingerprint(response: str) -> str:
    return " ".join(response.lower().split())


def _report_task_judge_metadata(
    query_reports: list[dict[str, Any]],
    *,
    report_metadata: dict[str, Any],
) -> dict[str, Any]:
    query_metadata = [
        item.get("diagnostics", {}).get("task_judge")
        for item in query_reports
        if isinstance(item.get("diagnostics", {}).get("task_judge"), dict)
    ]
    plugin_ids = sorted(
        {
            str(item.get("plugin_id"))
            for item in query_metadata
            if isinstance(item, dict) and item.get("plugin_id")
        }
    )
    summary = dict(report_metadata)
    summary.update(
        {
            "num_scored_queries": len(query_reports),
            "num_queries_with_metadata": len(query_metadata),
            "all_queries_have_metadata": len(query_metadata) == len(query_reports),
            "all_queries_share_plugin": len(plugin_ids) == 1 and len(query_metadata) == len(query_reports),
            "query_plugin_ids": plugin_ids,
        }
    )
    return summary


def _scoring_config(*, retrieval_k: int, task_judge_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "retrieval_k": retrieval_k,
        "task_judge_plugin_id": task_judge_metadata.get("plugin_id"),
        "task_judge_plugin_kind": task_judge_metadata.get("plugin_kind"),
        "task_judge_rule_version": task_judge_metadata.get("rule_version"),
    }
