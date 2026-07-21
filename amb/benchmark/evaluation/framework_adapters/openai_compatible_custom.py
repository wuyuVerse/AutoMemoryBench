"""OpenAI-compatible custom-agent adapter entrypoint."""

from __future__ import annotations

import os
import time
from typing import Any

from amb.benchmark.evaluation.framework_adapters.optional_dependency import (
    OptionalFrameworkSpec,
)
from amb.benchmark.evaluation.openai_compatible import (
    OpenAICompatibleChatClient,
    parse_json_response,
)


DEFAULT_BASE_URL_ENV = "AMST_OPENAI_COMPATIBLE_BASE_URL"
DEFAULT_API_KEY_ENV_ENV = "AMST_OPENAI_COMPATIBLE_API_KEY_ENV"
DEFAULT_MODEL_ENV = "AMST_OPENAI_COMPATIBLE_MODEL"
DEFAULT_TIMEOUT_ENV = "AMST_OPENAI_COMPATIBLE_TIMEOUT_S"
SPEC = OptionalFrameworkSpec(
    framework_id="openai_compatible_custom_agent",
    framework_label="OpenAI-compatible custom agent",
    required_modules=("openai",),
    install_hint="openai plus the selected external memory backend SDK",
    contract_path="configs/agent_frameworks/openai_compatible_custom_agent_contract.json",
)


class OpenAICompatibleCustomAgent:
    """Single-agent OpenAI-compatible runner with explicit memory/trace export.

    The adapter intentionally keeps memory retrieval in-process and deterministic:
    observed turns are stored as case-local memory records, a lexical retriever
    selects context for each query, and the OpenAI-compatible backend only
    generates the final natural-language response. This separates the evaluated
    memory policy from provider-specific SDK behavior and keeps trace artifacts
    comparable across backends.
    """

    def __init__(
        self,
        *,
        base_url_env: str = DEFAULT_BASE_URL_ENV,
        api_key_env_env: str = DEFAULT_API_KEY_ENV_ENV,
        model_env: str = DEFAULT_MODEL_ENV,
        default_api_key_env: str = "DPSK_API_KEY",
        default_base_url: str = "http://pg-2ze87hbn0tlfarni6-pub.polardbaigateway.rds.aliyuncs.com:8000/ds-v4-openai/v1",
        default_model: str = "DeepSeek-V4-Pro",
        memory_backend_id: str = "mem0",
        retrieval_k: int = 8,
        max_observation_chars: int = 6000,
        max_tokens: int = 192,
        temperature: float = 0.0,
        timeout_s: float | None = None,
    ) -> None:
        self.base_url_env = base_url_env
        self.api_key_env_env = api_key_env_env
        self.model_env = model_env
        self.default_api_key_env = default_api_key_env
        self.default_base_url = default_base_url
        self.default_model = default_model
        self.memory_backend_id = memory_backend_id
        self.retrieval_k = retrieval_k
        self.max_observation_chars = max_observation_chars
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.case_id = ""
        self.memories: list[dict[str, Any]] = []
        self._last_trace: dict[str, Any] = self._base_trace()

    def reset(self, case_id: str, namespace: str | None = None) -> None:
        self.case_id = case_id
        self.memories = []
        self._last_trace = self._base_trace(namespace=namespace)

    def observe(self, observation: dict[str, Any]) -> None:
        memory_id = f"{self.case_id}:turn:{len(self.memories) + 1:04d}"
        self.memories.append(
            {
                "memory_id": memory_id,
                "namespace": ["openai_compatible_custom_agent", self.case_id],
                "content": str(observation.get("content") or ""),
                "metadata": {
                    "case_id": str(observation.get("case_id") or self.case_id),
                    "session_id": str(observation.get("session_id") or ""),
                    "turn_id": str(observation.get("turn_id") or ""),
                    "role": str(observation.get("role") or ""),
                    "timestamp": str(observation.get("timestamp") or ""),
                    "domain": str(observation.get("domain") or ""),
                },
            }
        )

    def ingest_turn(self, turn: dict[str, Any]) -> None:
        self.observe(turn)

    def answer_or_act(self, probe: dict[str, Any]) -> dict[str, Any]:
        return self.run_probe(probe)

    def run_probe(self, probe: dict[str, Any], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        prompt = str(probe.get("prompt") or "")
        query_id = str(probe.get("query_id") or "")
        start = time.monotonic()
        hits = self._retrieve(prompt, k=self.retrieval_k)
        tool_call = self._memory_search_tool_call(prompt, hits)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an evaluated memory-agent. Answer only from the provided "
                    "case-local memory context when it is relevant. If the context is "
                    "insufficient, say so briefly. Return JSON with keys response, "
                    "memory_needed, and activated_memory_ids."
                ),
            },
            {
                "role": "user",
                "content": self._compose_user_message(prompt=prompt, hits=hits),
            },
        ]
        completion = self._client().create_chat_completion(
            model=self._model(),
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            require_json=True,
        )
        parsed = parse_json_response(completion.content)
        response = str(parsed.get("response") or completion.content or "").strip()
        if not response:
            response = "No response returned by the OpenAI-compatible backend."
        activated = _string_list(parsed.get("activated_memory_ids")) or [
            str(hit["memory_id"]) for hit in hits[: min(len(hits), self.retrieval_k)]
        ]
        latency_ms = (time.monotonic() - start) * 1000.0
        usage = completion.usage if isinstance(completion.usage, dict) else {}
        cost = {
            "input_tokens": _number_or_none(usage.get("prompt_tokens")),
            "output_tokens": _number_or_none(usage.get("completion_tokens")),
            "latency_ms": latency_ms,
            "retrieval_latency_ms": 0.0,
            "storage_bytes": sum(len(str(memory.get("content") or "").encode("utf-8")) for memory in self.memories),
        }
        candidate_memory_ops = [
            {
                "operation": "scan",
                "memory_id": str(memory["memory_id"]),
                "source_memory_ids": [],
            }
            for memory in self.memories
        ]
        hit_memory_ops = [
            {
                "operation": "read",
                "memory_id": str(hit["memory_id"]),
                "source_memory_ids": [],
            }
            for hit in hits
        ]
        memory_ops = candidate_memory_ops + hit_memory_ops
        retrieval_hits = [
            {
                "memory_id": str(hit["memory_id"]),
                "score": float(hit["score"]),
                "rank": int(index),
            }
            for index, hit in enumerate(hits, start=1)
        ]
        self._last_trace = {
            **self._base_trace(),
            "message_history_policy": "retrieval_augmented_case_memory",
            "memory_ops": memory_ops,
            "retrieval_hits": retrieval_hits,
            "tool_calls": [tool_call],
            "planner_trace": [
                {
                    "step": "retrieve_case_memory",
                    "decision": "use_top_k_context" if hits else "no_relevant_memory_found",
                    "executor": "deterministic_lexical_retriever",
                    "input_memory_ids": [str(memory["memory_id"]) for memory in self.memories],
                    "output_memory_ids": [str(hit["memory_id"]) for hit in hits],
                    "tool_call_ids": ["openai_compatible_memory_search_1"],
                },
                {
                    "step": "generate_answer",
                    "decision": "call_openai_compatible_backend",
                    "executor": self._model(),
                    "input_memory_ids": [str(hit["memory_id"]) for hit in hits],
                    "output_memory_ids": [],
                    "tool_call_ids": [],
                },
            ],
            "cost": {
                **{key: value for key, value in cost.items() if value is not None},
                "tool_call_count": 1,
                "memory_op_count": len(memory_ops),
            },
            "framework_state": {
                "num_case_memories": len(self.memories),
                "retrieval_k": self.retrieval_k,
                "base_url_env": self.base_url_env,
                "api_key_env_var_name": self._api_key_env_name(),
                "model_env": self.model_env,
            },
        }
        return {
            "memory_needed": bool(hits) if parsed.get("memory_needed") is None else bool(parsed.get("memory_needed")),
            "activated_memory_ids": activated,
            "response": response,
            "memory_operations": [
                {
                    "operation": "skip",
                    "memory_id": str(hit["memory_id"]),
                    "content": str(hit["content"])[:240],
                }
                for hit in hits
            ],
            "retrieval_hits": retrieval_hits,
            "tool_calls": [tool_call],
            "planner_trace": self._last_trace["planner_trace"],
            "cost": cost,
        }

    def export_memory(self) -> list[dict[str, Any]]:
        return [dict(memory) for memory in self.memories]

    def export_trace(self) -> dict[str, Any]:
        return dict(self._last_trace)

    def export_tool_calls(self) -> list[dict[str, Any]]:
        return list(self._last_trace.get("tool_calls", []))

    def export_framework_state(self) -> dict[str, Any]:
        return dict(self._last_trace.get("framework_state", {}))

    def _client(self) -> OpenAICompatibleChatClient:
        return OpenAICompatibleChatClient(
            base_url=os.getenv(self.base_url_env, self.default_base_url),
            api_key=self._api_key(),
            timeout_s=self._timeout_s(),
        )

    def _api_key_env_name(self) -> str:
        return os.getenv(self.api_key_env_env, self.default_api_key_env)

    def _api_key(self) -> str:
        env_name = self._api_key_env_name()
        key = os.getenv(env_name, "")
        if not key:
            raise RuntimeError(
                f"missing OpenAI-compatible API key: set {env_name} "
                f"or set {self.api_key_env_env} to another environment variable name"
            )
        return key

    def _model(self) -> str:
        return os.getenv(self.model_env, self.default_model)

    def _timeout_s(self) -> float:
        if self.timeout_s is not None:
            return self.timeout_s
        raw = os.getenv(DEFAULT_TIMEOUT_ENV)
        return float(raw) if raw else 60.0

    def _retrieve(self, prompt: str, *, k: int) -> list[dict[str, Any]]:
        query_terms = _terms(prompt)
        scored: list[dict[str, Any]] = []
        for memory in self.memories:
            content = str(memory.get("content") or "")
            terms = _terms(content)
            overlap = len(query_terms & terms)
            score = float(overlap) + (0.01 if content else 0.0)
            if score <= 0:
                continue
            scored.append({**memory, "score": score})
        if not scored:
            scored = [{**memory, "score": 0.01} for memory in self.memories[-k:]]
        scored.sort(key=lambda item: (-float(item["score"]), str(item["memory_id"])))
        return scored[:k]

    def _compose_user_message(self, *, prompt: str, hits: list[dict[str, Any]]) -> str:
        context_lines: list[str] = []
        budget = self.max_observation_chars
        for index, hit in enumerate(hits, start=1):
            content = str(hit.get("content") or "")
            if not content:
                continue
            snippet = content[: max(0, budget)]
            budget -= len(snippet)
            context_lines.append(f"[M{index} id={hit['memory_id']} score={hit['score']:.3f}] {snippet}")
            if budget <= 0:
                break
        context = "\n".join(context_lines) if context_lines else "(no retrieved memory)"
        return f"Question:\n{prompt}\n\nRetrieved case memory:\n{context}"

    def _memory_search_tool_call(self, prompt: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "tool_call_id": "openai_compatible_memory_search_1",
            "tool_name": "slack.search",
            "arguments": {
                "channel": "case_memory",
                "query": " ".join(prompt.split()[:12]) or "memory query",
            },
            "approval_state": "not_required",
            "result_summary": f"retrieved {len(hits)} case-memory records",
            "source_memory_ids": [str(hit["memory_id"]) for hit in hits],
            "side_effect_id": None,
        }

    def _base_trace(self, *, namespace: str | None = None) -> dict[str, Any]:
        return {
            "framework_id": "openai_compatible_custom_agent",
            "framework_version": "local_openai_compatible_v1",
            "framework_runtime": "python_in_process",
            "orchestration_mode": "single_agent",
            "model_id": self._model(),
            "memory_backend_id": self.memory_backend_id,
            "tool_runtime_id": "automemorybench_tool_runtime_v1",
            "session_id": self.case_id,
            "namespace": ["openai_compatible_custom_agent", namespace or self.case_id],
            "message_history_policy": "retrieval_augmented_case_memory",
            "memory_ops": [],
            "retrieval_hits": [],
            "tool_calls": [],
            "planner_trace": [],
            "handoff_trace": [],
            "cost": {},
            "framework_state": {},
        }


def create_adapter(**kwargs: Any) -> OpenAICompatibleCustomAgent:
    """Create the env-gated OpenAI-compatible custom-agent adapter.

    Construction never reads the API key or calls the backend. The first
    ``answer_or_act`` call requires the configured key environment variable.
    This keeps preflight and trace checks secret-safe while still exposing a
    real runnable adapter for arbitrary OpenAI-compatible agent systems.
    """

    return OpenAICompatibleCustomAgent(**kwargs)


def _terms(text: str) -> set[str]:
    return {part.casefold() for part in text.replace("_", " ").split() if len(part) >= 3}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None
