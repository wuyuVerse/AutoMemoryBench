"""Task-utility judge plugins and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from amb.benchmark.metrics.task import expected_behavior_scores
from amb.benchmark.schemas.models import ExpectedBehavior, Query


TASK_JUDGE_METADATA_SCHEMA_VERSION = "amst-task-judge-metadata-v1"
DEFAULT_TASK_JUDGE_PLUGIN_ID = "deterministic_expected_behavior_v1"
DEFAULT_TASK_JUDGE_ALIASES = ("default", "deterministic", DEFAULT_TASK_JUDGE_PLUGIN_ID)


@dataclass(frozen=True)
class TaskJudgePluginMetadata:
    plugin_id: str
    plugin_kind: str
    provider: str | None = None
    model: str | None = None
    prompt_template_id: str | None = None
    rule_version: str | None = None
    supports_partial_credit: bool = True
    supports_reference_free_judging: bool = False
    supports_structured_tool_checks: bool = True
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TASK_JUDGE_METADATA_SCHEMA_VERSION,
            "plugin_id": self.plugin_id,
            "plugin_kind": self.plugin_kind,
            "provider": self.provider,
            "model": self.model,
            "prompt_template_id": self.prompt_template_id,
            "rule_version": self.rule_version,
            "supports_partial_credit": self.supports_partial_credit,
            "supports_reference_free_judging": self.supports_reference_free_judging,
            "supports_structured_tool_checks": self.supports_structured_tool_checks,
            "settings": dict(self.settings),
        }


@dataclass(frozen=True)
class TaskJudgeOutcome:
    scores: dict[str, float]
    metadata: dict[str, Any]


class TaskJudgePlugin(Protocol):
    plugin_id: str

    def judge(
        self,
        response: str,
        expected: ExpectedBehavior,
        *,
        tool_name: str | None = None,
        parameters: dict[str, Any] | None = None,
        query: Query | None = None,
    ) -> TaskJudgeOutcome: ...

    def report_metadata(self) -> dict[str, Any]: ...


class DeterministicTaskJudgePlugin:
    """Built-in rule-based judge that preserves the original task scoring semantics."""

    plugin_id = DEFAULT_TASK_JUDGE_PLUGIN_ID

    def __init__(self) -> None:
        self._metadata = TaskJudgePluginMetadata(
            plugin_id=self.plugin_id,
            plugin_kind="deterministic",
            provider="builtin",
            model=None,
            prompt_template_id=None,
            rule_version="expected_behavior_scores_v1",
            supports_partial_credit=True,
            supports_reference_free_judging=False,
            supports_structured_tool_checks=True,
            settings={
                "must_include_clause_overlap_threshold": 0.6,
                "refusal_marker_policy": "default_refusal_markers_v1",
            },
        )

    def judge(
        self,
        response: str,
        expected: ExpectedBehavior,
        *,
        tool_name: str | None = None,
        parameters: dict[str, Any] | None = None,
        query: Query | None = None,
    ) -> TaskJudgeOutcome:
        scores = expected_behavior_scores(
            response,
            expected,
            tool_name=tool_name,
            parameters=parameters,
            scoring_rule=None if query is None else query.scoring_rule,
        )
        metadata = self.report_metadata()
        metadata.update(
            {
                "used_structured_tool_name": tool_name is not None,
                "used_structured_parameters": bool(parameters),
                "probe_type": None if query is None else query.probe_type,
                "task_type": None if query is None else query.task_type,
            }
        )
        return TaskJudgeOutcome(scores=scores, metadata=metadata)

    def report_metadata(self) -> dict[str, Any]:
        return self._metadata.to_dict()


def available_task_judge_plugins() -> tuple[str, ...]:
    return ("deterministic", DEFAULT_TASK_JUDGE_PLUGIN_ID)


def load_task_judge_plugin(name: str | None = None) -> TaskJudgePlugin:
    normalized = str(name or "deterministic").strip().lower()
    if normalized in DEFAULT_TASK_JUDGE_ALIASES:
        return DeterministicTaskJudgePlugin()
    raise ValueError(f"unknown task judge plugin: {name}")
