"""Local official-source MemInsight client factory for AutoMemoryBench integration runs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from amb.clients.core.common import ensure_site_packages_from_venv, ensure_source_path

# The openai SDK (used by the OpenAI-compatible annotator) lives in the meminsight
# venv, not the pod's base interpreter; add its site-packages before importing openai.
DEFAULT_MEMINSIGHT_VENV = Path(__file__).resolve().parents[4] / ".venv-meminsight"


AnnotationFn = Callable[[str], str]


class MemInsightOfficialSourceClient:
    """Thin adapter over MemInsight's annotation-and-retrieval interface.

    The official repository is experiment-code oriented rather than a stable
    package. This wrapper preserves the official data contract: memories and
    queries are converted to ``[attribute]<value>`` annotations, then retrieval
    prioritizes matching attributes before falling back to lexical overlap.
    """

    def __init__(self, annotation_fn: AnnotationFn, *, default_limit: int = 5) -> None:
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        self.annotation_fn = annotation_fn
        self.default_limit = default_limit
        self.case_id: str | None = None
        self._rows: list[dict[str, Any]] = []

    def reset(self, *, case_id: str | None = None, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        self.case_id = case_id or user_id
        self._rows = []
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
        text = _content_from_payload(content=content, messages=messages)
        annotations = self.annotation_fn(text).strip().lower()
        attributes = _parse_annotations(annotations)
        row = {
            "id": f"meminsight-{len(self._rows) + 1}",
            "content": text,
            "annotations": annotations,
            "attributes": attributes,
            "metadata": dict(metadata or {}),
            "user_id": user_id or self.case_id,
        }
        self._rows.append(row)
        return {"id": row["id"], "annotations": annotations, "attributes": attributes, "user_id": row["user_id"]}

    def search(
        self,
        query: str | None = None,
        *,
        limit: int | None = None,
        top_k: int | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required for MemInsight search")
        k = int(limit or top_k or self.default_limit)
        query_annotations = self.annotation_fn(str(query)).strip().lower()
        query_attributes = _parse_annotations(query_annotations)
        scored = [
            _scored_row(row, query=str(query), query_annotations=query_annotations, query_attributes=query_attributes)
            for row in self._rows
        ]
        return [row for row in sorted(scored, key=lambda item: item["score"], reverse=True) if row["score"] > 0][:k]

    def get_all(self, **_: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "metadata": {
                    "annotations": row["annotations"],
                    "attributes": row["attributes"],
                    **dict(row.get("metadata") or {}),
                },
            }
            for row in self._rows
        ]

    def delete(self, memory_id: str | None = None, **_: Any) -> dict[str, Any]:
        if not memory_id:
            raise ValueError("memory_id is required for MemInsight delete")
        before = len(self._rows)
        self._rows = [row for row in self._rows if row["id"] != memory_id]
        return {"deleted": len(self._rows) != before, "memory_id": memory_id}


def create_client(
    *,
    source_root: str = "related_work/repos/MemInsight",
    default_limit: int = 5,
    annotation_fn: AnnotationFn | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    api_key: str | None = None,
    **_: Any,  # absorb embedding_* injected by apply_model (MemInsight is attribute-based, no vectors)
) -> MemInsightOfficialSourceClient:
    """Create a MemInsight client from the locally cloned official source tree.

    MemInsight's official annotation surface uses AWS Bedrock Claude. When a ``model``
    (+ ``base_url``/``api_key_env``) is supplied, build an OpenAI-compatible annotation
    function instead so MemInsight runs on any OpenAI-compatible chat endpoint — the attribute
    schema (``[key]<value>`` pairs) is identical; only the annotator backbone changes.
    """

    ensure_source_path(source_root)
    if annotation_fn is None:
        if model:
            annotation_fn = _openai_annotation_fn(model=model, base_url=base_url,
                                                  api_key_env=api_key_env, api_key=api_key)
        else:
            annotation_fn = _official_annotation_fn(source_root)
    return MemInsightOfficialSourceClient(annotation_fn, default_limit=default_limit)


def _openai_annotation_fn(*, model: str, base_url: str | None, api_key_env: str | None,
                          api_key: str | None) -> AnnotationFn:
    """MemInsight annotator over an OpenAI-compatible chat endpoint.

    Returns attributes in MemInsight's ``[attribute]<value>`` format (parsed by
    ``_parse_annotations``), matching the official Bedrock annotation schema.
    """
    import os
    ensure_site_packages_from_venv(DEFAULT_MEMINSIGHT_VENV)
    from openai import OpenAI

    key = api_key or (os.environ.get(api_key_env or "OPENAI_API_KEY") or "")
    client = OpenAI(base_url=base_url, api_key=key) if base_url else OpenAI(api_key=key)
    instruction = (
        "Extract the salient, query-relevant attributes from the text as a flat list of "
        "[attribute]<value> pairs (e.g. [topic]<billing> [entity]<acme corp> [date]<2026-01>). "
        "Use lowercase. Output ONLY the bracketed pairs on one line, no prose."
    )

    def annotate(text: str) -> str:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": instruction},
                          {"role": "user", "content": str(text)}],
                temperature=0.0,
                max_tokens=256,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            return ""

    return annotate


def _official_annotation_fn(source_root: str) -> AnnotationFn:
    """Resolve the official annotation surface without patching upstream code."""

    ensure_source_path(f"{source_root}/memory")
    try:
        from global_methods import run_claude_for_annotations  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MemInsight official source requires boto3/botocore/retry and AWS Bedrock credentials for annotation"
        ) from exc
    return lambda text: str(run_claude_for_annotations(text, type="turn"))


def _content_from_payload(*, content: Any, messages: Any) -> str:
    if messages is not None:
        values = messages if isinstance(messages, list) else [messages]
        return "\n".join(_message_text(value) for value in values)
    if content is None:
        raise ValueError("content or messages is required")
    return str(content)


def _message_text(message: Any) -> str:
    if isinstance(message, dict):
        role = str(message.get("role") or "user")
        return f"{role}: {message.get('content', '')}"
    return str(message)


def _parse_annotations(raw: str) -> dict[str, list[str]]:
    attributes: dict[str, list[str]] = {}
    for key, value in re.findall(r"\[(.*?)\]<([^>]*)>", raw):
        key = key.strip().lower()
        value = value.strip().lower()
        if key and value:
            attributes.setdefault(key, []).append(value)
    return attributes


def _scored_row(
    row: dict[str, Any],
    *,
    query: str,
    query_annotations: str,
    query_attributes: dict[str, list[str]],
) -> dict[str, Any]:
    attr_score = 0.0
    row_attributes = row["attributes"]
    for key, values in query_attributes.items():
        row_values = row_attributes.get(key, [])
        for value in values:
            if value in row_values or any(value in row_value or row_value in value for row_value in row_values):
                attr_score += 2.0
    lexical_score = _lexical_overlap(query, row["content"]) + _lexical_overlap(query_annotations, row["annotations"])
    score = attr_score + lexical_score
    return {
        "id": row["id"],
        "content": row["content"],
        "score": score,
        "metadata": {
            "annotations": row["annotations"],
            "query_annotations": query_annotations,
            "attributes": row_attributes,
            **dict(row.get("metadata") or {}),
        },
    }


def _lexical_overlap(left: str, right: str) -> float:
    left_terms = {term for term in re.findall(r"[a-z0-9_]+", left.lower()) if len(term) > 2}
    right_terms = {term for term in re.findall(r"[a-z0-9_]+", right.lower()) if len(term) > 2}
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(len(left_terms), 1)
