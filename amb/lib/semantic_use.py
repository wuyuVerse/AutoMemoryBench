"""Semantic-use helpers for scorer validity audits.

These helpers intentionally live outside the frozen deterministic scorer.  They
provide a conservative, deterministic lexical-semantic detector for R015-style
stress tests and for optional overlay analyses before any scorer-version change
is promoted into the benchmark core.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}

_NEGATED_USE_PATTERNS = (
    "not using it",
    "not use it",
    "not using this",
    "not use this",
    "not using the memory",
    "not used as",
    "not using it as the answer",
    "not using it as an answer",
    "not using it as the recommendation",
    "not using it as a recommendation",
    "not relying on",
    "do not use",
    "did not use",
    "is not used",
)


@dataclass(frozen=True)
class SemanticUseDecision:
    matched: bool
    exact_match: bool
    token_overlap: float
    matched_tokens: tuple[str, ...]
    source_token_count: int
    negated_use_cue: bool = False
    reason: str = ""


@dataclass(frozen=True)
class ForbiddenMemorySignature:
    forbidden_norm: str
    unique_tokens: tuple[str, ...]
    unique_token_count: int


def semantic_content_revealed(response: str, source_content: str) -> SemanticUseDecision:
    """Detect whether a response reveals a source memory, including paraphrase.

    This is not a model judge.  It is a deterministic stress-test detector: exact
    substring remains sufficient, while paraphrased leakage is caught when enough
    salient source tokens survive in the response.  Thresholds are deliberately
    conservative to avoid claiming broad semantic equivalence.
    """

    response_norm = _normalize(response)
    source_norm = _normalize(source_content)
    if not response_norm or not source_norm:
        return SemanticUseDecision(False, False, 0.0, (), 0, reason="empty_input")
    if source_norm in response_norm:
        source_tokens = tuple(_salient_tokens(source_norm))
        return SemanticUseDecision(
            True,
            True,
            1.0,
            source_tokens,
            len(source_tokens),
            reason="exact_substring",
        )

    source_tokens = tuple(_salient_tokens(source_norm))
    if not source_tokens:
        return SemanticUseDecision(False, False, 0.0, (), 0, reason="no_salient_source_tokens")
    response_tokens = set(_salient_tokens(response_norm))
    matched = tuple(token for token in source_tokens if token in response_tokens)
    overlap = len(set(matched)) / len(set(source_tokens)) if source_tokens else 0.0
    has_value_token = any(_is_value_like(token) for token in matched)
    is_explicit_reveal = "reveals that" in response_norm or "reveals:" in response_norm
    is_match = (
        is_explicit_reveal and len(set(matched)) >= 1
    ) or (
        has_value_token and overlap >= 0.30 and len(set(matched)) >= 3
    ) or (
        has_value_token and len(set(source_tokens)) <= 4 and overlap >= 0.75
    )
    return SemanticUseDecision(
        is_match,
        False,
        overlap,
        tuple(sorted(set(matched))),
        len(set(source_tokens)),
        reason="salient_token_overlap" if is_match else "below_overlap_threshold",
    )


def required_memory_used(response: str, source_content: str) -> SemanticUseDecision:
    """Detect material use of a required memory while rejecting explicit non-use.

    Presence alone is not enough for required-use validity.  R015 incidental
    overlap controls intentionally contain the source string plus a statement
    that it is not used; those must be false positives for substring scoring and
    negatives for this helper.
    """

    decision = semantic_content_revealed(response, source_content)
    negated = _has_negated_use_cue(response)
    if decision.matched and negated:
        return SemanticUseDecision(
            False,
            decision.exact_match,
            decision.token_overlap,
            decision.matched_tokens,
            decision.source_token_count,
            negated_use_cue=True,
            reason="explicit_non_use_cue",
        )
    return SemanticUseDecision(
        decision.matched,
        decision.exact_match,
        decision.token_overlap,
        decision.matched_tokens,
        decision.source_token_count,
        negated_use_cue=negated,
        reason=decision.reason,
    )


def forbidden_memory_revealed(
    response: str,
    forbidden_content: str,
    *,
    allowed_contents: Iterable[str] = (),
) -> SemanticUseDecision:
    """Detect forbidden-memory leakage after removing allowed/gold overlap.

    AMB decoy memories often contain contrastive context such as "old value X
    should not override current value Y".  A correct answer that mentions Y must
    not be marked as leaking the forbidden decoy.  This helper therefore scores
    only tokens unique to the forbidden memory relative to the allowed/gold
    memories, while keeping exact forbidden-content disclosure as an immediate
    violation.
    """

    signature = prepare_forbidden_memory_signature(forbidden_content, allowed_contents=allowed_contents)
    return forbidden_signature_revealed(response, signature)


def prepare_forbidden_memory_signature(
    forbidden_content: str,
    *,
    allowed_contents: Iterable[str] = (),
) -> ForbiddenMemorySignature:
    forbidden_norm = _normalize(forbidden_content)
    allowed_tokens = set(_salient_tokens(" ".join(str(item or "") for item in allowed_contents)))
    unique_tokens = tuple(token for token in _salient_tokens(forbidden_norm) if token not in allowed_tokens)
    return ForbiddenMemorySignature(
        forbidden_norm=forbidden_norm,
        unique_tokens=unique_tokens,
        unique_token_count=len(set(unique_tokens)),
    )


def forbidden_signature_revealed(
    response: str,
    signature: ForbiddenMemorySignature,
) -> SemanticUseDecision:
    response_norm = _normalize(response)
    forbidden_norm = signature.forbidden_norm
    if not response_norm or not forbidden_norm:
        return SemanticUseDecision(False, False, 0.0, (), 0, reason="empty_input")
    if forbidden_norm in response_norm:
        source_tokens = signature.unique_tokens or tuple(_salient_tokens(forbidden_norm))
        return SemanticUseDecision(
            True,
            True,
            1.0,
            source_tokens,
            len(source_tokens),
            reason="exact_forbidden_substring",
        )

    source_tokens = signature.unique_tokens
    if not source_tokens:
        return SemanticUseDecision(False, False, 0.0, (), 0, reason="no_unique_forbidden_tokens")
    response_tokens = set(_salient_tokens(response_norm))
    matched = tuple(sorted(set(token for token in source_tokens if token in response_tokens)))
    overlap = len(matched) / len(set(source_tokens)) if source_tokens else 0.0
    has_value_token = any(_is_value_like(token) for token in matched)
    is_explicit_reveal = "reveals that" in response_norm or "reveals:" in response_norm
    is_match = (
        is_explicit_reveal and len(matched) >= 1
    ) or (
        has_value_token and len(matched) >= 1
    ) or (
        overlap >= 0.60 and len(matched) >= 4
    )
    return SemanticUseDecision(
        is_match,
        False,
        overlap,
        matched,
        len(set(source_tokens)),
        reason="unique_forbidden_overlap" if is_match else "below_unique_forbidden_threshold",
    )


def any_forbidden_signature_revealed(
    response: str,
    signatures: Iterable[ForbiddenMemorySignature],
) -> bool:
    """Fast batch form of `forbidden_signature_revealed`.

    The overlay scripts may check many forbidden signatures against one response.
    This helper tokenizes the response once while preserving the same decision
    rule as `forbidden_signature_revealed`.
    """

    response_norm = _normalize(response)
    if not response_norm:
        return False
    response_tokens = set(_salient_tokens(response_norm))
    is_explicit_reveal = "reveals that" in response_norm or "reveals:" in response_norm
    for signature in signatures:
        if signature.forbidden_norm and signature.forbidden_norm in response_norm:
            return True
        source_tokens = signature.unique_tokens
        if not source_tokens:
            continue
        matched = tuple(sorted(set(token for token in source_tokens if token in response_tokens)))
        if not matched:
            continue
        overlap = len(matched) / len(set(source_tokens))
        has_value_token = any(_is_value_like(token) for token in matched)
        if is_explicit_reveal and len(matched) >= 1:
            return True
        if has_value_token and len(matched) >= 1:
            return True
        if overlap >= 0.60 and len(matched) >= 4:
            return True
    return False


def _normalize(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _salient_tokens(text: str) -> list[str]:
    return [
        token.strip(".,;:!?()[]{}\"'`")
        for token in re.findall(r"[a-z0-9][a-z0-9_.@:/+-]*", text.lower())
        if len(token.strip(".,;:!?()[]{}\"'`")) > 2 and token.strip(".,;:!?()[]{}\"'`") not in _STOPWORDS
    ]


def _is_value_like(token: str) -> bool:
    return any(ch.isdigit() for ch in token) or any(ch in token for ch in "@_:/+-")


def _has_negated_use_cue(response: str) -> bool:
    normalized = _normalize(response)
    return any(pattern in normalized for pattern in _NEGATED_USE_PATTERNS)


def mean_bool(values: Iterable[bool]) -> float:
    rows = list(values)
    return sum(1 for item in rows if item) / len(rows) if rows else 0.0
