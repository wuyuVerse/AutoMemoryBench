"""Run-validity gate — keep silent agent failures out of the scores.

A sandbox/agent run can "succeed" (returncode 0, status=passed) yet produce no
real output: the CLI failed to authenticate or never wrote its answer, so every
response is empty. Scoring such a run yields a meaningless ~31% artifact (safety
trivially passes on empty text; empty-gold queries score recall 1.0).

This module classifies a prediction set as VALID / INVALID so callers can refuse
to emit an official score for an empty-output run. It does NOT touch the frozen
scorer — it is a gate around it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValidityVerdict:
    valid: bool
    num_predictions: int
    nonempty_response_rate: float
    model_call_rate: float | None
    reason: str


def assess_predictions(predictions, *, min_nonempty_rate: float = 0.05) -> ValidityVerdict:
    """Assess whether a run produced real agent output.

    `predictions` is a PredictionSet (or any obj with `.predictions`) or a list of
    prediction-like objects/dicts exposing `response` (str) and optional `cost`.
    A run is INVALID when almost all responses are empty — the signature of a
    silent CLI/auth failure.
    """
    rows = getattr(predictions, "predictions", predictions)
    rows = list(rows or [])
    n = len(rows)
    if n == 0:
        return ValidityVerdict(False, 0, 0.0, None, "no predictions")

    def _resp(r):
        return (getattr(r, "response", None) if not isinstance(r, dict) else r.get("response")) or ""

    def _has_cost(r):
        c = getattr(r, "cost", None) if not isinstance(r, dict) else r.get("cost")
        if c is None:
            return False
        d = c if isinstance(c, dict) else getattr(c, "__dict__", {})
        return any(v for v in (d or {}).values())

    nonempty = sum(1 for r in rows if str(_resp(r)).strip())
    called = sum(1 for r in rows if _has_cost(r))
    ne_rate = nonempty / n
    call_rate = called / n
    valid = ne_rate >= min_nonempty_rate
    reason = (
        "ok" if valid
        else f"empty-output artifact: only {nonempty}/{n} non-empty responses "
             f"({ne_rate:.1%} < {min_nonempty_rate:.0%}); likely silent CLI/auth failure"
    )
    return ValidityVerdict(valid, n, round(ne_rate, 4), round(call_rate, 4), reason)
