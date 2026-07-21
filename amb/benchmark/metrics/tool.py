"""Tool-call metrics for action-oriented probes."""

from __future__ import annotations

from typing import Any

from amb.benchmark.schemas.models import ExpectedBehavior, QueryPrediction


def tool_call_scores(prediction: QueryPrediction, expected: ExpectedBehavior) -> dict[str, float | None]:
    """Score structured tool calls and parameter completion.

    The benchmark accepts structured `tool_name` and `parameters` on
    predictions. For backward compatibility, textual responses can still satisfy
    task `must_include`, but tool metrics require structured fields.
    """

    if not expected.tool_name:
        return {
            "tool_name_exact": None,
            "parameter_precision": None,
            "parameter_recall": None,
            "parameter_f1": None,
            "tool_call_success": None,
        }

    tool_name_exact = float(prediction.tool_name == expected.tool_name)
    expected_params = {str(key): str(value) for key, value in expected.parameters.items()}
    predicted_params = {str(key): str(value) for key, value in prediction.parameters.items()}
    if not expected_params:
        parameter_precision = 1.0
        parameter_recall = 1.0
        parameter_f1 = 1.0
    else:
        matching = sum(1 for key, value in predicted_params.items() if expected_params.get(key) == value)
        parameter_precision = matching / len(predicted_params) if predicted_params else 0.0
        parameter_recall = matching / len(expected_params)
        denominator = parameter_precision + parameter_recall
        parameter_f1 = 0.0 if denominator == 0.0 else 2 * parameter_precision * parameter_recall / denominator
    return {
        "tool_name_exact": tool_name_exact,
        "parameter_precision": parameter_precision,
        "parameter_recall": parameter_recall,
        "parameter_f1": parameter_f1,
        "tool_call_success": float(tool_name_exact == 1.0 and parameter_f1 == 1.0),
    }
