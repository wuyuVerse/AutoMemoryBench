"""Classification-style metrics."""

from __future__ import annotations


def precision_recall_f1(true_positive: int, predicted_positive: int, actual_positive: int) -> dict[str, float]:
    precision = safe_div(true_positive, predicted_positive)
    recall = safe_div(true_positive, actual_positive)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator

