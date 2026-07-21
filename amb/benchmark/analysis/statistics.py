"""Statistical primitives used by AutoMemoryBench report analysis."""

from __future__ import annotations

import math
import random
from typing import Any

DEFAULT_CI_LEVEL = 0.95


def bootstrap_mean_ci(
    values: list[float],
    *,
    rng: random.Random,
    samples: int,
    confidence: float = DEFAULT_CI_LEVEL,
) -> dict[str, Any]:
    """Estimate a percentile bootstrap confidence interval for a mean."""

    point = mean(values)
    if point is None:
        return {
            "mean": None,
            "lower": None,
            "upper": None,
            "confidence": confidence,
            "num_observations": 0,
            "samples": 0,
        }

    if len(values) == 1 or samples <= 0:
        lower = point
        upper = point
        actual_samples = 0
    else:
        sample_means = []
        n = len(values)
        for _ in range(samples):
            sample_means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
        sample_means.sort()
        alpha = (1.0 - confidence) / 2.0
        lower = quantile(sample_means, alpha)
        upper = quantile(sample_means, 1.0 - alpha)
        actual_samples = samples

    return {
        "mean": point,
        "lower": lower,
        "upper": upper,
        "confidence": confidence,
        "num_observations": len(values),
        "samples": actual_samples,
    }


def sign_flip_p_value(differences: list[float], *, rng: random.Random, samples: int) -> float | None:
    """Monte Carlo paired sign-flip test for a zero-mean difference."""

    if not differences:
        return None

    observed = abs(sum(differences) / len(differences))
    if observed == 0.0:
        return 1.0

    n = len(differences)
    if samples <= 0:
        samples = 1

    exceedances = 0
    for _ in range(samples):
        permuted_mean = sum(value if rng.random() < 0.5 else -value for value in differences) / n
        if abs(permuted_mean) >= observed:
            exceedances += 1
    return (exceedances + 1) / (samples + 1)


def numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def spearman_correlation(first: list[float], second: list[float]) -> float | None:
    if len(first) != len(second):
        raise ValueError("correlation inputs must have the same length")
    if len(first) < 2:
        return None
    return pearson_correlation(_ranks(first), _ranks(second))


def kendall_tau_b(first: list[float], second: list[float]) -> float | None:
    if len(first) != len(second):
        raise ValueError("correlation inputs must have the same length")
    n = len(first)
    if n < 2:
        return None
    concordant = discordant = ties_first = ties_second = 0
    for left in range(n):
        for right in range(left + 1, n):
            first_cmp = _compare(first[left], first[right])
            second_cmp = _compare(second[left], second[right])
            if first_cmp == 0 and second_cmp == 0:
                continue
            if first_cmp == 0:
                ties_first += 1
            elif second_cmp == 0:
                ties_second += 1
            elif first_cmp == second_cmp:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_first) * (concordant + discordant + ties_second))
    if denominator == 0:
        return None
    return (concordant - discordant) / denominator


def pearson_correlation(first: list[float], second: list[float]) -> float | None:
    if len(first) != len(second):
        raise ValueError("correlation inputs must have the same length")
    if len(first) < 2:
        return None
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    numerator = sum((x - first_mean) * (y - second_mean) for x, y in zip(first, second))
    first_var = sum((x - first_mean) ** 2 for x in first)
    second_var = sum((y - second_mean) ** 2 for y in second)
    denominator = math.sqrt(first_var * second_var)
    if denominator == 0:
        return None
    return numerator / denominator


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for index in range(cursor, end):
            ranks[indexed[index][0]] = average_rank
        cursor = end
    return ranks


def _compare(left: float, right: float) -> int:
    if left < right:
        return -1
    if left > right:
        return 1
    return 0
