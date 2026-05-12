"""Validation helpers based on Hellinger affinity."""

from __future__ import annotations

from collections import Counter
from math import exp, isfinite, lgamma, log, sqrt
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np


def _probability_vector(p: Iterable[float]) -> np.ndarray:
    probs = np.asarray(list(p), dtype=np.float64)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("probabilities must be a nonempty one-dimensional vector")
    if np.any(~np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    total = float(probs.sum())
    if total <= 0:
        raise ValueError("probabilities must have positive total mass")
    return probs / total


def bhattacharyya_iid_bounds(affinity_bounds: Tuple[float, float], threshold: float = 0.5) -> dict:
    """Translate one-sample affinity bounds into iid sample-size bounds.

    Returns a conservative lower/upper crossing range for the sample size at
    which total variation can no longer be guaranteed below ``threshold`` using
    the standard Hellinger-to-TVD upper bound ``TVD <= sqrt(1 - A^2)``.
    """
    lo, hi = map(float, affinity_bounds)
    if not (0.0 <= lo <= hi <= 1.0):
        raise ValueError("affinity_bounds must satisfy 0 <= lo <= hi <= 1")
    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must be in (0, 1)")
    target_affinity = sqrt(1.0 - threshold * threshold)

    def crossing(a: float) -> float:
        if a <= 0.0:
            return 1.0
        if a >= 1.0:
            return float("inf")
        return log(target_affinity) / log(a)

    lower = crossing(lo)
    upper = crossing(hi)
    return {
        "affinity_lower": lo,
        "affinity_upper": hi,
        "threshold": threshold,
        "target_affinity": target_affinity,
        "n_cross_lower": lower,
        "n_cross_upper": upper,
    }


def empirical_affinity(exact: Mapping[object, float], observed: Mapping[object, int]) -> float:
    """Estimate Hellinger affinity from exact probabilities and observed counts."""
    total = int(sum(observed.values()))
    if total <= 0:
        raise ValueError("observed counts must have positive total")
    affinity = 0.0
    for key, p in exact.items():
        q = observed.get(key, 0) / total
        if p > 0 and q > 0:
            affinity += sqrt(float(p) * q)
    return float(min(1.0, max(0.0, affinity)))


def categorical_exact_pmf(p: Iterable[float]) -> dict:
    """Return the exact categorical PMF as an index-keyed dict."""
    probs = _probability_vector(p)
    return {int(i): float(v) for i, v in enumerate(probs)}


def categorical_counts(draws: Sequence[int], index_base: int = 0) -> Counter:
    """Convert categorical draws to zero-based count keys."""
    base = int(index_base)
    return Counter(int(x) - base for x in draws)


def _compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def multinomial_exact_pmf(K: int, p: Iterable[float]) -> dict:
    """Enumerate a small multinomial PMF.

    This helper is intended for verification slices, not large production
    distributions.
    """
    trials = int(K)
    if trials < 0:
        raise ValueError("K must be nonnegative")
    probs = _probability_vector(p)
    pmf = {}
    for counts in _compositions(trials, probs.size):
        logp = lgamma(trials + 1)
        possible = True
        for c, prob in zip(counts, probs):
            logp -= lgamma(c + 1)
            if c > 0:
                if prob <= 0:
                    possible = False
                    break
                logp += c * log(float(prob))
        pmf[counts] = 0.0 if not possible else exp(logp)
    total = sum(pmf.values())
    if total > 0 and isfinite(total):
        pmf = {key: value / total for key, value in pmf.items()}
    return pmf


def multinomial_counts(draws: np.ndarray) -> Counter:
    """Convert multinomial draw rows to count-vector keys."""
    array = np.asarray(draws)
    if array.ndim != 2:
        raise ValueError("draws must be a two-dimensional array")
    return Counter(tuple(int(v) for v in row) for row in array)
