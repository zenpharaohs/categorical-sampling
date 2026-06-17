"""Validation helpers based on Hellinger affinity and randomized PIT samples."""

from __future__ import annotations

from collections import Counter
from math import exp, isfinite, lgamma, log, sqrt
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

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


RandomSource = Union[np.random.Generator, int, None]


def _rng_from_seed(rng: RandomSource) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def randomized_categorical_pit(
    draws: Sequence[int],
    p: Iterable[float],
    *,
    rng: RandomSource = None,
    index_base: int = 0,
) -> np.ndarray:
    """Map categorical draws to randomized PIT samples on ``(0, 1)``.

    For category ``j`` with target mass ``p[j]`` and left CDF edge
    ``F(j-1)``, this returns ``F(j-1) + V p[j]``. Under the exact categorical
    target and independent ``V ~ U(0,1)``, the transformed samples are uniform.
    """

    probs = _probability_vector(p)
    labels = np.asarray(draws, dtype=np.int64).reshape(-1) - int(index_base)
    if labels.size == 0:
        return np.empty(0, dtype=np.float64)
    if np.any(labels < 0) or np.any(labels >= probs.size):
        raise ValueError("draws contain labels outside the probability vector")
    generator = _rng_from_seed(rng)
    left_edges = np.concatenate(([0.0], np.cumsum(probs[:-1])))
    u = left_edges[labels] + generator.random(labels.size) * probs[labels]
    eps = np.finfo(np.float64).tiny
    return np.clip(u, eps, 1.0 - np.finfo(np.float64).eps)


def validate_categorical_draws(
    draws: Sequence[int],
    p: Iterable[float],
    *,
    rng: RandomSource = None,
    index_base: int = 0,
    d_max: int = 128,
    points_per_cell: Optional[int] = None,
    beta_cv: bool = True,
) -> dict:
    """Validate categorical draws with discrete and PIT-based diagnostics.

    The direct discrete affinity is always computed. If the sibling packages
    ``streaming-pit-validate`` and ``hellinger-qualify`` are importable, the
    randomized PIT sample is also passed through their streaming Legendre and
    smoothed-spacing Hellinger estimators. Missing optional packages simply
    leave the corresponding report as ``None``.
    """

    probs = _probability_vector(p)
    generator = _rng_from_seed(rng)
    pit = randomized_categorical_pit(draws, probs, rng=generator, index_base=index_base)
    exact = categorical_exact_pmf(probs)
    counts = categorical_counts(draws, index_base=index_base)
    affinity = empirical_affinity(exact, counts)
    out = {
        "n": int(pit.size),
        "discrete_affinity": affinity,
        "discrete_h2": 1.0 - affinity,
        "pit": pit,
        "legendre": None,
        "hellinger": None,
        "hellinger_cv": None,
    }

    try:
        from streaming_pit_validate import StreamingLegendreValidator
    except ModuleNotFoundError:
        StreamingLegendreValidator = None
    if StreamingLegendreValidator is not None and pit.size:
        validator = StreamingLegendreValidator(d_max=d_max)
        validator.update(pit)
        out["legendre"] = validator.report()

    try:
        from hellinger_qualify import beta_control_variate_fixed_estimate, estimate_hellinger
    except ModuleNotFoundError:
        estimate_hellinger = None
        beta_control_variate_fixed_estimate = None
    if estimate_hellinger is not None and pit.size:
        out["hellinger"] = estimate_hellinger(pit, points_per_cell=points_per_cell)
        if beta_cv and beta_control_variate_fixed_estimate is not None and pit.size >= 2:
            out["hellinger_cv"] = beta_control_variate_fixed_estimate(
                pit,
                generator,
                points_per_cell=points_per_cell,
            )
    return out


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
