"""Public sampler API.

The initial backend delegates to NumPy. The signatures mirror the contract
files so native kernels can be dropped in without reshaping the user API.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Iterable, Optional

import numpy as np

from . import _backend
from .tuning import MultinomialTuner


Seed = Optional[int]


def _rng(seed: Seed) -> np.random.Generator:
    return np.random.default_rng(seed)


def _probability_vector(p: Iterable[float]) -> np.ndarray:
    probs = np.asarray(list(p), dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("p must be a one-dimensional probability vector")
    if probs.size == 0:
        raise ValueError("p must contain at least one category")
    if not np.all(np.isfinite(probs)):
        raise ValueError("p must contain only finite values")
    if np.any(probs < 0):
        raise ValueError("p must be nonnegative")
    total = float(probs.sum())
    if total <= 0:
        raise ValueError("p must have positive total mass")
    return probs / total


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be nonnegative")
    return out


def _positive_buffer_size(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError("buffer_size must be an integer")
    out = int(value)
    if out <= 0:
        raise ValueError("buffer_size must be positive")
    return out


def binomial(
    n: int,
    p: float,
    size: int = 1,
    *,
    seed: Seed = None,
    method: str = "auto",
) -> np.ndarray:
    """Draw binomial variates.

    Native methods are used when the extension is available; ``method="numpy"``
    always selects NumPy's implementation.
    """
    trials = _nonnegative_int("n", n)
    count = _nonnegative_int("size", size)
    prob = float(p)
    if not np.isfinite(prob) or prob < 0.0 or prob > 1.0:
        raise ValueError("p must be in [0, 1]")
    if method not in {"auto", "centerout", "wait2", "btrd", "numpy"}:
        raise ValueError("method must be 'auto', 'centerout', 'wait2', 'btrd', or 'numpy'")
    if _backend.native_available() and method in {"auto", "centerout", "wait2", "btrd"}:
        return _backend.binomial(trials, prob, count, seed, method)
    return _rng(seed).binomial(trials, prob, size=count).astype(np.int64, copy=False)


def categorical(
    p: Iterable[float],
    size: int = 1,
    *,
    seed: Seed = None,
    method: str = "auto",
    index_base: int = 0,
) -> np.ndarray:
    """Draw categorical labels from a probability vector."""
    if method not in {"auto", "numpy"}:
        raise ValueError(
            "method must be 'auto' or 'numpy' until native categorical "
            "dispatch is available"
        )
    count = _nonnegative_int("size", size)
    base = _nonnegative_int("index_base", index_base)
    if base not in {0, 1}:
        raise ValueError("index_base must be 0 or 1")
    probs = _probability_vector(p)
    draws = _rng(seed).choice(probs.size, size=count, p=probs)
    return draws.astype(np.int64, copy=False) + base


def multinomial(
    K: int,
    p: Iterable[float],
    size: int = 1,
    *,
    seed: Seed = None,
    method: str = "auto",
    tuner: Optional[MultinomialTuner] = None,
) -> np.ndarray:
    """Draw multinomial count vectors.

    Supplying a :class:`MultinomialTuner` enables explicit, stateful native
    kernel selection for repeated workloads. Calls without one remain
    stateless.
    """
    trials = _nonnegative_int("K", K)
    count = _nonnegative_int("size", size)
    probs = _probability_vector(p)
    if tuner is not None:
        if method not in {"auto", "adaptive"}:
            raise ValueError(
                "tuner may only be used with method='auto' or method='adaptive'"
            )
        if not _backend.native_available():
            raise RuntimeError(
                "adaptive multinomial dispatch requires the native backend"
            )
        if count == 0:
            return _backend.multinomial(trials, probs, count, seed, "pivot")
        chosen = tuner.choose(trials, probs, count)
        started_ns = perf_counter_ns()
        draws = _backend.multinomial(trials, probs, count, seed, chosen)
        elapsed_ns = max(1, perf_counter_ns() - started_ns)
        tuner.observe(trials, probs, count, chosen, elapsed_ns)
        return draws
    if method == "adaptive":
        raise ValueError("method='adaptive' requires a MultinomialTuner")
    native_methods = {
        "auto",
        "pivot",
        "cascade",
        "smallk-cdf",
        "smallK-cdf",
        "cdf",
        "rep",
        "thin",
        "alias",
    }
    if _backend.native_available() and method in native_methods:
        return _backend.multinomial(trials, probs, count, seed, method)
    if method not in {"auto", "numpy"}:
        raise ValueError(
            "method must be 'auto', 'pivot', 'cascade', 'smallk-cdf', 'rep', "
            "'thin', 'alias', or 'numpy'"
        )
    return _rng(seed).multinomial(trials, probs, size=count).astype(np.int64, copy=False)


def _mix_seed(seed: Seed, counter: int) -> Seed:
    if seed is None:
        return None
    x = (int(seed) + 0x9E3779B97F4A7C15 * (counter + 1)) & ((1 << 64) - 1)
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return int(x ^ (x >> 31))


@dataclass
class BinomialStream:
    """Buffered binomial sampler."""

    n: int
    p: float
    seed: Seed = None
    buffer_size: int = 4096
    method: str = "auto"

    def __post_init__(self) -> None:
        self.buffer_size = _positive_buffer_size(self.buffer_size)
        self._offset = 0
        self._buffer = np.empty(0, dtype=np.int64)

    def draw(self, size: int = 1) -> np.ndarray:
        count = _nonnegative_int("size", size)
        chunks = []
        while count > 0:
            if self._buffer.size == 0:
                refill = self.buffer_size
                self._buffer = binomial(
                    self.n,
                    self.p,
                    refill,
                    seed=_mix_seed(self.seed, self._offset),
                    method=self.method,
                )
                self._offset += 1
            take = min(count, self._buffer.size)
            chunks.append(self._buffer[:take])
            self._buffer = self._buffer[take:]
            count -= take
        if not chunks:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(chunks)


@dataclass
class CategoricalStream:
    """Buffered categorical sampler."""

    p: Iterable[float]
    seed: Seed = None
    buffer_size: int = 4096
    method: str = "auto"
    index_base: int = 0

    def __post_init__(self) -> None:
        self._p = _probability_vector(self.p)
        self.buffer_size = _positive_buffer_size(self.buffer_size)
        self._offset = 0
        self._buffer = np.empty(0, dtype=np.int64)

    def draw(self, size: int = 1) -> np.ndarray:
        count = _nonnegative_int("size", size)
        chunks = []
        while count > 0:
            if self._buffer.size == 0:
                refill = self.buffer_size
                self._buffer = categorical(
                    self._p,
                    refill,
                    seed=_mix_seed(self.seed, self._offset),
                    method=self.method,
                    index_base=self.index_base,
                )
                self._offset += 1
            take = min(count, self._buffer.size)
            chunks.append(self._buffer[:take])
            self._buffer = self._buffer[take:]
            count -= take
        if not chunks:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(chunks)


@dataclass
class MultinomialStream:
    """Buffered multinomial sampler."""

    K: int
    p: Iterable[float]
    seed: Seed = None
    buffer_size: int = 1024
    method: str = "auto"
    tuner: Optional[MultinomialTuner] = None

    def __post_init__(self) -> None:
        self._p = _probability_vector(self.p)
        self.buffer_size = _positive_buffer_size(self.buffer_size)
        self._offset = 0
        self._buffer = np.empty((0, self._p.size), dtype=np.int64)

    def draw(self, size: int = 1) -> np.ndarray:
        count = _nonnegative_int("size", size)
        chunks = []
        while count > 0:
            if self._buffer.shape[0] == 0:
                refill = self.buffer_size
                self._buffer = multinomial(
                    self.K,
                    self._p,
                    refill,
                    seed=_mix_seed(self.seed, self._offset),
                    method=self.method,
                    tuner=self.tuner,
                )
                self._offset += 1
            take = min(count, self._buffer.shape[0])
            chunks.append(self._buffer[:take, :])
            self._buffer = self._buffer[take:, :]
            count -= take
        if not chunks:
            return np.empty((0, self._p.size), dtype=np.int64)
        return np.vstack(chunks)
