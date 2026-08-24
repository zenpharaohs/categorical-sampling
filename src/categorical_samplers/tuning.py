"""Contextual continuous-Bernoulli Thompson dispatch for multinomial kernels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import blake2b
from math import ceil, exp, isfinite, log2
from threading import RLock
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:
    from cb_sampler import CbStream, draw_streams
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional install
    if exc.name != "cb_sampler":
        raise
    CbStream = None
    draw_streams = None


_METHOD_ALIASES = {
    "pivot": "pivot",
    "cascade": "pivot",
    "smallk-cdf": "smallk-cdf",
    "smallK-cdf": "smallk-cdf",
    "cdf": "smallk-cdf",
    "rep": "rep",
    "thin": "rep",
    "alias": "rep",
}
_DEFAULT_METHODS = ("pivot", "smallk-cdf", "rep")


def _exact_nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be nonnegative")
    return out


def _normalized_probabilities(p: Iterable[float]) -> np.ndarray:
    probs = np.asarray(list(p), dtype=np.float64)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("p must be a nonempty one-dimensional probability vector")
    if np.any(~np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("p must contain finite nonnegative values")
    probs.sort()
    total = float(probs.sum())
    if not (total > 0.0):
        raise ValueError("p must have positive total mass")
    return probs / total


def _power_of_two_ceiling(value: float) -> int:
    if value <= 1.0:
        return 1
    return 1 << int(ceil(log2(value)))


@dataclass(frozen=True)
class MultinomialContext:
    """Permutation-invariant timing context used to share dispatch evidence."""

    trials: int
    categories: int
    batch_size: int
    nonzero_categories: int
    effective_categories_bin: int
    concentration_bin: int


@dataclass
class _Arm:
    stream: object
    observations: int = 0
    selections: int = 0
    ewma_ns_per_row: Optional[float] = None
    score: Optional[float] = None


@dataclass
class _Bucket:
    context: MultinomialContext
    arms: Dict[str, _Arm]


class MultinomialTuner:
    """Learn context-specific multinomial dispatch with Thompson sampling.

    Each context warms every qualified kernel once. Thereafter, one draw from
    each continuous-Bernoulli pseudo-posterior selects the kernel with the
    largest sampled relative-throughput score. Timing observations update an
    exponentially weighted latency estimate, while capped pseudo-counts keep
    the policy able to adapt when machine load changes.

    Passing a tuner to :func:`categorical_samplers.multinomial` is explicit:
    ordinary ``method="auto"`` calls remain stateless and reproducible.
    """

    def __init__(
        self,
        *,
        methods: Iterable[str] = _DEFAULT_METHODS,
        seed: int = 0,
        ewma_alpha: float = 0.25,
        max_strength: float = 32.0,
        score_floor: float = 0.01,
    ) -> None:
        if CbStream is None or draw_streams is None:
            raise ModuleNotFoundError(
                "MultinomialTuner requires cb-sampler from the public "
                "continuous-bernoulli repository"
            )
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)
        ):
            raise TypeError("seed must be an integer")
        seed = int(seed)
        if seed < 0 or seed >= (1 << 64):
            raise ValueError("seed must be in [0, 2**64)")
        ewma_alpha = float(ewma_alpha)
        max_strength = float(max_strength)
        score_floor = float(score_floor)
        if not (0.0 < ewma_alpha <= 1.0):
            raise ValueError("ewma_alpha must be in (0, 1]")
        if not isfinite(max_strength) or max_strength <= 0.0:
            raise ValueError("max_strength must be positive and finite")
        if not (0.0 < score_floor < 0.5):
            raise ValueError("score_floor must be in (0, 0.5)")

        canonical = []
        for method in methods:
            try:
                name = _METHOD_ALIASES[str(method)]
            except KeyError as exc:
                raise ValueError(f"unsupported tuning method: {method!r}") from exc
            if name not in canonical:
                canonical.append(name)
        if len(canonical) < 2:
            raise ValueError("at least two distinct tuning methods are required")

        self.methods: Tuple[str, ...] = tuple(canonical)
        self.seed = seed
        self.ewma_alpha = ewma_alpha
        self.max_strength = max_strength
        self.score_floor = score_floor
        self._buckets: Dict[MultinomialContext, _Bucket] = {}
        self._lock = RLock()
        self._closed = False

    def __enter__(self) -> "MultinomialTuner":
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("multinomial tuner is closed")

    @staticmethod
    def context(K: int, p: Iterable[float], size: int) -> MultinomialContext:
        trials = _exact_nonnegative_int("K", K)
        batch_size = _exact_nonnegative_int("size", size)
        probs = _normalized_probabilities(p)
        positive = probs[probs > 0.0]
        entropy = -float(np.sum(positive * np.log(positive)))
        effective = exp(entropy)
        concentration = 1.0 / float(np.max(probs))
        return MultinomialContext(
            trials=trials,
            categories=int(probs.size),
            batch_size=batch_size,
            nonzero_categories=int(positive.size),
            effective_categories_bin=_power_of_two_ceiling(effective),
            concentration_bin=_power_of_two_ceiling(concentration),
        )

    def _stream_index(self, context: MultinomialContext, method: str) -> int:
        payload = (repr(context) + "|" + method).encode("utf-8")
        digest = blake2b(payload, digest_size=8, person=b"cat-tuning-v1").digest()
        return int.from_bytes(digest, "little")

    def _bucket(self, context: MultinomialContext) -> _Bucket:
        bucket = self._buckets.get(context)
        if bucket is not None:
            return bucket
        arms = {
            method: _Arm(
                CbStream(
                    0.0,
                    0.0,
                    seed=self.seed,
                    stream_idx=self._stream_index(context, method),
                    buf_size=64,
                )
            )
            for method in self.methods
        }
        bucket = _Bucket(context=context, arms=arms)
        self._buckets[context] = bucket
        return bucket

    def choose(self, K: int, p: Iterable[float], size: int) -> str:
        """Choose one canonical native method for the supplied context."""

        self._require_open()
        context = self.context(K, p, size)
        with self._lock:
            bucket = self._bucket(context)
            for method in self.methods:
                arm = bucket.arms[method]
                if arm.observations == 0 and arm.selections == 0:
                    arm.selections += 1
                    return method

            if not all(arm.observations > 0 for arm in bucket.arms.values()):
                method = min(
                    self.methods,
                    key=lambda name: (
                        bucket.arms[name].selections,
                        self.methods.index(name),
                    ),
                )
            else:
                samples = draw_streams([bucket.arms[name].stream for name in self.methods])
                method = self.methods[int(np.argmax(samples))]
            bucket.arms[method].selections += 1
            return method

    def observe(
        self,
        K: int,
        p: Iterable[float],
        size: int,
        method: str,
        elapsed_ns: float,
    ) -> None:
        """Record one completed kernel call and refresh its context policy."""

        self._require_open()
        context = self.context(K, p, size)
        try:
            canonical = _METHOD_ALIASES[str(method)]
        except KeyError as exc:
            raise ValueError(f"unsupported tuning method: {method!r}") from exc
        elapsed_ns = float(elapsed_ns)
        if not isfinite(elapsed_ns) or elapsed_ns <= 0.0:
            raise ValueError("elapsed_ns must be positive and finite")

        with self._lock:
            bucket = self._bucket(context)
            if canonical not in bucket.arms:
                raise ValueError(
                    f"method {canonical!r} is not configured for this tuner"
                )
            arm = bucket.arms[canonical]
            per_row = elapsed_ns / max(context.batch_size, 1)
            if arm.ewma_ns_per_row is None:
                arm.ewma_ns_per_row = per_row
            else:
                arm.ewma_ns_per_row = (
                    (1.0 - self.ewma_alpha) * arm.ewma_ns_per_row
                    + self.ewma_alpha * per_row
                )
            arm.observations += 1
            self._refresh(bucket)

    def _refresh(self, bucket: _Bucket) -> None:
        if not all(arm.ewma_ns_per_row is not None for arm in bucket.arms.values()):
            return
        best = min(float(arm.ewma_ns_per_row) for arm in bucket.arms.values())
        for arm in bucket.arms.values():
            # Bradley-Terry-style speed share against the current best arm.
            # The best arm is centered at 0.5 rather than near the closed
            # endpoint, so a strong incumbent does not extinguish exploration.
            raw_score = best / (best + float(arm.ewma_ns_per_row))
            score = min(1.0 - self.score_floor, max(self.score_floor, raw_score))
            strength = min(float(arm.observations), self.max_strength)
            arm.score = score
            arm.stream.set_stats(strength * score, strength)

    def diagnostics(self) -> dict:
        """Return a JSON-friendly snapshot of learned contexts and arms."""

        self._require_open()
        with self._lock:
            contexts = []
            for context, bucket in sorted(
                self._buckets.items(), key=lambda item: repr(item[0])
            ):
                contexts.append(
                    {
                        "context": asdict(context),
                        "ready": all(
                            arm.observations > 0 for arm in bucket.arms.values()
                        ),
                        "methods": {
                            method: {
                                "observations": arm.observations,
                                "selections": arm.selections,
                                "ewma_ns_per_row": arm.ewma_ns_per_row,
                                "score": arm.score,
                            }
                            for method, arm in bucket.arms.items()
                        },
                    }
                )
            return {
                "methods": self.methods,
                "seed": self.seed,
                "ewma_alpha": self.ewma_alpha,
                "max_strength": self.max_strength,
                "score_floor": self.score_floor,
                "contexts": contexts,
            }

    def reset(self) -> None:
        """Discard all learned contexts while keeping the tuner usable."""

        self._require_open()
        with self._lock:
            for bucket in self._buckets.values():
                for arm in bucket.arms.values():
                    arm.stream.close()
            self._buckets.clear()

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            for bucket in self._buckets.values():
                for arm in bucket.arms.values():
                    arm.stream.close()
            self._buckets.clear()
            self._closed = True
