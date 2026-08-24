"""Exercise self-tuning multinomial dispatch against the native kernels.

Run from the repository root with ``cb-sampler`` installed::

    PYTHONPATH=src python benchmarks/bench_adaptive_multinomial.py

The benchmark is diagnostic, not a CI gate: wall-clock results depend on the
host, compiler, thread runtime, and current machine load.
"""

from __future__ import annotations

import argparse
from collections import Counter
from time import perf_counter_ns

import numpy as np

import categorical_samplers as cs


METHODS = ("pivot", "smallk-cdf", "rep")


def fixed_timings(K: int, p: np.ndarray, size: int, repeats: int) -> dict[str, float]:
    timings: dict[str, float] = {}
    for method_index, method in enumerate(METHODS):
        cs.multinomial(K, p, size=size, seed=method_index, method=method)
        started = perf_counter_ns()
        for repeat in range(repeats):
            cs.multinomial(
                K,
                p,
                size=size,
                seed=10_000 * method_index + repeat,
                method=method,
            )
        elapsed = perf_counter_ns() - started
        timings[method] = elapsed / (repeats * max(size, 1))
    return timings


def bench_case(
    name: str,
    K: int,
    p: np.ndarray,
    size: int,
    rounds: int,
    repeats: int,
) -> None:
    p = p / p.sum()
    baseline = fixed_timings(K, p, size, repeats)
    choices = []
    with cs.MultinomialTuner(seed=137) as tuner:
        for round_index in range(rounds):
            draws = cs.multinomial(
                K,
                p,
                size=size,
                seed=100_000 + round_index,
                tuner=tuner,
            )
            if not np.all(draws.sum(axis=1) == K):
                raise AssertionError(f"{name}: invalid multinomial row sum")
        report = tuner.diagnostics()["contexts"][0]
        for method, state in report["methods"].items():
            choices.extend([method] * state["selections"])

    oracle = min(baseline, key=baseline.__getitem__)
    learned = max(Counter(choices), key=Counter(choices).__getitem__)
    times = ",".join(f"{method}:{baseline[method]:.1f}" for method in METHODS)
    counts = ",".join(
        f"{method}:{report['methods'][method]['selections']}" for method in METHODS
    )
    ewmas = ",".join(
        f"{method}:{report['methods'][method]['ewma_ns_per_row']:.1f}"
        for method in METHODS
    )
    print(
        f"{name};K={K};d={p.size};size={size};oracle={oracle};learned={learned};"
        f"fixed_ns_per_row={times};selections={counts};ewma_ns_per_row={ewmas}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.rounds < len(METHODS):
        parser.error(f"--rounds must be at least {len(METHODS)}")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if not cs.native_available():
        raise SystemExit("native backend is required")

    cases = (
        ("small_d_small_K", 8, np.array([0.2, 0.3, 0.5]), 100_000),
        ("small_d_large_K", 1000, np.array([0.2, 0.3, 0.5]), 50_000),
        ("medium_d", 25, np.linspace(1, 64, 64, dtype=np.float64), 50_000),
        ("large_d_sparse_K", 10, np.linspace(1, 1000, 1000, dtype=np.float64), 10_000),
    )
    for case in cases:
        bench_case(*case, rounds=args.rounds, repeats=args.repeats)


if __name__ == "__main__":
    main()
