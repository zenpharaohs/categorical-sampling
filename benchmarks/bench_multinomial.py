"""Manual multinomial benchmark against NumPy.

Run from the repository root after building/installing the native extension:

    PYTHONPATH=src python benchmarks/bench_multinomial.py
"""

from __future__ import annotations

import time

import numpy as np

import categorical_samplers as cs


def bench_case(name: str, K: int, p: np.ndarray, size: int) -> None:
    p = p / p.sum()
    cs.multinomial(K, p, size=100, seed=1, method="pivot")
    np.random.default_rng(1).multinomial(K, p, size=100)

    t0 = time.perf_counter()
    x = cs.multinomial(K, p, size=size, seed=123, method="pivot")
    t1 = time.perf_counter()

    rng = np.random.default_rng(123)
    t2 = time.perf_counter()
    y = rng.multinomial(K, p, size=size)
    t3 = time.perf_counter()

    native = t1 - t0
    numpy = t3 - t2
    speedup = numpy / native if native > 0 else float("inf")
    max_mean_err = float(np.max(np.abs(x.mean(axis=0) - K * p)))
    numpy_max_mean_err = float(np.max(np.abs(y.mean(axis=0) - K * p)))
    print(
        f"{name},{K},{p.size},{size},{native:.6f},{numpy:.6f},"
        f"{speedup:.2f},{max_mean_err:.6g},{numpy_max_mean_err:.6g}"
    )


def main() -> None:
    print(f"native_available={cs.native_available()}")
    print("case,K,d,size,native_s,numpy_s,speedup,max_mean_err,numpy_max_mean_err")

    bench_case("small_d_small_K", 8, np.array([0.2, 0.3, 0.5]), 500_000)
    bench_case("small_d_large_K", 1000, np.array([0.2, 0.3, 0.5]), 100_000)
    bench_case("medium_d", 25, np.linspace(1, 64, 64, dtype=np.float64), 100_000)
    bench_case("large_d_sparse_K", 10, np.linspace(1, 1000, 1000, dtype=np.float64), 20_000)


if __name__ == "__main__":
    main()
