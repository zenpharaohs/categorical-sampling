"""Installed-wheel smoke check for cibuildwheel."""

from __future__ import annotations

import sys

import numpy as np

import categorical_samplers as cs


def main() -> int:
    if not cs.native_available():
        print("native backend is not available", file=sys.stderr)
        return 1

    draws = cs.binomial(5000, 0.002, size=20000, seed=101, method="auto")
    if draws.dtype != np.int64:
        print(f"unexpected dtype: {draws.dtype}", file=sys.stderr)
        return 1
    if draws.shape != (20000,):
        print(f"unexpected shape: {draws.shape}", file=sys.stderr)
        return 1
    mean = float(draws.mean())
    if abs(mean - 10.0) > 0.2:
        print(f"unexpected binomial mean: {mean}", file=sys.stderr)
        return 1

    cats = cs.categorical([0.2, 0.3, 0.5], size=10, seed=7)
    if cats.shape != (10,):
        print(f"unexpected categorical shape: {cats.shape}", file=sys.stderr)
        return 1

    multi = cs.multinomial(8, [0.2, 0.3, 0.5], size=10, seed=7)
    if multi.shape != (10, 3) or not np.all(multi.sum(axis=1) == 8):
        print("unexpected multinomial output", file=sys.stderr)
        return 1

    print(f"native_available=True binomial_mean={mean:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
