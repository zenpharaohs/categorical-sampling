"""Large-ish categorical sampler validation for local installs.

Run from the repo root with:

    PYTHONPATH=src python benchmarks/validate_install.py --n 1000000

If sibling packages ``hellinger-qualify`` and ``streaming-pit-validate`` are on
``PYTHONPATH`` or installed in the environment, their PIT diagnostics are
included in the report.
"""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

import categorical_samplers as cs


def _fmt_scale(value: float) -> str:
    if value >= 1e299:
        return "Inf"
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}k"
    return f"{value:.0f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200_000, help="number of categorical draws")
    parser.add_argument("--k", type=int, default=64, help="number of categories")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    weights = rng.lognormal(mean=0.0, sigma=1.0, size=args.k)
    p = weights / weights.sum()

    t0 = perf_counter()
    draws = cs.categorical(p, size=args.n, seed=args.seed + 1)
    draw_seconds = perf_counter() - t0

    t1 = perf_counter()
    report = cs.validate_categorical_draws(draws, p, rng=args.seed + 2)
    validation_seconds = perf_counter() - t1

    print(f"N={args.n:,} K={args.k:,}")
    print(f"draw_rate={args.n / draw_seconds:,.0f}/s validation_rate={args.n / validation_seconds:,.0f}/s")
    print(f"discrete_affinity={report['discrete_affinity']:.12f} discrete_H2={report['discrete_h2']:.3e}")

    legendre = report["legendre"]
    if legendre is None:
        print("legendre=not available (install streaming-pit-validate)")
    else:
        print(
            "legendre "
            f"H2={legendre.h2:.3e} SNR={legendre.snr:+.2f} "
            f"d_eff={legendre.d_eff} blocks={tuple(round(x, 2) for x in legendre.block_snr)}"
        )

    hq = report["hellinger"]
    if hq is None:
        print("hellinger=not available (install hellinger-qualify)")
    else:
        print(f"hellinger_smooth H2={hq.h2:.3e} Db={hq.db:.3e}")

    hq_cv = report["hellinger_cv"]
    if hq_cv is not None:
        scales = hq_cv.adjusted_scales
        print(
            "hellinger_cv "
            f"H2={hq_cv.adjusted_h2:.3e} "
            f"Beta=({hq_cv.alpha:.6g},{hq_cv.beta:.6g}) "
            f"no={_fmt_scale(scales.n_no_classifier_total_error_below_half)} "
            f"some={_fmt_scale(scales.n_some_classifier_total_error_below_half)}"
        )


if __name__ == "__main__":
    main()
