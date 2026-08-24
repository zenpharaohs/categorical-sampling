# Architecture Notes

This package is moving toward a dual-frontend architecture:

```text
MATLAB API -> MEX shim           -> shared native core
Python API -> CPython/NumPy shim -> shared native core
```

The shared native core is where the sampling algorithms, RNG discipline,
threading model, and validation-relevant behavior should live. Frontends should
stay as thin as practical: parse arguments, allocate language-native arrays,
adapt memory layout, translate errors, and expose idiomatic APIs.

## Why This Matters

The ziggurat exponential bug in the `wait2` binomial path was a useful warning.
The waiting-times method was mathematically sound, but the optimized exponential
generator had an implementation error. Because equivalent logic existed in both
MATLAB/MEX and Python-facing code, the fix had to be propagated carefully.

The long-term target is to fix algorithmic bugs once in the shared backend and
have MATLAB and Python inherit the same correction.

## Threading Is Part Of The Algorithm

The OpenMP batch paths are not simple compiler-level vectorization. They depend
on sampler-specific design:

- each output row has an independent deterministic RNG substream
- OpenMP scheduling must not affect sampled values
- there is no shared mutable RNG state
- rows write to independent output memory regions
- validation checks distributional correctness, not only memory safety

This is why the backend owns the RNG and threading discipline. A JIT or generic
automatic parallelizer should not be expected to infer this safely.

## Current Multinomial Kernels

The Python package now wraps shared C implementations for:

- `pivot` / sequential binomial cascade
- `smallk-cdf`
- `rep` / alias-thin histogram

OpenMP row-parallelism is enabled when available. This is essential for the
large-batch performance story and makes the Python results much closer to the
MATLAB/MEX results.

## Contracts And Triage

The `mcm/` directory should be treated as the contract layer between frontends
and backend. It should specify:

- parameter domains and dtype/shape behavior
- RNG determinism promises
- exactness and validation requirements
- performance regimes and triage context features
- frontend-specific packaging constraints

The opt-in `MultinomialTuner` implements contextual Thompson triage over these
three kernels using continuous-Bernoulli pseudo-posteriors. It warms each arm,
learns from exponentially weighted time per output row, and keeps the default
`method="auto"` path stateless. See
[Self-tuning multinomial dispatch](self_tuning_dispatch.md) for the policy and
its reproducibility boundary.

Adaptive triage must choose only among kernels that have passed validation for
the relevant context. Correctness is a qualification gate; triage is a
performance decision among qualified methods.
