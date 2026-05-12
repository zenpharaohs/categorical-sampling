---
mcm_version: '0.1'
contract_type: module
id: categorical_python_api
title: 'Public Python API for categorical, binomial, and multinomial samplers'
language: python
artifact_class: public_api
status: draft
iteration: 1
depends_on:
- categorical_python_backend
consumed_by:
- categorical_python_validation
- categorical_python_packaging
- categorical_python_torch
output_file: python/src/categorical_samplers/__init__.py
---

# Module: `categorical_python_api`

## Purpose

Provide a NumPy-first public API for exact high-performance sampling while
preserving the MATLAB package's core semantics: method selection, deterministic
seeding, stream amortization, and validation hooks.

## Required Public Surface

Recommended top-level functions:

- `binomial(n, p, size=1, seed=None, method="auto")`
- `categorical(p, size=1, seed=None, method="auto")`
- `multinomial(K, p, size=1, seed=None, method="auto")`

Recommended stream classes:

- `BinomStreamSampler(n, p, buffer_size=10000, seed=None, method="auto")`
- `MultinomStreamSampler(K, p, buffer_size=10000, seed=None, method="auto")`

Stream objects must support:

- `draw(n)`
- `stats()`
- `close()`
- context-manager usage

Multinomial stream objects may support `cats(n)` only when `K == 1`.

## Shape and Dtype Contract

- `binomial(..., size=m)` returns shape `(m,)`.
- `categorical(..., size=m)` returns shape `(m,)`.
- `multinomial(..., size=m)` returns shape `(m, d)`.
- `size=None` behavior must be explicit. Prefer stable array return shapes for
  the first release.
- Probability inputs are accepted as one-dimensional NumPy-compatible arrays.
- Output dtype choices must be documented and tested.

## Indexing Contract

Python categorical outputs should be zero-based by default, matching NumPy and
PyTorch conventions.  If one-based output is offered for MATLAB parity, it must
be an explicit option such as `index_base=1`.

## Method Names

Public method strings should be stable and explicit:

- binomial: `"auto"`, `"btrd"`, `"wait2"`, `"dev"`
- multinomial: `"auto"`, `"pivot"`, `"smallk-cdf"`, `"rep"`, `"numpy"`
- categorical: `"auto"`, `"alias"`, `"cdf"`, `"numpy"`

Benchmark labels should avoid ambiguous `"direct"` unless qualified as
`"numpy-direct"`.

## Seed Contract

- `seed=None` may use entropy.
- Explicit seeds must be deterministic.
- Repeated calls with the same seed and parameters must repeat.
- Stream classes must diversify refill seeds deterministically.

## Triage Contract

The Python `method="auto"` planner should initially port the MATLAB threshold
logic and calibration defaults.  Host-specific calibration may be added after
the base port but must not block a usable first release.

## Acceptance Evidence

- API smoke tests cover all public functions and stream classes.
- Shape/dtype tests cover scalar, vector, and zero-size draws.
- Determinism tests cover functions and streams.
- Method-forcing tests cover every supported route.
- NumPy baseline tests cover distributional agreement, not bitwise equality.

