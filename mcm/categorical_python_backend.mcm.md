---
mcm_version: '0.1'
contract_type: module
id: categorical_python_backend
title: 'Native backend contract for Python categorical samplers'
language: c_python
artifact_class: native_backend
status: draft
iteration: 1
depends_on: []
consumed_by:
- categorical_python_api
- categorical_python_validation
- categorical_python_packaging
output_file: python/src/categorical_samplers/_backend.c
---

# Module: `categorical_python_backend`

## Purpose

Expose the validated C kernels to Python without changing their statistical
semantics.  The Python backend may reorganize entry points, memory ownership,
and error handling, but it must not silently change sampler laws, seed
discipline, or method-selection behavior.

## Kernel Scope

The first backend must expose:

- binomial fresh draw kernels: `btrd`, `wait2`, `dev`
- binomial prebuilt BTRD state build/draw
- categorical alias/sample helpers where useful
- multinomial histogram kernels: `pivot`, `smallK-cdf`, `rep` or their native
  equivalents
- OpenMP thread control if supported on the platform

The backend may omit MATLAB-only helper paths, but omitted behavior must be
called out in Python release notes.

## Binding Technology

Preferred options:

- CPython C extension using the NumPy C API
- Cython, if it materially reduces binding complexity

Avoid introducing C++ solely for binding convenience.  A pybind11 layer is
allowed only if the implementation remains thin and does not convert the
project into a C++ sampler library.

## Memory Contract

- Output arrays must be NumPy-owned arrays with stable dtype and shape.
- No pointer returned to Python may outlive its owning capsule/object.
- Prebuilt sampler states must free native memory deterministically when the
  Python object is closed or garbage-collected.
- Native functions must fail with Python exceptions, not process aborts, for
  ordinary invalid inputs.

## Dtype and Shape Contract

- Binomial draws return `np.uint32` unless the parameter range requires a wider
  dtype and the API explicitly documents that upgrade.
- Multinomial histograms return signed or unsigned 32-bit integer arrays,
  consistently documented.
- Multinomial histogram output shape is `(size, d)`.
- Categorical draws return integer category indices using Python's documented
  indexing convention; the API contract decides zero- vs one-based indexing.

## Seed Contract

- Explicit seeds are Python integers converted to unsigned 64-bit values.
- Same seed, method, parameters, backend version, and platform should be
  reproducible unless release notes explicitly state otherwise.
- Parallel execution must not make draws schedule-dependent.
- Stream refills must use deterministic seed mixing equivalent in spirit to
  the MATLAB stream samplers.

## Error Handling

The backend must reject:

- negative counts or sizes
- nonfinite probabilities
- all-zero probability vectors
- probabilities outside `[0, 1]` for scalar distributions
- unsupported sizes that would overflow native counters or output dimensions

## Acceptance Evidence

- Native smoke tests cover every exported backend route.
- Edge-case tests cover degenerate probabilities and zero-size draws.
- Determinism tests compare repeated seeded calls.
- Row-sum tests cover multinomial outputs.
- If OpenMP is enabled, tests compare deterministic output across thread
  counts where feasible.

