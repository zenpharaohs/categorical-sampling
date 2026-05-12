# categorical-samplers

Python port of the MATLAB/MEX categorical sampler package.

This repo is intentionally contract-first. The `mcm/` directory carries the
port contracts written from the MATLAB implementation, and `native/` carries
the C and MEX reference sources so the Python backend work can proceed without
reinventing the sampler design.

## Current Status

This is the initial Python repository scaffold.

- `categorical_samplers` exposes a NumPy-backed API for smoke tests and contract
  development.
- `categorical_samplers.validation` exposes first Hellinger/Bhattacharyya
  helpers for sampler qualification.
- `native/core` contains the extracted C core starting point.
- `native/matlab_reference` contains the MATLAB MEX reference sources.
- The optimized native Python extension is the next implementation step.

## Quick Start

From a fresh clone:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .[test]
python -m pytest
```

Without installing, the smoke tests can also be run from the repo root:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## API Sketch

```python
import categorical_samplers as cs

x = cs.binomial(100, 0.37, size=1000, seed=123)
y = cs.categorical([0.2, 0.3, 0.5], size=1000, seed=123)
z = cs.multinomial(25, [0.2, 0.3, 0.5], size=1000, seed=123)
```

The initial implementation delegates to NumPy. That keeps the public contract
testable while the native backend is built.

## Validation

Sampler qualification is based on Hellinger affinity. For iid samples, the
affinity compounds as:

```text
A(P^N, Q^N) = A(P, Q)^N
```

The validation helpers translate one-sample affinity bounds into sample-size
qualification ranges for indistinguishability or detectability claims.
