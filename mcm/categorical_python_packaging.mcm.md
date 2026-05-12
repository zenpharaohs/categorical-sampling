---
mcm_version: '0.1'
contract_type: module
id: categorical_python_packaging
title: 'Python packaging and distribution contract'
language: python
artifact_class: packaging
status: draft
iteration: 1
depends_on:
- categorical_python_backend
- categorical_python_api
- categorical_python_validation
consumed_by: []
output_file: python/pyproject.toml
---

# Module: `categorical_python_packaging`

## Purpose

Define a maintainable Python packaging path for the native sampler port.  The
first release should be easy to install from source and test locally; wheel
coverage can expand after the API and backend stabilize.

## Package Layout

Recommended layout:

```text
python/
  pyproject.toml
  README.md
  src/categorical_samplers/
    __init__.py
    _backend.*
    validation.py
    torch.py
  tests/
```

## Build System

Acceptable build systems:

- `setuptools` with a NumPy-aware extension build
- `scikit-build-core` if CMake materially improves portability
- Cython only if selected by the backend contract

The build must document how OpenMP is enabled or disabled on macOS, Linux, and
Windows.

## Dependencies

Required runtime dependencies should be minimal:

- NumPy

Optional dependencies:

- SciPy for validation helpers if needed
- PyTorch for torch convenience wrappers
- pytest for tests

The core sampler API must not require PyTorch.

## Wheel and Source Distribution Contract

- Source installs must work from a clean checkout.
- Wheels should include native extension binaries when practical.
- Unsupported platforms must fail with clear build messages.
- The package must expose a version string.

## Test Commands

The Python README must include commands equivalent to:

```bash
python -m pip install -e .[test]
python -m pytest
```

If validation tests are slower than smoke tests, provide markers such as:

```bash
python -m pytest -m "not slow"
python -m pytest -m validation
```

## Acceptance Evidence

- Clean editable install succeeds.
- Import smoke test succeeds.
- Full pytest suite succeeds.
- Source distribution builds.
- At least one platform build log documents OpenMP behavior.

