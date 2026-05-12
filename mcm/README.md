---
mcm_version: '0.1'
contract_type: bundle
id: categorical_python_port_contracts
title: 'Categorical sampler Python port contracts'
language: c_python_matlab
artifact_class: porting_contract_bundle
status: draft
iteration: 1
depends_on: []
consumed_by:
- categorical_python_backend
- categorical_python_api
- categorical_python_validation
- categorical_python_packaging
- categorical_python_torch
---

# Categorical Sampler Python Port MCM Bundle

These contracts describe the Python version as a clean port of the validated
MATLAB/MEX/C sampler package.  The intent is to preserve the sampling kernels,
seed discipline, validation theory, and performance lessons already developed
here, while exposing a Pythonic NumPy-first API.

The contracts are deliberately small and composable:

- `categorical_python_backend.mcm.md` defines the native C extension boundary.
- `categorical_python_api.mcm.md` defines the public Python surface.
- `categorical_python_validation.mcm.md` defines qualification evidence.
- `categorical_python_packaging.mcm.md` defines build and distribution
  expectations.
- `categorical_python_torch.mcm.md` defines optional PyTorch interop.

## Porting Principle

Do not re-invent the sampler algorithms during the first Python port.  The C
kernels in this repository are the behavioral authority.  Python should first
wrap, test, validate, and document those kernels.  New Python-native or GPU
implementations may be added later only as separately validated artifacts.

## Completion Bar

The Python port is release-ready only when:

- NumPy API tests pass for shape, dtype, determinism, and edge cases.
- Native backend tests pass under sanitizer/debug builds where available.
- Hellinger/Bhattacharyya qualification tests pass with confidence intervals.
- Performance tests compare against explicit NumPy baselines.
- Packaging tests pass from a clean environment.

