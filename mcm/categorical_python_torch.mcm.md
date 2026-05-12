---
mcm_version: '0.1'
contract_type: module
id: categorical_python_torch
title: 'Optional PyTorch integration contract'
language: python
artifact_class: interop
status: draft
iteration: 1
depends_on:
- categorical_python_api
consumed_by:
- categorical_python_packaging
output_file: python/src/categorical_samplers/torch.py
---

# Module: `categorical_python_torch`

## Purpose

Make the Python sampler convenient for PyTorch users without making PyTorch a
required dependency of the core package.

## Scope

The first PyTorch integration should be a thin convenience layer over the
NumPy-first API.  It should not introduce separate sampler laws, GPU kernels,
or autograd behavior.

Recommended functions:

- `torch_binomial(n, p, size=1, seed=None, method="auto", device=None)`
- `torch_categorical(p, size=1, seed=None, method="auto", device=None)`
- `torch_multinomial(K, p, size=1, seed=None, method="auto", device=None)`

Stream wrappers may be added later if needed.

## Tensor Contract

- CPU tensor inputs may be accepted by converting to NumPy safely.
- Outputs may be returned as PyTorch tensors.
- `device=None` returns CPU tensors.
- Non-CPU devices may be supported by generating on CPU and transferring, but
  the documentation must state that sampling itself is CPU-side.
- Returned tensors are samples, not differentiable operations.

## Dependency Contract

- Importing `categorical_samplers` must not import PyTorch.
- Importing `categorical_samplers.torch` may require PyTorch.
- If PyTorch is absent, errors should be clear and local to torch wrappers.

## Acceptance Evidence

- Tests skip cleanly when PyTorch is unavailable.
- CPU tensor input tests cover probability vectors.
- Output dtype and shape tests cover all wrappers.
- Determinism tests compare repeated seeded calls.

