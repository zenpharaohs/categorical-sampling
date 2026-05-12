---
mcm_version: '0.1'
contract_type: module
id: categorical_python_validation
title: 'Validation and sampler qualification contract for Python port'
language: python_matlab
artifact_class: validation
status: draft
iteration: 1
depends_on:
- categorical_python_api
- categorical_python_backend
consumed_by:
- categorical_python_packaging
output_file: python/tests/test_validation.py
---

# Module: `categorical_python_validation`

## Purpose

Port the MATLAB validation philosophy to Python: samplers are qualified by
Bhattacharyya/Hellinger affinity intervals, tail accounting, recursive
multinomial decomposition, and confidence-bound reporting.

## Required Validation Helpers

The Python package should expose or internally test equivalents of:

- `affinity_counts_ci`
- `bhattacharyya_iid_bounds`
- `binom_hellinger_affinity`
- `multinom_exact_affinity`
- `multinom_projection_affinity`
- `multinom_recursive_affinity`

Names may be Pythonic, but the mathematical contract must remain recognizable.

## Qualification Semantics

For exact law `P`, sampler law `Q`, and one-draw affinity `rho = A(P,Q)`:

```text
rho_N = rho^N
1 - rho_N <= TVD_N <= sqrt(1 - rho_N^2)
alpha + beta = 1 - TVD_N
```

Reports should communicate:

- affinity interval `[rho_lo, rho_hi]`
- Hellinger-squared interval
- TVD interval at user-selected sample size `N`
- trust/detection sample-size ranges for a target TVD
- confidence level or `alpha` used for empirical intervals

## Binomial Validation

Binomial validation must:

- compute exact bulk probabilities stably
- account for ignored tail mass
- include empirical tail mass
- support conservative confidence intervals for empirical probabilities
- handle large `n` without enumerating the entire support

## Multinomial Validation

Multinomial validation must include:

- exact finite-support enumeration for small `(K,d)`
- random or specified projection checks reduced to binomial laws
- recursive conditional split checks
- exact leaf checks for small recursive leaves
- clear reporting that recursive/projection certificates validate a chosen
  path, not every possible projection

## Baselines

Python validation may use NumPy samplers as distributional baselines, but the
exact law is the authority.  Baseline agreement is not sufficient without exact
law checks.

## Acceptance Evidence

- Unit tests for every validation helper.
- Validation tests against NumPy-generated exact-like samples.
- Validation tests against the package's native samplers.
- At least one recursive multinomial test with exact leaf checks.
- CI tests must verify that plug-in estimates lie within reported confidence
  intervals where expected.

