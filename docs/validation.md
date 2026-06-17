# Validation

The categorical sampler has two complementary validation paths.

## Direct Discrete Affinity

For a categorical distribution with exact probabilities `p_j` and observed
counts `q_j`, `categorical_samplers.validation.empirical_affinity` estimates

```text
A(p, q) = sum_j sqrt(p_j q_j).
```

This is the native discrete check. It is cheap and does not need any auxiliary
randomness.

## Randomized PIT

To reuse the shared PIT validators, each categorical draw is mapped into the
unit interval by randomized probability integral transform:

```text
u = F(j - 1) + V p_j,    V ~ U(0,1).
```

If the sampler is exactly drawing from `p`, then `u` is uniform on `(0,1)`.
This lets the categorical repo use the same two validation packages as the
continuous-Bernoulli sampler:

- `streaming-pit-validate` for streaming shifted-Legendre diagnostics.
- `hellinger-qualify` for smoothed-spacing Hellinger certification and
  classifier sample-scale estimates.

The integration is optional. A normal install of `categorical-sampling` still
works by itself; if the sibling packages are installed or visible on
`PYTHONPATH`, `validate_categorical_draws` includes their reports.

For local development with sibling clones:

```bash
PYTHONPATH=src:../hellinger-qualify/src:../streaming-pit-validate/src \
  python benchmarks/validate_install.py --n 1000000
```

The report includes direct discrete affinity, randomized-PIT Legendre output,
smoothed Hellinger, and fitted-Beta control-variate Hellinger when available.
