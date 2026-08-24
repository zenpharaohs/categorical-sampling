# Self-tuning multinomial dispatch

The native multinomial kernels have complementary performance regimes. A
single static threshold is brittle across batch sizes, probability shapes,
thread counts, and host machines. `MultinomialTuner` therefore treats each
qualified kernel as an arm in a contextual Thompson sampler.

## Semantics

Self-tuning is explicit. Calls without a tuner retain the deterministic,
stateless behavior of `method="auto"`:

```python
import categorical_samplers as cs

with cs.MultinomialTuner(seed=137) as tuner:
    for seed in range(20):
        x = cs.multinomial(25, [0.2, 0.3, 0.5], size=100_000,
                           method="auto", tuner=tuner, seed=seed)
    report = tuner.diagnostics()
```

The tuner requires `cb-sampler` from the public `continuous-bernoulli`
repository. Until it is available from a package index, install it directly
from GitHub:

```bash
python -m pip install \
  "cb-sampler @ git+https://github.com/zenpharaohs/continuous-bernoulli.git#subdirectory=python"
```

Each permutation-invariant context records `K`, category count, batch size,
nonzero category count, and coarse entropy/concentration bins. Every configured
kernel receives one warmup call. Later calls draw once from every arm's
continuous-Bernoulli pseudo-posterior and run the method with the largest draw.
`MultinomialStream` also accepts a `tuner`; its fixed-size buffer refills then
form a natural repeated workload for learning.

The observed score is a Bradley–Terry-style relative-speed share. If `b` is
the best exponentially weighted time per output row and `t` is the arm's time,
the score is `b / (b + t)`, clipped away from the closed-family endpoints. The
best arm therefore sits at `0.5` instead of being driven toward a point mass at
one, preserving exploration. Posterior state is replaced with

```text
(chi, nu) = (strength * score, strength)
```

where `strength` is capped. This sample-and-hold update tracks current relative
throughput and preserves exploration if host load or threading behavior changes.

## Reproducibility boundary

Explicit method names and `method="auto"` without a tuner remain reproducible
for fixed inputs and seeds. A supplied tuner is intentionally stateful and
learns from wall-clock timings, so its future dispatch choices are not promised
to repeat across machines or runs. Every selected kernel remains an exact
sampler; tuning changes performance policy, not the target distribution.

Only kernels that have independently passed correctness qualification belong in
the tuner's candidate set. The initial set is `pivot`, `smallk-cdf`, and `rep`.
