# categorical-sampling

Categorical-family random samplers and validation utilities.

This repo is intentionally contract-first. The `mcm/` directory carries the
port contracts written from the MATLAB implementation, and `native/` carries
the C and MEX reference sources so the Python backend work can proceed without
reinventing the sampler design.

See [Architecture Notes](docs/architecture.md) for the shared-backend,
frontend-shim, RNG/threading, and triage rationale.

## Current Status

This is the initial Python repository scaffold with native backend paths wired
for binomial and multinomial sampling.

- `categorical_samplers` exposes a NumPy-backed API for smoke tests and contract
  development.
- `categorical_samplers.binomial` uses the native C backend when the extension
  has been built.
- `categorical_samplers.multinomial` uses native C multinomial kernels when the
  extension has been built: `pivot`, `smallk-cdf`, and `rep`.
- `categorical_samplers.validation` exposes direct discrete
  Hellinger/Bhattacharyya helpers plus randomized-PIT hooks for the shared
  `hellinger-qualify` and `streaming-pit-validate` packages.
- `native/core` contains the extracted C core starting point for binomial and
  multinomial kernels.
- `native/matlab_reference` contains the MATLAB MEX reference sources.
- Native multinomial batch kernels use OpenMP when available. Native
  categorical kernels, prebuilt multinomial state, and broader multinomial
  triage are the next implementation steps.

## Quick Start

From a fresh clone on macOS or Linux:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests
```

From a fresh clone in Windows `cmd`:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -e .
python -m unittest discover -s tests
```

The final `.` in `python -m pip install -e .` means "install this repository
from the current folder"; it is required.

Without installing, the smoke tests can also be run from the repo root:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

In that mode, native backend tests are skipped unless the extension has already
been built in place:

```bash
python setup.py build_ext --inplace
PYTHONPATH=src python -m unittest discover -s tests
```

## Build Wheels

Wheel builds are configured with `cibuildwheel` in `pyproject.toml`.

Local platform wheel:

```bash
python -m pip install build cibuildwheel
python -m cibuildwheel --platform auto
```

Linux builders can run:

```bash
python -m pip install build cibuildwheel
python -m cibuildwheel --platform linux
```

macOS and Windows builders use the same command with `--platform macos` or
`--platform windows`. Each wheel is tested after installation with:

```bash
python -m categorical_samplers._wheel_smoke
```

That smoke check confirms that the native extension imports, native binomial
and multinomial sampling work, and the categorical API still behaves as
expected.

## Benchmarks

Manual benchmarks live in `benchmarks/`. They are not CI gates because speed is
platform- and compiler-sensitive.

```bash
PYTHONPATH=src python benchmarks/bench_multinomial.py
```

On macOS, OpenMP builds require `libomp` (for example from Homebrew). Set
`CATEGORICAL_SAMPLERS_NO_OPENMP=1` to force a serial native build.

## API Sketch

```python
import categorical_samplers as cs

x = cs.binomial(100, 0.37, size=1000, seed=123)
y = cs.categorical([0.2, 0.3, 0.5], size=1000, seed=123)
z = cs.multinomial(25, [0.2, 0.3, 0.5], size=1000, seed=123)
```

The initial implementation delegates to NumPy. That keeps the public contract
testable while the native backend is completed. Binomial sampling already uses
the native extension when it is available:

```python
import categorical_samplers as cs

assert cs.native_available()
x = cs.binomial(100_000, 0.01, size=1_000_000, seed=123, method="wait2")
```

Multinomial methods currently include:

```python
H1 = cs.multinomial(10, p, size=1000, method="smallk-cdf")
H2 = cs.multinomial(10, p, size=1000, method="rep")
H3 = cs.multinomial(1000, p, size=1000, method="pivot")
```

`method="auto"` is still a simple native dispatch placeholder. Learned or
calibrated triage is planned.

## Validation

Sampler qualification is based on Hellinger affinity and randomized PIT
diagnostics. For iid samples, the affinity compounds as:

```text
A(P^N, Q^N) = A(P, Q)^N
```

The validation helpers translate one-sample affinity bounds into sample-size
qualification ranges for indistinguishability or detectability claims.

For categorical draws, the repo also computes randomized PIT samples
`u = F(j - 1) + V p_j`. Under the exact sampler these are uniform, so the same
PIT validators used by `continuous-bernoulli` can be reused here:

```bash
PYTHONPATH=src:../hellinger-qualify/src:../streaming-pit-validate/src \
  python benchmarks/validate_install.py --n 1000000
```

See [Validation](docs/validation.md) for the direct discrete and PIT-based
validation paths.
