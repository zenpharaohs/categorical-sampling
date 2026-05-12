"""Native backend dispatch."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from . import _native
except ImportError:  # pragma: no cover - exercised when extension is absent
    _native = None


def native_available() -> bool:
    """Return whether the optimized native backend is importable."""
    return _native is not None and bool(_native.native_available())


def binomial(n: int, p: float, size: int, seed: Optional[int], method: str) -> np.ndarray:
    """Draw binomial variates using the native backend."""
    if _native is None:
        raise RuntimeError("native backend is not available")
    return _native.binomial(int(n), float(p), int(size), seed, str(method))
