"""Native backend status.

The first checked-in Python surface is NumPy-backed. This module is deliberately
small so callers can distinguish the scaffold from the future optimized backend.
"""


def native_available() -> bool:
    """Return whether the optimized native backend is importable."""
    return False
