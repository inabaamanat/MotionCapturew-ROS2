"""Master clock.

Everything on the single capture PC shares one wall-clock timebase so the two
camera streams and the force samples can be aligned by timestamp. We use
``time.time()`` (Unix epoch, seconds) to stay compatible with the existing
treadmill acquisition schema (functions_HILO.py stores ``START = time.time()``).
``perf_counter`` is exposed for high-resolution interval timing.
"""
from __future__ import annotations

import time

# Epoch captured at import; the session START is set explicitly by the engine.
_PROCESS_START = time.time()


def now() -> float:
    """Wall-clock timestamp (Unix epoch seconds)."""
    return time.time()


def mono() -> float:
    """Monotonic high-resolution counter (seconds); for measuring intervals."""
    return time.perf_counter()


def since(t0: float) -> float:
    """Seconds elapsed since a previous :func:`now` timestamp."""
    return time.time() - t0
