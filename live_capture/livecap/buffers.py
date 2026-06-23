"""Thread-safe primitives shared between producer threads and the GUI.

* :class:`LatestSlot` -- holds only the most recent item (frame, pose, ...).
  Producers overwrite; consumers read the freshest value. This is the core
  latency-minimizing pattern: nothing queues up behind a slow consumer.
* :class:`RingBuffer` -- fixed-capacity circular buffer of timestamped numeric
  vectors, used for the scrolling live plots and for in-memory recording.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Stamped:
    """A value with its capture/arrival timestamp (epoch seconds)."""
    t: float
    value: Any


class LatestSlot:
    """Single-slot mailbox: keep only the newest item."""

    def __init__(self):
        self._lock = threading.Lock()
        self._item: Stamped | None = None
        self._seq = 0  # increments on every set; lets consumers detect new data

    def set(self, value: Any, t: float):
        with self._lock:
            self._item = Stamped(t=t, value=value)
            self._seq += 1

    def get(self) -> tuple[Stamped | None, int]:
        with self._lock:
            return self._item, self._seq


class RingBuffer:
    """Fixed-capacity circular buffer of (timestamp, vector) rows."""

    def __init__(self, capacity: int, width: int):
        self.capacity = int(capacity)
        self.width = int(width)
        self._t = np.full(self.capacity, np.nan)
        self._x = np.full((self.capacity, self.width), np.nan)
        self._head = 0          # next write index
        self._count = 0
        self._lock = threading.Lock()

    def append(self, t: float, vec):
        vec = np.asarray(vec, float).reshape(-1)
        with self._lock:
            self._t[self._head] = t
            self._x[self._head, :len(vec)] = vec
            self._head = (self._head + 1) % self.capacity
            self._count = min(self._count + 1, self.capacity)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (t, x) in chronological order (oldest -> newest)."""
        with self._lock:
            if self._count < self.capacity:
                idx = np.arange(0, self._count)
            else:
                idx = (self._head + np.arange(self.capacity)) % self.capacity
            return self._t[idx].copy(), self._x[idx].copy()

    def latest(self):
        with self._lock:
            if self._count == 0:
                return None, None
            i = (self._head - 1) % self.capacity
            return self._t[i], self._x[i].copy()
