"""Small real-time filters for 3D pose streams.

The capture loop needs low latency, so this keeps the filtering deliberately
simple: an exponential smoother per joint coordinate with gap-aware resets. Raw
triangulated points remain available to the recorder for later offline work.
"""
from __future__ import annotations

import numpy as np


class PoseSmoother:
    def __init__(self, alpha: float = 0.55, max_gap_s: float = 0.20):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.max_gap_s = float(max_gap_s)
        self._last = None
        self._last_t = None

    def reset(self):
        self._last = None
        self._last_t = None

    def update(self, pts3d: np.ndarray, valid: np.ndarray, t: float):
        pts = np.asarray(pts3d, float)
        valid = np.asarray(valid, bool)
        if self._last is None or self._last_t is None or t - self._last_t > self.max_gap_s:
            self._last = pts.copy()
            self._last[~valid] = np.nan
            self._last_t = t
            return self._last.copy(), valid.copy()

        out = self._last.copy()
        for i in range(pts.shape[0]):
            if not valid[i] or not np.all(np.isfinite(pts[i])):
                continue
            if not np.all(np.isfinite(out[i])):
                out[i] = pts[i]
            else:
                out[i] = self.alpha * pts[i] + (1.0 - self.alpha) * out[i]
        out[~np.isfinite(out)] = np.nan
        self._last = out
        self._last_t = t
        return out.copy(), np.isfinite(out[:, 0])
