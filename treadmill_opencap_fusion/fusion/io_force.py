"""Loader for the Bertec treadmill force-plate .npz file.

The acquisition script (Treadmill_python/functions_HILO.py) stores:
  DATA  (14, N) : rows 0..11 = 12 analog force channels (raw volts),
                  row 12      = HILO control target (not a force),
                  row 13      = elapsed seconds since START (the sync clock).
  COMMS (3, N), EXO (4, N), TREAD (6, N) : auxiliary streams (often all zero).
  START scalar  : Unix epoch (s) at recording start.

Only the first ~N_valid frames are populated; the array is preallocated for a
5-minute trial, so we trim to the region where the time row is monotonic > 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt


@dataclass
class ForceData:
    """Trimmed, channel-labelled force-plate recording (still in volts)."""

    t: np.ndarray                 # (N,) elapsed seconds since START
    start_epoch: float            # Unix epoch of t[0] == 0
    channels: dict[str, np.ndarray]   # label -> (N,) volts
    fs: float                     # estimated mean sampling rate (Hz)
    n_samples: int

    @property
    def epoch(self) -> np.ndarray:
        """Absolute Unix-epoch timestamp of every sample."""
        return self.start_epoch + self.t


def load_force(npz_path: str, time_row: int, channel_order: list[str],
               filter_cfg: dict | None = None) -> ForceData:
    raw = np.load(npz_path, allow_pickle=True)
    DATA = np.asarray(raw["DATA"], dtype=float)
    start_epoch = float(raw["START"])

    t_all = DATA[time_row]
    # Valid frames: time strictly increasing and positive. Frame 0 has t==0 and
    # is legitimately the first sample, so include it explicitly.
    valid = np.where(t_all > 0)[0]
    if valid.size == 0:
        raise ValueError(f"No valid samples found in {npz_path}")
    lo, hi = 0, valid.max() + 1

    t = t_all[lo:hi].copy()
    # Guard against any stray zero rows inside the populated region.
    good = np.ones(t.shape, dtype=bool)
    good[1:] = np.diff(t) > 0
    t = t[good]

    channels = {}
    for i, label in enumerate(channel_order):
        channels[label] = DATA[i, lo:hi][good]

    fs = 1.0 / np.median(np.diff(t)) if t.size > 1 else float("nan")

    if filter_cfg and filter_cfg.get("enabled", False):
        b, a = butter(int(filter_cfg.get("order", 4)),
                      float(filter_cfg["cutoff_hz"]) / (0.5 * fs),
                      btype="low")
        for label in channels:
            channels[label] = filtfilt(b, a, channels[label])

    return ForceData(t=t, start_epoch=start_epoch, channels=channels,
                     fs=fs, n_samples=t.size)
