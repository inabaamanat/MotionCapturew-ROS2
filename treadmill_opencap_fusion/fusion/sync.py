"""Synchronize the force-PC clock with the OpenCap clock.

The two systems run on independent clocks, so absolute timestamps cannot be
trusted. We align them on a *shared physical event*: foot contact.

For each foot we build two continuous signals on a common grid and find the lag
that maximizes their cross-correlation:

  * Force  : the plate vertical GRF (Fz) -- high during that foot's stance.
  * Kinematics : the negative vertical height of the foot markers
                 (-height) -- also high during stance (foot is on the belt).

Both signals are band-pass filtered to isolate the gait oscillation. The Left
and Right correlations are summed so the search has a single unambiguous optimum
(L and R stance are out of phase, which breaks the one-stride cycle ambiguity
that a single foot would suffer from).

Result convention:
    force_time = opencap_time + offset_s
A heel-strike residual against the (clean) force events is reported as a
cross-check and surfaced in the QC report.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

VERTICAL_AXIS = 1  # Y is up in OpenCap world frame


@dataclass
class SyncResult:
    offset_s: float          # force_time = opencap_time + offset_s
    correlation: float       # mean per-foot peak correlation [-1, 1]
    corr_left: float
    corr_right: float
    method: str
    heelstrike_rms_s: float
    heelstrike_matched: int
    diagnostics: dict


def _uniform_grid(t, x, fs, fill=0.0):
    n = int(np.floor((t[-1] - t[0]) * fs)) + 1
    grid = t[0] + np.arange(n) / fs
    f = interp1d(t, x, bounds_error=False, fill_value=fill)
    return grid, f(grid)


def _foot_height(markers, side: str) -> np.ndarray:
    prefix = "r_" if side == "R" else "l_"
    cands = [f"{prefix}5meta", f"{prefix}toe", f"{prefix}ankle",
             f"{prefix}calc", f"{prefix}heel"]
    cols = [markers.data[c][:, VERTICAL_AXIS] for c in cands if c in markers.data]
    if not cols:
        raise KeyError(f"no foot markers for side {side}")
    return np.nanmean(np.vstack(cols), axis=0)


def _corr_at(short, long, lag):
    """Pearson r of `short` placed at integer sample `lag` within `long`."""
    n = min(len(short), len(long) - lag)
    if lag < 0 or n < int(0.6 * len(short)):
        return -np.inf
    a = short[:n] - np.mean(short[:n])
    b = long[lag:lag + n] - np.mean(long[lag:lag + n])
    d = np.std(a) * np.std(b) * n
    return float(np.sum(a * b) / d) if d > 0 else -np.inf


def _parabolic(y0, y1, y2):
    denom = (y0 - 2 * y1 + y2)
    return 0.5 * (y0 - y2) / denom if denom != 0 else 0.0


def synchronize(force_t, cforce, kin, markers, force_gait, sync_cfg) -> SyncResult:
    method = sync_cfg.get("method", "contact_xcorr")
    fs = float(sync_cfg.get("resample_hz", 100))
    max_lag_s = float(sync_cfg.get("max_lag_s", 30.0))

    if method == "manual":
        off = float(sync_cfg["manual_offset_s"])
        rms, nm = _heelstrike_residual(force_gait, kin, markers, off, fs)
        return SyncResult(off, float("nan"), float("nan"), float("nan"),
                          "manual", rms, nm, {})

    b, a = butter(2, [0.5 / (fs / 2), 6.0 / (fs / 2)], btype="band")

    # Per-foot band-passed signals on uniform grids.
    sig = {}
    for side, plate in (("R", cforce.right), ("L", cforce.left)):
        _, fz = _uniform_grid(force_t, plate.Fz, fs)
        kg, h = _uniform_grid(markers.time, _foot_height(markers, side), fs)
        sig[side] = (filtfilt(b, a, h) * -1.0, filtfilt(b, a, fz))

    # Physical constraint: the OpenCap walking window must fall inside the force
    # *loaded* bout (the subject cannot be walking on video while the plates read
    # zero). This rejects spurious correlation peaks over the dead-time at the
    # start of the force recording.
    bw = (cforce.left.Fz + cforce.right.Fz)
    _, total_grid = _uniform_grid(force_t, bw, fs)
    active = np.where(total_grid > 0.10 * np.nanmax(total_grid))[0]
    kin_dur_samp = len(sig["R"][0])
    margin = int(0.5 * fs)
    if active.size:
        lag_min = max(0, active[0] - margin)
        lag_max = min(int(max_lag_s * fs), active[-1] - kin_dur_samp + margin)
    else:
        lag_min, lag_max = 0, int(max_lag_s * fs)

    longN = len(sig["R"][1])
    lag_max = min(lag_max, longN - int(0.6 * kin_dur_samp))
    if lag_max <= lag_min:                      # fall back to full search
        lag_min, lag_max = 0, longN - int(0.6 * kin_dur_samp)
    lags = np.arange(lag_min, lag_max + 1)

    rR = np.array([_corr_at(sig["R"][0], sig["R"][1], k) for k in lags])
    rL = np.array([_corr_at(sig["L"][0], sig["L"][1], k) for k in lags])
    combined = rR + rL
    ki = int(np.argmax(combined))
    frac = _parabolic(combined[ki - 1], combined[ki], combined[ki + 1]) \
        if 0 < ki < len(combined) - 1 else 0.0
    offset_s = (lags[ki] + frac) / fs

    corr_R, corr_L = float(rR[ki]), float(rL[ki])
    rms, nm = _heelstrike_residual(force_gait, kin, markers, offset_s, fs)

    diagnostics = {
        "lags_s": lags / fs,
        "corr_R": rR, "corr_L": rL, "corr_combined": combined,
        "fs": fs,
    }
    return SyncResult(offset_s=offset_s, correlation=0.5 * (corr_R + corr_L),
                      corr_left=corr_L, corr_right=corr_R, method=method,
                      heelstrike_rms_s=rms, heelstrike_matched=nm,
                      diagnostics=diagnostics)


def _kin_heel_strikes(markers, side, fs):
    """Rising edges of kinematic foot contact (height drops into its low band)."""
    h = _foot_height(markers, side)
    g, hg = _uniform_grid(markers.time, h, fs)
    thr = np.nanmin(hg) + 0.35 * (np.nanmax(hg) - np.nanmin(hg))
    stance = hg < thr
    rise = np.where((~stance[:-1]) & stance[1:])[0] + 1
    return g[rise]


def _heelstrike_residual(force_gait, kin, markers, offset_s, fs):
    """RMS time error of matched heel strikes (force vs kinematics, per foot)."""
    errs = []
    for side, fev in (("R", force_gait.right), ("L", force_gait.left)):
        khs = _kin_heel_strikes(markers, side, fs)
        fhs = fev.heel_strikes
        if fhs.size == 0:
            continue
        for h in khs:
            d = np.min(np.abs(fhs - (h + offset_s)))
            if d < 0.25:  # within a quarter second counts as a match
                errs.append(d)
    if not errs:
        return float("nan"), 0
    return float(np.sqrt(np.mean(np.square(errs)))), len(errs)
