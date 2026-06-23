"""Gait-event detection and spatiotemporal metrics.

Two independent event sources:
  * Force plates  -> stance is defined by vertical GRF above a body-weight
    fraction. Heel strike = rising edge, toe off = falling edge.
  * Kinematics    -> per-foot vertical marker signal, used to cross-check the
    synchronization (heel strike ~ local minimum of foot height).

Marker frame convention (verified for this OpenCap export): Y is vertical.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

VERTICAL_AXIS = 1  # Y is up in OpenCap world frame


@dataclass
class FootEvents:
    side: str
    stance: np.ndarray          # bool (N,)
    heel_strikes: np.ndarray    # times (s)
    toe_offs: np.ndarray        # times (s)


@dataclass
class GaitResult:
    left: FootEvents
    right: FootEvents
    metrics: dict = field(default_factory=dict)


def _rising_falling(mask: np.ndarray, t: np.ndarray, min_interval: float):
    """Debounced rising (heel strike) and falling (toe off) edge times."""
    m = mask.astype(int)
    rise = np.where(np.diff(m) == 1)[0] + 1
    fall = np.where(np.diff(m) == -1)[0] + 1

    def debounce(idx):
        if idx.size == 0:
            return idx
        keep = [idx[0]]
        for i in idx[1:]:
            if t[i] - t[keep[-1]] >= min_interval:
                keep.append(i)
        return np.array(keep)

    return t[debounce(rise)], t[debounce(fall)]


def detect_force_events(cforce, t, body_mass_kg, gait_cfg) -> GaitResult:
    """Stance/heel-strike/toe-off per foot from calibrated vertical GRF."""
    bw = body_mass_kg * 9.80665
    thr = float(gait_cfg.get("stance_threshold_bw", 0.05)) * bw
    min_int = float(gait_cfg.get("min_step_interval_s", 0.30))

    out = {}
    for side, plate in (("L", cforce.left), ("R", cforce.right)):
        stance = plate.Fz > thr
        hs, to = _rising_falling(stance, t, min_int)
        out[side] = FootEvents(side=side, stance=stance, heel_strikes=hs, toe_offs=to)

    res = GaitResult(left=out["L"], right=out["R"])
    res.metrics = _spatiotemporal(out["L"], out["R"], t)
    return res


def _spatiotemporal(left: FootEvents, right: FootEvents, t: np.ndarray) -> dict:
    """Stride/step/stance/swing times, cadence, duty factor, symmetry."""
    def strides(hs):
        return np.diff(hs) if hs.size > 1 else np.array([])

    def stance_swing(ev: FootEvents):
        # Pair each heel strike with the next toe off and the following heel strike.
        st, sw = [], []
        for i, hs in enumerate(ev.heel_strikes[:-1]):
            nxt_hs = ev.heel_strikes[i + 1]
            to = ev.toe_offs[(ev.toe_offs > hs) & (ev.toe_offs < nxt_hs)]
            if to.size:
                st.append(to[0] - hs)
                sw.append(nxt_hs - to[0])
        return np.array(st), np.array(sw)

    l_stride, r_stride = strides(left.heel_strikes), strides(right.heel_strikes)
    l_st, l_sw = stance_swing(left)
    r_st, r_sw = stance_swing(right)

    all_hs = np.sort(np.concatenate([left.heel_strikes, right.heel_strikes]))
    step_times = np.diff(all_hs) if all_hs.size > 1 else np.array([])
    # Walking-bout duration = span of heel strikes (excludes treadmill-mount
    # dead time), so cadence is not diluted by the unloaded period.
    walk_duration = (all_hs[-1] - all_hs[0]) if all_hs.size > 1 else (t[-1] - t[0])

    def m(x):
        return float(np.mean(x)) if len(x) else float("nan")

    def sym(a, b):
        a, b = m(a), m(b)
        if not (np.isfinite(a) and np.isfinite(b)) or (a + b) == 0:
            return float("nan")
        return float(2 * abs(a - b) / (a + b) * 100)  # symmetry index (%)

    return {
        "n_steps": int(all_hs.size),
        "n_strides_L": int(l_stride.size),
        "n_strides_R": int(r_stride.size),
        "walk_duration_s": float(walk_duration),
        "cadence_steps_per_min": float(60.0 / m(step_times)) if step_times.size else float("nan"),
        "stride_time_L_s": m(l_stride),
        "stride_time_R_s": m(r_stride),
        "step_time_s": m(step_times),
        "stance_time_L_s": m(l_st),
        "stance_time_R_s": m(r_st),
        "swing_time_L_s": m(l_sw),
        "swing_time_R_s": m(r_sw),
        "duty_factor_L": m(l_st) / m(l_stride) if l_stride.size else float("nan"),
        "duty_factor_R": m(r_st) / m(r_stride) if r_stride.size else float("nan"),
        "stride_time_symmetry_pct": sym(l_stride, r_stride),
        "stance_time_symmetry_pct": sym(l_st, r_st),
    }


def kinematic_foot_signal(markers, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-foot vertical contact proxy and heel-strike times (OpenCap clock).

    Uses the mean vertical height of available foot markers. Heel strikes are
    local minima of that height (foot lowest = in contact). Returns
    (height_signal, heelstrike_times).
    """
    from scipy.signal import find_peaks

    prefix = "r_" if side == "R" else "l_"
    candidates = [f"{prefix}ankle", f"{prefix}5meta", f"{prefix}toe",
                  f"{prefix}calc", f"{prefix}heel"]
    cols = [markers.data[c][:, VERTICAL_AXIS] for c in candidates if c in markers.data]
    if not cols:
        raise KeyError(f"no foot markers found for side {side}")
    height = np.nanmean(np.vstack(cols), axis=0)

    # Heel strike ~ minima of height: find peaks of the inverted signal.
    span = np.nanmax(height) - np.nanmin(height)
    peaks, _ = find_peaks(-height, prominence=0.25 * span if span > 0 else None,
                          distance=int(0.3 * markers.fs))
    return height, markers.time[peaks]
