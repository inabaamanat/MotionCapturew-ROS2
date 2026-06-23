"""Incremental gait tracker driven by per-foot vertical GRF (force clock).

Detects heel strike (stance rising edge) / toe off (falling edge) per foot as
samples arrive, and maintains live spatiotemporal metrics over a sliding window
of recent strides. Mirrors the offline definitions in
treadmill_opencap_fusion/fusion/gait.py but in streaming form.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .detector import IDX

GRAVITY = 9.80665


class LiveGait:
    def __init__(self, body_mass_kg, stance_threshold_bw=0.05,
                 min_step_interval_s=0.30, history=12):
        self.thr = stance_threshold_bw * body_mass_kg * GRAVITY
        self.min_int = min_step_interval_s
        self.in_stance = {"L": False, "R": False}
        self.last_hs = {"L": deque(maxlen=history), "R": deque(maxlen=history)}
        self.last_to = {"L": deque(maxlen=history), "R": deque(maxlen=history)}
        self.all_hs = deque(maxlen=2 * history)
        self.last_event = None     # ("HS"/"TO", side, t)
        self.phase = {"L": "unknown", "R": "unknown"}
        self.contact = {"L": "swing", "R": "swing"}
        self.current_hs = {"L": np.nan, "R": np.nan}
        self.current_to = {"L": np.nan, "R": np.nan}
        self.pose_ap = {"L": np.nan, "R": np.nan}
        self.last_hs_ap = {"L": np.nan, "R": np.nan}
        self.camera_stride = {"L": deque(maxlen=history), "R": deque(maxlen=history)}

    def update(self, t, fz_l, fz_r):
        """Feed one force sample (calibrated vertical N). Returns event or None."""
        ev = None
        for side, fz in (("L", fz_l), ("R", fz_r)):
            stance = fz > self.thr
            if stance and not self.in_stance[side]:
                if not self.last_hs[side] or (t - self.last_hs[side][-1]) >= self.min_int:
                    self.last_hs[side].append(t)
                    self.all_hs.append(t)
                    self.current_hs[side] = t
                    self._record_camera_stride(side)
                    ev = ("HS", side, t)
                    self.last_event = ev
            elif not stance and self.in_stance[side]:
                self.last_to[side].append(t)
                self.current_to[side] = t
                ev = ("TO", side, t)
                self.last_event = ev
            self.in_stance[side] = stance
            self.contact[side] = "stance" if stance else "swing"
            self.phase[side] = self._phase_for(side, t)
        return ev

    def update_pose(self, pts3d, valid):
        """Track foot AP position from 3D pose when stereo data is available.

        On treadmills, belt speed x stride time is the better stride-length
        estimate. This camera-derived value is a fallback and a useful QC signal.
        """
        if pts3d is None or valid is None:
            return
        needed = ["Lhip", "Rhip", "Lshoulder", "Rshoulder"]
        if not all(valid[IDX[n]] and np.all(np.isfinite(pts3d[IDX[n]])) for n in needed):
            return
        hip_mid = 0.5 * (pts3d[IDX["Lhip"]] + pts3d[IDX["Rhip"]])
        sho_mid = 0.5 * (pts3d[IDX["Lshoulder"]] + pts3d[IDX["Rshoulder"]])
        up = sho_mid - hip_mid
        up_norm = np.linalg.norm(up)
        if up_norm < 1e-9:
            return
        up = up / up_norm
        ml = pts3d[IDX["Rhip"]] - pts3d[IDX["Lhip"]]
        ml = ml - np.dot(ml, up) * up
        ml_norm = np.linalg.norm(ml)
        if ml_norm < 1e-9:
            return
        ml = ml / ml_norm
        anterior = np.cross(up, ml)
        anterior_norm = np.linalg.norm(anterior)
        if anterior_norm < 1e-9:
            return
        anterior = anterior / anterior_norm
        for side, ankle in (("L", "Lankle"), ("R", "Rankle")):
            if valid[IDX[ankle]] and np.all(np.isfinite(pts3d[IDX[ankle]])):
                self.pose_ap[side] = float(np.dot(pts3d[IDX[ankle]] - hip_mid, anterior))

    def _record_camera_stride(self, side):
        ap = self.pose_ap[side]
        prev = self.last_hs_ap[side]
        if np.isfinite(ap) and np.isfinite(prev):
            self.camera_stride[side].append(abs(ap - prev))
        if np.isfinite(ap):
            self.last_hs_ap[side] = ap

    def _phase_for(self, side, t):
        stance_mean, swing_mean = self._stance_swing(side)
        if self.in_stance[side]:
            start = self.current_hs[side]
            if not np.isfinite(start):
                return "stance"
            denom = stance_mean if np.isfinite(stance_mean) and stance_mean > 0 else 0.65
            frac = np.clip((t - start) / denom, 0.0, 1.2)
            if frac < 0.17:
                return "loading response"
            if frac < 0.50:
                return "mid stance"
            if frac < 0.83:
                return "terminal stance"
            return "pre swing"
        start = self.current_to[side]
        if not np.isfinite(start):
            return "swing"
        denom = swing_mean if np.isfinite(swing_mean) and swing_mean > 0 else 0.40
        frac = np.clip((t - start) / denom, 0.0, 1.2)
        if frac < 0.33:
            return "initial swing"
        if frac < 0.66:
            return "mid swing"
        return "terminal swing"

    @staticmethod
    def _mean_diff(dq):
        a = np.array(dq)
        return float(np.mean(np.diff(a))) if a.size > 1 else float("nan")

    def _stance_swing(self, side):
        hs = list(self.last_hs[side])
        to = list(self.last_to[side])
        st, sw = [], []
        for i in range(len(hs) - 1):
            nxt = hs[i + 1]
            t_off = [x for x in to if hs[i] < x < nxt]
            if t_off:
                st.append(t_off[0] - hs[i])
                sw.append(nxt - t_off[0])
        return (float(np.mean(st)) if st else np.nan,
                float(np.mean(sw)) if sw else np.nan)

    def metrics(self, belt_speed_mps=0.0) -> dict:
        stl, swl = self._stance_swing("L")
        str_, swr = self._stance_swing("R")
        step_t = self._mean_diff(sorted(self.all_hs))
        strL = self._mean_diff(self.last_hs["L"])
        strR = self._mean_diff(self.last_hs["R"])

        def sym(a, b):
            if not (np.isfinite(a) and np.isfinite(b)) or (a + b) == 0:
                return np.nan
            return 2 * abs(a - b) / (a + b) * 100

        def duty(stance, stride):
            if not (np.isfinite(stance) and np.isfinite(stride)) or stride <= 0:
                return np.nan
            return stance / stride

        def spatial(side, duration):
            if np.isfinite(duration) and belt_speed_mps and belt_speed_mps > 0.03:
                return float(abs(belt_speed_mps) * duration)
            arr = np.array(self.camera_stride[side], float)
            return float(np.nanmean(arr)) if arr.size and np.any(np.isfinite(arr)) else np.nan

        stride_len_L = spatial("L", strL)
        stride_len_R = spatial("R", strR)
        stance_len_L = float(abs(belt_speed_mps) * stl) if np.isfinite(stl) and belt_speed_mps > 0.03 else np.nan
        stance_len_R = float(abs(belt_speed_mps) * str_) if np.isfinite(str_) and belt_speed_mps > 0.03 else np.nan

        return {
            "cadence_steps_per_min": 60.0 / step_t if np.isfinite(step_t) and step_t > 0 else np.nan,
            "stride_time_L_s": strL, "stride_time_R_s": strR,
            "stance_time_L_s": stl, "stance_time_R_s": str_,
            "swing_time_L_s": swl, "swing_time_R_s": swr,
            "duty_factor_L": duty(stl, strL), "duty_factor_R": duty(str_, strR),
            "stride_length_L_m": stride_len_L, "stride_length_R_m": stride_len_R,
            "stance_length_L_m": stance_len_L, "stance_length_R_m": stance_len_R,
            "contact_L": self.contact["L"], "contact_R": self.contact["R"],
            "phase_L": self.phase["L"], "phase_R": self.phase["R"],
            "stride_time_symmetry_pct": sym(strL, strR),
            "stance_time_symmetry_pct": sym(stl, str_),
            "stride_length_symmetry_pct": sym(stride_len_L, stride_len_R),
        }
