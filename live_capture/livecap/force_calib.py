"""Per-sample force calibration (volts -> N/Nm) + COP, for the live stream.

Mirrors treadmill_opencap_fusion/fusion/calibration.py but applies to a single
12-vector at a time. Modes: matrix | gains | bodyweight.
"""
from __future__ import annotations

import numpy as np

GRAVITY = 9.80665
COMP = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]


class LiveForceCalibrator:
    def __init__(self, mode="bodyweight", body_mass_kg=None, matrix=None,
                 gains=None, scale_n_per_v=None, surface_dz_m=0.0,
                 cop_fz_threshold_n=20.0):
        self.mode = mode
        self.body_mass_kg = body_mass_kg
        self.dz = float(surface_dz_m)
        self.cop_thr = float(cop_fz_threshold_n)
        self.scale = scale_n_per_v          # bodyweight mode (set later if None)
        self.ML = np.asarray(matrix["left"], float) if (matrix and matrix.get("left")) else None
        self.MR = np.asarray(matrix["right"], float) if (matrix and matrix.get("right")) else None
        if gains:
            self.gL = np.array([gains[f"L_{c}"] for c in COMP])
            self.gR = np.array([gains[f"R_{c}"] for c in COMP])
        else:
            self.gL = self.gR = None

    def set_bodyweight_scale_from_volts(self, fz_l_volts, fz_r_volts,
                                        min_abs_total_volts=0.0):
        """Estimate N/V so quiet-standing total Fz equals body weight.

        The sign of the scale is intentionally preserved: some Bertec channel
        mappings report vertical load as negative volts.
        """
        total = np.asarray(fz_l_volts) + np.asarray(fz_r_volts)
        finite = np.isfinite(total)
        if not finite.any():
            return self.scale
        contact = finite & (np.abs(total) >= float(min_abs_total_volts))
        if not contact.any():
            return self.scale
        mean_v = np.nanmedian(total[contact])
        if self.body_mass_kg and np.isfinite(mean_v) and abs(mean_v) > 1e-9:
            self.scale = self.body_mass_kg * GRAVITY / mean_v
        return self.scale

    def apply(self, volts12) -> dict:
        v = np.asarray(volts12, float)
        vL, vR = v[:6], v[6:]
        if self.mode == "matrix" and self.ML is not None:
            sL, sR = self.ML @ vL, self.MR @ vR
        elif self.mode == "gains" and self.gL is not None:
            sL, sR = vL * self.gL, vR * self.gR
        else:  # bodyweight
            s = self.scale if self.scale else 1.0
            sL, sR = vL * s, vR * s

        out = {}
        for side, s6 in (("L", sL), ("R", sR)):
            Fx, Fy, Fz, Mx, My, Mz = s6
            for c, val in zip(COMP, s6):
                out[f"{side}_{c}"] = float(val)
            if abs(Fz) >= self.cop_thr:
                out[f"{side}_COPx"] = float((-My + Fx * self.dz) / Fz)
                out[f"{side}_COPy"] = float((Mx + Fy * self.dz) / Fz)
            else:
                out[f"{side}_COPx"] = np.nan
                out[f"{side}_COPy"] = np.nan
        return out


def build_calibrator(cfg):
    fc = cfg.get("calibration", "force")
    return LiveForceCalibrator(
        mode=fc.get("mode", "bodyweight"),
        body_mass_kg=fc.get("body_mass_kg"),
        matrix=fc.get("matrix"),
        gains=fc.get("gains"),
        scale_n_per_v=fc.get("scale_n_per_v"),
        surface_dz_m=fc.get("surface_dz_m", 0.0),
        cop_fz_threshold_n=fc.get("cop_fz_threshold_n", 20.0),
    )
