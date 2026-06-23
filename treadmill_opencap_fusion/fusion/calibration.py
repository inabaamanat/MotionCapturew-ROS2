"""Convert raw force-plate volts to Newtons / Newton-metres and compute COP.

Three calibration modes (config -> calibration.mode):
  matrix     : per-plate 6x6 Bertec matrix, SI = M @ volts   (most accurate)
  gains      : per-channel scalar, SI = volts * gain
  bodyweight : derive the Fz volts->N scale from known body mass so the mean
               total stance force equals body weight; reuse that scale for the
               other channels (vertical accurate, shear/moments approximate).

Centre of pressure (per plate), standard instrumented-treadmill convention:
  COPx = (-My + Fx * dz) / Fz
  COPy = ( Mx + Fy * dz) / Fz
with dz the height of the belt surface above the sensor origin. COP is set to
NaN whenever |Fz| is below a contact threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.80665
COMPONENTS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]


@dataclass
class PlateSignals:
    """Calibrated 6-DOF signals + COP for a single plate (SI units)."""

    Fx: np.ndarray
    Fy: np.ndarray
    Fz: np.ndarray
    Mx: np.ndarray
    My: np.ndarray
    Mz: np.ndarray
    COPx: np.ndarray
    COPy: np.ndarray

    def as_dict(self, prefix: str) -> dict[str, np.ndarray]:
        return {f"{prefix}_{k}": getattr(self, k)
                for k in ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "COPx", "COPy"]}


@dataclass
class CalibratedForce:
    left: PlateSignals
    right: PlateSignals
    units: dict[str, str]
    mode: str
    notes: str = ""

    @property
    def total_Fz(self) -> np.ndarray:
        return self.left.Fz + self.right.Fz


def _plate_volts(channels: dict[str, np.ndarray], side: str) -> np.ndarray:
    """Stack the 6 raw-volt channels for a side into (6, N): [Fx Fy Fz Mx My Mz]."""
    return np.vstack([channels[f"{side}_{c}"] for c in COMPONENTS])


def _compute_cop(Fx, Fy, Fz, Mx, My, dz, thresh):
    with np.errstate(divide="ignore", invalid="ignore"):
        copx = (-My + Fx * dz) / Fz
        copy = (Mx + Fy * dz) / Fz
    bad = np.abs(Fz) < thresh
    copx[bad] = np.nan
    copy[bad] = np.nan
    return copx, copy


def _build_plate(si6, dz, thresh) -> PlateSignals:
    Fx, Fy, Fz, Mx, My, Mz = si6
    copx, copy = _compute_cop(Fx, Fy, Fz, Mx, My, dz, thresh)
    return PlateSignals(Fx, Fy, Fz, Mx, My, Mz, copx, copy)


def calibrate(channels: dict[str, np.ndarray], cal_cfg: dict,
              body_mass_kg: float | None) -> CalibratedForce:
    mode = cal_cfg.get("mode", "bodyweight")
    dz = float(cal_cfg.get("surface_dz_m", 0.0))
    thresh = float(cal_cfg.get("cop_fz_threshold_n", 20.0))
    notes = ""

    volts_L = _plate_volts(channels, "L")
    volts_R = _plate_volts(channels, "R")

    if mode == "matrix":
        mats = cal_cfg.get("matrix", {})
        if mats.get("left") is None or mats.get("right") is None:
            raise ValueError("calibration.mode=matrix but matrix.left/right not set")
        ML = np.asarray(mats["left"], dtype=float)
        MR = np.asarray(mats["right"], dtype=float)
        if ML.shape != (6, 6) or MR.shape != (6, 6):
            raise ValueError("calibration matrices must be 6x6")
        si_L = ML @ volts_L
        si_R = MR @ volts_R
        units = {"F": "N", "M": "N*m"}

    elif mode == "gains":
        g = cal_cfg["gains"]
        gl = np.array([g[f"L_{c}"] for c in COMPONENTS]).reshape(6, 1)
        gr = np.array([g[f"R_{c}"] for c in COMPONENTS]).reshape(6, 1)
        si_L = volts_L * gl
        si_R = volts_R * gr
        units = {"F": "N", "M": "N*m"}

    elif mode == "bodyweight":
        mass = cal_cfg.get("bodyweight", {}).get("body_mass_kg") or body_mass_kg
        if mass is None:
            raise ValueError("bodyweight calibration needs body_mass_kg or .osim mass")
        bodyweight_n = mass * GRAVITY
        # Estimate volts->N from the mean total vertical force during stance.
        # During steady walking the time-averaged total vertical GRF equals
        # body weight, so scale = bodyweight / mean(total Fz volts in contact).
        fz_volts_total = channels["L_Fz"] + channels["R_Fz"]
        contact = fz_volts_total > 0.15 * np.nanmax(fz_volts_total)
        mean_v = np.mean(fz_volts_total[contact]) if contact.any() else np.nan
        scale = bodyweight_n / mean_v if mean_v and np.isfinite(mean_v) else np.nan
        si_L = volts_L * scale
        si_R = volts_R * scale
        units = {"F": "N (approx)", "M": "N*m (approx)"}
        notes = (f"bodyweight calibration: mass={mass:.1f} kg, "
                 f"scale={scale:.1f} N/V (shear & moments approximate)")
    else:
        raise ValueError(f"unknown calibration mode: {mode}")

    left = _build_plate(si_L, dz, thresh)
    right = _build_plate(si_R, dz, thresh)
    return CalibratedForce(left=left, right=right, units=units,
                           mode=mode, notes=notes)
