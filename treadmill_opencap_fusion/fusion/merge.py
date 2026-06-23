"""Merge force, kinematics and gait data onto the raw-video frame timeline.

Produces one row per raw video frame with:
  * timing columns (raw_frame, opencap_time_s, force_time_s, flags)
  * every joint angle (deg) and its angular velocity (deg/s)
  * pelvis translations and velocities
  * calibrated GRF for both plates (Fx Fy Fz Mx My Mz), COP, and totals
  * GRF normalized to body weight
  * per-foot stance flag and gait-cycle phase (%)

Marker trajectories (needed for the skeleton overlay) are returned separately as
an (n_frames, n_markers, 3) array aligned to the same frame timeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

GRAVITY = 9.80665


@dataclass
class MergedTrial:
    table: pd.DataFrame
    markers: np.ndarray          # (n_frames, n_markers, 3), NaN where unavailable
    marker_names: list[str]
    meta: dict


def _interp_cols(time_src, value_cols, t_query):
    """Linear interpolation of each column at t_query; NaN outside source range."""
    out = {}
    lo, hi = time_src[0], time_src[-1]
    inside = (t_query >= lo) & (t_query <= hi)
    for name, vals in value_cols.items():
        f = interp1d(time_src, vals, bounds_error=False, fill_value=np.nan)
        col = f(t_query)
        col[~inside] = np.nan
        out[name] = col
    return out, inside


def _angular_velocity(time, angle, fs):
    """Smoothed time-derivative (deg/s) on a uniform-ish grid."""
    win = max(5, int(round(0.10 * fs)) | 1)
    if len(angle) < win:
        return np.gradient(angle, time)
    return savgol_filter(angle, win, 3, deriv=1, delta=1.0 / fs)


def _gait_phase(t_query, heel_strikes):
    """Percent of gait cycle (0..100) for each query time; NaN outside strides."""
    phase = np.full(t_query.shape, np.nan)
    hs = np.sort(heel_strikes)
    for i in range(len(hs) - 1):
        a, b = hs[i], hs[i + 1]
        m = (t_query >= a) & (t_query < b)
        phase[m] = (t_query[m] - a) / (b - a) * 100.0
    return phase


def build_merged_table(cfg, force, cforce, kin, markers, force_gait, sync,
                       align, raw_info, body_mass_kg) -> MergedTrial:
    fps = raw_info.fps
    n = raw_info.n_frames
    opencap_fps = kin.fs

    raw_idx = np.arange(n)
    # raw_frame = slope*sync_frame + intercept ; sync_frame = opencap_time*fps_oc
    sync_idx = (raw_idx - align.intercept) / align.slope
    t_oc = sync_idx / opencap_fps                 # OpenCap clock (s)
    t_f = t_oc + sync.offset_s                     # force clock (s)

    df = pd.DataFrame({
        "raw_frame": raw_idx,
        "opencap_time_s": t_oc,
        "force_time_s": t_f,
    })

    # ---- joint angles + angular velocities (sampled from the .mot) ----
    kin_cols = {name: kin.coords[name].to_numpy() for name in kin.names}
    vel_cols = {f"{name}_vel": _angular_velocity(kin.time, v, kin.fs)
                for name, v in kin_cols.items()}
    ang, has_kin = _interp_cols(kin.time, kin_cols, t_oc)
    vel, _ = _interp_cols(kin.time, vel_cols, t_oc)
    for name, col in ang.items():
        df[f"{name}_deg" if kin.in_degrees else name] = col
    for name, col in vel.items():
        df[f"{name}_degps" if kin.in_degrees else name] = col
    df["has_kinematics"] = has_kin

    # ---- calibrated forces, COP, totals (sampled from the force clock) ----
    fcols = {}
    for prefix, plate in (("L", cforce.left), ("R", cforce.right)):
        fcols.update(plate.as_dict(prefix))
    fres, has_force = _interp_cols(force.t, fcols, t_f)
    for name, col in fres.items():
        df[name] = col
    df["has_force"] = has_force

    bw = body_mass_kg * GRAVITY
    df["total_Fx_N"] = df["L_Fx"] + df["R_Fx"]
    df["total_Fy_N"] = df["L_Fy"] + df["R_Fy"]
    df["total_Fz_N"] = df["L_Fz"] + df["R_Fz"]
    df["total_Fz_BW"] = df["total_Fz_N"] / bw
    df["L_Fz_BW"] = df["L_Fz"] / bw
    df["R_Fz_BW"] = df["R_Fz"] / bw

    # rename plate force columns with units for clarity
    df.rename(columns={f"{s}_{c}": f"{s}_{c}_{u}"
                       for s in ("L", "R")
                       for c, u in (("Fx", "N"), ("Fy", "N"), ("Fz", "N"),
                                    ("Mx", "Nm"), ("My", "Nm"), ("Mz", "Nm"),
                                    ("COPx", "m"), ("COPy", "m"))},
              inplace=True)

    # ---- stance flags + gait-cycle phase (force clock) ----
    for side, fev in (("L", force_gait.left), ("R", force_gait.right)):
        st = interp1d(force.t, fev.stance.astype(float), bounds_error=False,
                      fill_value=0.0)(t_f)
        df[f"{side}_stance"] = (st > 0.5) & has_force
        df[f"{side}_gait_phase_pct"] = _gait_phase(t_f, fev.heel_strikes)

    # ---- marker array aligned to the frame timeline ----
    marker_names = list(markers.data.keys())
    M = np.full((n, len(marker_names), 3), np.nan)
    for mi, name in enumerate(marker_names):
        arr = markers.data[name]
        for ax in range(3):
            f = interp1d(markers.time, arr[:, ax], bounds_error=False,
                         fill_value=np.nan)
            col = f(t_oc)
            col[~has_kin] = np.nan
            M[:, mi, ax] = col

    meta = {
        "trial": cfg.trial_name,
        "sync_offset_s": sync.offset_s,
        "sync_correlation": sync.correlation,
        "sync_corr_L": sync.corr_left,
        "sync_corr_R": sync.corr_right,
        "sync_heelstrike_rms_s": sync.heelstrike_rms_s,
        "sync_heelstrike_matched": sync.heelstrike_matched,
        "video_align_slope": align.slope,
        "video_align_intercept": align.intercept,
        "video_align_quality": align.quality,
        "body_mass_kg": body_mass_kg,
        "bodyweight_N": bw,
        "calibration_mode": cforce.mode,
        "calibration_notes": cforce.notes,
        "force_fs_hz": force.fs,
        "kinematics_fs_hz": kin.fs,
        "video_fps": fps,
        "n_frames": int(n),
        "gait_metrics": force_gait.metrics,
    }
    return MergedTrial(table=df, markers=M, marker_names=marker_names, meta=meta)


def export_table(merged: MergedTrial, out_dir: str):
    import os
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "merged.csv")
    merged.table.to_csv(csv_path, index=False)
    try:
        merged.table.to_parquet(os.path.join(out_dir, "merged.parquet"), index=False)
    except Exception:
        pass  # parquet optional
    np.savez_compressed(os.path.join(out_dir, "markers.npz"),
                        markers=merged.markers, names=np.array(merged.marker_names))
    return csv_path
