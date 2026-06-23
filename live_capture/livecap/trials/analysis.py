"""Build normalized trial artifacts from the live recorder outputs.

The recorder keeps the original fusion-compatible raw files at the recording
root. This module adds a stable trial layer beside them:

  metadata.json
  frame_data.npz
  step_events.json
  derived_metrics.json
  pressure_timeseries.npz
  quality_metrics.json
  raw/manifest.json
  exports/

Raw pose/force/video files remain separate from derived metrics so later
analyses can be rerun from the same source data.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..pose.detector import COCO_NAMES, IDX
from ..state import ANGLE_NAMES

GRAVITY = 9.80665
ANGLE_KEYS = list(ANGLE_NAMES)
FORCE_KEYS = ["L_Fx", "L_Fy", "L_Fz", "L_Mx", "L_My", "L_Mz",
              "R_Fx", "R_Fy", "R_Fz", "R_Mx", "R_My", "R_Mz",
              "L_COPx", "L_COPy", "R_COPx", "R_COPy"]


def _finite_mean(values, default=float("nan")):
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else default


def _finite_sum(values, default=float("nan")):
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    return float(np.sum(arr)) if arr.size else default


def _safe_float(v):
    try:
        f = float(v)
    except Exception:
        return None
    return f if math.isfinite(f) else None


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_ready(obj.tolist())
    if isinstance(obj, (np.integer, np.floating)):
        return _json_ready(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _write_json(path: Path, payload: dict | list):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _nearest_indices(source_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    if source_t.size == 0 or target_t.size == 0:
        return np.zeros(target_t.size, dtype=int)
    idx = np.searchsorted(source_t, target_t, side="left")
    idx = np.clip(idx, 0, source_t.size - 1)
    left = np.clip(idx - 1, 0, source_t.size - 1)
    use_left = np.abs(source_t[left] - target_t) < np.abs(source_t[idx] - target_t)
    idx[use_left] = left[use_left]
    return idx


def _gradient(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    if x.shape[0] < 2 or t.size < 2:
        return np.full_like(x, np.nan, dtype=float)
    tt = np.asarray(t, float)
    if np.nanmax(tt) - np.nanmin(tt) <= 1e-9:
        return np.full_like(x, np.nan, dtype=float)
    return np.gradient(x, tt, axis=0, edge_order=1)


def _load_npz(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _camera_confidence(kp0: np.ndarray, kp1: np.ndarray | None = None) -> np.ndarray:
    c0 = kp0[:, :, 2] if kp0.ndim == 3 and kp0.shape[-1] >= 3 else np.full(kp0.shape[:2], np.nan)
    if kp1 is None or kp1.size == 0:
        return c0
    c1 = kp1[:, :, 2] if kp1.ndim == 3 and kp1.shape[-1] >= 3 else np.full_like(c0, np.nan)
    return np.nanmean(np.stack([c0, c1]), axis=0)


def _pelvis_and_body(kp3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = kp3d.shape[0]
    pelvis = np.full((n, 3), np.nan)
    body = np.full((n, 3), np.nan)
    com = np.full((n, 3), np.nan)
    hip_idx = [IDX["Lhip"], IDX["Rhip"]]
    body_idx = [IDX["Lhip"], IDX["Rhip"], IDX["Lshoulder"], IDX["Rshoulder"]]

    def mean_or_nan(points):
        if not np.any(np.isfinite(points)):
            return np.full(3, np.nan)
        return np.nanmean(points, axis=0)

    for i in range(n):
        pelvis[i] = mean_or_nan(kp3d[i, hip_idx, :])
        body[i] = mean_or_nan(kp3d[i, body_idx, :])
        com[i] = mean_or_nan(kp3d[i, :, :])
    return pelvis, body, com


def _angle_table(pose: dict[str, Any], frame_t: np.ndarray) -> dict[str, Any]:
    angle_names = [str(x) for x in pose.get("angle_names", np.array(ANGLE_KEYS)).tolist()]
    angles = np.asarray(pose.get("angles", np.zeros((frame_t.size, 0))), float)
    if angles.ndim != 2:
        angles = np.zeros((frame_t.size, 0))

    series: dict[str, list[float]] = {}
    for i, name in enumerate(angle_names):
        if i < angles.shape[1]:
            series[name] = angles[:, i].astype(float).tolist()

    def from_existing(name, fallback=None, scale=1.0):
        if name in series:
            return np.asarray(series[name], float)
        if fallback and fallback in series:
            return np.asarray(series[fallback], float) * scale
        return np.full(frame_t.size, np.nan)

    requested = {
        "hip_flexion_l": from_existing("hip_flexion_l"),
        "hip_flexion_r": from_existing("hip_flexion_r"),
        "hip_abduction_l": from_existing("hip_abduction_l", "hip_adduction_l", -1.0),
        "hip_abduction_r": from_existing("hip_abduction_r", "hip_adduction_r", -1.0),
        "hip_internal_rotation_l": from_existing("hip_internal_rotation_l"),
        "hip_internal_rotation_r": from_existing("hip_internal_rotation_r"),
        "knee_flexion_l": from_existing("knee_flexion_l", "knee_angle_l"),
        "knee_flexion_r": from_existing("knee_flexion_r", "knee_angle_r"),
        "ankle_dorsiflexion_l": from_existing("ankle_dorsiflexion_l", "ankle_angle_l"),
        "ankle_dorsiflexion_r": from_existing("ankle_dorsiflexion_r", "ankle_angle_r"),
        "ankle_plantarflexion_l": -from_existing("ankle_dorsiflexion_l", "ankle_angle_l"),
        "ankle_plantarflexion_r": -from_existing("ankle_dorsiflexion_r", "ankle_angle_r"),
        "ankle_inversion_l": from_existing("ankle_inversion_l"),
        "ankle_inversion_r": from_existing("ankle_inversion_r"),
        "ankle_eversion_l": from_existing("ankle_eversion_l"),
        "ankle_eversion_r": from_existing("ankle_eversion_r"),
    }
    velocity = {k: _gradient(v.reshape(-1, 1), frame_t).reshape(-1).tolist()
                for k, v in requested.items()}
    acceleration = {k: _gradient(np.asarray(velocity[k]).reshape(-1, 1), frame_t).reshape(-1).tolist()
                    for k in requested}
    return {
        "time_s": frame_t.tolist(),
        "angles_deg": {k: v.tolist() for k, v in requested.items()},
        "angular_velocity_deg_s": velocity,
        "angular_acceleration_deg_s2": acceleration,
        "source_angle_names": angle_names,
    }


def _derive_force_events(force: dict[str, Any], body_mass_kg: float,
                         threshold_bw: float) -> list[dict[str, Any]]:
    data = np.asarray(force.get("DATA", np.zeros((14, 0))), float)
    cal = np.asarray(force.get("CAL", np.zeros((0, len(FORCE_KEYS)))), float)
    if data.shape[1] == 0:
        return []
    t = np.asarray(data[13], float)
    if cal.shape[0] != t.size:
        return []
    fz_l, fz_r = cal[:, FORCE_KEYS.index("L_Fz")], cal[:, FORCE_KEYS.index("R_Fz")]
    thr = threshold_bw * body_mass_kg * GRAVITY
    events = []
    for side, fz in (("L", fz_l), ("R", fz_r)):
        stance = np.asarray(fz > thr, bool)
        if stance.size < 2:
            continue
        changes = np.where(np.diff(stance.astype(int)) != 0)[0] + 1
        last_hs = -1e9
        for i in changes:
            if stance[i]:
                if float(t[i]) - last_hs >= 0.20:
                    events.append({"type": "HS", "side": side, "timestamp": float(t[i])})
                    last_hs = float(t[i])
            else:
                events.append({"type": "TO", "side": side, "timestamp": float(t[i])})
    return sorted(events, key=lambda e: e["timestamp"])


def _load_recorded_events(force: dict[str, Any], body_mass_kg: float,
                          threshold_bw: float) -> list[dict[str, Any]]:
    if {"EVENT_TYPE", "EVENT_SIDE", "EVENT_T"} <= set(force):
        out = []
        types = force["EVENT_TYPE"].astype(str)
        sides = force["EVENT_SIDE"].astype(str)
        times = np.asarray(force["EVENT_T"], float)
        for typ, side, ts in zip(types, sides, times):
            if typ and typ != "nan" and side:
                out.append({"type": typ, "side": side, "timestamp": float(ts)})
        if out:
            return sorted(out, key=lambda e: e["timestamp"])
    return _derive_force_events(force, body_mass_kg, threshold_bw)


def _ankle_ap_width_height(frame_data: dict[str, np.ndarray], frame_idx: int, side: str):
    P = frame_data["world_positions"]
    if P.size == 0:
        return np.nan, np.nan, np.nan
    side_idx = IDX[f"{side}ankle"]
    other_idx = IDX["Rankle" if side == "L" else "Lankle"]
    p = P[frame_idx, side_idx]
    other = P[frame_idx, other_idx]
    step_width = abs(p[0] - other[0]) if np.all(np.isfinite([p[0], other[0]])) else np.nan
    foot_height = p[1] if np.isfinite(p[1]) else np.nan
    return float(p[2]) if np.isfinite(p[2]) else np.nan, step_width, foot_height


def _pressure_metrics(force: dict[str, Any], events: list[dict[str, Any]],
                      cfg, recording_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = np.asarray(force.get("DATA", np.zeros((14, 0))), float)
    cal = np.asarray(force.get("CAL", np.zeros((0, len(FORCE_KEYS)))), float)
    if data.shape[1] == 0 or cal.shape[0] == 0:
        empty = {"time_s": [], "left": {}, "right": {}}
        np.savez_compressed(recording_dir / "pressure_timeseries.npz", time_s=np.zeros(0))
        return empty, []
    t = np.asarray(data[13], float)
    left = {k[2:]: cal[:, FORCE_KEYS.index(k)] for k in FORCE_KEYS if k.startswith("L_")}
    right = {k[2:]: cal[:, FORCE_KEYS.index(k)] for k in FORCE_KEYS if k.startswith("R_")}
    np.savez_compressed(
        recording_dir / "pressure_timeseries.npz",
        time_s=t,
        L_Fz=left["Fz"], R_Fz=right["Fz"],
        L_COPx=left["COPx"], L_COPy=left["COPy"],
        R_COPx=right["COPx"], R_COPy=right["COPy"],
    )

    contact_area = float(cfg.get("pressure", "estimated_contact_area_m2", default=0.015) or 0.015)
    step_pressure = []
    for ev in events:
        if ev["type"] != "HS":
            continue
        side = ev["side"]
        next_to = next((x for x in events
                        if x["side"] == side and x["type"] == "TO"
                        and x["timestamp"] > ev["timestamp"]), None)
        end = next_to["timestamp"] if next_to else ev["timestamp"] + 0.75
        mask = (t >= ev["timestamp"]) & (t <= end)
        if not np.any(mask):
            continue
        src = left if side == "L" else right
        fz = np.asarray(src["Fz"], float)[mask]
        copx = np.asarray(src["COPx"], float)[mask]
        copy = np.asarray(src["COPy"], float)[mask]
        pressure = np.maximum(fz, 0.0) / contact_area
        segments = {
            "heel_pressure": pressure[copy < -0.15],
            "forefoot_pressure": pressure[(copy >= -0.15) & (copy < 0.20)],
            "toe_pressure": pressure[copy >= 0.20],
        }
        cop = np.column_stack([copx, copy])
        cop_v = np.linalg.norm(_gradient(cop, t[mask]), axis=1) if np.sum(mask) > 1 else np.array([np.nan])
        step_pressure.append({
            "step_id": None,
            "side": side,
            "start_time": float(ev["timestamp"]),
            "end_time": float(end),
            "recording_ref": "pressure_timeseries.npz",
            "peak_pressure_pa": _safe_float(np.nanmax(pressure)),
            "average_pressure_pa": _safe_float(np.nanmean(pressure)),
            "contact_area_m2": contact_area,
            "heel_pressure_pa": _safe_float(_finite_mean(segments["heel_pressure"])),
            "forefoot_pressure_pa": _safe_float(_finite_mean(segments["forefoot_pressure"])),
            "toe_pressure_pa": _safe_float(_finite_mean(segments["toe_pressure"])),
            "center_of_pressure_path_m": _json_ready(cop[::max(1, len(cop) // 40)]),
            "center_of_pressure_velocity_m_s": _safe_float(_finite_mean(cop_v)),
        })
    return {"time_s": t.tolist()}, step_pressure


def _build_step_events(force: dict[str, Any], frame_data: dict[str, np.ndarray],
                       pressure_by_hs: list[dict[str, Any]], cfg) -> tuple[list[dict], list[dict]]:
    body_mass = float(cfg.get("calibration", "force", "body_mass_kg", default=75.0) or 75.0)
    threshold_bw = float(cfg.get("gait", "stance_threshold_bw", default=0.05) or 0.05)
    events = _load_recorded_events(force, body_mass, threshold_bw)
    frame_t = frame_data["timestamp"]
    steps = []
    hs = [e for e in events if e["type"] == "HS"]
    treadmill_speed = 0.0
    if "TREAD" in force and np.asarray(force["TREAD"]).size:
        tread = np.asarray(force["TREAD"], float)
        if tread.ndim == 2 and tread.shape[0] >= 2:
            treadmill_speed = abs(_finite_mean(tread[1]))
    for i, ev in enumerate(hs):
        side = ev["side"]
        hs_t = ev["timestamp"]
        same_next = next((x for x in hs[i + 1:] if x["side"] == side), None)
        any_next = hs[i + 1] if i + 1 < len(hs) else None
        toe_off = next((x for x in events
                        if x["side"] == side and x["type"] == "TO" and x["timestamp"] > hs_t
                        and (same_next is None or x["timestamp"] < same_next["timestamp"])), None)
        end_t = same_next["timestamp"] if same_next else (toe_off["timestamp"] if toe_off else hs_t)
        start_frame = int(_nearest_indices(frame_t, np.array([hs_t]))[0]) if frame_t.size else 0
        end_frame = int(_nearest_indices(frame_t, np.array([end_t]))[0]) if frame_t.size else start_frame
        stride_time = (same_next["timestamp"] - hs_t) if same_next else np.nan
        step_time = (any_next["timestamp"] - hs_t) if any_next else np.nan
        toe_t = toe_off["timestamp"] if toe_off else np.nan
        stance = toe_t - hs_t if np.isfinite(toe_t) else np.nan
        swing = stride_time - stance if np.isfinite(stride_time) and np.isfinite(stance) else np.nan
        step_len = treadmill_speed * step_time if treadmill_speed > 0.03 and np.isfinite(step_time) else np.nan
        stride_len = treadmill_speed * stride_time if treadmill_speed > 0.03 and np.isfinite(stride_time) else np.nan
        _, width, foot_h = _ankle_ap_width_height(frame_data, start_frame, side)
        ankle_idx = IDX[f"{side}ankle"]
        seg = frame_data["world_positions"][start_frame:max(end_frame + 1, start_frame + 1), ankle_idx, 1]
        clearance = _safe_float(np.nanmax(seg) - np.nanmin(seg)) if seg.size else None
        pressure = next((p for p in pressure_by_hs
                         if p["side"] == side and abs(p["start_time"] - hs_t) < 1e-6), None)
        step_id = f"step_{i + 1:04d}"
        if pressure:
            pressure["step_id"] = step_id
            pressure["recording_ref"] = f"pressure_timeseries.npz#{step_id}"
        cadence = 60.0 / step_time if np.isfinite(step_time) and step_time > 0 else np.nan
        steps.append({
            "step_id": step_id,
            "timestamp": hs_t,
            "foot": "left" if side == "L" else "right",
            "side": side,
            "heel_strike_time": hs_t,
            "toe_off_time": _safe_float(toe_t),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "spatial_metrics": {
                "step_length_m": _safe_float(step_len),
                "stride_length_m": _safe_float(stride_len),
                "step_width_m": _safe_float(width),
                "foot_progression_angle_deg": None,
                "toe_angle_deg": None,
                "foot_clearance_m": clearance,
                "maximum_foot_height_m": _safe_float(np.nanmax(seg)) if seg.size else None,
                "walking_direction_deg": cfg.get("cameras", "mount_geometry", "walking_direction_deg", default=0),
            },
            "temporal_metrics": {
                "stance_time_s": _safe_float(stance),
                "swing_time_s": _safe_float(swing),
                "stride_time_s": _safe_float(stride_time),
                "step_time_s": _safe_float(step_time),
                "cadence_steps_per_min": _safe_float(cadence),
                "walking_speed_m_s": _safe_float(treadmill_speed) if treadmill_speed > 0.03 else None,
                "double_support_time_s": None,
                "single_support_time_s": None,
            },
            "pressure_data": pressure or {"recording_ref": None},
            "event_markers": [x for x in events if hs_t <= x["timestamp"] <= end_t],
        })
    return steps, events


def _summary(metadata: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    left = [s for s in steps if s["side"] == "L"]
    right = [s for s in steps if s["side"] == "R"]
    spatial = [s["spatial_metrics"] for s in steps]
    temporal = [s["temporal_metrics"] for s in steps]
    avg_step_len = _finite_mean([s.get("step_length_m") for s in spatial])
    total_steps = len(steps)
    duration = metadata.get("duration_s") or 0.0
    dist = _finite_sum([s.get("step_length_m") for s in spatial], default=np.nan)
    if not np.isfinite(dist):
        speed = _finite_mean([s.get("walking_speed_m_s") for s in temporal])
        dist = speed * duration if np.isfinite(speed) else np.nan
    return {
        "trial_id": metadata.get("trial_id"),
        "total_steps": total_steps,
        "left_step_count": len(left),
        "right_step_count": len(right),
        "average_stride_length_m": _safe_float(_finite_mean([s.get("stride_length_m") for s in spatial])),
        "average_step_length_m": _safe_float(avg_step_len),
        "average_cadence_steps_per_min": _safe_float(_finite_mean([s.get("cadence_steps_per_min") for s in temporal])),
        "average_walking_speed_m_s": _safe_float(_finite_mean([s.get("walking_speed_m_s") for s in temporal])),
        "average_stance_time_s": _safe_float(_finite_mean([s.get("stance_time_s") for s in temporal])),
        "average_swing_time_s": _safe_float(_finite_mean([s.get("swing_time_s") for s in temporal])),
        "total_distance_walked_m": _safe_float(dist),
        "recording_duration_s": _safe_float(duration),
    }


def build_trial_artifacts(recording_dir: str | os.PathLike, cfg) -> dict[str, Any]:
    """Create normalized trial files and return the metadata payload."""
    rec_dir = Path(recording_dir)
    rec_dir.mkdir(parents=True, exist_ok=True)
    (rec_dir / "raw").mkdir(exist_ok=True)
    (rec_dir / "exports").mkdir(exist_ok=True)

    force = _load_npz(rec_dir / "force.npz")
    pose = _load_npz(rec_dir / "pose.npz")
    frame_epoch = np.asarray(pose.get("t", np.zeros(0)), float)
    start_epoch = float(pose.get("start_epoch", force.get("START", time.time())))
    if frame_epoch.size:
        frame_t = frame_epoch - start_epoch
    else:
        frame_t = np.zeros(0)

    kp3d = np.asarray(pose.get("kp3d", np.zeros((0, len(COCO_NAMES), 3))), float)
    if kp3d.ndim != 3:
        kp3d = np.zeros((0, len(COCO_NAMES), 3))
    kp3d_raw = np.asarray(pose.get("kp3d_raw", kp3d), float)
    if kp3d_raw.shape != kp3d.shape:
        kp3d_raw = kp3d.copy()
    valid = np.asarray(pose.get("valid", np.isfinite(kp3d[:, :, 0]) if kp3d.size else np.zeros((0, len(COCO_NAMES)), bool)), bool)
    kp0 = np.asarray(pose.get("kp0", np.zeros((kp3d.shape[0], len(COCO_NAMES), 3))), float)
    kp1 = np.asarray(pose.get("kp1", np.zeros_like(kp0)), float)
    conf = _camera_confidence(kp0, kp1)
    tracking_conf = np.nanmean(conf, axis=1) if conf.size else np.zeros(0)
    pelvis, body, com = _pelvis_and_body(kp3d)
    local = kp3d - pelvis[:, None, :] if kp3d.size else kp3d
    velocity = _gradient(kp3d, frame_t)
    acceleration = _gradient(velocity, frame_t)
    body_velocity = _gradient(body, frame_t)
    body_acceleration = _gradient(body_velocity, frame_t)
    com_velocity = _gradient(com, frame_t)
    expected_dt = 1.0 / float(cfg.get("pose", "target_fps", default=30) or 30)
    dropped = np.zeros(frame_t.size, bool)
    if frame_t.size > 1:
        dropped[1:] = np.diff(frame_t) > expected_dt * 1.75
    missing = np.sum(~np.isfinite(kp3d[:, :, 0]), axis=1) if kp3d.size else np.zeros(0)
    occluded = np.sum(~valid, axis=1) if valid.size else missing
    reproj = np.asarray(pose.get("reprojection_error_px",
                                 np.full(frame_t.size, np.nan)), float)
    if reproj.shape[0] != frame_t.size:
        reproj = np.full(frame_t.size, np.nan)
    frame_data = {
        "frame_number": np.arange(frame_t.size, dtype=np.int64),
        "timestamp": frame_t,
        "epoch": frame_epoch,
        "tracking_confidence": tracking_conf,
        "dropped_frame": dropped,
        "world_positions": kp3d,
        "world_positions_raw": kp3d_raw,
        "local_positions": local,
        "rotations_quaternion": np.tile(np.array([np.nan, np.nan, np.nan, np.nan]), (frame_t.size, len(COCO_NAMES), 1)),
        "velocities": velocity,
        "accelerations": acceleration,
        "joint_confidence": conf,
        "pelvis_position": pelvis,
        "body_position": body,
        "center_of_mass": com,
        "center_of_mass_velocity": com_velocity,
        "body_velocity": body_velocity,
        "body_acceleration": body_acceleration,
        "quality_missing_joint_count": missing,
        "quality_interpolated_frame": np.zeros(frame_t.size, bool),
        "quality_occluded_joint_count": occluded,
        "quality_processing_latency_ms": np.full(frame_t.size, np.nan),
        "quality_reprojection_error_px": reproj,
    }
    np.savez_compressed(rec_dir / "frame_data.npz",
                        joint_names=np.array(COCO_NAMES), **frame_data)

    pressure_series, pressure_by_hs = _pressure_metrics(force, [], cfg, rec_dir)
    steps, events = _build_step_events(force, frame_data, pressure_by_hs, cfg)
    # Recompute pressure step ids now that step IDs exist.
    _, pressure_by_hs = _pressure_metrics(force, events, cfg, rec_dir)
    steps, events = _build_step_events(force, frame_data, pressure_by_hs, cfg)

    duration = 0.0
    if frame_t.size:
        duration = float(frame_t[-1] - frame_t[0])
    elif "DATA" in force and np.asarray(force["DATA"]).shape[1]:
        duration = float(np.asarray(force["DATA"])[13, -1] - np.asarray(force["DATA"])[13, 0])
    name = rec_dir.name
    existing = {}
    meta_path = rec_dir / "metadata.json"
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    metadata = {
        "session_id": existing.get("session_id") or str(uuid.uuid4()),
        "trial_id": existing.get("trial_id") or str(uuid.uuid4()),
        "subject_id": cfg.get("recording", "subject_id", default=None),
        "recording_name": cfg.get("recording", "name", default=name) or name,
        "notes": cfg.get("recording", "notes", default=""),
        "date_time": datetime.fromtimestamp(start_epoch).isoformat(),
        "start_epoch": start_epoch,
        "duration_s": duration,
        "camera_fps": {
            "target": cfg.get("pose", "target_fps", default=30),
        },
        "recording_fps": _safe_float(frame_t.size / duration) if duration > 0 else None,
        "tracking_model_version": cfg.get("pose", "model", default="unknown"),
        "camera_calibration": {
            "extrinsics": cfg.path("calibration", "cameras", "extrinsics"),
            "frame_pair_tolerance_s": cfg.get("cameras", "frame_pair_tolerance_s", default=None),
        },
        "walking_condition": cfg.get("recording", "walking_condition", default=None),
        "surface_type": cfg.get("recording", "surface_type", default="treadmill"),
        "shoe_type": cfg.get("recording", "shoe_type", default=None),
        "processing_latency_ms": _safe_float(_finite_mean(frame_data["quality_processing_latency_ms"])),
        "paths": {
            "raw_force": "force.npz",
            "raw_pose": "pose.npz",
            "frame_data": "frame_data.npz",
            "step_events": "step_events.json",
            "derived_metrics": "derived_metrics.json",
            "pressure": "pressure_timeseries.npz",
            "exports": "exports",
        },
        "schema_version": "trial-recording-v1",
    }
    _write_json(meta_path, metadata)

    quality = {
        "tracking_confidence_mean": _safe_float(_finite_mean(tracking_conf)),
        "missing_joints_total": int(np.nansum(missing)) if missing.size else 0,
        "interpolated_frames": 0,
        "occluded_joints_total": int(np.nansum(occluded)) if occluded.size else 0,
        "dropped_frames": int(np.sum(dropped)),
        "reprojection_error_px_mean": _safe_float(_finite_mean(reproj)),
        "reprojection_error_px_p95": _safe_float(
            np.nanpercentile(reproj[np.isfinite(reproj)], 95)
            if np.any(np.isfinite(reproj)) else np.nan),
        "processing_latency_ms_mean": None,
    }
    joint_angles = _angle_table(pose, frame_t)
    summary = _summary(metadata, steps)
    derived = {
        "summary": summary,
        "joint_angles": joint_angles,
        "charts": {
            "walking_speed_vs_time": _series_from_steps(steps, "walking_speed_m_s", "temporal_metrics"),
            "cadence_vs_time": _series_from_steps(steps, "cadence_steps_per_min", "temporal_metrics"),
            "stride_length_vs_step": _series_from_steps(steps, "stride_length_m", "spatial_metrics", x="step_id"),
            "step_length_vs_step": _series_from_steps(steps, "step_length_m", "spatial_metrics", x="step_id"),
            "ground_contact_timeline": [
                {"timestamp": e["timestamp"], "side": e["side"], "type": e["type"]} for e in events
            ],
            "left_vs_right_symmetry": _symmetry(steps),
        },
    }
    _write_json(rec_dir / "quality_metrics.json", quality)
    _write_json(rec_dir / "step_events.json", {"events": events, "steps": steps})
    _write_json(rec_dir / "derived_metrics.json", derived)
    _write_json(rec_dir / "summary.json", summary)
    raw_files = [
        {"kind": "force", "path": "../force.npz"},
        {"kind": "pose_landmarks", "path": "../pose.npz"},
        {"kind": "metrics_stream", "path": "../live_metrics.csv"},
    ]
    if (rec_dir / "cam0.mp4").exists():
        raw_files.append({"kind": "video_cam0", "path": "../cam0.mp4"})
    if (rec_dir / "cam1.mp4").exists():
        raw_files.append({"kind": "video_cam1", "path": "../cam1.mp4"})
    _write_json(rec_dir / "raw" / "manifest.json", {
        "raw_files": raw_files,
        "note": "Derived artifacts can be regenerated from these raw capture files.",
    })
    _update_index(rec_dir.parent)
    return metadata


def _series_from_steps(steps: list[dict], metric: str, group: str, x: str = "timestamp") -> list[dict]:
    out = []
    for s in steps:
        val = s.get(group, {}).get(metric)
        if val is not None:
            out.append({"x": s.get(x), "value": val, "side": s.get("side"), "step_id": s.get("step_id")})
    return out


def _symmetry(steps: list[dict]) -> dict[str, Any]:
    def means(group, metric):
        l = _finite_mean([s[group].get(metric) for s in steps if s.get("side") == "L"])
        r = _finite_mean([s[group].get(metric) for s in steps if s.get("side") == "R"])
        if not np.isfinite(l) or not np.isfinite(r) or (l + r) == 0:
            pct = np.nan
        else:
            pct = 2 * abs(l - r) / (l + r) * 100
        return {"left": _safe_float(l), "right": _safe_float(r), "asymmetry_pct": _safe_float(pct)}
    return {
        "stride_length": means("spatial_metrics", "stride_length_m"),
        "step_length": means("spatial_metrics", "step_length_m"),
        "stance_time": means("temporal_metrics", "stance_time_s"),
        "swing_time": means("temporal_metrics", "swing_time_s"),
    }


def _update_index(recordings_dir: Path):
    trials = []
    for meta_path in sorted(recordings_dir.glob("*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            summary_path = meta_path.parent / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            trials.append({
                "trial_id": meta.get("trial_id"),
                "session_id": meta.get("session_id"),
                "recording_name": meta.get("recording_name") or meta_path.parent.name,
                "date_time": meta.get("date_time"),
                "path": meta_path.parent.name,
                "duration_s": summary.get("recording_duration_s", meta.get("duration_s")),
                "total_steps": summary.get("total_steps"),
                "distance_m": summary.get("total_distance_walked_m"),
                "average_speed_m_s": summary.get("average_walking_speed_m_s"),
                "average_cadence_steps_per_min": summary.get("average_cadence_steps_per_min"),
            })
        except Exception:
            continue
    _write_json(recordings_dir / "trials_index.json", {"trials": trials})


def copy_raw_to_archive(recording_dir: Path, archive_dir: Path):
    """Copy raw capture files into a binary/archive staging folder."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in ("force.npz", "pose.npz", "cam0.mp4", "cam1.mp4", "live_metrics.csv"):
        src = recording_dir / name
        if src.exists():
            shutil.copy2(src, archive_dir / name)
