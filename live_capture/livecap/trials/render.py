"""OpenCV renderers for the Trials review page."""
from __future__ import annotations

import cv2
import numpy as np

from ..gui.render import (ACCENT, BG, CENTER_C, FG, GRID, LEFT_C, MUTED, OK_C,
                          PANEL, RIGHT_C, WARN, _fmt, _panel, _text)
from ..pose.detector import IDX, SKELETON
from .manager import Trial


def _line_plot(img, x, y, w, h, title, xs, ys, color, ymin=None, ymax=None):
    _panel(img, x, y, w, h, title)
    px, py, pw, ph = x + 8, y + 28, w - 16, h - 38
    cv2.rectangle(img, (px, py), (px + pw, py + ph), BG, -1)
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    good = np.isfinite(xs) & np.isfinite(ys)
    if np.sum(good) < 2:
        _text(img, "no data", (px + 8, py + ph // 2), 0.42, MUTED, 1)
        return
    xs, ys = xs[good], ys[good]
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin = float(np.nanmin(ys)) if ymin is None else ymin
    ymax = float(np.nanmax(ys)) if ymax is None else ymax
    if abs(ymax - ymin) < 1e-9:
        ymax = ymin + 1.0
    pts = []
    for xx, yy in zip(xs, ys):
        u = int(px + (xx - xmin) / (xmax - xmin + 1e-9) * pw)
        v = int(py + ph - (yy - ymin) / (ymax - ymin + 1e-9) * ph)
        pts.append((u, v))
    for gv in np.linspace(ymin, ymax, 3):
        vv = int(py + ph - (gv - ymin) / (ymax - ymin + 1e-9) * ph)
        cv2.line(img, (px, vv), (px + pw, vv), GRID, 1)
    cv2.polylines(img, [np.asarray(pts, np.int32)], False, color, 2, cv2.LINE_AA)


def _draw_skeleton(img, x, y, w, h, frames, frame_idx):
    _panel(img, x, y, w, h, "Skeleton replay")
    P = frames.get("world_positions")
    valid = np.isfinite(P[frame_idx, :, 0]) if P is not None and P.size else None
    if P is None or not P.size or valid is None or not np.any(valid):
        _text(img, "3D skeleton unavailable", (x + 16, y + h // 2), 0.45, MUTED, 1)
        return
    pts = P[frame_idx]
    center = np.nanmean(pts[valid], axis=0)
    span = np.nanmax(np.linalg.norm(pts[valid] - center, axis=1)) + 1e-6
    scale = min(w, h) / (2.4 * span)
    cx, cy = x + w // 2, y + h // 2 + 20

    def proj(p):
        return int(cx + (p[0] - center[0]) * scale), int(cy - (p[1] - center[1]) * scale)
    for a, b, tag in SKELETON:
        ia, ib = IDX[a], IDX[b]
        if valid[ia] and valid[ib]:
            color = LEFT_C if tag == "L" else RIGHT_C if tag == "R" else CENTER_C
            cv2.line(img, proj(pts[ia]), proj(pts[ib]), color, 3, cv2.LINE_AA)
    for name, i in IDX.items():
        if valid[i] and i >= IDX["Lshoulder"]:
            color = LEFT_C if name.startswith("L") else RIGHT_C if name.startswith("R") else CENTER_C
            cv2.circle(img, proj(pts[i]), 5, color, -1, cv2.LINE_AA)


def _trajectory(img, x, y, w, h, title, points, color):
    _panel(img, x, y, w, h, title)
    if points is None or not np.asarray(points).size:
        _text(img, "no trajectory", (x + 12, y + h // 2), 0.42, MUTED, 1)
        return
    pts = np.asarray(points, float)
    good = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 2])
    if np.sum(good) < 2:
        return
    q = pts[good][:, [0, 2]]
    mn, mx = np.nanmin(q, axis=0), np.nanmax(q, axis=0)
    px, py, pw, ph = x + 10, y + 30, w - 20, h - 40
    draw = []
    for a, b in q:
        u = int(px + (a - mn[0]) / (mx[0] - mn[0] + 1e-9) * pw)
        v = int(py + ph - (b - mn[1]) / (mx[1] - mn[1] + 1e-9) * ph)
        draw.append((u, v))
    cv2.polylines(img, [np.asarray(draw, np.int32)], False, color, 2, cv2.LINE_AA)


def _pressure_heatmap(img, x, y, w, h, trial: Trial, frame_time: float):
    _panel(img, x, y, w, h, "Pressure heatmap")
    pressure = trial.pressure
    t = pressure.get("time_s", np.zeros(0))
    if not len(t):
        _text(img, "pressure unavailable", (x + 12, y + h // 2), 0.42, MUTED, 1)
        return
    j = int(np.argmin(np.abs(np.asarray(t) - frame_time)))
    vals = [
        float(pressure.get("L_Fz", np.zeros_like(t))[j]),
        float(pressure.get("R_Fz", np.zeros_like(t))[j]),
    ]
    vmax = max(max(vals), 1.0)
    for i, (label, val, color) in enumerate([("L", vals[0], LEFT_C), ("R", vals[1], RIGHT_C)]):
        cx = x + w // 4 + i * w // 2
        cy = y + h // 2 + 16
        intensity = int(np.clip(val / vmax, 0.0, 1.0) * 255)
        heat = (min(255, color[0] + intensity // 4), min(255, color[1] + intensity // 4), min(255, color[2]))
        cv2.ellipse(img, (cx, cy), (34, 78), 0, 0, 360, heat, -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), (34, 78), 0, 0, 360, GRID, 2, cv2.LINE_AA)
        _text(img, f"{label} {_fmt(val, 0, ' N')}", (cx - 42, y + 34), 0.42, FG, 1)


def _timeline(img, x, y, w, h, trial: Trial, frame_time: float):
    _panel(img, x, y, w, h, "Interactive timeline")
    events = trial.events
    summary = trial.summary
    duration = summary.get("recording_duration_s") or 0.0
    px, py, pw = x + 16, y + h // 2, w - 32
    cv2.line(img, (px, py), (px + pw, py), GRID, 2)
    for ev in events:
        t = ev.get("timestamp", 0)
        u = int(px + (t / (duration + 1e-9)) * pw) if duration else px
        color = LEFT_C if ev.get("side") == "L" else RIGHT_C
        cv2.line(img, (u, py - 18), (u, py + 18), color, 1)
    cur = int(px + (frame_time / (duration + 1e-9)) * pw) if duration else px
    cv2.line(img, (cur, py - 28), (cur, py + 28), WARN, 2)
    _text(img, f"{frame_time:.2f}s / {duration:.2f}s", (px, y + h - 10), 0.42, FG, 1)


def build_trial_review(trial: Trial | None, frame_idx: int = 0, size=(1600, 900)):
    W, H = size
    img = np.full((H, W, 3), BG, np.uint8)
    if trial is None:
        _text(img, "Select a trial to inspect, replay, export, or compare.", (40, H // 2), 0.7, MUTED, 1)
        return img
    frames = trial.frames
    ts = np.asarray(frames.get("timestamp", np.zeros(0)), float)
    if ts.size:
        frame_idx = int(np.clip(frame_idx, 0, ts.size - 1))
        frame_time = float(ts[frame_idx])
    else:
        frame_idx, frame_time = 0, 0.0
    meta = trial.metadata
    summ = trial.summary
    _text(img, meta.get("recording_name", trial.path.name), (24, 34), 0.8, FG, 2)
    _text(img, meta.get("date_time", ""), (24, 58), 0.42, MUTED, 1)
    stats = [
        ("Steps", summ.get("total_steps")),
        ("Distance", _fmt(summ.get("total_distance_walked_m"), 2, " m")),
        ("Speed", _fmt(summ.get("average_walking_speed_m_s"), 2, " m/s")),
        ("Cadence", _fmt(summ.get("average_cadence_steps_per_min"), 0, "/min")),
        ("Duration", _fmt(summ.get("recording_duration_s"), 1, " s")),
    ]
    for i, (k, v) in enumerate(stats):
        x = 24 + i * 180
        _text(img, k, (x, 96), 0.42, MUTED, 1)
        _text(img, str(v), (x, 122), 0.62, ACCENT, 2)

    _timeline(img, 20, 140, W - 40, 86, trial, frame_time)
    _draw_skeleton(img, 20, 240, 360, 310, frames, frame_idx)
    _pressure_heatmap(img, 396, 240, 260, 310, trial, frame_time)

    joint = trial.jointAngles
    angle_t = np.asarray(joint.get("time_s", []), float)
    angles = joint.get("angles_deg", {})
    _line_plot(img, 672, 240, 430, 145, "Joint angles over time",
               angle_t, np.asarray(angles.get("knee_flexion_l", []), float), LEFT_C, -20, 100)
    _line_plot(img, 672, 405, 430, 145, "Walking speed vs time",
               [s.get("timestamp") for s in trial.steps],
               [s.get("temporal_metrics", {}).get("walking_speed_m_s") for s in trial.steps],
               OK_C, 0, None)
    _line_plot(img, 1118, 240, 430, 145, "Cadence vs time",
               [s.get("timestamp") for s in trial.steps],
               [s.get("temporal_metrics", {}).get("cadence_steps_per_min") for s in trial.steps],
               ACCENT, 0, None)
    _line_plot(img, 1118, 405, 430, 145, "Stride length vs step",
               np.arange(len(trial.steps)),
               [s.get("spatial_metrics", {}).get("stride_length_m") for s in trial.steps],
               FG, 0, None)

    P = frames.get("world_positions")
    if P is not None and np.asarray(P).size:
        _trajectory(img, 20, 568, 360, 230, "Center of mass trajectory",
                    frames.get("center_of_mass"), CENTER_C)
        _trajectory(img, 396, 568, 260, 230, "Foot trajectories",
                    P[:, IDX["Lankle"], :], LEFT_C)
        _trajectory(img, 396, 568, 260, 230, "Foot trajectories",
                    P[:, IDX["Rankle"], :], RIGHT_C)

    _panel(img, 672, 568, 876, 230, "Step table")
    for i, step in enumerate(trial.steps[:8]):
        yy = 604 + i * 22
        color = LEFT_C if step.get("side") == "L" else RIGHT_C
        _text(img, step.get("step_id", ""), (690, yy), 0.38, color, 1)
        _text(img, step.get("foot", ""), (790, yy), 0.38, FG, 1)
        _text(img, _fmt(step.get("temporal_metrics", {}).get("stance_time_s"), 2, "s"),
              (900, yy), 0.38, FG, 1)
        _text(img, _fmt(step.get("spatial_metrics", {}).get("stride_length_m"), 2, "m"),
              (1010, yy), 0.38, FG, 1)
        _text(img, _fmt(step.get("pressure_data", {}).get("peak_pressure_pa"), 0, "Pa"),
              (1130, yy), 0.38, FG, 1)
    return img
