"""OpenCV renderers for the Trials review page."""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from ..gui.render import (ACCENT, BG, CENTER_C, FG, GRID, LEFT_C, MUTED, OK_C,
                          PANEL, RIGHT_C, WARN, _fmt, _fit_image,
                          _panel, _text)
from ..pose.detector import IDX, SKELETON
from ..state import ANGLE_NAMES
from .manager import Trial

_VIDEO_CACHE = {}


def _read_video_frame(path: Path, frame_idx: int):
    key = str(path)
    if not path.exists():
        return None
    cap = _VIDEO_CACHE.get(key)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        _VIDEO_CACHE[key] = cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ok, frame = cap.read()
    return frame if ok and frame is not None else None


def _camera_keypoint_score(kp):
    arr = np.asarray(kp, float)
    if arr.ndim != 3 or arr.shape[-1] < 3 or arr.shape[0] == 0:
        return -1.0
    body = arr[:, IDX["Lshoulder"]:, :]
    finite = np.isfinite(body[:, :, 0]) & np.isfinite(body[:, :, 1])
    conf = np.where(np.isfinite(body[:, :, 2]), body[:, :, 2], 0.0)
    return float(np.nanmean(finite.astype(float)) + np.nanmean(conf))


def _replay_camera_id(trial: Trial, requested="auto"):
    if str(requested).lower() in ("cam0", "0"):
        return 0
    if str(requested).lower() in ("cam1", "1"):
        return 1
    pose = trial.landmarks
    return 0 if _camera_keypoint_score(pose.get("kp0")) >= _camera_keypoint_score(pose.get("kp1")) else 1


def _camera_scene_skeleton(img, x, y, w, h, trial: Trial, frame_idx: int,
                           angles: dict, replay_camera="auto", show_labels=True):
    cam_id = _replay_camera_id(trial, replay_camera)
    _panel(img, x, y, w, h, f"Camera skeleton replay cam{cam_id}")
    frame = _read_video_frame(trial.path / f"cam{cam_id}.mp4", frame_idx)
    px, py, pw, ph = x + 8, y + 26, w - 16, h - 34
    if frame is None:
        _draw_skeleton(img, x, y, w, h, trial.frames, frame_idx)
        return
    canvas, scale, ox, oy = _fit_image_with_transform_for_trial(frame, pw, ph)
    pose = trial.landmarks
    key = "kp0" if cam_id == 0 else "kp1"
    kp_all = np.asarray(pose.get(key, np.zeros((0, 17, 3))), float)
    kp = kp_all[frame_idx].copy() if kp_all.ndim == 3 and frame_idx < kp_all.shape[0] else None
    if kp is not None:
        good = np.isfinite(kp[:, 0]) & np.isfinite(kp[:, 1])
        kp[good, 0] = kp[good, 0] * scale + ox
        kp[good, 1] = kp[good, 1] * scale + oy
        _draw_line_skeleton_overlay(canvas, kp, angles,
                                    show_labels=(show_labels and pw >= 520 and ph >= 340))
    img[py:py + ph, px:px + pw] = canvas


def _side_color(tag):
    return {"L": LEFT_C, "R": RIGHT_C, "C": CENTER_C}.get(tag, CENTER_C)


def _joint_confidence(p):
    return float(np.clip(p[2], 0.0, 1.0)) if len(p) > 2 and np.isfinite(p[2]) else 0.0


def _quality_color(tag, q):
    base = np.asarray(_side_color(tag), dtype=float)
    low = np.asarray(WARN, dtype=float)
    t = np.clip((q - 0.20) / 0.65, 0.0, 1.0)
    return tuple(int(v) for v in low * (1.0 - t) + base * t)


def _draw_line_skeleton_overlay(frame, kp, angles=None, show_labels=True):
    if kp is None:
        return frame
    for an, bn, tag in SKELETON:
        a, b = kp[IDX[an]], kp[IDX[bn]]
        if np.all(np.isfinite(a[:2])) and np.all(np.isfinite(b[:2])):
            q = min(_joint_confidence(a), _joint_confidence(b))
            c = _quality_color(tag, q)
            p0 = (int(round(a[0])), int(round(a[1])))
            p1 = (int(round(b[0])), int(round(b[1])))
            cv2.line(frame, p0, p1, (5, 5, 6), 7, cv2.LINE_AA)
            cv2.line(frame, p0, p1, c, 4, cv2.LINE_AA)
    for name, i in IDX.items():
        p = kp[i]
        if i >= IDX["Lshoulder"] and np.all(np.isfinite(p[:2])):
            tag = "L" if name.startswith("L") else ("R" if name.startswith("R") else "C")
            q = _joint_confidence(p)
            c = _quality_color(tag, q)
            xy = (int(round(p[0])), int(round(p[1])))
            cv2.circle(frame, xy, 8, (5, 5, 6), -1, cv2.LINE_AA)
            cv2.circle(frame, xy, 6, c, -1, cv2.LINE_AA)
            cv2.circle(frame, xy, 8, FG, 1, cv2.LINE_AA)
    if angles and show_labels:
        for jn, key, label in [
            ("Lhip", "hip_flexion_l", "L hip"),
            ("Rhip", "hip_flexion_r", "R hip"),
            ("Lknee", "knee_angle_l", "L knee"),
            ("Rknee", "knee_angle_r", "R knee"),
        ]:
            p = kp[IDX[jn]]
            v = angles.get(key)
            if np.all(np.isfinite(p[:2])) and v is not None and np.isfinite(v):
                org = (int(p[0]) + 10, int(p[1]) - 8)
                _text(frame, f"{label} {v:.0f}", (org[0] + 1, org[1] + 1), 0.55, (5, 5, 6), 3)
                _text(frame, f"{label} {v:.0f}", org, 0.55,
                      LEFT_C if label.startswith("L") else RIGHT_C, 1)
    return frame


def _fit_image_with_transform_for_trial(frame, w, h):
    canvas = np.full((h, w, 3), (12, 12, 14), np.uint8)
    fh, fw = frame.shape[:2]
    scale = min(w / max(1, fw), h / max(1, fh))
    nw, nh = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas, scale, x0, y0


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


def _foot_trajectories(img, x, y, w, h, title, P):
    _panel(img, x, y, w, h, title)
    if P is None or not np.asarray(P).size:
        _text(img, "no trajectory", (x + 12, y + h // 2), 0.42, MUTED, 1)
        return
    left = np.asarray(P[:, IDX["Lankle"], :], float)
    right = np.asarray(P[:, IDX["Rankle"], :], float)
    all_pts = np.vstack([left[:, [0, 2]], right[:, [0, 2]]])
    good_all = np.isfinite(all_pts[:, 0]) & np.isfinite(all_pts[:, 1])
    if np.sum(good_all) < 2:
        return
    mn, mx = np.nanmin(all_pts[good_all], axis=0), np.nanmax(all_pts[good_all], axis=0)
    px, py, pw, ph = x + 10, y + 30, w - 20, h - 40

    def draw(points, color):
        q = np.asarray(points[:, [0, 2]], float)
        good = np.isfinite(q[:, 0]) & np.isfinite(q[:, 1])
        if np.sum(good) < 2:
            return
        draw_pts = []
        for a, b in q[good]:
            u = int(px + (a - mn[0]) / (mx[0] - mn[0] + 1e-9) * pw)
            v = int(py + ph - (b - mn[1]) / (mx[1] - mn[1] + 1e-9) * ph)
            draw_pts.append((u, v))
        cv2.polylines(img, [np.asarray(draw_pts, np.int32)], False, color, 2, cv2.LINE_AA)

    draw(left, LEFT_C)
    draw(right, RIGHT_C)


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


def _plot_points_from_steps(steps, group, key):
    return ([s.get("timestamp") for s in steps],
            [s.get(group, {}).get(key) for s in steps])


def _chart_view(img, x, y, w, h, trial: Trial, chart_name: str):
    chart_name = chart_name or "Overview"
    if chart_name == "Walking speed vs time":
        xs, ys = _plot_points_from_steps(trial.steps, "temporal_metrics", "walking_speed_m_s")
        _line_plot(img, x, y, w, h, chart_name, xs, ys, OK_C, 0, None)
    elif chart_name == "Cadence vs time":
        xs, ys = _plot_points_from_steps(trial.steps, "temporal_metrics", "cadence_steps_per_min")
        _line_plot(img, x, y, w, h, chart_name, xs, ys, ACCENT, 0, None)
    elif chart_name == "Stride length vs step":
        _line_plot(img, x, y, w, h, chart_name, np.arange(len(trial.steps)),
                   [s.get("spatial_metrics", {}).get("stride_length_m") for s in trial.steps],
                   FG, 0, None)
    elif chart_name == "Step length vs step":
        _line_plot(img, x, y, w, h, chart_name, np.arange(len(trial.steps)),
                   [s.get("spatial_metrics", {}).get("step_length_m") for s in trial.steps],
                   FG, 0, None)
    elif chart_name == "Joint angles over time":
        joint = trial.jointAngles
        angle_t = np.asarray(joint.get("time_s", []), float)
        angles = joint.get("angles_deg", {})
        _line_plot(img, x, y, w, h, chart_name,
                   angle_t, np.asarray(angles.get("knee_flexion_l", []), float),
                   LEFT_C, -20, 100)
        _line_plot(img, x, y + h // 2, w, h // 2, "Right knee flexion",
                   angle_t, np.asarray(angles.get("knee_flexion_r", []), float),
                   RIGHT_C, -20, 100)
    elif chart_name == "Center of mass trajectory":
        _trajectory(img, x, y, w, h, chart_name, trial.frames.get("center_of_mass"), CENTER_C)
    elif chart_name == "Foot trajectories":
        P = trial.frames.get("world_positions")
        _foot_trajectories(img, x, y, w, h, chart_name, P)
    elif chart_name == "Pressure over time":
        p = trial.pressure
        t = np.asarray(p.get("time_s", []), float)
        _line_plot(img, x, y, w, h, chart_name, t, np.asarray(p.get("L_Fz", []), float), LEFT_C, 0, None)
        _line_plot(img, x, y + h // 2, w, h // 2, "Right GRF", t, np.asarray(p.get("R_Fz", []), float), RIGHT_C, 0, None)
    elif chart_name == "Ground contact timeline":
        _timeline(img, x, y, w, h, trial, 0.0)
    elif chart_name == "Left vs right symmetry":
        sym = trial.derivedMetrics.get("charts", {}).get("left_vs_right_symmetry", {})
        _panel(img, x, y, w, h, chart_name)
        yy = y + 52
        for metric, vals in sym.items():
            _text(img, metric.replace("_", " "), (x + 22, yy), 0.52, FG, 1)
            _text(img, f"L {_fmt(vals.get('left'), 2)}", (x + 260, yy), 0.52, LEFT_C, 1)
            _text(img, f"R {_fmt(vals.get('right'), 2)}", (x + 390, yy), 0.52, RIGHT_C, 1)
            _text(img, f"asym {_fmt(vals.get('asymmetry_pct'), 1, '%')}", (x + 530, yy), 0.52, ACCENT, 1)
            yy += 34


def _frame_angle_labels(trial: Trial, frame_idx: int) -> dict:
    joint = trial.jointAngles
    angles = joint.get("angles_deg", {}) or {}

    def at(name):
        arr = np.asarray(angles.get(name, []), float)
        return float(arr[frame_idx]) if frame_idx < arr.size and np.isfinite(arr[frame_idx]) else np.nan

    return {
        "hip_flexion_l": at("hip_flexion_l"),
        "hip_flexion_r": at("hip_flexion_r"),
        "knee_angle_l": at("knee_flexion_l"),
        "knee_angle_r": at("knee_flexion_r"),
    }


def build_trial_review(trial: Trial | None, frame_idx: int = 0, size=(1600, 900),
                       chart_name: str = "Overview", replay_camera: str = "auto"):
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
    frame_angles = _frame_angle_labels(trial, frame_idx)
    if chart_name == "Camera skeleton replay":
        _camera_scene_skeleton(img, 12, 12, W - 24, H - 24, trial,
                               frame_idx, frame_angles, replay_camera,
                               show_labels=True)
        duration = summ.get("recording_duration_s") or (float(np.nanmax(ts)) if ts.size else 0.0)
        cv2.rectangle(img, (24, H - 54), (W - 24, H - 18), (12, 12, 14), -1)
        _text(img, f"{meta.get('recording_name', trial.path.name)}  {frame_time:.2f}s / {duration:.2f}s",
              (36, H - 30), 0.62, FG, 2)
        return img
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
    if chart_name and chart_name != "Overview":
        _camera_scene_skeleton(img, 20, 240, 360, 310, trial, frame_idx,
                               frame_angles, replay_camera, show_labels=False)
        _pressure_heatmap(img, 396, 240, 260, 310, trial, frame_time)
        _chart_view(img, 672, 240, W - 700, 558, trial, chart_name)
        return img

    left_x, top_y = 20, 240
    gap = 16
    top_h = max(340, min(430, int(H * 0.42)))
    pressure_w = max(180, min(300, int(W * 0.20)))
    right_min = 260
    max_left = max(260, W - 40 - pressure_w - right_min - 2 * gap)
    left_w = max(300, min(max_left, int(W * 0.46)))
    pressure_x = left_x + left_w + gap
    right_x = pressure_x + pressure_w + gap
    right_w = max(240, W - right_x - 20)

    _camera_scene_skeleton(img, left_x, top_y, left_w, top_h, trial, frame_idx,
                           frame_angles, replay_camera, show_labels=False)
    _pressure_heatmap(img, pressure_x, top_y, pressure_w, top_h, trial, frame_time)

    joint = trial.jointAngles
    angle_t = np.asarray(joint.get("time_s", []), float)
    angles = joint.get("angles_deg", {})
    if right_w >= 700:
        plot_gap = 16
        plot_w = (right_w - plot_gap) // 2
        _line_plot(img, right_x, 240, plot_w, 145, "Joint angles over time",
                   angle_t, np.asarray(angles.get("knee_flexion_l", []), float), LEFT_C, -20, 100)
        _line_plot(img, right_x, 405, plot_w, 145, "Walking speed vs time",
                   [s.get("timestamp") for s in trial.steps],
                   [s.get("temporal_metrics", {}).get("walking_speed_m_s") for s in trial.steps],
                   OK_C, 0, None)
        _line_plot(img, right_x + plot_w + plot_gap, 240, plot_w, 145, "Cadence vs time",
                   [s.get("timestamp") for s in trial.steps],
                   [s.get("temporal_metrics", {}).get("cadence_steps_per_min") for s in trial.steps],
                   ACCENT, 0, None)
        _line_plot(img, right_x + plot_w + plot_gap, 405, plot_w, 145, "Stride length vs step",
                   np.arange(len(trial.steps)),
                   [s.get("spatial_metrics", {}).get("stride_length_m") for s in trial.steps],
                   FG, 0, None)
        plot_bottom = 550
    else:
        plot_h = max(70, min(104, (top_h - 30) // 4))
        y = 240
        _line_plot(img, right_x, y, right_w, plot_h, "Joint angles over time",
                   angle_t, np.asarray(angles.get("knee_flexion_l", []), float), LEFT_C, -20, 100)
        y += plot_h + 10
        _line_plot(img, right_x, y, right_w, plot_h, "Walking speed vs time",
                   [s.get("timestamp") for s in trial.steps],
                   [s.get("temporal_metrics", {}).get("walking_speed_m_s") for s in trial.steps],
                   OK_C, 0, None)
        y += plot_h + 10
        _line_plot(img, right_x, y, right_w, plot_h, "Cadence vs time",
                   [s.get("timestamp") for s in trial.steps],
                   [s.get("temporal_metrics", {}).get("cadence_steps_per_min") for s in trial.steps],
                   ACCENT, 0, None)
        y += plot_h + 10
        _line_plot(img, right_x, y, right_w, plot_h, "Stride length vs step",
                   np.arange(len(trial.steps)),
                   [s.get("spatial_metrics", {}).get("stride_length_m") for s in trial.steps],
                   FG, 0, None)
        plot_bottom = y + plot_h

    P = frames.get("world_positions")
    lower_y = max(top_y + top_h + 18, min(H - 150, plot_bottom + 16))
    if P is not None and np.asarray(P).size:
        _trajectory(img, left_x, lower_y, left_w, max(160, H - lower_y - 18), "Center of mass trajectory",
                    frames.get("center_of_mass"), CENTER_C)
        _foot_trajectories(img, pressure_x, lower_y, pressure_w, max(160, H - lower_y - 18), "Foot trajectories", P)

    _panel(img, right_x, lower_y, right_w, max(160, H - lower_y - 18), "Step table")
    for i, step in enumerate(trial.steps[:8]):
        yy = lower_y + 36 + i * 22
        color = LEFT_C if step.get("side") == "L" else RIGHT_C
        _text(img, step.get("step_id", ""), (right_x + 18, yy), 0.38, color, 1)
        _text(img, step.get("foot", ""), (right_x + 118, yy), 0.38, FG, 1)
        _text(img, _fmt(step.get("temporal_metrics", {}).get("stance_time_s"), 2, "s"),
              (right_x + 228, yy), 0.38, FG, 1)
        _text(img, _fmt(step.get("spatial_metrics", {}).get("stride_length_m"), 2, "m"),
              (right_x + 338, yy), 0.38, FG, 1)
        _text(img, _fmt(step.get("pressure_data", {}).get("peak_pressure_pa"), 0, "Pa"),
              (right_x + 458, yy), 0.38, FG, 1)
    return img


def _angle_dict(pose: dict, idx: int) -> dict:
    names = [str(x) for x in pose.get("angle_names", np.array(ANGLE_NAMES)).tolist()]
    arr = np.asarray(pose.get("angles", np.zeros((0, 0))), float)
    if arr.ndim != 2 or idx >= arr.shape[0]:
        return {}
    return {name: float(arr[idx, i]) for i, name in enumerate(names)
            if i < arr.shape[1] and np.isfinite(arr[idx, i])}


def _overlay_video_frame(frame, kp, angles, label):
    if frame is None:
        frame = np.full((720, 1280, 3), BG, np.uint8)
    out = frame.copy()
    draw_precision_pose_overlay(out, kp, angles)
    _text(out, label, (12, out.shape[0] - 14), 0.55, FG, 1)
    return out


def render_processed_playback(recording_dir: str | Path, cfg, max_width: int = 1600):
    """Render synchronized camera playback with skeleton overlay.

    The recorder stores one video frame per processed pose frame, not
    necessarily one frame per 1/30 s of wall time. Render the output on a
    regular 30 fps timeline and hold the nearest recorded frame so playback
    duration matches the trial timestamps instead of being compressed.
    """
    rec_dir = Path(recording_dir)
    trial = Trial(rec_dir)
    pose = trial.landmarks
    t_epoch = np.asarray(pose.get("t", []), float)
    if t_epoch.size == 0:
        return None
    start_epoch = float(t_epoch[0])
    with np.load(rec_dir / "pose.npz", allow_pickle=True) as z:
        if "start_epoch" in z:
            start_epoch = float(np.asarray(z["start_epoch"]).item())
        pose_npz = {k: z[k] for k in z.files}
    t = t_epoch - start_epoch
    t = t - (float(t[0]) if t.size else 0.0)
    kp0 = np.asarray(pose.get("kp0", np.zeros((t.size, 17, 3))), float)
    kp1 = np.asarray(pose.get("kp1", np.zeros_like(kp0)), float)
    raw_pose = trial.rawForce  # touch raw data lazily before long video work
    _ = raw_pose.get("START")

    caps = []
    labels = []
    for cam_id in (0, 1):
        path = rec_dir / f"cam{cam_id}.mp4"
        cap = cv2.VideoCapture(str(path)) if path.exists() else None
        if cap is not None and cap.isOpened():
            caps.append(cap)
            labels.append(f"cam{cam_id}")
        else:
            if cap is not None:
                cap.release()
            caps.append(None)
            labels.append(f"cam{cam_id} unavailable")
    if caps[0] is None and caps[1] is None:
        out_dir = rec_dir / "playback"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "skeleton_overlay.mp4"
        fps = max(30.0, float(cfg.get("pose", "target_fps", default=30) or 30))
        duration = float(np.nanmax(t)) if t.size else 0.0
        n_out = max(len(t), int(np.ceil(duration * fps)) + 1)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (max_width, 900))
        for out_i in range(n_out):
            out_t = out_i / fps
            i = int(np.clip(np.searchsorted(t, out_t, side="left"), 0, len(t) - 1))
            if i > 0 and abs(t[i - 1] - out_t) <= abs(t[i] - out_t):
                i -= 1
            writer.write(build_trial_review(trial, i, size=(max_width, 900)))
        writer.release()
        return out_path

    fps = max(30.0, float(cfg.get("pose", "target_fps", default=30) or 30))
    panel_w = max_width // 2
    panel_h = int(round(panel_w * 9 / 16))
    out_size = (panel_w * 2, panel_h)
    out_dir = rec_dir / "playback"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "skeleton_overlay.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    duration = float(np.nanmax(t)) if t.size else 0.0
    n_out = max(len(t), int(np.ceil(duration * fps)) + 1)
    frame_cache = {}
    for out_i in range(n_out):
        out_t = out_i / fps
        i = int(np.clip(np.searchsorted(t, out_t, side="left"), 0, len(t) - 1))
        if i > 0 and abs(t[i - 1] - out_t) <= abs(t[i] - out_t):
            i -= 1
        panels = []
        for cam_id, cap in enumerate(caps):
            frame = frame_cache.get(cam_id)
            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if not ok:
                    frame = frame_cache.get(cam_id)
                elif frame is not None:
                    frame_cache[cam_id] = frame
            kp = kp0[i] if cam_id == 0 and i < len(kp0) else (kp1[i] if i < len(kp1) else None)
            panel = _overlay_video_frame(frame, kp, _angle_dict(pose_npz, i), labels[cam_id])
            panels.append(_fit_image(panel, panel_w, panel_h))
        writer.write(np.hstack(panels))
    writer.release()
    for cap in caps:
        if cap is not None:
            cap.release()
    return out_path
