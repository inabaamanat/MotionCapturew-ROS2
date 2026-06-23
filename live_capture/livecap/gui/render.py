"""Composite live dashboard, drawn with OpenCV from the SharedState.

Produces one BGR image combining: both camera feeds with the on-body 2D
skeleton + joint-angle labels, scrolling GRF and joint-angle plots, a COP belt
map, front/side 3D-skeleton orthographic views, live gait metrics, and a status
bar. The DearPyGui app displays this image as a texture; it is also renderable
standalone (save to PNG) for headless testing.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..pose.detector import IDX, SKELETON
from ..state import ANGLE_NAMES

BG = (24, 24, 28)
PANEL = (36, 36, 42)
FG = (230, 230, 235)
MUTED = (140, 140, 150)
LEFT_C = (90, 180, 255)
RIGHT_C = (120, 220, 120)
CENTER_C = (205, 205, 210)
ACCENT = (210, 160, 70)
OK_C = (120, 220, 120)
WARN = (70, 130, 240)
GRID = (60, 60, 68)
HANDLE_C = (255, 210, 105)
HANDLE_HI = (255, 235, 165)
FONT = cv2.FONT_HERSHEY_SIMPLEX
BONE = (224, 226, 220)
BONE_HI = (255, 255, 248)
BONE_EDGE = (125, 130, 130)
BONE_JOINT = (236, 238, 232)
GRID_BG = (14, 16, 19)
GRID_3D = (42, 54, 64)
FOCUS_KEYS = {
    "cam0", "cam1", "skeleton", "grf", "joint_plot", "cop",
    "treadmill", "metrics", "angles",
}


def _side_color(tag):
    return {"L": LEFT_C, "R": RIGHT_C, "C": CENTER_C}[tag]


def _text(img, s, org, sc=0.5, c=FG, th=1):
    cv2.putText(img, s, org, FONT, sc, c, th, cv2.LINE_AA)


def _panel(img, x, y, w, h, title=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), GRID, 1)
    if title:
        _text(img, title, (x + 8, y + 18), 0.5, ACCENT, 1)


def _fit_image(frame, w, h, bg=(12, 12, 14)):
    canvas = np.full((h, w, 3), bg, np.uint8)
    if frame is None:
        return canvas
    fh, fw = frame.shape[:2]
    scale = min(w / max(1, fw), h / max(1, fh))
    nw, nh = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _mid(a, b):
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _interp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _as_pt(p):
    return int(round(p[0])), int(round(p[1]))


def _bone_line(img, a, b, thickness=7, color=BONE):
    a, b = _as_pt(a), _as_pt(b)
    cv2.line(img, a, b, BONE_EDGE, thickness + 3, cv2.LINE_AA)
    cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
    cv2.line(img, a, b, BONE_HI, max(1, thickness // 3), cv2.LINE_AA)


def _double_bone(img, a, b, sep=8, thickness=4):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-6:
        _joint(img, a, thickness + 2)
        return
    ox, oy = -dy / n * sep * 0.5, dx / n * sep * 0.5
    _bone_line(img, (ax + ox, ay + oy), (bx + ox, by + oy), thickness)
    _bone_line(img, (ax - ox, ay - oy), (bx - ox, by - oy), thickness)


def _joint(img, p, r=6):
    c = _as_pt(p)
    cv2.circle(img, c, r + 2, BONE_EDGE, -1, cv2.LINE_AA)
    cv2.circle(img, c, r, BONE_JOINT, -1, cv2.LINE_AA)
    cv2.circle(img, c, max(1, r // 3), BONE_HI, -1, cv2.LINE_AA)


def _ellipse_bone(img, center, axes, angle=0, start=0, end=360, thickness=4):
    c = _as_pt(center)
    ax = max(1, int(round(axes[0])))
    ay = max(1, int(round(axes[1])))
    cv2.ellipse(img, c, (ax, ay), angle, start, end, BONE_EDGE, thickness + 3, cv2.LINE_AA)
    cv2.ellipse(img, c, (ax, ay), angle, start, end, BONE, thickness, cv2.LINE_AA)
    cv2.ellipse(img, c, (ax, ay), angle, start, end, BONE_HI, max(1, thickness // 3), cv2.LINE_AA)


def _pose_points_2d(kp):
    if kp is None:
        return None
    pts = []
    for name, i in IDX.items():
        p = kp[i]
        if i >= IDX["Lshoulder"] and np.all(np.isfinite(p[:2])):
            pts.append(p[:2])
    return np.asarray(pts, float) if pts else None


def _project_2d_pose(kp, x, y, w, h):
    pts = _pose_points_2d(kp)
    if pts is None:
        return None
    lo = np.nanmin(pts, axis=0)
    hi = np.nanmax(pts, axis=0)
    center = 0.5 * (lo + hi)
    span = np.maximum(hi - lo, 1.0)
    scale = min((w * 0.72) / span[0], (h * 0.80) / span[1])
    cx, cy = x + w * 0.50, y + h * 0.54

    def proj(p):
        u = cx + (p[0] - center[0]) * scale
        v = cy + (p[1] - center[1]) * scale
        return int(round(u)), int(round(v))

    return proj


def _safe(v):
    return v if (v is not None and np.isfinite(v)) else np.nan


def _fmt(v, digits=1, suffix=""):
    return f"{v:.{digits}f}{suffix}" if np.isfinite(_safe(v)) else "--"


def dashboard_layout(size, controls=None):
    W, H = size
    m, gap, status_h = 8, 8, 28
    controls = controls or {}
    content_h = max(1, H - status_h - 3 * m)
    left_frac = float(controls.get("left_frac", 0.18))
    right_frac = float(controls.get("right_frac", 0.30))
    left_min, right_min = 170, 280
    main_min = max(240, int(W * 0.22))
    available = max(1, W - 2 * m - 2 * gap)
    left_w = int(np.clip(W * left_frac, left_min, max(left_min, int(W * 0.36))))
    right_w = int(np.clip(W * right_frac, right_min, max(right_min, int(W * 0.42))))
    overflow = left_w + right_w + main_min - available
    if overflow > 0:
        take = min(overflow, max(0, right_w - right_min))
        right_w -= take
        overflow -= take
    if overflow > 0:
        take = min(overflow, max(0, left_w - left_min))
        left_w -= take
    main_x = m + left_w + gap
    right_x = W - right_w - m
    main_w = max(220, right_x - main_x - gap)
    cam_h = int(round(left_w * 9 / 16))
    cam0 = (m, m, left_w, cam_h + 28)
    cam1_y = m + cam_h + 36
    cam1 = (m, cam1_y, left_w, cam_h + 28)
    left_y = cam1_y + cam_h + 36
    treadmill_h = 140
    cop_h = max(150, content_h - (left_y - m) - treadmill_h - gap)
    rx, rw = right_x, right_w
    plot_h = max(160, min(190, int(content_h * 0.21)))
    y2 = m + plot_h + gap
    y3 = y2 + plot_h + gap
    metrics_h = max(160, min(210, int(content_h * 0.23)))
    y4 = y3 + metrics_h + gap
    return {
        "cam0": cam0,
        "cam1": cam1,
        "skeleton": (main_x, m, main_w, content_h),
        "cop": (m, left_y, left_w, cop_h),
        "treadmill": (m, left_y + cop_h + gap, left_w, treadmill_h),
        "grf": (rx, m, rw, plot_h),
        "joint_plot": (rx, y2, rw, plot_h),
        "metrics": (rx, y3, rw, metrics_h),
        "angles": (rx, y4, rw, max(132, m + content_h - y4)),
        "status": (m, H - status_h - m, W - 2 * m, status_h),
        "__split_left": (m + left_w + gap // 2, m, 0, content_h),
        "__split_right": (right_x - gap // 2, m, 0, content_h),
    }


def dashboard_hit_test(size, point, controls=None):
    x, y = point
    for key, (rx, ry, rw, rh) in dashboard_layout(size, controls).items():
        if key == "status" or key.startswith("__"):
            continue
        if rx <= x <= rx + rw and ry <= y <= ry + rh:
            return key
    return None


def dashboard_resize_hit_test(size, point, controls=None, tol=10):
    x, y = point
    layout = dashboard_layout(size, controls)
    for name in ("__split_left", "__split_right"):
        sx, sy, _, sh = layout[name]
        if sy <= y <= sy + sh and abs(x - sx) <= tol:
            return "left" if name.endswith("left") else "right"
    return None


def draw_resize_handles(img, layout):
    for name in ("__split_left", "__split_right"):
        sx, sy, _, sh = layout[name]
        y0, y1 = sy + 10, sy + sh - 10
        cv2.line(img, (sx, y0), (sx, y1), HANDLE_C, 2, cv2.LINE_AA)
        cy = sy + sh // 2
        cv2.circle(img, (sx, cy), 12, (45, 43, 36), -1, cv2.LINE_AA)
        cv2.circle(img, (sx, cy), 12, HANDLE_C, 1, cv2.LINE_AA)
        left_tri = np.asarray([(sx - 8, cy), (sx - 2, cy - 6), (sx - 2, cy + 6)], np.int32)
        right_tri = np.asarray([(sx + 8, cy), (sx + 2, cy - 6), (sx + 2, cy + 6)], np.int32)
        cv2.fillConvexPoly(img, left_tri, HANDLE_HI, cv2.LINE_AA)
        cv2.fillConvexPoly(img, right_tri, HANDLE_HI, cv2.LINE_AA)


# ---------- 2D skeleton on a camera frame ----------
def draw_skeleton_2d(frame, kp, angles):
    if kp is None:
        return frame
    ov = frame.copy()
    for an, bn, tag in SKELETON:
        a, b = kp[IDX[an]], kp[IDX[bn]]
        if np.all(np.isfinite(a[:2])) and np.all(np.isfinite(b[:2])):
            cv2.line(ov, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     _side_color(tag), 4, cv2.LINE_AA)
    for n, i in IDX.items():
        p = kp[i]
        if np.all(np.isfinite(p[:2])) and i >= IDX["Lshoulder"]:
            tag = "L" if n.startswith("L") else ("R" if n.startswith("R") else "C")
            cv2.circle(ov, (int(p[0]), int(p[1])), 5, _side_color(tag), -1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)

    if angles:
        for jn, key, tag in [("Lknee", "knee_angle_l", "L"), ("Rknee", "knee_angle_r", "R"),
                             ("Lhip", "hip_flexion_l", "L"), ("Rhip", "hip_flexion_r", "R")]:
            J = kp[IDX[jn]][:2]
            v = _safe(angles.get(key))
            if np.all(np.isfinite(J)) and np.isfinite(v):
                dx = 8 if tag == "R" else -52
                _text(frame, f"{v:.0f}", (int(J[0]) + dx, int(J[1])), 0.5,
                      _side_color(tag), 2)
    return frame


# ---------- scrolling plot ----------
def scrolling_plot(img, x, y, w, h, title, t, series, window_s, ymin, ymax, units=""):
    _panel(img, x, y, w, h, title)
    px, py, pw, ph = x + 6, y + 26, w - 12, h - 34
    cv2.rectangle(img, (px, py), (px + pw, py + ph), BG, -1)
    if t is None or np.sum(np.isfinite(t)) < 2:
        return
    good = np.isfinite(t)
    t = t[good]
    tmax = t[-1]
    t0 = tmax - window_s

    def yto(v):
        v = np.clip(v, ymin, ymax)
        return int(py + ph - (v - ymin) / (ymax - ymin + 1e-9) * ph)

    def xto(tt):
        return int(px + (tt - t0) / window_s * pw)

    for gv in np.linspace(ymin, ymax, 4):
        yy = yto(gv)
        cv2.line(img, (px, yy), (px + pw, yy), GRID, 1)
        _text(img, f"{gv:.0f}", (px + 2, yy - 2), 0.32, MUTED, 1)
    for si, (arr, color, lbl) in enumerate(series):
        if arr is None:
            continue
        arr = np.asarray(arr, float)[good]
        pts = [(xto(t[i]), yto(arr[i])) for i in range(len(t))
               if np.isfinite(arr[i]) and t[i] >= t0]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, color, 2, cv2.LINE_AA)
        cur = arr[-1] if arr.size and np.isfinite(arr[-1]) else np.nan
        legend = f"{lbl} {cur:.1f}{units}" if np.isfinite(cur) else f"{lbl} --"
        _text(img, legend, (px + 6 + si * 92, py + 12), 0.4, color, 1)


# ---------- COP belt map ----------
def cop_map(img, x, y, w, h, cop, approx=False):
    title = "Centre of Pressure" + (" (approx)" if approx else "")
    _panel(img, x, y, w, h, title)
    cx0, cy0 = x + w // 2, y + h // 2 + 4
    for sx, color, (copx, copy) in [(-1, LEFT_C, (cop[0], cop[1])),
                                    (1, RIGHT_C, (cop[2], cop[3]))]:
        bx = cx0 + sx * (w // 4)
        cv2.rectangle(img, (bx - 22, cy0 - 60), (bx + 22, cy0 + 60), GRID, 1)
        if np.isfinite(copx) and np.isfinite(copy):
            px = int(bx + np.clip(copx, -0.25, 0.25) / 0.25 * 20)
            py = int(cy0 - np.clip(copy, -0.5, 0.5) / 0.5 * 56)
            cv2.circle(img, (px, py), 6, color, -1)


# ---------- 3D orthographic skeleton ----------
def _body_axes(P, valid):
    needed = ("Lhip", "Rhip", "Lshoulder", "Rshoulder")
    if not all(valid[IDX[n]] and np.all(np.isfinite(P[IDX[n]])) for n in needed):
        return None
    hip_mid = 0.5 * (P[IDX["Lhip"]] + P[IDX["Rhip"]])
    sho_mid = 0.5 * (P[IDX["Lshoulder"]] + P[IDX["Rshoulder"]])
    up = sho_mid - hip_mid
    up_n = np.linalg.norm(up)
    ml = P[IDX["Rhip"]] - P[IDX["Lhip"]]
    ml_n = np.linalg.norm(ml)
    if up_n < 1e-9 or ml_n < 1e-9:
        return None
    up = up / up_n
    ml = ml - np.dot(ml, up) * up
    ml_n = np.linalg.norm(ml)
    if ml_n < 1e-9:
        return None
    ml = ml / ml_n
    anterior = np.cross(up, ml)
    ant_n = np.linalg.norm(anterior)
    if ant_n < 1e-9:
        return None
    return ml, up, anterior / ant_n


def draw_3d(img, x, y, w, h, kp3d, valid):
    _panel(img, x, y, w, h, "3D skeleton (front | side)")
    if kp3d is None or valid is None or not np.any(valid):
        _text(img, "needs stereo calibration", (x + 10, y + h // 2), 0.45, MUTED, 1)
        return
    P = kp3d
    pts = P[valid]
    c = np.nanmean(pts, axis=0)
    span = np.nanmax(np.linalg.norm(pts - c, axis=1)) + 1e-6
    axes = _body_axes(P, valid)
    if axes is None:
        ml = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, -1.0, 0.0])
        anterior = np.array([0.0, 0.0, 1.0])
    else:
        ml, up, anterior = axes
    views = [(x + 6, "front", ml), (x + w // 2 + 3, "side", anterior)]
    pw = w // 2 - 9
    for x0, name, h_axis in views:
        _text(img, name, (x0 + 4, y + 34), 0.4, MUTED, 1)
        cx, cy = x0 + pw // 2, y + h // 2 + 10
        sc = (h - 60) / (2.2 * span)

        def proj(p):
            q = p - c
            u = cx + np.dot(q, h_axis) * sc
            v = cy - np.dot(q, up) * sc
            return int(u), int(v)
        for an, bn, tag in SKELETON:
            a, b = P[IDX[an]], P[IDX[bn]]
            if valid[IDX[an]] and valid[IDX[bn]]:
                cv2.line(img, proj(a), proj(b), _side_color(tag), 2, cv2.LINE_AA)
        for n, i in IDX.items():
            if valid[i] and i >= IDX["Lshoulder"]:
                tag = "L" if n.startswith("L") else ("R" if n.startswith("R") else "C")
                cv2.circle(img, proj(P[i]), 3, _side_color(tag), -1)


def _draw_blank_3d_grid(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), GRID_BG, -1)
    horizon = y + int(h * 0.58)
    floor = y + h - 26
    vanish = (x + w // 2, horizon)
    for i in range(9):
        t = i / 8
        yy = int(horizon + (floor - horizon) * (t ** 1.7))
        cv2.line(img, (x + 22, yy), (x + w - 22, yy), GRID_3D, 1, cv2.LINE_AA)
    for i in range(-8, 9):
        bx = x + w // 2 + int(i * w * 0.075)
        cv2.line(img, (bx, floor), vanish, GRID_3D, 1, cv2.LINE_AA)
    for i in range(-4, 5):
        xx = x + w // 2 + int(i * w * 0.105)
        cv2.line(img, (xx, y + 34), (xx, floor), (28, 36, 43), 1, cv2.LINE_AA)
    cv2.line(img, (x + 18, floor), (x + w - 18, floor), (60, 70, 78), 1, cv2.LINE_AA)


def _project_anatomy_3d(kp3d, valid, x, y, w, h):
    if kp3d is None or valid is None or not np.any(valid):
        return None
    P = kp3d
    valid_idx = [i for i in range(len(P)) if valid[i] and np.all(np.isfinite(P[i])) and i >= IDX["Lshoulder"]]
    if not valid_idx:
        return None
    axes = _body_axes(P, valid)
    if axes is None:
        h_axis = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, -1.0, 0.0])
    else:
        ml, up, _ = axes
        h_axis = ml
    pts = P[valid_idx]
    center = np.nanmean(pts, axis=0)
    u = np.asarray([np.dot(p - center, h_axis) for p in pts])
    v = np.asarray([np.dot(p - center, up) for p in pts])
    span_u = max(float(np.nanmax(u) - np.nanmin(u)), 0.25)
    span_v = max(float(np.nanmax(v) - np.nanmin(v)), 0.80)
    scale = min((w * 0.50) / span_u, (h * 0.64) / span_v)
    cx, cy = x + w * 0.50, y + h * 0.58

    def proj(p):
        q = p - center
        return (cx + np.dot(q, h_axis) * scale,
                cy - np.dot(q, up) * scale)

    out = {}
    for name, idx in IDX.items():
        if valid[idx] and np.all(np.isfinite(P[idx])):
            out[name] = proj(P[idx])
    return out


def _project_anatomy_2d(kp2d, x, y, w, h):
    proj = _project_2d_pose(kp2d, x, y, w, h)
    if proj is None:
        return None
    out = {}
    for name, idx in IDX.items():
        p = kp2d[idx]
        if np.all(np.isfinite(p[:2])):
            out[name] = proj(p[:2])
    return out


def _anatomy_points_from_kp2d(kp2d):
    if kp2d is None:
        return None
    out = {}
    for name, idx in IDX.items():
        p = kp2d[idx]
        if np.all(np.isfinite(p[:2])):
            out[name] = (float(p[0]), float(p[1]))
    return out if out else None


def _required_points(pts, *names):
    return all(name in pts for name in names)


def _draw_skull(img, pts, scale):
    if "nose" not in pts:
        return
    if _required_points(pts, "Lshoulder", "Rshoulder"):
        shoulder_w = abs(pts["Rshoulder"][0] - pts["Lshoulder"][0])
    else:
        shoulder_w = scale * 0.30
    r = max(12, min(42, int(shoulder_w * 0.36)))
    nose = pts["nose"]
    head = (nose[0], nose[1] - r * 0.55)
    c = _as_pt(head)
    axes = (max(1, int(r * 0.78)), max(1, int(r)))
    cv2.ellipse(img, c, axes, 0, 0, 360, BONE_EDGE, -1, cv2.LINE_AA)
    cv2.ellipse(img, c, (max(1, axes[0] - 3), max(1, axes[1] - 3)), 0, 0, 360,
                BONE, -1, cv2.LINE_AA)
    cv2.ellipse(img, c, axes, 0, 0, 360, BONE_HI, max(2, r // 10), cv2.LINE_AA)
    cv2.circle(img, _as_pt((head[0] - r * 0.22, head[1] - r * 0.08)),
               max(2, r // 8), (30, 32, 34), -1, cv2.LINE_AA)
    cv2.circle(img, _as_pt((head[0] + r * 0.20, head[1] - r * 0.08)),
               max(2, r // 8), (30, 32, 34), -1, cv2.LINE_AA)
    jaw = np.asarray([
        (head[0] - r * 0.36, head[1] + r * 0.34),
        (head[0] - r * 0.22, head[1] + r * 0.66),
        (head[0] + r * 0.18, head[1] + r * 0.66),
        (head[0] + r * 0.34, head[1] + r * 0.34),
    ], np.int32)
    cv2.polylines(img, [jaw], True, BONE_EDGE, max(2, r // 10), cv2.LINE_AA)
    cv2.polylines(img, [jaw], True, BONE, max(1, r // 14), cv2.LINE_AA)
    for i in range(5):
        tx = head[0] - r * 0.18 + i * r * 0.09
        cv2.line(img, _as_pt((tx, head[1] + r * 0.48)),
                 _as_pt((tx, head[1] + r * 0.62)), BONE_EDGE, 1, cv2.LINE_AA)


def _draw_ribcage(img, pts, scale):
    if not _required_points(pts, "Lshoulder", "Rshoulder", "Lhip", "Rhip"):
        return
    lsh, rsh = pts["Lshoulder"], pts["Rshoulder"]
    lhp, rhp = pts["Lhip"], pts["Rhip"]
    shoulder = _mid(lsh, rsh)
    hip = _mid(lhp, rhp)
    torso_h = max(28, abs(hip[1] - shoulder[1]))
    torso_w = max(28, abs(rsh[0] - lsh[0]) * 0.78)
    spine_top = _interp(shoulder, hip, 0.10)
    spine_bot = _interp(shoulder, hip, 0.94)
    _bone_line(img, spine_top, spine_bot, max(4, int(scale * 0.010)))
    for t in np.linspace(0.10, 0.92, 10):
        p = _interp(spine_top, spine_bot, float(t))
        _joint(img, p, max(2, int(scale * 0.006)))
    for i, t in enumerate(np.linspace(0.18, 0.78, 7)):
        center = _interp(shoulder, hip, float(t))
        rib_w = torso_w * (1.0 - abs(t - 0.44) * 0.75)
        rib_h = torso_h * 0.16
        th = max(2, int(scale * 0.006))
        _ellipse_bone(img, center, (rib_w * 0.50, rib_h), 0, 198, 342, th)
        _ellipse_bone(img, center, (rib_w * 0.50, rib_h), 0, 18, 162, th)
    _bone_line(img, lsh, rsh, max(4, int(scale * 0.012)))
    _joint(img, lsh, max(5, int(scale * 0.012)))
    _joint(img, rsh, max(5, int(scale * 0.012)))


def _draw_pelvis(img, pts, scale):
    if not _required_points(pts, "Lhip", "Rhip"):
        return
    lhp, rhp = pts["Lhip"], pts["Rhip"]
    center = _mid(lhp, rhp)
    hip_w = max(28, abs(rhp[0] - lhp[0]) * 1.08)
    hip_h = max(20, scale * 0.10)
    _ellipse_bone(img, (center[0] - hip_w * 0.19, center[1] + hip_h * 0.12),
                  (hip_w * 0.28, hip_h * 0.54), -18, 205, 35, max(3, int(scale * 0.008)))
    _ellipse_bone(img, (center[0] + hip_w * 0.19, center[1] + hip_h * 0.12),
                  (hip_w * 0.28, hip_h * 0.54), 18, 145, 335, max(3, int(scale * 0.008)))
    _bone_line(img, lhp, rhp, max(4, int(scale * 0.010)))
    _joint(img, lhp, max(6, int(scale * 0.012)))
    _joint(img, rhp, max(6, int(scale * 0.012)))


def _draw_hand_or_foot(img, root, tip, scale, is_foot=False):
    dx, dy = tip[0] - root[0], tip[1] - root[1]
    n = max(1e-6, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    length = max(scale * (0.08 if is_foot else 0.06), n * 0.35)
    spread = scale * (0.018 if is_foot else 0.014)
    palm = (root[0] + ux * length * 0.28, root[1] + uy * length * 0.28)
    for i in range(-2, 3):
        base = (palm[0] + px * i * spread * 0.45, palm[1] + py * i * spread * 0.45)
        end = (root[0] + ux * length + px * i * spread, root[1] + uy * length + py * i * spread)
        _bone_line(img, base, end, max(1, int(scale * 0.004)))


def _draw_limb(img, pts, side, scale):
    shoulder, elbow, wrist = f"{side}shoulder", f"{side}elbow", f"{side}wrist"
    hip, knee, ankle = f"{side}hip", f"{side}knee", f"{side}ankle"
    if _required_points(pts, shoulder, elbow, wrist):
        _bone_line(img, pts[shoulder], pts[elbow], max(5, int(scale * 0.014)))
        _double_bone(img, pts[elbow], pts[wrist], max(5, int(scale * 0.018)),
                     max(3, int(scale * 0.008)))
        _joint(img, pts[elbow], max(5, int(scale * 0.011)))
        direction = pts[wrist]
        if shoulder in pts:
            direction = (pts[wrist][0] + (pts[wrist][0] - pts[elbow][0]) * 0.45,
                         pts[wrist][1] + (pts[wrist][1] - pts[elbow][1]) * 0.45)
        _draw_hand_or_foot(img, pts[wrist], direction, scale, False)
    if _required_points(pts, hip, knee, ankle):
        _bone_line(img, pts[hip], pts[knee], max(7, int(scale * 0.018)))
        _double_bone(img, pts[knee], pts[ankle], max(6, int(scale * 0.022)),
                     max(4, int(scale * 0.010)))
        _joint(img, pts[knee], max(6, int(scale * 0.013)))
        direction = (pts[ankle][0] + (pts[ankle][0] - pts[knee][0]) * 0.45,
                     pts[ankle][1] + (pts[ankle][1] - pts[knee][1]) * 0.45)
        _draw_hand_or_foot(img, pts[ankle], direction, scale, True)
        _joint(img, pts[ankle], max(5, int(scale * 0.011)))


def _draw_anatomical_skeleton(img, pts, scale):
    if pts is None:
        return False
    _draw_ribcage(img, pts, scale)
    _draw_pelvis(img, pts, scale)
    _draw_limb(img, pts, "L", scale)
    _draw_limb(img, pts, "R", scale)
    if _required_points(pts, "nose", "Lshoulder", "Rshoulder"):
        neck = _mid(pts["Lshoulder"], pts["Rshoulder"])
        _bone_line(img, neck, pts["nose"], max(4, int(scale * 0.010)))
    _draw_skull(img, pts, scale)
    for name in ("Lshoulder", "Rshoulder", "Lhip", "Rhip", "Lwrist", "Rwrist"):
        if name in pts:
            _joint(img, pts[name], max(4, int(scale * 0.010)))
    return True


def draw_anatomical_overlay_2d(frame, kp):
    pts = _anatomy_points_from_kp2d(kp)
    if pts is None:
        return False
    body_pts = np.asarray([p for name, p in pts.items() if name in {
        "nose", "Lshoulder", "Rshoulder", "Lhip", "Rhip",
        "Lknee", "Rknee", "Lankle", "Rankle",
    }], float)
    if body_pts.size == 0:
        return False
    lo = np.nanmin(body_pts, axis=0)
    hi = np.nanmax(body_pts, axis=0)
    scale = max(80.0, float(max(hi - lo)))

    skel = np.zeros_like(frame)
    if not _draw_anatomical_skeleton(skel, pts, scale):
        return False

    gray = cv2.cvtColor(skel, cv2.COLOR_BGR2GRAY)
    mask = gray > 8
    if not np.any(mask):
        return False
    glow = cv2.dilate(mask.astype(np.uint8), np.ones((13, 13), np.uint8), iterations=1).astype(bool)
    cyan = np.zeros_like(frame)
    cyan[:, :] = (235, 255, 185)
    frame[glow] = cv2.addWeighted(frame[glow], 0.72, cyan[glow], 0.28, 0)

    tint = np.zeros_like(frame)
    intensity = np.clip(gray.astype(np.float32) / 255.0, 0.0, 1.0)
    tint[:, :, 0] = np.clip(255 * intensity, 0, 255).astype(np.uint8)
    tint[:, :, 1] = np.clip(248 * intensity, 0, 255).astype(np.uint8)
    tint[:, :, 2] = np.clip(200 * intensity, 0, 255).astype(np.uint8)
    frame[mask] = cv2.addWeighted(frame[mask], 0.36, tint[mask], 0.90, 0)
    return True


def draw_walking_skeleton(img, x, y, w, h, kp3d, valid, kp2d=None, gait=None):
    _panel(img, x, y, w, h, "Walking skeleton")
    px, py, pw, ph = x + 12, y + 30, w - 24, h - 44
    _draw_blank_3d_grid(img, px, py, pw, ph)

    pts = None
    if kp3d is not None and valid is not None and np.any(valid):
        pts = _project_anatomy_3d(kp3d, valid, px, py, pw, ph)
        source = "3D"
    else:
        pts = _project_anatomy_2d(kp2d, px, py, pw, ph)
        source = "2D"

    if not _draw_anatomical_skeleton(img, pts, min(pw, ph)):
        _text(img, "waiting for pose", (px + 18, py + ph // 2), 0.55, MUTED, 1)
        return

    if gait:
        left = gait.get("contact_L", "--")
        right = gait.get("contact_R", "--")
        cadence = _fmt(gait.get("cadence_steps_per_min"), 0)
        _text(img, f"{source}  L {left}  R {right}  Cadence {cadence}/min",
              (px + 16, py + 24), 0.48, FG, 1)
    else:
        _text(img, source, (px + 16, py + 24), 0.48, MUTED, 1)


# ---------- status bar ----------
def status_bar(img, x, y, w, h, st):
    _panel(img, x, y, w, h)
    rec = st.get("recording")
    cv2.circle(img, (x + 16, y + h // 2), 8, WARN if rec else MUTED, -1)
    _text(img, "REC" if rec else "idle", (x + 30, y + h // 2 + 4), 0.5,
          WARN if rec else MUTED, 1)
    items = [
        f"t={st.get('session_t',0):.1f}s",
        f"cam0 {st.get('cam0_fps',0):.0f}fps", f"cam1 {st.get('cam1_fps',0):.0f}fps",
        f"pose {st.get('pose_fps',0):.0f}fps", f"force {st.get('force_fps',0):.0f}Hz",
        f"{st.get('pose_device','?')}{'/fp16' if st.get('pose_half') else ''}",
        f"lat {st.get('cam0_latency_ms',0):.0f}/{st.get('cam1_latency_ms',0):.0f}ms",
        f"3D {'ON' if st.get('calibrated_3d') else 'off'}",
        f"joints {st.get('pose_valid_joints',0):.0f}/17",
    ]
    if st.get("pose_reproj_error_px", 0):
        items.append(f"reproj {st.get('pose_reproj_error_px',0):.1f}px")
    if st.get("pose_smoothing"):
        items.append("smooth")
    if st.get("force_scale_n_per_v", 0.0):
        items.append(f"BW scale {st.get('force_scale_n_per_v',0):.1f}N/V")
    elif st.get("force_auto_bw_needed", 0):
        items.append(f"BW cal {st.get('force_auto_bw_samples',0)}/"
                     f"{st.get('force_auto_bw_needed',0)}")
    xx = x + 90
    for s in items:
        _text(img, s, (xx, y + h // 2 + 4), 0.45, FG, 1)
        xx += int(11 * len(s))


# ---------- treadmill ----------
def treadmill_panel(img, x, y, w, h, st):
    _panel(img, x, y, w, h, "Treadmill")
    mode = st.get("treadmill_mode", "FIXED")
    conn = st.get("treadmill_connected", False)
    cv2.circle(img, (x + w - 16, y + 12), 6, OK_C if conn else WARN, -1)
    _text(img, "online" if conn else "offline", (x + w - 76, y + 16), 0.36,
          OK_C if conn else WARN, 1)
    items = [
        ("Mode", mode),
        ("Target", f"{st.get('treadmill_target_vel', 0):.2f} m/s"),
        ("Current", f"{st.get('treadmill_current_vel', 0):.2f} m/s"),
        ("Incline", f"{st.get('treadmill_incline', 0):.1f} deg"),
    ]
    for i, (k, v) in enumerate(items):
        yy = y + 44 + i * 24
        _text(img, k, (x + 10, yy), 0.46, MUTED, 1)
        c = WARN if (k == "Mode" and mode == "STOP") else FG
        _text(img, v, (x + 110, yy), 0.46, c, 1)


# ---------- metrics ----------
def metrics_panel(img, x, y, w, h, gait):
    _panel(img, x, y, w, h, "Gait cycle + metrics")
    if not gait:
        _text(img, "walking not detected yet", (x + 10, y + 40), 0.45, MUTED, 1)
        return
    left_phase = gait.get("phase_L", "unknown")
    right_phase = gait.get("phase_R", "unknown")
    left_contact = gait.get("contact_L", "--")
    right_contact = gait.get("contact_R", "--")
    _text(img, "L", (x + 10, y + 42), 0.48, LEFT_C, 2)
    _text(img, f"{left_contact} | {left_phase}", (x + 34, y + 42), 0.42, FG, 1)
    _text(img, "R", (x + 10, y + 66), 0.48, RIGHT_C, 2)
    _text(img, f"{right_contact} | {right_phase}", (x + 34, y + 66), 0.42, FG, 1)
    items = [
        ("Cadence", f"{_fmt(gait.get('cadence_steps_per_min'), 0)} /min"),
        ("Stride L/R", f"{_fmt(gait.get('stride_time_L_s'), 2)}/"
                       f"{_fmt(gait.get('stride_time_R_s'), 2)} s"),
        ("Stance L/R", f"{_fmt(gait.get('stance_time_L_s'), 2)}/"
                       f"{_fmt(gait.get('stance_time_R_s'), 2)} s"),
        ("Duty L/R", f"{_fmt(gait.get('duty_factor_L'), 2)}/"
                     f"{_fmt(gait.get('duty_factor_R'), 2)}"),
        ("Stride len", f"{_fmt(gait.get('stride_length_L_m'), 2)}/"
                       f"{_fmt(gait.get('stride_length_R_m'), 2)} m"),
        ("Stance len", f"{_fmt(gait.get('stance_length_L_m'), 2)}/"
                       f"{_fmt(gait.get('stance_length_R_m'), 2)} m"),
        ("Stride sym", f"{_fmt(gait.get('stride_time_symmetry_pct'), 1)} %"),
        ("Stance sym", f"{_fmt(gait.get('stance_time_symmetry_pct'), 1)} %"),
    ]
    for i, (k, v) in enumerate(items):
        col = 0 if i < 4 else 1
        row = i if i < 4 else i - 4
        x0 = x + 10 + col * (w // 2)
        yy = y + 92 + row * 18
        _text(img, k, (x0, yy), 0.35, MUTED, 1)
        _text(img, v, (x0 + 104, yy), 0.35, FG, 1)


def angle_table_panel(img, x, y, w, h, angles):
    _panel(img, x, y, w, h, "Joint angles (deg)")
    if not angles:
        _text(img, "3D angles unavailable", (x + 10, y + 42), 0.45, MUTED, 1)
        return
    rows = [
        ("Hip flex", "hip_flexion_l", "hip_flexion_r"),
        ("Hip add", "hip_adduction_l", "hip_adduction_r"),
        ("Hip abd", "hip_abduction_l", "hip_abduction_r"),
        ("Knee", "knee_angle_l", "knee_angle_r"),
        ("Ankle", "ankle_angle_l", "ankle_angle_r"),
        ("Thigh m", "thigh_length_l_m", "thigh_length_r_m"),
        ("Shank m", "shank_length_l_m", "shank_length_r_m"),
    ]
    col0, col1, col2 = x + 12, x + int(w * 0.50), x + int(w * 0.73)
    _text(img, "Angle", (col0, y + 42), 0.44, MUTED, 1)
    _text(img, "Left", (col1, y + 42), 0.44, LEFT_C, 1)
    _text(img, "Right", (col2, y + 42), 0.44, RIGHT_C, 1)
    row_step = max(18, min(24, (h - 58) // max(1, len(rows))))
    for i, (label, lk, rk) in enumerate(rows):
        yy = y + 64 + i * row_step
        _text(img, label, (col0, yy), 0.43, FG, 1)
        _text(img, _fmt(angles.get(lk), 1), (col1, yy), 0.43, LEFT_C, 1)
        _text(img, _fmt(angles.get(rk), 1), (col2, yy), 0.43, RIGHT_C, 1)


def _pose_state(state):
    kp3d_s, _ = state.kp3d.get()
    if kp3d_s is not None:
        pts3d, valid = kp3d_s.value
    else:
        pts3d, valid = None, None
    kp2d_s, _ = state.kp2d[1].get()
    if kp2d_s is None:
        kp2d_s, _ = state.kp2d[0].get()
    gait_s, _ = state.gait.get()
    ang_s, _ = state.angles.get()
    return pts3d, valid, kp2d_s, gait_s, ang_s


def _camera_panel(img, x, y, w, h, state, ci, angles=None, title=None):
    _panel(img, x, y, w, h, title or f"cam{ci}")
    px, py, pw, ph = x + 8, y + 26, w - 16, h - 34
    slot, _ = state.frame[ci].get()
    kp_s, _ = state.kp2d[ci].get()
    frame = None
    if slot is not None:
        frame = slot.value.copy()
        if not draw_anatomical_overlay_2d(frame, kp_s.value if kp_s else None):
            draw_skeleton_2d(frame, kp_s.value if kp_s else None,
                             angles.value if angles else None)
    fitted = _fit_image(frame, pw, ph)
    img[py:py + ph, px:px + pw] = fitted
    if frame is None:
        _text(img, "no signal", (px + 18, py + ph // 2), 0.55, MUTED, 1)


def _overlay_camera_id(state, preferred=1):
    s, _ = state.frame[preferred].get()
    if s is not None and s.value is not None:
        return preferred
    other = 0 if preferred == 1 else 1
    s, _ = state.frame[other].get()
    if s is not None and s.value is not None:
        return other
    return preferred


def _build_focus_dashboard(state, size, focus, window_s, body_mass_kg):
    W, H = size
    cv = np.full((H, W, 3), BG, np.uint8)
    focus = focus if focus in FOCUS_KEYS else "skeleton"
    st = state.get_status()
    status_rect = dashboard_layout(size)["status"]
    m = 8
    fx, fy, fw, fh = m, m, W - 2 * m, max(180, status_rect[1] - 2 * m)
    bw = body_mass_kg * 9.80665
    pts3d, valid, kp2d_s, gait_s, ang_s = _pose_state(state)

    if focus in ("cam0", "cam1"):
        _camera_panel(cv, fx, fy, fw, fh, state, int(focus[-1]), ang_s,
                      f"{focus} focused")
    elif focus == "skeleton":
        ci = _overlay_camera_id(state)
        _camera_panel(cv, fx, fy, fw, fh, state, ci, ang_s,
                      f"skeleton overlay cam{ci}")
    elif focus == "grf":
        ft, fxv = state.force_hist.snapshot()
        lfz = fxv[:, 2] / bw * 100 if fxv.size else None
        rfz = fxv[:, 8] / bw * 100 if fxv.size else None
        tot = (fxv[:, 2] + fxv[:, 8]) / bw * 100 if fxv.size else None
        ymax = 160.0
        if tot is not None and np.any(np.isfinite(tot)):
            peak = np.nanpercentile(np.abs(tot[np.isfinite(tot)]), 98)
            if np.isfinite(peak):
                ymax = max(ymax, float(np.ceil(peak * 1.15 / 20.0) * 20.0))
        scrolling_plot(cv, fx, fy, fw, fh, "Vertical GRF (%BW)", ft,
                       [(tot, FG, "Tot"), (lfz, LEFT_C, "L"), (rfz, RIGHT_C, "R")],
                       window_s, 0.0, ymax)
    elif focus == "joint_plot":
        at, ax = state.angle_hist.snapshot()
        kL = ax[:, ANGLE_NAMES.index("knee_angle_l")] if ax.size else None
        kR = ax[:, ANGLE_NAMES.index("knee_angle_r")] if ax.size else None
        hL = ax[:, ANGLE_NAMES.index("hip_flexion_l")] if ax.size else None
        hR = ax[:, ANGLE_NAMES.index("hip_flexion_r")] if ax.size else None
        scrolling_plot(cv, fx, fy, fw, fh, "Joint angles (deg)", at,
                       [(kL, LEFT_C, "kneeL"), (kR, RIGHT_C, "kneeR"),
                        (hL, ACCENT, "hipL"), (hR, OK_C, "hipR")],
                       window_s, -20, 90)
    elif focus == "cop":
        _, cx = state.cop_hist.latest()
        cop = cx[:4] if cx is not None else [np.nan] * 4
        cop_map(cv, fx, fy, fw, fh, cop, approx=True)
    elif focus == "treadmill":
        treadmill_panel(cv, fx, fy, fw, fh, st)
    elif focus == "metrics":
        metrics_panel(cv, fx, fy, fw, fh, gait_s.value if gait_s else None)
    elif focus == "angles":
        angle_table_panel(cv, fx, fy, fw, fh, ang_s.value if ang_s else None)

    _text(cv, "double-click to return", (fx + fw - 190, fy + 20), 0.45, MUTED, 1)
    status_bar(cv, *status_rect, st)
    return cv


# ---------- top-level composite ----------
def build_dashboard(state, size=(1600, 900), window_s=6.0, body_mass_kg=82.0,
                    focus=None, layout_controls=None):
    if focus:
        return _build_focus_dashboard(state, size, focus, window_s, body_mass_kg)
    W, H = size
    cv = np.full((H, W, 3), BG, np.uint8)
    bw = body_mass_kg * 9.80665

    layout = dashboard_layout(size, layout_controls)

    # compact camera previews, fitted to 16:9 without stretching
    ang_s, _ = state.angles.get()
    for ci in (0, 1):
        _camera_panel(cv, *layout[f"cam{ci}"], state, ci, ang_s)

    # primary live camera view with anatomical skeleton overlay
    pts3d, valid, kp2d_s, gait_s, ang_s = _pose_state(state)
    ci = _overlay_camera_id(state)
    _camera_panel(cv, *layout["skeleton"], state, ci, ang_s,
                  f"skeleton overlay cam{ci}")

    # left column support panels
    st = state.get_status()
    ct, cx = state.cop_hist.latest()
    cop = cx[:4] if cx is not None else [np.nan] * 4
    cop_map(cv, *layout["cop"], cop, approx=True)
    treadmill_panel(cv, *layout["treadmill"], st)

    # right column plots and metrics
    # force plot
    ft, fx = state.force_hist.snapshot()
    lfz = fx[:, 2] / bw * 100 if fx.size else None     # %BW
    rfz = fx[:, 8] / bw * 100 if fx.size else None
    tot = (fx[:, 2] + fx[:, 8]) / bw * 100 if fx.size else None
    grf_ymax = 160.0
    if tot is not None and np.any(np.isfinite(tot)):
        peak = np.nanpercentile(np.abs(tot[np.isfinite(tot)]), 98)
        if np.isfinite(peak):
            grf_ymax = max(grf_ymax, float(np.ceil(peak * 1.15 / 20.0) * 20.0))
    scrolling_plot(cv, *layout["grf"], "Vertical GRF (%BW)", ft,
                   [(tot, FG, "Tot"), (lfz, LEFT_C, "L"), (rfz, RIGHT_C, "R")],
                   window_s, 0.0, grf_ymax)
    # angle plot
    at, ax = state.angle_hist.snapshot()
    kL = ax[:, ANGLE_NAMES.index("knee_angle_l")] if ax.size else None
    kR = ax[:, ANGLE_NAMES.index("knee_angle_r")] if ax.size else None
    hL = ax[:, ANGLE_NAMES.index("hip_flexion_l")] if ax.size else None
    hR = ax[:, ANGLE_NAMES.index("hip_flexion_r")] if ax.size else None
    scrolling_plot(cv, *layout["joint_plot"], "Joint angles (deg)", at,
                   [(kL, LEFT_C, "kneeL"), (kR, RIGHT_C, "kneeR"),
                    (hL, ACCENT, "hipL"), (hR, OK_C, "hipR")],
                   window_s, -20, 90)
    metrics_panel(cv, *layout["metrics"], gait_s.value if gait_s else None)
    angle_table_panel(cv, *layout["angles"], ang_s.value if ang_s else None)
    status_bar(cv, *layout["status"], st)
    draw_resize_handles(cv, layout)
    return cv
