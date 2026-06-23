"""Render the synchronized data as an annotated overlay video.

The raw Cam0 frame is placed on the left; a dark instrument panel on the right
shows, for the current frame:
  * header (trial, time, frame, calibration mode, sync quality)
  * a sagittal stick figure driven by the hip/knee/ankle angles
  * live joint-angle readouts (hip / knee / ankle, L & R)
  * left/right vertical GRF bars in %body-weight with a stance indicator
  * a centre-of-pressure foot map (flagged when calibration is approximate)
  * scrolling time-series of total GRF and knee angles
  * spatiotemporal metrics (cadence, stride time, symmetry)

All drawing is done with OpenCV primitives for speed (no per-frame matplotlib).
"""
from __future__ import annotations

import cv2
import numpy as np

from .pose2d import IDX, SKELETON

# ---- palette (BGR) ----
BG = (24, 24, 28)
PANEL = (36, 36, 42)
FG = (230, 230, 235)
MUTED = (140, 140, 150)
LEFT_C = (90, 180, 255)      # orange-ish for Left
RIGHT_C = (120, 220, 120)    # green for Right
ACCENT = (210, 160, 70)
WARN = (70, 130, 240)
GRID = (60, 60, 68)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.5, color=FG, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _panel(img, x, y, w, h, title=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), GRID, 1)
    if title:
        _text(img, title, (x + 10, y + 22), 0.52, ACCENT, 1)
    return x, y, w, h


def _safe(v, default=np.nan):
    return default if v is None or (isinstance(v, float) and np.isnan(v)) else v


class OverlayRenderer:
    def __init__(self, merged, raw_video_path, cfg, kp2d=None):
        self.m = merged
        self.df = merged.table
        self.cfg = cfg
        self.raw_path = raw_video_path
        self.kp2d = kp2d   # (n_frames, 17, 3) [x, y, conf] or None
        self.window_s = float(cfg.get("overlay", "scrolling_window_s", default=4.0))
        self.draw_skel = bool(cfg.get("overlay", "draw_skeleton", default=True))

        # Pre-extract series used by the scrolling plots / bars.
        self.t = self.df["opencap_time_s"].to_numpy()
        self.total_bw = self.df["total_Fz_BW"].to_numpy()
        self.lfz_bw = self.df["L_Fz_BW"].to_numpy()
        self.rfz_bw = self.df["R_Fz_BW"].to_numpy()
        self.knee_r = self.df.get("knee_angle_r_deg")
        self.knee_l = self.df.get("knee_angle_l_deg")
        self.bw = merged.meta["bodyweight_N"]
        self.cal_mode = merged.meta["calibration_mode"]

    # ---------- scrolling time-series ----------
    def _plot(self, img, x, y, w, h, series_list, frame, title,
              ymin, ymax, units=""):
        _panel(img, x, y, w, h, title)
        px, py, pw, ph = x + 8, y + 30, w - 16, h - 42
        cv2.rectangle(img, (px, py), (px + pw, py + ph), BG, -1)
        fps = self.m.meta["video_fps"]
        win = int(self.window_s * fps)
        f0 = max(0, frame - win)
        idx = np.arange(f0, frame + 1)
        if len(idx) < 2:
            return
        # zero / reference gridline
        def yto(v):
            v = np.clip(v, ymin, ymax)
            return int(py + ph - (v - ymin) / (ymax - ymin + 1e-9) * ph)
        for gv in np.linspace(ymin, ymax, 4):
            yy = yto(gv)
            cv2.line(img, (px, yy), (px + pw, yy), GRID, 1)
            _text(img, f"{gv:.1f}", (px + 2, yy - 2), 0.34, MUTED, 1)
        for si, (series, color, lbl) in enumerate(series_list):
            arr = np.asarray(series, float)[idx]
            xs = (px + (idx - f0) / max(1, (frame - f0)) * pw).astype(int)
            pts = [(int(xs[i]), yto(arr[i])) for i in range(len(idx))
                   if np.isfinite(arr[i])]
            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts, np.int32)], False, color, 2,
                              cv2.LINE_AA)
            # current value label
            cur = np.asarray(series, float)[frame]
            if np.isfinite(cur):
                _text(img, f"{lbl} {cur:.2f}{units}",
                      (px + 6, py + 14 + 12 * si), 0.42, color, 1)

    # ---------- GRF bars ----------
    def _grf_bars(self, img, x, y, w, h, row):
        _panel(img, x, y, w, h, "Vertical GRF (%BW)")
        base_y = y + h - 24
        top_y = y + 36
        full = base_y - top_y
        maxbw = 1.5
        for i, (lbl, val, color, stance) in enumerate([
                ("L", _safe(row.get("L_Fz_BW")), LEFT_C, row.get("L_stance")),
                ("R", _safe(row.get("R_Fz_BW")), RIGHT_C, row.get("R_stance"))]):
            bx = x + 30 + i * (w // 2 - 10)
            bw_px = w // 2 - 70
            cv2.rectangle(img, (bx, top_y), (bx + bw_px, base_y), BG, -1)
            cv2.line(img, (bx, base_y - int(full / maxbw)),
                     (bx + bw_px, base_y - int(full / maxbw)), MUTED, 1)  # 1 BW line
            if np.isfinite(val):
                hh = int(np.clip(val, 0, maxbw) / maxbw * full)
                cv2.rectangle(img, (bx, base_y - hh), (bx + bw_px, base_y), color, -1)
                _text(img, f"{val*100:.0f}%", (bx, top_y - 6), 0.45, color, 1)
            ind = color if stance else MUTED
            _text(img, lbl, (bx + bw_px // 2 - 6, base_y + 18), 0.5, ind, 2)

    # ---------- COP foot map ----------
    def _cop_map(self, img, x, y, w, h, row):
        title = "Centre of Pressure"
        if self.cal_mode == "bodyweight":
            title += " (approx)"
        _panel(img, x, y, w, h, title)
        cx0, cy0 = x + w // 2, y + h // 2 + 6
        # two belts
        for i, (sx, color, cxk, cyk) in enumerate([
                (-1, LEFT_C, "L_COPx_m", "L_COPy_m"),
                (1, RIGHT_C, "R_COPx_m", "R_COPy_m")]):
            belt_cx = cx0 + sx * (w // 4)
            cv2.rectangle(img, (belt_cx - 26, cy0 - 70), (belt_cx + 26, cy0 + 70),
                          GRID, 1)
            copx, copy = _safe(row.get(cxk)), _safe(row.get(cyk))
            if np.isfinite(copx) and np.isfinite(copy):
                # map +-0.25 m to belt box
                px = int(belt_cx + np.clip(copx, -0.25, 0.25) / 0.25 * 24)
                py = int(cy0 - np.clip(copy, -0.5, 0.5) / 0.5 * 66)
                cv2.circle(img, (px, py), 6, color, -1)

    # ---------- sagittal stick figure ----------
    def _stick(self, img, x, y, w, h, row):
        _panel(img, x, y, w, h, "Sagittal pose")
        cx, hip_y = x + w // 2, y + 52
        L_thigh, L_shank, L_foot, trunk = 46, 44, 20, 52

        def deg(v):
            return np.deg2rad(_safe(v, 0.0))

        def leg(side, color, dx):
            hf = deg(row.get(f"hip_flexion_{side}_deg"))
            ka = deg(row.get(f"knee_angle_{side}_deg"))
            aa = deg(row.get(f"ankle_angle_{side}_deg"))
            hip = np.array([cx + dx, hip_y])
            # thigh: hip flexion measured from vertical-down, forward +
            th = np.array([np.sin(hf), np.cos(hf)])
            knee = hip + L_thigh * th
            # knee flexion bends shank backward relative to thigh
            sh = np.array([np.sin(hf - ka), np.cos(hf - ka)])
            ankle = knee + L_shank * sh
            # foot forward, ankle dorsi/plantar
            ft = np.array([np.cos(hf - ka + aa), -np.sin(hf - ka + aa)])
            toe = ankle + L_foot * ft
            pts = [hip, knee, ankle, toe]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, 3,
                         cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, tuple(p.astype(int)), 3, color, -1)
            return hip

        if bool(row.get("has_kinematics", False)):
            hipL = leg("l", LEFT_C, -6)
            hipR = leg("r", RIGHT_C, 6)
            # trunk from pelvis
            lext = np.deg2rad(_safe(row.get("lumbar_extension_deg"), 0.0))
            top = np.array([cx + np.sin(-lext) * trunk, hip_y - np.cos(lext) * trunk])
            cv2.line(img, (cx, hip_y), tuple(top.astype(int)), FG, 3, cv2.LINE_AA)
            cv2.circle(img, tuple(top.astype(int)), 8, FG, 2)
        else:
            _text(img, "no pose", (cx - 28, hip_y + 30), 0.5, MUTED, 1)

    # ---------- joint readouts ----------
    def _readouts(self, img, x, y, w, h, row):
        _panel(img, x, y, w, h, "Joint angles (deg)")
        rows = [("Hip", "hip_flexion"), ("Knee", "knee_angle"),
                ("Ankle", "ankle_angle")]
        _text(img, "L", (x + 120, y + 22), 0.45, LEFT_C, 1)
        _text(img, "R", (x + 200, y + 22), 0.45, RIGHT_C, 1)
        for i, (lbl, key) in enumerate(rows):
            yy = y + 48 + i * 26
            _text(img, lbl, (x + 12, yy), 0.5, FG, 1)
            for cx, side, color in [(110, "l", LEFT_C), (190, "r", RIGHT_C)]:
                v = _safe(row.get(f"{key}_{side}_deg"))
                s = f"{v:6.1f}" if np.isfinite(v) else "  --"
                _text(img, s, (x + cx, yy), 0.5, color, 1)

    # ---------- gait phase bar ----------
    def _phase(self, img, x, y, w, h, row):
        _panel(img, x, y, w, h, "Gait cycle (%)")
        for i, (side, color) in enumerate([("L", LEFT_C), ("R", RIGHT_C)]):
            yy = y + 40 + i * 30
            ph = _safe(row.get(f"{side}_gait_phase_pct"))
            stance = bool(row.get(f"{side}_stance"))
            cv2.rectangle(img, (x + 30, yy), (x + w - 16, yy + 16), BG, -1)
            # stance ~0-60%, swing 60-100% shading
            split = int((x + 30) + 0.6 * (w - 46))
            cv2.rectangle(img, (x + 30, yy), (split, yy + 16), (50, 50, 58), -1)
            if np.isfinite(ph):
                px = int((x + 30) + ph / 100 * (w - 46))
                cv2.line(img, (px, yy - 2), (px, yy + 18), color, 2)
            _text(img, side, (x + 12, yy + 13), 0.5, color, 1)
            _text(img, "stance", (x + 36, yy + 12), 0.32, MUTED, 1)
            _text(img, "swing", (split + 6, yy + 12), 0.32, MUTED, 1)

    # ---------- metrics ----------
    def _metrics(self, img, x, y, w, h):
        _panel(img, x, y, w, h, "Spatiotemporal")
        g = self.m.meta["gait_metrics"]
        items = [
            ("Cadence", f"{g['cadence_steps_per_min']:.0f} steps/min"),
            ("Stride L/R", f"{g['stride_time_L_s']:.2f} / {g['stride_time_R_s']:.2f} s"),
            ("Stance L/R", f"{g['stance_time_L_s']:.2f} / {g['stance_time_R_s']:.2f} s"),
            ("Duty L/R", f"{g['duty_factor_L']:.2f} / {g['duty_factor_R']:.2f}"),
            ("Stride sym", f"{g['stride_time_symmetry_pct']:.1f} %"),
            ("Stance sym", f"{g['stance_time_symmetry_pct']:.1f} %"),
        ]
        for i, (k, v) in enumerate(items):
            yy = y + 44 + i * 22
            _text(img, k, (x + 12, yy), 0.44, MUTED, 1)
            _text(img, v, (x + 150, yy), 0.44, FG, 1)

    def _header(self, img, x, y, w, h, row, frame):
        _panel(img, x, y, w, h)
        _text(img, f"{self.m.meta['trial']}", (x + 12, y + 26), 0.7, FG, 2)
        _text(img, f"t={row['opencap_time_s']:.2f}s  frame {frame}",
              (x + 12, y + 50), 0.5, MUTED, 1)
        meta = self.m.meta
        _text(img, f"cal: {self.cal_mode}  |  sync r={meta['sync_correlation']:.2f}"
                   f" off={meta['sync_offset_s']:.2f}s",
              (x + 12, y + 70), 0.42, MUTED, 1)

    # ---------- on-body skeleton (2D keypoints + OpenCap angles) ----------
    @staticmethod
    def _side_color(tag):
        return {"L": LEFT_C, "R": RIGHT_C, "C": (205, 205, 210)}[tag]

    def _label_box(self, img, text, org, color):
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
        x, y = org
        cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + 3), (20, 20, 24), -1)
        cv2.putText(img, text, (x, y), FONT, 0.5, color, 1, cv2.LINE_AA)

    def _angle_arc(self, img, J, A, B, color, radius=26):
        """Draw an arc at joint J between segments J->A and J->B."""
        if not (np.all(np.isfinite(J)) and np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
            return
        a1 = np.degrees(np.arctan2(A[1] - J[1], A[0] - J[0]))
        a2 = np.degrees(np.arctan2(B[1] - J[1], B[0] - J[0]))
        delta = ((a2 - a1 + 180) % 360) - 180   # shortest signed sweep
        cv2.ellipse(img, (int(J[0]), int(J[1])), (radius, radius), 0,
                    a1, a1 + delta, color, 2, cv2.LINE_AA)

    def _draw_skeleton(self, frame, f, row):
        if self.kp2d is None or f >= len(self.kp2d):
            return
        kp = self.kp2d[f]
        overlay = frame.copy()

        # bones
        for an, bn, tag in SKELETON:
            a, b = kp[IDX[an]], kp[IDX[bn]]
            if np.all(np.isfinite(a[:2])) and np.all(np.isfinite(b[:2])):
                cv2.line(overlay, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                         self._side_color(tag), 4, cv2.LINE_AA)
        # joints
        for n, i in IDX.items():
            p = kp[i]
            if np.all(np.isfinite(p[:2])) and i >= IDX["Lshoulder"]:
                tag = "L" if n.startswith("L") else ("R" if n.startswith("R") else "C")
                cv2.circle(overlay, (int(p[0]), int(p[1])), 5,
                           self._side_color(tag), -1, cv2.LINE_AA)
                cv2.circle(overlay, (int(p[0]), int(p[1])), 5, (20, 20, 24), 1,
                           cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # angle arcs + OpenCap angle labels at the major joints
        annot = [
            ("knee", "Lknee", "Lhip", "Lankle", "knee_angle_l_deg", "L"),
            ("knee", "Rknee", "Rhip", "Rankle", "knee_angle_r_deg", "R"),
            ("hip", "Lhip", "Lshoulder", "Lknee", "hip_flexion_l_deg", "L"),
            ("hip", "Rhip", "Rshoulder", "Rknee", "hip_flexion_r_deg", "R"),
            ("elbow", "Lelbow", "Lshoulder", "Lwrist", "elbow_flex_l_deg", "L"),
            ("elbow", "Relbow", "Rshoulder", "Rwrist", "elbow_flex_r_deg", "R"),
        ]
        for _, jn, an, bn, col, tag in annot:
            J, A, B = kp[IDX[jn]][:2], kp[IDX[an]][:2], kp[IDX[bn]][:2]
            color = self._side_color(tag)
            self._angle_arc(frame, J, A, B, color)
            val = _safe(row.get(col))
            if np.all(np.isfinite(J)) and np.isfinite(val):
                dx = 12 if tag == "R" else -58
                self._label_box(frame, f"{val:.0f}deg",
                                (int(J[0]) + dx, int(J[1]) - 6), color)
        # ankle labels (no reliable distal keypoint for an arc)
        for jn, col, tag in [("Lankle", "ankle_angle_l_deg", "L"),
                             ("Rankle", "ankle_angle_r_deg", "R")]:
            J = kp[IDX[jn]][:2]
            val = _safe(row.get(col))
            if np.all(np.isfinite(J)) and np.isfinite(val):
                dx = 12 if tag == "R" else -58
                self._label_box(frame, f"{val:.0f}deg",
                                (int(J[0]) + dx, int(J[1]) + 14),
                                self._side_color(tag))

    # ---------- main ----------
    def render(self, out_path):
        cap = cv2.VideoCapture(self.raw_path)
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cfg.get("overlay", "fps") or cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        panel_w = 600
        W, H = vw + panel_w, vh
        codec = self.cfg.get("overlay", "codec", default="mp4v")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*codec),
                                 fps, (W, H))

        df = self.df
        for f in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            row = df.iloc[f] if f < len(df) else df.iloc[-1]
            if self.draw_skel:
                self._draw_skeleton(frame, f, row)
            canvas = np.full((H, W, 3), BG, np.uint8)
            canvas[:vh, :vw] = frame

            x0 = vw + 12
            pw = panel_w - 24
            self._header(canvas, x0, 10, pw, 84, row, f)
            self._readouts(canvas, x0, 104, pw // 2 - 6, 140, row)
            self._stick(canvas, x0 + pw // 2 + 6, 104, pw // 2 - 6, 140, row)
            self._grf_bars(canvas, x0, 254, pw // 2 - 6, 180, row)
            self._cop_map(canvas, x0 + pw // 2 + 6, 254, pw // 2 - 6, 180, row)
            self._phase(canvas, x0, 444, pw, 96, row)
            self._plot(canvas, x0, 550, pw, 150, [
                (self.total_bw, FG, "Total"),
                (self.lfz_bw, LEFT_C, "L"),
                (self.rfz_bw, RIGHT_C, "R")], f,
                "Vertical GRF (%BW)", 0.0, 1.5)
            if self.knee_r is not None:
                self._plot(canvas, x0, 710, pw, 150, [
                    (self.knee_l.to_numpy(), LEFT_C, "Knee L"),
                    (self.knee_r.to_numpy(), RIGHT_C, "Knee R")], f,
                    "Knee angle (deg)", -5.0, 90.0)
            self._metrics(canvas, x0, 870, pw, 170)

            writer.write(canvas)

        cap.release()
        writer.release()
        return out_path
