"""Capture engine: orchestrates camera/force sources, the pose worker, force
calibration + live gait, the shared state, and the recorder.

Thread model (single PC, one wall clock):
  * camera grabber threads (inside each CameraSource) -> latest-frame slots
  * force producer thread -> on_force() : calibrate, gait, ring buffer, record
  * pose worker thread -> pulls newest frame pair, detects 2D, triangulates,
    computes 3D angles, publishes to shared state, records
  * (GUI runs on the main thread and only reads shared state)
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque

import numpy as np

from . import clock
from .config import load_config
from .state import SharedState, ANGLE_NAMES
from .sources.camera import make_cameras
from .sources.force import make_force
from .force_calib import build_calibrator
from .pose.livegait import LiveGait
from .calib import load_stereo
from .treadmill_control import TreadmillController


def _selected_pose_config(cfg):
    pose = dict(cfg.get("pose", default={}) or {})
    profiles = pose.get("profiles") or {}
    name = pose.get("profile")
    if name and isinstance(profiles, dict):
        profile = profiles.get(name)
        if profile is None:
            print(f"[engine] pose.profile={name!r} not found; using pose defaults")
        else:
            merged = dict(pose)
            merged.update(profile or {})
            merged["profile"] = name
            pose = merged
    return pose


class Engine:
    def __init__(self, config_path: str, mode: str | None = None):
        self.cfg = load_config(config_path)
        # Apply any mode override (e.g. --mode replay) BEFORE building sources,
        # since make_cameras/make_force select live vs replay from cfg["mode"].
        if mode:
            self.cfg.raw["mode"] = mode
        self.session_t0 = None

        self.cams = make_cameras(self.cfg)
        self.force = make_force(self.cfg)
        self.calib = build_calibrator(self.cfg)
        self.stereo = load_stereo(self.cfg.abspath(
            self.cfg.get("calibration", "cameras", "extrinsics")))
        sp = self.cfg.get("pose", "smoothing", default={}) or {}
        self.pose_smoothing_enabled = bool(sp.get("enabled", True))
        self.pose_smoother_alpha = float(sp.get("alpha", 0.55))
        self.pose_smoother_max_gap_s = float(sp.get("max_gap_s", 0.20))
        self._pose_smoother = None

        body_mass = self.cfg.get("calibration", "force", "body_mass_kg") or 75.0
        g = self.cfg.get("gait", default={})
        self.gait = LiveGait(body_mass,
                             stance_threshold_bw=g.get("stance_threshold_bw", 0.05),
                             min_step_interval_s=g.get("min_step_interval_s", 0.30))

        force_rate = (self.cfg.get("force", "live", "rate_hz", default=1000)
                      if self.cfg.get("mode") == "live" else 100)
        self.state = SharedState(
            plot_window_s=self.cfg.get("gui", "plot_window_s", default=6.0),
            force_rate_hz=force_rate)
        self.state.set_stereo_calib(self.stereo)

        # treadmill controller (faithful port of the existing GUI controls)
        self.treadmill_ctrl = TreadmillController(
            write_tread_cb=self._on_tread,
            status_cb=self.state.set_status,
            log_cb=lambda m: print(f"[treadmill] {m}"))

        self.recorder = None      # set by attach_recorder()
        self._stop = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._running = False
        self._pose_thread = None
        self._pose_det = None
        self._pose_fps_n = 0
        self._pose_fps_t = clock.mono()
        fc = self.cfg.get("calibration", "force", default={})
        self._bw_auto_seconds = float(fc.get("auto_bodyweight_seconds", 0.0) or 0.0)
        self._bw_min_total_fz_volts = float(fc.get("min_total_fz_volts", 0.05) or 0.0)
        self._bw_target_samples = max(1, int(self._bw_auto_seconds * force_rate))
        self._bw_max_cv = float(fc.get("auto_bodyweight_max_cv", 0.08) or 0.08)
        self._bw_left = deque(maxlen=self._bw_target_samples)
        self._bw_right = deque(maxlen=self._bw_target_samples)
        self._bw_last_log_t = 0.0

        # bodyweight scale bootstrap for replay (whole-file estimate)
        if self.calib.mode == "bodyweight" and self.calib.scale is None:
            self._bootstrap_bodyweight()

    def _bootstrap_bodyweight(self):
        src = self.force
        if hasattr(src, "volts"):           # replay source has the full array
            self.calib.set_bodyweight_scale_from_volts(
                src.volts[2], src.volts[8],
                min_abs_total_volts=self._bw_min_total_fz_volts)

    def _maybe_auto_bodyweight_scale(self, volts12):
        if (self.calib.mode != "bodyweight" or self.calib.scale is not None
                or self._bw_auto_seconds <= 0):
            return
        v = np.asarray(volts12, float)
        total = v[2] + v[8]
        if not np.isfinite(total) or abs(total) < self._bw_min_total_fz_volts:
            return
        self._bw_left.append(v[2])
        self._bw_right.append(v[8])
        if len(self._bw_left) < self._bw_target_samples:
            self.state.set_status(
                force_auto_bw_ready=False,
                force_auto_bw_samples=len(self._bw_left),
                force_auto_bw_needed=self._bw_target_samples)
            return
        totals = np.asarray(self._bw_left) + np.asarray(self._bw_right)
        abs_total = np.abs(totals[np.isfinite(totals)])
        mean_abs = float(np.nanmean(abs_total)) if abs_total.size else np.nan
        cv = (float(np.nanstd(abs_total) / mean_abs)
              if np.isfinite(mean_abs) and mean_abs > 1e-9 else np.inf)
        if cv > self._bw_max_cv:
            now = clock.mono()
            if now - self._bw_last_log_t > 2.0:
                print(f"[force] waiting for quiet BW window "
                      f"(vertical CV={cv:.3f}, need <= {self._bw_max_cv:.3f})")
                self._bw_last_log_t = now
            self.state.set_status(
                force_auto_bw_ready=False,
                force_auto_bw_samples=len(self._bw_left),
                force_auto_bw_needed=self._bw_target_samples,
                force_auto_bw_cv=cv)
            return
        scale = self.calib.set_bodyweight_scale_from_volts(
            np.asarray(self._bw_left), np.asarray(self._bw_right),
            min_abs_total_volts=self._bw_min_total_fz_volts)
        if scale is not None:
            print(f"[force] bodyweight scale estimated: {scale:.3f} N/V")
            self.state.set_status(force_scale_n_per_v=float(scale),
                                  force_auto_bw_ready=True,
                                  force_auto_bw_samples=len(self._bw_left),
                                  force_auto_bw_needed=self._bw_target_samples,
                                  force_auto_bw_cv=cv)

    def attach_recorder(self, recorder):
        self.recorder = recorder

    def reload_calibration(self):
        self.stereo = load_stereo(self.cfg.abspath(
            self.cfg.get("calibration", "cameras", "extrinsics")))
        self.state.set_stereo_calib(self.stereo)
        self.state.set_status(calibrated_3d=self.stereo is not None)
        return self.stereo

    # ---------- force path ----------
    def _on_force(self, t, volts12):
        if self.session_t0 is None:
            self.session_t0 = t
        self._maybe_auto_bodyweight_scale(volts12)
        f = self.calib.apply(volts12)
        # ring buffer of 12 calibrated channels for plotting/record
        vec = [f[k] for k in ("L_Fx", "L_Fy", "L_Fz", "L_Mx", "L_My", "L_Mz",
                              "R_Fx", "R_Fy", "R_Fz", "R_Mx", "R_My", "R_Mz")]
        self.state.force_hist.append(t, vec)
        self.state.cop_hist.append(t, [f["L_COPx"], f["L_COPy"],
                                       f["R_COPx"], f["R_COPy"]])
        ev = self.gait.update(t, f["L_Fz"], f["R_Fz"])
        self.state.gait.set(self.gait.metrics(self._belt_speed_for_metrics()), t)
        # drive self-paced treadmill control from the raw force channels
        self.treadmill_ctrl.feed_force(volts12)
        if self.recorder:
            self.recorder.on_force(t, volts12, f, ev)

    def _on_tread(self, vec6):
        """Treadmill telemetry callback: [pos, v, curr, a, cv, vr]."""
        t = clock.now()
        if self.recorder and self.recorder.armed:
            self.recorder.on_tread(t, vec6)

    def _belt_speed_for_metrics(self):
        """Best available belt speed for treadmill spatial gait estimates."""
        cur = getattr(self.treadmill_ctrl, "current_speed", 0.0)
        target = getattr(self.treadmill_ctrl, "fixed_vel", 0.0)
        return abs(cur) if abs(cur) > 0.03 else abs(target)

    # ---------- pose path ----------
    def _pose_loop(self):
        from .pose.detector import PoseDetector
        from .pose.filtering import PoseSmoother
        from .pose.triangulate import triangulate, reprojection_errors
        from .pose.angles import compute_angles
        p = _selected_pose_config(self.cfg)
        device = p.get("device", "cpu")
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("[engine] CUDA unavailable -> falling back to CPU pose")
                    device = "cpu"
                else:
                    torch.cuda.set_device(0)
            except Exception:
                device = "cpu"
        model = p.get("model", "yolo11n-pose.pt")
        model_abs = self.cfg.abspath(model)
        model = model_abs if model_abs and os.path.exists(model_abs) else model
        self._pose_det = PoseDetector(model,
                                      device=device,
                                      conf_thresh=p.get("conf_thresh", 0.3),
                                      imgsz=p.get("imgsz", 640),
                                      half=p.get("half", True),
                                      max_det=p.get("max_det", 1),
                                      fuse=p.get("fuse", True),
                                      warmup=p.get("warmup", True),
                                      warmup_shape=p.get("warmup_shape", [1280, 720]),
                                      warmup_batch=p.get("warmup_batch", 2))
        self._pose_smoother = PoseSmoother(
            alpha=self.pose_smoother_alpha,
            max_gap_s=self.pose_smoother_max_gap_s)
        tol = float(self.cfg.get("cameras", "frame_pair_tolerance_s", default=0.02))
        target_dt = 1.0 / float(p.get("target_fps", 30))
        last = {0: -1, 1: -1}
        self.state.set_status(calibrated_3d=self.stereo is not None,
                              pose_device=device,
                              pose_half=bool(p.get("half", True) and device == "cuda"),
                              pose_imgsz=int(p.get("imgsz", 640) or 640),
                              pose_model=os.path.basename(str(model)),
                              pose_profile=p.get("profile", "custom"))

        while not self._stop.is_set():
            t_loop = clock.mono()
            s0, seq0 = self.cams[0].read_latest()
            s1, seq1 = self.cams[1].read_latest()
            if s0 is None or (seq0 == last[0] and seq1 == last[1]):
                time.sleep(0.002)
                continue
            last[0], last[1] = seq0, seq1

            kp1 = None
            if s1 is not None:
                kp0, kp1 = self._pose_det.detect_many([s0.value, s1.value])
                self.state.frame[0].set(s0.value, s0.t)
                self.state.kp2d[0].set(kp0, s0.t)
                self.state.frame[1].set(s1.value, s1.t)
                self.state.kp2d[1].set(kp1, s1.t)
            else:
                kp0 = self._pose_det.detect(s0.value)
                self.state.frame[0].set(s0.value, s0.t)
                self.state.kp2d[0].set(kp0, s0.t)

            if self.recorder and self.recorder.armed:
                self.recorder.on_frame(0, s0.value, s0.t)
                if s1 is not None:
                    self.recorder.on_frame(1, s1.value, s1.t)

            # triangulate only if calibrated and frames are time-aligned
            if self.stereo is not None and kp1 is not None and abs(s0.t - s1.t) <= tol:
                pts3d_raw, valid_raw = triangulate(kp0, kp1, self.stereo,
                                                   conf_thresh=self._pose_det.conf_thresh)
                if self.pose_smoothing_enabled:
                    pts3d, valid = self._pose_smoother.update(pts3d_raw, valid_raw, s0.t)
                else:
                    pts3d, valid = pts3d_raw, valid_raw
                joint_err = reprojection_errors(pts3d_raw, valid_raw, kp0, kp1, self.stereo)
                mean_err = (float(np.nanmean(joint_err))
                            if np.any(np.isfinite(joint_err)) else np.nan)
                quality = {
                    "reprojection_error_px": mean_err,
                    "joint_reprojection_error_px": joint_err,
                    "valid_joint_count": int(np.sum(valid_raw)),
                    "smoothed": bool(self.pose_smoothing_enabled),
                }
                self.gait.update_pose(pts3d, valid)
                angles = compute_angles(pts3d, valid)
                self.state.kp3d.set((pts3d, valid), s0.t)
                self.state.angles.set(angles, s0.t)
                self.state.angle_hist.append(
                    s0.t, [angles.get(k, np.nan) for k in ANGLE_NAMES])
                self.state.set_status(pose_valid_joints=int(np.sum(valid_raw)),
                                      pose_reproj_error_px=mean_err,
                                      pose_smoothing=bool(self.pose_smoothing_enabled))
                if self.recorder:
                    self.recorder.on_pose(s0.t, kp0, kp1, pts3d, valid, angles,
                                          self.gait.metrics(self._belt_speed_for_metrics()),
                                          quality=quality, kp3d_raw=pts3d_raw)
            else:
                if self._pose_smoother is not None:
                    self._pose_smoother.reset()
                if self.recorder:
                    self.recorder.on_pose(s0.t, kp0, kp1, None, None, None,
                                          self.gait.metrics(self._belt_speed_for_metrics()))

            # fps + status
            self._pose_fps_n += 1
            if clock.mono() - self._pose_fps_t >= 0.5:
                self.state.set_status(
                    pose_fps=self._pose_fps_n / (clock.mono() - self._pose_fps_t),
                    cam0_fps=self.cams[0].fps_est, cam1_fps=self.cams[1].fps_est,
                    force_fps=getattr(self.force, "fps_est", 0.0),
                    cam0_latency_ms=(clock.now() - s0.t) * 1e3,
                    cam1_latency_ms=(clock.now() - s1.t) * 1e3 if s1 else 0.0,
                    session_t=(clock.now() - self.session_t0) if self.session_t0 else 0.0,
                    recording=bool(self.recorder and self.recorder.armed))
                self._pose_fps_n = 0
                self._pose_fps_t = clock.mono()

            # pace to target fps
            sleep = target_dt - (clock.mono() - t_loop)
            if sleep > 0:
                time.sleep(sleep)

    # ---------- lifecycle ----------
    def start(self):
        with self._lifecycle_lock:
            if self._running:
                return
            self._stop.clear()
            self.session_t0 = None
            self._pose_fps_n = 0
            self._pose_fps_t = clock.mono()
            if self.calib.mode == "bodyweight" and self.calib.scale is None:
                self._bw_left.clear()
                self._bw_right.clear()
                self.state.set_status(force_auto_bw_ready=False,
                                      force_auto_bw_samples=0,
                                      force_auto_bw_needed=self._bw_target_samples)
            for c in self.cams.values():
                c.start()
            self.force.start(self._on_force)
            self._pose_thread = threading.Thread(target=self._pose_loop, daemon=True,
                                                 name="pose")
            self._pose_thread.start()
            self._running = True

    def stop(self):
        with self._lifecycle_lock:
            if not self._running:
                if self.recorder and self.recorder.armed:
                    self.recorder.stop_and_save()
                return
            self._stop.set()
            for c in self.cams.values():
                c.stop()
            self.force.stop()
            if self._pose_thread:
                self._pose_thread.join(timeout=3.0)
            self._pose_thread = None
            self._running = False
            if self.recorder and self.recorder.armed:
                self.recorder.stop_and_save()
