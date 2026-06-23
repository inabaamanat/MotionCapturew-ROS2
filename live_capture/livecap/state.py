"""Shared, thread-safe live state read by the GUI and written by workers.

Holds the freshest frame/pose per camera, recent force and joint-angle history
(ring buffers for scrolling plots), and live gait metrics. Producers never block
on the GUI; the GUI reads snapshots.
"""
from __future__ import annotations

import threading

from .buffers import LatestSlot, RingBuffer

# 12 force channels + total/derived handled downstream; store the 12 raw->N here.
FORCE_WIDTH = 12
# 3D joint-angle vector layout for the angle ring buffer / plots.
ANGLE_NAMES = [
    "hip_flexion_l", "hip_flexion_r",
    "hip_adduction_l", "hip_adduction_r",
    "hip_abduction_l", "hip_abduction_r",
    "knee_angle_l", "knee_angle_r",
    "knee_flexion_l", "knee_flexion_r",
    "ankle_angle_l", "ankle_angle_r",
    "ankle_dorsiflexion_l", "ankle_dorsiflexion_r",
    "thigh_length_l_m", "thigh_length_r_m",
    "shank_length_l_m", "shank_length_r_m",
]


class SharedState:
    def __init__(self, plot_window_s: float, force_rate_hz: float,
                 angle_rate_hz: float = 60.0):
        # latest frames + 2D keypoints per camera
        self.frame = {0: LatestSlot(), 1: LatestSlot()}
        self.kp2d = {0: LatestSlot(), 1: LatestSlot()}
        # latest triangulated 3D keypoints (COCO-17, metres) and joint angles
        self.kp3d = LatestSlot()
        self.angles = LatestSlot()          # dict name -> deg
        self.gait = LatestSlot()            # dict of live spatiotemporal metrics

        # scrolling histories
        cap_force = int(max(1.0, plot_window_s) * force_rate_hz * 1.5)
        cap_ang = int(max(1.0, plot_window_s) * angle_rate_hz * 1.5)
        self.force_hist = RingBuffer(cap_force, FORCE_WIDTH)     # calibrated N
        self.cop_hist = RingBuffer(cap_ang, 4)                   # Lx,Ly,Rx,Ry
        self.angle_hist = RingBuffer(cap_ang, len(ANGLE_NAMES))

        # status / diagnostics
        self._lock = threading.Lock()
        self.status = {
            "cam0_fps": 0.0, "cam1_fps": 0.0, "pose_fps": 0.0,
            "force_fps": 0.0, "cam0_latency_ms": 0.0, "cam1_latency_ms": 0.0,
            "dropped": 0, "calibrated_3d": False, "recording": False,
            "session_t": 0.0, "force_scale_n_per_v": 0.0,
            "pose_valid_joints": 0, "pose_reproj_error_px": 0.0,
            "pose_smoothing": False, "pose_device": "unknown",
            "pose_half": False, "pose_imgsz": 0,
            "force_auto_bw_ready": False, "force_auto_bw_samples": 0,
            "force_auto_bw_needed": 0, "force_auto_bw_cv": 0.0,
            # treadmill
            "treadmill_mode": "FIXED", "treadmill_target_vel": 0.0,
            "treadmill_current_vel": 0.0, "treadmill_incline": 0.0,
            "treadmill_connected": False,
        }

    def set_status(self, **kw):
        with self._lock:
            self.status.update(kw)

    def get_status(self) -> dict:
        with self._lock:
            return dict(self.status)
