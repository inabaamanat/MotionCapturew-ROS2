"""Synchronized recorder.

Writes, for a session:
  * force.npz  -- fusion-compatible schema: DATA (14 x N) with rows 0-11 = volts,
                  row 12 = 0 (target placeholder), row 13 = elapsed seconds;
                  plus START (epoch). Loads directly in
                  treadmill_opencap_fusion/fusion/io_force.py.
  * pose.npz   -- timestamps, 2D keypoints (both cams), 3D keypoints, validity,
                  and joint angles.
  * live_metrics.csv -- per pose-frame: time, joint angles, live gait metrics,
                  nearest per-foot GRF and COP.
  * cam0.mp4 / cam1.mp4 (optional) -- frames actually processed, with timestamps.

So that the same offline overlay/QC used for the OpenCap trials runs unchanged.
"""
from __future__ import annotations

import os
import re
import time
import uuid
import json

import numpy as np

from . import clock
from .pose.detector import COCO_NAMES
from .pose.angles import compute_angles  # noqa: F401  (schema reference)
from .state import ANGLE_NAMES
from .trials.analysis import build_trial_artifacts

ANGLE_KEYS = list(ANGLE_NAMES)
GAIT_KEYS = ["contact_L", "contact_R", "phase_L", "phase_R",
             "cadence_steps_per_min", "stride_time_L_s", "stride_time_R_s",
             "stance_time_L_s", "stance_time_R_s",
             "swing_time_L_s", "swing_time_R_s",
             "duty_factor_L", "duty_factor_R",
             "stride_length_L_m", "stride_length_R_m",
             "stance_length_L_m", "stance_length_R_m",
             "stride_time_symmetry_pct", "stance_time_symmetry_pct",
             "stride_length_symmetry_pct"]
FORCE_KEYS = ["L_Fx", "L_Fy", "L_Fz", "L_Mx", "L_My", "L_Mz",
              "R_Fx", "R_Fy", "R_Fz", "R_Mx", "R_My", "R_Mz",
              "L_COPx", "L_COPy", "R_COPx", "R_COPy"]


class Recorder:
    def __init__(self, cfg, session_name=None):
        self.cfg = cfg
        self.armed = False
        self.session_name = session_name
        self.save_video = bool(cfg.get("recording", "save_raw_video", default=True))
        self._reset()
        self._writers = {}
        self.out_dir = None

    def _reset(self):
        self.session_id = str(uuid.uuid4())
        self.trial_id = str(uuid.uuid4())
        self.start_epoch = None
        self.f_t, self.f_volts, self.f_cal, self.f_events = [], [], [], []
        self.p_t, self.p_kp0, self.p_kp1 = [], [], []
        self.p_kp3d, self.p_kp3d_raw, self.p_valid = [], [], []
        self.p_ang, self.p_gait, self.p_quality = [], [], []
        self.tr_t, self.tr = [], []        # treadmill telemetry [pos,v,curr,a,cv,vr]

    def _trial_name(self):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if not self.session_name:
            return stamp
        label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(self.session_name)).strip("._")
        return f"{label}_{stamp}" if label else stamp

    @staticmethod
    def _unique_dir(path):
        if not os.path.exists(path):
            return path
        for i in range(2, 1000):
            cand = f"{path}_{i:02d}"
            if not os.path.exists(cand):
                return cand
        raise RuntimeError(f"could not create a unique recording folder near {path}")

    def arm(self):
        name = self._trial_name()
        base = self.cfg.abspath(self.cfg.get("recording", "output_dir",
                                             default="recordings"))
        self.out_dir = self._unique_dir(os.path.join(base, name))
        os.makedirs(self.out_dir, exist_ok=True)
        self._reset()
        self._write_initial_metadata(name)
        self.armed = True
        return self.out_dir

    def _write_initial_metadata(self, name):
        meta = {
            "session_id": self.session_id,
            "trial_id": self.trial_id,
            "recording_name": self.cfg.get("recording", "name", default=name) or name,
            "subject_id": self.cfg.get("recording", "subject_id", default=None),
            "notes": self.cfg.get("recording", "notes", default=""),
            "date_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "recording",
            "schema_version": "trial-recording-v1",
        }
        with open(os.path.join(self.out_dir, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    # ---- stream callbacks (called from engine threads) ----
    def on_force(self, t, volts12, calibrated, event):
        if not self.armed:
            return
        if self.start_epoch is None:
            self.start_epoch = t
        self.f_t.append(t)
        self.f_volts.append(np.asarray(volts12, float))
        self.f_cal.append([calibrated.get(k, np.nan) for k in FORCE_KEYS])
        self.f_events.append(event if event else ("", "", t))

    def on_pose(self, t, kp0, kp1, kp3d, valid, angles, gait=None,
                quality=None, kp3d_raw=None):
        if not self.armed:
            return
        if self.start_epoch is None:
            self.start_epoch = t
        self.p_t.append(t)
        self.p_kp0.append(kp0)
        self.p_kp1.append(kp1 if kp1 is not None else np.full((17, 3), np.nan))
        self.p_kp3d.append(kp3d if kp3d is not None else np.full((17, 3), np.nan))
        self.p_kp3d_raw.append(kp3d_raw if kp3d_raw is not None else
                               (kp3d if kp3d is not None else np.full((17, 3), np.nan)))
        self.p_valid.append(valid if valid is not None else np.zeros(17, bool))
        self.p_ang.append(angles if angles is not None else {})
        self.p_gait.append(gait if gait is not None else {})
        self.p_quality.append(quality if quality is not None else {})

    def on_tread(self, t, vec6):
        if not self.armed:
            return
        self.tr_t.append(t)
        self.tr.append(np.asarray(vec6, float))

    def on_frame(self, cam_id, frame, t):
        if not (self.armed and self.save_video):
            return
        import cv2
        w = self._writers.get(cam_id)
        if w is None:
            fps = max(1.0, self.cfg.get("gui", "refresh_hz", default=30))
            path = os.path.join(self.out_dir, f"cam{cam_id}.mp4")
            h, ww = frame.shape[:2]
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                self.cfg.get("pose", "target_fps", default=30),
                                (ww, h))
            self._writers[cam_id] = w
        w.write(frame)

    # ---- finalize ----
    def stop_and_save(self, process=False):
        if not self.armed:
            return None
        self.armed = False
        for w in self._writers.values():
            w.release()
        self._writers = {}
        if not self.f_t and not self.p_t:
            self._mark_raw_saved()
            return self.out_dir

        self._save_force()
        self._save_pose()
        self._save_csv()
        self._mark_raw_saved()
        if process:
            try:
                build_trial_artifacts(self.out_dir, self.cfg)
            except Exception as exc:
                print(f"[recorder] trial artifact build failed: {exc}")
        return self.out_dir

    def _mark_raw_saved(self):
        if not self.out_dir:
            return
        path = os.path.join(self.out_dir, "metadata.json")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            meta = {
                "session_id": self.session_id,
                "trial_id": self.trial_id,
                "recording_name": os.path.basename(self.out_dir),
                "schema_version": "trial-recording-v1",
            }
        duration = 0.0
        if self.start_epoch is not None:
            last = max(self.p_t[-1] if self.p_t else self.start_epoch,
                       self.f_t[-1] if self.f_t else self.start_epoch)
            duration = float(last - self.start_epoch)
        meta.update({
            "status": "raw_saved",
            "duration_s": duration,
            "raw_saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "raw_files": {
                "force": "force.npz" if os.path.exists(os.path.join(self.out_dir, "force.npz")) else None,
                "pose": "pose.npz" if os.path.exists(os.path.join(self.out_dir, "pose.npz")) else None,
                "cam0": "cam0.mp4" if os.path.exists(os.path.join(self.out_dir, "cam0.mp4")) else None,
                "cam1": "cam1.mp4" if os.path.exists(os.path.join(self.out_dir, "cam1.mp4")) else None,
            },
        })
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    def _save_force(self):
        if not self.f_t:
            return
        t = np.array(self.f_t)
        volts = np.array(self.f_volts).T            # 12 x N
        N = volts.shape[1]
        DATA = np.zeros((14, N))
        DATA[:12] = volts
        DATA[13] = t - t[0]                         # elapsed seconds
        # treadmill telemetry on its own (sparse) timeline: [pos,v,curr,a,cv,vr]
        tread = np.array(self.tr).T if self.tr else np.zeros((6, 0))
        tread_t = (np.array(self.tr_t) - self.start_epoch) if self.tr_t else np.zeros(0)
        ev_type, ev_side, ev_t = self._event_arrays()
        np.savez(os.path.join(self.out_dir, "force.npz"),
                 DATA=DATA, COMMS=np.zeros((3, N)), EXO=np.zeros((4, N)),
                 TREAD=tread, TREAD_t=tread_t, START=np.array(self.start_epoch),
                 CAL=np.array(self.f_cal),           # extra: calibrated + COP
                 EVENT_TYPE=ev_type, EVENT_SIDE=ev_side, EVENT_T=ev_t)

    def _event_arrays(self):
        seen = set()
        types, sides, times = [], [], []
        for ev in self.f_events:
            if not ev:
                continue
            typ, side, t = ev
            if not typ or not side:
                continue
            key = (str(typ), str(side), round(float(t) - float(self.start_epoch or t), 6))
            if key in seen:
                continue
            seen.add(key)
            types.append(str(typ))
            sides.append(str(side))
            times.append(float(t) - float(self.start_epoch or t))
        return np.array(types), np.array(sides), np.array(times, float)

    def _save_pose(self):
        if not self.p_t:
            return
        ang = np.array([[a.get(k, np.nan) for k in ANGLE_KEYS] for a in self.p_ang])
        np.savez_compressed(
            os.path.join(self.out_dir, "pose.npz"),
            t=np.array(self.p_t), start_epoch=np.array(self.start_epoch),
            kp0=np.array(self.p_kp0), kp1=np.array(self.p_kp1),
            kp3d=np.array(self.p_kp3d), kp3d_raw=np.array(self.p_kp3d_raw),
            valid=np.array(self.p_valid),
            angles=ang, angle_names=np.array(ANGLE_KEYS),
            reprojection_error_px=np.array([
                q.get("reprojection_error_px", np.nan) for q in self.p_quality
            ], float),
            joint_reprojection_error_px=np.array([
                q.get("joint_reprojection_error_px", np.full(17, np.nan))
                for q in self.p_quality
            ], float),
            valid_joint_count=np.array([
                q.get("valid_joint_count", np.nan) for q in self.p_quality
            ], float),
            coco_names=np.array(COCO_NAMES))

    def _save_csv(self):
        if not self.p_t:
            return
        import pandas as pd
        ft = np.array(self.f_t) if self.f_t else np.array([])
        fcal = np.array(self.f_cal) if self.f_cal else None
        rows = []
        for i, t in enumerate(self.p_t):
            row = {"time_s": t - self.start_epoch, "epoch": t}
            row.update({k: self.p_ang[i].get(k, np.nan) for k in ANGLE_KEYS})
            if i < len(self.p_quality):
                row["reprojection_error_px"] = self.p_quality[i].get(
                    "reprojection_error_px", np.nan)
                row["valid_joint_count"] = self.p_quality[i].get(
                    "valid_joint_count", np.nan)
            if i < len(self.p_gait):
                row.update({k: self.p_gait[i].get(k, np.nan) for k in GAIT_KEYS})
            if ft.size:
                j = int(np.argmin(np.abs(ft - t)))
                for ki, k in enumerate(FORCE_KEYS):
                    row[k] = fcal[j, ki]
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            os.path.join(self.out_dir, "live_metrics.csv"), index=False)
