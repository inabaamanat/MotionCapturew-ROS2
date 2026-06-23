"""High-precision offline pose processing for saved trial videos."""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..calib import load_stereo
from ..pose.angles import compute_angles
from ..pose.detector import COCO_NAMES, PoseDetector
from ..pose.filtering import PoseSmoother
from ..pose.triangulate import reprojection_errors, triangulate
from ..state import ANGLE_NAMES


def _profile_config(cfg) -> dict[str, Any]:
    """Resolve the offline processing config, falling back to pose profiles."""
    live_pose = dict(cfg.get("pose", default={}) or {})
    offline = dict(cfg.get("offline_processing", default={}) or {})
    profiles = live_pose.get("profiles") or {}
    name = offline.get("profile") or "precision_4080"
    pose = dict(live_pose)
    if isinstance(profiles, dict) and name in profiles:
        pose.update(profiles.get(name) or {})
    pose.update({k: v for k, v in offline.items() if k != "enabled"})
    pose["profile"] = name
    return pose


def _video_frame_count(path: Path) -> int:
    if not path.exists():
        return 0
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def _source_times(rec_dir: Path, n_frames: int, fps: float) -> tuple[np.ndarray, float]:
    pose_path = rec_dir / "pose.npz"
    if pose_path.exists():
        with np.load(pose_path, allow_pickle=True) as z:
            old_t = np.asarray(z.get("t", np.zeros(0)), float)
            start_epoch = float(z.get("start_epoch", old_t[0] if old_t.size else time.time()))
        if old_t.size:
            if n_frames <= old_t.size:
                return old_t[:n_frames], start_epoch
            extra_dt = 1.0 / max(fps, 1.0)
            extra = old_t[-1] + extra_dt * np.arange(1, n_frames - old_t.size + 1)
            return np.concatenate([old_t, extra]), start_epoch
        return start_epoch + np.arange(n_frames, dtype=float) / max(fps, 1.0), start_epoch
    start_epoch = time.time()
    return start_epoch + np.arange(n_frames, dtype=float) / max(fps, 1.0), start_epoch


def _open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    return cap if cap.isOpened() else None


def _read_pair(cap0, cap1):
    ok0, f0 = cap0.read() if cap0 is not None else (False, None)
    ok1, f1 = cap1.read() if cap1 is not None else (False, None)
    return (f0 if ok0 else None), (f1 if ok1 else None)


def _selected_model(pose_cfg: dict[str, Any], cfg) -> str:
    model = pose_cfg.get("model", "yolo11x-pose.pt")
    model_abs = cfg.abspath(model)
    return model_abs if model_abs and os.path.exists(model_abs) else model


def reprocess_trial_pose(recording_dir: str | os.PathLike, cfg,
                         force: bool = False, progress=None) -> Path | None:
    """Rerun pose estimation from saved MP4s with the offline precision profile.

    The live pose file is retained as ``pose_live.npz`` the first time this runs.
    ``pose.npz`` is then replaced with high-precision 2D detections, triangulated
    3D joints, quality metrics, and joint angles.
    """
    def emit(stage, current=0, total=1, message=""):
        if progress is not None:
            progress(stage, current, total, message)

    offline = cfg.get("offline_processing", default={}) or {}
    if not bool(offline.get("enabled", True)):
        emit("precision_pose", 1, 1, "Offline precision processing disabled")
        return None
    rec_dir = Path(recording_dir)
    cam0_path = rec_dir / "cam0.mp4"
    cam1_path = rec_dir / "cam1.mp4"
    if not cam0_path.exists() and not cam1_path.exists():
        emit("precision_pose", 1, 1, "No saved camera videos found")
        return None
    out_path = rec_dir / "pose.npz"
    marker = rec_dir / "pose_precision.npz"
    if marker.exists() and not force:
        emit("precision_pose", 1, 1, "Using existing precision pose file")
        shutil.copy2(marker, out_path)
        return out_path
    if out_path.exists() and not (rec_dir / "pose_live.npz").exists():
        shutil.copy2(out_path, rec_dir / "pose_live.npz")

    cap0 = _open_video(cam0_path)
    cap1 = _open_video(cam1_path)
    src_cap = cap0 if cap0 is not None else cap1
    if src_cap is None:
        emit("precision_pose", 1, 1, "Could not open saved camera videos")
        return None
    fps = float(src_cap.get(cv2.CAP_PROP_FPS) or
                cfg.get("pose", "target_fps", default=30) or 30)
    n0, n1 = _video_frame_count(cam0_path), _video_frame_count(cam1_path)
    n_frames = max(n0, n1)
    if cap0 is not None and cap1 is not None:
        n_frames = min(n0, n1) if n0 and n1 else max(n0, n1)
    if n_frames <= 0:
        emit("precision_pose", 1, 1, "Saved camera videos contain no frames")
        return None

    pose_cfg = _profile_config(cfg)
    emit("load_model", 0, 1,
         f"Loading {pose_cfg.get('model', 'pose model')} at imgsz {pose_cfg.get('imgsz', 960)}")
    device = pose_cfg.get("device", "cuda")
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"
    det = PoseDetector(
        _selected_model(pose_cfg, cfg),
        device=device,
        conf_thresh=pose_cfg.get("conf_thresh", 0.20),
        imgsz=pose_cfg.get("imgsz", 960),
        half=pose_cfg.get("half", True),
        max_det=pose_cfg.get("max_det", 1),
        fuse=pose_cfg.get("fuse", True),
        warmup=pose_cfg.get("warmup", True),
        warmup_shape=pose_cfg.get("warmup_shape", [720, 1280]),
        warmup_batch=pose_cfg.get("warmup_batch", 2),
    )
    emit("load_model", 1, 1, f"Model ready on {device}")
    stereo = load_stereo(cfg.abspath(cfg.get("calibration", "cameras", "extrinsics")))
    smoothing_cfg = pose_cfg.get("smoothing", {}) or {}
    smoother = PoseSmoother(alpha=smoothing_cfg.get("alpha", 0.45),
                            max_gap_s=smoothing_cfg.get("max_gap_s", 0.35))
    smoothing_enabled = bool(smoothing_cfg.get("enabled", True))
    times, start_epoch = _source_times(rec_dir, n_frames, fps)

    kp0_rows, kp1_rows = [], []
    kp3d_rows, kp3d_raw_rows, valid_rows = [], [], []
    angles_rows, reproj_rows, joint_reproj_rows, valid_count_rows = [], [], [], []
    t0 = time.perf_counter()
    try:
        for i in range(n_frames):
            if i == 0:
                emit("precision_pose", 0, n_frames, "Running precision pose on saved videos")
            f0, f1 = _read_pair(cap0, cap1)
            if f0 is None and f1 is None:
                break
            if f0 is not None and f1 is not None:
                kp0, kp1 = det.detect_many([f0, f1])
            elif f0 is not None:
                kp0 = det.detect(f0)
                kp1 = np.full_like(kp0, np.nan)
            else:
                kp1 = det.detect(f1)
                kp0 = np.full_like(kp1, np.nan)

            pts_raw = np.full((len(COCO_NAMES), 3), np.nan)
            pts = pts_raw.copy()
            valid = np.zeros(len(COCO_NAMES), bool)
            joint_err = np.full(len(COCO_NAMES), np.nan)
            if stereo is not None and f0 is not None and f1 is not None:
                pts_raw, valid_raw = triangulate(
                    kp0, kp1, stereo,
                    conf_thresh=float(pose_cfg.get("conf_thresh", 0.20)))
                joint_err = reprojection_errors(pts_raw, valid_raw, kp0, kp1, stereo)
                if smoothing_enabled:
                    pts, valid = smoother.update(pts_raw, valid_raw, float(times[i]))
                else:
                    pts, valid = pts_raw, valid_raw

            angles = compute_angles(pts, valid)
            kp0_rows.append(kp0)
            kp1_rows.append(kp1)
            kp3d_rows.append(pts)
            kp3d_raw_rows.append(pts_raw)
            valid_rows.append(valid)
            angles_rows.append([angles.get(k, np.nan) for k in ANGLE_NAMES])
            valid_count_rows.append(int(np.sum(valid)))
            joint_reproj_rows.append(joint_err)
            reproj_rows.append(float(np.nanmean(joint_err))
                               if np.any(np.isfinite(joint_err)) else np.nan)
            if (i + 1) % 10 == 0 or i + 1 == n_frames:
                emit("precision_pose", i + 1, n_frames,
                     f"Precision pose frame {i + 1}/{n_frames}")
    finally:
        if cap0 is not None:
            cap0.release()
        if cap1 is not None:
            cap1.release()

    n = len(kp0_rows)
    if n == 0:
        emit("precision_pose", 1, 1, "No pose frames were processed")
        return None
    payload = {
        "t": times[:n],
        "start_epoch": np.array(start_epoch),
        "kp0": np.asarray(kp0_rows),
        "kp1": np.asarray(kp1_rows),
        "kp3d": np.asarray(kp3d_rows),
        "kp3d_raw": np.asarray(kp3d_raw_rows),
        "valid": np.asarray(valid_rows),
        "angles": np.asarray(angles_rows, float),
        "angle_names": np.asarray(ANGLE_NAMES),
        "reprojection_error_px": np.asarray(reproj_rows, float),
        "joint_reprojection_error_px": np.asarray(joint_reproj_rows, float),
        "valid_joint_count": np.asarray(valid_count_rows, float),
        "coco_names": np.asarray(COCO_NAMES),
        "offline_precision": np.array(True),
        "offline_processing_seconds": np.array(time.perf_counter() - t0),
        "offline_pose_profile": np.array(str(pose_cfg.get("profile", "precision"))),
        "offline_pose_model": np.array(str(pose_cfg.get("model", ""))),
        "offline_pose_imgsz": np.array(int(pose_cfg.get("imgsz", 960) or 960)),
    }
    np.savez_compressed(marker, **payload)
    shutil.copy2(marker, out_path)
    emit("precision_pose", n_frames, n_frames, "Precision pose file written")
    return out_path
