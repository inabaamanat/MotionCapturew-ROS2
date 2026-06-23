"""2D body-keypoint detection on the raw video (YOLO pose), for drawing the
skeleton registered to the subject's body.

OpenCap markers live in 3D world coordinates and the session export does not
include the camera calibration, so they cannot be projected onto the image
directly. Instead we detect 2D keypoints on the video itself (perfectly
registered to the body) and annotate them with the trustworthy OpenCap joint
angles.

Keypoints are cached to disk so the (slow) detection runs only once per video.
COCO-17 keypoint order is used throughout.
"""
from __future__ import annotations

import os

import numpy as np

COCO_NAMES = ["nose", "Leye", "Reye", "Lear", "Rear", "Lshoulder", "Rshoulder",
              "Lelbow", "Relbow", "Lwrist", "Rwrist", "Lhip", "Rhip",
              "Lknee", "Rknee", "Lankle", "Rankle"]
IDX = {n: i for i, n in enumerate(COCO_NAMES)}

# bones as (a, b) index pairs and a side tag for colouring
SKELETON = [
    ("Lshoulder", "Rshoulder", "C"),
    ("Lshoulder", "Lhip", "L"), ("Rshoulder", "Rhip", "R"),
    ("Lhip", "Rhip", "C"),
    ("Lshoulder", "Lelbow", "L"), ("Lelbow", "Lwrist", "L"),
    ("Rshoulder", "Relbow", "R"), ("Relbow", "Rwrist", "R"),
    ("Lhip", "Lknee", "L"), ("Lknee", "Lankle", "L"),
    ("Rhip", "Rknee", "R"), ("Rknee", "Rankle", "R"),
]


def detect_pose(video_path: str, cache_path: str, conf_thresh: float = 0.3,
                model_name: str = "yolo11n-pose.pt", smooth: bool = True,
                force: bool = False) -> np.ndarray:
    """Return (n_frames, 17, 3) array of [x, y, confidence]. Cached to npz."""
    if cache_path and os.path.exists(cache_path) and not force:
        d = np.load(cache_path)
        return d["kp"]

    import cv2
    from ultralytics import YOLO

    model = YOLO(model_name)
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    kp = np.full((n, 17, 3), np.nan)

    for f in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        res = model(frame, verbose=False)[0]
        if res.keypoints is None or res.keypoints.xy is None or len(res.keypoints.xy) == 0:
            continue
        # pick the largest detected person (most central / highest box area)
        if res.boxes is not None and len(res.boxes) > 1:
            areas = (res.boxes.xywh[:, 2] * res.boxes.xywh[:, 3]).cpu().numpy()
            p = int(np.argmax(areas))
        else:
            p = 0
        xy = res.keypoints.xy[p].cpu().numpy()
        conf = (res.keypoints.conf[p].cpu().numpy()
                if res.keypoints.conf is not None else np.ones(len(xy)))
        kp[f, :, :2] = xy
        kp[f, :, 2] = conf
    cap.release()

    # low-confidence points -> NaN, then per-keypoint temporal smoothing
    kp[kp[:, :, 2] < conf_thresh, :2] = np.nan
    if smooth:
        kp = _smooth(kp)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, kp=kp)
    return kp


def _smooth(kp: np.ndarray, win: int = 7) -> np.ndarray:
    """Interpolate short gaps and apply a light moving-average per keypoint."""
    n, k, _ = kp.shape
    t = np.arange(n)
    out = kp.copy()
    kernel = np.ones(win) / win
    for j in range(k):
        for ax in range(2):
            v = kp[:, j, ax]
            good = np.isfinite(v)
            if good.sum() < 2:
                continue
            v = np.interp(t, t[good], v[good])
            v = np.convolve(v, kernel, mode="same")
            # restore NaN where there was a long gap (>0.5 s ~ 30 frames)
            out[:, j, ax] = v
    return out
