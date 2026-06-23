"""Video metadata, frame access, and raw<->synced alignment.

OpenCap's kinematics/markers are sampled 1:1 with the *trimmed sync* video
(both 60 Hz). To overlay onto the higher-resolution *raw* Cam0 video we must
recover which raw frame corresponds to OpenCap time 0. We do this by matching a
few sync frames against the raw frames (normalized grayscale correlation) and
fitting raw_frame = slope * opencap_frame + intercept.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int
    duration: float


@dataclass
class VideoAlignment:
    slope: float        # raw_frame index per opencap (sync) frame
    intercept: float    # raw frame index at opencap time 0
    quality: float      # mean match correlation [0,1]

    def opencap_time_for_raw_frame(self, raw_idx: int, opencap_fps: float) -> float:
        sync_idx = (raw_idx - self.intercept) / self.slope
        return sync_idx / opencap_fps


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return VideoInfo(path, fps, n, w, h, n / fps if fps else float("nan"))


def _read_gray(cap, idx, size):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        return None
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, size).astype(np.float32)


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.sum(a * b) / d) if d > 0 else -1.0


def align_raw_to_sync(raw_path: str, sync_path: str,
                      n_probe: int = 5, match_size=(96, 54)) -> VideoAlignment:
    """Find raw_frame = slope*sync_frame + intercept by template matching."""
    raw = probe(raw_path)
    sync = probe(sync_path)
    cap_raw = cv2.VideoCapture(raw_path)
    cap_sync = cv2.VideoCapture(sync_path)

    # match_size is (w, h); sync may be portrait, so size by its aspect.
    sw = match_size[0]
    sh = int(round(sw * sync.height / sync.width))
    size = (sw, sh)

    # The raw frame for sync frame s lies near s + (raw.n - sync.n) ... but the
    # trim could be at either end, so search the full plausible band.
    sync_idxs = np.linspace(5, sync.n_frames - 6, n_probe).astype(int)
    matches = []
    band = abs(raw.n_frames - sync.n_frames) + 15
    for s in sync_idxs:
        tmpl = _read_gray(cap_sync, s, size)
        if tmpl is None:
            continue
        best_r, best_j = -1.0, s
        for j in range(max(0, s - band), min(raw.n_frames, s + band + 1)):
            rg = _read_gray(cap_raw, j, size)
            if rg is None:
                continue
            r = _ncc(tmpl, rg)
            if r > best_r:
                best_r, best_j = r, j
        matches.append((s, best_j, best_r))

    cap_raw.release()
    cap_sync.release()

    s_arr = np.array([m[0] for m in matches], float)
    j_arr = np.array([m[1] for m in matches], float)
    q = float(np.mean([m[2] for m in matches])) if matches else 0.0
    if len(matches) >= 2:
        slope, intercept = np.polyfit(s_arr, j_arr, 1)
    else:
        slope, intercept = 1.0, (raw.n_frames - sync.n_frames)
    return VideoAlignment(slope=float(slope), intercept=float(intercept), quality=q)
