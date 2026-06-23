"""2D body-keypoint detection (YOLO-pose), GPU-accelerated on the capture PC.

Returns COCO-17 keypoints as (17, 3) = [x, y, confidence] for the most prominent
person in a frame. The model is loaded once and reused.
"""
from __future__ import annotations

import numpy as np

# COCO-17 ordering (shared with treadmill_opencap_fusion/fusion/pose2d.py)
COCO_NAMES = ["nose", "Leye", "Reye", "Lear", "Rear", "Lshoulder", "Rshoulder",
              "Lelbow", "Relbow", "Lwrist", "Rwrist", "Lhip", "Rhip",
              "Lknee", "Rknee", "Lankle", "Rankle"]
IDX = {n: i for i, n in enumerate(COCO_NAMES)}

SKELETON = [
    ("Lshoulder", "Rshoulder", "C"),
    ("Lshoulder", "Lhip", "L"), ("Rshoulder", "Rhip", "R"),
    ("Lhip", "Rhip", "C"),
    ("Lshoulder", "Lelbow", "L"), ("Lelbow", "Lwrist", "L"),
    ("Rshoulder", "Relbow", "R"), ("Relbow", "Rwrist", "R"),
    ("Lhip", "Lknee", "L"), ("Lknee", "Lankle", "L"),
    ("Rhip", "Rknee", "R"), ("Rknee", "Rankle", "R"),
]


class PoseDetector:
    def __init__(self, model_name="yolo11n-pose.pt", device="cuda",
                 conf_thresh=0.3, imgsz=640, half=True, max_det=1,
                 fuse=True, warmup=True, warmup_shape=(1280, 720),
                 warmup_batch=2):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.device = device
        self.conf_thresh = float(conf_thresh)
        self.imgsz = int(imgsz or 640)
        self.half = bool(half and str(device).startswith(("cuda", "0")))
        self.max_det = int(max_det or 1)
        self.warmup_shape = tuple(int(x) for x in (warmup_shape or (1280, 720)))
        self.warmup_batch = max(1, int(warmup_batch or 1))
        self._torch = None
        try:
            import torch
            self._torch = torch
            if str(device).startswith(("cuda", "0")) and torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
        except Exception:
            self._torch = None
        if fuse:
            try:
                self.model.fuse()
            except Exception:
                pass
        if warmup:
            self.warmup()

    def _predict(self, frames):
        kwargs = {
            "verbose": False,
            "device": self.device,
            "imgsz": self.imgsz,
            "half": self.half,
            "max_det": self.max_det,
        }
        if self._torch is None:
            return self.model(frames, **kwargs)
        with self._torch.inference_mode():
            return self.model(frames, **kwargs)

    def warmup(self):
        if len(self.warmup_shape) != 2:
            self.warmup_shape = (1280, 720)
        frame = np.zeros((self.warmup_shape[0], self.warmup_shape[1], 3),
                         dtype=np.uint8)
        try:
            frames = [frame] * self.warmup_batch
            self._predict(frames if self.warmup_batch > 1 else frame)
        except Exception:
            pass

    def detect(self, frame) -> np.ndarray:
        """Return (17,3) [x,y,conf]; NaN where below confidence threshold."""
        return self._result_to_kp(self._predict(frame)[0])

    def detect_many(self, frames) -> list[np.ndarray]:
        """Return one COCO-17 keypoint array per frame using one batched call."""
        results = self._predict(list(frames))
        return [self._result_to_kp(res) for res in results]

    def _result_to_kp(self, res) -> np.ndarray:
        kp = np.full((17, 3), np.nan)
        if res.keypoints is None or res.keypoints.xy is None or len(res.keypoints.xy) == 0:
            return kp
        if res.boxes is not None and len(res.boxes) > 1:
            areas = (res.boxes.xywh[:, 2] * res.boxes.xywh[:, 3]).cpu().numpy()
            p = int(np.argmax(areas))
        else:
            p = 0
        xy = res.keypoints.xy[p].cpu().numpy()
        conf = (res.keypoints.conf[p].cpu().numpy()
                if res.keypoints.conf is not None else np.ones(len(xy)))
        kp[:, :2] = xy
        kp[:, 2] = conf
        kp[kp[:, 2] < self.conf_thresh, :2] = np.nan
        return kp
