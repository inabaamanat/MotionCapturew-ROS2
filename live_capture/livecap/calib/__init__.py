"""Camera calibration: per-camera intrinsics + stereo extrinsics -> projection
matrices used for 3D triangulation."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class StereoCalib:
    """Everything needed to triangulate: intrinsics, distortion, projections."""
    K0: np.ndarray
    dist0: np.ndarray
    K1: np.ndarray
    dist1: np.ndarray
    P0: np.ndarray            # 3x4 projection matrix, camera 0 (world = cam0)
    P1: np.ndarray            # 3x4 projection matrix, camera 1
    R: np.ndarray             # rotation cam0->cam1
    T: np.ndarray             # translation cam0->cam1 (metres)
    image_size: tuple

    def save(self, path: str, metadata: dict | None = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        extra = metadata or {}
        np.savez(path, K0=self.K0, dist0=self.dist0, K1=self.K1, dist1=self.dist1,
                 P0=self.P0, P1=self.P1, R=self.R, T=self.T,
                 image_size=np.array(self.image_size), **extra)


def load_stereo(path: str) -> StereoCalib | None:
    if not path or not os.path.exists(path):
        return None
    d = np.load(path)
    return StereoCalib(K0=d["K0"], dist0=d["dist0"], K1=d["K1"], dist1=d["dist1"],
                       P0=d["P0"], P1=d["P1"], R=d["R"], T=d["T"],
                       image_size=tuple(d["image_size"]))


def projection_matrices(K0, K1, R, T) -> tuple[np.ndarray, np.ndarray]:
    """World frame = camera 0. P0 = K0[I|0], P1 = K1[R|T]."""
    P0 = K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K1 @ np.hstack([R, T.reshape(3, 1)])
    return P0, P1
