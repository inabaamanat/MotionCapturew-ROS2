"""Stereo triangulation of matched 2D keypoints into 3D (world = camera 0)."""
from __future__ import annotations

import cv2
import numpy as np


def _undistort(pts, K, dist):
    """Map pixel points to normalized->pixel coords with distortion removed.

    Returns points in the same pixel convention expected by the projection
    matrices (P = K[R|t]), i.e. re-applies K after undistortion.
    """
    p = pts.reshape(-1, 1, 2).astype(np.float64)
    und = cv2.undistortPoints(p, K, dist, P=K)
    return und.reshape(-1, 2)


def triangulate(kp0, kp1, calib, conf_thresh=0.3):
    """Triangulate COCO-17 keypoints from two views.

    kp0, kp1: (17,3) [x,y,conf]. Returns (pts3d (17,3) metres, valid mask (17,)).
    Points missing/low-confidence in either view are NaN.
    """
    n = kp0.shape[0]
    pts3d = np.full((n, 3), np.nan)
    valid = np.zeros(n, bool)

    ok = (np.isfinite(kp0[:, 0]) & np.isfinite(kp1[:, 0])
          & (kp0[:, 2] >= conf_thresh) & (kp1[:, 2] >= conf_thresh))
    if not ok.any():
        return pts3d, valid

    p0 = _undistort(kp0[ok, :2], calib.K0, calib.dist0).T   # 2 x m
    p1 = _undistort(kp1[ok, :2], calib.K1, calib.dist1).T
    X = cv2.triangulatePoints(calib.P0, calib.P1, p0, p1)   # 4 x m
    X = (X[:3] / X[3]).T                                    # m x 3
    pts3d[ok] = X
    valid[ok] = True
    return pts3d, valid


def reprojection_errors(pts3d, valid, kp0, kp1, calib) -> np.ndarray:
    """Per-joint mean reprojection error (px), NaN for invalid joints."""
    out = np.full(pts3d.shape[0], np.nan)
    if not valid.any():
        return out
    X = pts3d[valid]
    Xh = np.hstack([X, np.ones((X.shape[0], 1))]).T
    per_view = []
    for P, kp, K, dist in ((calib.P0, kp0, calib.K0, calib.dist0),
                           (calib.P1, kp1, calib.K1, calib.dist1)):
        proj = (P @ Xh)
        proj = (proj[:2] / proj[2]).T
        obs = _undistort(kp[valid, :2], K, dist)
        per_view.append(np.linalg.norm(proj - obs, axis=1))
    out[valid] = np.nanmean(np.vstack(per_view), axis=0)
    return out


def reprojection_error(pts3d, valid, kp0, kp1, calib) -> float:
    """Mean reprojection error (px) over valid points, for QC."""
    errs = reprojection_errors(pts3d, valid, kp0, kp1, calib)
    return float(np.nanmean(errs)) if np.any(np.isfinite(errs)) else float("nan")
    return float(np.nanmean(np.concatenate(errs)))


def make_virtual_stereo(image_size=(1280, 720), baseline=0.6, f=1000.0):
    """Build a synthetic StereoCalib (two parallel cameras) for unit testing."""
    from ..calib import StereoCalib, projection_matrices
    w, h = image_size
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)
    dist = np.zeros(5)
    R = np.eye(3)
    T = np.array([baseline, 0, 0.0])   # cam1 shifted along +x by baseline
    P0, P1 = projection_matrices(K, K, R, T)
    return StereoCalib(K0=K, dist0=dist, K1=K, dist1=dist, P0=P0, P1=P1,
                       R=R, T=T, image_size=image_size)
