"""Checkerboard calibration routines (intrinsics + stereo extrinsics)."""
from __future__ import annotations

import numpy as np
import cv2

from . import StereoCalib, projection_matrices


def _object_points(cols, rows, square_m):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * square_m


def find_corners(gray, cols, rows):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)


def calibrate_intrinsics(images_gray, cols, rows, square_m):
    """images_gray: list of grayscale frames containing the board."""
    objp = _object_points(cols, rows, square_m)
    objpoints, imgpoints = [], []
    size = None
    for g in images_gray:
        size = g.shape[::-1]
        c = find_corners(g, cols, rows)
        if c is not None:
            objpoints.append(objp)
            imgpoints.append(c)
    if len(objpoints) < 5:
        raise RuntimeError(f"need >=5 valid board views, got {len(objpoints)}")
    rms, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
    return {"K": K, "dist": dist, "rms": rms, "image_size": size,
            "n_views": len(objpoints)}


def calibrate_stereo(pairs_gray, K0, dist0, K1, dist1, cols, rows, square_m):
    """pairs_gray: list of (gray0, gray1) with the board visible in both."""
    objp = _object_points(cols, rows, square_m)
    objpoints, ip0, ip1 = [], [], []
    size = None
    for g0, g1 in pairs_gray:
        size = g0.shape[::-1]
        c0 = find_corners(g0, cols, rows)
        c1 = find_corners(g1, cols, rows)
        if c0 is not None and c1 is not None:
            objpoints.append(objp)
            ip0.append(c0)
            ip1.append(c1)
    if len(objpoints) < 5:
        raise RuntimeError(f"need >=5 valid stereo pairs, got {len(objpoints)}")
    flags = cv2.CALIB_FIX_INTRINSIC
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    rms, *_, R, T, _, _ = cv2.stereoCalibrate(
        objpoints, ip0, ip1, K0, dist0, K1, dist1, size,
        criteria=crit, flags=flags)
    P0, P1 = projection_matrices(K0, K1, R, T)
    calib = StereoCalib(K0=K0, dist0=dist0, K1=K1, dist1=dist1, P0=P0, P1=P1,
                        R=R, T=T, image_size=size)
    return calib, rms
