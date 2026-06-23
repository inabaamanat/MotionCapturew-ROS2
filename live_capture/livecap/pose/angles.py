"""3D joint angles from triangulated COCO-17 keypoints.

A pelvis-anchored anatomical frame is built each frame:
    up (U)        ~ shoulder_mid - hip_mid        (trunk long axis)
    medio-lat (M) ~ Rhip - Lhip, orthogonalized   (points to subject's right)
    anterior (A)  = U x M                          (forward)

Angles returned (degrees):
    knee_flexion_{l,r}     flexion (0 = straight leg)
    hip_flexion_{l,r}      thigh forward(+)/back(-) in the sagittal plane
    hip_adduction_{l,r}    thigh toward(+)/away(-) from midline (frontal plane)

Ankle angle needs foot keypoints, which COCO-17 lacks; it is reported as NaN
here (use a foot-aware model such as Halpe-26, or the offline OpenSim path).
"""
from __future__ import annotations

import numpy as np

from .detector import IDX


def _u(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * np.nan


def _pelvis_frame(P):
    hip_mid = 0.5 * (P[IDX["Lhip"]] + P[IDX["Rhip"]])
    sho_mid = 0.5 * (P[IDX["Lshoulder"]] + P[IDX["Rshoulder"]])
    U = _u(sho_mid - hip_mid)
    M = _u(P[IDX["Rhip"]] - P[IDX["Lhip"]])
    M = _u(M - np.dot(M, U) * U)          # orthogonalize to U
    A = _u(np.cross(U, M))                 # anterior
    return hip_mid, U, M, A


def _ok(P, valid, *names):
    return all(valid[IDX[n]] and np.all(np.isfinite(P[IDX[n]])) for n in names)


def _segment(P, valid, a, b):
    if not _ok(P, valid, a, b):
        return np.full(3, np.nan)
    return P[IDX[b]] - P[IDX[a]]


def compute_angles(pts3d: np.ndarray, valid: np.ndarray) -> dict:
    P = pts3d
    valid = np.asarray(valid, bool)
    out = {k: np.nan for k in (
        "knee_angle_l", "knee_angle_r", "knee_flexion_l", "knee_flexion_r",
        "hip_flexion_l", "hip_flexion_r", "hip_adduction_l", "hip_adduction_r",
        "hip_abduction_l", "hip_abduction_r", "ankle_angle_l", "ankle_angle_r",
        "ankle_dorsiflexion_l", "ankle_dorsiflexion_r",
        "thigh_length_l_m", "thigh_length_r_m", "shank_length_l_m",
        "shank_length_r_m", "valid_joint_count")}
    out["valid_joint_count"] = int(np.sum(valid))
    if not _ok(P, valid, "Lhip", "Rhip", "Lshoulder", "Rshoulder"):
        return out
    _, U, M, A = _pelvis_frame(P)
    down = -U

    for side in ("l", "r"):
        S = side.upper()
        hip, knee, ankle = f"{S}hip", f"{S}knee", f"{S}ankle"
        thigh_seg = _segment(P, valid, hip, knee)
        shank_seg = _segment(P, valid, knee, ankle)
        if np.all(np.isfinite(thigh_seg)):
            out[f"thigh_length_{side}_m"] = float(np.linalg.norm(thigh_seg))
            thigh = _u(P[IDX[knee]] - P[IDX[hip]])
            out[f"hip_flexion_{side}"] = np.degrees(
                np.arctan2(np.dot(thigh, A), np.dot(thigh, down)))
            toward_mid = -M if side == "r" else M
            out[f"hip_adduction_{side}"] = np.degrees(
                np.arctan2(np.dot(thigh, toward_mid), np.dot(thigh, down)))
            out[f"hip_abduction_{side}"] = -out[f"hip_adduction_{side}"]
        if np.all(np.isfinite(shank_seg)):
            out[f"shank_length_{side}_m"] = float(np.linalg.norm(shank_seg))
        if _ok(P, valid, hip, knee, ankle):
            thigh_v = P[IDX[hip]] - P[IDX[knee]]
            shank_v = P[IDX[ankle]] - P[IDX[knee]]
            cosang = np.dot(_u(thigh_v), _u(shank_v))
            inc = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            flex = 180.0 - inc
            out[f"knee_angle_{side}"] = flex
            out[f"knee_flexion_{side}"] = flex
            out[f"ankle_dorsiflexion_{side}"] = out[f"ankle_angle_{side}"]
    return out
