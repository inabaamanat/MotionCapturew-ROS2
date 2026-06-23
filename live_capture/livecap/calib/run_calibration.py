"""Interactive stereo calibration tool.

Usage:
    python -m livecap.calib.run_calibration --config ../config.yaml

Hold a printed checkerboard so BOTH cameras see it; press SPACE to capture a
pair, in varied positions/orientations (>=12 recommended). Press 'c' to compute
and save intrinsics + extrinsics, 'q' to quit. Works with live or replay cameras.
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from ..config import load_config
from ..sources.camera import make_cameras
from . import projection_matrices, StereoCalib
from .calibrate import calibrate_intrinsics, calibrate_stereo, find_corners


def _resize_to_height(frame, target_h):
    h, w = frame.shape[:2]
    scale = target_h / float(h)
    return cv2.resize(frame, (int(round(w * scale)), target_h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cb = cfg.get("calibration", "cameras", "checkerboard")
    cols, rows, sq = cb["cols"], cb["rows"], cb["square_m"]
    min_pairs = int(cb.get("min_pairs", 12))

    cams = make_cameras(cfg)
    for c in cams.values():
        c.start()

    grabs0, grabs1 = [], []
    print(f"SPACE=capture pair, c=compute+save, q=quit")
    print(f"checkerboard: {cols}x{rows} inner corners, {sq*1000:.0f} mm squares, "
          f"minimum {min_pairs} pairs")
    cv2.namedWindow("stereo calibration", cv2.WINDOW_NORMAL)
    while True:
        s0, _ = cams[0].read_latest()
        s1, _ = cams[1].read_latest()
        if s0 is None or s1 is None:
            continue
        f0, f1 = s0.value.copy(), s1.value.copy()
        g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        c0, c1 = find_corners(g0, cols, rows), find_corners(g1, cols, rows)
        for f, c in ((f0, c0), (f1, c1)):
            if c is not None:
                cv2.drawChessboardCorners(f, (cols, rows), c, True)
        disp = np.hstack([_resize_to_height(f0, 480), _resize_to_height(f1, 480)])
        cv2.putText(disp, f"pairs: {len(grabs0)}  both-visible: "
                    f"{c0 is not None and c1 is not None}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("stereo calibration", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" ") and c0 is not None and c1 is not None:
            grabs0.append(g0)
            grabs1.append(g1)
            print(f"captured pair {len(grabs0)}")
        elif k == ord("c"):
            if len(grabs0) < min_pairs:
                print(f"need >={min_pairs} captured pairs before computing, got {len(grabs0)}")
                continue
            break
        elif k == ord("q"):
            cv2.destroyAllWindows()
            for c in cams.values():
                c.stop()
            return

    cv2.destroyAllWindows()
    for c in cams.values():
        c.stop()

    print("computing intrinsics...")
    i0 = calibrate_intrinsics(grabs0, cols, rows, sq)
    i1 = calibrate_intrinsics(grabs1, cols, rows, sq)
    print(f"  cam0 rms={i0['rms']:.3f}px  cam1 rms={i1['rms']:.3f}px")
    print("computing stereo extrinsics...")
    calib, rms = calibrate_stereo(list(zip(grabs0, grabs1)), i0["K"], i0["dist"],
                                  i1["K"], i1["dist"], cols, rows, sq)
    print(f"  stereo rms={rms:.3f}px  baseline={np.linalg.norm(calib.T):.3f} m")

    out = cfg.abspath(cfg.get("calibration", "cameras", "extrinsics"))
    cam0_out = cfg.abspath(cfg.get("calibration", "cameras", "intrinsics_cam0"))
    cam1_out = cfg.abspath(cfg.get("calibration", "cameras", "intrinsics_cam1"))
    for path in (out, cam0_out, cam1_out):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    calib.save(out, metadata={
        "checkerboard_cols": np.array(cols),
        "checkerboard_rows": np.array(rows),
        "checkerboard_square_m": np.array(sq),
        "captured_pairs": np.array(len(grabs0)),
        "stereo_rms_px": np.array(rms),
    })
    np.savez(cam0_out, K=i0["K"], dist=i0["dist"],
             image_size=np.array(i0["image_size"]))
    np.savez(cam1_out, K=i1["K"], dist=i1["dist"],
             image_size=np.array(i1["image_size"]))
    print(f"saved calibration -> {out}")


if __name__ == "__main__":
    main()
