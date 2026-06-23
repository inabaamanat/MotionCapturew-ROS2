"""Trial-readiness checks for the live treadmill mocap system.

Run before a real session:
    python -m livecap.preflight --config config.yaml

This does not move the treadmill. It checks that the control-panel TCP port is
reachable, but speed/incline commands still happen only from the GUI.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import sys
import time

import numpy as np

from .config import load_config
from .calib import load_stereo
from .sources.camera import make_cameras
from .engine import _selected_pose_config


class Report:
    def __init__(self):
        self.failures = 0
        self.warnings = 0

    def ok(self, msg):
        print(f"[ OK ] {msg}")

    def warn(self, msg):
        self.warnings += 1
        print(f"[WARN] {msg}")

    def fail(self, msg):
        self.failures += 1
        print(f"[FAIL] {msg}")


def _check_output_dir(cfg, rep):
    base = cfg.abspath(cfg.get("recording", "output_dir", default="recordings"))
    probe = os.path.join(base, ".preflight_write_test")
    try:
        os.makedirs(probe, exist_ok=True)
        with open(os.path.join(probe, "ok.txt"), "w", encoding="utf-8") as fh:
            fh.write("ok\n")
        shutil.rmtree(probe)
        rep.ok(f"recording output is writable: {base}")
    except Exception as e:
        rep.fail(f"recording output is not writable: {base} ({e})")


def _check_checkerboard(cfg, rep):
    cb = cfg.get("calibration", "cameras", "checkerboard", default={})
    cols, rows = cb.get("cols"), cb.get("rows")
    square = float(cb.get("square_m", 0.0) or 0.0)
    min_pairs = int(cb.get("min_pairs", 0) or 0)
    if abs(square - 0.100) < 1e-6:
        rep.ok(f"checkerboard square size is 100 mm ({cols}x{rows} inner corners)")
    else:
        rep.fail(f"checkerboard square_m should be 0.100 for the 100 mm grid, got {square}")
    if min_pairs >= 12:
        rep.ok(f"camera calibration requires at least {min_pairs} captured pairs")
    else:
        rep.warn(f"camera calibration min_pairs is low ({min_pairs}); use at least 12")


def _check_stereo_calib(cfg, rep):
    path = cfg.abspath(cfg.get("calibration", "cameras", "extrinsics"))
    calib = load_stereo(path)
    if calib is None:
        rep.fail(f"stereo calibration missing: {path}")
        return
    baseline = float(np.linalg.norm(calib.T))
    rep.ok(f"stereo calibration present: {path}")
    rep.ok(f"stereo baseline: {baseline:.3f} m")
    try:
        d = np.load(path)
        if "checkerboard_square_m" in d:
            rep.ok(f"calibration metadata square: {float(d['checkerboard_square_m']):.3f} m")
        if "stereo_rms_px" in d:
            rms = float(d["stereo_rms_px"])
            msg = f"stereo RMS reprojection error: {rms:.3f} px"
            rep.ok(msg) if rms <= 1.5 else rep.warn(msg)
    except Exception as e:
        rep.warn(f"could not read calibration metadata ({e})")


def _check_model_and_cuda(cfg, rep):
    p = _selected_pose_config(cfg)
    model = p.get("model", "yolo11n-pose.pt")
    model_abs = cfg.abspath(model)
    if p.get("profile"):
        rep.ok(f"pose profile selected: {p.get('profile')}")
    rep.ok(f"pose model configured: {model}, imgsz {p.get('imgsz', 640)}, target {p.get('target_fps', 30)} fps")
    if model_abs and os.path.exists(model_abs):
        rep.ok(f"pose model found: {model_abs}")
    else:
        rep.warn(f"pose model not found locally ({model}); ultralytics may try to download it")
    if p.get("device", "cpu") == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                rep.ok(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                rep.warn("config requests CUDA, but torch reports CUDA unavailable")
        except Exception as e:
            rep.warn(f"could not check CUDA ({e})")


def _check_force_calibration(cfg, rep):
    fc = cfg.get("calibration", "force", default={})
    mode = fc.get("mode", "bodyweight")
    if mode == "matrix":
        mat = fc.get("matrix") or {}
        ok = True
        for side in ("left", "right"):
            arr = np.asarray(mat.get(side), float) if mat.get(side) is not None else None
            ok = ok and arr is not None and arr.shape == (6, 6)
        rep.ok("Bertec matrix calibration configured") if ok else rep.fail(
            "calibration.force.mode is matrix, but left/right 6x6 matrices are incomplete")
    elif mode == "gains":
        gains = fc.get("gains") or {}
        keys = [f"{s}_{c}" for s in ("L", "R") for c in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]
        missing = [k for k in keys if k not in gains]
        rep.ok("per-channel force gains configured") if not missing else rep.fail(
            f"calibration.force.gains is missing: {', '.join(missing)}")
    else:
        mass = fc.get("body_mass_kg")
        auto_s = float(fc.get("auto_bodyweight_seconds", 0.0) or 0.0)
        if mass and auto_s > 0:
            rep.warn(f"force calibration is bodyweight fallback: {mass} kg, auto-scale {auto_s:.1f}s")
        else:
            rep.fail("bodyweight force fallback needs body_mass_kg and auto_bodyweight_seconds")


def _check_cameras(cfg, rep, timeout_s):
    cams = make_cameras(cfg)
    try:
        for c in cams.values():
            c.start()
        deadline = time.time() + timeout_s
        seen = {0: None, 1: None}
        while time.time() < deadline and any(v is None for v in seen.values()):
            for i, c in cams.items():
                if seen[i] is None:
                    s, _ = c.read_latest()
                    if s is not None and s.value is not None:
                        seen[i] = s.value.shape
            time.sleep(0.05)
        for i in (0, 1):
            if seen[i] is None:
                rep.fail(f"cam{i} did not deliver a frame within {timeout_s:.1f}s")
            else:
                rep.ok(f"cam{i} frame received: {seen[i]}")
    finally:
        for c in cams.values():
            c.stop()


def _check_daq(cfg, rep):
    if cfg.get("mode", default="replay") != "live":
        path = cfg.abspath(cfg.get("force", "replay", "npz"))
        rep.ok(f"replay force file present: {path}") if path and os.path.exists(path) else rep.fail(
            f"replay force file missing: {path}")
        return
    live = cfg.get("force", "live")
    try:
        import nidaqmx
        from nidaqmx.system import System
        system = System.local()
        devices = {d.name: d for d in system.devices}
        dev_name = live["device"]
        if dev_name not in devices:
            rep.fail(f"NI-DAQ device {dev_name} not found; available: {sorted(devices)}")
            return
        rep.ok(f"NI-DAQ device found: {dev_name}")
        try:
            chans = {ch.name.split("/")[-1] for ch in devices[dev_name].ai_physical_chans}
            missing = [ch for ch in live["channels"] if ch not in chans]
            rep.ok("all configured analog input channels exist") if not missing else rep.fail(
                f"missing NI-DAQ channels on {dev_name}: {missing}")
        except Exception as e:
            rep.warn(f"could not enumerate analog input channels ({e})")
    except Exception as e:
        rep.fail(f"NI-DAQmx is not ready ({e})")


def _check_treadmill_port(rep, host="127.0.0.1", port=4000, timeout_s=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            rep.ok(f"treadmill control panel TCP is reachable at {host}:{port}")
    except Exception as e:
        rep.fail(f"treadmill TCP {host}:{port} is not reachable ({e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", choices=["live", "replay"], default=None)
    ap.add_argument("--camera-timeout", type=float, default=8.0)
    ap.add_argument("--skip-cameras", action="store_true")
    ap.add_argument("--skip-daq", action="store_true")
    ap.add_argument("--skip-treadmill", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.mode:
        cfg.raw["mode"] = args.mode
    rep = Report()
    print(f"Preflight mode: {cfg.get('mode', default='replay')}")
    _check_output_dir(cfg, rep)
    _check_checkerboard(cfg, rep)
    _check_stereo_calib(cfg, rep)
    _check_model_and_cuda(cfg, rep)
    _check_force_calibration(cfg, rep)
    if args.skip_cameras:
        rep.warn("camera check skipped")
    else:
        _check_cameras(cfg, rep, args.camera_timeout)
    if args.skip_daq:
        rep.warn("DAQ check skipped")
    else:
        _check_daq(cfg, rep)
    if args.skip_treadmill:
        rep.warn("treadmill TCP check skipped")
    else:
        _check_treadmill_port(rep)

    print(f"\nPreflight complete: {rep.failures} failure(s), {rep.warnings} warning(s)")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
