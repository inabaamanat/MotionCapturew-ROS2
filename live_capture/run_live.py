#!/usr/bin/env python3
"""Launch the live capture application.

    python run_live.py --config config.yaml                 # DearPyGui app
    python run_live.py --config config.yaml --backend cv2    # OpenCV-window GUI
    python run_live.py --config config.yaml --mode live      # use live devices
    python run_live.py --config config.yaml --headless 8     # run 8s, no GUI (test)

Calibrate the two cameras first:
    python -m livecap.calib.run_calibration --config config.yaml
"""
import argparse
import os
import sys
import time

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from livecap.engine import Engine
from livecap.recorder import Recorder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    ap.add_argument("--backend", choices=["dpg", "cv2"], default="dpg")
    ap.add_argument("--mode", choices=["replay", "live"], default=None,
                    help="override config mode")
    ap.add_argument("--record", action="store_true",
                    help="arm recording immediately on start")
    ap.add_argument("--session", default=None,
                    help="optional recording label; a timestamp is appended")
    ap.add_argument("--headless", type=float, default=0.0,
                    help="run N seconds with no GUI (smoke test) then exit")
    args = ap.parse_args()

    engine = Engine(args.config, mode=args.mode)
    recorder = Recorder(engine.cfg,
                        session_name=args.session or engine.cfg.get("session_name"))
    engine.attach_recorder(recorder)

    if args.headless > 0:
        if args.record:
            recorder.arm()
        engine.start()
        time.sleep(args.headless)
        engine.stop()
        print("status:", engine.state.get_status())
        return

    if args.backend == "cv2":
        from livecap.gui.app import run_cv2_backend
        if args.record:
            recorder.arm()
        run_cv2_backend(engine, recorder, engine.cfg)
    else:
        from livecap.gui.app import DearPyGuiApp
        app = DearPyGuiApp(engine, recorder, engine.cfg)
        if args.record:
            recorder.arm()
        app.run()


if __name__ == "__main__":
    main()
