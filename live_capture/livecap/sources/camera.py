"""Camera sources with a latest-frame-only grabber (the key latency lever).

A dedicated grabber thread continuously reads the stream and stores only the most
recent frame. Consumers (the pose thread) pull that latest frame, so a slow
consumer never causes a growing backlog / latency creep.

* :class:`CameraSource`      -- live RTSP/SRT/UVC via OpenCV (low-buffer).
* :class:`ReplayCameraSource`-- streams a video file at real-time cadence.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import threading
import time

import cv2

try:
    cv2.setLogLevel(0)
except Exception:
    pass

from .. import clock
from ..buffers import LatestSlot


class _BaseCamera:
    def __init__(self, cam_id: int, transport_latency_s: float = 0.0):
        self.cam_id = cam_id
        self.transport_latency_s = float(transport_latency_s)
        self.slot = LatestSlot()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.fps_est = 0.0
        self._n = 0
        self._t0 = clock.mono()
        self.width = 0
        self.height = 0

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"cam{self.cam_id}")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def read_latest(self):
        """Return (Stamped(frame), seq). Stamped.t already includes latency offset."""
        return self.slot.get()

    def _publish(self, frame, capture_t: float):
        self.width, self.height = frame.shape[1], frame.shape[0]
        # subtract estimated transport latency so the timestamp reflects capture.
        self.slot.set(frame, capture_t - self.transport_latency_s)
        self._n += 1
        dt = clock.mono() - self._t0
        if dt >= 0.5:
            self.fps_est = self._n / dt
            self._n = 0
            self._t0 = clock.mono()

    def _run(self):  # pragma: no cover - overridden
        raise NotImplementedError


class CameraSource(_BaseCamera):
    """Live network/USB camera via OpenCV, tuned for low latency."""

    def __init__(self, cam_id, url, buffersize=1, ffmpeg_opts="",
                 transport_latency_s=0.0, width=None, height=None, fps=None,
                 backend=None):
        super().__init__(cam_id, transport_latency_s)
        self.url = url
        self.buffersize = buffersize
        self.ffmpeg_opts = ffmpeg_opts
        self.width_request = width
        self.height_request = height
        self.fps_request = fps
        self.backend_request = str(backend or "").strip().lower()

    def _open(self):
        if self.ffmpeg_opts:
            # OpenCV reads ffmpeg options from this env var (|-separated key;val).
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = self.ffmpeg_opts
        src = int(self.url) if str(self.url).isdigit() else self.url
        if self.backend_request in ("msmf", "media_foundation"):
            backend = cv2.CAP_MSMF
        elif self.backend_request in ("dshow", "directshow"):
            backend = cv2.CAP_DSHOW
        elif self.backend_request in ("any", "auto"):
            backend = cv2.CAP_ANY
        elif isinstance(src, str):
            backend = cv2.CAP_FFMPEG
        else:
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(src, backend)
        if self.width_request:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width_request))
        if self.height_request:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height_request))
        if self.fps_request:
            cap.set(cv2.CAP_PROP_FPS, float(self.fps_request))
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffersize)
        except Exception:
            pass
        return cap

    def _run(self):
        cap = self._open()
        backoff = 0.5
        while not self._stop.is_set():
            if not cap.isOpened():
                time.sleep(backoff)
                cap = self._open()
                continue
            ok, frame = cap.read()
            t = clock.now()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            self._publish(frame, t)
        cap.release()


class ReplayCameraSource(_BaseCamera):
    """Stream a video file at its native frame rate for hardware-free testing."""

    def __init__(self, cam_id, path, loop=True, fps=None, transport_latency_s=0.0):
        super().__init__(cam_id, transport_latency_s)
        self.path = path
        self.loop = loop
        self.fps_override = fps

    def _run(self):
        cap = cv2.VideoCapture(self.path)
        fps = self.fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
        period = 1.0 / fps
        next_t = clock.mono()
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                if self.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            self._publish(frame, clock.now())
            next_t += period
            sleep = next_t - clock.mono()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = clock.mono()   # fell behind; resync pacing
        cap.release()


def make_cameras(cfg):
    """Build the two camera sources from config (mode = live | replay)."""
    mode = cfg.get("mode", default="replay")
    tol_lat = float(cfg.get("cameras", "transport_latency_s", default=0.0))
    cams = {}
    if mode == "live":
        c = cfg.get("cameras", "live")
        for i, key in enumerate(("cam0", "cam1")):
            cams[i] = CameraSource(i, c[key], buffersize=c.get("buffersize", 1),
                                   ffmpeg_opts=c.get("ffmpeg_opts", ""),
                                   transport_latency_s=tol_lat,
                                   width=c.get("width"),
                                   height=c.get("height"),
                                   fps=c.get("fps"),
                                   backend=c.get("backend"))
    else:
        c = cfg.get("cameras", "replay")
        for i, key in enumerate(("cam0", "cam1")):
            cams[i] = ReplayCameraSource(i, cfg.abspath(c[key]),
                                         loop=c.get("loop", True))
    return cams
