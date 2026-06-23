"""Force-plate sources (push model): a producer loop emits (t, volts12).

* :class:`LiveForceSource`   -- Bertec via NI-DAQ (nidaqmx), reusing the channel
  map and acquisition approach from Treadmill_python/functions_HILO.py.
* :class:`ReplayForceSource` -- streams an existing .npz at its real cadence.

``nidaqmx`` is imported lazily so the package works on machines without it.
``emit`` is a callback ``emit(t_epoch, volts_vector_of_12)``.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .. import clock


class LiveForceSource:
    def __init__(self, device, channels, terminal_config, rate_hz, read_chunk):
        self.device = device
        self.channels = channels
        self.terminal_config = int(terminal_config)
        self.rate_hz = int(rate_hz)
        self.read_chunk = int(read_chunk)
        self._stop = threading.Event()
        self._thread = None
        self.fps_est = 0.0

    def start(self, emit):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(emit,),
                                        daemon=True, name="force")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self, emit):
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, TerminalConfiguration
        with nidaqmx.Task() as task:
            cfg = TerminalConfiguration(self.terminal_config)
            for ch in self.channels:
                task.ai_channels.add_ai_voltage_chan(f"{self.device}/{ch}",
                                                     terminal_config=cfg)
            task.timing.cfg_samp_clk_timing(
                rate=self.rate_hz, sample_mode=AcquisitionType.CONTINUOUS)
            task.start()
            period = 1.0 / self.rate_hz
            n = 0
            t0 = clock.mono()
            while not self._stop.is_set():
                data = task.read(number_of_samples_per_channel=self.read_chunk)
                arr = np.asarray(data, float)          # (12, read_chunk)
                t_arrival = clock.now()
                # back-date each sample within the chunk by the sample period.
                for j in range(arr.shape[1]):
                    t = t_arrival - (arr.shape[1] - 1 - j) * period
                    emit(t, arr[:, j])
                n += arr.shape[1]
                dt = clock.mono() - t0
                if dt >= 0.5:
                    self.fps_est = n / dt
                    n = 0
                    t0 = clock.mono()


class ReplayForceSource:
    """Replay an existing .npz at the true cadence encoded in its time row."""

    def __init__(self, npz_path, data_time_row=13, loop=True):
        self.npz_path = npz_path
        self.data_time_row = int(data_time_row)
        self.loop = loop
        self._stop = threading.Event()
        self._thread = None
        self.fps_est = 0.0
        d = np.load(npz_path, allow_pickle=True)
        DATA = np.asarray(d["DATA"], float)
        t = DATA[self.data_time_row]
        valid = np.where(t > 0)[0]
        hi = valid.max() + 1 if valid.size else DATA.shape[1]
        self.t = t[:hi]
        self.volts = DATA[:12, :hi]          # 12 channels x N
        self.fs = 1.0 / np.median(np.diff(self.t[self.t > 0])) if hi > 2 else 100.0

    def start(self, emit):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(emit,),
                                        daemon=True, name="force_replay")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self, emit):
        while not self._stop.is_set():
            t_wall0 = clock.mono()
            t_rec0 = self.t[0]
            n = 0
            tmark = clock.mono()
            for i in range(self.volts.shape[1]):
                if self._stop.is_set():
                    return
                # pace to the recording's own timeline
                target = t_wall0 + (self.t[i] - t_rec0)
                sleep = target - clock.mono()
                if sleep > 0:
                    time.sleep(sleep)
                emit(clock.now(), self.volts[:, i])
                n += 1
                if clock.mono() - tmark >= 0.5:
                    self.fps_est = n / (clock.mono() - tmark)
                    n = 0
                    tmark = clock.mono()
            if not self.loop:
                break


def make_force(cfg):
    mode = cfg.get("mode", default="replay")
    if mode == "live":
        c = cfg.get("force", "live")
        return LiveForceSource(c["device"], c["channels"], c["terminal_config"],
                               c["rate_hz"], c["read_chunk"])
    c = cfg.get("force", "replay")
    return ReplayForceSource(cfg.abspath(c["npz"]),
                             data_time_row=c.get("data_time_row", 13),
                             loop=c.get("loop", True))
