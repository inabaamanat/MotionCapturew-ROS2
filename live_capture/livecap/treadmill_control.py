"""Treadmill control — faithful port of the existing GUI's treadmill features.

Reproduces, byte-for-byte in behaviour, the controls from
Treadmill_python/functions_HILO.py:

  * SET (fixed speed)  -> set_speed_fixed(): set_treadmill(v*1000, v*1000,
                          a*1000, a*1000); turns Self-Paced off; updates readout.
  * STOP (big red)     -> big_red_stop(): velocity=0, acceleration=0.25.
  * Self-Paced Mode    -> the position-centering controller (v_gain=0.1,
                          a_gain=1, pos_zero=0.75) with the exact safety limits
                          (accel<=0.25, |jerk|<0.15, |delta|<=0.15, 0<=v<=2),
                          driven by per-foot COP estimated from the raw force
                          channels (frame[3]/frame[2], frame[9]/frame[8]) at
                          gait edges (Fz threshold 0.1 V).
  * Incline            -> additional control exposed by the hardware
                          (treadmill.set_treadmill incline arg); defaults to 0 so
                          fixed/stop behaviour is identical to the original.

Uses the canonical Treadmill_python/treadmill.py (TCP/IP to the Bertec control
panel on localhost:4000). All hardware calls are guarded so the GUI never
crashes if the control panel is closed; failures surface via status/log.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

# Gait-edge threshold (volts) and control gains — copied from functions_HILO.py.
FZ_THRESHOLD = 0.1
V_GAIN = 0.1
A_GAIN = 1.0
POS_ZERO = 0.75
MAX_ACC = 0.25
MAX_JERK = 0.15
MAX_DELTA = 0.15
MAX_SPEED = 2.0


def _import_treadmill():
    """Import the canonical Treadmill_python/treadmill.py (sibling of live_capture)."""
    try:
        import treadmill            # already on path
        return treadmill
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", ".."))   # OneDrive root
        cand = os.path.join(root, "Treadmill_python")
        if os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)
        import treadmill
        return treadmill


class TreadmillController:
    def __init__(self, write_tread_cb=None, status_cb=None, log_cb=None):
        self._tm = None
        self.write_tread_cb = write_tread_cb     # called with [pos, v, curr, a, cv, vr]
        self.status_cb = status_cb               # called with status dict
        self.log = log_cb or (lambda m: None)

        self.fixed_vel = 0.0
        self.fixed_acc = 0.0
        self.incline_deg = 0.0
        self.current_speed = 0.0
        self.self_paced = False
        self.connected = False
        self.last_error = ""

        self._lock = threading.Lock()
        self._reset_self_paced()
        self._last_volts = None

    # ---- hardware ----
    def _treadmill(self):
        if self._tm is None:
            self._tm = _import_treadmill()
        return self._tm

    def _command(self, vel_ms, acc_ms2):
        """Send a speed/accel/incline command; returns current speed (m/s) or None."""
        try:
            tm = self._treadmill()
            r, l, inc = tm.set_treadmill(vel_ms * 1000, vel_ms * 1000,
                                         acc_ms2 * 1000, acc_ms2 * 1000,
                                         self.incline_deg * 100)
            self.connected = True
            self.last_error = ""
            self.current_speed = (r or 0) / 1000.0
            return self.current_speed
        except Exception as e:
            self.connected = False
            self.last_error = str(e)
            self.log(f"treadmill command failed: {e}")
            return None

    def _emit_status(self, mode):
        if self.status_cb:
            self.status_cb(
                treadmill_mode=mode,
                treadmill_target_vel=(self.sp_speed if self.self_paced else self.fixed_vel),
                treadmill_current_vel=self.current_speed,
                treadmill_incline=self.incline_deg,
                treadmill_connected=self.connected)

    # ---- SET (fixed speed) : mirrors set_speed_fixed ----
    def set_fixed(self, vel_ms, acc_ms2):
        with self._lock:
            self.self_paced = False                 # set_btn("Self Paced Mode", False)
            self.fixed_vel = float(vel_ms)
            self.fixed_acc = float(acc_ms2)
            self._command(self.fixed_vel, self.fixed_acc)
            if self.write_tread_cb:
                self.write_tread_cb([0, self.fixed_vel, self.current_speed,
                                     self.fixed_acc, 0, 0])
            self.log(f"SET v={self.fixed_vel:.2f} m/s a={self.fixed_acc:.2f} m/s^2")
            self._emit_status("FIXED")
            return self.current_speed

    # ---- STOP (big red) : mirrors big_red_stop ----
    def stop(self):
        with self._lock:
            self.self_paced = False
            self.fixed_vel = 0.0
            self.fixed_acc = 0.25
            self._command(0.0, 0.25)
            if self.write_tread_cb:
                self.write_tread_cb([0, 0.0, self.current_speed, 0.25, 0, 0])
            self.log("STOP (emergency)")
            self._emit_status("STOP")
            return self.current_speed

    # ---- incline ----
    def set_incline(self, deg):
        with self._lock:
            self.incline_deg = float(deg)
            self._command(self.fixed_vel if not self.self_paced else self.sp_speed,
                          self.fixed_acc if not self.self_paced else 0.1)
            self.log(f"incline {self.incline_deg:.1f} deg")
            self._emit_status("SELF-PACED" if self.self_paced else "FIXED")

    # ---- Self-Paced Mode ----
    def _reset_self_paced(self):
        self.l_foot_pos = 0.0
        self.r_foot_pos = 0.0
        self.last_position = POS_ZERO
        self.last_step_time = 0.0
        self.last_delta = 0.0
        self.sp_speed = self.fixed_vel
        self.sp_curr = self.fixed_vel

    def start_self_paced(self, initial_vel=None):
        with self._lock:
            if initial_vel is not None:
                self.fixed_vel = float(initial_vel)
            self._reset_self_paced()
            self.self_paced = True
            self._last_volts = None
            self._command(self.fixed_vel, 0.1)      # original: accel 100 mm/s^2
            self.log(f">>>Self-Paced start (v0={self.fixed_vel:.2f} m/s)")
            self._emit_status("SELF-PACED")

    def stop_self_paced(self):
        with self._lock:
            self.self_paced = False
            self.log("Self-Paced off")
            self._emit_status("FIXED")

    def feed_force(self, volts12):
        """Per-sample force input (12 raw volts). Drives self-paced control only."""
        if not self.self_paced:
            self._last_volts = volts12
            return
        last = self._last_volts
        self._last_volts = volts12
        if last is None:
            return
        frame = volts12
        state = self._transition(frame, last)
        msg_ready = False
        # exact index convention from position_estimate()
        if state == "leading edge right":
            self.r_foot_pos = self._cop(frame[3], frame[2], self.r_foot_pos)
            msg_ready = True
        if state == "leading edge left":
            self.l_foot_pos = self._cop(frame[9], frame[8], self.l_foot_pos)
            msg_ready = True
        if state == "falling edge right":
            self.r_foot_pos = self._cop(frame[3], frame[2], self.r_foot_pos)
        if state == "falling edge left":
            self.l_foot_pos = self._cop(frame[9], frame[8], self.l_foot_pos)
        if msg_ready:
            self._self_paced_update([self.l_foot_pos, self.r_foot_pos])

    @staticmethod
    def _cop(mx, fz, prev):
        return mx / fz if fz != 0 else prev

    @staticmethod
    def _transition(frame, last):
        thr = FZ_THRESHOLD
        if frame[2] < thr and last[2] >= thr:
            return "falling edge left"
        if frame[8] < thr and last[8] >= thr:
            return "falling edge right"
        if frame[2] > thr and last[2] <= thr:
            return "leading edge left"
        if frame[8] > thr and last[8] <= thr:
            return "leading edge right"
        return "none"

    def _self_paced_update(self, message):
        """Exact replica of the self_paced control law."""
        # NOTE: matches the original exactly -- last_step_time starts at 0, so the
        # first delta_t is ~epoch-seconds (huge) and v_rel~0 on the first event.
        now = time.time()
        delta_t = now - self.last_step_time
        self.last_step_time = now

        position_estimate = float(np.mean(message))
        if position_estimate < 0 or position_estimate > 2:
            position_estimate = POS_ZERO

        v_rel = (position_estimate - self.last_position) / delta_t
        centering_velocity = position_estimate - POS_ZERO
        if abs(centering_velocity) < 0.1:
            centering_velocity = 0
        self.last_position = position_estimate

        delta = (centering_velocity + v_rel) * V_GAIN
        acceleration = abs(self.sp_speed - self.sp_curr) * A_GAIN

        # SAFETY LIMITS (identical to original)
        if acceleration >= MAX_ACC:
            acceleration = MAX_ACC
        jerk = delta - self.last_delta
        if abs(jerk) >= MAX_JERK:
            delta = 0
        if delta >= MAX_DELTA:
            delta = MAX_DELTA
        if delta <= -MAX_DELTA:
            delta = -MAX_DELTA

        self.sp_speed = self.sp_speed + delta
        self.last_delta = delta
        if self.sp_speed < 0:
            self.sp_speed = 0
        if self.sp_speed > MAX_SPEED:
            self.sp_speed = MAX_SPEED

        curr = self._command(self.sp_speed, acceleration)
        if curr is not None:
            self.sp_curr = curr
        if self.write_tread_cb:
            self.write_tread_cb([position_estimate, self.sp_speed, self.sp_curr,
                                 acceleration, centering_velocity, v_rel])
        self._emit_status("SELF-PACED")
