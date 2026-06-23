"""
Treadmill Interface Toolbox (Python port of MATLAB code by Pablo Iturralde)
Controls a Bertec Treadmill via TCP/IP on localhost:4000.

Requires the Treadmill Control Panel to be open with TCP/IP remote control
enabled on port 4000.

Units:
  Speed   : mm/s  (range -6500 to 6500)
  Accel   : mm/s² (range 1 to 3000)
  Incline : hundredths of a degree (e.g. 200 = 2.0°, range 0–1500)
"""

import socket
import struct
import time
import math
import warnings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 4000
MAX_VEL  =  6500
MIN_VEL  = -6500
MAX_ACC  =  3000
MAX_INC  =  1500
PACKET_SIZE = 32

# ---------------------------------------------------------------------------
# Byte / integer helpers
# ---------------------------------------------------------------------------

def int16_to_bytes(numbers: list[int]) -> list[tuple[int, int]]:
    """Convert a list of int16 values to (MSB, LSB) byte pairs."""
    result = []
    for n in numbers:
        n = max(-32768, min(32767, n))
        b = n & 0xFFFF
        result.append((b >> 8, b & 0xFF))
    return result


def bytes_to_int(pairs: list[tuple[int, int]]) -> list[int]:
    """Convert (MSB, LSB) byte pairs back to signed int16 values."""
    result = []
    for msb, lsb in pairs:
        val = msb * 256 + lsb
        if msb > 127:
            val -= 65536
        result.append(val)
    return result

# ---------------------------------------------------------------------------
# TCP/IP communication
# ---------------------------------------------------------------------------

def open_comm() -> socket.socket:
    """Open TCP connection to the treadmill controller."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    return s


def close_comm(s: socket.socket) -> None:
    """Close TCP connection."""
    s.close()


def send_packet(s: socket.socket, payload: list[int]) -> None:
    """Write a TCP/IP packet with the given payload."""
    s.sendall(bytes(payload))


def _recv_exact(s: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from socket."""
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed before all bytes received")
        buf += chunk
    return buf


def read_packet(s: socket.socket) -> tuple[int, int, int]:
    """
    Read one 32-byte packet from the treadmill and return
    (speed_R mm/s, speed_L mm/s, incline hundredths-of-degree).
    """
    data = _recv_exact(s, PACKET_SIZE)
    # Byte 0: format byte
    # Bytes 1-8: four int16 speeds (big-endian): R, L, Rr, Lr
    # Bytes 9-10: int16 incline
    # Bytes 11-31: padding
    speed_r, speed_l, _, _ = struct.unpack_from(">4h", data, 1)
    (incline,) = struct.unpack_from(">h", data, 9)
    return speed_r, speed_l, incline


def read_current_data(s: socket.socket) -> tuple[int, int, int]:
    """
    Drain stale packets and return the most recent treadmill state.
    """
    # Drain buffered data: peek at available bytes, discard in 32-byte chunks
    s.setblocking(False)
    while True:
        try:
            stale = s.recv(PACKET_SIZE)
            if not stale:
                break
        except BlockingIOError:
            break
    s.setblocking(True)
    return read_packet(s)

# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def get_payload(speed_r: float, speed_l: float,
                acc_r: float, acc_l: float,
                incline: float) -> list[int]:
    """Build a 64-byte command payload for the treadmill."""
    # Clamp inputs
    acc_r   = max(1,       min(MAX_ACC, acc_r))
    acc_l   = max(1,       min(MAX_ACC, acc_l))
    speed_r = max(MIN_VEL, min(MAX_VEL, speed_r))
    speed_l = max(MIN_VEL, min(MAX_VEL, speed_l))
    incline = max(0,       min(MAX_INC, incline))

    values = [round(v) for v in [speed_r, speed_l, 0, 0, acc_r, acc_l, 0, 0, incline]]
    pairs  = int16_to_bytes(values)

    actual_data = [b for pair in pairs for b in pair]   # flatten pairs → 18 bytes
    sec_check   = [255 - b for b in actual_data]
    padding     = [0] * 27

    return [0] + actual_data + sec_check + padding       # 1+18+18+27 = 64 bytes

# ---------------------------------------------------------------------------
# Mid-level API
# ---------------------------------------------------------------------------

def set_treadmill(speed_r: float, speed_l: float,
                  acc_r: float = 1000, acc_l: float = 1000,
                  incline: float = 0) -> tuple[int, int, int]:
    """
    Open connection, set treadmill speed/incline, read back current state,
    close connection.  Returns (cur_speed_R, cur_speed_L, cur_incline).
    """
    payload = get_payload(speed_r, speed_l, acc_r, acc_l, incline)
    s = open_comm()
    try:
        send_packet(s, payload)
        return read_current_data(s)
    finally:
        close_comm(s)


def read_treadmill() -> tuple[int, int, int]:
    """Open connection, read current treadmill state, close connection."""
    s = open_comm()
    try:
        return read_packet(s)
    finally:
        close_comm(s)


def set_speed_profile(speed_r: list[float], speed_l: list[float],
                      timing: list[float]) -> dict:
    """
    Send a timed speed profile to the treadmill.

    Parameters
    ----------
    speed_r, speed_l : lists of speeds (mm/s) at each time point
    timing           : list of times (s), must start at 0 and be monotonically
                       increasing; intervals should be ≥ 140 ms

    Returns
    -------
    dict with keys: cur_speed_r, cur_speed_l, cur_inc,
                    sent_timer, read_timer, sent_vr, sent_vl
    """
    n = len(timing)
    if not (len(speed_r) == len(speed_l) == n):
        raise ValueError("speed_r, speed_l, and timing must be the same length")
    if timing[0] != 0:
        raise ValueError("timing must start at 0")
    if any(d < 0 for d in (timing[i+1]-timing[i] for i in range(n-1))):
        raise ValueError("timing must be monotonically increasing")
    if any((timing[i+1]-timing[i]) < 0.14 for i in range(n-1)):
        warnings.warn("Trying to send commands too fast (< 140 ms apart)")

    dt = [timing[i+1] - timing[i] for i in range(n-1)]
    acc_r = [100] + [math.ceil(abs((speed_r[i+1]-speed_r[i])/dt[i])) for i in range(n-1)]
    acc_l = [100] + [math.ceil(abs((speed_l[i+1]-speed_l[i])/dt[i])) for i in range(n-1)]
    acc_r = [min(round(2*a), MAX_ACC) for a in acc_r]
    acc_l = [min(round(2*a), MAX_ACC) for a in acc_l]

    results = {k: [] for k in
               ("cur_speed_r", "cur_speed_l", "cur_inc",
                "sent_timer", "read_timer", "sent_vr", "sent_vl")}

    s = open_comm()
    try:
        base_time = time.perf_counter()
        for i in range(n):
            payload = get_payload(speed_r[i], speed_l[i], acc_r[i], acc_l[i], 0)
            # Wait until scheduled time
            while time.perf_counter() - base_time < timing[i]:
                pass
            elapsed = time.perf_counter() - base_time
            send_packet(s, payload)
            r, l, inc = read_current_data(s)
            read_t = time.perf_counter() - base_time

            pairs = int16_to_bytes([round(speed_r[i]), round(speed_l[i])])
            results["cur_speed_r"].append(r)
            results["cur_speed_l"].append(l)
            results["cur_inc"].append(inc)
            results["sent_timer"].append(elapsed)
            results["read_timer"].append(read_t)
            results["sent_vr"].append(bytes_to_int([pairs[0]])[0])
            results["sent_vl"].append(bytes_to_int([pairs[1]])[0])
    finally:
        close_comm(s)

    return results

# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def tight_speed_control(speed_r: list[float], speed_l: list[float],
                        time_pts: list[float]) -> dict:
    """
    Interpolate a speed profile at ≤140 ms resolution and execute it.

    Parameters
    ----------
    speed_r, speed_l : speed waypoints (mm/s), |speed| ≤ 4000
    time_pts         : time waypoints (s)

    Returns
    -------
    Same dict as set_speed_profile, plus keys:
    tight_speed_r, tight_speed_l, tight_time
    """
    n = len(time_pts)
    if not (len(speed_r) == len(speed_l) == n):
        raise ValueError("All input vectors must be the same length")
    if any(abs(v) > 4000 for v in speed_r + speed_l):
        raise ValueError("Speeds must be ≤ 4000 mm/s")
    for i in range(n-1):
        dt = time_pts[i+1] - time_pts[i]
        if (abs(speed_l[i+1]-speed_l[i])/dt > 2000 or
                abs(speed_r[i+1]-speed_r[i])/dt > 2000):
            raise ValueError("Requested accelerations too high (> 2000 mm/s²)")

    tight_r, tight_l, tight_t = [speed_r[0]], [speed_l[0]], [time_pts[0]]

    for i in range(1, n):
        steps = max(1, math.floor(7 * (time_pts[i] - time_pts[i-1])))
        segment_r = [tight_r[-1] + (speed_r[i]-tight_r[-1]) * k/(steps)
                     for k in range(1, steps+1)]
        segment_l = [tight_l[-1] + (speed_l[i]-tight_l[-1]) * k/(steps)
                     for k in range(1, steps+1)]
        segment_t = [time_pts[i-1] + (time_pts[i]-time_pts[i-1]) * k/(steps)
                     for k in range(1, steps+1)]
        tight_r.extend(segment_r)
        tight_l.extend(segment_l)
        tight_t.extend(segment_t)

    results = set_speed_profile(tight_r, tight_l, tight_t)
    results["tight_speed_r"] = tight_r
    results["tight_speed_l"] = tight_l
    results["tight_time"]    = tight_t
    return results


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Walk at 1 m/s on both belts for 5 s, then stop
    r, l, inc = set_treadmill(1000, 1000)          # 1000 mm/s = 1 m/s
    print(f"After set: R={r} mm/s  L={l} mm/s  incline={inc/100:.2f}°")

    results = tight_speed_control(
        speed_r=[0, 1000, 1000, 0],
        speed_l=[0, 1000, 1000, 0],
        time_pts=[0,  1.0,  4.0, 5.0],
    )
    print("Profile complete.")
