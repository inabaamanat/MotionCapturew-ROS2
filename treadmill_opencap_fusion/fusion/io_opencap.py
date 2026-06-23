"""Loaders for OpenCap / OpenSim outputs: .mot, .trc, .osim."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Kinematics:
    """Joint-angle time series from an OpenSim .mot file."""

    time: np.ndarray          # (N,) seconds
    coords: pd.DataFrame      # columns = coordinate names, index = frame
    in_degrees: bool
    fs: float

    @property
    def names(self) -> list[str]:
        return list(self.coords.columns)


@dataclass
class Markers:
    """3D marker trajectories from an OpenSim .trc file (metres)."""

    time: np.ndarray              # (N,)
    data: dict[str, np.ndarray]   # marker name -> (N, 3) XYZ in metres
    fs: float
    units: str


def load_mot(path: str) -> Kinematics:
    """Parse an OpenSim storage (.mot) file (header terminated by 'endheader')."""
    in_degrees = True
    with open(path) as fh:
        lines = fh.readlines()

    header_end = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.lower().startswith("indegrees"):
            in_degrees = s.split("=")[1].strip().lower() == "yes"
        if s.lower() == "endheader":
            header_end = i
            break
    if header_end is None:
        raise ValueError(f"No 'endheader' found in {path}")

    col_line = lines[header_end + 1].strip().split("\t")
    data = np.loadtxt(lines[header_end + 2:])
    df = pd.DataFrame(data, columns=col_line)
    time = df["time"].to_numpy()
    coords = df.drop(columns=["time"])
    fs = 1.0 / np.median(np.diff(time)) if time.size > 1 else float("nan")
    return Kinematics(time=time, coords=coords, in_degrees=in_degrees, fs=fs)


def load_trc(path: str) -> Markers:
    """Parse an OpenSim .trc marker file."""
    with open(path) as fh:
        lines = fh.readlines()

    # Line 3 (index 2): data-rate metadata; line 4 (index 3): marker names.
    meta_vals = lines[2].strip().split("\t")
    data_rate = float(meta_vals[0])
    units = meta_vals[4] if len(meta_vals) > 4 else "m"

    name_tokens = lines[3].rstrip("\n").split("\t")
    # First two columns are Frame# and Time; markers follow, one name per X col.
    marker_names = [tok for tok in name_tokens[2:] if tok != ""]

    # Numeric data starts at line index 5 (after the X1/Y1/Z1 sub-header).
    rows = []
    for line in lines[5:]:
        if line.strip() == "":
            continue
        rows.append([float(x) if x.strip() != "" else np.nan
                     for x in line.rstrip("\n").split("\t")])
    arr = np.array(rows)
    time = arr[:, 1]

    data: dict[str, np.ndarray] = {}
    for m, name in enumerate(marker_names):
        cols = slice(2 + 3 * m, 2 + 3 * m + 3)
        data[name] = arr[:, cols]

    fs = data_rate if data_rate > 0 else (
        1.0 / np.median(np.diff(time)) if time.size > 1 else float("nan"))
    return Markers(time=time, data=data, fs=fs, units=units)


def load_model_mass(path: str) -> dict:
    """Return total body mass (kg) and per-body masses from a scaled .osim."""
    with open(path) as fh:
        text = fh.read()
    # Each <Body name="..."> block contains one <mass> value.
    bodies = re.findall(r'<Body name="([^"]+)">.*?<mass>([\d.eE+\-]+)</mass>',
                        text, flags=re.DOTALL)
    per_body = {name: float(m) for name, m in bodies}
    return {"total_mass_kg": float(sum(per_body.values())),
            "per_body_kg": per_body}
