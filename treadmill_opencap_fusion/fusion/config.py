"""Configuration loading and path resolution."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class Config:
    """Parsed configuration with resolved absolute paths."""

    raw: dict[str, Any]
    config_dir: str

    # resolved paths
    force_npz: str = ""
    mot: str = ""
    trc: str = ""
    osim: str = ""
    raw_video: str = ""
    sync_video: str = ""
    output_dir: str = ""

    trial_name: str = ""

    def get(self, *keys, default=None):
        """Nested lookup, e.g. cfg.get('force', 'filter', 'cutoff_hz')."""
        node = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


def _resolve(base_dir: str, path: str | None) -> str | None:
    if path is None:
        return None
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base_dir, path))


def _first_glob(pattern: str) -> str | None:
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def load_config(config_path: str) -> Config:
    """Load YAML config and auto-resolve OpenCap file locations from trial name."""
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    with open(config_path) as fh:
        raw = yaml.safe_load(fh)

    cfg = Config(raw=raw, config_dir=config_dir)
    cfg.trial_name = raw["trial_name"]
    paths = raw["paths"]

    cfg.force_npz = _resolve(config_dir, paths["force_npz"])
    opencap_dir = _resolve(config_dir, paths["opencap_dir"])
    cfg.output_dir = _resolve(config_dir, paths.get("output_dir", "output"))

    trial = cfg.trial_name

    def auto(explicit, pattern):
        if explicit:
            return _resolve(config_dir, explicit)
        return _first_glob(os.path.join(opencap_dir, pattern))

    cfg.mot = auto(paths.get("mot"), f"OpenSimData/Kinematics/{trial}.mot")
    cfg.trc = auto(paths.get("trc"), f"MarkerData/{trial}.trc")
    cfg.osim = auto(paths.get("osim"), f"OpenSimData/Model/{trial}/*.osim")
    cfg.raw_video = auto(paths.get("raw_video"),
                         f"Videos/Cam0/InputMedia/{trial}/{trial}.mov")
    cfg.sync_video = auto(paths.get("sync_video"),
                          f"Videos/trimmed/InputMedia/{trial}/{trial}_sync.mp4")

    return cfg
