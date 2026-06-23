"""Central data access API for recordings and trials."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _npz_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


@dataclass
class Trial:
    """Lazy loaded trial record.

    The property names intentionally match the short access style requested by
    the app: trial.frames, trial.steps, trial.metadata, trial.summary, etc.
    """

    path: Path

    @property
    def metadata(self) -> dict[str, Any]:
        return _read_json(self.path / "metadata.json", {})

    @property
    def frames(self) -> dict[str, Any]:
        return _npz_dict(self.path / "frame_data.npz")

    @property
    def steps(self) -> list[dict[str, Any]]:
        return _read_json(self.path / "step_events.json", {"steps": []}).get("steps", [])

    @property
    def events(self) -> list[dict[str, Any]]:
        return _read_json(self.path / "step_events.json", {"events": []}).get("events", [])

    @property
    def summary(self) -> dict[str, Any]:
        return _read_json(self.path / "summary.json", {})

    @property
    def pressure(self) -> dict[str, Any]:
        return _npz_dict(self.path / "pressure_timeseries.npz")

    @property
    def jointAngles(self) -> dict[str, Any]:
        return self.derivedMetrics.get("joint_angles", {})

    @property
    def derivedMetrics(self) -> dict[str, Any]:
        return _read_json(self.path / "derived_metrics.json", {})

    @property
    def landmarks(self) -> dict[str, Any]:
        pose = _npz_dict(self.path / "pose.npz")
        return {k: pose.get(k) for k in ("t", "kp0", "kp1", "kp3d", "valid", "coco_names") if k in pose}

    @property
    def rawForce(self) -> dict[str, Any]:
        return _npz_dict(self.path / "force.npz")

    def frame_count(self) -> int:
        frames = self.frames
        return int(len(frames.get("timestamp", [])))

    def is_processed(self) -> bool:
        return ((self.path / "summary.json").exists()
                and (self.path / "frame_data.npz").exists())


class RecordingDataManager:
    """Browse, search, filter, and load trial recordings from one root folder."""

    def __init__(self, recordings_dir: str | os.PathLike):
        self.recordings_dir = Path(recordings_dir)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

    @property
    def index_path(self) -> Path:
        return self.recordings_dir / "trials_index.json"

    def list_trials(self) -> list[dict[str, Any]]:
        trials = []
        for meta_path in self.recordings_dir.glob("*/metadata.json"):
            trial = Trial(meta_path.parent)
            meta = trial.metadata
            summ = trial.summary
            trials.append({
                "trial_id": meta.get("trial_id"),
                "session_id": meta.get("session_id"),
                "recording_name": meta.get("recording_name") or meta_path.parent.name,
                "date_time": meta.get("date_time"),
                "path": meta_path.parent.name,
                "status": meta.get("status") or ("processed" if trial.is_processed() else "raw_saved"),
                "duration_s": summ.get("recording_duration_s", meta.get("duration_s")),
                "total_steps": summ.get("total_steps"),
                "distance_m": summ.get("total_distance_walked_m"),
                "average_speed_m_s": summ.get("average_walking_speed_m_s"),
                "average_cadence_steps_per_min": summ.get("average_cadence_steps_per_min"),
            })
        return sorted(trials, key=lambda x: x.get("date_time") or "", reverse=True)

    def get_trial(self, trial_id_or_name: str) -> Trial | None:
        for item in self.list_trials():
            if trial_id_or_name in (item.get("trial_id"), item.get("recording_name"), item.get("path")):
                return Trial(self.recordings_dir / item["path"])
        direct = self.recordings_dir / trial_id_or_name
        return Trial(direct) if (direct / "metadata.json").exists() else None

    def search(self, text: str = "", **filters) -> list[dict[str, Any]]:
        text = (text or "").lower()
        out = []
        for item in self.list_trials():
            hay = " ".join(str(item.get(k, "")) for k in ("recording_name", "date_time", "trial_id")).lower()
            if text and text not in hay:
                continue
            ok = True
            for key, val in filters.items():
                if val is None:
                    continue
                if item.get(key) != val:
                    ok = False
                    break
            if ok:
                out.append(item)
        return out

    def as_dataframe(self, trials: Iterable[dict[str, Any]] | None = None) -> pd.DataFrame:
        return pd.DataFrame(list(trials if trials is not None else self.list_trials()))

    def process_trial(self, trial_id_or_name: str | Trial, cfg,
                      render_playback: bool = True, progress=None) -> Trial | None:
        """Run the offline trial pipeline immediately for one recording."""
        def emit(stage, current=0, total=1, message=""):
            if progress is not None:
                progress(stage, current, total, message)

        trial = (trial_id_or_name if isinstance(trial_id_or_name, Trial)
                 else self.get_trial(str(trial_id_or_name)))
        if trial is None:
            return None
        emit("start", 0, 1, f"Preparing {trial.path.name}")
        from .offline_pose import reprocess_trial_pose
        reprocess_trial_pose(trial.path, cfg, force=True, progress=progress)
        from .analysis import build_trial_artifacts
        emit("artifacts", 0, 1, "Building normalized trial files")
        metadata = build_trial_artifacts(trial.path, cfg)
        emit("artifacts", 1, 1, "Trial files written")
        if render_playback:
            from .render import render_processed_playback
            emit("playback", 0, 1, "Rendering skeleton overlay playback")
            playback = render_processed_playback(trial.path, cfg)
            if playback is not None:
                meta_path = trial.path / "metadata.json"
                meta = _read_json(meta_path, metadata)
                paths = dict(meta.get("paths") or {})
                paths["playback_video"] = str(playback.relative_to(trial.path))
                meta["paths"] = paths
                meta["status"] = "processed"
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            emit("playback", 1, 1, "Playback rendering complete")
        emit("complete", 1, 1, "Processing complete")
        return Trial(trial.path)

    def pending_trials(self) -> list[Trial]:
        pending = []
        for item in self.list_trials():
            trial = Trial(self.recordings_dir / item["path"])
            if trial.is_processed():
                continue
            if (trial.path / "force.npz").exists() or (trial.path / "pose.npz").exists():
                pending.append(trial)
        return pending

    def process_pending(self, cfg, render_playback: bool = True, progress=None) -> list[Trial]:
        processed = []
        for trial in self.pending_trials():
            out = self.process_trial(trial, cfg, render_playback=render_playback,
                                     progress=progress)
            if out is not None:
                processed.append(out)
        return processed
