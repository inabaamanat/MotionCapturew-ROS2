"""Trial export services."""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .manager import Trial


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_ready(obj.tolist())
    if isinstance(obj, (np.integer, np.floating)):
        val = obj.item()
        return None if isinstance(val, float) and not np.isfinite(val) else val
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj


class TrialExporter:
    def __init__(self, trial: Trial):
        self.trial = trial
        self.export_dir = trial.path / "exports"
        self.export_dir.mkdir(exist_ok=True)

    def payload(self, scope: str = "both") -> dict[str, Any]:
        include_raw = scope in ("raw", "both")
        include_processed = scope in ("processed", "both")
        data: dict[str, Any] = {"metadata": self.trial.metadata}
        if include_processed:
            data.update({
                "summary": self.trial.summary,
                "steps": self.trial.steps,
                "events": self.trial.events,
                "derived_metrics": self.trial.derivedMetrics,
            })
        if include_raw:
            data["raw_manifest"] = {
                "force": "force.npz",
                "pose": "pose.npz",
                "live_metrics": "live_metrics.csv",
            }
        return data

    def to_json(self, scope: str = "both") -> Path:
        out = self.export_dir / f"{self.trial.path.name}_{scope}.json"
        out.write_text(json.dumps(_json_ready(self.payload(scope)), indent=2), encoding="utf-8")
        return out

    def to_csv(self, scope: str = "processed") -> Path:
        out = self.export_dir / f"{self.trial.path.name}_{scope}.csv"
        rows = []
        for step in self.trial.steps:
            row = {
                "trial_id": self.trial.metadata.get("trial_id"),
                "step_id": step.get("step_id"),
                "timestamp": step.get("timestamp"),
                "side": step.get("side"),
                "foot": step.get("foot"),
                "start_frame": step.get("start_frame"),
                "end_frame": step.get("end_frame"),
            }
            row.update(step.get("spatial_metrics", {}))
            row.update(step.get("temporal_metrics", {}))
            row.update({k: v for k, v in step.get("pressure_data", {}).items()
                        if not isinstance(v, (list, dict))})
            rows.append(row)
        if not rows:
            rows.append({"trial_id": self.trial.metadata.get("trial_id")})
        pd.DataFrame(rows).to_csv(out, index=False)
        return out

    def to_excel(self, scope: str = "both") -> Path:
        out = self.export_dir / f"{self.trial.path.name}_{scope}.xlsx"
        with pd.ExcelWriter(out) as writer:
            pd.DataFrame([self.trial.metadata]).to_excel(
                writer, sheet_name="metadata", index=False)
            pd.DataFrame([self.trial.summary]).to_excel(
                writer, sheet_name="summary", index=False)
            if scope in ("processed", "both"):
                step_rows = []
                for step in self.trial.steps:
                    row = {
                        "step_id": step.get("step_id"),
                        "timestamp": step.get("timestamp"),
                        "side": step.get("side"),
                        "start_frame": step.get("start_frame"),
                        "end_frame": step.get("end_frame"),
                    }
                    row.update(step.get("spatial_metrics", {}))
                    row.update(step.get("temporal_metrics", {}))
                    row.update({k: v for k, v in step.get("pressure_data", {}).items()
                                if not isinstance(v, (list, dict))})
                    step_rows.append(row)
                pd.DataFrame(step_rows).to_excel(writer, sheet_name="steps", index=False)
                pd.DataFrame(self.trial.events).to_excel(writer, sheet_name="events", index=False)
            if scope in ("raw", "both") and (self.trial.path / "live_metrics.csv").exists():
                pd.read_csv(self.trial.path / "live_metrics.csv").to_excel(
                    writer, sheet_name="live_metrics", index=False)
        return out

    def to_archive(self, scope: str = "both") -> Path:
        out = self.export_dir / f"{self.trial.path.name}_{scope}.zip"
        include_raw = scope in ("raw", "both")
        include_processed = scope in ("processed", "both")
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            if include_processed:
                for name in ("metadata.json", "summary.json", "step_events.json",
                             "derived_metrics.json", "quality_metrics.json",
                             "frame_data.npz", "pressure_timeseries.npz"):
                    p = self.trial.path / name
                    if p.exists():
                        zf.write(p, arcname=name)
            if include_raw:
                for name in ("force.npz", "pose.npz", "live_metrics.csv", "cam0.mp4", "cam1.mp4"):
                    p = self.trial.path / name
                    if p.exists():
                        zf.write(p, arcname=f"raw/{name}")
        return out

    def export(self, fmt: str, scope: str = "both") -> Path:
        fmt = fmt.lower()
        if fmt == "json":
            return self.to_json(scope)
        if fmt == "csv":
            return self.to_csv(scope)
        if fmt in ("excel", "xlsx"):
            return self.to_excel(scope)
        if fmt in ("binary", "archive", "zip"):
            return self.to_archive(scope)
        raise ValueError(f"unsupported export format: {fmt}")
