"""End-to-end orchestration: load -> calibrate -> sync -> merge -> export -> render."""
from __future__ import annotations

import json
import os

from . import io_force, io_opencap, io_video, calibration, gait, sync as syncmod
from . import merge as mergemod, qc, overlay, pose2d
from .config import load_config


def _section(msg):
    print(f"\n=== {msg} ===")


def run(config_path: str, do_overlay: bool | None = None,
        do_qc: bool = True) -> dict:
    cfg = load_config(config_path)
    out_dir = os.path.join(cfg.output_dir, cfg.trial_name)
    os.makedirs(out_dir, exist_ok=True)

    _section("Loading data")
    force = io_force.load_force(cfg.force_npz, cfg.get("force", "time_row"),
                                cfg.get("force", "channel_order"),
                                cfg.get("force", "filter"))
    kin = io_opencap.load_mot(cfg.mot)
    markers = io_opencap.load_trc(cfg.trc)
    model = io_opencap.load_model_mass(cfg.osim)
    body_mass = (cfg.get("calibration", "bodyweight", "body_mass_kg")
                 or model["total_mass_kg"])
    print(f"force: {force.n_samples} samp @ {force.fs:.1f} Hz, {force.t[-1]:.1f}s")
    print(f"kinematics: {len(kin.time)} frames @ {kin.fs:.1f} Hz, {len(kin.names)} coords")
    print(f"markers: {len(markers.data)} @ {markers.fs:.1f} Hz | body mass {body_mass:.1f} kg")

    _section("Calibrating force plates")
    cforce = calibration.calibrate(force.channels, cfg.raw["calibration"], body_mass)
    print(f"mode={cforce.mode}  {cforce.notes}")

    _section("Detecting gait events")
    force_gait = gait.detect_force_events(cforce, force.t, body_mass, cfg.raw["gait"])
    print(f"force heel strikes  L={len(force_gait.left.heel_strikes)} "
          f"R={len(force_gait.right.heel_strikes)}  "
          f"cadence={force_gait.metrics['cadence_steps_per_min']:.0f}/min")

    _section("Synchronizing clocks")
    sync = syncmod.synchronize(force.t, cforce, kin, markers, force_gait,
                               cfg.raw["sync"])
    print(f"offset={sync.offset_s:.3f}s  r={sync.correlation:.3f} "
          f"(R={sync.corr_right:.3f} L={sync.corr_left:.3f})  "
          f"HS-RMS={sync.heelstrike_rms_s:.3f}s matched={sync.heelstrike_matched}")

    _section("Aligning raw video")
    raw_info = io_video.probe(cfg.raw_video)
    align = io_video.align_raw_to_sync(cfg.raw_video, cfg.sync_video)
    print(f"raw {raw_info.width}x{raw_info.height}@{raw_info.fps:.2f} "
          f"{raw_info.n_frames}f | align slope={align.slope:.3f} "
          f"intercept={align.intercept:.1f} quality={align.quality:.3f}")

    _section("Merging onto frame timeline")
    merged = mergemod.build_merged_table(cfg, force, cforce, kin, markers,
                                         force_gait, sync, align, raw_info,
                                         body_mass)
    csv_path = mergemod.export_table(merged, out_dir)
    print(f"table {merged.table.shape} -> {csv_path}")

    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(merged.meta, fh, indent=2, default=float)

    qc_path = None
    if do_qc:
        _section("Writing QC report")
        qc_path = qc.make_qc_report(force, cforce, kin, markers, force_gait,
                                    sync, merged, out_dir)
        print(qc_path)

    overlay_path = None
    if do_overlay is None:
        do_overlay = cfg.get("overlay", "enabled", default=True)
    if do_overlay:
        kp2d = None
        if cfg.get("overlay", "draw_skeleton", default=True):
            _section("Detecting 2D body keypoints (for on-body skeleton)")
            kp2d = pose2d.detect_pose(
                cfg.raw_video, os.path.join(out_dir, "pose2d.npz"),
                model_name=cfg.get("overlay", "pose_model",
                                   default="yolo11n-pose.pt"))
            print(f"pose keypoints: {kp2d.shape}")
        _section("Rendering overlay video")
        overlay_path = os.path.join(out_dir, f"{cfg.trial_name}_overlay.mp4")
        r = overlay.OverlayRenderer(merged, cfg.raw_video, cfg, kp2d=kp2d)
        r.render(overlay_path)
        print(overlay_path)

    return {"out_dir": out_dir, "csv": csv_path, "qc": qc_path,
            "overlay": overlay_path, "meta": merged.meta}
