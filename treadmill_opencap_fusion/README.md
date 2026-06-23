# Treadmill ↔ OpenCap Fusion

Synchronize Bertec instrumented-treadmill force-plate data with markerless
OpenCap motion capture, merge them onto a common per-frame timeline, and render
a fully-annotated overlay video plus a tidy analysis table.

Built for the Medical Robotics Lab. Designed to be reused for arbitrary trials,
not just the one it was first written for.

## What it does

1. **Loads** every data source for a trial:
   - Bertec force/treadmill `.npz` (raw analog voltages, 12 channels @ ~100 Hz)
   - OpenCap inverse-kinematics `.mot` (joint angles, 60 Hz)
   - OpenCap markers `.trc` (3D marker trajectories, 60 Hz)
   - OpenCap scaled OpenSim model `.osim` (segment masses → body mass)
   - The raw Cam0 video
2. **Calibrates** the force plates from raw volts to Newtons / Newton-metres
   (Bertec calibration matrix, with a bodyweight-derived fallback) and computes
   the centre of pressure (COP) for each plate.
3. **Synchronizes** the independent force-PC clock and OpenCap clock using a
   physics-based shared signal (centre-of-mass vertical acceleration vs. measured
   vertical GRF) cross-checked against per-foot heel-strike event matching.
4. **Merges** everything onto the video frame timeline and exports a tidy table
   (`merged.csv` / `merged.parquet`) with one row per video frame.
5. **Renders** an annotated overlay video with joint angles, ground-reaction
   forces, COP, gait phase, and spatiotemporal metrics.
6. **Writes a QC report** so you can verify the synchronization and signal quality.

## Quick start

```bash
# from the repo root (OneDrive_1_6-15-2026/)
/opt/anaconda3/bin/python treadmill_opencap_fusion/run_fusion.py \
    --config treadmill_opencap_fusion/config.yaml
```

Outputs land in `treadmill_opencap_fusion/output/<trial>/`.

## Calibration

Force channels in the `.npz` are **raw analog voltages**. To get true Newtons
edit `config.yaml -> calibration`:

- `mode: matrix`  — provide the per-plate 6×6 Bertec calibration matrix (V→[Fx Fy Fz Mx My Mz]).
- `mode: gains`   — provide per-channel scalar V→N / V→Nm gains.
- `mode: bodyweight` (default fallback) — derives the vertical scale from the
  known body mass in the `.osim`; vertical GRF is accurate, shear/moments are approximate.

## Scope

Joint **angles, velocities, GRF, COP, gait events and spatiotemporal metrics**
are produced here. Joint **moments/forces** (inverse dynamics) are intentionally
deferred — they require OpenSim ID with calibrated GRF+COP mapped to the feet.
