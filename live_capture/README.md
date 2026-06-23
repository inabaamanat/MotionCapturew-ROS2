# Live Capture — 2 iPhones + Bertec Force → Real-Time 3D MoCap

Replaces the web-driven OpenCap loop with an **on-premise, real-time** system:
two iPhone camera feeds + the Bertec force plates are captured live, fused on a
single clock, turned into 3D joint kinematics + ground-reaction metrics, shown in
a live GUI, and recorded — no waiting for a cloud session to finish.

Runs on the single Windows + GPU treadmill PC. Reuses the lab's existing DAQ /
treadmill code and the offline `treadmill_opencap_fusion` analysis package.

## Pipeline

```
 iPhone cam0 ─┐                       ┌─ 2D pose (YOLO, GPU)
 iPhone cam1 ─┤ latest-frame grabbers ┤─ triangulate ─ 3D angles ─┐
              │  (kill buffering)      └─ pair by timestamp        │→ SharedState ─→ GUI + Recorder
 Bertec DAQ ──┴─ force thread ─ calibrate (V→N) + COP + live gait ─┘
```

- **Low latency:** each camera has a grabber thread that keeps only the newest
  frame, so a slow consumer never builds a backlog. Use a low-buffer RTSP/SRT app.
- **One clock:** every sample is stamped with the PC wall clock; camera frame
  pairs are matched by timestamp before triangulation.
- **3D angles:** triangulated COCO-17 keypoints → pelvis-frame hip/knee angles
  (validated at r≈0.92–0.99 vs OpenCap's OpenSim angles). Offline OpenSim IK
  available for publication-grade absolute angles.

## Setup

```bash
pip install -r requirements.txt          # + CUDA torch + nidaqmx on the PC
```

Edit `config.yaml`: set `mode: live`, the two `cameras.live` RTSP URLs, the
force `device`/`channels`, and `calibration.force` (body mass, or the Bertec
matrix/gains for true Newtons).

## Calibrate the cameras (required for 3D)

```bash
python -m livecap.calib.run_calibration --config config.yaml
```

Use the 100 mm checkerboard/grid configured in `config.yaml`
(`square_m: 0.100`). The configured `cols`/`rows` are inner corners, not printed
squares. Show the board to **both** cameras, SPACE to capture pairs (at least 12
in varied positions/orientations), `c` to compute + save. Without this, the GUI shows per-camera 2D
skeletons but no 3D triangulation.

## Trial-day preflight

With the two iPhones streaming and the Bertec treadmill control panel open:

```bash
python -m livecap.preflight --config config.yaml
```

This checks camera frames, the 100 mm checkerboard config, stereo calibration
file, YOLO model, CUDA availability, NI-DAQ device/channels, treadmill TCP
control on `localhost:4000`, and recording-folder writability. It does not move
the treadmill.

## Run

```bash
python run_live.py --config config.yaml                  # DearPyGui app
python run_live.py --config config.yaml --backend cv2    # OpenCV-window GUI
python run_live.py --config config.yaml --mode replay    # test with existing files
python run_live.py --config config.yaml --headless 10    # smoke test, no GUI
python run_live.py --config config.yaml --session S01_walk_1 --record
```

GUI controls: Start/Stop Capture, Arm Recording, Stop+Save.
Every recording creates a fresh timestamped folder under `recordings/`, so trial
data is not overwritten.
Stop+Save also builds the research trial layer automatically: normalized frame
data, step events, derived metrics, quality metrics, pressure time series,
summary files, and a `trials_index.json` browser index.

### Treadmill control (full, faithful to the existing GUI)

The Bertec treadmill is controlled from this GUI with identical behaviour to
`Treadmill_python/functions_HILO.py` (via `treadmill.py`, TCP to the control
panel on `localhost:4000`):

- **SET** — drive both belts to the entered Velocity (m/s) and Acceleration
  (m/s²); turns Self-Paced off.
- **STOP** (red) — emergency stop (velocity 0, acceleration 0.25 m/s²).
- **Self-Paced** — position-centering controller (the exact v_gain=0.1 / a_gain=1
  law with the same accel/jerk/speed safety limits), driven live by per-foot COP
  from the force plates.
- **Incline** (deg) — additional hardware control (defaults to 0).

Velocity + mode are shown live in the GUI and on the dashboard "Treadmill" panel;
treadmill telemetry is saved to `force.npz` (`TREAD`, `TREAD_t`).

cv2 backend keys: `r` arm · `s` save · `q` quit · `[`/`]` target speed ∓0.1 ·
`g` SET · `x` STOP · `,`/`.` incline ∓0.5 · `p` toggle Self-Paced.

> **Zip note:** treadmill control imports `treadmill.py` from the sibling
> `Treadmill_python/` folder — include it in the zip (i.e. zip the whole project
> folder, not just `live_capture/`). The Treadmill Control Panel must be open
> with TCP/IP remote enabled on port 4000; otherwise controls no-op gracefully.

## Output (per recording)

Mirrors `treadmill_opencap_fusion/output/` so the offline overlay/QC tools run on
live recordings unchanged:

- `force.npz` — fusion-compatible (`DATA` 14×N, `START`; plus calibrated `CAL`)
- `pose.npz` — timestamps, 2D keypoints (both cams), 3D keypoints, joint angles
- `live_metrics.csv` — per-frame angles + live gait metrics + per-foot GRF/COP
- `cam0.mp4`, `cam1.mp4` — processed frames (for offline OpenSim if desired)
- `metadata.json`, `frame_data.npz`, `step_events.json`,
  `derived_metrics.json`, `quality_metrics.json`, `pressure_timeseries.npz`,
  `summary.json` — normalized trial artifacts for browsing, replay, export, and
  rerunning analysis while keeping raw data separate.
- `exports/` — JSON, CSV, Excel, and binary archive exports from the Trials page.

Programmatic access:

```python
from livecap.trials import RecordingDataManager

trial = RecordingDataManager("recordings").get_trial("my_trial_name")
trial.frames
trial.steps
trial.metadata
trial.summary
trial.pressure
trial.jointAngles
trial.landmarks
trial.events
```

The DearPyGui app includes a **Trials** tab with a searchable recording table,
summary/replay view, synchronized skeleton and pressure playback, gait plots,
trajectories, step table, click-to-jump step navigation, and export buttons.

## Testing without hardware

`mode: replay` streams the existing `../iggdl_RENAME.npz` (force) and
`Stationary_Walking_2.mov` (cameras) through the full engine + GUI in real time.
Triangulation needs two genuinely different calibrated views, so 3D is exercised
by the unit checks (`livecap/pose/triangulate.py:make_virtual_stereo`, exact
reconstruction) until two iPhones are connected.
