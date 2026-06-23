# Trial Day Checklist

Use this sequence for a live 2-iPhone + Bertec treadmill capture.

## 1. Hardware Setup

1. Open the Bertec treadmill control panel on the capture PC.
2. Enable TCP/IP remote control on `localhost:4000`.
3. Confirm the treadmill emergency stop is physically reachable.
4. Connect the Bertec/NI-DAQ hardware and confirm the configured NI device is `Dev1`.
5. Put both iPhones on the same network as the capture PC.
6. Start the local stream relay if you are using the included MediaMTX server:

```powershell
cd "..\mediamtx"
.\mediamtx.exe .\mediamtx.yml
```

## 2. Camera Placement

Use the walking direction as 0 degrees.

1. Mount `cam0` at 45 degrees from the walking direction.
2. Mount `cam1` at 315 degrees, also written as -45 degrees.
3. Aim both cameras at the subject's pelvis/torso so the full body remains visible through the gait cycle.
4. Keep both phones rigidly mounted. Do not move them after calibration.
5. Use portrait orientation unless you intentionally update the capture layout.
6. Lock focus and exposure in the streaming app if possible.
7. Stream the phones to:

```text
rtsp://127.0.0.1:8554/cam0
rtsp://127.0.0.1:8554/cam1
```

The 45/315 degree setup is the physical starting geometry. The stereo calibration step below measures the exact real geometry used by the 3D reconstruction.

## 3. Config Check

From `live_capture`:

```powershell
python -m livecap.preflight --config config.yaml
```

Expected before camera calibration: the only hard failure may be missing `calib/stereo_extrinsics.npz`.

Fix any camera, DAQ, treadmill TCP, or output-folder failures before continuing.

## 4. Stereo Calibration

Use the 100 mm checkerboard/grid configured in `config.yaml`:

```yaml
checkerboard: {cols: 4, rows: 7, square_m: 0.100, min_pairs: 12}
```

`cols` and `rows` are inner corners, not printed squares.

Run:

```powershell
python -m livecap.calib.run_calibration --config config.yaml
```

1. Hold the board so both cameras can see it.
2. Press Space to capture a pair.
3. Capture at least 12 pairs.
4. Move the board through varied positions, depths, heights, and rotations.
5. Keep the board flat and still for each capture.
6. Press `c` to compute and save calibration.
7. Do not move either camera after this step.

Calibration writes:

```text
calib/cam0_intrinsics.npz
calib/cam1_intrinsics.npz
calib/stereo_extrinsics.npz
```

## 5. Final Preflight

Run:

```powershell
python -m livecap.preflight --config config.yaml
```

Proceed only when there are no failures.

Warnings about bodyweight force fallback are acceptable for gait timing tests, but calibrated forces/COP need the Bertec matrix or gains.

## 6. Subject Setup

1. Enter the subject mass in `config.yaml` under `calibration.force.body_mass_kg`.
2. Put the subject on the treadmill.
3. Keep the subject standing still for the first 2 seconds after capture starts so bodyweight fallback scaling can initialize.
4. Confirm both camera panels show the full body.
5. Confirm vertical GRF rises when the subject stands on the belts.
6. Confirm the treadmill panel shows the expected mode and connection status.

## 7. Start A Trial

Run the GUI with a trial label:

```powershell
python run_live.py --config config.yaml --session S01_walk_1
```

In the GUI:

1. Confirm capture is running.
2. Click `Arm Recording`.
3. Set treadmill velocity and acceleration.
4. Click `SET` for fixed-speed walking, or enable `Self-Paced` when ready.
5. Monitor live joint angles, GRF, gait phase, cadence, stride metrics, and treadmill status.
6. Click `STOP` immediately if anything looks unsafe.
7. Click `Stop+Save` at the end of the trial.

Each trial saves to a fresh timestamped folder under:

```text
recordings/
```

Expected outputs:

```text
force.npz
pose.npz
live_metrics.csv
cam0.mp4
cam1.mp4
```

## 8. Between Trials

1. Do not move cameras unless you plan to recalibrate.
2. Use a new `--session` label for each trial, for example `S01_walk_2`.
3. Re-run preflight after any hardware, stream, DAQ, or camera change.
4. Re-run stereo calibration after any camera movement.

## 9. Shutdown

1. Click `STOP` in the GUI.
2. Save any active recording.
3. Stop the GUI.
4. Stop iPhone streams.
5. Close MediaMTX if used.
6. Close the Bertec control panel.
