# How to Run & Use SafeWheel

This quick guide walks you through installing dependencies, launching the driver-drowsiness detector, and using its real-time controls effectively.

---

## 1. Prerequisites
- Windows with a working webcam (default target platform).
- Python 3.12.x installed (matches the repo’s virtual environment).
- Git LFS (already configured for the dlib landmark model).

Verify Python:
```powershell
python --version
# Expected: Python 3.12.x
```

---

## 2. Install Dependencies
From the repository root (`Driver-Drowsiness-Detection`):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r Requirements.txt
```

> Tip: If activating fails in PowerShell, ensure script execution is allowed (`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`).

---

## 3. Launch the Application
With the virtual environment activated:

```powershell
python "Driver Drowsiness Detection.py"
```

Startup behavior:
- Automatically tries camera indices 0 → 2.
- Resizes input frames to 960×540 by default (edit `frame_width`/`frame_height` if needed).
- Loads the dlib 68-point landmark model from `dlib_shape_predictor/`.

### Alternate inputs & telemetry
- Pick cameras explicitly: `python "Driver Drowsiness Detection.py" --cameras 1 2`
- Replay a recorded drive: `python "Driver Drowsiness Detection.py" --video data\drive.mp4` (loops by default)
- Start directly from the video feed: add `--prefer-video`; stop looping with `--no-loop-video`
- Expose status over HTTP/WebSocket: `python "Driver Drowsiness Detection.py" --api-port 5001 --ws-port 5002`
  - HTTP: `GET http://localhost:5001/status`
  - WebSocket: subscribe to `ws://localhost:5002/`
- Feed ambient sensors: supply an HTTP endpoint that returns `{"ambient_light": <lux>, "cabin_temp": <°C>}` via `--sensor-http http://localhost:5055/env`

---

## 4. Live Controls (Hotkeys)
| Key | Action |
| --- | --- |
| `q` | Quit the application |
| `o` | Toggle all overlays (status, boxes, charts) |
| `k` | Toggle landmark points (must be ON to see pose lines/text) |
| `j` | Toggle landmark IDs |
| `p` | Toggle pose overlay (requires `k` ON to visualize) |
| `c` | Start/stop calibration (~6s neutral face) |
| `1` | Performance preset: speed (lower accuracy, higher FPS) |
| `2` | Preset: balanced (default) |
| `3` | Preset: quality (higher accuracy, lower FPS) |
| `[` / `]` | Cycle backward/forward through the configured camera indices |
| `v` | Toggle between live camera and loaded video file (when `--video` supplied) |

Status text at the bottom of the window confirms which overlays/presets are active.

---

## 5. Calibration Workflow
1. Press `c` to start.
2. Hold a neutral expression for ~6 seconds; the banner shows the countdown.
3. Baselines (`baseline_ear`, `baseline_mar`) persist in `calibration.json`.
4. Press `c` again to abort early if needed.

These baselines adjust thresholds to your face, reducing false positives.

---

## 6. Adaptive Drift & Environment Context
- The app monitors a 120 s rolling median of EAR/MAR. If it drifts >12 % from your stored baseline, an on-screen banner plus `calibration_advice.log` remind you to recalibrate.
- Optional ambient-light / cabin-temperature readings (via `--sensor-http`) tweak sensitivity on the fly: darker cabins require longer eye closures; hotter cabins make yawns contribute more to debt.
- When you exit, a snapshot of session averages, drift count, and environment stats is appended to `session_metrics.jsonl`.

---

## 7. Reading the UI
- **Face counter (top left)**: number of faces detected.
- **MAR / Score / Tilt**: live metrics used for scoring.
- **Debt bar**: shows cumulative fatigue (0–100). Fills when risk scores exceed the alert threshold and decays otherwise.
- **Alert banner**: “DROWSINESS ALERT!” appears with audible alarms when the combined score crosses the threshold.

---

## 8. Logging & Incidents
- `events_log.csv`: records eyes-closed events, yawns, and score-triggered alerts with timestamps.
- `incidents/`: each alert triggers an `incident_YYYYMMDD_HHMMSS/` folder containing:
  - `snapshot.jpg` — still frame at alert time.
  - `clip.mp4` — ~10-second rolling video buffer.
- Generate a quick report with charts + embedded videos:
  ```powershell
  python tools/review_session.py --log events_log.csv --incidents incidents --output reports --open
  ```
  This produces `reports/session_metrics.png` plus an HTML summary referencing your incident media.

Ensure `incidents/`, `events_log.csv`, and `calibration.json` are ignored in Git (already covered in `.gitignore`).

---

## 9. Health Checks
- **Quick dependency smoke test** (verifies OpenCV/dlib/numpy stack):
  ```powershell
  python smoke_test.py
  ```
- **Analytics/report sanity check** (ensures `tools/review_session.py` can read your logs/incidents):
  ```powershell
  python tools\review_session.py --log events_log.csv --incidents incidents --output reports --open
  ```
  Inspect the generated chart/HTML in `reports/`.

---

## 10. Troubleshooting
- **No pose overlay**: Press `k` to enable landmark points; pose lines only render when landmarks are visible.
- **No faces detected**: Improve lighting, keep your face centered, or check if another app is using the camera.
- **High CPU usage / lag**: Drop to preset `1`, reduce resolution, and turn off pose/IDs.
- **Camera busy**: Close other camera apps, then press `[`/`]` to retry other indices or launch with `--cameras <idx...>`.
- **Need to regression-test without a webcam**: pass `--video path\to\clip.mp4 --prefer-video` to replay files instead of camera input.
- **Remote monitoring**: ensure `--api-port`/`--ws-port` are open on the firewall; visit `/status` to verify telemetry JSON.
- **Sensor endpoint errors**: the app silently retries; check the console and ensure your `--sensor-http` target returns JSON with `ambient_light` and/or `cabin_temp`.

---

Happy driving, and stay alert!

