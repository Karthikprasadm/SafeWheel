<h1 align="center">SafeWheel – Driver Drowsiness Detection</h1>

Real‑time, driver‑focused drowsiness detection using eye aspect ratio (EAR), mouth aspect ratio (MAR), and head‑tilt. Includes a combined drowsiness score, a persistent Debt meter (cumulative fatigue), audible alarms, incident capture (snapshot + 10s clip), performance presets, MediaPipe face detection fallback, multi‑source video input (live cameras or replay files), and live telemetry over HTTP/WebSocket.

### Features
- Drowsiness cues:
  - Eye closure via EAR (blink/microsleep)
  - Yawn via MAR
  - Head‑tilt estimate (pose)
- Scoring and persistence:
  - Combined score = PERCLOS (60s) + yawns/min + tilt (weighted)
  - Debt meter (0–100) accumulates sustained risk and decays during recovery
- Alerts and UX:
  - Audible alarm with cooldown; large on‑screen alert banner
  - Overlays you can toggle live (points, IDs, pose)
  - Performance presets (speed/balanced/quality)
- Logging and review:
  - `events_log.csv` (timestamped events/counters)
  - Incident auto‑save: `incidents/incident_YYYYMMDD_HHMMSS/` with `snapshot.jpg` and `clip.mp4` (~10s)
  - Session summary JSONL + HTML report generator (`tools/review_session.py`)
- Personalization:
  - Calibration (EAR/MAR baselines) saved to `calibration.json`
- Inputs & telemetry:
  - Multi‑camera cycling + optional video file replay (`--video`) for regression tests
  - Live telemetry API (`/status`) and WebSocket feed for fleet dashboards/remote monitoring
- Adaptive environment awareness:
  - Rolling EAR/MAR medians detect drift and prompt re‑calibration
  - Optional ambient-light & cabin-temperature sensors auto-adjust thresholds

### Requirements
Install Python 3.12+ (Windows recommended). Then:

```sh
pip install -r Requirements.txt
```

This installs OpenCV (contrib), dlib (prebuilt on Windows), imutils, numpy, scipy, mediapipe (for robust face detection), `websockets`, `pandas`, `matplotlib`, and dependencies.

If you are on Windows and see build issues for dlib on non‑Windows environments, set the Windows wheel `dlib-bin` or install from source as needed.

### Model file
The dlib 68‑landmark model is expected at:
```
dlib_shape_predictor/shape_predictor_68_face_landmarks.dat
```
This file is tracked via Git LFS in this repository.

### Quick Start
```sh
python -m venv .venv
.venv\Scripts\pip install -r Requirements.txt
python "Driver Drowsiness Detection.py"
```
- The app will try camera indices 0 → 2 and pick the first that returns frames.
- Default output size is 960×540; change `frame_width`/`frame_height` in the script for other resolutions.
- If you prefer activating the virtual environment, run `.\.venv\Scripts\activate` (PowerShell) before launching the script.

#### Alternate inputs
- Use specific cameras: `python "Driver Drowsiness Detection.py" --cameras 1 2`
- Start from a video file (loops by default): `python "Driver Drowsiness Detection.py" --video sample.mp4`
- Force video first: add `--prefer-video`; disable looping with `--no-loop-video`
- Runtime hotkeys: `[`/`]` cycle through camera list, `v` toggles between live camera and the loaded video file
- Poll environment sensors (HTTP JSON with `ambient_light`, `cabin_temp`): `python "Driver Drowsiness Detection.py" --sensor-http http://localhost:5055/env`
- Export live metrics: `python "Driver Drowsiness Detection.py" --api-port 5001 --ws-port 5002`

### Health Checks & QA
- Validate dependencies + major libraries: `python smoke_test.py` (prints OpenCV/numpy/dlib versions).
- Generate analytics artifacts for any recorded run:  
  `python tools/review_session.py --log events_log.csv --incidents incidents --output reports --open`
- The script creates `reports/session_metrics.png` and `session_report_YYYYMMDD_HHMMSS.html`. Open the HTML to confirm incidents render.

### Controls (Hotkeys)
- q: Quit
- o: Toggle overlays (master)
- k: Toggle landmark points
- j: Toggle landmark IDs
- p: Toggle pose lines/text
- c: Start/stop calibration (~6s, neutral face)
- 1/2/3: Performance preset (speed/balanced/quality)
- [: Previous camera in the configured list
- ]: Next camera in the configured list
- v: Toggle between live camera and video file (when `--video` supplied)

### Calibration
- Press c to start; hold a neutral face for ~6 seconds.
- Baselines saved to `calibration.json`.
- After calibration, thresholds adapt to your EAR/MAR to reduce false alarms.

### Adaptive Drift & Environment Sensors
- A rolling 120 s window of EAR/MAR is monitored. If medians drift >12 % from your saved baseline you’ll see a banner plus a note in `calibration_advice.log`, guiding you to press `c` and refresh thresholds.
- When available, `ambient_light` (lux) and `cabin_temp` (°C) values from `--sensor-http` slightly retune detection (e.g., dim cabins require longer eye-closure streaks; hot cabins treat yawns with higher weight).
- Session summaries (averages, drift count, environment min/max) append to `session_metrics.jsonl` when you exit.

### Drowsiness Score and Debt
- Score blends PERCLOS (60s window), yawns/min, and head‑tilt using weights in the script (`SCORE_WEIGHTS`).
- When score ≥ threshold (default 0.6), an alert triggers and Debt accumulates; otherwise Debt decays.
- The Debt bar (0–100) highlights sustained risk.

### Telemetry API & WebSocket
- Start the HTTP API with `--api-port 5001` (binds to `0.0.0.0` by default). Fetch live metrics at `http://HOST:5001/status`.
- Enable the WebSocket feed with `--ws-port 5002` (or omit to auto‑use `api-port+1`). Subscribe for JSON push updates of score, debt, faces, etc.
- Endpoints return current frame metrics (`score`, `debt_score`, `perclos`, `yawns_per_min`, `input_source`, alert flags).

### Session Review Tools
- Build quick-look charts + incident gallery:  
  `python tools/review_session.py --log events_log.csv --incidents incidents --output reports --open`
- The script produces `reports/session_metrics.png` (PERCLOS/Debt plots) and an HTML file with embedded snapshots/video clips.
- `events_log.csv` now includes `score`, `perclos`, `debt_score`, `yawns_per_min`, `ambient_light`, and `cabin_temp`, so you can import it straight into pandas as well.

### Incidents and Logging
- Events are appended to `events_log.csv`.
- On alert, a snapshot and ~10s video clip are saved to `incidents/incident_YYYYMMDD_HHMMSS/`.

### Performance Tips
- Use presets:
  - 1 (speed): detect every 6 frames @ 0.5x; pose off; IDs off
  - 2 (balanced): detect every 3 frames @ 0.6x (default)
  - 3 (quality): detect every 2 frames @ 0.8x
- Keep overlays minimal while driving (o/k/j/p) to maximize FPS.
- Ensure even lighting; reduce screen/cabin glare.

### Troubleshooting
- No faces detected: improve lighting, face the camera squarely, ensure the face is fully in frame. MediaPipe fallback is enabled.
- Pose overlay looks missing: press `k` to turn landmark points ON; pose lines/text only draw when points are visible.
- High CPU/lag: lower resolution, use preset 1, turn off pose and IDs.
- Camera busy: close other apps using the camera, or change index in the script.

### Project Structure
```
Driver-Drowsiness-Detection/
├─ Driver Drowsiness Detection.py      # Main application
├─ EAR.py                              # Eye aspect ratio (EAR)
├─ MAR.py                              # Mouth aspect ratio (MAR)
├─ HeadPose.py                         # Head pose (PnP, tilt)
├─ tools/
│  └─ review_session.py                # Build analytics charts + HTML report
├─ dlib_shape_predictor/
│  └─ shape_predictor_68_face_landmarks.dat  # Landmark model (LFS)
├─ reports/                            # (Generated) charts + HTML from review_session.py
├─ Requirements.txt
└─ README.md
```

### License
MIT. See `LICENSE`.
