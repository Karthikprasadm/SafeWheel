<h1 align="center">SafeWheel – Driver Drowsiness Detection</h1>

Real‑time, driver‑focused drowsiness detection using eye aspect ratio (EAR), mouth aspect ratio (MAR), and head‑tilt. Includes a combined drowsiness score, a persistent Debt meter (cumulative fatigue), audible alarms, incident capture (snapshot + 10s clip), performance presets, and a MediaPipe face detection fallback.

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
- Personalization:
  - Calibration (EAR/MAR baselines) saved to `calibration.json`
- Robustness:
  - MediaPipe face detection fallback; dlib landmarks; camera auto‑selection

### Requirements
Install Python 3.12+ (Windows recommended). Then:

```sh
pip install -r Requirements.txt
```

This installs OpenCV (contrib), dlib (prebuilt on Windows), imutils, numpy, scipy, mediapipe (for robust face detection) and dependencies.

If you are on Windows and see build issues for dlib on non‑Windows environments, set the Windows wheel `dlib-bin` or install from source as needed.

### Model file
The dlib 68‑landmark model is expected at:
```
dlib_shape_predictor/shape_predictor_68_face_landmarks.dat
```
This file is tracked via Git LFS in this repository.

### Quick Start
```sh
python "Driver Drowsiness Detection.py"
```
- The app will try camera indices 0 → 2 and pick the first that returns frames.
- Default output size is 960×540; change `frame_width`/`frame_height` in the script for other resolutions.

### Controls (Hotkeys)
- q: Quit
- o: Toggle overlays (master)
- k: Toggle landmark points
- j: Toggle landmark IDs
- p: Toggle pose lines/text
- c: Start/stop calibration (~6s, neutral face)
- 1/2/3: Performance preset (speed/balanced/quality)

### Calibration
- Press c to start; hold a neutral face for ~6 seconds.
- Baselines saved to `calibration.json`.
- After calibration, thresholds adapt to your EAR/MAR to reduce false alarms.

### Drowsiness Score and Debt
- Score blends PERCLOS (60s window), yawns/min, and head‑tilt using weights in the script (`SCORE_WEIGHTS`).
- When score ≥ threshold (default 0.6), an alert triggers and Debt accumulates; otherwise Debt decays.
- The Debt bar (0–100) highlights sustained risk.

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
- High CPU/lag: lower resolution, use preset 1, turn off pose and IDs.
- Camera busy: close other apps using the camera, or change index in the script.

### Project Structure
```
Driver-Drowsiness-Detection/
├─ Driver Drowsiness Detection.py      # Main application
├─ EAR.py                              # Eye aspect ratio (EAR)
├─ MAR.py                              # Mouth aspect ratio (MAR)
├─ HeadPose.py                         # Head pose (PnP, tilt)
├─ dlib_shape_predictor/
│  └─ shape_predictor_68_face_landmarks.dat  # Landmark model (LFS)
├─ Requirements.txt
└─ README.md
```

### License
MIT. See `LICENSE`.
