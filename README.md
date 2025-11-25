<h1 align="center">SafeWheel – Driver Drowsiness Detection</h1>

SafeWheel monitors live video (or recorded drives) for the classic drowsiness cues—eye closures, yawns, and head pose—then fuses them into a single risk score, an accumulating “Debt” meter, and audible/visual alerts. The system also logs every event to CSV, saves incident clips, exposes metrics via HTTP/WebSocket, and ships a reporting script for after-drive analysis.

---

## 1. Prerequisites & Environment

| Requirement | Notes |
| --- | --- |
| OS | Windows 10/11 preferred (shipped wheels target Windows). Linux/macOS also work if you install `dlib` from source. |
| Python | 3.12.x (matches `.venv`). |
| Hardware | Webcam capable of 640×480+ @ 20 FPS, microphone for alarms (optional but recommended). |
| Optional sensors | Any device that can publish `{"ambient_light": <lux>, "cabin_temp": <°C>}` over HTTP (ESP32, Pi, etc.). |

Clone and set up once:
```powershell
git clone https://github.com/Karthikprasadm/SafeWheel.git
cd SafeWheel
git lfs pull                                  # downloads dlib landmark model
python -m venv .venv
.\.venv\Scripts\activate
pip install -r Requirements.txt
```
> PowerShell script execution disabled? Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, open a new shell, and activate again.

### Dependency Stack
- Vision / geometry: `opencv-contrib-python`, `dlib`, `mediapipe`, `imutils`.
- Scientific: `numpy`, `scipy`.
- Telemetry: `websockets`, stdlib `http.server`.
- Reporting: `pandas`, `matplotlib`.

---

## 2. Running the Application

```powershell
cd SafeWheel
.\.venv\Scripts\activate
python "Driver Drowsiness Detection.py"
```

Key behavior:
- SafeWheel cycles camera indices **0 → 2** until it finds one that returns frames (override with `--cameras 1 3`).
- Frames are resized to **960×540** (changing `frame_width`/`frame_height` affects overlays + video clips).
- Exit with `q`. Use `[`/`]` to hop between configured cameras, `v` to swap between a video file and the live device.

### CLI Reference
| Flag | Description |
| --- | --- |
| `--cameras 0 2 4` | Specify camera indices to probe (order matters). |
| `--video path\drive.mp4` | Replay footage instead of live camera (loops by default). |
| `--prefer-video` / `--no-loop-video` | Start directly on the video source / stop rewinding at EOF. |
| `--api-port 5001 --ws-port 5002` | Enable HTTP JSON (`/status`) + WebSocket telemetry. |
| `--sensor-http http://host:5055/env` | Poll an HTTP endpoint for ambient-light & cabin-temperature readings. |
| `--stats-window 180` | Adjust the EAR/MAR drift window (seconds). |
| `--stats-log C:\logs\session_metrics.jsonl` | Custom path for per-session summaries. |

### Hotkeys (while window focused)
- `q` quit · `o` master overlay toggle · `k` landmark dots · `j` landmark IDs · `p` pose lines
- `[` / `]` previous/next camera · `v` toggle camera vs. `--video` source
- `1/2/3` performance presets (speed / balanced / quality)
- `c` start/stop calibration (6 s neutral expression). Status bar shows Calib: ON/YES/NO.

---

## 3. Calibration, Drift & Sensors

1. Press `c`, hold a neutral face for ~6 s, press `c` again to cancel early.
2. Baselines write to `calibration.json`.
3. Each frame recomputes thresholds as:
   - `ear_thresh = max(0.12, baseline_ear - 0.04)` (clamped to <= global default).
   - `mar_thresh = clamp(max(baseline_mar + 0.08, baseline_mar × 1.15), 0.55, 0.95)`.

### Drift Detection
- Rolling **120 s** median of EAR/MAR is compared against stored baselines.
- >12 % deviation triggers a banner + entry in `calibration_advice.log`.
- Session summaries (`session_metrics.jsonl`) capture average EAR/MAR, score mean, drift count, and environment stats.

### Environment Sensors
- Provide `--sensor-http URL` returning `{"ambient_light": <lux>, "cabin_temp": <°C>}`.
- Low light (<30 lux) lengthens the required consecutive eye-closure frames; bright cabins can shorten it slightly.
- Hot cabins (>32 °C) lower the yawn threshold; cold cabins (<15 °C) raise it.
- Sensor readings appear in the status bar (`Env: 42lx/29.1C`) and in telemetry/logs.

---

## 4. Scoring, Debt & Alerts

- **PERCLOS**: fraction of frames in the last 60 s window where EAR < threshold.
- **Yawn Rate**: yawns per minute (rolling 60 s, with cooldown to avoid double counting).
- **Head Tilt**: degrees above the configured alert angle (default 15°).
- **Score**: `0.6 * perclos + 0.4 * yawns + 0.0 * tilt` (weights tweak-able).

### Debt Meter
- Accumulates when eyes/yawns trigger risk, decays otherwise.
- Warning threshold `≥80` → three short beeps, repeats every 10 s while still ≥80.
- Critical threshold `=100` → 10 s tone, repeats every 5 s until the bar drops.
- The “DROWSINESS ALERT” banner flashes whenever risk is active; snapshots + clips are saved once per alert window.

---

## 5. Telemetry & Remote Monitoring

Start servers:
```powershell
python "Driver Drowsiness Detection.py" --api-port 5001 --ws-port 5002
```
- `GET http://localhost:5001/status` → JSON snapshot (score, debt, perclos, yawns/min, tilt, faces, env data, recalibration flag).
- WebSocket clients connect to `ws://localhost:5002/` and receive a JSON payload every time metrics update.
- Intended use cases: fleet dashboards, mobile notifications, control-room display.

---

## 6. Logging, Incidents & Reporting

### events_log.csv
Header:
```
timestamp,event,ear,mar,head_tilt_deg,faces,score,perclos,yawns_per_min,debt_score,ambient_light,cabin_temp
```
Event types include:
- `eyes_closed`, `yawn`, `score_alert` (risk crosses threshold), `metrics` (every ~2 s snapshot), etc.
Logs append across runs; if the header ever changes the old file is renamed to `events_log.csv.legacy`.

### incidents/
Each alert writes `incidents/incident_YYYYMMDD_HHMMSS/` with:
- `snapshot.jpg` – still image at alert start.
- `clip.mp4` – ~10 s video pulled from the rolling buffer.

### Reporting Script
Transform raw data into dashboards:
```powershell
python tools/review_session.py --log events_log.csv --incidents incidents --output reports --open
```
Outputs:
- `reports/session_metrics.png` – stacked chart of PERCLOS + Debt over time.
- `reports/session_report_*.html` – interactive summary with tables, chart, embedded snapshots, and playable MP4 clips.

---

## 7. Health Checks
- `python smoke_test.py` – verifies OpenCV, numpy, scipy, dlib, mediapipe import cleanly.
- `python tools/review_session.py ... --open` – ensures analytics stack (pandas/matplotlib) can parse latest logs.
- Check `session_metrics.jsonl` after each run to confirm drift/environment summaries save correctly.

---

## 8. Troubleshooting
- **No camera feed**: close Teams/Zoom, run `Get-Process python | Stop-Process` to free the device, then relaunch.
- **Pose overlay missing**: enable landmark points (`k`); pose lines only draw when points are visible.
- **High CPU / dropped frames**: switch to preset `1` (speed), lower `frame_width`, disable pose (`p`) + IDs (`j`).
- **Activation blocked**: set PowerShell execution policy or use `.\.venv\Scripts\activate.bat` from CMD.
- **Telemetry offline**: verify firewall ports, ensure `--api-port` isn’t 0, monitor console for HTTP binding errors.
- **Sensor endpoint failures**: console prints `[WARN] failed to start sensor poller`; confirm the URL returns JSON floats.
- **Debt beeps absent**: confirm speakers aren’t muted; Winsound falls back to console bell (`\a`) if audio hardware blocks.

---

## 9. Project Structure
```
Driver-Drowsiness-Detection/
├─ Driver Drowsiness Detection.py      # Main loop, CLI parser, telemetry, sensors
├─ EAR.py                              # Eye aspect ratio helper
├─ MAR.py                              # Mouth aspect ratio helper
├─ HeadPose.py                         # Head pose solver (PnP)
├─ tools/
│  └─ review_session.py                # CSV → charts + HTML report
├─ dlib_shape_predictor/
│  └─ shape_predictor_68_face_landmarks.dat  # 68-point landmark model (Git LFS)
├─ incidents/                          # Generated per-alert snapshots/videos
├─ reports/                            # Generated analytics artifacts
├─ events_log.csv                      # Rolling event log (CSV, auto-rotates if schema changes)
├─ session_metrics.jsonl               # One JSON line per app run
├─ Requirements.txt
└─ README.md / how_to_run&use.md       # Documentation
```

---

## 10. License
MIT – see `LICENSE`.
