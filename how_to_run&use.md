# How to Run & Use SafeWheel

This guide is a practical, copy/paste-friendly companion to `README.md`. Follow it from top to bottom when setting up a new machine or handing SafeWheel to another teammate.

---

## 1. Verify Requirements
| Item | Minimum | Notes |
| --- | --- | --- |
| OS | Windows 10/11 | Linux/macOS work if you can install `dlib` from source. |
| Python | 3.12.x | Matches the committed `.venv`. |
| Camera | 640×480 @ 20 FPS | USB or built-in webcam. |
| Git LFS | Enabled | Required for the dlib landmark model. |

```powershell
python --version     # expect 3.12.x
git lfs install
```

---

## 2. Create the Virtual Environment
Run these once per clone (from repo root `Driver-Drowsiness-Detection`):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # if blocked, run Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r Requirements.txt
```
Key packages installed: OpenCV, dlib-bin, mediapipe, numpy/scipy, websockets, pandas, matplotlib.

> Forgot to activate? Prefix commands with `.\.venv\Scripts\python.exe` (still works without touching execution policy).

---

## 3. Launch Scenarios
Base command (with venv active):
```powershell
python "Driver Drowsiness Detection.py"
```
SafeWheel auto-probes camera indices 0→2, resizes frames to 960×540, and shows the overlay window.

| Scenario | Command |
| --- | --- |
| Specify cameras | `python "Driver Drowsiness Detection.py" --cameras 2 4` |
| Replay a clip (loops) | `python "Driver Drowsiness Detection.py" --video data\drive.mp4` |
| Prefer the video feed first | add `--prefer-video` |
| Stop looping video | add `--no-loop-video` |
| Enable telemetry | `python "Driver Drowsiness Detection.py" --api-port 5001 --ws-port 5002` |
| Poll ambient sensors | `python "Driver Drowsiness Detection.py" --sensor-http http://localhost:5055/env` |
| Longer drift window / custom summary log | `python "Driver Drowsiness Detection.py" --stats-window 180 --stats-log C:\logs\session_metrics.jsonl` |

The console prints `[INFO]`/`[WARN]` messages if the camera or sensor endpoint fails. Fix issues there before looking elsewhere.

---

## 4. Live Controls & Status Bar
| Key | Action |
| --- | --- |
| `q` | Quit the app. |
| `o` | Master overlay toggle (hides everything except alert banner). |
| `k` / `j` | Landmark points / IDs. Pose lines render only when points are ON. |
| `p` | Pose overlay (blue/red orientation lines). |
| `[` / `]` | Previous/next camera from the configured list. |
| `v` | Switch between live camera and `--video` source. |
| `1 / 2 / 3` | Performance presets (speed / balanced / quality). |
| `c` | Start/stop calibration (~6 s neutral face). |

Status bar (bottom-left) shows overlay state, calibration flag, perf mode, and active input (`CAM 0`, `VIDEO clip.mp4`). If sensors are enabled you’ll also see `Env: 42lx/29.1C`.

---

## 5. Calibration & Drift
1. Press `c`; hold a neutral expression for 6 seconds (progress text appears).
2. Baselines write to `calibration.json` and immediately apply to EAR/MAR thresholds.
3. A rolling 120 s median monitors for drift. If medians move >12 % from the saved baseline you’ll see a yellow banner and `calibration_advice.log` gains a hint to recalibrate.
4. Session summaries append to `session_metrics.jsonl` with averages, drift count, and environment stats when you exit.

Tips:
- Recalibrate whenever lighting changes drastically (e.g., day → night) or if the “Calibrated” text disappears after editing the JSON manually.
- You can delete `calibration.json` to return to stock thresholds.

---

## 6. Environment Sensors & Telemetry
### Sensors
Provide an HTTP endpoint returning JSON:
```json
{ "ambient_light": 38.5, "cabin_temp": 30.2 }
```
Launch with `--sensor-http http://<device>/env`. The app polls every 15 s (configurable via `--sensor-interval`). Low light increases the required closed-eye frames; high cabin temps lower the yawning threshold.

### Telemetry API / WebSocket
```powershell
python "Driver Drowsiness Detection.py" --api-port 5001 --ws-port 5002
```
- `GET http://localhost:5001/status` → JSON snapshot (`score`, `perclos`, `debt_score`, `faces_detected`, `ambient_light`, etc.).
- `ws://localhost:5002/` pushes the same payload whenever metrics update (great for dashboards or fleet monitoring).

---

## 7. UI Elements
- **Face counter** (top-left) – active faces. Only the largest face drives scoring.
- **Score readout** – `Score`, `PERCLOS`, `YPM`, `Tilt`, plus “Calibrated” label if baselines exist.
- **Debt bar** – 0–100 scale. ≥80 triggers repeated warning beeps; 100 triggers repeated 10 s alarms until it drops.
- **Alert banner** – “DROWSINESS ALERT!” draws even when overlays are off. Incident media is saved exactly once per continuous alert.

---

## 8. Logging, Incidents & Reports
| Artifact | Location | Notes |
| --- | --- | --- |
| Event log | `events_log.csv` | Header: `timestamp,event,ear,mar,head_tilt_deg,faces,score,perclos,yawns_per_min,debt_score,ambient_light,cabin_temp`. Appends every run. If schema changes the previous file is renamed `.legacy`. |
| Incidents | `incidents/incident_YYYYMMDD_HHMMSS/` | Contains `snapshot.jpg` and `clip.mp4` (~10 s). |
| Session summary | `session_metrics.jsonl` | One JSON line per app exit with averages and drift stats. |
| Analytics output | `reports/` | Generated by `tools/review_session.py` (chart + HTML gallery). |

Generate a report at any time:
```powershell
python tools\review_session.py --log events_log.csv --incidents incidents --output reports --open
```
Open the HTML report to review charts and incident media without relaunching the app.

---

## 9. Health Checks
| Check | Command | Purpose |
| --- | --- | --- |
| Dependency smoke test | `python smoke_test.py` | Ensures OpenCV / numpy / dlib / mediapipe import cleanly. |
| Analytics sanity check | `python tools\review_session.py --log events_log.csv --incidents incidents --output reports --open` | Confirms pandas/matplotlib can parse the latest logs and incidents. |
| Camera free? | `Get-Process python | Stop-Process -Force` | Releases any orphaned python processes still holding the webcam. |

Run these when onboarding a new machine or before demos.

---

## 10. Troubleshooting
- **PowerShell blocks Activate.ps1** → Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, reopen PowerShell, activate again. Alternatively use `cmd.exe` with `.\.venv\Scripts\activate.bat`.
- **“No working camera found”** → close other apps, unplug/replug USB camera, optionally reduce `--cameras` to the exact index used by Windows.
- **Pose overlay missing** → press `k`. Pose lines require landmarks to be visible.
- **High CPU / low FPS** → switch to preset `1`, lower frame resolution, disable pose (`p`) and IDs (`j`), or run from a more powerful machine.
- **Telemetry unreachable** → ensure Windows Firewall allows the chosen ports, check console for binding errors.
- **Sensor endpoint errors** → console prints `[WARN] failed to start sensor poller`. Hit the URL in a browser and confirm it returns valid JSON numbers.
- **Debt beeps not audible** → speakers muted? When Winsound fails, SafeWheel falls back to console bell (`\a`). Consider external buzzers for noisy environments.

---

Happy driving, stay alert, and review your logs often!
