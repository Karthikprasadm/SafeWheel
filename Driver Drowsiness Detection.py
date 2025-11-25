#!/usr/bin/env python
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import asyncio
import imutils
import time
import dlib
import math
import os
import csv
from datetime import datetime
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import statistics as stats
import urllib.request
import urllib.error
import cv2
import numpy as np
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords
from collections import deque
try:
    import websockets
except Exception:
    websockets = None
parser = argparse.ArgumentParser(description="Driver Drowsiness Detection")
parser.add_argument(
    "--cameras",
    type=int,
    nargs="*",
    help="List of camera indices to probe (default: 0 1 2)."
)
parser.add_argument(
    "--video",
    type=str,
    help="Optional video file to replay as an input source/fallback."
)
parser.add_argument(
    "--prefer-video",
    action="store_true",
    help="Start with the video file when provided (otherwise use cameras first)."
)
parser.add_argument(
    "--loop-video",
    dest="loop_video",
    action="store_true",
    help="Loop the provided video file (default)."
)
parser.add_argument(
    "--no-loop-video",
    dest="loop_video",
    action="store_false",
    help="Stop when the provided video file ends."
)
parser.set_defaults(loop_video=True)
parser.add_argument(
    "--api-port",
    type=int,
    default=0,
    help="Expose a JSON telemetry API on this port (0 disables)."
)
parser.add_argument(
    "--ws-port",
    type=int,
    default=None,
    help="Port for the telemetry WebSocket feed (defaults to api-port+1 when API is enabled)."
)
parser.add_argument(
    "--api-host",
    type=str,
    default="0.0.0.0",
    help="Bind address for telemetry HTTP/WebSocket servers."
)
parser.add_argument(
    "--sensor-http",
    type=str,
    help="Optional HTTP endpoint returning JSON with ambient_light (lux) and cabin_temp (C)."
)
parser.add_argument(
    "--sensor-interval",
    type=float,
    default=15.0,
    help="Seconds between environment sensor polls (default: 15)."
)
parser.add_argument(
    "--stats-window",
    type=float,
    default=120.0,
    help="Seconds of EAR/MAR history to use for drift detection (default: 120)."
)
parser.add_argument(
    "--stats-log",
    type=str,
    default=None,
    help="Optional path for session metric summaries (defaults to session_metrics.jsonl)."
)
args = parser.parse_args()
camera_indices = args.cameras if args.cameras else [0, 1, 2]
video_path = os.path.abspath(args.video) if args.video else None
loop_video = args.loop_video
prefer_video = args.prefer_video and bool(video_path)
api_host = args.api_host
api_port = max(0, args.api_port)
ws_port = args.ws_port if args.ws_port is not None else (api_port + 1 if api_port else 0)
sensor_http_endpoint = args.sensor_http
sensor_poll_interval = max(5.0, args.sensor_interval)
stats_window_secs = max(30.0, args.stats_window)
session_metrics_path = os.path.abspath(args.stats_log) if args.stats_log else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'session_metrics.jsonl')

try:
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
    mp_fd = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
except Exception:
    mp_fd = None

# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
script_dir = os.path.dirname(os.path.abspath(__file__))
predictor_path = os.path.join(
    script_dir, 'dlib_shape_predictor', 'shape_predictor_68_face_landmarks.dat')
if not os.path.exists(predictor_path):
    raise FileNotFoundError(
        "Cannot find dlib predictor file at: {}".format(predictor_path))
predictor = dlib.shape_predictor(predictor_path)

def clamp(value, min_value, max_value):
    if value is None:
        return None
    return max(min_value, min(max_value, value))

class VideoInputManager:
    """Manage multiple cameras and optional video file input."""

    def __init__(self, camera_indices, frame_width, frame_height, video_path=None,
                 loop_video=True, prefer_video=False):
        self.camera_indices = camera_indices or []
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.video_path = video_path
        self.loop_video = loop_video
        self.prefer_video = prefer_video and bool(video_path)
        self._current_camera_pos = 0
        self.active_camera_index = None
        self.mode = None
        self.stream = None
        self.video_cap = None
        self._lock = threading.Lock()
        self._initialize()

    @property
    def status_label(self):
        if self.mode == 'camera' and self.active_camera_index is not None:
            return f'CAM {self.active_camera_index}'
        if self.mode == 'video' and self.video_path:
            return f'VIDEO {os.path.basename(self.video_path)}'
        return 'NONE'

    def _initialize(self):
        if self.prefer_video and self._open_video():
            return
        if self.camera_indices and self._open_any_camera():
            return
        if self.video_path and self.mode != 'video' and self._open_video():
            return
        raise RuntimeError("No working camera or video input could be initialized.")

    def _open_any_camera(self):
        for offset in range(len(self.camera_indices)):
            pos = (self._current_camera_pos + offset) % len(self.camera_indices)
            if self._open_camera_at(pos):
                self._current_camera_pos = pos
                return True
        return False

    def _open_camera_at(self, position):
        if position >= len(self.camera_indices):
            return False
        index = self.camera_indices[position]
        # stop any previous camera stream
        self._stop_camera()
        vs_try = None
        try:
            vs_try = VideoStream(src=index).start()
            time.sleep(1.0)
            test_frame = vs_try.read()
            if test_frame is None:
                raise RuntimeError("Camera returned empty frame")
            try:
                vs_try.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                vs_try.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            except Exception:
                pass
            self.stream = vs_try
            self.mode = 'camera'
            self.active_camera_index = index
            print(f"[INFO] camera opened at index {index}")
            return True
        except Exception as exc:
            print(f"[WARN] failed to open camera {index}: {exc}")
            if vs_try is not None:
                try:
                    vs_try.stop()
                except Exception:
                    pass
            self.stream = None
            self.active_camera_index = None
            return False

    def _open_video(self):
        if not self.video_path:
            return False
        self._stop_video()
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[WARN] cannot open video file: {self.video_path}")
            cap.release()
            return False
        self.video_cap = cap
        self.mode = 'video'
        print(f"[INFO] streaming from video file {self.video_path} (loop={'on' if self.loop_video else 'off'})")
        return True

    def _stop_camera(self):
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
            self.stream = None
        self.active_camera_index = None

    def _stop_video(self):
        if self.video_cap is not None:
            try:
                self.video_cap.release()
            except Exception:
                pass
            self.video_cap = None

    def read(self):
        with self._lock:
            if self.mode == 'camera':
                if self.stream is None and not self._open_any_camera():
                    return None
                frame = self.stream.read() if self.stream else None
                if frame is None and self._open_any_camera():
                    frame = self.stream.read() if self.stream else None
                return frame
            if self.mode == 'video':
                if self.video_cap is None and not self._open_video():
                    return None
                ret, frame = self.video_cap.read()
                if not ret:
                    if self.loop_video:
                        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.video_cap.read()
                    if not ret:
                        return None
                return frame
            return None

    def cycle_camera(self, step=1):
        if not self.camera_indices:
            return False
        self.mode = 'camera'
        if step == 0:
            step = 1
        target = (self._current_camera_pos + step) % len(self.camera_indices)
        start = target
        for _ in range(len(self.camera_indices)):
            if self._open_camera_at(target):
                self._current_camera_pos = target
                return True
            target = (target + step) % len(self.camera_indices)
            if target == start:
                break
        return False

    def toggle_video(self):
        if not self.video_path:
            return False
        if self.mode == 'video':
            return self._open_any_camera()
        return self._open_video()

    def stop(self):
        self._stop_camera()
        self._stop_video()


# 960x540 for higher clarity (adjust if needed)
frame_width = 960
frame_height = 540

# initialize the video stream / file input
print("[INFO] initializing video input...")
vs = VideoInputManager(
    camera_indices=camera_indices,
    frame_width=frame_width,
    frame_height=frame_height,
    video_path=video_path,
    loop_video=loop_video,
    prefer_video=prefer_video
)
time.sleep(0.5)

# Alarm, scoring and logging configuration
ALARM_COOLDOWN_SECS = 3.0
last_alarm_time = {"eyes": 0.0, "yawn": 0.0}

# Drowsiness score configuration
PERCLOS_WINDOW_SECS = 60.0
YAWN_COOLDOWN_SECS = 2.0
YAWN_RATE_WINDOW_SECS = 60.0
HEAD_TILT_ALERT_DEG = 15.0
YAWNS_PER_MIN_ALERT = 3.0
SCORE_WEIGHTS = {"perclos": 0.6, "yawn": 0.4, "tilt": 0.0}
EYE_ALERT_SCORE = 0.75
YAWN_ALERT_SCORE = 0.65
ALERT_SCORE_THRESHOLD = 0.6

# Incident capture configuration
INCIDENT_DIR = os.path.join(script_dir, 'incidents')
os.makedirs(INCIDENT_DIR, exist_ok=True)
VIDEO_CLIP_SECS = 10.0

# Calibration configuration (personalized thresholds)
CALIB_SECONDS = 6.0
calibrating = False
calib_start_ts = 0.0
calib_ear_samples: deque = deque()
calib_mar_samples: deque = deque()
baseline_ear = None
baseline_mar = None
calib_path = os.path.join(script_dir, 'calibration.json')
CALIB_MIN_EAR = 0.18   # ignore samples if eyes look closed
CALIB_MAX_MAR = 0.7    # ignore samples if mouth is wide open
BASELINE_EAR_RANGE = (0.15, 0.4)
BASELINE_MAR_RANGE = (0.30, 0.65)
try:
    if os.path.exists(calib_path):
        with open(calib_path, 'r', encoding='utf-8') as f:
            _c = json.load(f)
            baseline_ear = clamp(_c.get('baseline_ear'), *BASELINE_EAR_RANGE)
            baseline_mar = clamp(_c.get('baseline_mar'), *BASELINE_MAR_RANGE)
except Exception:
    pass

def persist_calibration(ear_value, mar_value):
    """Persist baseline EAR/MAR thresholds to memory and disk."""
    global baseline_ear, baseline_mar
    ear_value = clamp(ear_value, *BASELINE_EAR_RANGE) if ear_value is not None else None
    mar_value = clamp(mar_value, *BASELINE_MAR_RANGE) if mar_value is not None else None
    baseline_ear = ear_value
    baseline_mar = mar_value
    try:
        with open(calib_path, 'w', encoding='utf-8') as f:
            json.dump({'baseline_ear': baseline_ear, 'baseline_mar': baseline_mar}, f)
    except Exception:
        pass

# Drowsiness Debt Meter (accumulate fatigue over time with decay)
debt_score = 0.0  # 0..100
DEBT_DECAY_PER_SEC_LOW = 6.0
DEBT_DECAY_PER_SEC_HIGH = 2.0
DEBT_ACCUM_FACTOR = 30.0  # points/sec when risk score == 1.0
DEBT_WARN_THRESHOLD = 80.0
DEBT_CRITICAL_THRESHOLD = 100.0
DEBT_WARN_REPEAT_SECS = 10.0
DEBT_CRITICAL_REPEAT_SECS = 5.0
prev_ts = None

def play_alarm_async(duration_ms=500, repeat=2):
    try:
        import winsound
        def _beep():
            try:
                for _ in range(repeat):
                    winsound.Beep(1500, duration_ms // repeat)
                    time.sleep(0.05)
            except Exception:
                pass
    except Exception:
        def _beep():
            try:
                for _ in range(repeat):
                    print("\a", end="")
                    time.sleep(duration_ms / 1000.0 / max(1, repeat))
            except Exception:
                pass
    threading.Thread(target=_beep, daemon=True).start()

telemetry_state = {}
telemetry_lock = threading.Lock()
telemetry_server = None


class TelemetryServer:
    """Lightweight HTTP/WebSocket telemetry broadcaster."""

    def __init__(self, host, http_port, ws_port, snapshot_cb):
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port
        self.snapshot_cb = snapshot_cb
        self.http_thread = None
        self.ws_thread = None
        self.ws_loop = None
        self.ws_clients = set()
        self._clients_lock = threading.Lock()

    def start(self):
        if self.http_port:
            self.http_thread = threading.Thread(target=self._run_http, daemon=True)
            self.http_thread.start()
            print(f"[INFO] telemetry HTTP API listening on {self.host}:{self.http_port}/status")
        if self.ws_port:
            if websockets is None:
                print("[WARN] websockets package not installed; WebSocket feed disabled.")
            else:
                self.ws_loop = asyncio.new_event_loop()
                self.ws_thread = threading.Thread(target=self._run_ws_loop, daemon=True)
                self.ws_thread.start()
                print(f"[INFO] telemetry WebSocket listening on ws://{self.host}:{self.ws_port}/")

    def _run_http(self):
        snapshot_cb = self.snapshot_cb

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self_inner):
                path = self_inner.path.split('?', 1)[0]
                if path in ('/', '/status'):
                    payload = json.dumps(snapshot_cb()).encode('utf-8')
                    self_inner.send_response(200)
                    self_inner.send_header('Content-Type', 'application/json')
                    self_inner.send_header('Content-Length', str(len(payload)))
                    self_inner.end_headers()
                    self_inner.wfile.write(payload)
                elif path == '/healthz':
                    self_inner.send_response(200)
                    self_inner.end_headers()
                else:
                    self_inner.send_response(404)
                    self_inner.end_headers()

            def log_message(self_inner, format, *args):
                # Silence default HTTP request logs
                return

        httpd = ThreadingHTTPServer((self.host, self.http_port), Handler)
        httpd.serve_forever()

    async def _ws_handler(self, websocket):
        with self._clients_lock:
            self.ws_clients.add(websocket)
        try:
            await websocket.send(json.dumps(self.snapshot_cb()))
            async for _ in websocket:
                # Keep the connection open; clients pull data via pushes
                await asyncio.sleep(0.1)
        except Exception:
            pass
        finally:
            with self._clients_lock:
                self.ws_clients.discard(websocket)

    def _run_ws_loop(self):
        asyncio.set_event_loop(self.ws_loop)
        server = websockets.serve(self._ws_handler, self.host, self.ws_port)
        self.ws_loop.run_until_complete(server)
        self.ws_loop.run_forever()

    def broadcast(self, payload):
        if not self.ws_loop or not self.ws_port or websockets is None:
            return
        message = json.dumps(payload)
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self.ws_loop)

    async def _broadcast(self, message):
        with self._clients_lock:
            targets = list(self.ws_clients)
        if not targets:
            return
        coros = []
        for ws in targets:
            coros.append(self._safe_send(ws, message))
        await asyncio.gather(*coros, return_exceptions=True)

    async def _safe_send(self, ws, message):
        try:
            await ws.send(message)
        except Exception:
            with self._clients_lock:
                self.ws_clients.discard(ws)


def snapshot_telemetry():
    with telemetry_lock:
        return dict(telemetry_state)


def update_telemetry(**kwargs):
    with telemetry_lock:
        telemetry_state.update(kwargs)
        payload = dict(telemetry_state)
    if telemetry_server:
        telemetry_server.broadcast(payload)


environment_state = {"ambient_light": None, "cabin_temp": None, "last_updated": None}
environment_stats = {
    "ambient_light": {"min": None, "max": None, "sum": 0.0, "count": 0},
    "cabin_temp": {"min": None, "max": None, "sum": 0.0, "count": 0}
}
environment_lock = threading.Lock()


def update_environment_data(**kwargs):
    changed = False
    now = time.time()
    with environment_lock:
        for key in ("ambient_light", "cabin_temp"):
            value = kwargs.get(key)
            if value is None:
                continue
            environment_state[key] = value
            stats_entry = environment_stats[key]
            stats_entry["min"] = value if stats_entry["min"] is None else min(stats_entry["min"], value)
            stats_entry["max"] = value if stats_entry["max"] is None else max(stats_entry["max"], value)
            stats_entry["sum"] += value
            stats_entry["count"] += 1
            changed = True
        if changed:
            environment_state["last_updated"] = now
    if changed:
        update_telemetry(
            ambient_light=environment_state.get("ambient_light"),
            cabin_temp=environment_state.get("cabin_temp")
        )


def get_environment_snapshot():
    with environment_lock:
        return dict(environment_state)


class SensorPoller:
    def __init__(self, url, interval, callback):
        self.url = url
        self.interval = interval
        self.callback = callback
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        while True:
            try:
                with urllib.request.urlopen(self.url, timeout=5) as resp:
                    payload = resp.read()
                    data = json.loads(payload.decode('utf-8'))
                    reading = {}
                    if 'ambient_light' in data:
                        try:
                            reading['ambient_light'] = float(data['ambient_light'])
                        except Exception:
                            pass
                    if 'cabin_temp' in data:
                        try:
                            reading['cabin_temp'] = float(data['cabin_temp'])
                        except Exception:
                            pass
                    if reading:
                        self.callback(**reading)
            except Exception:
                pass
            time.sleep(self.interval)


session_window_secs = stats_window_secs
ear_history: deque = deque()
mar_history: deque = deque()
session_ear_stats = {"sum": 0.0, "count": 0, "min": None, "max": None}
session_mar_stats = {"sum": 0.0, "count": 0, "min": None, "max": None}
session_score_stats = {"sum": 0.0, "count": 0}
drift_notice_until = 0.0
last_drift_log_ts = 0.0
drift_event_count = 0
session_start_ts = time.time()
last_metrics_log_ts = 0.0
drift_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration_advice.log')


def _append_history(history, ts, value):
    history.append((ts, value))
    while history and (ts - history[0][0]) > session_window_secs:
        history.popleft()


def _update_session_stats(stats_bucket, value):
    if value is None:
        return
    stats_bucket["sum"] += value
    stats_bucket["count"] += 1
    stats_bucket["min"] = value if stats_bucket["min"] is None else min(stats_bucket["min"], value)
    stats_bucket["max"] = value if stats_bucket["max"] is None else max(stats_bucket["max"], value)


def _running_median(history):
    if not history:
        return None
    values = [v for _, v in history]
    try:
        return float(stats.median(values))
    except Exception:
        return float(sum(values) / len(values))


def _write_drift_log(message):
    global last_drift_log_ts
    now = time.time()
    if now - last_drift_log_ts < 60:
        return
    last_drift_log_ts = now
    try:
        with open(drift_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} - {message}\n")
    except Exception:
        pass


log_path = os.path.join(script_dir, 'events_log.csv')
LOG_HEADER = [
    'timestamp', 'event', 'ear', 'mar', 'head_tilt_deg', 'faces',
    'score', 'perclos', 'yawns_per_min', 'debt_score', 'ambient_light', 'cabin_temp'
]
if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
    try:
        with open(log_path, 'r', encoding='utf-8') as existing:
            first_line = existing.readline().strip()
        if first_line != ','.join(LOG_HEADER):
            backup = log_path + '.legacy'
            os.replace(log_path, backup)
            print(f"[WARN] existing events_log header differed; moved legacy log to {backup}")
    except Exception:
        pass
log_file = open(log_path, 'a', newline='', encoding='utf-8')
csv_writer = csv.writer(log_file)
if log_file.tell() == 0:
    csv_writer.writerow(LOG_HEADER)


def log_event(event_type, ear=None, mar=None, head_tilt_deg=None, faces=0,
              score=None, perclos=None, yawns_per_min=None, debt_score=None,
              ambient_light=None, cabin_temp=None):
    csv_writer.writerow([
        datetime.now().isoformat(timespec='seconds'),
        event_type,
        None if ear is None else float(ear),
        None if mar is None else float(mar),
        None if head_tilt_deg is None else float(head_tilt_deg),
        int(faces),
        None if score is None else float(score),
        None if perclos is None else float(perclos),
        None if yawns_per_min is None else float(yawns_per_min),
        None if debt_score is None else float(debt_score),
        None if ambient_light is None else float(ambient_light),
        None if cabin_temp is None else float(cabin_temp)
    ])
    try:
        log_file.flush()
    except Exception:
        pass


def summarize_stats(bucket):
    if bucket["count"] == 0:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": bucket["sum"] / bucket["count"],
        "min": bucket["min"],
        "max": bucket["max"]
    }


def summarize_env(bucket):
    if bucket["count"] == 0:
        return {"avg": None, "min": None, "max": None}
    return {
        "avg": bucket["sum"] / bucket["count"],
        "min": bucket["min"],
        "max": bucket["max"]
    }


def write_session_summary():
    summary = {
        "ended_at": datetime.now().isoformat(timespec='seconds'),
        "duration_secs": time.time() - session_start_ts,
        "baseline_ear": baseline_ear,
        "baseline_mar": baseline_mar,
        "ear_stats": summarize_stats(session_ear_stats),
        "mar_stats": summarize_stats(session_mar_stats),
        "score_mean": (session_score_stats["sum"] / session_score_stats["count"])
        if session_score_stats["count"] else None,
        "drift_events": drift_event_count,
        "ambient_light": summarize_env(environment_stats["ambient_light"]),
        "cabin_temp": summarize_env(environment_stats["cabin_temp"])
    }
    try:
        with open(session_metrics_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(summary) + "\n")
    except Exception:
        pass

if api_port or ws_port:
    telemetry_server = TelemetryServer(api_host, api_port, ws_port, snapshot_telemetry)
    telemetry_server.start()
else:
    telemetry_server = None

sensor_poller = None
if sensor_http_endpoint:
    try:
        sensor_poller = SensorPoller(sensor_http_endpoint, sensor_poll_interval, update_environment_data)
        sensor_poller.start()
        print(f"[INFO] polling environment sensor at {sensor_http_endpoint} every {sensor_poll_interval}s")
    except Exception as exc:
        print(f"[WARN] failed to start sensor poller: {exc}")

# loop over the frames from the video stream
# 2D image points. If you change the image, you need to change vector
image_points = np.array([
    (359, 391),     # Nose tip 34
    (399, 561),     # Chin 9
    (337, 297),     # Left eye left corner 37
    (513, 301),     # Right eye right corne 46
    (345, 465),     # Left Mouth corner 49
    (453, 469)      # Right mouth corner 55
], dtype="double")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

EYE_AR_THRESH = 0.23
MOUTH_AR_THRESH = 0.75
EYE_AR_CONSEC_FRAMES = 5
COUNTER = 0

overlay_enabled = True
pose_enabled = False
draw_landmark_points = False
draw_landmark_ids = False

# Faster face detection: downscale + cache
DETECT_SCALE = 0.6  # slightly larger for more reliable detection
DETECT_EVERY_N_FRAMES = 3
cached_rects = []
frame_idx = 0
no_face_frames = 0

# Performance presets (1=speed, 2=balanced, 3=quality)
perf_mode = 'balanced'
def apply_perf_mode(mode):
    global DETECT_SCALE, DETECT_EVERY_N_FRAMES, pose_enabled, draw_landmark_ids
    if mode == 'speed':
        DETECT_SCALE = 0.5
        DETECT_EVERY_N_FRAMES = 6
        pose_enabled = False
        draw_landmark_ids = False
    elif mode == 'quality':
        DETECT_SCALE = 0.8
        DETECT_EVERY_N_FRAMES = 2
        pose_enabled = True
        draw_landmark_ids = True
    else:
        DETECT_SCALE = 0.6
        DETECT_EVERY_N_FRAMES = 3
        pose_enabled = True
        draw_landmark_ids = False
apply_perf_mode(perf_mode)

# Enable OpenCV optimizations and limit threads for stable CPU usage
try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(2)
except Exception:
    pass

# Rolling windows and state for scoring/recording
closed_flags: deque = deque()  # (ts, is_closed)
yawn_events: deque = deque()   # ts of yawn start
yawn_active = False
last_yawn_time = 0.0

frame_buffer: deque = deque()  # (ts, frame)
time_stamps: deque = deque()   # for FPS estimate

alert_active = False
incident_written_for_current_alert = False
HEAD_TILT_EVERY_N_FRAMES = 3
last_head_deg_value = 0.0
last_pose_points = None  # (start_point, end_point, end_point_alt)
debt_alarm_state = "idle"
debt_alarm_last_ts = 0.0

def save_incident_async(snapshot_frame, frames_with_ts, fps, incident_ts):
    def _worker():
        incident_dir = os.path.join(INCIDENT_DIR, f'incident_{incident_ts}')
        os.makedirs(incident_dir, exist_ok=True)
        snapshot_path = os.path.join(incident_dir, 'snapshot.jpg')
        try:
            cv2.imwrite(snapshot_path, snapshot_frame)
        except Exception:
            pass
        clip_path = os.path.join(incident_dir, 'clip.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(clip_path, fourcc, fps, (snapshot_frame.shape[1], snapshot_frame.shape[0]))
        try:
            for _, f in frames_with_ts:
                writer.write(f)
        finally:
            writer.release()
    threading.Thread(target=_worker, daemon=True).start()

# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)

while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    if frame is None:
        if getattr(vs, 'mode', None) == 'video' and not vs.loop_video:
            print("[INFO] video file reached the end; exiting.")
            break
        time.sleep(0.02)
        continue
    # If camera did not honor requested size, resize to target
    if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
        frame = imutils.resize(frame, width=frame_width, height=frame_height)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape
    env_snapshot = get_environment_snapshot()
    ambient_light = env_snapshot.get('ambient_light')
    cabin_temp = env_snapshot.get('cabin_temp')
    eye_frames_required = EYE_AR_CONSEC_FRAMES
    env_eye_offset = 0.0
    env_mar_offset = 0.0
    if ambient_light is not None:
        if ambient_light < 30:
            env_eye_offset -= 0.01
            eye_frames_required += 2
        elif ambient_light > 200:
            eye_frames_required = max(3, eye_frames_required - 1)
    if cabin_temp is not None:
        if cabin_temp > 32:
            env_mar_offset -= 0.02
        elif cabin_temp < 15:
            env_mar_offset += 0.01
    latest_ear_median = None
    latest_mar_median = None
    recalibration_flag = False

    now_ts = time.time()
    if prev_ts is None:
        prev_ts = now_ts
    dt = max(1e-3, now_ts - prev_ts)
    prev_ts = now_ts
    # Buffer frames for incident clips and FPS estimation
    frame_buffer.append((now_ts, frame.copy()))
    # trim buffer to last VIDEO_CLIP_SECS seconds
    while frame_buffer and (now_ts - frame_buffer[0][0]) > VIDEO_CLIP_SECS:
        frame_buffer.popleft()
    time_stamps.append(now_ts)
    while time_stamps and (now_ts - time_stamps[0]) > 5.0:
        time_stamps.popleft()

    # detect faces in the grayscale frame (downscaled, cached)
    if frame_idx % DETECT_EVERY_N_FRAMES == 0 or not cached_rects:
        scaled = []
        # Try MediaPipe first (RGB expected)
        if mp_fd is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mp_fd.process(rgb)
            if res and res.detections:
                h, w = frame.shape[:2]
                for det in res.detections:
                    box = det.location_data.relative_bounding_box
                    x = int(box.xmin * w)
                    y = int(box.ymin * h)
                    ww = int(box.width * w)
                    hh = int(box.height * h)
                    scaled.append(dlib.rectangle(max(0,x), max(0,y), min(w-1,x+ww), min(h-1,y+hh)))
        # Fallback to dlib HOG if none
        if not scaled:
            small = cv2.resize(gray, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
            rects_small = detector(small, 0)
            inv = 1.0 / DETECT_SCALE
            for r in rects_small:
                scaled.append(dlib.rectangle(
                    int(r.left() * inv), int(r.top() * inv),
                    int(r.right() * inv), int(r.bottom() * inv)))
        cached_rects = scaled
    # fallback: if no faces for a while, try full-res detect once
    if not cached_rects:
        no_face_frames += 1
        if no_face_frames >= DETECT_EVERY_N_FRAMES * 3:
            rects_full = detector(gray, 0)
            if rects_full:
                cached_rects = rects_full
            no_face_frames = 0
    else:
        no_face_frames = 0
    rects = cached_rects
    has_face_this_frame = bool(rects)
    if not rects:
        COUNTER = 0
        yawn_active = False
    frame_idx += 1
    update_telemetry(
        faces_detected=len(rects),
        alert_active=bool(alert_active),
        input_source=vs.status_label
    )

    # draw number of faces (always show for clarity)
    if overlay_enabled:
        text = "{} face(s) found".format(len(rects))
        cv2.putText(frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Process only the largest face for speed
    if len(rects) > 1:
        rects = [max(rects, key=lambda r: (r.right()-r.left()) * (r.bottom()-r.top()))]

    # loop over the face detections
    for idx, rect in enumerate(rects):
        # compute the bounding box of the face and draw it on the
        # frame
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        if overlay_enabled:
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        # Determine thresholds, applying calibration baselines if available
        ear_thresh = EYE_AR_THRESH
        mar_thresh = MOUTH_AR_THRESH
        if baseline_ear is not None:
            ear_thresh = clamp(baseline_ear - 0.04, 0.12, EYE_AR_THRESH)
        if baseline_mar is not None:
            adaptive_mar = max(baseline_mar + 0.08, baseline_mar * 1.15)
            mar_thresh = clamp(adaptive_mar, 0.55, 0.95)

        ear_thresh = clamp(ear_thresh + env_eye_offset, 0.12, EYE_AR_THRESH)
        mar_thresh = clamp(mar_thresh + env_mar_offset, 0.55, 0.95)

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        if overlay_enabled and draw_landmark_points:
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < ear_thresh:
            COUNTER += 1
            # if the eyes were closed for a sufficient number of times
            # then show the warning
            if COUNTER >= eye_frames_required:
                if overlay_enabled:
                    cv2.putText(frame, "Eyes Closed!", (500, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # alarm + log with cooldown
                now = time.time()
                if now - last_alarm_time["eyes"] > ALARM_COOLDOWN_SECS:
                    play_alarm_async()
                    last_alarm_time["eyes"] = now
                    # head tilt may be computed later; we log with None here and update below if available
                    log_event('eyes_closed', ear, None, None, len(rects),
                              ambient_light=ambient_light, cabin_temp=cabin_temp)
            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
        else:
            COUNTER = 0

        mouth = shape[mStart:mEnd]

        mouthMAR = mouth_aspect_ratio(mouth)
        mar = mouthMAR
        _append_history(ear_history, now_ts, ear)
        _append_history(mar_history, now_ts, mar)
        _update_session_stats(session_ear_stats, ear)
        _update_session_stats(session_mar_stats, mar)
        ear_recent_median = _running_median(ear_history)
        mar_recent_median = _running_median(mar_history)
        # compute the convex hull for the mouth, then
        # visualize the mouth
        if overlay_enabled and draw_landmark_points:
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        if overlay_enabled:
            cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw text if mouth is open
        if mar > mar_thresh:
            if overlay_enabled:
                cv2.putText(frame, "Yawning!", (800, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            now = time.time()
            if now - last_alarm_time["yawn"] > ALARM_COOLDOWN_SECS:
                play_alarm_async()
                last_alarm_time["yawn"] = now
                log_event('yawn', None, mar, None, len(rects),
                          ambient_light=ambient_light, cabin_temp=cabin_temp)

        # ---- Metrics for drowsiness score on the first detected face only ----
        if idx == 0:
            latest_ear_median = ear_recent_median
            latest_mar_median = mar_recent_median
            recalibration_needed = False
            drift_msgs = []
            if baseline_ear is not None and ear_recent_median is not None and baseline_ear > 0:
                ear_drift_ratio = (ear_recent_median - baseline_ear) / baseline_ear
                if abs(ear_drift_ratio) >= 0.12:
                    recalibration_needed = True
                    drift_msgs.append(f"EAR {ear_drift_ratio:+.0%}")
            if baseline_mar is not None and mar_recent_median is not None and baseline_mar > 0:
                mar_drift_ratio = (mar_recent_median - baseline_mar) / baseline_mar
                if abs(mar_drift_ratio) >= 0.12:
                    recalibration_needed = True
                    drift_msgs.append(f"MAR {mar_drift_ratio:+.0%}")
            if recalibration_needed:
                drift_notice_until = now_ts + 10
                drift_event_count += 1
                _write_drift_log("Calibration drift detected: " + ", ".join(drift_msgs) + ". Press 'c' to recalibrate.")
            recalibration_flag = recalibration_needed
            # PERCLOS window update
            closed_now = ear < ear_thresh
            closed_flags.append((now_ts, 1 if closed_now else 0))
            while closed_flags and (now_ts - closed_flags[0][0]) > PERCLOS_WINDOW_SECS:
                closed_flags.popleft()
            if closed_flags:
                perclos = sum(v for _, v in closed_flags) / float(len(closed_flags))
            else:
                perclos = 0.0

            # Yawns per minute with cooldown to avoid counting continuous frames
            if mar > mar_thresh and (not yawn_active) and (now_ts - last_yawn_time > YAWN_COOLDOWN_SECS):
                yawn_active = True
                last_yawn_time = now_ts
                yawn_events.append(now_ts)
            elif mar <= mar_thresh * 0.9:
                yawn_active = False
            while yawn_events and (now_ts - yawn_events[0]) > YAWN_RATE_WINDOW_SECS:
                yawn_events.popleft()
            yawns_per_min = float(len(yawn_events))  # window is 60s

            # Head tilt degree compute ALWAYS (even if pose overlay is off)
            head_deg_value = last_head_deg_value
            if frame_idx % HEAD_TILT_EVERY_N_FRAMES == 0:
                try:
                    (head_tilt_degree, start_point, end_point, 
                        end_point_alt) = getHeadTiltAndCoords(size, image_points, size[0])
                    head_deg_value = float(head_tilt_degree[0]) if head_tilt_degree is not None else last_head_deg_value
                    last_head_deg_value = head_deg_value
                    last_pose_points = (start_point, end_point, end_point_alt)
                except Exception:
                    pass

            # Score normalization (dynamic thresholds)
            perclos_component = perclos  # already 0..1
            yawn_component = min(1.0, yawns_per_min / YAWNS_PER_MIN_ALERT)
            tilt_component = min(1.0, head_deg_value / HEAD_TILT_ALERT_DEG)
            score = (
                SCORE_WEIGHTS["perclos"] * perclos_component +
                SCORE_WEIGHTS["yawn"] * yawn_component +
                SCORE_WEIGHTS["tilt"] * tilt_component
            )
            session_score_stats["sum"] += score
            session_score_stats["count"] += 1

            risk_trigger = False

            if COUNTER >= eye_frames_required:
                score = max(score, EYE_ALERT_SCORE)
                risk_trigger = True
            if mar > mar_thresh:
                score = max(score, YAWN_ALERT_SCORE)
                risk_trigger = True

            # Show score overlay if overlay enabled
            if overlay_enabled:
                cv2.putText(frame, "Score: {:.2f}  PERCLOS: {:.2f}  YPM: {}  Tilt: {:.0f}".format(
                    score, perclos, int(yawns_per_min), head_deg_value),
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                if baseline_ear or baseline_mar:
                    cv2.putText(frame, "Calibrated", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

            # Drowsiness debt meter accumulate/decay (only on eyes/yawn triggers)
            if risk_trigger:
                debt_score = min(100.0, debt_score + DEBT_ACCUM_FACTOR * dt)
            else:
                decay = DEBT_DECAY_PER_SEC_LOW if score < ALERT_SCORE_THRESHOLD * 0.5 else DEBT_DECAY_PER_SEC_HIGH
                debt_score = max(0.0, debt_score - decay * dt)

            now_alarm = time.time()
            if debt_score >= DEBT_CRITICAL_THRESHOLD:
                if (debt_alarm_state != "critical" or
                        (now_alarm - debt_alarm_last_ts) >= DEBT_CRITICAL_REPEAT_SECS):
                    play_alarm_async(duration_ms=10000, repeat=20)
                    debt_alarm_last_ts = now_alarm
                debt_alarm_state = "critical"
            elif debt_score >= DEBT_WARN_THRESHOLD:
                if debt_alarm_state not in ("warn", "critical") or (
                        debt_alarm_state == "warn" and
                        (now_alarm - debt_alarm_last_ts) >= DEBT_WARN_REPEAT_SECS):
                    play_alarm_async(duration_ms=600, repeat=3)
                    debt_alarm_last_ts = now_alarm
                if debt_alarm_state != "critical":
                    debt_alarm_state = "warn"
            elif debt_score < DEBT_WARN_THRESHOLD * 0.8:
                debt_alarm_state = "idle"

            if overlay_enabled:
                # draw debt bar at top left
                bar_w = int(200 * (debt_score / 100.0))
                cv2.rectangle(frame, (10, 70), (210, 90), (60, 60, 60), 1)
                cv2.rectangle(frame, (10, 70), (10 + bar_w, 90), (0, 140, 255), -1)
                cv2.putText(frame, 'Debt {:.0f}/100'.format(debt_score), (220, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

            update_telemetry(
                score=float(score),
                perclos=float(perclos),
                yawns_per_min=float(yawns_per_min),
                head_tilt=float(head_deg_value),
                eyes_closed=bool(COUNTER >= eye_frames_required),
                yawning=bool(mar > mar_thresh),
                risk_trigger=bool(risk_trigger),
                debt_score=float(debt_score),
                ear_median=None if latest_ear_median is None else float(latest_ear_median),
                mar_median=None if latest_mar_median is None else float(latest_mar_median),
                recalibration=recalibration_flag,
                ambient_light=None if ambient_light is None else float(ambient_light),
                cabin_temp=None if cabin_temp is None else float(cabin_temp),
                eye_frames_required=eye_frames_required
            )

            if now_ts - last_metrics_log_ts >= 2.0:
                log_event(
                    'metrics',
                    ear=ear,
                    mar=mar,
                    head_tilt_deg=head_deg_value,
                    faces=len(rects),
                    score=score,
                    perclos=perclos,
                    yawns_per_min=yawns_per_min,
                    debt_score=debt_score,
                    ambient_light=ambient_light,
                    cabin_temp=cabin_temp
                )
                last_metrics_log_ts = now_ts

            # Thresholded alert banner and alarm (only eyes/yawn triggers)
            if risk_trigger:
                alert_active = True
                # Big banner regardless of overlay setting
                cv2.putText(frame, "DROWSINESS ALERT!", (int(frame.shape[1]*0.15), 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                if time.time() - last_alarm_time.get("score", 0.0) > ALARM_COOLDOWN_SECS:
                    play_alarm_async()
                    last_alarm_time["score"] = time.time()
                # Auto-save incident media only once per continuous alert
                if not incident_written_for_current_alert:
                    incident_ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    # estimate fps
                    fps = 0.0
                    if len(time_stamps) >= 2:
                        duration = time_stamps[-1] - time_stamps[0]
                        if duration > 0:
                            fps = (len(time_stamps) - 1) / duration
                    if fps <= 0:
                        fps = 20.0
                    save_incident_async(frame.copy(), list(frame_buffer), fps, incident_ts)
                incident_written_for_current_alert = True
            else:
                # Alert cleared
                alert_active = False
                incident_written_for_current_alert = False


        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw each of them
        for (i, (x, y)) in enumerate(shape):
            if i == 33:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[0] = np.array([x, y], dtype='double')
                # write on frame in Green
                if overlay_enabled and draw_landmark_points:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    if draw_landmark_ids:
                        cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 8:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[1] = np.array([x, y], dtype='double')
                # write on frame in Green
                if overlay_enabled and draw_landmark_points:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    if draw_landmark_ids:
                        cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 36:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[2] = np.array([x, y], dtype='double')
                # write on frame in Green
                if overlay_enabled and draw_landmark_points:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    if draw_landmark_ids:
                        cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 45:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[3] = np.array([x, y], dtype='double')
                # write on frame in Green
                if overlay_enabled and draw_landmark_points:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    if draw_landmark_ids:
                        cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 48:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[4] = np.array([x, y], dtype='double')
                # write on frame in Green
                if overlay_enabled and draw_landmark_points:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    if draw_landmark_ids:
                        cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 54:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[5] = np.array([x, y], dtype='double')
                # write on frame in Green
                if overlay_enabled and draw_landmark_points:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    if draw_landmark_ids:
                        cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            else:
                # everything to all other landmarks
                # write on frame in Red
                if overlay_enabled and draw_landmark_points:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                    if draw_landmark_ids:
                        cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        #Draw the determinant image points onto the person's face
        if overlay_enabled and draw_landmark_points:
            for p in image_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        head_tilt_degree = None
        if overlay_enabled and pose_enabled:
            try:
                # draw last cached pose lines to avoid heavy per-frame solve
                if last_pose_points and draw_landmark_points:
                    sp, ep, ep_alt = last_pose_points
                    cv2.line(frame, sp, ep, (255, 0, 0), 2)
                    cv2.line(frame, sp, ep_alt, (0, 0, 255), 2)
                if draw_landmark_points:
                    cv2.putText(frame, 'Head Tilt Degree: ' + str(last_head_deg_value), (170, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            except Exception:
                pass
            # If we just logged an event without degree, update it now
            # (best-effort: we log fresh events as they happen again)

        # extract the mouth coordinates, then use the
        # coordinates to compute the mouth aspect ratio
    # show the frameq
    if calibrating and overlay_enabled:
        cv2.putText(frame, 'Calibrating... keep neutral face ({}s)'.format(int(max(0, CALIB_SECONDS - (now_ts - calib_start_ts)))),
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    if overlay_enabled:
        status = 'Overlay: ON | Pose: ' + ('ON' if pose_enabled else 'OFF')
        status += ' | Points: ' + ('ON' if draw_landmark_points else 'OFF')
        status += ' | IDs: ' + ('ON' if draw_landmark_ids else 'OFF')
        status += ' | Calib: ' + ('ON' if calibrating else ('YES' if (baseline_ear or baseline_mar) else 'NO'))
        status += ' | Mode: ' + perf_mode.upper()
        status += ' | Input: ' + vs.status_label
        if ambient_light is not None or cabin_temp is not None:
            light_txt = '---' if ambient_light is None else f"{ambient_light:.0f}lx"
            temp_txt = '---' if cabin_temp is None else f"{cabin_temp:.1f}C"
            status += f' | Env: {light_txt}/{temp_txt}'
        status += '  (1/2/3 modes | O/P/K/J overlay | C calib | [ ] cams | V video)'
        cv2.putText(frame, status, (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        # minimal hint when overlay is off
        cv2.putText(frame, 'Overlay: OFF (press O to toggle)', (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if overlay_enabled and now_ts < drift_notice_until:
        cv2.putText(frame, 'Calibration drift detected - press C to refresh baselines', (10, frame.shape[0]-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    if key == ord("o"):
        overlay_enabled = not overlay_enabled
    if key == ord("p"):
        pose_enabled = not pose_enabled
    if key == ord("k"):
        draw_landmark_points = not draw_landmark_points
    if key == ord("j"):
        draw_landmark_ids = not draw_landmark_ids
    if key == ord('1'):
        perf_mode = 'speed'
        apply_perf_mode(perf_mode)
    if key == ord('2'):
        perf_mode = 'balanced'
        apply_perf_mode(perf_mode)
    if key == ord('3'):
        perf_mode = 'quality'
        apply_perf_mode(perf_mode)
    if key == ord("c"):
        if not calibrating:
            calibrating = True
            calib_start_ts = time.time()
            calib_ear_samples.clear()
            calib_mar_samples.clear()
        else:
            calibrating = False
    if key == ord('v'):
        vs.toggle_video()
    if key == ord('['):
        vs.cycle_camera(-1)
    if key == ord(']'):
        vs.cycle_camera(1)

    # Handle calibration sampling
    if calibrating:
        if has_face_this_frame:
            if ear is not None and ear >= CALIB_MIN_EAR:
                calib_ear_samples.append(ear)
            if mar is not None and mar <= CALIB_MAX_MAR:
                calib_mar_samples.append(mar)
        if now_ts - calib_start_ts >= CALIB_SECONDS and (calib_ear_samples or calib_mar_samples):
            # compute robust baselines (medians)
            new_baseline_ear = baseline_ear
            new_baseline_mar = baseline_mar
            try:
                if calib_ear_samples:
                    new_baseline_ear = float(stats.median(calib_ear_samples))
                if calib_mar_samples:
                    new_baseline_mar = float(stats.median(calib_mar_samples))
            except Exception:
                if calib_ear_samples:
                    new_baseline_ear = float(sum(calib_ear_samples) / len(calib_ear_samples))
                if calib_mar_samples:
                    new_baseline_mar = float(sum(calib_mar_samples) / len(calib_mar_samples))
            persist_calibration(new_baseline_ear, new_baseline_mar)
            calibrating = False

# print(image_points)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
write_session_summary()
try:
    log_file.close()
except Exception:
    pass
