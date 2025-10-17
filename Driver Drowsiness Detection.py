#!/usr/bin/env python
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import math
import os
import csv
from datetime import datetime
import threading
import json
import cv2
import numpy as np
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords
from collections import deque
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

# initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up
print("[INFO] initializing camera...")
def start_video_stream():
    for index in [0, 1, 2]:
        try:
            vs_try = VideoStream(src=index).start()
            time.sleep(1.0)
            test_frame = vs_try.read()
            if test_frame is not None:
                print("[INFO] camera opened at index {}".format(index))
                # try to set native capture size for performance
                try:
                    vs_try.stream.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
                    vs_try.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
                except Exception:
                    pass
                return vs_try
            vs_try.stop()
        except Exception:
            try:
                vs_try.stop()
            except Exception:
                pass
    raise RuntimeError("No working camera found (tried indices 0-2).")

vs = start_video_stream()
time.sleep(2.0)

# 960x540 for higher clarity (adjust if needed)
frame_width = 960
frame_height = 540

# Alarm, scoring and logging configuration
ALARM_COOLDOWN_SECS = 3.0
last_alarm_time = {"eyes": 0.0, "yawn": 0.0}

# Drowsiness score configuration
PERCLOS_WINDOW_SECS = 60.0
YAWN_COOLDOWN_SECS = 2.0
YAWN_RATE_WINDOW_SECS = 60.0
HEAD_TILT_ALERT_DEG = 15.0
YAWNS_PER_MIN_ALERT = 3.0
SCORE_WEIGHTS = {"perclos": 0.5, "yawn": 0.3, "tilt": 0.2}
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
try:
    if os.path.exists(calib_path):
        with open(calib_path, 'r', encoding='utf-8') as f:
            _c = json.load(f)
            baseline_ear = _c.get('baseline_ear')
            baseline_mar = _c.get('baseline_mar')
except Exception:
    pass

# Drowsiness Debt Meter (accumulate fatigue over time with decay)
debt_score = 0.0  # 0..100
DEBT_DECAY_PER_SEC_LOW = 6.0
DEBT_DECAY_PER_SEC_HIGH = 2.0
DEBT_ACCUM_FACTOR = 30.0  # points/sec when risk score == 1.0
prev_ts = None

def play_alarm_async():
    try:
        import winsound
        def _beep():
            try:
                for _ in range(2):
                    winsound.Beep(1500, 250)
                    time.sleep(0.05)
            except Exception:
                pass
    except Exception:
        def _beep():
            try:
                print("\a", end="")
            except Exception:
                pass
    threading.Thread(target=_beep, daemon=True).start()

script_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(script_dir, 'events_log.csv')
log_file = open(log_path, 'a', newline='', encoding='utf-8')
csv_writer = csv.writer(log_file)
if log_file.tell() == 0:
    csv_writer.writerow(['timestamp', 'event', 'ear', 'mar', 'head_tilt_deg', 'faces'])

def log_event(event_type, ear, mar, head_tilt_deg, faces):
    csv_writer.writerow([
        datetime.now().isoformat(timespec='seconds'),
        event_type,
        None if ear is None else float(ear),
        None if mar is None else float(mar),
        None if head_tilt_deg is None else float(head_tilt_deg),
        int(faces)
    ])
    try:
        log_file.flush()
    except Exception:
        pass

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
    else:
        DETECT_SCALE = 0.6
        DETECT_EVERY_N_FRAMES = 3
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
        continue
    # If camera did not honor requested size, resize to target
    if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
        frame = imutils.resize(frame, width=frame_width, height=frame_height)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

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
    frame_idx += 1

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

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        if overlay_enabled and draw_landmark_points:
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            # if the eyes were closed for a sufficient number of times
            # then show the warning
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if overlay_enabled:
                    cv2.putText(frame, "Eyes Closed!", (500, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # alarm + log with cooldown
                now = time.time()
                if now - last_alarm_time["eyes"] > ALARM_COOLDOWN_SECS:
                    play_alarm_async()
                    last_alarm_time["eyes"] = now
                    # head tilt may be computed later; we log with None here and update below if available
                    log_event('eyes_closed', ear, None, None, len(rects))
            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
        else:
            COUNTER = 0

        mouth = shape[mStart:mEnd]

        mouthMAR = mouth_aspect_ratio(mouth)
        mar = mouthMAR
        # compute the convex hull for the mouth, then
        # visualize the mouth
        if overlay_enabled and draw_landmark_points:
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        if overlay_enabled:
            cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw text if mouth is open
        if mar > MOUTH_AR_THRESH:
            if overlay_enabled:
                cv2.putText(frame, "Yawning!", (800, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            now = time.time()
            if now - last_alarm_time["yawn"] > ALARM_COOLDOWN_SECS:
                play_alarm_async()
                last_alarm_time["yawn"] = now
                log_event('yawn', None, mar, None, len(rects))

        # ---- Metrics for drowsiness score on the first detected face only ----
        if idx == 0:
            # PERCLOS window update
            closed_now = ear < EYE_AR_THRESH
            closed_flags.append((now_ts, 1 if closed_now else 0))
            while closed_flags and (now_ts - closed_flags[0][0]) > PERCLOS_WINDOW_SECS:
                closed_flags.popleft()
            if closed_flags:
                perclos = sum(v for _, v in closed_flags) / float(len(closed_flags))
            else:
                perclos = 0.0

            # Yawns per minute with cooldown to avoid counting continuous frames
            if mar > MOUTH_AR_THRESH and (not yawn_active) and (now_ts - last_yawn_time > YAWN_COOLDOWN_SECS):
                yawn_active = True
                last_yawn_time = now_ts
                yawn_events.append(now_ts)
            elif mar <= MOUTH_AR_THRESH * 0.9:
                yawn_active = False
            while yawn_events and (now_ts - yawn_events[0]) > YAWN_RATE_WINDOW_SECS:
                yawn_events.popleft()
            yawns_per_min = float(len(yawn_events))  # window is 60s

            # Head tilt degree compute ALWAYS (even if pose overlay is off)
            head_deg_value = last_head_deg_value
            if frame_idx % HEAD_TILT_EVERY_N_FRAMES == 0:
                try:
                    (head_tilt_degree, start_point, end_point, 
                        end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)
                    head_deg_value = float(head_tilt_degree[0]) if head_tilt_degree is not None else last_head_deg_value
                    last_head_deg_value = head_deg_value
                    last_pose_points = (start_point, end_point, end_point_alt)
                except Exception:
                    pass

            # Apply personalized thresholds if calibrated
            ear_thresh = EYE_AR_THRESH
            mar_thresh = MOUTH_AR_THRESH
            if baseline_ear:
                ear_thresh = max(0.12, baseline_ear * 0.8)
            if baseline_mar:
                mar_thresh = min(1.2, baseline_mar * 1.35)

            # Score normalization (dynamic thresholds)
            perclos_component = perclos  # already 0..1
            yawn_component = min(1.0, yawns_per_min / YAWNS_PER_MIN_ALERT)
            tilt_component = min(1.0, head_deg_value / HEAD_TILT_ALERT_DEG)
            score = (
                SCORE_WEIGHTS["perclos"] * perclos_component +
                SCORE_WEIGHTS["yawn"] * yawn_component +
                SCORE_WEIGHTS["tilt"] * tilt_component
            )

            # Show score overlay if overlay enabled
            if overlay_enabled:
                cv2.putText(frame, "Score: {:.2f}  PERCLOS: {:.2f}  YPM: {}  Tilt: {:.0f}".format(
                    score, perclos, int(yawns_per_min), head_deg_value),
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                if baseline_ear or baseline_mar:
                    cv2.putText(frame, "Calibrated", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

            # Drowsiness debt meter accumulate/decay
            if score >= ALERT_SCORE_THRESHOLD:
                decay = DEBT_DECAY_PER_SEC_HIGH
                debt_score = min(100.0, debt_score + DEBT_ACCUM_FACTOR * dt)
            else:
                decay = DEBT_DECAY_PER_SEC_LOW if score < ALERT_SCORE_THRESHOLD * 0.5 else DEBT_DECAY_PER_SEC_HIGH
                debt_score = max(0.0, debt_score - decay * dt)

            if overlay_enabled:
                # draw debt bar at top left
                bar_w = int(200 * (debt_score / 100.0))
                cv2.rectangle(frame, (10, 70), (210, 90), (60, 60, 60), 1)
                cv2.rectangle(frame, (10, 70), (10 + bar_w, 90), (0, 140, 255), -1)
                cv2.putText(frame, 'Debt {:.0f}/100'.format(debt_score), (220, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

            # Thresholded alert banner and alarm
            if score >= ALERT_SCORE_THRESHOLD:
                alert_active = True
                # Big banner regardless of overlay setting
                cv2.putText(frame, "DROWSINESS ALERT!", (int(frame.shape[1]*0.15), 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                if time.time() - last_alarm_time.get("score", 0.0) > ALARM_COOLDOWN_SECS:
                    play_alarm_async()
                    last_alarm_time["score"] = time.time()
                # Auto-save incident media only once per continuous alert
                if not incident_written_for_current_alert:
                    incident_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
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
        status += ' | Mode: ' + perf_mode.upper() + '  (1/2/3 to set)  (O/P/K/J/C to toggle)'
        cv2.putText(frame, status, (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        # minimal hint when overlay is off
        cv2.putText(frame, 'Overlay: OFF (press O to toggle)', (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
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

    # Handle calibration sampling
    if calibrating:
        calib_ear_samples.append(ear if 'ear' in locals() else 0.0)
        calib_mar_samples.append(mar if 'mar' in locals() else 0.0)
        if now_ts - calib_start_ts >= CALIB_SECONDS:
            # compute robust baselines (medians)
            try:
                import statistics as stats
                baseline_ear = float(stats.median(calib_ear_samples))
                baseline_mar = float(stats.median(calib_mar_samples))
            except Exception:
                if calib_ear_samples:
                    baseline_ear = float(sum(calib_ear_samples) / len(calib_ear_samples))
                if calib_mar_samples:
                    baseline_mar = float(sum(calib_mar_samples) / len(calib_mar_samples))
            # persist to file
            try:
                with open(calib_path, 'w', encoding='utf-8') as f:
                    json.dump({'baseline_ear': baseline_ear, 'baseline_mar': baseline_mar}, f)
            except Exception:
                pass
            calibrating = False

# print(image_points)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
try:
    log_file.close()
except Exception:
    pass
