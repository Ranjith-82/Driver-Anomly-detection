"""
Cleaned multiprocessing pipeline for real-time driver monitoring.

Notes:
- Recognition is requested only once per track (cached).
- Yawn detection counts per-track yawns and sends a Telegram alert after threshold.
- Replace Config.BOT_TOKEN and Config.CHAT_ID with your values below (already set).
"""

import time
import os
import cv2
import pickle
import logging
import queue
import numpy as np
import threading
import requests
from datetime import datetime
from multiprocessing import Process, Queue, Manager, Event
from pathlib import Path

try:
    import face_recognition
except Exception:
    face_recognition = None
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
except Exception:
    mp = None
    mp_tasks = None
    vision = None

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('pipeline')


class Config:
    CAMERA_ID = 0
    CAP_WIDTH = 1280
    CAP_HEIGHT = 720

    DETECT_SCALE = 0.25
    RECOG_SCALE = 0.5
    FACE_DETECT_SKIP = 5
    RECOG_SKIP = 10
    YAWN_SKIP = 6
    YAWN_THRESHOLD = 3

    CAPTURE_QUEUE_SIZE = 6
    DETECT_QUEUE_SIZE = 4
    DETECT_RESULT_QUEUE_SIZE = 6
    RECOG_QUEUE_SIZE = 4
    RECOG_RESULT_QUEUE_SIZE = 8

    FACE_DETECTION_MODEL = 'hog'
    FACE_RECOGNITION_TOLERANCE = 0.5
    FACEMESH_MODEL_PATH = 'face_landmarker.task'
    FACEMESH_CLASSIFIER_PATH = 'facemesh_classifier.pkl'
    FACEMESH_LABEL_ENCODER_PATH = 'label_encoder.pkl'

    # Keep Telegram credentials outside source control.
    BOT_TOKEN = os.getenv('DRIVER_MONITOR_BOT_TOKEN', '')
    CHAT_ID = os.getenv('DRIVER_MONITOR_CHAT_ID', '')

    LOG_INTERVAL = 1.0


def create_tracker():
    if hasattr(cv2, 'TrackerKCF_create'):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
        return cv2.legacy.TrackerKCF_create()
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError('No suitable OpenCV tracker available')


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    denom = float(boxAArea + boxBArea - interArea)
    if denom <= 0:
        return 0.0
    return interArea / denom


def detect_yawn(roi):
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        c = max(contours, key=cv2.contourArea)
        _, _, _, ch = cv2.boundingRect(c)
        if ch / float(h) > 0.28:
            return True
    except Exception:
        return False
    return False


def normalize_landmarks(coords):
    center = coords.mean(axis=0)
    scale = np.maximum(coords.max(axis=0) - coords.min(axis=0), 1e-6)
    return ((coords - center) / scale).flatten()


def create_face_mesh_extractor(model_path=None):
    if mp is not None and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
        return ('solutions', mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        ))

    if mp_tasks is None or vision is None or model_path is None:
        return None

    model_path = Path(model_path)
    if not model_path.exists():
        return None

    base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return ('tasks', vision.FaceLandmarker.create_from_options(options))


def extract_face_mesh_features(image, extractor):
    if extractor is None:
        return None
    mode, handle = extractor
    try:
        if mode == 'solutions':
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = handle.process(image_rgb)
            image_rgb.flags.writeable = True
            if not results.multi_face_landmarks:
                return None
            landmarks = results.multi_face_landmarks[0].landmark
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            results = handle.detect(mp_image)
            if not results.face_landmarks:
                return None
            landmarks = results.face_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z]
                          for lm in landmarks], dtype=np.float32)
        return normalize_landmarks(coords)
    except Exception:
        return None


def load_expression_classifier():
    if not os.path.exists(Config.FACEMESH_CLASSIFIER_PATH) or not os.path.exists(Config.FACEMESH_LABEL_ENCODER_PATH):
        logger.warning(
            'Expression classifier files not found. Using heuristic yawn detection only.')
        return None, None, None
    try:
        with open(Config.FACEMESH_CLASSIFIER_PATH, 'rb') as f:
            classifier = pickle.load(f)
        with open(Config.FACEMESH_LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        extractor = create_face_mesh_extractor(Config.FACEMESH_MODEL_PATH)
        if extractor is None:
            logger.warning(
                'Failed to initialize facemesh extractor. Using heuristic yawn detection only.')
            return None, None, None
        return classifier, label_encoder, extractor
    except Exception:
        logger.exception('Failed to load expression classifier')
        return None, None, None


def predict_expression(roi, classifier, label_encoder, extractor):
    if classifier is None or label_encoder is None or extractor is None:
        return None, 0.0
    features = extract_face_mesh_features(roi, extractor)
    if features is None:
        return None, 0.0
    try:
        prediction = classifier.predict([features])[0]
        confidence = max(classifier.predict_proba([features])[0])
        label = label_encoder.inverse_transform([prediction])[0]
        return label, float(confidence)
    except Exception:
        return None, 0.0


def load_driver_database():
    drivers = {}
    db_path = 'drivers_database.pkl'
    if not os.path.exists(db_path):
        logger.warning('Driver database not found: %s', db_path)
        return drivers
    try:
        with open(db_path, 'rb') as f:
            drivers = pickle.load(f)
    except Exception:
        logger.exception('Failed to load driver database')
        drivers = {}
    return drivers


def send_telegram_alert(driver_details, reason, frame):
    try:
        if not driver_details:
            return False
        token = Config.BOT_TOKEN
        chat_id = Config.CHAT_ID
        if not token or not chat_id:
            logger.error(
                'Telegram credentials are missing. Set DRIVER_MONITOR_BOT_TOKEN '
                'and DRIVER_MONITOR_CHAT_ID.')
            return False

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        name = driver_details.get(
            'full_name', driver_details.get('name', 'N/A'))
        phone = driver_details.get('phone', 'N/A')
        dob = driver_details.get('dob', 'N/A')
        blood_group = driver_details.get('blood_group', 'N/A')
        address = driver_details.get(
            'address', driver_details.get('city', 'N/A'))

        message = (
            f'⚠️ DRIVER FATIGUE ALERT\n\n'
            f'Name: {name}\n'
            f'Phone: {phone}\n'
            f'DOB: {dob}\n'
            f'Blood Group: {blood_group}\n'
            f'Address: {address}\n'
            f'Reason: {reason}\n'
            f'Time: {current_time}'
        )

        alert_frame = frame.copy()
        overlay = alert_frame.copy()
        cv2.rectangle(overlay, (0, 0),
                      (alert_frame.shape[1], 100), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, alert_frame, 0.7, 0, alert_frame)
        text = '⚠️ FATIGUE DETECTED ⚠️'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (alert_frame.shape[1] - text_width) // 2
        y = 60
        cv2.putText(alert_frame, text, (x, y), font,
                    font_scale, (0, 255, 255), thickness)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        screenshot_path = f'alert_{timestamp}.jpg'
        cv2.imwrite(screenshot_path, alert_frame)

        url_msg = f'https://api.telegram.org/bot{token}/sendMessage'
        params_msg = {'chat_id': chat_id, 'text': message}
        response_msg = requests.post(url_msg, params=params_msg, timeout=5)

        url_photo = f'https://api.telegram.org/bot{token}/sendPhoto'
        with open(screenshot_path, 'rb') as photo_file:
            files = {'photo': photo_file}
            params_photo = {'chat_id': chat_id}
            response_photo = requests.post(
                url_photo, params=params_photo, files=files, timeout=5)

        if response_msg.status_code != 200:
            logger.error('Telegram sendMessage failed: %s', response_msg.text)
        if response_photo.status_code != 200:
            logger.error('Telegram sendPhoto failed: %s', response_photo.text)

        if os.path.exists(screenshot_path):
            os.remove(screenshot_path)

        if response_msg.status_code == 200:
            if response_photo.status_code != 200:
                logger.warning(
                    'Telegram text delivered, but alert photo failed.')
            return True
        return False
    except Exception:
        logger.exception('Telegram error')
        if 'screenshot_path' in locals() and os.path.exists(screenshot_path):
            os.remove(screenshot_path)
        return False


class Stats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.capture_time = 0.0
        self.detect_time = 0.0
        self.recog_time = 0.0
        self.track_time = 0.0
        self.render_time = 0.0
        self.count = 0

    def update(self, capture=0, detect=0, recog=0, track=0, render=0):
        self.capture_time += capture
        self.detect_time += detect
        self.recog_time += recog
        self.track_time += track
        self.render_time += render
        self.count += 1

    def summary(self):
        if self.count == 0:
            return {}
        return {
            'capture_ms': (self.capture_time / self.count) * 1000,
            'detect_ms': (self.detect_time / self.count) * 1000,
            'recog_ms': (self.recog_time / self.count) * 1000,
            'track_ms': (self.track_time / self.count) * 1000,
            'render_ms': (self.render_time / self.count) * 1000,
        }


def detector_process(detect_q, detect_result_q, stop_event):
    logger.info('Detector process started')
    while not stop_event.is_set():
        try:
            item = detect_q.get(timeout=0.5)
        except Exception:
            continue
        try:
            frame_id, ts, frame_small, _ = item
        except Exception:
            try:
                frame_id, ts, frame_small = item
            except Exception:
                continue
        if frame_id % Config.FACE_DETECT_SKIP != 0:
            continue
        t0 = time.perf_counter()
        boxes = []
        try:
            rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            if face_recognition is not None:
                boxes = face_recognition.face_locations(
                    rgb, model=Config.FACE_DETECTION_MODEL)
        except Exception:
            logger.exception('Detector error')
            boxes = []
        detect_time = time.perf_counter() - t0
        try:
            detect_result_q.put_nowait((frame_id, ts, boxes, detect_time))
        except Exception:
            pass
    logger.info('Detector process stopping')


def recognizer_process(recog_q, recog_result_q, manager_dict, stop_event):
    logger.info('Recognizer process started')
    drivers_db = {}
    encodings = []
    driver_ids = []
    db_path = 'drivers_database.pkl'
    if os.path.exists(db_path):
        try:
            with open(db_path, 'rb') as f:
                drivers_db = pickle.load(f)
        except Exception:
            drivers_db = {}
    if drivers_db and face_recognition is not None:
        for driver_id, info in drivers_db.items():
            p = info.get('photo')
            if p and os.path.exists(p):
                try:
                    img = face_recognition.load_image_file(p)
                    locs = face_recognition.face_locations(
                        img, model=Config.FACE_DETECTION_MODEL)
                    if locs:
                        enc = face_recognition.face_encodings(img, locs)[0]
                        encodings.append(enc)
                        driver_ids.append(driver_id)
                except Exception:
                    continue

    while not stop_event.is_set():
        try:
            item = recog_q.get(timeout=0.5)
        except Exception:
            continue
        try:
            track_id, roi_small, ts = item
        except Exception:
            continue
        t0 = time.perf_counter()
        result = {'track_id': track_id, 'identity': None,
                  'confidence': 0.0, 'recog_ms': 0.0}
        try:
            if face_recognition is not None and roi_small is not None and encodings:
                rgb = cv2.cvtColor(roi_small, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(
                    rgb, model=Config.FACE_DETECTION_MODEL)
                if locs:
                    encs = face_recognition.face_encodings(rgb, locs)
                    if encs:
                        enc = encs[0]
                        distances = face_recognition.face_distance(
                            encodings, enc)
                        best = np.argmin(distances)
                        if distances[best] < Config.FACE_RECOGNITION_TOLERANCE:
                            result['identity'] = driver_ids[best]
                            result['confidence'] = float(1.0 - distances[best])
                            manager_dict[str(track_id)] = (
                                result['identity'], result['confidence'], time.time())
        except Exception:
            logger.exception('Recognizer error')
        result['recog_ms'] = (time.perf_counter() - t0) * 1000.0
        try:
            recog_result_q.put_nowait(result)
        except Exception:
            pass
    logger.info('Recognizer process stopping')


class CaptureThread(threading.Thread):
    def __init__(self, cap_q, detect_q, stop_event):
        super().__init__(daemon=True)
        self.cap_q = cap_q
        self.detect_q = detect_q
        self.stop_event = stop_event
        self.cap = cv2.VideoCapture(Config.CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAP_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAP_HEIGHT)

    def run(self):
        frame_id = 0
        while not self.stop_event.is_set():
            t0 = time.perf_counter()
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            ts = time.time()
            small = cv2.resize(
                frame, (0, 0), fx=Config.DETECT_SCALE, fy=Config.DETECT_SCALE)
            payload = (frame_id, ts, small, frame)
            try:
                self.cap_q.put_nowait(payload)
            except queue.Full:
                try:
                    _ = self.cap_q.get_nowait()
                    self.cap_q.put_nowait(payload)
                except Exception:
                    pass
            try:
                self.detect_q.put_nowait(payload)
            except queue.Full:
                try:
                    _ = self.detect_q.get_nowait()
                    self.detect_q.put_nowait(payload)
                except Exception:
                    pass
            frame_id += 1
            elapsed = time.perf_counter() - t0
            if elapsed < 0.001:
                time.sleep(0.001)
        self.cap.release()


def main():
    manager = Manager()
    identity_cache = manager.dict()
    stop_event = Event()

    capture_q = Queue(maxsize=Config.CAPTURE_QUEUE_SIZE)
    detect_q = Queue(maxsize=Config.DETECT_QUEUE_SIZE)
    detect_result_q = Queue(maxsize=Config.DETECT_RESULT_QUEUE_SIZE)
    recog_q = Queue(maxsize=Config.RECOG_QUEUE_SIZE)
    recog_result_q = Queue(maxsize=Config.RECOG_RESULT_QUEUE_SIZE)

    detector = Process(target=detector_process, args=(
        detect_q, detect_result_q, stop_event), daemon=True)
    recognizer = Process(target=recognizer_process, args=(
        recog_q, recog_result_q, identity_cache, stop_event), daemon=True)
    detector.start()
    recognizer.start()

    capture_thread = CaptureThread(capture_q, detect_q, stop_event)
    capture_thread.start()
    drivers_db = load_driver_database()
    classifier, label_encoder, facemesh_extractor = load_expression_classifier()

    if classifier is None:
        logger.info(
            'Running with heuristic yawn detection only. Train facemesh classifier for better accuracy.')

    trackers = {}
    tracks_meta = {}
    next_track_id = 0

    stats = Stats()
    last_log = time.time()
    frame_count = 0

    try:
        while True:
            try:
                frame_item = capture_q.get(timeout=1.0)
            except Exception:
                continue
            try:
                frame_id, ts, small_frame, full_frame = frame_item
            except Exception:
                continue
            frame_count += 1

            t_capture = 0.0
            t_detect = 0.0
            t_recog = 0.0
            t_track = 0.0
            t_render = 0.0

            # send for detection at interval
            if frame_id % Config.FACE_DETECT_SKIP == 0:
                try:
                    # detector reads from capture_q directly, so nothing to do here
                    pass
                except Exception:
                    pass

            # collect detection results
            try:
                while True:
                    det = detect_result_q.get_nowait()
                    _, det_ts, boxes, detect_time = det
                    t_detect += detect_time
                    for box in boxes:
                        top, right, bottom, left = box
                        scale = 1.0 / Config.DETECT_SCALE
                        x = int(left * scale)
                        y = int(top * scale)
                        w = int((right - left) * scale)
                        h = int((bottom - top) * scale)
                        bbox = (x, y, w, h)
                        # avoid duplicates
                        skip = False
                        for otid, ometa in tracks_meta.items():
                            ob = ometa.get('bbox')
                            if ob and iou(bbox, ob) > 0.35:
                                skip = True
                                ometa['bbox'] = bbox
                                ometa['last_seen'] = time.time()
                                break
                        if skip:
                            continue
                        # create tracker
                        try:
                            tracker = create_tracker()
                            tracker.init(full_frame, bbox)
                        except Exception:
                            continue
                        tid = next_track_id
                        next_track_id += 1
                        trackers[tid] = tracker
                        tracks_meta[tid] = {'bbox': bbox, 'last_seen': time.time(
                        ), 'missed': 0, 'identity': None, 'yawn_count': 0, 'alerted': False, 'frames': 0}
                        # request recognition if not cached
                        if identity_cache.get(str(tid)) is None:
                            x1, y1, w1, h1 = bbox
                            roi = full_frame[max(0, y1):min(
                                full_frame.shape[0], y1 + h1), max(0, x1):min(full_frame.shape[1], x1 + w1)]
                            if roi.size != 0:
                                small_roi = cv2.resize(
                                    roi, (0, 0), fx=Config.RECOG_SCALE, fy=Config.RECOG_SCALE)
                                try:
                                    recog_q.put_nowait(
                                        (tid, small_roi, time.time()))
                                except Exception:
                                    pass
            except Exception:
                pass

            # update trackers
            t_tr0 = time.perf_counter()
            to_delete = []
            for tid, tracker in list(trackers.items()):
                try:
                    ok, bbox = tracker.update(full_frame)
                except Exception:
                    ok = False
                    bbox = None
                if not ok or bbox is None:
                    tracks_meta[tid]['missed'] += 1
                    if tracks_meta[tid]['missed'] > 30:
                        to_delete.append(tid)
                    continue
                x, y, w, h = [int(v) for v in bbox]
                tracks_meta[tid]['bbox'] = (x, y, w, h)
                tracks_meta[tid]['last_seen'] = time.time()
                tracks_meta[tid]['missed'] = 0
                tracks_meta[tid]['frames'] = tracks_meta[tid].get(
                    'frames', 0) + 1
                known = tracks_meta[tid].get('identity')
                if (known is None) and (identity_cache.get(str(tid)) is None):
                    if tracks_meta[tid]['frames'] % Config.RECOG_SKIP == 0:
                        roi = full_frame[max(0, y):min(
                            full_frame.shape[0], y + h), max(0, x):min(full_frame.shape[1], x + w)]
                        if roi.size != 0:
                            small_roi = cv2.resize(
                                roi, (0, 0), fx=Config.RECOG_SCALE, fy=Config.RECOG_SCALE)
                            try:
                                recog_q.put_nowait(
                                    (tid, small_roi, time.time()))
                            except Exception:
                                pass
                # yawn detection
                if tracks_meta[tid]['frames'] % Config.YAWN_SKIP == 0:
                    roi = full_frame[max(0, y):min(
                        full_frame.shape[0], y + h), max(0, x):min(full_frame.shape[1], x + w)]
                    if roi.size != 0:
                        label = None
                        if classifier is not None:
                            label, _ = predict_expression(
                                roi, classifier, label_encoder, facemesh_extractor)
                        if label == 'yawning' or (classifier is None and detect_yawn(roi)):
                            tracks_meta[tid]['yawn_count'] = tracks_meta[tid].get(
                                'yawn_count', 0) + 1
                        if tracks_meta[tid]['yawn_count'] >= Config.YAWN_THRESHOLD and not tracks_meta[tid].get('alerted'):
                            driver_details = tracks_meta[tid].get(
                                'driver_details')
                            if driver_details:
                                reason = f'Yawning detected {tracks_meta[tid]["yawn_count"]} times.'
                                send_telegram_alert(
                                    driver_details, reason, full_frame)
                                tracks_meta[tid]['alerted'] = True
            for tid in to_delete:
                trackers.pop(tid, None)
                tracks_meta.pop(tid, None)
            t_track += (time.perf_counter() - t_tr0)

            # process recognition results
            try:
                while True:
                    res = recog_result_q.get_nowait()
                    tid = res.get('track_id')
                    if tid in tracks_meta:
                        if res.get('identity'):
                            tracks_meta[tid]['identity'] = (
                                res['identity'], res['confidence'])
                            tracks_meta[tid]['driver_details'] = drivers_db.get(
                                res['identity'], {})
                            identity_cache[str(tid)] = (
                                res['identity'], res['confidence'], time.time())
                        t_recog += res.get('recog_ms', 0) / 1000.0
            except Exception:
                pass

            # render
            t_r0 = time.perf_counter()
            for tid, meta in tracks_meta.items():
                x, y, w, h = meta['bbox']
                iden = meta.get('identity')
                if iden:
                    name = str(iden[0])
                    conf = iden[1]
                    color = (0, 255, 0)
                    label = f"{name}: {conf*100:.0f}%"
                else:
                    color = (0, 0, 255)
                    label = 'Unknown'
                cv2.rectangle(full_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(full_frame, label, (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            stats_text = f"F:{frame_count} Tks:{len(trackers)}"
            cv2.putText(full_frame, stats_text, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Pipeline', full_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            t_render += (time.perf_counter() - t_r0)

            stats.update(capture=t_capture, detect=t_detect,
                         recog=t_recog, track=t_track, render=t_render)
            if time.time() - last_log >= Config.LOG_INTERVAL:
                logger.info(f"Stage ms avg: {stats.summary()}")
                stats.reset()
                last_log = time.time()

    except KeyboardInterrupt:
        logger.info('Interrupted by user')
    finally:
        stop_event.set()
        detector.terminate()
        recognizer.terminate()
        capture_thread.join(timeout=1)
        detector.join(timeout=1)
        recognizer.join(timeout=1)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
