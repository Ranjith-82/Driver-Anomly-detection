"""Buffered integration of the existing face-recognition and behavior programs.

The imported modules remain the algorithm owners. This file adds only the
camera/worker/result-buffer integration layer.
"""

import argparse
import os
import pickle
import queue
import threading
import time
# used for Counter = "How many times did each thing happen?,deque = "Keep the most recent items."
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import face_recognition_system

import numpy as np

import face_recognition_system
import feature_fusion
import gaze_detection
import train_and_run_driver_detector as behavior_detection

try:
    import integrated_driver_system_optimized as telegram_source
except Exception:
    telegram_source = None


SCRIPT_DIR = Path(__file__).resolve().parent
YOLO_MODEL_PATH = SCRIPT_DIR / 'best_phone_finetuned_v1.pt'
OBJECT_DETECTION_IMAGE_SIZE = 416
OBJECT_DETECTION_BASE_CONFIDENCE = 0.10
OBJECT_DETECTION_IOU_THRESHOLD = 0.45
OBJECT_DETECTION_MAX_DETECTIONS = 20
OBJECT_DETECTION_FRAME_SKIP = 2
OBJECT_DETECTION_HISTORY_LENGTH = 3
OBJECT_DETECTION_MIN_HITS = 1
OBJECT_DETECTION_CLASS_NAMES = {
    0: 'Phone',
    1: 'Cigarette',
    2: 'Cup',
    3: 'Bottle',
}
OBJECT_DETECTION_CLASS_CONFIDENCE = {
    0: 0.20,
    1: 0.50,
    2: 0.35,
    3: 0.40,
}
OBJECT_DETECTION_ALERTS = {
    0: ('MOBILE PHONE DETECTED', 'mobile_phone'),
    1: ('CIGARETTE DETECTED', 'cigarette'),
    2: ('CUP DETECTED', 'cup'),
    3: ('BOTTLE DETECTED', 'bottle'),
}


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else SCRIPT_DIR / path


@dataclass
class FramePacket:
    frame_id: int
    timestamp: float
    frame: np.ndarray


class LatestFrameBuffer:
    """Thread-safe one-slot buffer. New frames replace stale frames."""

    def __init__(self):
        # helps threads wait for new frames.
        self.condition = threading.Condition()
        self.packet = None  # stores the newest frame.

    def put(self, packet):
        with self.condition:
            self.packet = packet
            self.condition.notify_all()

    def get_newest(self, last_frame_id, stop_event):
        with self.condition:
            while (self.packet is None or self.packet.frame_id <= last_frame_id) and not stop_event.is_set():
                self.condition.wait(timeout=0.1)
            if stop_event.is_set() or self.packet is None:
                return None
            return self.packet


class ResultBuffer:
    """Merged latest results; each worker updates only its own namespace."""

    def __init__(self):
        self.lock = threading.Lock()
        self.values = {
            'driver_name': 'IDENTIFYING...',
            'recognition_confidence': 0.0,
            'recognition_frame_id': -1,
            'recognition_timestamp': 0.0,
            'recognition_location': None,
            'recognition_complete': False,
            'driver_confirmed': False,
            'recognition_active': True,
            'behavior': 'FACE NOT DETECTED',
            'left_ear': None,
            'right_ear': None,
            'average_ear': None,
            'eyes_state': 'EYE STATE UNCERTAIN',
            'blink_count_10s': 0,
            'blink_rate': 0.0,
            'eyes_closed_duration': 0.0,
            'mar': None,
            'pitch': None,
            'yaw': None,
            'roll': None,
            'head_position': 'HEAD POSITION UNCERTAIN',
            'fatigue_alert': False,
            'distracted': False,
            'yawning': False,
            'talking': False,
            'gaze_direction': 'UNKNOWN',
            'gaze_x': None,
            'gaze_y': None,
            'gaze_distraction_elapsed': 0.0,
            'gaze_distraction_confirmed': False,
            'behavior_confidence': 0.0,
            'behavior_frame_id': -1,
            'behavior_timestamp': 0.0,
            'face_present': False,
            'face_lost_since': None,
            'yawn_event': False,
            'behavior_error': '',
            'recognition_error': '',
            'object_detections': [],
            'object_detection_frame_id': -1,
            'object_detection_timestamp': 0.0,
            'object_detection_error': '',
            'feature_fusion': None,
            'feature_vector': [],
            'feature_names': list(feature_fusion.FEATURE_NAMES),
        }

    def update(self, values):
        with self.lock:
            self.values.update(values)

    def snapshot(self):
        with self.lock:
            return self.values.copy()


class TelegramWorker(threading.Thread):
    """Sends alert snapshots off the camera and behavior threads."""

    def __init__(self, stop_event, database_path, cooldown):
        super().__init__(name='telegram-worker', daemon=True)
        self.stop_event = stop_event
        self.cooldown = cooldown
        self.queue = queue.Queue(maxsize=8)  # maximum 8 alerts can wait
        self.database_path = database_path
        self.bot_token = os.getenv('DRIVER_MONITOR_BOT_TOKEN', '')
        self.chat_id = os.getenv('DRIVER_MONITOR_CHAT_ID', '')

    def submit(self, reason, values, frame):
        item = (reason, values.copy(), frame.copy())
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            print('[TELEGRAM] Alert queue full; dropping stale alert')

    def run(self):
        print('[TELEGRAM] Worker started')
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                reason, values, frame = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.send_alert(reason, values, frame)
            except Exception as error:
                print(f'[TELEGRAM] ERROR: {error}')
            finally:
                self.queue.task_done()
        print('[TELEGRAM] Worker stopped')

    def send_alert(self, reason, values, frame):
        if telegram_source is None:
            raise RuntimeError(
                'Existing Telegram module could not be imported')
        if not self.bot_token or not self.chat_id:
            raise RuntimeError(
                'Telegram credentials are missing. Set DRIVER_MONITOR_BOT_TOKEN '
                'and DRIVER_MONITOR_CHAT_ID in this PowerShell window.')
        telegram_source.Config.BOT_TOKEN = self.bot_token
        telegram_source.Config.CHAT_ID = self.chat_id
        driver_details = {}
        if values['driver_name'] not in {'UNKNOWN', 'IDENTIFYING...'}:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'rb') as file:
                    drivers = pickle.load(file)
                for driver in drivers.values():
                    if driver.get('full_name') == values['driver_name']:
                        driver_details = driver
                        break
        if not driver_details:
            if reason in {
                    alert[0] for alert in OBJECT_DETECTION_ALERTS.values()}:
                driver_details = {'full_name': 'Unknown driver'}
            else:
                print('[TELEGRAM] Skipped: no confirmed driver details')
                return
        details = (
            f'{reason}; EAR={values["average_ear"]}; MAR={values["mar"]}; '
            f'head={values["head_position"]}; pitch={values["pitch"]}; yaw={values["yaw"]}'
        )
        sent = telegram_source.send_telegram_alert(
            driver_details, details, frame)
        status = 'Sent' if sent else 'Failed'
        print(
            f'[TELEGRAM] {status}: {reason} for {values["driver_name"]}')

    def close(self):
        self.stop_event.set()
        self.join(timeout=8.0)


class AlertManager:
    """Applies priority and cooldowns to behavior events."""

    def __init__(self, telegram_worker, cooldown=60.0, distraction_delay=0.5,
                 yawn_cooldown=10.0, fatigue_events=3,
                 fatigue_window=60.0):
        self.telegram_worker = telegram_worker
        self.cooldown = cooldown
        self.distraction_delay = distraction_delay
        self.yawn_cooldown = yawn_cooldown
        self.fatigue_events_required = fatigue_events
        self.fatigue_window = fatigue_window
        self.fatigue_event_times = deque()
        self.fatigue_active = False
        self.fatigue_notification_sent = False
        self.last_sent = {}
        self.distraction_started = None
        self.gaze_distraction_alerted = False
        self.object_classes_active = set()
        self.pending_object_alerts = deque()
        self.last_alert = 'NONE'

    def update(self, values, frame, now):
        behavior = values['behavior']
        detected_object_classes = {
            detection['class_id']
            for detection in values.get('object_detections', [])
            if detection.get('class_id') in OBJECT_DETECTION_ALERTS
        }
        new_object_classes = detected_object_classes - self.object_classes_active
        for class_id in sorted(new_object_classes):
            alert = OBJECT_DETECTION_ALERTS[class_id]
            if class_id == 0:
                self.pending_object_alerts.appendleft(alert)
            else:
                self.pending_object_alerts.append(alert)
        self.object_classes_active = detected_object_classes

        # Only detect fatigue if face is present
        fatigue_detected = (values['face_present'] and
                            (behavior == 'FATIGUE ALERT' or values['fatigue_alert']))
        if fatigue_detected and not self.fatigue_active:
            self.fatigue_event_times.append(now)
            self.fatigue_active = True
            print(
                f'[ALERT] Fatigue event detected at {now:.2f}. Total events: {len(self.fatigue_event_times)}')
        elif not fatigue_detected:
            self.fatigue_active = False

        fatigue_cutoff = now - self.fatigue_window
        while self.fatigue_event_times and self.fatigue_event_times[0] < fatigue_cutoff:
            self.fatigue_event_times.popleft()
        if not self.fatigue_event_times:
            self.fatigue_notification_sent = False

        if behavior == 'DISTRACTED':
            if self.distraction_started is None:
                self.distraction_started = now
        else:
            self.distraction_started = None
        distracted_duration = 0.0 if self.distraction_started is None else now - \
            self.distraction_started

        gaze_distraction = values.get('gaze_distraction_confirmed', False)
        if not gaze_distraction:
            self.gaze_distraction_alerted = False

        reason = None
        key = None
        if self.pending_object_alerts:
            reason, key = self.pending_object_alerts.popleft()
        elif (len(self.fatigue_event_times) >= self.fatigue_events_required
                and not self.fatigue_notification_sent):
            reason, key = 'FATIGUE ALERT', 'fatigue'
            print(f'[ALERT] FATIGUE ALERT triggered! Events: {len(self.fatigue_event_times)}, '
                  f'Required: {self.fatigue_events_required}')
        elif gaze_distraction and not self.gaze_distraction_alerted:
            reason = (
                'WARNING: DRIVER DISTRACTION DETECTED. '
                'PLEASE LOOK AT THE ROAD.'
            )
            key = 'gaze_distraction'
            self.gaze_distraction_alerted = True
        elif behavior == 'DISTRACTED' and distracted_duration >= self.distraction_delay:
            reason, key = 'DRIVER DISTRACTED', 'distraction'
        elif values.get('yawn_event', False):
            reason, key = 'YAWNING DETECTED', 'yawn'

        if reason is None:
            self.last_alert = 'NONE'
        elif now - self.last_sent.get(key, -float('inf')) >= (self.yawn_cooldown if key == 'yawn' else self.cooldown):
            self.last_sent[key] = now
            self.last_alert = reason
            if key == 'fatigue':
                self.fatigue_notification_sent = True
            self.telegram_worker.submit(reason, values, frame)
            print(f'[ALERT] Alert submitted: {reason}')
        else:
            self.last_alert = f'{reason} (cooldown)'
        return distracted_duration


class CameraProducer(threading.Thread):
    def __init__(self, frame_buffer, stop_event, camera_index, width, height):
        super().__init__(name='camera-producer', daemon=True)
        self.frame_buffer = frame_buffer
        self.stop_event = stop_event
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.capture = None
        self.error = None

    def run(self):
        self.capture = cv2.VideoCapture(self.camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.capture.isOpened():
            self.error = RuntimeError(
                f'Cannot open camera index {self.camera_index}')
            print(f'[CAMERA] ERROR: {self.error}')
            self.stop_event.set()
            return
        print('[CAMERA] Camera opened')
        frame_id = 0
        while not self.stop_event.is_set():
            ok, frame = self.capture.read()
            if not ok:
                print('[CAMERA] Failed to read frame')
                self.stop_event.set()
                break
            self.frame_buffer.put(FramePacket(
                frame_id, time.monotonic(), frame))
            frame_id += 1
        self.capture.release()
        print('[CAMERA] Camera stopped')


class FaceRecognitionWorker(threading.Thread):
    """Existing face_recognition algorithm, driven by latest packets."""

    def __init__(self, frame_buffer, result_buffer, stop_event, interval, tolerance,
                 resize_scale, confirmations, face_lost_grace, database_path):
        super().__init__(name='face-recognition-worker', daemon=True)
        self.frame_buffer = frame_buffer
        self.result_buffer = result_buffer
        self.stop_event = stop_event
        self.interval = interval
        self.tolerance = tolerance
        self.resize_scale = resize_scale
        self.recognizer = None
        self.last_frame_id = -1
        self.next_run = 0.0
        self.confirmation_counts = Counter()
        self.confirmation_required = confirmations
        self.face_lost_grace = face_lost_grace
        self.database_path = database_path

    def reset_identity(self):
        self.confirmation_counts.clear()
        self.result_buffer.update({
            'driver_name': 'UNKNOWN',
            'recognition_confidence': 0.0,
            'recognition_location': None,
            'recognition_complete': False,
            'driver_confirmed': False,
            'recognition_active': True,
            'recognition_error': '',
        })

    def set_recognition_error(self, message):
        self.result_buffer.update({
            'driver_name': 'UNKNOWN',
            'recognition_confidence': 0.0,
            'recognition_location': None,
            'recognition_complete': False,
            'driver_confirmed': False,
            'recognition_active': False,
            'recognition_error': message,
        })

    def run(self):
        try:
            self.recognizer = face_recognition_system.LiveDriverRecognition(
                self.database_path)
            print('[FACE REC] Worker started')
            if not self.recognizer.known_encodings:
                message = (
                    'No usable face encodings; check driver photos and logs')
                self.set_recognition_error(message)
                print(f'[FACE REC] ERROR: {message}')
                return
            while not self.stop_event.is_set():
                packet = self.frame_buffer.get_newest(
                    self.last_frame_id, self.stop_event)
                if packet is None:
                    continue
                self.last_frame_id = packet.frame_id
                state = self.result_buffer.snapshot()
                face_lost_since = state['face_lost_since']
                face_missing = (face_lost_since is not None and
                                packet.timestamp - face_lost_since >= self.face_lost_grace)
                if state['driver_confirmed'] and not face_missing:
                    continue
                if face_missing and state['driver_confirmed']:
                    self.reset_identity()
                if packet.timestamp < self.next_run:
                    continue
                self.next_run = packet.timestamp + self.interval
                self.process_packet(packet)
        except Exception as error:
            self.set_recognition_error(str(error))
            print(f'[FACE REC] ERROR: {error}')
        finally:
            print('[FACE REC] Worker stopped')

    def process_packet(self, packet):
        # Preserved from LiveDriverRecognition.run_live_recognition:
        # resize, HOG locations, encodings, distance, tolerance, confidence.
        small_frame = cv2.resize(packet.frame, (0, 0),
                                 fx=self.resize_scale, fy=self.resize_scale)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition_system.face_recognition.face_locations(
            rgb_small_frame, model='hog')
        encodings = face_recognition_system.face_recognition.face_encodings(
            rgb_small_frame, locations)
        results = []
        for encoding, location in zip(encodings, locations):
            distances = face_recognition_system.face_recognition.face_distance(
                self.recognizer.known_encodings, encoding)
            if len(distances) == 0:
                results.append((None, 0.0, location))
                continue
            best_index = int(np.argmin(distances))
            best_distance = float(distances[best_index])
            if best_distance < self.tolerance:
                results.append(
                    (self.recognizer.known_driver_ids[best_index], 1.0 - best_distance, location))
            else:
                results.append((None, 0.0, location))

        best_result = next(
            (item for item in results if item[0] is not None), None)
        if best_result is None:
            driver_name = 'UNKNOWN'
            confidence = 0.0
            location = results[0][2] if results else None
        else:
            driver_id, confidence, location = best_result
            driver = self.recognizer.drivers.get(driver_id, {})
            driver_name = driver.get('full_name', str(driver_id))
            current_state = self.result_buffer.snapshot()
            if (best_result is not None and current_state['face_lost_since'] is not None
                    and current_state['behavior_timestamp'] >= packet.timestamp):
                best_result = None
                driver_name = 'UNKNOWN'
                confidence = 0.0
                location = None
        if location is not None:
            top, right, bottom, left = location
            location = (
                int(top / self.resize_scale), int(right / self.resize_scale),
                int(bottom / self.resize_scale), int(left / self.resize_scale),
            )
        previous = self.result_buffer.snapshot()
        if best_result is not None:
            self.confirmation_counts[driver_id] += 1
            for known_id in list(self.confirmation_counts):
                if known_id != driver_id:
                    del self.confirmation_counts[known_id]
            confirmed = self.confirmation_counts[driver_id] >= self.confirmation_required
        else:
            confirmed = False
            self.confirmation_counts.clear()
        if confirmed:
            output_name = driver_name
        elif previous['driver_confirmed']:
            output_name = 'UNKNOWN'
        else:
            output_name = 'UNKNOWN'
        self.result_buffer.update({
            'driver_name': output_name,
            'recognition_confidence': confidence,
            'recognition_location': location,
            'recognition_frame_id': packet.frame_id,
            'recognition_timestamp': packet.timestamp,
            'recognition_complete': confirmed,
            'driver_confirmed': confirmed,
            'recognition_active': not confirmed,
            'recognition_error': '',
        })
        print(f'[FACE REC] {driver_name} on frame {packet.frame_id}')


class BehaviorWorker(threading.Thread):
    """Existing behavior classes, evaluated independently on newest packets."""

    def __init__(self, frame_buffer, result_buffer, stop_event, model_path, labels_path,
                 face_model, confidence, pose_yaw, pose_pitch, mar_threshold,
                 eye_close_threshold, eye_open_threshold, fatigue_seconds,
                 distraction_seconds):
        super().__init__(name='behavior-worker', daemon=True)
        self.frame_buffer = frame_buffer
        self.result_buffer = result_buffer
        self.stop_event = stop_event
        self.model_path = model_path
        self.labels_path = labels_path
        self.face_model = face_model
        self.confidence = confidence
        self.pose_yaw = pose_yaw
        self.pose_pitch = pose_pitch
        self.mar_threshold = mar_threshold
        self.eye_close_threshold = eye_close_threshold
        self.eye_open_threshold = eye_open_threshold
        self.fatigue_seconds = fatigue_seconds
        self.distraction_seconds = distraction_seconds
        self.landmark_detector = None
        self.gaze_thresholds = None
        self.last_frame_id = -1

    def run(self):
        detector = self.landmark_detector
        try:
            with open(self.model_path, 'rb') as file:
                classifier = pickle.load(file)
            with open(self.labels_path, 'rb') as file:
                encoder = pickle.load(file)
            if detector is None:
                detector = behavior_detection.FaceLandmarkDetector(
                    self.face_model)
            self.landmark_detector = detector
            if self.gaze_thresholds is None:
                raise RuntimeError('Gaze calibration has not completed')
            pose = behavior_detection.HeadPoseEstimator()
            mouth = behavior_detection.MouthAnalyzer(
                mar_threshold=self.mar_threshold)
            eyes = behavior_detection.EyeAnalyzer(
                close_threshold=self.eye_close_threshold,
                open_threshold=self.eye_open_threshold,
                fatigue_seconds=self.fatigue_seconds,
            )
            state = behavior_detection.BehaviorStateMachine(
                yaw_threshold=self.pose_yaw, pitch_threshold=self.pose_pitch)
            history = deque(maxlen=9)
            gaze_smoother = gaze_detection.GazeSmoother()
            gaze_distraction_timer = gaze_detection.DistractionTimer(
                self.distraction_seconds)
            print('[BEHAVIOR] Worker started')
            while not self.stop_event.is_set():
                packet = self.frame_buffer.get_newest(
                    self.last_frame_id, self.stop_event)
                if packet is None:
                    continue
                self.last_frame_id = packet.frame_id
                self.process_packet(
                    packet, detector, pose, mouth, eyes, state, classifier,
                    encoder, history, gaze_smoother, gaze_distraction_timer)
        except Exception as error:
            self.result_buffer.update({'behavior_error': str(error)})
            print(f'[BEHAVIOR] ERROR: {error}')
        finally:
            if detector is not None:
                detector.close()
            self.landmark_detector = None
            print('[BEHAVIOR] Worker stopped')

    def process_packet(self, packet, detector, pose, mouth, eyes, state,
                       classifier, encoder, history, gaze_smoother,
                       gaze_distraction_timer):
        landmarks = detector.extract(packet.frame)
        if landmarks is None:
            state.update_distraction('HEAD POSITION UNCERTAIN', False)
            gaze_distraction_timer.update(None, packet.timestamp)
            previous = self.result_buffer.snapshot()
            face_lost_since = previous['face_lost_since']
            if face_lost_since is None:
                face_lost_since = packet.timestamp
            self.result_buffer.update({
                'behavior': 'FACE NOT DETECTED',
                'face_present': False,
                'face_lost_since': face_lost_since,
                'fatigue_alert': False,
                'distracted': False,
                'yawning': False,
                'talking': False,
                'yawn_event': False,
                'gaze_direction': 'UNKNOWN',
                'gaze_x': None,
                'gaze_y': None,
                'gaze_distraction_elapsed': 0.0,
                'gaze_distraction_confirmed': False,
                'head_position': 'HEAD POSITION UNCERTAIN',
                'eyes_state': 'EYE STATE UNCERTAIN',
                'left_ear': None, 'right_ear': None, 'average_ear': None,
                'behavior_frame_id': packet.frame_id,
                'behavior_timestamp': packet.timestamp,
            })
            print(f'[BEHAVIOR] Face lost on frame {packet.frame_id}')
            return

        gaze_landmarks = [
            (float(point[0]), float(point[1])) for point in landmarks
        ]
        eye_landmarks = gaze_detection.extract_eye_landmarks(
            gaze_landmarks, packet.frame.shape[1], packet.frame.shape[0])
        gaze_position = (
            gaze_detection.calculate_gaze_position(eye_landmarks)
            if eye_landmarks else None
        )
        if gaze_position is None:
            gaze_direction = 'UNKNOWN'
            gaze_distraction_elapsed, gaze_distraction_confirmed = (
                gaze_distraction_timer.update(None, packet.timestamp)
            )
        else:
            gaze_candidate = gaze_detection.classify_gaze(
                gaze_position, self.gaze_thresholds)
            gaze_direction = gaze_smoother.update(gaze_candidate)
            gaze_distraction_elapsed, gaze_distraction_confirmed = (
                gaze_distraction_timer.update(
                    gaze_direction, packet.timestamp)
            )

        left_ear, right_ear, average_ear, eyes_state, blink_rate, _, closed_duration, fatigue = eyes.update(
            landmarks, packet.timestamp)
        _, rotation = behavior_detection.corrected_head_pose(
            landmarks, packet.frame.shape[1], packet.frame.shape[0])
        smoothed_pose = pose.update(rotation)
        if pose.calibrating or smoothed_pose is None:
            self.result_buffer.update({
                'behavior': 'CALIBRATING',
                'face_present': True,
                'face_lost_since': None,
                'left_ear': left_ear, 'right_ear': right_ear, 'average_ear': average_ear,
                'eyes_state': eyes_state, 'blink_count_10s': len(eyes.blink_times),
                'blink_rate': blink_rate, 'eyes_closed_duration': closed_duration,
                'fatigue_alert': fatigue, 'behavior_frame_id': packet.frame_id,
                'behavior_timestamp': packet.timestamp,
                'yawn_event': False,
                'gaze_direction': gaze_direction,
                'gaze_x': gaze_position.x if gaze_position else None,
                'gaze_y': gaze_position.y if gaze_position else None,
                'gaze_distraction_elapsed': gaze_distraction_elapsed,
                'gaze_distraction_confirmed': gaze_distraction_confirmed,
            })
            return

        pitch, yaw, roll = [float(value) for value in smoothed_pose]
        head_position = state.position(pitch, yaw)
        mar = behavior_detection.mouth_features(landmarks)
        yawn_event, talking, _ = mouth.update(mar, packet.timestamp)
        distracted = state.update_distraction(head_position, True)
        vector = behavior_detection.feature_vector(
            # creates the features for your ML model.
            landmarks, packet.frame.shape[1], packet.frame.shape[0])
        # "What behavior does this face currently represent?"
        probabilities = classifier.predict_proba([vector])[0]
        index = int(np.argmax(probabilities))  # pick the highest probabilities
        behavior_confidence = float(probabilities[index])
        ml_label = str(encoder.inverse_transform([index])[0])
        if fatigue:
            behavior = 'FATIGUE ALERT'
        elif distracted:
            behavior = 'DISTRACTED'
        elif yawn_event or mouth.yawn_active:
            behavior = 'YAWNING'
        elif talking:
            behavior = 'TALKING'
        elif ml_label in {'talking', 'yawning'} and behavior_confidence >= self.confidence:
            behavior = ml_label.upper()
        else:
            behavior = 'NORMAL'
        self.result_buffer.update({
            'behavior': behavior, 'left_ear': left_ear, 'right_ear': right_ear,
            'face_present': True,
            'face_lost_since': None,
            'average_ear': average_ear, 'eyes_state': eyes_state,
            'blink_count_10s': len(eyes.blink_times), 'blink_rate': blink_rate,
            'eyes_closed_duration': closed_duration, 'mar': mar,
            'pitch': pitch, 'yaw': yaw, 'roll': roll,
            'head_position': head_position, 'fatigue_alert': fatigue,
            'distracted': distracted, 'yawning': yawn_event or mouth.yawn_active,
            'yawn_event': yawn_event,
            'talking': talking, 'behavior_confidence': behavior_confidence,
            'behavior_frame_id': packet.frame_id, 'behavior_timestamp': packet.timestamp,
            'gaze_direction': gaze_direction,
            'gaze_x': gaze_position.x if gaze_position else None,
            'gaze_y': gaze_position.y if gaze_position else None,
            'gaze_distraction_elapsed': gaze_distraction_elapsed,
            'gaze_distraction_confirmed': gaze_distraction_confirmed,
        })


class ObjectDetectionWorker(threading.Thread):
    """Runs object detection on the shared latest-frame buffer."""

    def __init__(self, frame_buffer, result_buffer, stop_event, model_path):
        super().__init__(name='object-detection-worker', daemon=True)
        self.frame_buffer = frame_buffer
        self.result_buffer = result_buffer
        self.stop_event = stop_event
        self.model_path = model_path
        self.last_frame_id = -1

    def run(self):
        try:
            try:
                import torch
                from ultralytics import YOLO
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    f'Required YOLO runtime dependency is unavailable: '
                    f'{error.name}') from error
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f'YOLO model not found: {self.model_path}')

            use_gpu = torch.cuda.is_available()
            device = 0 if use_gpu else 'cpu'
            model = YOLO(str(self.model_path))
            self.validate_class_mapping(model.names)
            print(
                f'[YOLO] Object detection worker started using '
                f'{self.model_path.name} on '
                f'{"CUDA GPU" if use_gpu else "CPU"}')
            history = deque(maxlen=OBJECT_DETECTION_HISTORY_LENGTH)
            while not self.stop_event.is_set():
                packet = self.frame_buffer.get_newest(
                    self.last_frame_id, self.stop_event)
                if packet is None:
                    continue
                self.last_frame_id = packet.frame_id
                if ((packet.frame_id + 1)
                        % OBJECT_DETECTION_FRAME_SKIP != 0):
                    continue

                detections = self.detect(model, packet.frame, device, use_gpu)
                history.append({
                    detection['class_id'] for detection in detections
                })
                hit_counts = Counter(
                    class_id
                    for frame_classes in history
                    for class_id in frame_classes
                )
                detections = [
                    detection for detection in detections
                    if hit_counts[detection['class_id']]
                    >= OBJECT_DETECTION_MIN_HITS
                ]
                print(
                    f'[YOLO] Scanned frame {packet.frame_id}: '
                    f'{len(detections)} detection(s)')
                self.result_buffer.update({
                    'object_detections': detections,
                    'object_detection_frame_id': packet.frame_id,
                    'object_detection_timestamp': packet.timestamp,
                    'object_detection_error': '',
                })
        except Exception as error:
            self.result_buffer.update({
                'object_detections': [],
                'object_detection_error': str(error),
            })
            print(f'[YOLO] ERROR: {error}')
        finally:
            print('[YOLO] Object detection worker stopped')

    @staticmethod
    def validate_class_mapping(model_names):
        if isinstance(model_names, dict):
            actual_names = {
                int(class_id): str(name)
                for class_id, name in model_names.items()
            }
        else:
            actual_names = {
                class_id: str(name)
                for class_id, name in enumerate(model_names)
            }
        for class_id, expected_name in OBJECT_DETECTION_CLASS_NAMES.items():
            actual_name = actual_names.get(class_id, '').strip().casefold()
            if actual_name != expected_name.casefold():
                raise ValueError(
                    'Incompatible YOLO class mapping: expected '
                    f'{class_id}={expected_name}, got '
                    f'{actual_names.get(class_id, "<missing>")!r}; '
                    f'model classes are {actual_names}')

    def detect(self, model, frame, device, use_gpu):
        detections = []
        result = model.predict(
            source=frame,
            imgsz=OBJECT_DETECTION_IMAGE_SIZE,
            conf=OBJECT_DETECTION_BASE_CONFIDENCE,
            iou=OBJECT_DETECTION_IOU_THRESHOLD,
            device=device,
            half=use_gpu,
            max_det=OBJECT_DETECTION_MAX_DETECTIONS,
            verbose=False,
        )[0]
        if result.boxes is None:
            return detections

        frame_area = frame.shape[0] * frame.shape[1]
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            if class_id not in OBJECT_DETECTION_CLASS_NAMES:
                continue
            if confidence < OBJECT_DETECTION_CLASS_CONFIDENCE[class_id]:
                continue
            coordinates = tuple(
                int(value) for value in box.xyxy[0].tolist())
            x1, y1, x2, y2 = coordinates
            box_area = max(0, x2 - x1) * max(0, y2 - y1)
            if box_area > 0.85 * frame_area:
                continue
            detections.append({
                'class_id': class_id,
                'name': OBJECT_DETECTION_CLASS_NAMES[class_id],
                'confidence': confidence,
                'box': coordinates,
            })
        return detections


class BufferedDriverMonitor:
    def __init__(self, args):
        self.args = args
        self.stop_event = threading.Event()
        self.frame_buffer = LatestFrameBuffer()
        self.result_buffer = ResultBuffer()
        self.camera = CameraProducer(
            self.frame_buffer, self.stop_event, args.camera, args.width, args.height)
        self.recognition = FaceRecognitionWorker(
            self.frame_buffer, self.result_buffer, self.stop_event,
            args.recognition_interval, args.recognition_tolerance, args.resize_scale,
            args.recognition_confirmations, args.face_lost_grace, args.database)
        self.behavior = BehaviorWorker(
            self.frame_buffer, self.result_buffer, self.stop_event,
            args.model, args.labels, args.face_landmarker_model, args.confidence,
            args.pose_yaw, args.pose_pitch, args.mar_threshold,
            args.eye_close_threshold, args.eye_open_threshold, args.fatigue_seconds,
            args.distraction_seconds)
        self.gaze_thresholds = None
        self.telegram = TelegramWorker(
            self.stop_event, args.database, args.telegram_cooldown)
        self.alerts = AlertManager(
            self.telegram, cooldown=args.telegram_cooldown,
            distraction_delay=args.distraction_seconds,
            yawn_cooldown=args.yawn_cooldown)
        self.object_detection = ObjectDetectionWorker(
            self.frame_buffer, self.result_buffer, self.stop_event,
            YOLO_MODEL_PATH)

    @staticmethod
    def _gaze_position_from_landmarks(landmarks, frame):
        if landmarks is None:
            return None
        points = [(float(point[0]), float(point[1])) for point in landmarks]
        eyes = gaze_detection.extract_eye_landmarks(
            points, frame.shape[1], frame.shape[0])
        return gaze_detection.calculate_gaze_position(eyes) if eyes else None

    def calibrate_gaze(self):
        """Capture all five gaze poses before starting monitoring workers."""
        self.camera.start()
        self.behavior.landmark_detector = (
            behavior_detection.FaceLandmarkDetector(
                self.args.face_landmarker_model)
        )
        calibration = gaze_detection.CalibrationSession()
        last_frame_id = -1

        while not calibration.complete and not self.stop_event.is_set():
            packet = self.frame_buffer.get_newest(
                last_frame_id, self.stop_event)
            if packet is None:
                if self.camera.error is not None:
                    raise RuntimeError(str(self.camera.error))
                raise RuntimeError('Camera stopped during gaze calibration')
            last_frame_id = packet.frame_id
            landmarks = self.behavior.landmark_detector.extract(packet.frame)
            position = self._gaze_position_from_landmarks(
                landmarks, packet.frame)
            now = time.monotonic()
            calibration.update(now, position)

            screen = packet.frame.copy()
            direction = calibration.current_direction or 'COMPLETE'
            instruction = (
                'LOOK STRAIGHT' if direction == 'FORWARD'
                else 'LOOK ' + direction
            )
            cv2.rectangle(screen, (12, 12), (650, 190), (18, 25, 30), -1)
            cv2.putText(
                screen,
                'GAZE CALIBRATION - STEP {}/5'.format(
                    min(calibration.index + 1, len(gaze_detection.GAZE_DIRECTIONS))
                ),
                (28, 48), cv2.FONT_HERSHEY_DUPLEX, 0.75,
                (106, 222, 202), 2, cv2.LINE_AA,
            )
            cv2.putText(
                screen, instruction, (28, 91), cv2.FONT_HERSHEY_DUPLEX,
                0.9, (100, 210, 255), 2, cv2.LINE_AA,
            )
            if position is None:
                status = 'Face/eyes not detected - center your face in camera'
                color = (80, 100, 255)
            else:
                status = 'Face and eyes detected'
                color = (150, 225, 180)
            cv2.putText(
                screen, status, (28, 128), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 1, cv2.LINE_AA,
            )
            if calibration.captured:
                if calibration.current_direction == 'DOWN':
                    message = 'Final gaze samples captured. Starting monitoring.'
                else:
                    message = 'Samples captured. Press SPACE for next direction.'
            elif calibration.capture_started is not None:
                remaining = max(
                    0.0, gaze_detection.CALIBRATION_CAPTURE_SECONDS
                    - (now - calibration.capture_started)
                )
                message = 'Hold gaze steady - collecting samples ({:.1f}s)'.format(
                    remaining)
            elif calibration.retry_message:
                message = calibration.retry_message
            elif position is not None:
                message = 'Hold gaze steady, then press SPACE to capture.'
            else:
                message = 'Face/eyes must be detected before capture.'
            cv2.putText(
                screen, message, (28, 164), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (220, 230, 226), 1, cv2.LINE_AA,
            )
            cv2.imshow('AI Driver Safety Monitor - Gaze Calibration', screen)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                self.stop_event.set()
                cv2.destroyAllWindows()
                return False
            if key == 32:
                if calibration.captured:
                    calibration.start_capture(now)
                elif calibration.capture_started is None and position is not None:
                    calibration.start_capture(now)
            if (calibration.captured
                    and calibration.current_direction == 'DOWN'):
                calibration.start_capture(now)

        if self.stop_event.is_set():
            return False

        self.gaze_thresholds = gaze_detection.calibrate_gaze_thresholds(
            calibration.observations)
        self.behavior.gaze_thresholds = self.gaze_thresholds
        print('[GAZE] Calibration values (normalized gaze coordinates):')
        for direction in gaze_detection.GAZE_DIRECTIONS:
            position = calibration.observations[direction]
            print('[GAZE] {} -> X: {:.4f} Y: {:.4f}'.format(
                direction, position.x, position.y))
        print('[GAZE] Personalized thresholds: UP {:.4f}, DOWN {:.4f}'.format(
            self.gaze_thresholds.up, self.gaze_thresholds.down))
        print('[GAZE] Calibration successful; starting driver monitoring')
        cv2.destroyWindow('AI Driver Safety Monitor - Gaze Calibration')
        return True

    def start(self):
        if self.gaze_thresholds is None:
            raise RuntimeError(
                'Gaze calibration must complete before monitoring starts')
        self.recognition.start()
        self.behavior.start()
        self.object_detection.start()
        self.telegram.start()
        print('[BUFFER] Camera and independent workers started')

    def stop(self):
        self.stop_event.set()
        with self.frame_buffer.condition:
            self.frame_buffer.condition.notify_all()
        for worker in (self.camera, self.recognition, self.behavior,
                       self.object_detection):
            if worker.ident is not None:
                worker.join(timeout=3.0)
        if self.telegram.ident is not None:
            self.telegram.close()
        if (self.behavior.ident is None
                and self.behavior.landmark_detector is not None):
            self.behavior.landmark_detector.close()
            self.behavior.landmark_detector = None
        print('[BUFFER] Shutdown complete')

    def display(self):
        last_frame_id = -1
        previous_time = time.perf_counter()
        fps = 0.0
        while not self.stop_event.is_set():
            packet = self.frame_buffer.get_newest(
                last_frame_id, self.stop_event)
            if packet is None:
                break
            last_frame_id = packet.frame_id
            values = self.result_buffer.snapshot()

            # Feature fusion is intentionally inserted after the result buffer, so
            # the dashboard continues to consume the same upstream values without
            # any changes to its existing render path.
            fused = feature_fusion.fuse_frame_features(values)
            values['feature_fusion'] = fused
            values['feature_vector'] = fused['feature_vector']
            values['feature_names'] = fused['feature_names']
            self.result_buffer.update({
                'feature_fusion': fused,
                'feature_vector': fused['feature_vector'],
                'feature_names': fused['feature_names'],
            })

            self.alerts.update(values, packet.frame, packet.timestamp)
            values['alert_status'] = self.alerts.last_alert
            current = time.perf_counter()
            instant = 1.0 / max(current - previous_time, 1e-6)
            fps = instant if fps == 0 else 0.1 * instant + 0.9 * fps
            previous_time = current
            dashboard = self.render_dashboard(
                packet.frame, values, fps, packet)
            cv2.imshow('AI Driver Safety Monitor', dashboard)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                self.stop_event.set()
                break
        self.stop()
        cv2.destroyAllWindows()

    @staticmethod
    def rounded_panel(canvas, x, y, width, height, fill, radius=14,
                      border=None):
        cv2.rectangle(canvas, (x + radius, y),
                      (x + width - radius, y + height), fill, -1)
        cv2.rectangle(canvas, (x, y + radius),
                      (x + width, y + height - radius), fill, -1)
        cv2.circle(canvas, (x + radius, y + radius), radius, fill, -1)
        cv2.circle(canvas, (x + width - radius, y + radius), radius, fill, -1)
        cv2.circle(canvas, (x + radius, y + height - radius), radius, fill, -1)
        cv2.circle(canvas, (x + width - radius, y +
                   height - radius), radius, fill, -1)
        if border is not None:
            cv2.ellipse(canvas, (x + radius, y + radius),
                        (radius, radius), 0, 180, 270, border, 1)
            cv2.ellipse(canvas, (x + width - radius, y + radius),
                        (radius, radius), 0, 270, 360, border, 1)
            cv2.ellipse(canvas, (x + radius, y + height - radius),
                        (radius, radius), 0, 90, 180, border, 1)
            cv2.ellipse(canvas, (x + width - radius, y + height - radius),
                        (radius, radius), 0, 0, 90, border, 1)
            cv2.line(canvas, (x + radius, y),
                     (x + width - radius, y), border, 1)
            cv2.line(canvas, (x + radius, y + height),
                     (x + width - radius, y + height), border, 1)
            cv2.line(canvas, (x, y + radius),
                     (x, y + height - radius), border, 1)
            cv2.line(canvas, (x + width, y + radius),
                     (x + width, y + height - radius), border, 1)

    @staticmethod
    def put_label(canvas, text, x, y, color=(139, 153, 166), scale=0.42):
        cv2.putText(canvas, text.upper(), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)

    @staticmethod
    def put_value(canvas, text, x, y, color=(235, 240, 244), scale=0.72,
                  thickness=2):
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def fit_camera(frame, width, height):
        source_height, source_width = frame.shape[:2]
        ratio = min(width / source_width, height / source_height)
        resized = cv2.resize(frame, (max(1, int(source_width * ratio)),
                                     max(1, int(source_height * ratio))),
                             interpolation=cv2.INTER_AREA)
        output = np.full((height, width, 3), (14, 20, 27), dtype=np.uint8)
        offset_x = (width - resized.shape[1]) // 2
        offset_y = (height - resized.shape[0]) // 2
        output[offset_y:offset_y + resized.shape[0],
               offset_x:offset_x + resized.shape[1]] = resized
        return output

    @classmethod
    def render_dashboard(cls, frame, values, fps, packet):
        width, height = 1440, 840
        canvas = np.full((height, width, 3), (12, 18, 26), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (width, 92), (18, 27, 38), -1)
        cv2.line(canvas, (28, 91), (width - 28, 91), (42, 61, 76), 1)
        cls.put_label(canvas, 'AI DRIVER SAFETY MONITOR', 34, 38,
                      (119, 211, 198), 0.72)
        cv2.putText(canvas, 'Real-time driver behavior and fatigue detection',
                    (36, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (145, 159, 172), 1, cv2.LINE_AA)
        cv2.circle(canvas, (width - 205, 39), 6, (76, 214, 159), -1)
        cls.put_label(canvas, 'SYSTEM ONLINE', width - 188, 45,
                      (184, 220, 205), 0.48)

        left_x, left_w = 30, 430
        video_x, video_y, video_w, video_h = 490, 116, 920, 618
        bg_card = (22, 31, 42)
        border = (42, 59, 73)
        accent = (119, 211, 198)
        cls.rounded_panel(canvas, left_x, 116, left_w,
                          126, bg_card, border=border)
        cls.put_label(canvas, 'Driver Identified', left_x + 22, 143)
        driver_name = values['driver_name']
        cls.put_value(canvas, driver_name, left_x + 22, 184, accent, 0.92)
        cls.put_label(canvas, f'Confidence {values["recognition_confidence"] * 100:.0f}%',
                      left_x + 22, 216, (153, 181, 190), 0.44)
        if values['recognition_error']:
            recognition = 'ERROR'
            recognition_color = (239, 98, 92)
        elif values['driver_confirmed']:
            recognition = 'CONFIRMED'
            recognition_color = (112, 194, 166)
        elif values['recognition_active']:
            recognition = 'SCANNING'
            recognition_color = (231, 183, 91)
        else:
            recognition = 'WAITING'
            recognition_color = (153, 181, 190)
        cls.put_label(canvas, f'Recognition {recognition}', left_x + 250, 216,
                      recognition_color, 0.40)
        if values['recognition_error']:
            error_text = f'Error: {values["recognition_error"]}'[:58]
            cls.put_label(canvas, error_text, left_x + 22, 235,
                          (239, 98, 92), 0.32)

        behavior = values['behavior']
        behavior_color = ((76, 214, 159) if behavior == 'NORMAL' else
                          (243, 183, 76) if behavior in {'DISTRACTED', 'YAWNING', 'TALKING'} else
                          (239, 98, 92))
        cls.rounded_panel(canvas, left_x, 256, left_w, 126,
                          (29, 39, 51), border=behavior_color)
        cls.put_label(canvas, 'Current Behavior', left_x + 22, 283)
        cls.put_value(canvas, behavior, left_x + 22, 326, behavior_color, 0.82)
        cls.put_label(canvas, f'Confidence {values["behavior_confidence"] * 100:.0f}%',
                      left_x + 22, 352, (153, 181, 190), 0.44)
        gaze_x = '--' if values['gaze_x'] is None else f'{values["gaze_x"]:.3f}'
        gaze_y = '--' if values['gaze_y'] is None else f'{values["gaze_y"]:.3f}'
        cls.put_label(
            canvas,
            f'Gaze {values["gaze_direction"]}  |  X {gaze_x}  Y {gaze_y}',
            left_x + 22, 370, (154, 214, 190), 0.37,
        )

        cls.rounded_panel(canvas, left_x, 386, left_w,
                          158, bg_card, border=border)
        cls.put_label(canvas, 'Head Position', left_x + 22, 413)
        cls.put_value(canvas, values['head_position'],
                      left_x + 22, 450, accent, 0.68)
        cls.put_label(canvas, f'Pitch {values["pitch"]:+.1f} deg' if values['pitch']
                      is not None else 'Pitch --', left_x + 22, 484, scale=0.44)
        cls.put_label(canvas, f'Yaw {values["yaw"]:+.1f} deg' if values['yaw']
                      is not None else 'Yaw --', left_x + 158, 484, scale=0.44)
        cls.put_label(canvas, f'Roll {values["roll"]:+.1f} deg' if values['roll']
                      is not None else 'Roll --', left_x + 286, 484, scale=0.44)
        cls.put_label(canvas, 'Pose tracking via face mesh',
                      left_x + 22, 520, (101, 126, 142), 0.39)

        cls.rounded_panel(canvas, left_x, 558, left_w,
                          176, bg_card, border=border)
        cls.put_label(canvas, 'Eye Monitoring', left_x + 22, 585)
        cls.put_value(canvas, values['eyes_state'],
                      left_x + 22, 618, (154, 214, 190), 0.60)

        def metric(value):
            return '--' if value is None else f'{value:.3f}'
        cls.put_label(
            canvas, f'EAR L {metric(values["left_ear"])}', left_x + 22, 650, scale=0.42)
        cls.put_label(
            canvas, f'EAR R {metric(values["right_ear"])}', left_x + 158, 650, scale=0.42)
        cls.put_label(
            canvas, f'AVG {metric(values["average_ear"])}', left_x + 294, 650, scale=0.42)
        cls.put_label(
            canvas, f'Closed {values["eyes_closed_duration"]:.2f} s', left_x + 22, 682, scale=0.42)
        cls.put_label(
            canvas, f'Blinks {values["blink_count_10s"]}  |  {values["blink_rate"]:.1f}/min', left_x + 158, 682, scale=0.42)
        cls.put_label(
            canvas, f'MAR {metric(values["mar"])}', left_x + 22, 714, scale=0.42)
        alert_text = values.get('alert_status', 'NONE')
        alert_color = (239, 98, 92) if alert_text != 'NONE' else (76, 214, 159)
        cls.rounded_panel(canvas, left_x, 748, left_w, 52,
                          (25, 35, 45), border=alert_color, radius=10)
        cls.put_label(
            canvas, f'Alert Status  {alert_text}', left_x + 18, 780, alert_color, 0.43)

        frame = frame.copy()
        detected_object_classes = set()
        for detection in values.get('object_detections', []):
            x1, y1, x2, y2 = detection['box']
            class_id = detection['class_id']
            class_name = detection['name']
            detected_object_classes.add(class_id)
            color = (0, 0, 255) if class_id in {0, 1} else (0, 200, 90)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'{class_name} {detection["confidence"]:.2f}'
            cv2.putText(frame, label, (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        if detected_object_classes:
            object_alert = OBJECT_DETECTION_ALERTS[
                min(detected_object_classes)][0]
            cv2.putText(frame, f'ALERT: {object_alert}', (20, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2,
                        cv2.LINE_AA)
        if values.get('gaze_distraction_confirmed', False):
            cv2.putText(
                frame, 'WARNING: DRIVER DISTRACTION DETECTED.',
                (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                (0, 0, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, 'PLEASE LOOK AT THE ROAD.',
                (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                (0, 0, 255), 2, cv2.LINE_AA,
            )
        video = cls.fit_camera(frame, video_w, video_h)
        video_frame = np.full((video_h + 8, video_w + 8, 3),
                              (42, 59, 73), dtype=np.uint8)
        video_frame[4:-4, 4:-4] = video
        cls.rounded_panel(canvas, video_x - 5, video_y - 5, video_w + 10, video_h + 10,
                          (42, 59, 73), radius=16)
        canvas[video_y:video_y + video_h, video_x:video_x + video_w] = video
        cls.put_label(canvas, 'LIVE MONITORING', video_x + 22, video_y + 30,
                      (194, 231, 218), 0.45)
        cv2.circle(canvas, (video_x + 14, video_y + 24), 5, (76, 214, 159), -1)
        cls.put_label(canvas, f'FPS {fps:.1f}', video_x + video_w - 90, video_y + 30,
                      (198, 211, 218), 0.42)
        cls.rounded_panel(canvas, video_x, 754, video_w, 40,
                          (25, 35, 45), border=alert_color, radius=10)
        cls.put_label(
            canvas, f'Alert Status  {alert_text}', video_x + 18, 780, alert_color, 0.43)
        cls.put_label(canvas, f'Frame {packet.frame_id}  |  Results R/B {values["recognition_frame_id"]}/{values["behavior_frame_id"]}',
                      34, 825, (89, 111, 126), 0.38)
        object_status = 'YOLO ERROR' if values.get('object_detection_error') else (
            f'YOLO frame {values.get("object_detection_frame_id", -1)}')
        cls.put_label(canvas, f'Real-time monitoring  |  Face recognition  |  Behavior analysis  |  {object_status}',
                      video_x, 825, (89, 111, 126), 0.38)
        return canvas


def main():
    parser = argparse.ArgumentParser(
        description='Buffered face recognition plus driver behavior monitoring.')
    parser.add_argument('--model', default='driver_behavior_model.pkl')
    parser.add_argument('--labels', default='driver_behavior_labels.pkl')
    parser.add_argument('--face-landmarker-model',
                        default='face_landmarker.task')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=600)
    parser.add_argument('--recognition-interval', type=float, default=2.0)
    parser.add_argument('--recognition-confirmations', type=int, default=2)
    parser.add_argument('--recognition-tolerance',
                        '--tolerance', type=float, default=0.5)
    parser.add_argument('--resize-scale', type=float, default=0.5)
    parser.add_argument('--confidence', type=float, default=0.65)
    parser.add_argument('--pose-yaw', type=float, default=25.0)
    parser.add_argument('--pose-pitch', type=float, default=20.0)
    parser.add_argument('--mar-threshold', type=float, default=0.62)
    parser.add_argument('--eye-close-threshold', type=float, default=0.21)
    parser.add_argument('--eye-open-threshold', type=float, default=0.24)
    parser.add_argument('--fatigue-seconds', type=float, default=1.5)
    parser.add_argument('--face-lost-grace', type=float, default=1.0)
    parser.add_argument('--distraction-seconds', type=float, default=0.6)
    parser.add_argument('--telegram-cooldown', type=float, default=60.0)
    parser.add_argument('--yawn-cooldown', type=float, default=10.0)
    parser.add_argument('--database', default='drivers_database.pkl')
    args = parser.parse_args()
    args.model = resolve_path(args.model)
    args.labels = resolve_path(args.labels)
    args.face_landmarker_model = resolve_path(args.face_landmarker_model)
    args.database = resolve_path(args.database)
    monitor = BufferedDriverMonitor(args)
    try:
        if not monitor.calibrate_gaze():
            monitor.stop()
            return
        monitor.start()
        monitor.display()
    except KeyboardInterrupt:
        monitor.stop()
    except Exception as error:
        monitor.stop()
        raise RuntimeError(f'[MAIN] {error}') from error


if __name__ == '__main__':
    main()
