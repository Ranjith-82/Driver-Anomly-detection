import argparse
import json
import pickle
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
except ImportError:
    mp_tasks = None
    vision = None


CLASS_FOLDERS = {
    'yawning': 'yawning_processed',
    'distracted': 'distracted_pre',
    'talking': 'talking_pre',
    'normal': 'normal_pre',
}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]
POSE_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)
MOUTH_LANDMARKS = (61, 291, 13, 14)
# MediaPipe Face Mesh six-point contours: corners, upper lid, upper lid,
# lower lid, lower lid. The order matches the standard EAR formula.
LEFT_EYE_LANDMARKS = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_LANDMARKS = (362, 385, 387, 263, 373, 380)
FEATURE_VERSION = 'landmarks_xy_plus_relative_pnp_angles_v2'
# Observed webcam convention: positive yaw is physical right and positive
# pitch is physical up. Keep these mappings explicit and easy to recalibrate.
POSITIVE_YAW_LABEL = 'LOOKING RIGHT'
NEGATIVE_YAW_LABEL = 'LOOKING LEFT'
POSITIVE_PITCH_LABEL = 'LOOKING UP'
NEGATIVE_PITCH_LABEL = 'LOOKING DOWN'


class FaceLandmarkDetector:
    def __init__(self, model_path=None):
        """Initialize face landmark detector using MediaPipe solutions API."""
        self.mode = 'solutions'
        self.handle = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.timestamp_ms = 0

    def extract(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.mode == 'solutions':
            result = self.handle.process(rgb)
            if not result.multi_face_landmarks:
                return None
            points = result.multi_face_landmarks[0].landmark
        else:
            self.timestamp_ms += 33
            result = self.handle.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.timestamp_ms)
            if not result.face_landmarks:
                return None
            points = result.face_landmarks[0]
        return np.asarray([[point.x, point.y, point.z] for point in points], dtype=np.float32)

    def close(self):
        if self.handle is not None and hasattr(self.handle, 'close'):
            self.handle.close()


def normalized_landmarks(landmarks):
    coords = landmarks[:, :2]
    center = coords.mean(axis=0)
    scale = max(float(np.ptp(coords, axis=0).max()), 1e-6)
    return ((coords - center) / scale).flatten()


def camera_matrix(width, height):
    focal_length = float(width)
    return np.array([
        [focal_length, 0.0, width / 2.0],
        [0.0, focal_length, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def rotation_matrix_from_landmarks(landmarks, width, height):
    image_points = np.asarray([
        (landmarks[index][0] * width, landmarks[index][1] * height)
        for index in POSE_LANDMARKS
    ], dtype=np.float64)
    success, rotation_vector, _ = cv2.solvePnP(
        POSE_MODEL_POINTS, image_points, camera_matrix(width, height),
        np.zeros((4, 1), dtype=np.float64), flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    rotation, _ = cv2.Rodrigues(rotation_vector)
    return rotation


def angles_from_relative_rotation(relative_rotation):
    """Return calibrated camera-relative pitch, yaw, roll in degrees.

    The neutral rotation is removed before conversion. atan2-based angles are
    derived from the relative matrix, avoiding the +/-180 degree Euler output
    that caused a forward-facing head to appear near +176 degrees.
    """
    pitch = np.degrees(np.arctan2(
        -relative_rotation[2, 1],
        np.sqrt(relative_rotation[0, 0] ** 2 + relative_rotation[1, 1] ** 2),
    ))
    yaw = np.degrees(np.arctan2(
        relative_rotation[2, 0],
        np.sqrt(relative_rotation[2, 1] ** 2 + relative_rotation[2, 2] ** 2),
    ))
    roll = np.degrees(
        np.arctan2(-relative_rotation[0, 1], relative_rotation[1, 1]))
    return np.asarray([pitch, yaw, roll], dtype=np.float32)


def corrected_head_pose(landmarks, width, height, neutral_rotation=None):
    rotation = rotation_matrix_from_landmarks(landmarks, width, height)
    if rotation is None:
        return None, None
    reference = np.eye(3) if neutral_rotation is None else neutral_rotation
    return angles_from_relative_rotation(reference.T @ rotation), rotation


def mouth_features(landmarks):
    left, right, top, bottom = [landmarks[index][:2]
                                for index in MOUTH_LANDMARKS]
    return float(np.linalg.norm(top - bottom) / max(np.linalg.norm(left - right), 1e-6))


def eye_aspect_ratio(landmarks, eye_indices):
    points = np.asarray([landmarks[index][:2]
                        for index in eye_indices], dtype=np.float32)
    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])
    if horizontal <= 1e-6:
        return None
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def eye_landmark_pixels(landmarks, eye_indices, width, height):
    return [(int(landmarks[index][0] * width), int(landmarks[index][1] * height))
            for index in eye_indices]


class EyeAnalyzer:
    def __init__(self, close_threshold=0.21, open_threshold=0.24,
                 window_seconds=10.0, close_confirm_frames=2,
                 open_confirm_frames=2, fatigue_seconds=1.5,
                 recovery_seconds=0.35):
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.window_seconds = window_seconds
        self.close_confirm_frames = close_confirm_frames
        self.open_confirm_frames = open_confirm_frames
        self.fatigue_seconds = fatigue_seconds
        self.recovery_seconds = recovery_seconds
        self.state = 'EYES OPEN'
        self.blink_times = deque()
        self.first_timestamp = None
        self.below_threshold_frames = 0
        self.above_threshold_frames = 0
        self.closure_start_time = None
        self.closed_duration = 0.0
        self.fatigue_alert = False
        self.open_recovery_start = None

    def update(self, landmarks, now):
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_LANDMARKS)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_LANDMARKS)
        if left_ear is None or right_ear is None:
            self.state = 'EYE STATE UNCERTAIN'
            self._expire(now)
            duration = self._closed_duration(now)
            return (left_ear, right_ear, None, self.state, self.blink_rate(now),
                    False, duration, self.fatigue_alert)

        average_ear = (left_ear + right_ear) / 2.0
        if self.first_timestamp is None:
            self.first_timestamp = now
        blink = False
        if self.state in {'EYES OPEN', 'EYE STATE UNCERTAIN'}:
            self.above_threshold_frames = 0
            if average_ear <= self.close_threshold:
                self.below_threshold_frames += 1
                if self.below_threshold_frames >= self.close_confirm_frames:
                    self.state = 'EYES CLOSED'
                    self.closure_start_time = now
                    self.closed_duration = 0.0
                    self.open_recovery_start = None
            else:
                self.below_threshold_frames = 0
                self.state = 'EYES OPEN'
        elif self.state == 'EYES CLOSED':
            self.closed_duration = self._closed_duration(now)
            if self.closed_duration >= self.fatigue_seconds:
                self.fatigue_alert = True
            if average_ear >= self.open_threshold:
                self.above_threshold_frames += 1
                if self.above_threshold_frames >= self.open_confirm_frames:
                    self.state = 'EYES OPEN'
                    self.blink_times.append(now)
                    blink = True
                    self.closure_start_time = None
                    self.closed_duration = 0.0
                    self.below_threshold_frames = 0
                    self.above_threshold_frames = 0
                    self.open_recovery_start = now
            else:
                self.above_threshold_frames = 0
        if self.state == 'EYES OPEN' and self.fatigue_alert:
            if self.open_recovery_start is None:
                self.open_recovery_start = now
            elif now - self.open_recovery_start >= self.recovery_seconds:
                self.fatigue_alert = False
                self.open_recovery_start = None
        self._expire(now)
        return (left_ear, right_ear, average_ear, self.state, self.blink_rate(now),
                blink, self.closed_duration, self.fatigue_alert)

    def _closed_duration(self, now):
        if self.closure_start_time is None:
            return 0.0
        return max(0.0, now - self.closure_start_time)

    def _expire(self, now):
        cutoff = now - self.window_seconds
        while self.blink_times and self.blink_times[0] < cutoff:
            self.blink_times.popleft()

    def blink_rate(self, now):
        self._expire(now)
        return len(self.blink_times) * 60.0 / self.window_seconds


def feature_vector(landmarks, width, height):
    pose, _ = corrected_head_pose(landmarks, width, height)
    if pose is None:
        return None
    return np.concatenate((normalized_landmarks(landmarks), pose)).astype(np.float32)


class HeadPoseEstimator:
    def __init__(self, calibration_frames=45, smoothing=0.35):
        self.calibration_frames = calibration_frames
        self.smoothing = smoothing
        self.calibration_rotations = []
        self.neutral_rotation = None
        self.smoothed_angles = None

    @property
    def calibrating(self):
        return self.neutral_rotation is None

    def reset_calibration(self):
        self.calibration_rotations = []
        self.neutral_rotation = None
        self.smoothed_angles = None

    def _finish_calibration(self):
        median_matrix = np.median(np.asarray(
            self.calibration_rotations), axis=0)
        u, _, vh = np.linalg.svd(median_matrix)
        self.neutral_rotation = u @ vh
        if np.linalg.det(self.neutral_rotation) < 0:
            u[:, -1] *= -1
            self.neutral_rotation = u @ vh

    def update(self, rotation):
        if rotation is None:
            return None
        if self.calibrating:
            self.calibration_rotations.append(rotation)
            if len(self.calibration_rotations) >= self.calibration_frames:
                self._finish_calibration()
            return None
        angles = angles_from_relative_rotation(
            self.neutral_rotation.T @ rotation)
        if self.smoothed_angles is None:
            self.smoothed_angles = angles
        else:
            self.smoothed_angles = self.smoothing * angles + \
                (1.0 - self.smoothing) * self.smoothed_angles
        return self.smoothed_angles.copy()


class MouthAnalyzer:
    def __init__(self, mar_threshold=0.62, close_threshold=0.52,
                 yawn_frames=12, yawn_cooldown=4.0, talking_window=24,
                 movement_threshold=0.035):
        self.mar_threshold = mar_threshold
        self.close_threshold = close_threshold
        self.yawn_frames = yawn_frames
        self.yawn_cooldown = yawn_cooldown
        self.talking_window = talking_window
        self.movement_threshold = movement_threshold
        self.mar_history = deque(maxlen=talking_window)
        self.open_frames = 0
        self.last_yawn_time = -float('inf')
        self.yawn_active = False

    def update(self, mar, now):
        self.mar_history.append(float(mar))
        self.open_frames = self.open_frames + 1 if mar >= self.mar_threshold else 0
        yawn_event = (self.open_frames >= self.yawn_frames and not self.yawn_active
                      and now - self.last_yawn_time >= self.yawn_cooldown)
        if yawn_event:
            self.yawn_active = True
            self.last_yawn_time = now
        if self.yawn_active and mar <= self.close_threshold:
            self.yawn_active = False
        movement = 0.0
        transitions = 0
        if len(self.mar_history) >= 3:
            values = np.asarray(self.mar_history)
            movement = float(np.mean(np.abs(np.diff(values))))
            transitions = int(np.sum((values[1:] >= self.mar_threshold) != (
                values[:-1] >= self.mar_threshold)))
        talking = (not self.yawn_active and len(self.mar_history) >= self.talking_window
                   and movement >= self.movement_threshold and transitions >= 2)
        return yawn_event, talking, movement


class BehaviorStateMachine:
    def __init__(self, yaw_threshold=25.0, pitch_threshold=20.0,
                 distraction_frames=18, recovery_frames=8, lost_grace=12):
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.distraction_frames = distraction_frames
        self.recovery_frames = recovery_frames
        self.lost_grace = lost_grace
        self.turned_frames = 0
        self.forward_frames = 0
        self.face_lost_frames = 0
        self.distracted = False

    def position(self, pitch, yaw):
        if pitch is None or yaw is None:
            return 'HEAD POSITION UNCERTAIN'
        if abs(yaw) >= self.yaw_threshold and abs(yaw) >= abs(pitch):
            return POSITIVE_YAW_LABEL if yaw > 0 else NEGATIVE_YAW_LABEL
        if abs(pitch) >= self.pitch_threshold:
            return POSITIVE_PITCH_LABEL if pitch > 0 else NEGATIVE_PITCH_LABEL
        return 'FORWARD'

    def update_distraction(self, position, face_present):
        if not face_present:
            self.face_lost_frames += 1
            return self.distracted and self.face_lost_frames <= self.lost_grace
        self.face_lost_frames = 0
        turned = position in {'LOOKING LEFT',
                              'LOOKING RIGHT', 'LOOKING UP', 'LOOKING DOWN'}
        if turned:
            self.turned_frames += 1
            self.forward_frames = 0
        else:
            self.forward_frames += 1
            self.turned_frames = 0
        if self.turned_frames >= self.distraction_frames:
            self.distracted = True
        elif self.distracted and self.forward_frames >= self.recovery_frames:
            self.distracted = False
        return self.distracted


def image_files(folder):
    return sorted(path for path in folder.rglob('*') if path.suffix.lower() in IMAGE_EXTENSIONS)


def select_images(files, max_images):
    if max_images <= 0 or len(files) <= max_images:
        return files
    indices = np.linspace(0, len(files) - 1, max_images, dtype=np.int64)
    return [files[index] for index in indices]


def load_dataset(data_root, face_model, max_images_per_class):
    detector = FaceLandmarkDetector()
    features, labels = [], []
    try:
        for label, folder_name in CLASS_FOLDERS.items():
            folder = Path(data_root) / folder_name
            all_files = image_files(folder) if folder.exists() else []
            files = select_images(all_files, max_images_per_class)
            print(f'{label}: using {len(files)} of {len(all_files)} images')
            for image_path in files:
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                landmarks = detector.extract(image)
                if landmarks is None:
                    continue
                vector = feature_vector(
                    landmarks, image.shape[1], image.shape[0])
                if vector is not None:
                    features.append(vector)
                    labels.append(label)
    finally:
        detector.close()
    if len(set(labels)) < 2:
        raise ValueError(
            'At least two classes with detectable faces are required.')
    return np.asarray(features, dtype=np.float32), np.asarray(labels)


def train(data_root, model_out, labels_out, metrics_out, face_model, max_images_per_class):
    X, labels = load_dataset(data_root, face_model, max_images_per_class)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    classifier = RandomForestClassifier(
        n_estimators=300, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    metrics = {
        'feature_version': FEATURE_VERSION,
        'accuracy': float(accuracy_score(y_test, predictions)),
        'classes': encoder.classes_.tolist(),
        'test_samples': int(len(y_test)),
        'classification_report': classification_report(y_test, predictions, target_names=encoder.classes_, output_dict=True, zero_division=0),
    }
    with open(model_out, 'wb') as file:
        pickle.dump(classifier, file)
    with open(labels_out, 'wb') as file:
        pickle.dump(encoder, file)
    Path(metrics_out).write_text(json.dumps(
        metrics, indent=2), encoding='utf-8')
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(classification_report(y_test, predictions,
          target_names=encoder.classes_, zero_division=0))
    print(f'Saved: {model_out}, {labels_out}, {metrics_out}')


def stable_ml_label(classifier, encoder, vector, history, confidence):
    probabilities = classifier.predict_proba([vector])[0]
    index = int(np.argmax(probabilities))
    probability = float(probabilities[index])
    label = str(encoder.inverse_transform([index])[0])
    history.append(label if probability >= confidence else 'normal')
    return Counter(history).most_common(1)[0][0], probability


def draw_text(frame, lines):
    y = 28
    for text, color, scale in lines:
        cv2.putText(frame, text, (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
        y += 27


def live(model_path, labels_path, face_model, camera, confidence, pose_yaw, pose_pitch,
         mar_threshold, eye_close_threshold, eye_open_threshold,
         fatigue_seconds, debug, width, height):
    with open(model_path, 'rb') as file:
        classifier = pickle.load(file)
    with open(labels_path, 'rb') as file:
        encoder = pickle.load(file)
    if not hasattr(classifier, 'n_features_in_'):
        raise ValueError(
            'Saved classifier metadata is missing. Retrain with --mode train.')
    detector = FaceLandmarkDetector()
    pose_estimator = HeadPoseEstimator()
    mouth = MouthAnalyzer(mar_threshold=mar_threshold)
    eyes = EyeAnalyzer(close_threshold=eye_close_threshold,
                       open_threshold=eye_open_threshold,
                       fatigue_seconds=fatigue_seconds)
    state = BehaviorStateMachine(
        yaw_threshold=pose_yaw, pitch_threshold=pose_pitch)
    prediction_history = deque(maxlen=9)
    capture = cv2.VideoCapture(camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not capture.isOpened():
        detector.close()
        raise RuntimeError(f'Cannot open camera index {camera}')
    previous_time = time.perf_counter()
    fps = 0.0
    last_pose = None
    last_position = 'HEAD POSITION UNCERTAIN'
    left_ear = None
    right_ear = None
    average_ear = None
    eye_state = 'UNKNOWN'
    blink_rate = 0.0
    eyes_closed_duration = 0.0
    fatigue_alert = False
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            landmarks = detector.extract(frame)
            now = time.monotonic()
            behavior = 'NORMAL'
            probability = 0.0
            if landmarks is None:
                left_ear = None
                right_ear = None
                average_ear = None
                eye_state = 'EYE STATE UNCERTAIN'
                blink_rate = eyes.blink_rate(now)
                eyes_closed_duration = eyes._closed_duration(now)
                fatigue_alert = eyes.fatigue_alert
                state.update_distraction(last_position, False)
                if state.face_lost_frames > state.lost_grace:
                    last_position = 'HEAD POSITION UNCERTAIN'
                    last_pose = None
                    prediction_history.clear()
            else:
                (left_ear, right_ear, average_ear, eye_state, blink_rate, _,
                 eyes_closed_duration, fatigue_alert) = eyes.update(
                    landmarks, now)
                _, rotation = corrected_head_pose(
                    landmarks, frame.shape[1], frame.shape[0])
                smoothed_pose = pose_estimator.update(rotation)
                if pose_estimator.calibrating:
                    behavior = 'CALIBRATING'
                elif smoothed_pose is not None:
                    last_pose = smoothed_pose
                    last_position = state.position(
                        float(smoothed_pose[0]), float(smoothed_pose[1]))
                    mar = mouth_features(landmarks)
                    yawn_event, talking, _ = mouth.update(mar, now)
                    distracted = state.update_distraction(last_position, True)
                    vector = feature_vector(
                        landmarks, frame.shape[1], frame.shape[0])
                    ml_label, probability = stable_ml_label(
                        classifier, encoder, vector, prediction_history, confidence)
                    if fatigue_alert:
                        behavior = 'FATIGUE ALERT'
                    elif distracted:
                        behavior = 'DISTRACTED'
                    elif yawn_event or mouth.yawn_active:
                        behavior = 'YAWNING'
                    elif talking:
                        behavior = 'TALKING'
                    elif ml_label in {'talking', 'yawning'} and probability >= confidence:
                        behavior = ml_label.upper()
            calibration_text = (f'CALIBRATING... {len(pose_estimator.calibration_rotations)}/{pose_estimator.calibration_frames}'
                                if pose_estimator.calibrating else 'Calibration complete')
            current_time = time.perf_counter()
            instant_fps = 1.0 / max(current_time - previous_time, 1e-6)
            fps = instant_fps if fps == 0.0 else 0.1 * instant_fps + 0.9 * fps
            previous_time = current_time
            color = (0, 220, 255) if behavior == 'NORMAL' else (0, 80, 255)
            lines = [('Driver Monitoring', (255, 255, 255), 0.75),
                     (f'Behavior: {behavior}', color, 0.68),
                     (f'Confidence: {probability:.2f}', (220, 220, 220), 0.55)]
            if last_pose is not None:
                lines.extend([
                    (f'Pitch: {last_pose[0]:+5.1f} deg  Yaw: {last_pose[1]:+5.1f} deg',
                     (120, 255, 120), 0.55),
                    (f'Roll:  {last_pose[2]:+5.1f} deg  Position: {last_position}',
                     (120, 255, 120), 0.55),
                ])
            else:
                lines.append(
                    (f'Position: {last_position}', (120, 255, 120), 0.55))

            def ear_text(value):
                return '--' if value is None else f'{value:.3f}'
            lines.extend([
                (f'Left EAR: {ear_text(left_ear)}  Right EAR: {ear_text(right_ear)}',
                 (220, 220, 120), 0.52),
                (f'Avg EAR: {ear_text(average_ear)}  Eyes: {eye_state}',
                 (220, 220, 120), 0.52),
                (f'Eyes closed: {eyes_closed_duration:.2f} s',
                 (220, 220, 120), 0.52),
                (f'Blinks (10s): {len(eyes.blink_times)}  Rate: {blink_rate:.1f}/min',
                 (220, 220, 120), 0.52),
            ])
            lines.append(
                (f'FPS: {fps:.1f}   {calibration_text}', (220, 220, 220), 0.5))
            if debug:
                mar_value = mouth.mar_history[-1] if mouth.mar_history else 0.0
                lines.extend([
                    (f'DEBUG MAR: {mar_value:.3f} open_frames: {mouth.open_frames}',
                     (255, 180, 80), 0.48),
                    (f'DEBUG fatigue: {fatigue_alert} threshold: {fatigue_seconds:.2f}s',
                     (255, 180, 80), 0.48),
                    (f'DEBUG turned: {state.turned_frames} recovery: {state.forward_frames}', (
                        255, 180, 80), 0.48),
                    (f'DEBUG thresholds yaw/pitch: {pose_yaw:.1f}/{pose_pitch:.1f}',
                     (255, 180, 80), 0.48),
                ])
            cv2.rectangle(
                frame, (0, 0), (frame.shape[1], 340 if debug else 259), (20, 20, 20), -1)
            draw_text(frame, lines)
            if debug and landmarks is not None:
                for eye_indices in (LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS):
                    points = eye_landmark_pixels(
                        landmarks, eye_indices, frame.shape[1], frame.shape[0])
                    for point in points:
                        cv2.circle(frame, point, 2, (255, 255, 0), -1)
            cv2.imshow('Driver behavior and head pose', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('c') and landmarks is not None:
                pose_estimator.reset_calibration()
                state.distracted = False
                prediction_history.clear()
    finally:
        capture.release()
        cv2.destroyAllWindows()
        detector.close()


def main():
    parser = argparse.ArgumentParser(
        description='Real-time yawning, talking, distraction, and head-pose monitor.')
    parser.add_argument('--mode', choices=('train', 'live'), default='live')
    parser.add_argument('--data-root', default='.',
                        help='Driver_pr folder containing the four class folders')
    parser.add_argument('--model', default='driver_behavior_model.pkl')
    parser.add_argument('--labels', default='driver_behavior_labels.pkl')
    parser.add_argument('--metrics', default='driver_behavior_metrics.json')
    parser.add_argument('--face-landmarker-model',
                        default='face_landmarker.task')
    parser.add_argument('--max-images-per-class', type=int, default=1000,
                        help='Uniformly spaced images per class; use 0 for all')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--confidence', type=float, default=0.65)
    parser.add_argument('--pose-yaw', type=float, default=35.0)
    parser.add_argument('--pose-pitch', type=float, default=20.0)
    parser.add_argument('--mar-threshold', type=float, default=0.62)
    parser.add_argument('--eye-close-threshold', type=float, default=0.21,
                        help='EAR value used only to identify closed eyes')
    parser.add_argument('--eye-open-threshold', type=float, default=0.24,
                        help='EAR value required to reopen eyes after closure')
    parser.add_argument('--fatigue-seconds', type=float, default=1.5,
                        help='Project fatigue alert threshold for continuous eye closure')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args.data_root, args.model, args.labels, args.metrics,
              args.face_landmarker_model, args.max_images_per_class)
    else:
        live(args.model, args.labels, args.face_landmarker_model, args.camera, args.confidence,
             args.pose_yaw, args.pose_pitch, args.mar_threshold,
             args.eye_close_threshold, args.eye_open_threshold,
             args.fatigue_seconds, args.debug, args.width, args.height)


if __name__ == '__main__':
    main()
