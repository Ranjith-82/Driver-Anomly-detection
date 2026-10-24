"""Standalone webcam-based eye gaze detection using MediaPipe Face Mesh."""

import argparse
import os
import platform
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np


# The horizontal fallback prevents noisy calibration from making that
# classifier overly sensitive. Calibrated thresholds expand when needed.
GAZE_HORIZONTAL_THRESHOLD = 0.05
GAZE_DEBOUNCE_FRAMES = 5
GAZE_DEBOUNCE_VOTES = 4
GAZE_DISTRACTION_SECONDS = 1.0
CALIBRATION_CAPTURE_SECONDS = 1.5
CALIBRATION_MIN_SAMPLES = 8
DISPLAY_PANEL_WIDTH = 460

GAZE_DIRECTIONS = ("FORWARD", "LEFT", "RIGHT", "UP", "DOWN")

# Iris center landmark plus the eye corners and upper/lower lid points.
# The indices are from MediaPipe's refined 478-point face landmark model.
EYE_LANDMARKS = {
    "screen_left": {"corners": (33, 133), "upper": (159, 158),
                    "lower": (145, 153), "iris": 468,
                    "outline": (33, 160, 158, 133, 153, 144)},
    "screen_right": {"corners": (362, 263), "upper": (386, 385),
                     "lower": (374, 380), "iris": 473,
                     "outline": (362, 387, 385, 263, 380, 373)},
}

FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
    365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
    132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)


@dataclass(frozen=True)
class GazePosition:
    x: float
    y: float


@dataclass(frozen=True)
class GazeThresholds:
    left: float
    right: float
    up: float
    down: float
    left_is_lower: bool = True
    up_is_lower: bool = True
    down_is_lower: bool = False


def initialize_camera(camera_index: int) -> cv2.VideoCapture:
    """Open the requested camera, using DirectShow first on Windows."""
    if platform.system() == "Windows":
        camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return camera
        camera.release()

    camera = cv2.VideoCapture(camera_index)
    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def initialize_mediapipe(model_path: Optional[str] = None) -> Tuple[str, object]:
    """Create a refined Face Mesh detector, with a Tasks API fallback."""
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        detector = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return "solutions", detector

    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
    except ImportError as error:
        raise RuntimeError(
            "This MediaPipe installation provides neither Face Mesh nor "
            "the Tasks API. Install a MediaPipe version with face landmarks."
        ) from error

    if model_path is None:
        model_path = os.path.join(os.path.dirname(
            __file__), "face_landmarker.task")
    if not os.path.isfile(model_path):
        raise RuntimeError(
            "The MediaPipe Tasks API needs face_landmarker.task. Place that "
            "model beside gaze_detection.py or pass --model-path."
        )

    options = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    return "tasks", vision.FaceLandmarker.create_from_options(options)


def extract_face_landmarks(
    frame: np.ndarray, detector: Tuple[str, object]
) -> Optional[List[Tuple[float, float]]]:
    """Return normalized face landmarks, or None when no face is visible."""
    mode, handle = detector
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if mode == "solutions":
        results = handle.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None
        return [(point.x, point.y) for point in results.multi_face_landmarks[0].landmark]

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = handle.detect(image)
    if not results.face_landmarks:
        return None
    return [(point.x, point.y) for point in results.face_landmarks[0]]


def extract_eye_landmarks(
    landmarks: Sequence[Tuple[float, float]], image_width: int, image_height: int
) -> Optional[Dict[str, Dict[str, Tuple[float, float]]]]:
    """Extract eye corners, lid positions, and iris centers in pixel units."""
    if len(landmarks) <= 473:
        return None

    eyes = {}
    for name, indices in EYE_LANDMARKS.items():
        def to_pixel(index): return (
            landmarks[index][0] * image_width,
            landmarks[index][1] * image_height,
        )
        eyes[name] = {
            "corner_a": to_pixel(indices["corners"][0]),
            "corner_b": to_pixel(indices["corners"][1]),
            "upper": tuple(np.mean([to_pixel(i) for i in indices["upper"]], axis=0)),
            "lower": tuple(np.mean([to_pixel(i) for i in indices["lower"]], axis=0)),
            "iris": to_pixel(indices["iris"]),
        }
    return eyes


def calculate_gaze_position(
    eye_landmarks: Dict[str, Dict[str, Tuple[float, float]]]
) -> Optional[GazePosition]:
    """Normalize each iris position within its eye and average both eyes."""
    normalized_positions = []
    for eye in eye_landmarks.values():
        corner_a, corner_b = eye["corner_a"], eye["corner_b"]
        eye_width = abs(corner_b[0] - corner_a[0])
        if eye_width < 1.0:
            continue

        # Ratios are relative to the eye box, so face size and camera distance
        # have much less influence than raw pixel coordinates.
        x = (eye["iris"][0] - min(corner_a[0], corner_b[0])) / eye_width
        corner_slope = (corner_b[1] - corner_a[1]) / (
            corner_b[0] - corner_a[0]
        )
        canthus_line_y = corner_a[1] + (
            eye["iris"][0] - corner_a[0]
        ) * corner_slope
        y = (eye["iris"][1] - canthus_line_y) / eye_width
        normalized_positions.append((float(x), float(y)))

    if not normalized_positions:
        return None
    return GazePosition(
        x=float(np.mean([position[0] for position in normalized_positions])),
        y=float(np.mean([position[1] for position in normalized_positions])),
    )


def _calibrated_boundary(
    forward: float, observed: float, lower: bool, minimum_offset: float
) -> float:
    """Place a threshold halfway toward the calibrated direction, with a floor."""
    if lower:
        movement = forward - observed
        offset = max(minimum_offset, movement /
                     2.0) if movement > 0 else minimum_offset
        return max(0.0, forward - offset)
    movement = observed - forward
    offset = max(minimum_offset, movement /
                 2.0) if movement > 0 else minimum_offset
    return min(1.0, forward + offset)


def calibrate_gaze_thresholds(
    observations: Dict[str, GazePosition]
) -> GazeThresholds:
    """Derive personalized direction boundaries from the five calibration poses."""
    forward = observations["FORWARD"]
    left_is_lower = observations["LEFT"].x <= observations["RIGHT"].x
    up_is_lower = observations["UP"].y < forward.y
    down_is_lower = observations["DOWN"].y < forward.y
    return GazeThresholds(
        left=_calibrated_boundary(
            forward.x, observations["LEFT"].x, left_is_lower,
            GAZE_HORIZONTAL_THRESHOLD,
        ),
        right=_calibrated_boundary(
            forward.x, observations["RIGHT"].x, not left_is_lower,
            GAZE_HORIZONTAL_THRESHOLD,
        ),
        up=(forward.y + observations["UP"].y) / 2.0,
        down=(forward.y + observations["DOWN"].y) / 2.0,
        left_is_lower=left_is_lower,
        up_is_lower=up_is_lower,
        down_is_lower=down_is_lower,
    )


def classify_gaze(position: GazePosition, thresholds: GazeThresholds) -> str:
    """Classify the strongest threshold crossing, or return FORWARD."""
    crossings = []
    if (position.x < thresholds.left if thresholds.left_is_lower
            else position.x > thresholds.left):
        distance = (thresholds.left - position.x if thresholds.left_is_lower
                    else position.x - thresholds.left)
        crossings.append((distance, "LEFT"))
    if (position.x > thresholds.right if thresholds.left_is_lower
            else position.x < thresholds.right):
        distance = (position.x - thresholds.right if thresholds.left_is_lower
                    else thresholds.right - position.x)
        crossings.append((distance, "RIGHT"))
    if (position.y < thresholds.up if thresholds.up_is_lower
            else position.y > thresholds.up):
        distance = (thresholds.up - position.y if thresholds.up_is_lower
                    else position.y - thresholds.up)
        crossings.append((distance, "UP"))
    if (position.y < thresholds.down if thresholds.down_is_lower
            else position.y > thresholds.down):
        distance = (thresholds.down - position.y if thresholds.down_is_lower
                    else position.y - thresholds.down)
        crossings.append((distance, "DOWN"))
    return max(crossings)[1] if crossings else "FORWARD"


class CalibrationSession:
    """Collect one short sample for each requested gaze direction."""

    def __init__(self) -> None:
        self.index = 0
        self.capture_started: Optional[float] = None
        self.samples: List[GazePosition] = []
        self.observations: Dict[str, GazePosition] = {}
        self.retry_message = ""
        self.captured = False

    @property
    def complete(self) -> bool:
        return self.index >= len(GAZE_DIRECTIONS)

    @property
    def current_direction(self) -> Optional[str]:
        return None if self.complete else GAZE_DIRECTIONS[self.index]

    def start_capture(self, now: float) -> None:
        if self.captured:
            self.index += 1
            self.captured = False
        elif not self.complete and self.capture_started is None:
            self.capture_started = now
            self.samples.clear()
            self.retry_message = ""

    def update(self, now: float, position: Optional[GazePosition]) -> None:
        if self.capture_started is None or self.complete:
            return
        if position is not None:
            self.samples.append(position)
        if now - self.capture_started < CALIBRATION_CAPTURE_SECONDS:
            return

        if len(self.samples) < CALIBRATION_MIN_SAMPLES:
            self.capture_started = None
            self.samples.clear()
            self.retry_message = "Not enough eye samples; press SPACE to retry."
            return

        direction = self.current_direction
        self.observations[direction] = GazePosition(
            x=float(np.median([sample.x for sample in self.samples])),
            y=float(np.median([sample.y for sample in self.samples])),
        )
        self.captured = True
        self.capture_started = None
        self.samples.clear()


class GazeSmoother:
    """Require repeated votes before changing the displayed gaze direction."""

    def __init__(self) -> None:
        self.history: Deque[str] = deque(maxlen=GAZE_DEBOUNCE_FRAMES)
        self.current = "FORWARD"

    def update(self, candidate: str) -> str:
        self.history.append(candidate)
        direction, votes = Counter(self.history).most_common(1)[0]
        if votes >= GAZE_DEBOUNCE_VOTES:
            self.current = direction
        return self.current


class DistractionTimer:
    """Raise an alert only after non-forward gaze persists for the set duration."""

    def __init__(self, duration: float) -> None:
        self.duration = duration
        self.away_since: Optional[float] = None

    def update(self, direction: Optional[str], now: float) -> Tuple[float, bool]:
        if direction is None or direction == "FORWARD":
            self.away_since = None
            return 0.0, False
        if self.away_since is None:
            self.away_since = now
        elapsed = now - self.away_since
        return elapsed, elapsed >= self.duration


def draw_landmarks(
    frame: np.ndarray, landmarks: Sequence[Tuple[float, float]]
) -> None:
    """Draw the face outline, eye contours, and iris centers."""
    height, width = frame.shape[:2]

    def pixel(index: int) -> Tuple[int, int]:
        return int(landmarks[index][0] * width), int(landmarks[index][1] * height)

    face_points = np.array([pixel(index)
                           for index in FACE_OVAL], dtype=np.int32)
    cv2.polylines(frame, [face_points], True, (80, 190, 255), 1, cv2.LINE_AA)

    for indices in EYE_LANDMARKS.values():
        outline = np.array([pixel(index)
                           for index in indices["outline"]], dtype=np.int32)
        cv2.polylines(frame, [outline], True, (80, 255, 120), 1, cv2.LINE_AA)
        iris = pixel(indices["iris"])
        cv2.circle(frame, iris, 3, (0, 220, 255), -1, cv2.LINE_AA)


def _draw_panel(frame: np.ndarray, height: int) -> Tuple[int, int, int]:
    """Draw an opaque, high-contrast panel and return its origin and width."""
    frame_height, frame_width = frame.shape[:2]
    x, y = 12, 12
    width = min(430, frame_width - 24)
    height = min(height, frame_height - 24)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (18, 25, 30), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (82, 112, 115), 1)
    cv2.rectangle(frame, (x, y), (x + width, y + 5), (64, 190, 175), -1)
    return x, y, width


def _panel_text(
    frame: np.ndarray, text: str, x: int, y: int, scale: float,
    color: Tuple[int, int, int] = (240, 245, 242), thickness: int = 1,
) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_gaze_panel(
    frame: np.ndarray,
    direction: str,
    position: Optional[GazePosition],
    alert_elapsed: float,
    distracted: bool,
    fps: float,
    debug: bool = False,
    raw_direction: str = "UNKNOWN",
    thresholds: Optional[GazeThresholds] = None,
) -> None:
    """Draw the normal gaze result, with detailed values only in debug mode."""
    x, y, width = _draw_panel(frame, 452 if debug else 356)
    left = x + 18
    _panel_text(frame, "DRIVER EYE GAZE", left, y + 36, 0.72,
                (106, 222, 202), 2)
    _panel_text(frame, "GAZE: " + direction, left, y + 91, 1.28,
                (85, 225, 165) if direction == "FORWARD" else (100, 210, 255), 2)

    gaze_x = "{:.3f}".format(position.x) if position else "--"
    gaze_y = "{:.3f}".format(position.y) if position else "--"
    value_color = (240, 245, 242)
    _panel_text(frame, "GAZE X: " + gaze_x, left, y + 136, 0.57, value_color)
    _panel_text(frame, "GAZE Y: " + gaze_y, left, y + 168, 0.57, value_color)
    _panel_text(frame, "DISTRACTION: " + ("YES" if distracted else "NO"),
                left, y + 200, 0.57,
                (100, 120, 255) if distracted else (150, 225, 180), 2)
    _panel_text(frame, "DURATION: {:.1f} s".format(alert_elapsed),
                left, y + 232, 0.57, value_color)
    _panel_text(frame, "FPS: {:.1f}".format(fps), left, y + 264, 0.57,
                (185, 205, 210))

    if not debug:
        _panel_text(
            frame,
            "UP THRESHOLD: {:.3f}".format(thresholds.up)
            if thresholds else "UP THRESHOLD: --",
            left, y + 300, 0.53, value_color,
        )
        _panel_text(
            frame,
            "DOWN THRESHOLD: {:.3f}".format(thresholds.down)
            if thresholds else "DOWN THRESHOLD: --",
            left, y + 330, 0.53, value_color,
        )

    if debug:
        cv2.line(frame, (left, y + 281), (x + width - 18, y + 281),
                 (82, 112, 115), 1, cv2.LINE_AA)
        values = [
            "Current threshold LEFT: {:.3f}".format(
                thresholds.left) if thresholds else "Current threshold LEFT: --",
            "Current threshold RIGHT: {:.3f}".format(
                thresholds.right) if thresholds else "Current threshold RIGHT: --",
            "UP THRESHOLD: {:.3f}".format(
                thresholds.up) if thresholds else "UP THRESHOLD: --",
            "DOWN THRESHOLD: {:.3f}".format(
                thresholds.down) if thresholds else "DOWN THRESHOLD: --",
            "Raw direction: " + raw_direction,
            "Smoothed direction: " + direction,
        ]
        for index, text in enumerate(values):
            _panel_text(frame, text, left, y + 304 + index * 17, 0.43,
                        (205, 220, 220))


def draw_calibration_panel(
    frame: np.ndarray, calibration: CalibrationSession, now: float
) -> None:
    """Show one calibration instruction at a time in its own panel."""
    x, y, _ = _draw_panel(frame, 300)
    left = x + 18
    _panel_text(frame, "GAZE CALIBRATION", left, y + 38, 0.72,
                (106, 222, 202), 2)
    step = min(calibration.index + 1, len(GAZE_DIRECTIONS))
    _panel_text(frame, "STEP {} / {}".format(step, len(GAZE_DIRECTIONS)),
                left, y + 78, 0.58, (185, 205, 210))
    direction = calibration.current_direction or "COMPLETE"
    _panel_text(frame, "LOOK " + direction, left, y + 126, 0.95,
                (100, 210, 255), 2)

    if calibration.captured:
        _panel_text(frame, "SAMPLE CAPTURED", left, y + 174, 0.62,
                    (150, 225, 180), 2)
        instruction = "Press SPACE for next direction."
    elif calibration.capture_started is not None:
        elapsed = now - calibration.capture_started
        remaining = max(0.0, CALIBRATION_CAPTURE_SECONDS - elapsed)
        _panel_text(frame, "Hold your eyes in this direction.", left,
                    y + 174, 0.49)
        instruction = "Capturing: {:.1f} s".format(remaining)
    else:
        _panel_text(frame, "Hold your eyes in this direction.", left,
                    y + 174, 0.49)
        instruction = "Press SPACE to capture."
    _panel_text(frame, instruction, left, y + 210, 0.49,
                (220, 230, 226))
    if calibration.retry_message:
        _panel_text(frame, calibration.retry_message, left, y + 244, 0.43,
                    (120, 170, 255))


def run(
    camera_index: int,
    distraction_seconds: float,
    model_path: Optional[str],
    debug: bool = False,
) -> None:
    camera = initialize_camera(camera_index)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam {}.".format(camera_index))

    detector = initialize_mediapipe(model_path)
    calibration = CalibrationSession()
    smoother = GazeSmoother()
    distraction_timer = DistractionTimer(distraction_seconds)
    previous_frame_time = time.perf_counter()
    fps = 0.0
    thresholds: Optional[GazeThresholds] = None

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a frame from the webcam.")
                break

            # Mirror the preview so directions match the driver's on-screen view.
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            landmarks = extract_face_landmarks(frame, detector)
            eye_landmarks = (
                extract_eye_landmarks(landmarks, width, height)
                if landmarks is not None else None
            )
            position = calculate_gaze_position(
                eye_landmarks) if eye_landmarks else None
            now = time.monotonic()

            elapsed_frame = now - previous_frame_time
            if elapsed_frame > 0:
                instant_fps = 1.0 / elapsed_frame
                fps = instant_fps if fps == 0 else 0.85 * fps + 0.15 * instant_fps
            previous_frame_time = now

            if landmarks is not None:
                draw_landmarks(frame, landmarks)

            display_frame = np.zeros(
                (height, width + DISPLAY_PANEL_WIDTH, 3), dtype=frame.dtype
            )
            display_frame[:, DISPLAY_PANEL_WIDTH:] = frame
            cv2.line(
                display_frame,
                (DISPLAY_PANEL_WIDTH - 1, 0),
                (DISPLAY_PANEL_WIDTH - 1, height),
                (82, 112, 115),
                1,
                cv2.LINE_AA,
            )

            if not calibration.complete:
                calibration.update(now, position)
                draw_calibration_panel(display_frame, calibration, now)
            else:
                if thresholds is None:
                    thresholds = calibrate_gaze_thresholds(
                        calibration.observations)
                    print(
                        "Calibration values (normalized gaze coordinates):"
                    )
                    for direction in GAZE_DIRECTIONS:
                        observation = calibration.observations[direction]
                        print(
                            "{} -> X: {:.4f} Y: {:.4f}".format(
                                direction, observation.x, observation.y
                            )
                        )
                    print(
                        "UP is numerically {} than FORWARD; DOWN is "
                        "numerically {} than FORWARD.".format(
                            "lower" if thresholds.up_is_lower else "higher",
                            "lower" if thresholds.down_is_lower else "higher",
                        )
                    )
                    print(
                        "UP THRESHOLD: {:.4f}; DOWN THRESHOLD: {:.4f}".format(
                            thresholds.up, thresholds.down
                        )
                    )

                raw_direction = "UNKNOWN"
                if position is None:
                    direction = None
                    alert_elapsed, distracted = distraction_timer.update(
                        None, now)
                else:
                    raw_direction = classify_gaze(position, thresholds)
                    direction = smoother.update(raw_direction)
                    alert_elapsed, distracted = distraction_timer.update(
                        direction, now)

                draw_gaze_panel(
                    display_frame, direction or "UNKNOWN", position, alert_elapsed,
                    distracted, fps, debug, raw_direction, thresholds,
                )

            cv2.imshow("Driver Eye Gaze Detection", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == 32 and not calibration.complete:
                calibration.start_capture(now)
    finally:
        camera.release()
        mode, handle = detector
        if mode == "solutions":
            handle.close()
        else:
            handle.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone real-time driver eye gaze detection"
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam device index")
    parser.add_argument(
        "--distraction-seconds", type=float, default=GAZE_DISTRACTION_SECONDS,
        help="Seconds of sustained non-forward gaze before raising an alert",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Optional FaceLandmarker model path for the MediaPipe Tasks API",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Show raw gaze, calibrated thresholds, and direction decisions",
    )
    args = parser.parse_args()
    if args.distraction_seconds <= 0:
        parser.error("--distraction-seconds must be greater than zero")

    try:
        run(args.camera, args.distraction_seconds, args.model_path, args.debug)
    except RuntimeError as error:
        parser.exit(1, "ERROR: {}\n".format(error))


if __name__ == "__main__":
    main()
