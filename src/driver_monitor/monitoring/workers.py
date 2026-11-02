from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from driver_monitor.detection.emotion import (
    DeepFaceEmotionAdapter,
    EmotionThresholdPolicy,
    classify_scores,
)
from driver_monitor.detection.svm_emotion import SVMEmotionAnalyzer
from driver_monitor.models import DetectorStatus, EmotionEstimate, EmotionLabel


class EmotionWorker(threading.Thread):
    """Samples the shared latest-frame buffer without blocking other detectors."""

    def __init__(
        self,
        frame_buffer: object,
        stop_event: threading.Event,
        read_face_state: Callable[[], Mapping[str, object]],
        publish: Callable[[dict[str, object]], None],
        thresholds_path: Path,
        interval: float = 1.0,
    ) -> None:
        super().__init__(name="emotion-worker", daemon=True)
        if interval <= 0:
            raise ValueError("Emotion sampling interval must be positive")
        self.frame_buffer = frame_buffer
        self.stop_event = stop_event
        self.read_face_state = read_face_state
        self.publish = publish
        self.thresholds_path = thresholds_path
        self.interval = interval
        self.last_error = ""

    @staticmethod
    def _crop_face(frame: object, bounds: object):
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 4:
            return None
        height, width = frame.shape[:2]
        top, right, bottom, left = (int(value) for value in bounds)
        pad_x = max(1, (right - left) // 8)
        pad_y = max(1, (bottom - top) // 8)
        x1, x2 = max(0, left - pad_x), min(width, right + pad_x)
        y1, y2 = max(0, top - pad_y), min(height, bottom + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _publish(self, estimate: EmotionEstimate, error: str = "") -> None:
        self.publish({
            "emotion_label": estimate.label.value,
            "emotion_status": estimate.status.value,
            "emotion_frame_id": estimate.frame_id,
            "emotion_timestamp": estimate.captured_at,
            "emotion_error": error,
        })

    def run(self) -> None:
        if not self.thresholds_path.is_file():
            self._publish(
                EmotionEstimate(EmotionLabel.UNCERTAIN, DetectorStatus.UNCERTAIN),
                "Emotion classes are disabled until the evaluation gate passes.",
            )
            return
        try:
            policy = EmotionThresholdPolicy.load(self.thresholds_path)
            if not policy.thresholds:
                self._publish(
                    EmotionEstimate(
                        EmotionLabel.UNCERTAIN, DetectorStatus.UNCERTAIN
                    ),
                    "No emotion class passed the evaluation release gate.",
                )
                return
            adapter = DeepFaceEmotionAdapter()
            adapter.warm_up()
        except Exception as error:
            self.last_error = str(error)
            self._publish(
                EmotionEstimate(EmotionLabel.UNCERTAIN, DetectorStatus.UNAVAILABLE),
                self.last_error,
            )
            return

        last_frame_id = -1
        next_run = 0.0
        while not self.stop_event.is_set():
            packet = self.frame_buffer.get_newest(
                last_frame_id, self.stop_event
            )
            if packet is None:
                continue
            last_frame_id = packet.frame_id
            now = time.monotonic()
            if now < next_run:
                continue
            next_run = now + self.interval
            face_state = self.read_face_state()
            face_frame_id = face_state.get("behavior_frame_id", -1)
            bounds = face_state.get("face_bounds")
            if (
                face_state.get("face_present") is not True
                or not isinstance(face_frame_id, int)
                or abs(packet.frame_id - face_frame_id) > 5
            ):
                self._publish(EmotionEstimate(
                    EmotionLabel.UNCERTAIN,
                    DetectorStatus.NO_FACE,
                    frame_id=packet.frame_id,
                    captured_at=packet.timestamp,
                    completed_at=time.monotonic(),
                ))
                continue
            crop = self._crop_face(packet.frame, bounds)
            if crop is None or crop.size == 0:
                self._publish(EmotionEstimate(
                    EmotionLabel.UNCERTAIN,
                    DetectorStatus.NO_FACE,
                    frame_id=packet.frame_id,
                    captured_at=packet.timestamp,
                    completed_at=time.monotonic(),
                ))
                continue
            try:
                scores = adapter.analyze_face_crop(crop)
                estimate = classify_scores(
                    scores,
                    policy.thresholds,
                    frame_id=packet.frame_id,
                    captured_at=packet.timestamp,
                    completed_at=time.monotonic(),
                )
                self._publish(estimate)
            except Exception as error:
                self.last_error = str(error)
                self._publish(
                    EmotionEstimate(
                        EmotionLabel.UNCERTAIN,
                        DetectorStatus.UNAVAILABLE,
                        frame_id=packet.frame_id,
                        captured_at=packet.timestamp,
                        completed_at=time.monotonic(),
                    ),
                    self.last_error,
                )


class SVMEmotionWorker(threading.Thread):
    """Samples only a fresh face spatially associated with the selected driver."""

    MAX_FRAME_AGE_SECONDS = 2.0
    MAX_FACE_FRAME_DELTA = 5
    MIN_FACE_OVERLAP = 0.3

    def __init__(
        self,
        frame_buffer: object,
        stop_event: threading.Event,
        read_face_state: Callable[[], Mapping[str, object]],
        publish: Callable[[dict[str, object]], None],
        model_dir: Path,
        interval: float = 1.0,
    ) -> None:
        super().__init__(name="svm-emotion-worker", daemon=True)
        if interval <= 0:
            raise ValueError("Emotion sampling interval must be positive")
        self.frame_buffer = frame_buffer
        self.stop_event = stop_event
        self.read_face_state = read_face_state
        self.publish = publish
        self.model_dir = model_dir
        self.interval = interval
        self.last_error = ""

    @staticmethod
    def _bounds(bounds: object, width: int, height: int):
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 4:
            return None
        try:
            top, right, bottom, left = (int(value) for value in bounds)
        except (TypeError, ValueError, OverflowError):
            return None
        top, bottom = max(0, top), min(height, bottom)
        left, right = max(0, left), min(width, right)
        if right <= left or bottom <= top:
            return None
        return top, right, bottom, left

    @classmethod
    def _same_face_region(
        cls,
        driver_bounds: object,
        current_bounds: object,
        width: int,
        height: int,
    ) -> bool:
        driver = cls._bounds(driver_bounds, width, height)
        current = cls._bounds(current_bounds, width, height)
        if driver is None or current is None:
            return False
        top = max(driver[0], current[0])
        right = min(driver[1], current[1])
        bottom = min(driver[2], current[2])
        left = max(driver[3], current[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        driver_area = (driver[1] - driver[3]) * (driver[2] - driver[0])
        current_area = (current[1] - current[3]) * (current[2] - current[0])
        return intersection / min(driver_area, current_area) >= cls.MIN_FACE_OVERLAP

    @staticmethod
    def _crop_face(frame: Any, bounds: object):
        height, width = frame.shape[:2]
        normalized = SVMEmotionWorker._bounds(bounds, width, height)
        if normalized is None:
            return None
        top, right, bottom, left = normalized
        pad_x = max(1, (right - left) // 8)
        pad_y = max(1, (bottom - top) // 8)
        x1, x2 = max(0, left - pad_x), min(width, right + pad_x)
        y1, y2 = max(0, top - pad_y), min(height, bottom + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _publish(
        self,
        label: str,
        status: str,
        packet: object | None = None,
        error: str = "",
    ) -> None:
        self.publish({
            "emotion_label": label,
            "emotion_status": status,
            "emotion_frame_id": getattr(packet, "frame_id", -1),
            "emotion_timestamp": getattr(packet, "timestamp", 0.0),
            "emotion_error": error,
        })

    def run(self) -> None:
        try:
            analyzer = SVMEmotionAnalyzer(self.model_dir)
        except Exception as error:
            self.last_error = str(error)
            self._publish("unknown", "unavailable", error=self.last_error)
            print(f"[EMOTION] ERROR: {self.last_error}")
            return

        last_frame_id = -1
        next_run = 0.0
        try:
            while not self.stop_event.is_set():
                packet = self.frame_buffer.get_newest(
                    last_frame_id, self.stop_event
                )
                if packet is None:
                    continue
                last_frame_id = packet.frame_id
                now = time.monotonic()
                if now < next_run:
                    continue
                next_run = now + self.interval
                if now - packet.timestamp > self.MAX_FRAME_AGE_SECONDS:
                    self._publish("unknown", "stale", packet)
                    continue

                face_state = self.read_face_state()
                driver_name = str(face_state.get("driver_name", "")).casefold()
                if (
                    face_state.get("driver_confirmed") is not True
                    or driver_name in {"", "unknown", "identifying..."}
                ):
                    self._publish("unknown", "unknown", packet)
                    continue
                face_frame_id = face_state.get("behavior_frame_id")
                if (
                    face_state.get("face_present") is not True
                    or not isinstance(face_frame_id, int)
                    or abs(packet.frame_id - face_frame_id)
                    > self.MAX_FACE_FRAME_DELTA
                ):
                    self._publish("unknown", "no_face", packet)
                    continue

                frame_height, frame_width = packet.frame.shape[:2]
                driver_bounds = face_state.get("recognition_location")
                current_bounds = face_state.get("face_bounds")
                if not self._same_face_region(
                    driver_bounds, current_bounds, frame_width, frame_height
                ):
                    self._publish("unknown", "unknown", packet)
                    continue
                crop = self._crop_face(packet.frame, current_bounds)
                if crop is None or crop.size == 0:
                    self._publish("unknown", "no_face", packet)
                    continue
                try:
                    label = analyzer.predict(crop)
                    if label is None:
                        self._publish("unknown", "no_face", packet)
                    else:
                        self._publish(label, "ready", packet)
                except Exception as error:
                    self.last_error = str(error)
                    self._publish(
                        "unknown", "unavailable", packet, self.last_error
                    )
                    print(f"[EMOTION] ERROR: {self.last_error}")
        finally:
            analyzer.close()
