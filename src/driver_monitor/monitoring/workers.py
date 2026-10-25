from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Mapping

from driver_monitor.detection.emotion import (
    DeepFaceEmotionAdapter,
    EmotionThresholdPolicy,
    classify_scores,
)
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
            packet = self.frame_buffer.get_newest(last_frame_id, self.stop_event)
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
