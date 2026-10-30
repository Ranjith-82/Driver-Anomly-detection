import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from driver_monitor.config import ProjectPaths
from driver_monitor.detection.svm_emotion import (
    SUPPORTED_EMOTIONS,
    SVMEmotionAnalyzer,
)
from driver_monitor.models import FramePacket
from driver_monitor.monitoring.workers import SVMEmotionWorker


class OnePacketBuffer:
    def __init__(self, packet, stop_event):
        self.packet = packet
        self.stop_event = stop_event
        self.sent = False

    def get_newest(self, last_frame_id, stop_event):
        if not self.sent:
            self.sent = True
            return self.packet
        self.stop_event.set()
        return None


class SVMEmotionTests(unittest.TestCase):
    def test_packaged_model_loads_and_preserves_feature_order(self):
        analyzer = SVMEmotionAnalyzer(ProjectPaths.discover().emotion_model_dir)
        try:
            landmarks = [
                SimpleNamespace(x=(index % 28) / 27, y=(index // 28) / 17)
                for index in range(468)
            ]
            features = analyzer._extract_features(landmarks, 280, 170)
            feature_frame = analyzer._feature_frame(features)

            self.assertEqual(len(analyzer.feature_names), 68)
            self.assertEqual(tuple(feature_frame.columns), analyzer.feature_names)
            self.assertEqual(tuple(analyzer._model.classes_), SUPPORTED_EMOTIONS)
            self.assertEqual(feature_frame.shape, (1, 68))
            prediction = analyzer._model.predict(feature_frame)[0]
            self.assertIn(prediction, SUPPORTED_EMOTIONS)
        finally:
            analyzer.close()

    def test_worker_processes_confirmed_driver_only_and_closes_analyzer(self):
        stop_event = threading.Event()
        packet = FramePacket(
            12, time.monotonic(), np.zeros((120, 160, 3), dtype=np.uint8)
        )
        face_state = {
            "driver_confirmed": True,
            "driver_name": "Known Driver",
            "recognition_location": (15, 95, 105, 35),
            "recognition_confidence": 0.9,
            "face_present": True,
            "face_bounds": (18, 92, 102, 38),
            "behavior_frame_id": 12,
        }
        analyzer = Mock()
        analyzer.predict.return_value = "happy"
        published = []
        worker = SVMEmotionWorker(
            OnePacketBuffer(packet, stop_event),
            stop_event,
            lambda: face_state,
            published.append,
            Path("emotion-model"),
        )

        with patch(
            "driver_monitor.monitoring.workers.SVMEmotionAnalyzer",
            return_value=analyzer,
        ) as analyzer_factory:
            worker.start()
            worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        analyzer_factory.assert_called_once_with(Path("emotion-model"))
        analyzer.predict.assert_called_once()
        analyzer.close.assert_called_once()
        self.assertEqual(published[-1]["emotion_label"], "happy")
        self.assertEqual(published[-1]["emotion_status"], "ready")
        self.assertNotIn("emotion_score", published[-1])

    def test_worker_returns_unknown_for_unconfirmed_or_passenger_face(self):
        for face_state in (
            {
                "driver_confirmed": False,
                "driver_name": "UNKNOWN",
                "recognition_location": None,
                "face_present": True,
                "face_bounds": (18, 92, 102, 38),
                "behavior_frame_id": 12,
            },
            {
                "driver_confirmed": True,
                "driver_name": "Known Driver",
                "recognition_location": (15, 95, 105, 35),
                "face_present": True,
                "face_bounds": (18, 150, 102, 100),
                "behavior_frame_id": 12,
            },
        ):
            with self.subTest(driver_confirmed=face_state["driver_confirmed"]):
                stop_event = threading.Event()
                packet = FramePacket(
                    12, time.monotonic(), np.zeros((120, 160, 3), dtype=np.uint8)
                )
                analyzer = Mock()
                published = []
                worker = SVMEmotionWorker(
                    OnePacketBuffer(packet, stop_event),
                    stop_event,
                    lambda: face_state,
                    published.append,
                    Path("emotion-model"),
                )
                with patch(
                    "driver_monitor.monitoring.workers.SVMEmotionAnalyzer",
                    return_value=analyzer,
                ):
                    worker.start()
                    worker.join(timeout=2)

                self.assertFalse(worker.is_alive())
                analyzer.predict.assert_not_called()
                self.assertEqual(published[-1]["emotion_label"], "unknown")
                self.assertEqual(published[-1]["emotion_status"], "unknown")
                analyzer.close.assert_called_once()

    def test_worker_discards_stale_camera_frames(self):
        stop_event = threading.Event()
        packet = FramePacket(
            12,
            time.monotonic() - 10,
            np.zeros((120, 160, 3), dtype=np.uint8),
        )
        analyzer = Mock()
        published = []
        worker = SVMEmotionWorker(
            OnePacketBuffer(packet, stop_event),
            stop_event,
            lambda: {"driver_confirmed": True, "driver_name": "Known Driver"},
            published.append,
            Path("emotion-model"),
        )
        with patch(
            "driver_monitor.monitoring.workers.SVMEmotionAnalyzer",
            return_value=analyzer,
        ):
            worker.start()
            worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        analyzer.predict.assert_not_called()
        self.assertEqual(published[-1]["emotion_status"], "stale")

    def test_worker_rejects_nonpositive_interval(self):
        with self.assertRaises(ValueError):
            SVMEmotionWorker(
                object(),
                threading.Event(),
                lambda: {},
                lambda _result: None,
                Path("emotion-model"),
                interval=0,
            )


if __name__ == "__main__":
    unittest.main()
