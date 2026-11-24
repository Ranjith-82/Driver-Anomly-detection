import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from driver_monitor.config import ProjectPaths
from driver_monitor.detection.cnn_emotion import (
    CNNEmotionAnalyzer,
    EXPECTED_EMOTIONS,
)
from driver_monitor.models import FramePacket
from driver_monitor.monitoring.workers import CNNEmotionWorker
from driver_monitor.monitoring.temporal import EmotionPredictionStabilizer


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


class TwoPacketBuffer:
    def __init__(self, packets, stop_event):
        self.packets = packets
        self.stop_event = stop_event

    def get_newest(self, last_frame_id, stop_event):
        for packet in self.packets:
            if packet.frame_id > last_frame_id:
                if last_frame_id >= self.packets[0].frame_id:
                    time.sleep(0.02)
                return packet
        self.stop_event.set()
        return None


class CNNEmotionTests(unittest.TestCase):
    def test_emotion_stabilizer_requires_two_consecutive_predictions(self):
        stabilizer = EmotionPredictionStabilizer()
        self.assertIsNone(stabilizer.update("confused", 0.587))
        self.assertIsNone(stabilizer.update("neutral", 0.62))
        self.assertIsNone(stabilizer.update("confused", 0.59))
        self.assertEqual(
            stabilizer.update("confused", 0.61),
            ("confused", 0.60),
        )
        self.assertIsNone(stabilizer.update("neutral", 0.8))
        self.assertEqual(
            stabilizer.update("confused", 0.58),
            ("confused", 0.58),
        )
        self.assertIsNone(stabilizer.update("neutral", 0.7))
        self.assertEqual(
            stabilizer.update("neutral", 0.72),
            ("neutral", 0.71),
        )

    def test_emotion_stabilizer_reset_discards_previous_confirmations(self):
        stabilizer = EmotionPredictionStabilizer()
        self.assertIsNone(stabilizer.update("happy", 0.8))
        self.assertEqual(stabilizer.update("happy", 0.82), ("happy", 0.81))
        stabilizer.reset()
        self.assertIsNone(stabilizer.update("happy", 0.84))

    def test_checkpoint_predicts_with_its_saved_class_order(self):
        analyzer = CNNEmotionAnalyzer(ProjectPaths.discover().emotion_checkpoint)
        self.assertEqual(
            analyzer.classes,
            (
                "angry",
                "confused",
                "disgust",
                "fear",
                "happy",
                "neutral",
                "sad",
                "shy",
                "surprise",
            ),
        )
        self.assertEqual(set(analyzer.classes), EXPECTED_EMOTIONS)

        label, confidence = analyzer.predict(
            np.zeros((96, 128, 3), dtype=np.uint8)
        )
        self.assertIn(label, analyzer.classes)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_worker_loads_once_and_publishes_emotion_confidence(self):
        stop_event = threading.Event()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        packets = (
            FramePacket(12, time.monotonic(), frame),
            FramePacket(13, time.monotonic(), frame),
        )
        face_state = {
            "driver_confirmed": True,
            "driver_name": "Known Driver",
            "recognition_location": (15, 95, 105, 35),
            "face_present": True,
            "face_bounds": (18, 92, 102, 38),
            "behavior_frame_id": 12,
        }
        analyzer = Mock()
        analyzer.predict.return_value = ("happy", 0.875)
        published = []
        checkpoint = ProjectPaths.discover().emotion_checkpoint
        worker = CNNEmotionWorker(
            TwoPacketBuffer(packets, stop_event),
            stop_event,
            lambda: face_state,
            published.append,
            checkpoint,
            interval=0.01,
        )

        with patch(
            "driver_monitor.monitoring.workers.CNNEmotionAnalyzer",
            return_value=analyzer,
        ) as analyzer_factory:
            worker.start()
            worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        analyzer_factory.assert_called_once_with(checkpoint)
        self.assertEqual(analyzer.predict.call_count, 2)
        self.assertEqual(
            [result["emotion_status"] for result in published],
            ["uncertain", "ready"],
        )
        self.assertEqual(published[0]["emotion_label"], "unknown")
        self.assertIsNone(published[0]["emotion_confidence"])
        self.assertEqual(published[1]["emotion_label"], "happy")
        self.assertEqual(published[1]["emotion_confidence"], 0.875)

    def test_worker_suppresses_inconsistent_single_sample_predictions(self):
        stop_event = threading.Event()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        packets = tuple(
            FramePacket(index, time.monotonic(), frame)
            for index in (20, 21, 22, 23)
        )
        face_state = {
            "driver_confirmed": True,
            "driver_name": "Known Driver",
            "recognition_location": (15, 95, 105, 35),
            "face_present": True,
            "face_bounds": (18, 92, 102, 38),
            "behavior_frame_id": 20,
        }
        analyzer = Mock()
        analyzer.predict.side_effect = [
            ("confused", 0.587),
            ("neutral", 0.62),
            ("confused", 0.59),
            ("confused", 0.61),
        ]
        published = []
        worker = CNNEmotionWorker(
            TwoPacketBuffer(packets, stop_event),
            stop_event,
            lambda: face_state,
            published.append,
            ProjectPaths.discover().emotion_checkpoint,
            interval=0.01,
        )

        with patch(
            "driver_monitor.monitoring.workers.CNNEmotionAnalyzer",
            return_value=analyzer,
        ):
            worker.start()
            worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(
            [result["emotion_status"] for result in published],
            ["uncertain", "uncertain", "uncertain", "ready"],
        )
        self.assertEqual(published[-1]["emotion_label"], "confused")
        self.assertAlmostEqual(published[-1]["emotion_confidence"], 0.60)

    def test_worker_handles_missing_face_without_running_inference(self):
        stop_event = threading.Event()
        packet = FramePacket(
            12, time.monotonic(), np.zeros((120, 160, 3), dtype=np.uint8)
        )
        analyzer = Mock()
        published = []
        worker = CNNEmotionWorker(
            OnePacketBuffer(packet, stop_event),
            stop_event,
            lambda: {
                "driver_confirmed": True,
                "driver_name": "Known Driver",
                "face_present": False,
                "face_bounds": None,
                "behavior_frame_id": 12,
            },
            published.append,
            ProjectPaths.discover().emotion_checkpoint,
        )

        with patch(
            "driver_monitor.monitoring.workers.CNNEmotionAnalyzer",
            return_value=analyzer,
        ):
            worker.start()
            worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        analyzer.predict.assert_not_called()
        self.assertEqual(published[-1]["emotion_status"], "no_face")
        self.assertIsNone(published[-1]["emotion_confidence"])

    def test_worker_publishes_unavailable_when_inference_fails(self):
        stop_event = threading.Event()
        packet = FramePacket(
            12, time.monotonic(), np.zeros((120, 160, 3), dtype=np.uint8)
        )
        face_state = {
            "driver_confirmed": True,
            "driver_name": "Known Driver",
            "recognition_location": (15, 95, 105, 35),
            "face_present": True,
            "face_bounds": (18, 92, 102, 38),
            "behavior_frame_id": 12,
        }
        analyzer = Mock()
        analyzer.predict.side_effect = RuntimeError("test inference failure")
        published = []
        worker = CNNEmotionWorker(
            OnePacketBuffer(packet, stop_event),
            stop_event,
            lambda: face_state,
            published.append,
            ProjectPaths.discover().emotion_checkpoint,
        )

        with patch(
            "driver_monitor.monitoring.workers.CNNEmotionAnalyzer",
            return_value=analyzer,
        ):
            worker.start()
            worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(published[-1]["emotion_status"], "unavailable")
        self.assertIsNone(published[-1]["emotion_confidence"])
        self.assertIn("test inference failure", published[-1]["emotion_error"])


if __name__ == "__main__":
    unittest.main()
