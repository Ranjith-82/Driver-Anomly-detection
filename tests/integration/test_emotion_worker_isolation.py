import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from driver_monitor.monitoring.workers import EmotionWorker


class EmotionWorkerIsolationTests(unittest.TestCase):
    def test_missing_release_policy_is_visible_and_worker_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            stop_event = threading.Event()
            camera_and_behavior = threading.Event()
            emitted = []
            frame_buffer = type("FrameBuffer", (), {
                "get_newest": lambda self, last_frame_id, stop: None,
            })()
            worker = EmotionWorker(
                frame_buffer,
                stop_event,
                lambda: {"face_present": True},
                emitted.append,
                Path(temporary) / "absent-emotion-threshold-policy.json",
            )
            worker.start()
            worker.join(timeout=2)
            camera_and_behavior.set()
            self.assertFalse(worker.is_alive())
            self.assertFalse(stop_event.is_set())
            self.assertTrue(camera_and_behavior.is_set())
            self.assertEqual(emitted[0]["emotion_status"], "uncertain")
            self.assertEqual(emitted[0]["emotion_label"], "Uncertain")
            self.assertNotIn("emotion_scores", emitted[0])

    def test_worker_rejects_invalid_sampling_interval(self):
        with self.assertRaises(ValueError):
            EmotionWorker(
                object(),
                threading.Event(),
                lambda: {},
                lambda _result: None,
                Path("thresholds.json"),
                interval=0,
            )

    def test_model_initialization_failure_is_visible_without_stopping_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            thresholds_path = Path(temporary) / "thresholds.json"
            thresholds_path.write_text(json.dumps({
                "evaluation_version": "test",
                "classes": {
                    "angry": {
                        "enabled": True,
                        "threshold": 0.8,
                        "validation_precision": 0.8,
                        "held_out_precision": 0.8,
                    },
                },
            }), encoding="utf-8")
            stop_event = threading.Event()
            published = []
            worker = EmotionWorker(
                object(),
                stop_event,
                lambda: {},
                published.append,
                thresholds_path,
            )
            with patch(
                "driver_monitor.monitoring.workers.DeepFaceEmotionAdapter",
                side_effect=RuntimeError("test model initialization failure"),
            ):
                worker.start()
                worker.join(timeout=2)

            self.assertFalse(worker.is_alive())
            self.assertFalse(stop_event.is_set())
            self.assertEqual(published[0]["emotion_status"], "unavailable")
            self.assertIn("initialization failure", published[0]["emotion_error"])


if __name__ == "__main__":
    unittest.main()
