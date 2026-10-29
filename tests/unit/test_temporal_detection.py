import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from driver_monitor.monitoring.legacy_runtime import (
    ObjectDetectionWorker,
    ResultBuffer,
)
from driver_monitor.monitoring.temporal import (
    BehaviorTemporalFilter,
    PhoneDetectionValidator,
)
from driver_monitor.monitoring.buffers import LatestFrameBuffer


class PhoneDetectionValidatorTests(unittest.TestCase):
    @staticmethod
    def phone(box=(10, 30, 30, 60), confidence=0.8):
        return {
            "class_id": 0,
            "name": "Phone",
            "confidence": confidence,
            "box": box,
        }

    def test_early_phone_candidate_is_tracked_then_confirmed(self):
        validator = PhoneDetectionValidator(
            persistence_seconds=0.2,
            use_persistence_seconds=0.2,
        )
        first = validator.update([self.phone()], 0.0)
        self.assertTrue(first["phone_present"])
        self.assertEqual(first["phone_track_state"], "candidate")
        self.assertFalse(first["phone_detection_confirmed"])

        second = validator.update([self.phone()], 0.25)
        self.assertTrue(second["phone_detection_confirmed"])
        self.assertEqual(second["confirmed_phone_detections"][0]["confidence"], 0.8)
        self.assertEqual(second["confirmed_phone_detections"][0]["track_id"], 1)

    def test_only_phone_class_can_enter_phone_tracking(self):
        validator = PhoneDetectionValidator()
        result = validator.update(
            [{
                "class_id": 1,
                "name": "Cigarette",
                "confidence": 0.99,
                "box": (10, 10, 90, 90),
            }],
            1.0,
        )
        self.assertFalse(result["phone_present"])
        self.assertFalse(result["phone_detection_confirmed"])

    def test_driver_suspicion_requires_persistent_phone_near_confirmed_driver(self):
        validator = PhoneDetectionValidator(
            persistence_seconds=0.0,
            use_persistence_seconds=0.2,
            alert_persistence_seconds=0.2,
        )
        driver_location = (10, 40, 40, 10)
        for timestamp in (0.0, 0.1):
            result = validator.update(
                [self.phone()],
                timestamp,
                driver_confirmed=True,
                driver_location=driver_location,
            )
        self.assertFalse(result["driver_phone_alert"])

        result = validator.update(
            [self.phone()],
            0.35,
            driver_confirmed=True,
            driver_location=driver_location,
        )
        self.assertTrue(result["suspected_driver_phone_use"])
        self.assertFalse(result["driver_phone_alert"])
        result = validator.update(
            [self.phone()],
            0.6,
            driver_confirmed=True,
            driver_location=driver_location,
        )
        self.assertTrue(result["driver_phone_alert"])

    def test_passenger_side_phone_does_not_become_driver_use_alert(self):
        validator = PhoneDetectionValidator(
            persistence_seconds=0.0,
            use_persistence_seconds=0.0,
            alert_persistence_seconds=0.0,
        )
        passenger_phone = self.phone(box=(120, 10, 150, 40))
        result = validator.update(
            [passenger_phone],
            1.0,
            driver_confirmed=True,
            driver_location=(10, 40, 40, 10),
        )
        result = validator.update(
            [passenger_phone],
            1.1,
            driver_confirmed=True,
            driver_location=(10, 40, 40, 10),
        )
        self.assertTrue(result["phone_detection_confirmed"])
        self.assertFalse(result["suspected_driver_phone_use"])
        self.assertFalse(result["driver_phone_alert"])

    def test_stale_confirmed_phone_expires_after_missed_frames(self):
        validator = PhoneDetectionValidator(
            persistence_seconds=0.0,
            max_missed_frames=1,
        )
        validator.update([self.phone()], 0.0)
        self.assertTrue(
            validator.update([self.phone()], 0.1)["phone_detection_confirmed"])
        retained = validator.update([], 0.2)
        self.assertFalse(retained["phone_present"])
        self.assertTrue(retained["phone_detection_confirmed"])
        expired = validator.update([], 0.3)
        self.assertFalse(expired["phone_detection_confirmed"])
        self.assertEqual(expired["phone_track_state"], "none")


class BehaviorTemporalFilterTests(unittest.TestCase):
    def test_transient_talking_does_not_activate_but_persistent_talking_does(self):
        smoother = BehaviorTemporalFilter(
            talking_seconds=0.3,
            clear_seconds=0.1,
        )
        self.assertEqual(smoother.update("TALKING", 0.0), "NORMAL")
        self.assertEqual(smoother.update("NORMAL", 0.1), "NORMAL")
        self.assertEqual(smoother.update("TALKING", 0.2), "NORMAL")
        self.assertEqual(smoother.update("TALKING", 0.49), "NORMAL")
        self.assertEqual(smoother.update("TALKING", 0.51), "TALKING")
        self.assertEqual(smoother.update("NORMAL", 0.55), "TALKING")
        self.assertEqual(smoother.update("NORMAL", 0.66), "NORMAL")

    def test_fatigue_is_never_delayed_by_behavior_smoothing(self):
        smoother = BehaviorTemporalFilter(talking_seconds=10.0)
        self.assertEqual(
            smoother.update("FATIGUE ALERT", 0.0),
            "FATIGUE ALERT",
        )


class ObjectDetectionWorkerStartupTests(unittest.TestCase):
    def test_model_load_failure_is_timestamped_and_visible(self):
        results = ResultBuffer()
        worker = ObjectDetectionWorker(
            LatestFrameBuffer(),
            results,
            threading.Event(),
            Path("missing-model.pt"),
        )
        with patch(
            "driver_monitor.monitoring.legacy_runtime.YoloObjectDetector",
            side_effect=FileNotFoundError("missing model"),
        ), patch("builtins.print") as output:
            worker.run()

        snapshot = results.snapshot()
        self.assertEqual(snapshot["object_detection_status"], "failed")
        self.assertEqual(snapshot["object_detection_error"], "missing model")
        self.assertTrue(
            any("YOLO_WORKER_FAILED" in str(call) for call in output.call_args_list))

    def test_detector_initializes_once_and_waits_for_a_frame_to_report_ready(self):
        results = ResultBuffer()
        stop_event = threading.Event()
        stop_event.set()

        class NoFrames:
            def get_newest(self, last_frame_id, event):
                return None

        worker = ObjectDetectionWorker(
            NoFrames(),
            results,
            stop_event,
            Path("model.pt"),
        )
        detector = SimpleNamespace(device_name="CPU")
        with patch(
            "driver_monitor.monitoring.legacy_runtime.YoloObjectDetector",
            return_value=detector,
        ) as load_model, patch("builtins.print"):
            worker.run()

        load_model.assert_called_once_with(Path("model.pt"))
        self.assertEqual(results.snapshot()["object_detection_status"], "loading")

    def test_first_successful_inference_marks_detector_ready(self):
        results = ResultBuffer()
        stop_event = threading.Event()

        class OneFrame:
            def get_newest(self, last_frame_id, event):
                event.set()
                return SimpleNamespace(
                    frame_id=1,
                    timestamp=1.0,
                    frame=np.zeros((80, 80, 3)),
                )

        detector = SimpleNamespace(
            device_name="CPU",
            last_inference_seconds=0.01,
            last_postprocess_seconds=0.001,
            detect=lambda *args, **kwargs: [],
        )
        worker = ObjectDetectionWorker(
            OneFrame(),
            results,
            stop_event,
            Path("model.pt"),
        )
        with patch(
            "driver_monitor.monitoring.legacy_runtime.YoloObjectDetector",
            return_value=detector,
        ), patch("builtins.print"):
            worker.run()

        self.assertEqual(results.snapshot()["object_detection_status"], "ready")


if __name__ == "__main__":
    unittest.main()
