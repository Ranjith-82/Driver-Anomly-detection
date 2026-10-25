import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from driver_monitor.fusion.features import FEATURE_NAMES, build_feature_vector
from driver_monitor.monitoring.alerts import (
    AlertManager,
    TelegramWorker,
    build_safety_alerts,
    notification_values,
)
from driver_monitor.profiles.model import DriverProfile
from driver_monitor.detection.objects import OBJECT_CLASSES, YoloObjectDetector
from driver_monitor.detection.recognition import (
    create_live_recognizer,
    recognize_frame,
)
from driver_monitor.profiles.repository import DriverProfileRepository


class ExistingDetectionCompatibilityTests(unittest.TestCase):
    def test_behavior_feature_order_remains_unchanged(self):
        expected = [
            "recognition_confidence", "recognition_status", "left_ear",
            "right_ear", "average_ear", "blink_count_10s", "blink_rate",
            "eyes_closed_duration", "mar", "pitch", "yaw", "roll",
            "fatigue_alert", "distracted", "yawning", "yawn_event",
            "talking", "behavior_confidence",
        ]
        self.assertEqual(FEATURE_NAMES, expected)
        vector = build_feature_vector({
            "recognition_confidence": 0.9,
            "driver_confirmed": True,
            "left_ear": 0.2,
            "fatigue_alert": True,
            "emotion": "anger",
        })
        self.assertEqual(len(vector), len(expected))
        self.assertEqual(vector[:3], [0.9, 1.0, 0.2])
        self.assertNotIn("emotion", FEATURE_NAMES)

    def test_yolo_class_mapping_is_preserved(self):
        YoloObjectDetector._validate_class_names(OBJECT_CLASSES)
        with self.assertRaisesRegex(ValueError, "Incompatible YOLO class mapping"):
            YoloObjectDetector._validate_class_names({
                0: "Phone",
                1: "Cigarette",
                2: "Cup",
                3: "Book",
            })

    def test_yolo_adapter_preserves_live_worker_thresholds_and_box_filter(self):
        detector = object.__new__(YoloObjectDetector)
        detector._torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False)
        )
        boxes = [
            SimpleNamespace(
                cls=[SimpleNamespace(item=lambda: 0)],
                conf=[SimpleNamespace(item=lambda: 0.25)],
                xyxy=[SimpleNamespace(tolist=lambda: [5, 5, 25, 25])],
            ),
            SimpleNamespace(
                cls=[SimpleNamespace(item=lambda: 1)],
                conf=[SimpleNamespace(item=lambda: 0.49)],
                xyxy=[SimpleNamespace(tolist=lambda: [5, 5, 25, 25])],
            ),
            SimpleNamespace(
                cls=[SimpleNamespace(item=lambda: 2)],
                conf=[SimpleNamespace(item=lambda: 0.5)],
                xyxy=[SimpleNamespace(tolist=lambda: [0, 0, 100, 100])],
            ),
        ]
        detector._model = SimpleNamespace(
            predict=Mock(return_value=[SimpleNamespace(boxes=boxes)])
        )

        detections = detector.detect(
            np.zeros((100, 100, 3)),
            image_size=416,
            confidence=0.1,
            iou=0.45,
            max_detections=20,
            class_confidence={0: 0.2, 1: 0.5, 2: 0.35},
            max_box_fraction=0.85,
        )

        self.assertEqual(
            detections,
            [{
                "class_id": 0,
                "name": "Phone",
                "confidence": 0.25,
                "box": (5, 5, 25, 25),
            }],
        )
        detector._model.predict.assert_called_once()
        self.assertEqual(
            detector._model.predict.call_args.kwargs["max_det"], 20
        )

    def test_recognition_adapter_uses_profile_repository_and_legacy_matching(self):
        class LegacyRecognizer:
            def __init__(self, database_path):
                self.database_path = database_path
                self.known_encodings = ["known-encoding"]
                self.known_driver_ids = ["driver-1"]

        face_recognition = SimpleNamespace(
            face_locations=Mock(return_value=[(10, 30, 40, 5)]),
            face_encodings=Mock(return_value=["candidate"]),
            face_distance=Mock(return_value=[0.25]),
        )
        module = SimpleNamespace(
            LiveDriverRecognition=LegacyRecognizer,
            cv2=SimpleNamespace(
                COLOR_BGR2RGB=1,
                resize=Mock(return_value="resized"),
                cvtColor=Mock(return_value="rgb"),
            ),
            face_recognition=face_recognition,
            np=SimpleNamespace(argmin=lambda values: 0),
        )
        profiles = DriverProfileRepository("temporary-driver-profiles.pkl")
        with patch(
            "driver_monitor.detection.recognition.load_recognition_module",
            return_value=module,
        ):
            recognizer = create_live_recognizer(profiles)
            result = recognize_frame(
                "frame",
                recognizer,
                tolerance=0.6,
                resize_scale=0.5,
            )

        self.assertEqual(
            recognizer.database_path,
            str(profiles.database_path),
        )
        self.assertEqual(result, [("driver-1", 0.75, (10, 30, 40, 5))])
        face_recognition.face_distance.assert_called_once_with(
            ["known-encoding"], "candidate"
        )

    def test_safety_alerts_ignore_emotion(self):
        values = {
            "behavior": "NORMAL",
            "face_present": True,
            "fatigue_alert": False,
            "gaze_distraction_confirmed": False,
            "yawn_event": False,
            "object_detections": [],
            "emotion": "anger",
        }
        self.assertEqual(build_safety_alerts(values), [])

    def test_telegram_snapshot_excludes_emotion_and_face_bounds(self):
        filtered = notification_values({
            "behavior": "NORMAL",
            "emotion_label": "anger",
            "emotion_status": "ready",
            "emotion_raw_score": 90,
            "face_bounds": (1, 2, 3, 4),
            "driver_name": "Test Driver",
        })
        self.assertEqual(filtered, {
            "behavior": "NORMAL",
            "driver_name": "Test Driver",
        })

    def test_normal_driving_and_uncertain_inputs_do_not_create_alerts(self):
        for face_present in (True, False):
            values = {
                "behavior": "NORMAL",
                "face_present": face_present,
                "fatigue_alert": False,
                "gaze_distraction_confirmed": False,
                "yawn_event": False,
                "object_detections": [],
                "emotion": "Uncertain",
            }
            self.assertEqual(build_safety_alerts(values), [])

    def test_existing_behavior_labels_keep_their_alert_semantics(self):
        scenarios = (
            ("NORMAL", [], True),
            ("DISTRACTED", ["DRIVER DISTRACTED"], True),
            ("FATIGUE ALERT", ["FATIGUE ALERT"], True),
            ("FATIGUE ALERT", [], False),
        )
        for behavior, expected, face_present in scenarios:
            with self.subTest(behavior=behavior, face_present=face_present):
                self.assertEqual(
                    build_safety_alerts({
                        "behavior": behavior,
                        "face_present": face_present,
                        "fatigue_alert": False,
                        "gaze_distraction_confirmed": False,
                        "yawn_event": False,
                        "object_detections": [],
                    }),
                    expected,
                )

    def test_extracted_alert_manager_preserves_priority_and_sanitizes_payload(self):
        worker = Mock()
        manager = AlertManager(worker, cooldown=1.0)
        values = {
            "behavior": "DISTRACTED",
            "face_present": True,
            "fatigue_alert": False,
            "gaze_distraction_confirmed": True,
            "yawn_event": False,
            "object_detections": [{"class_id": 0}],
            "emotion_label": "anger",
            "emotion_status": "ready",
        }
        manager.update(values, "frame", now=10.0)
        worker.submit.assert_called_once()
        reason, payload, frame = worker.submit.call_args.args
        self.assertEqual(reason, "MOBILE PHONE DETECTED")
        self.assertEqual(frame, "frame")
        self.assertFalse(any(key.startswith("emotion") for key in payload))

    def test_extracted_telegram_delivery_uses_repository_without_emotion(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = DriverProfileRepository(
                Path(temporary) / "profiles.pkl"
            )
            repository.create(
                DriverProfile(
                    "D-01",
                    full_name="Test Driver",
                    phone="5550100",
                )
            )
            source = SimpleNamespace(
                Config=SimpleNamespace(BOT_TOKEN="", CHAT_ID=""),
                send_telegram_alert=Mock(return_value=True),
            )
            stop_event = SimpleNamespace(set=Mock())
            with patch.dict(
                "os.environ",
                {
                    "DRIVER_MONITOR_BOT_TOKEN": "test-token",
                    "DRIVER_MONITOR_CHAT_ID": "test-chat",
                },
            ):
                worker = TelegramWorker(
                    stop_event,
                    repository.database_path,
                    cooldown=60.0,
                    telegram_source=source,
                )
            worker.send_alert(
                "DRIVER DISTRACTED",
                {
                    "driver_name": "Test Driver",
                    "average_ear": 0.2,
                    "mar": 0.1,
                    "head_position": "FORWARD",
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "emotion_label": "anger",
                },
                "frame",
            )
            sent_profile, details, _ = source.send_telegram_alert.call_args.args
            self.assertEqual(sent_profile["phone"], "5550100")
            self.assertNotIn("emotion", details)
            self.assertEqual(source.Config.BOT_TOKEN, "test-token")
            self.assertEqual(source.Config.CHAT_ID, "test-chat")


if __name__ == "__main__":
    unittest.main()
