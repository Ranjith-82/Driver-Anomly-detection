import unittest
from types import SimpleNamespace

import numpy as np

from driver_monitor.monitoring.alerts import build_safety_alerts
from driver_monitor.monitoring.legacy_runtime import (
    BufferedDriverMonitor,
    ResultBuffer,
)
from driver_monitor.ui.live_dashboard import (
    draw_detection_annotation,
    emotion_confidence_text,
    emotion_display,
    emotion_emoji,
)


class LiveDashboardTests(unittest.TestCase):
    def test_detection_labels_stay_inside_frame_at_each_edge(self):
        boxes = (
            (0, 0, 50, 30),
            (270, 0, 319, 30),
            (0, 210, 50, 239),
            (270, 210, 319, 239),
        )
        for box in boxes:
            with self.subTest(box=box):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                bounds = draw_detection_annotation(
                    frame, box, "Cup confirmed 0.53", (0, 200, 90)
                )
                self.assertIsNotNone(bounds)
                left, top, right, bottom = bounds
                self.assertGreaterEqual(left, 0)
                self.assertGreaterEqual(top, 0)
                self.assertLessEqual(right, frame.shape[1])
                self.assertLessEqual(bottom, frame.shape[0])

    def test_detection_labels_avoid_existing_label_regions(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        first = draw_detection_annotation(
            frame, (100, 70, 150, 100), "Cup candidate 0.53", (0, 200, 90)
        )
        second = draw_detection_annotation(
            frame,
            (160, 70, 210, 100),
            "Bottle candidate 0.62",
            (0, 200, 90),
            (first,),
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertFalse(
            first[0] < second[2]
            and second[0] < first[2]
            and first[1] < second[3]
            and second[1] < first[3]
        )

    def test_invalid_or_off_frame_detection_boxes_are_not_drawn(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertIsNone(
            draw_detection_annotation(
                frame, (20, 20, 20, 40), "invalid", (0, 200, 90)
            )
        )
        self.assertIsNone(
            draw_detection_annotation(
                frame, (120, 20, 150, 40), "outside", (0, 200, 90)
            )
        )

    def test_dashboard_alerts_do_not_obscure_the_camera_image(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        values = ResultBuffer().snapshot()
        values.update({
            "alert_status": "DISTRACTED",
            "gaze_distraction_confirmed": True,
            "face_present": True,
        })
        dashboard = BufferedDriverMonitor.render_dashboard(
            frame, values, 30.0, SimpleNamespace(frame_id=1)
        )
        np.testing.assert_array_equal(
            dashboard[180:660, 490:1130], frame
        )

    def test_displays_valid_label_only_for_ready_status(self):
        self.assertEqual(
            emotion_display({
                "emotion_label": "smiling (estimated from happy)",
                "emotion_status": "ready",
            }),
            ("smiling (estimated from happy)", "ready"),
        )

    def test_distinguishes_no_face_worker_failure_and_uncertainty(self):
        self.assertEqual(
            emotion_display({"emotion_status": "no_face"}),
            ("No face", "no_face"),
        )
        self.assertEqual(
            emotion_display({"emotion_status": "unavailable", "emotion_label": "anger"}),
            ("Uncertain", "unavailable"),
        )
        self.assertEqual(
            emotion_display({"emotion_status": "stale", "emotion_label": "anger"}),
            ("Uncertain", "stale"),
        )

    def test_displays_emotion_labels_with_emojis(self):
        expected = {
            "angry": "😠",
            "confused": "😕",
            "disgust": "🤢",
            "fear": "😨",
            "happy": "😊",
            "neutral": "😐",
            "sad": "😢",
            "shy": "😳",
            "surprise": "😲",
        }
        for label, emoji in expected.items():
            with self.subTest(label=label):
                self.assertEqual(
                    emotion_display({
                        "emotion_label": label,
                        "emotion_status": "ready",
                    }),
                    (label, "ready"),
                )
                self.assertEqual(emotion_emoji(label), emoji)
        self.assertEqual(
            emotion_display({
                "emotion_label": "unknown",
                "emotion_status": "unknown",
            }),
            ("unknown", "unknown"),
        )
        self.assertEqual(emotion_emoji("unknown"), "❓")

    def test_displays_only_valid_ready_emotion_confidence(self):
        self.assertEqual(
            emotion_confidence_text({
                "emotion_status": "ready",
                "emotion_confidence": 0.875,
            }),
            "Score: 87.5%",
        )
        for values in (
            {"emotion_status": "no_face", "emotion_confidence": 0.9},
            {"emotion_status": "uncertain", "emotion_confidence": 0.9},
            {"emotion_status": "ready", "emotion_confidence": float("nan")},
            {"emotion_status": "ready", "emotion_confidence": 1.1},
        ):
            with self.subTest(values=values):
                self.assertEqual(emotion_confidence_text(values), "")

    def test_emotion_unavailability_does_not_hide_safety_alerts(self):
        values = {
            "behavior": "DISTRACTED",
            "face_present": True,
            "fatigue_alert": False,
            "gaze_distraction_confirmed": True,
            "yawn_event": False,
            "raw_object_detections": [],
            "emotion_label": "anger",
            "emotion_status": "unavailable",
        }
        self.assertEqual(emotion_display(values), ("Uncertain", "unavailable"))
        self.assertIn("DRIVER DISTRACTION DETECTED", build_safety_alerts(values))


if __name__ == "__main__":
    unittest.main()
