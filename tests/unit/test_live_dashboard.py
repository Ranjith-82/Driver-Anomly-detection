import unittest

from driver_monitor.monitoring.alerts import build_safety_alerts
from driver_monitor.ui.live_dashboard import emotion_display


class LiveDashboardTests(unittest.TestCase):
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

    def test_emotion_unavailability_does_not_hide_safety_alerts(self):
        values = {
            "behavior": "DISTRACTED",
            "face_present": True,
            "fatigue_alert": False,
            "gaze_distraction_confirmed": True,
            "yawn_event": False,
            "object_detections": [],
            "emotion_label": "anger",
            "emotion_status": "unavailable",
        }
        self.assertEqual(emotion_display(values), ("Uncertain", "unavailable"))
        self.assertIn("DRIVER DISTRACTION DETECTED", build_safety_alerts(values))


if __name__ == "__main__":
    unittest.main()
