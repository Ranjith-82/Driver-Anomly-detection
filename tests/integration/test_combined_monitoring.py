import unittest

from driver_monitor.monitoring.alerts import build_safety_alerts
from driver_monitor.monitoring.buffers import ResultBuffer


class CombinedMonitoringTests(unittest.TestCase):
    def test_detector_failure_does_not_hide_available_results(self):
        results = ResultBuffer()
        results.update("identity", {"status": "ready", "driver": "known"})
        results.update("behavior", {"status": "ready", "label": "NORMAL"})
        results.update("objects", {"status": "ready", "items": []})
        results.update("camera", {"status": "unavailable", "error": "not found"})
        results.update("emotion", {"status": "unavailable", "label": "Uncertain"})
        snapshot = results.snapshot()
        self.assertEqual(snapshot["identity"]["driver"], "known")
        self.assertEqual(snapshot["behavior"]["label"], "NORMAL")
        self.assertEqual(snapshot["objects"]["items"], [])
        self.assertEqual(snapshot["camera"]["status"], "unavailable")
        self.assertEqual(snapshot["emotion"]["status"], "unavailable")

    def test_controlled_normal_occlusion_and_low_light_inputs_do_not_false_alert(self):
        scenarios = {
            "normal": {
                "face_present": True,
                "behavior": "NORMAL",
                "emotion": "neutral",
            },
            "partial occlusion": {
                "face_present": False,
                "behavior": "FACE NOT DETECTED",
                "emotion": "Uncertain",
            },
            "low light": {
                "face_present": True,
                "behavior": "UNCERTAIN",
                "emotion": "Uncertain",
            },
        }
        for name, values in scenarios.items():
            with self.subTest(scenario=name):
                values.update({
                    "fatigue_alert": False,
                    "gaze_distraction_confirmed": False,
                    "yawn_event": False,
                    "object_detections": [],
                })
                self.assertEqual(build_safety_alerts(values), [])

    def test_concurrent_existing_alerts_are_unchanged_by_emotion(self):
        base = {
            "face_present": True,
            "behavior": "DISTRACTED",
            "fatigue_alert": False,
            "gaze_distraction_confirmed": True,
            "yawn_event": False,
            "object_detections": [{"class_id": 0}],
            "driver_confirmed": True,
            "phone_detection_confirmed": True,
            "driver_phone_alert": True,
        }
        without_emotion = build_safety_alerts(base)
        with_emotion = build_safety_alerts({**base, "emotion": "anger"})
        self.assertEqual(with_emotion, without_emotion)
        self.assertGreaterEqual(len(with_emotion), 2)

    def test_no_face_and_low_confidence_emotion_never_become_neutral(self):
        results = ResultBuffer()
        results.update("emotion", {
            "status": "no_face",
            "label": "Uncertain",
        })
        self.assertEqual(results.snapshot()["emotion"]["label"], "Uncertain")
        results.update("emotion", {
            "status": "uncertain",
            "label": "Uncertain",
        })
        self.assertEqual(results.snapshot()["emotion"]["label"], "Uncertain")


if __name__ == "__main__":
    unittest.main()
