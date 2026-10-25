import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from driver_monitor.detection.emotion import (
    DeepFaceEmotionAdapter,
    EmotionThresholdPolicy,
    classify_scores,
)
from driver_monitor.models import DetectorStatus, EmotionLabel


class EmotionMappingTests(unittest.TestCase):
    def setUp(self):
        self.thresholds = {
            "angry": 40.0,
            "happy": 40.0,
            "neutral": 40.0,
        }

    def test_maps_only_validated_target_classes(self):
        cases = (
            ({"angry": 60, "happy": 10, "neutral": 10, "sad": 20},
             EmotionLabel.ANGER),
            ({"angry": 5, "happy": 70, "neutral": 5, "sad": 20},
             EmotionLabel.SMILING),
            ({"angry": 5, "happy": 5, "neutral": 70, "sad": 20},
             EmotionLabel.NEUTRAL),
        )
        for scores, expected in cases:
            with self.subTest(expected=expected):
                estimate = classify_scores(scores, self.thresholds)
                self.assertEqual(estimate.label, expected)
                self.assertEqual(estimate.status, DetectorStatus.READY)

    def test_non_target_and_below_threshold_are_uncertain(self):
        non_target = classify_scores(
            {"angry": 10, "happy": 10, "neutral": 10, "sad": 70},
            self.thresholds,
        )
        below_threshold = classify_scores(
            {"angry": 30, "happy": 20, "neutral": 20},
            self.thresholds,
        )
        self.assertEqual(non_target.label, EmotionLabel.UNCERTAIN)
        self.assertEqual(below_threshold.label, EmotionLabel.UNCERTAIN)

    def test_disabled_classes_and_invalid_values_are_uncertain(self):
        disabled = classify_scores(
            {"angry": 80, "happy": 10, "neutral": 10},
            {"angry": None, "happy": None, "neutral": None},
        )
        malformed = classify_scores({"angry": float("nan")}, self.thresholds)
        self.assertEqual(disabled.label, EmotionLabel.UNCERTAIN)
        self.assertEqual(malformed.label, EmotionLabel.UNCERTAIN)
        self.assertEqual(malformed.status, DetectorStatus.UNCERTAIN)

    def test_empty_face_result_has_no_face_status(self):
        estimate = classify_scores({}, self.thresholds)
        self.assertEqual(estimate.status, DetectorStatus.NO_FACE)
        self.assertEqual(estimate.label, EmotionLabel.UNCERTAIN)

    def test_adapter_requests_only_emotion_action(self):
        deepface = Mock()
        deepface.analyze.return_value = [{
            "emotion": {"angry": 1.0, "happy": 2.0, "neutral": 97.0}
        }]
        with patch.dict("sys.modules", {"deepface": SimpleNamespace(DeepFace=deepface)}):
            adapter = DeepFaceEmotionAdapter()
            adapter.warm_up()
            result = adapter.analyze_face_crop(object())
        self.assertEqual(result["neutral"], 97.0)
        deepface.build_model.assert_called_once_with("Emotion")
        self.assertEqual(deepface.analyze.call_args.kwargs["actions"], ["emotion"])
        self.assertEqual(deepface.analyze.call_args.kwargs["detector_backend"], "skip")

    def test_threshold_policy_rejects_classes_that_fail_evaluation(self):
        import json
        import tempfile
        from pathlib import Path

        policy = {
            "evaluation_version": "evaluation-v1",
            "classes": {
                "angry": {
                    "threshold": 40,
                    "validation_precision": 0.9,
                    "held_out_precision": 0.8,
                    "enabled": True,
                },
                "happy": {
                    "threshold": 40,
                    "validation_precision": 0.9,
                    "held_out_precision": 0.7,
                    "enabled": False,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "thresholds.json"
            path.write_text(json.dumps(policy), encoding="utf-8")
            loaded = EmotionThresholdPolicy.load(path)
        self.assertEqual(loaded.thresholds, {"angry": 40.0})


if __name__ == "__main__":
    unittest.main()
