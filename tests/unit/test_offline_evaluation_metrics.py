import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from evaluation.common.evaluators import evaluate_fatigue_events
from evaluation.common.inputs import (
    EvaluationInputError,
    validate_prediction_rows,
    verify_independent_split,
)
from evaluation.common.metrics import classification_metrics, regression_metrics


class OfflineEvaluationMetricTests(unittest.TestCase):
    def test_multiclass_metrics_and_one_vs_rest_rates(self):
        result = classification_metrics(
            ["normal", "normal", "drowsy"],
            ["normal", "drowsy", "normal"],
            ["normal", "drowsy"],
        )
        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertEqual(result["confusion_matrix"]["normal"]["normal"], 1)
        self.assertEqual(result["confusion_matrix"]["normal"]["drowsy"], 1)
        self.assertEqual(result["per_class"]["drowsy"]["false_positive"], 1)
        self.assertEqual(result["per_class"]["drowsy"]["false_negative"], 1)
        self.assertAlmostEqual(result["per_class"]["drowsy"]["false_positive_rate"], 0.5)
        self.assertAlmostEqual(result["per_class"]["drowsy"]["false_negative_rate"], 1.0)

    def test_rejects_mismatched_or_undeclared_class_predictions(self):
        with self.assertRaisesRegex(ValueError, "counts differ"):
            classification_metrics(["normal"], [], ["normal"])
        with self.assertRaisesRegex(ValueError, "undeclared"):
            validate_prediction_rows(
                [
                    {
                        "sample_id": "sample-1",
                        "identity_id": "",
                        "sha256": "a" * 64,
                        "model_sha256": "b" * 64,
                        "actual": "wrong",
                        "predicted": "normal",
                    }
                ],
                ["normal"],
            )

    def test_regression_error_calculation(self):
        result = regression_metrics([10.0, 20.0], [12.0, 17.0])
        self.assertAlmostEqual(result["mean_error"], -0.5)
        self.assertAlmostEqual(result["mae"], 2.5)
        self.assertAlmostEqual(result["rmse"], (6.5) ** 0.5)

    def test_split_check_rejects_subject_and_hash_leakage(self):
        rows = [
            {
                "identity_id": "person-a",
                "sha256": "a" * 64,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            metadata = Path(temporary) / "split.json"
            metadata.write_text(
                json.dumps(
                    {
                        "independent_test": True,
                        "split_strategy": "subject-independent",
                        "training_identity_ids": ["person-a"],
                        "test_identity_ids": ["person-a"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EvaluationInputError, "Identity leakage"):
                verify_independent_split(metadata, rows)

    def test_fatigue_event_matching_rejects_empty_positive_intervals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings, model_hash = self._fatigue_fixture(root)
            event_file = root / "events.csv"
            event_file.write_text(
                "clip_id,identity_id,sha256,model_sha256,source,start_seconds,end_seconds\n"
                f"clip-1,test-1,{'a' * 64},,ground_truth,,\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EvaluationInputError, "timestamps"):
                evaluate_fatigue_events(root, settings)

    def test_fatigue_event_matching_on_valid_clips(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings, model_hash = self._fatigue_fixture(root)
            event_file = root / "events.csv"
            event_file.write_text(
                "clip_id,identity_id,sha256,model_sha256,source,start_seconds,end_seconds\n"
                f"clip-1,test-1,{'a' * 64},,ground_truth,1,3\n"
                f"clip-1,test-1,{'a' * 64},{model_hash},prediction,1.2,3.2\n"
                f"clip-2,test-2,{'b' * 64},,ground_truth,4,5\n"
                f"clip-3,test-3,{'c' * 64},{model_hash},prediction,1,2\n"
                f"clip-4,test-4,{'d' * 64},,negative_clip,,\n",
                encoding="utf-8",
            )
            report, matched = evaluate_fatigue_events(root, settings)
            metrics = report["metrics"]
            self.assertEqual(metrics["true_positive"], 1)
            self.assertEqual(metrics["true_negative"], 1)
            self.assertEqual(metrics["false_positive"], 1)
            self.assertEqual(metrics["false_negative"], 1)
            self.assertEqual(metrics["missed_event_count"], 1)
            self.assertAlmostEqual(metrics["mean_detection_delay_seconds"], 0.2)
            self.assertEqual(len(matched), 3)

    @staticmethod
    def _fatigue_fixture(root):
        model_path = root / "model.py"
        model_path.write_bytes(b"offline-rule-fixture")
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        split_path = root / "split.json"
        split_path.write_text(
            json.dumps(
                {
                    "independent_test": True,
                    "split_strategy": "recording-independent",
                    "training_identity_ids": ["train-person"],
                    "test_identity_ids": [
                        "test-1",
                        "test-2",
                        "test-3",
                        "test-4",
                    ],
                    "training_sha256": ["e" * 64],
                    "test_sha256": [
                        "a" * 64,
                        "b" * 64,
                        "c" * 64,
                        "d" * 64,
                    ],
                }
            ),
            encoding="utf-8",
        )
        return (
            {
                "events_csv": str(root / "events.csv"),
                "model_path": str(model_path),
                "split_metadata": str(split_path),
                "dataset": "synthetic unit fixture",
                "dataset_version": "test-only",
                "dataset_path": str(root),
            },
            model_hash,
        )


if __name__ == "__main__":
    unittest.main()
