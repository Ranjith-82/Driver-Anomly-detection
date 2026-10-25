import unittest

from tools.evaluate_emotion_model import (
    build_report,
    evaluate_partition,
    select_class_thresholds,
    validate_partitions,
)


def make_rows(partition, examples_per_class, prefix):
    labels = ("anger", "smiling", "neutral")
    source_labels = ("angry", "happy", "neutral")
    rows = []
    for label, source_label in zip(labels, source_labels):
        for index in range(examples_per_class):
            scores = {name: 1.0 for name in (
                "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
            )}
            scores[source_label] = 99.0
            rows.append({
                "identity_id": f"{prefix}-{label}-{index % 5}",
                "partition": partition,
                "label": label,
                "scores": scores,
            })
    return rows


class EmotionEvaluationTests(unittest.TestCase):
    def test_rejects_identity_overlap_and_insufficient_examples(self):
        rows = make_rows("validation", 1, "same") + make_rows("held-out", 1, "same")
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_partitions(rows, min_validation_examples=1, min_held_out_examples=1,
                                min_validation_identities=1, min_held_out_identities=1)

        disjoint = make_rows("validation", 1, "validation") + make_rows("held-out", 1, "held")
        with self.assertRaisesRegex(ValueError, "examples"):
            validate_partitions(
                disjoint,
                min_validation_identities=1,
                min_held_out_identities=1,
            )

    def test_selects_thresholds_and_reports_metrics(self):
        validation = make_rows("validation", 20, "validation")
        held_out = make_rows("held-out", 100, "held")
        validate_partitions(
            validation + held_out,
            min_validation_examples=20,
            min_held_out_examples=100,
            min_validation_identities=5,
            min_held_out_identities=10,
        )
        thresholds = select_class_thresholds(validation)
        self.assertEqual(set(thresholds), {"anger", "smiling", "neutral"})
        self.assertTrue(all(value is not None for value in thresholds.values()))
        report = evaluate_partition(held_out, thresholds)
        self.assertEqual(report["uncertain_rate"], 0.0)
        self.assertTrue(all(
            report["classes"][label]["precision"] == 1.0
            for label in ("anger", "smiling", "neutral")
        ))

    def test_report_disables_class_that_fails_held_out_precision(self):
        validation = make_rows("validation", 20, "validation")
        held_out = make_rows("held-out", 100, "held")
        for row in held_out:
            if row["label"] == "anger":
                row["label"] = "neutral"

        report, policy = build_report(
            validation + held_out,
            "approved test fixture",
            "test-model",
        )

        self.assertFalse(policy["classes"]["angry"]["enabled"])
        self.assertIsNotNone(policy["classes"]["angry"]["threshold"])
        self.assertGreaterEqual(
            policy["classes"]["angry"]["validation_precision"], 0.75
        )
        self.assertLess(
            policy["classes"]["angry"]["held_out_precision"], 0.75
        )
        self.assertIn("Held-out confusion matrix", report)
        self.assertNotIn("validation-anger-", report)
        self.assertNotIn("held-anger-", report)
        self.assertNotIn("image_path", report)


if __name__ == "__main__":
    unittest.main()
