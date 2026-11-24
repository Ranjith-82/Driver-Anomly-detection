import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.evaluate_video import (
    _behavior_classification_metrics,
    _behavior_evaluation_status,
    _behavior_video_validation,
    _evaluate_classification,
    _gaze_classification_metrics,
    _gaze_evaluation_status,
    _load_box_annotations,
    _load_gaze_calibration,
    _write_pending_fatigue_report,
)
from evaluation.common.inputs import (
    EvaluationInputError,
    GAZE_LABELS,
    label_at,
    load_ground_truth,
    normalize_gaze_label,
    parse_video_time,
)
from evaluation.common.metrics import classification_metrics, detection_metrics
from evaluation.common.reports import (
    write_confusion_matrix_png,
    write_metrics_csv,
)
from evaluation.common.video import iter_video_frames
from evaluation.cvat_to_yolo import convert_cvat_video_xml


class EvaluationInputTests(unittest.TestCase):
    def test_behavior_evaluation_status_explains_each_row_selection(self):
        self.assertEqual(
            _behavior_evaluation_status({
                "actual_label": "",
                "predicted_label": "NORMAL",
            }),
            "UNLABELLED",
        )
        self.assertEqual(
            _behavior_evaluation_status({
                "actual_label": "NORMAL",
                "predicted_label": "calibrating",
            }),
            "EXCLUDED_CALIBRATING",
        )
        self.assertEqual(
            _behavior_evaluation_status({
                "actual_label": "NORMAL",
                "predicted_label": "FACE NOT DETECTED",
            }),
            "EXCLUDED_FACE_NOT_DETECTED",
        )

    def test_behavior_metrics_exclude_calibration_but_count_it_and_face_misses(self):
        rows = [
            {
                "actual_label": "normal",
                "predicted_label": "CALIBRATING",
                "calibration_status": "CALIBRATING",
                "raw_behavior": "CALIBRATING",
                "face_present": True,
            },
            {
                "actual_label": "NORMAL",
                "predicted_label": "FACE NOT DETECTED",
                "raw_behavior": "FACE NOT DETECTED",
                "face_present": False,
            },
            {
                "actual_label": "YAWNING",
                "predicted_label": "YAWNING",
                "raw_behavior": "YAWNING",
                "face_present": True,
            },
            {
                "actual_label": "DISTRACTED",
                "predicted_label": "NORMAL",
                "raw_behavior": "DISTRACTED",
                "face_present": True,
            },
            {
                "actual_label": "TALKING",
                "predicted_label": "TALKING",
                "raw_behavior": "TALKING",
                "face_present": True,
            },
            {
                "actual_label": "",
                "predicted_label": "TALKING",
                "raw_behavior": "TALKING",
                "face_present": True,
            },
        ]

        result = _behavior_classification_metrics(rows)

        self.assertEqual(result["total_processed_frames"], 6)
        self.assertEqual(result["labelled_frames"], 5)
        self.assertEqual(result["unlabelled_frames_excluded"], 1)
        self.assertEqual(result["calibration_frames_excluded"], 1)
        self.assertEqual(result["excluded_frames"], 3)
        self.assertEqual(result["evaluated_frames"], 3)
        self.assertEqual(result["classes"], [
            "NORMAL", "YAWNING", "DISTRACTED", "TALKING",
        ])
        self.assertEqual(result["confusion_matrix"]["NORMAL"], {
            "NORMAL": 0, "YAWNING": 0, "DISTRACTED": 0, "TALKING": 0,
        })
        self.assertEqual(result["per_class"]["DISTRACTED"]["false_negative"], 1)
        self.assertEqual(result["face_not_detected_excluded_frames"], 1)
        self.assertEqual(result["face_detection_expected_frames"], 5)
        self.assertEqual(result["face_detection_failures"], 1)
        self.assertEqual(result["face_not_detected_output_frames"], 1)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)

    def test_behavior_video_validation_checks_filename_timestamps_and_intervals(self):
        video = Path("driver.mp4")
        rows = [
            {
                "video_name": "driver.mp4",
                "frame_index": 0,
                "timestamp_seconds": 0.0,
                "actual_label": "NORMAL",
            },
            {
                "video_name": "driver.mp4",
                "frame_index": 1,
                "timestamp_seconds": 0.5,
                "actual_label": "TALKING",
            },
        ]
        intervals = {
            "driver.mp4": [
                {"start_time": 0.0, "end_time": 0.5, "actual_label": "NORMAL"},
                {"start_time": 0.5, "end_time": 1.0, "actual_label": "TALKING"},
            ],
        }
        with patch(
            "evaluation.evaluate_video._video_metadata",
            return_value={
                "fps": 2.0,
                "reported_frame_count": 2,
                "duration_seconds": 1.0,
            },
        ):
            result = _behavior_video_validation([video], rows, intervals)

        self.assertTrue(result["all_ground_truth_video_names_match"])
        self.assertTrue(result["all_video_timestamps_monotonic"])
        self.assertTrue(result["all_frame_counts_match_metadata"])
        self.assertTrue(result["all_intervals_within_video_durations"])
        self.assertTrue(result["all_interval_frame_labels_match"])
        self.assertEqual(
            [item["decoded_frames_in_interval"]
             for item in result["videos"][0]["ground_truth_intervals"]],
            [1, 1],
        )

    def test_fatigue_frame_report_separates_eye_and_final_alert_stages(self):
        rows = [
            {
                "actual_label": "normal",
                "predicted_label": "negative",
                "eyes_state": "EYES OPEN",
                "average_ear": 0.3,
                "eyes_closed_duration": 0.0,
                "face_present": True,
            },
            {
                "actual_label": "positive",
                "predicted_label": "negative",
                "eyes_state": "EYES CLOSED",
                "average_ear": 0.1,
                "eyes_closed_duration": 0.5,
                "face_present": True,
            },
            {
                "actual_label": "positive",
                "predicted_label": "positive",
                "eyes_state": "EYES CLOSED",
                "average_ear": 0.1,
                "eyes_closed_duration": 1.5,
                "face_present": True,
            },
            {
                "actual_label": "positive",
                "predicted_label": "positive",
                "eyes_state": "EYE STATE UNCERTAIN",
                "average_ear": None,
                "eyes_closed_duration": 2.0,
                "face_present": True,
            },
        ]

        result = _evaluate_classification(rows, "fatigue")

        self.assertEqual(result["evaluated_frames"], 3)
        self.assertEqual(result["invalid_eye_state_frames_excluded"], 1)
        stages = result["decision_stages"]
        self.assertEqual(
            stages["eye_closure_detection"]["per_class"]["positive"]["recall"],
            1.0,
        )
        self.assertEqual(
            stages["sustained_eye_closure"]["per_class"]["positive"]["recall"],
            0.5,
        )
        self.assertEqual(
            stages["final_fatigue_alert"]["per_class"]["positive"]["recall"],
            0.5,
        )

    def test_pending_fatigue_report_has_no_metrics_or_stale_confusion_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            stale_image = output_dir / "confusion_matrix.png"
            stale_image.write_bytes(b"old confusion matrix")

            report = _write_pending_fatigue_report(
                output_dir,
                Path("fatigue-labels.csv"),
                "The referenced video and event manifest are unavailable.",
            )

            self.assertEqual(report["status"], "pending_data")
            self.assertIsNone(report["metrics"])
            self.assertFalse(stale_image.exists())
            self.assertEqual(
                (output_dir / "predictions.csv").read_text(encoding="utf-8").strip(),
                "video_name,frame_index,timestamp_seconds",
            )

    def test_confusion_report_writers_support_operational_prediction_columns(self):
        matrix = {
            "NORMAL": {"NORMAL": 1, "FACE NOT DETECTED": 1},
            "YAWNING": {"NORMAL": 0, "FACE NOT DETECTED": 0},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "report.csv"
            png_path = root / "confusion.png"
            write_metrics_csv(csv_path, {
                "metrics": {"confusion_matrix": matrix},
            })
            write_confusion_matrix_png(
                png_path,
                matrix,
                ["NORMAL", "YAWNING"],
                column_labels=["NORMAL", "FACE NOT DETECTED"],
            )

            self.assertIn(
                "confusion_matrix.NORMAL.FACE NOT DETECTED,1",
                csv_path.read_text(encoding="utf-8"),
            )
            import cv2

            image = cv2.imread(str(png_path))
            self.assertEqual(tuple(image.shape[:2]), (302, 530))

    def test_yolo_loader_explains_interval_csv_cannot_score_boxes(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "boxes.csv"
            csv_path.write_text(
                "video_name,start_time,end_time,actual_label\n"
                "driver1.mp4,10,28,Phone\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                EvaluationInputError,
                "video-level label intervals, not bounding boxes",
            ):
                _load_box_annotations(csv_path)

    def test_cvat_conversion_requires_coverage_and_preserves_verified_empty_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            xml_path = root / "annotations.xml"
            coverage_path = root / "coverage.csv"
            output_path = root / "boxes.csv"
            xml_path.write_text(
                '<annotations><track id="1" label="Phone">'
                '<box frame="0" xtl="1.5" ytl="2" xbr="9" ybr="12" outside="0"/>'
                '<box frame="1" xtl="1" ytl="1" xbr="2" ybr="2" outside="1"/>'
                '</track></annotations>',
                encoding="utf-8",
            )
            coverage_path.write_text(
                "video_name,frame_index,status\n"
                "drive.avi,0,annotated\n"
                "drive.avi,1,verified_empty\n",
                encoding="utf-8",
            )

            result = convert_cvat_video_xml(xml_path, coverage_path, output_path)

            self.assertEqual(result, {
                "covered_frames": 2,
                "object_boxes": 1,
                "verified_empty_frames": 1,
            })
            self.assertEqual(
                output_path.read_text(encoding="utf-8").splitlines(),
                [
                    "video_name,frame_index,class_name,x_min,y_min,x_max,y_max",
                    "drive.avi,0,Phone,1.5,2.0,9.0,12.0",
                    "drive.avi,1,,,,,",
                ],
            )

    def test_cvat_conversion_rejects_unreviewed_box_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            xml_path = root / "annotations.xml"
            coverage_path = root / "coverage.csv"
            xml_path.write_text(
                '<annotations><track id="1" label="Bottle">'
                '<box frame="3" xtl="1" ytl="1" xbr="2" ybr="2"/>'
                '</track></annotations>',
                encoding="utf-8",
            )
            coverage_path.write_text(
                "video_name,frame_index,status\n"
                "drive.avi,0,verified_empty\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(EvaluationInputError, "uncovered"):
                convert_cvat_video_xml(xml_path, coverage_path, root / "boxes.csv")

    def test_gaze_calibration_uses_module_that_provides_file_loader(self):
        calibration_module = SimpleNamespace(load_calibration=lambda path: {"path": path})
        with patch(
            "driver_monitor.legacy.load_legacy_module",
            return_value=calibration_module,
        ) as load_module:
            result = _load_gaze_calibration(Path("repo"), Path("saved.json"))
        self.assertEqual(result, {"path": Path("saved.json")})
        load_module.assert_called_once_with(
            "gaze_calibration",
            Path("repo") / "Phase-2" / "Gaze_detection" / "gaze_detection.py",
        )

    def test_video_reader_preserves_capture_timestamps_and_releases(self):
        class FakeCapture:
            def __init__(self):
                self.index = 0
                self.timestamps = [0.0, 100.0, 225.0]
                self.frames = [object(), object(), object()]
                self.released = False

            def isOpened(self):
                return True

            def get(self, property_id):
                if property_id == 1:
                    return 10.0
                return self.timestamps[max(0, self.index - 1)]

            def read(self):
                if self.index == len(self.frames):
                    return False, None
                frame = self.frames[self.index]
                self.index += 1
                return True, frame

            def release(self):
                self.released = True

        capture = FakeCapture()
        fake_cv2 = SimpleNamespace(
            VideoCapture=lambda path: capture if path == "drive.mp4" else None,
            CAP_PROP_FPS=1,
            CAP_PROP_POS_MSEC=2,
        )
        with patch.dict("sys.modules", {"cv2": fake_cv2}):
            frames = list(iter_video_frames(Path("drive.mp4")))
        self.assertEqual([frame[1] for frame in frames], [0.0, 0.1, 0.225])
        self.assertTrue(capture.released)

    def test_video_time_and_half_open_interval(self):
        self.assertAlmostEqual(parse_video_time("00:01:02.5"), 62.5)
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "labels.csv"
            csv_path.write_text(
                "video_name,start_time,end_time,actual_label\n"
                "drive.mp4,0,1.5,NORMAL\n",
                encoding="utf-8",
            )
            labels = load_ground_truth(csv_path)
        self.assertEqual(label_at(labels, "drive.mp4", 0), "NORMAL")
        self.assertIsNone(label_at(labels, "drive.mp4", 1.5))

    def test_video_time_rejects_invalid_components(self):
        for value in ("00:60", "01:60:00", "1.5:02", "1:-1", "nan"):
            with self.subTest(value=value), self.assertRaises(EvaluationInputError):
                parse_video_time(value)

    def test_gaze_ground_truth_labels_normalize_to_model_classes(self):
        self.assertEqual(normalize_gaze_label("STRAIGHT"), "FORWARD")
        self.assertEqual(normalize_gaze_label("LOOKING_UP"), "UP")
        self.assertEqual(normalize_gaze_label("unknown"), "UNKNOWN")
        self.assertEqual(
            GAZE_LABELS,
            ("FORWARD", "RIGHT", "LEFT", "UP", "DOWN", "UNKNOWN"),
        )
        with self.assertRaisesRegex(EvaluationInputError, "Unsupported gaze"):
            normalize_gaze_label("SIDEWAYS")

    def test_gaze_metrics_exclude_unknown_calibration_and_invalid_rows(self):
        rows = [
            {
                "video_name": "driver.avi",
                "actual_label": "FORWARD",
                "predicted_label": "UP",
                "raw_gaze_label": "FORWARD",
                "calibration_status": "READY",
                "face_present": True,
                "gaze_y": 0.25,
                "gaze_distraction_confirmed": True,
            },
            {
                "video_name": "driver.avi",
                "actual_label": "UP",
                "predicted_label": "UP",
                "raw_gaze_label": "UP",
                "calibration_status": "CALIBRATING",
                "face_present": True,
            },
            {
                "video_name": "driver.avi",
                "actual_label": "UNKNOWN",
                "predicted_label": "FORWARD",
                "raw_gaze_label": "FORWARD",
                "calibration_status": "READY",
                "face_present": True,
            },
            {
                "video_name": "driver.avi",
                "actual_label": "LEFT",
                "predicted_label": "UNKNOWN",
                "raw_gaze_label": "LEFT",
                "calibration_status": "READY",
                "face_present": False,
            },
            {
                "video_name": "driver.avi",
                "actual_label": "",
                "predicted_label": "RIGHT",
                "raw_gaze_label": "RIGHT",
                "calibration_status": "READY",
                "face_present": True,
            },
        ]
        intervals = {
            "driver.avi": [{
                "start_time": 0.0,
                "end_time": 2.0,
                "actual_label": "UNKNOWN",
            }],
        }
        result = _gaze_classification_metrics(rows, intervals)
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["confusion_matrix"]["FORWARD"]["UP"], 1)
        self.assertEqual(result["calibration_frames_excluded"], 1)
        self.assertEqual(result["unknown_ground_truth_frames_excluded"], 1)
        self.assertEqual(result["unknown_ground_truth_duration_seconds"], 2.0)
        self.assertEqual(result["invalid_prediction_frames_excluded"], 1)
        self.assertEqual(result["unlabelled_frames_excluded"], 1)
        self.assertEqual(result["forward_predicted_up"]["frames"], 1)
        self.assertEqual(
            result["forward_predicted_up"]["mean_gaze_y_when_predicted_up"],
            0.25,
        )
        self.assertEqual(
            result["distraction_decision_metrics"][
                "final_sustained_distraction_alert"
            ]["per_class"]["positive"]["false_positive"],
            1,
        )
        self.assertEqual(
            _gaze_evaluation_status({
                "actual_label": "UNKNOWN",
                "predicted_label": "FORWARD",
                "calibration_status": "READY",
                "face_present": True,
            }),
            "EXCLUDED_UNKNOWN_GROUND_TRUTH",
        )

    def test_empty_classification_metrics_keep_declared_classes(self):
        result = classification_metrics([], [], ["FORWARD", "UP"])
        self.assertEqual(result["samples"], 0)
        self.assertEqual(result["weighted_f1"], 0.0)
        self.assertEqual(result["confusion_matrix"]["FORWARD"]["UP"], 0)

    def test_overlapping_ground_truth_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "labels.csv"
            csv_path.write_text(
                "video_name,start_time,end_time,actual_label\n"
                "drive.mp4,0,2,NORMAL\n"
                "drive.mp4,1,3,DROWSY\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EvaluationInputError, "overlap"):
                load_ground_truth(csv_path)

    def test_classification_macro_f1_and_rates(self):
        result = classification_metrics(
            ["normal", "normal", "drowsy"],
            ["normal", "drowsy", "normal"],
            ["normal", "drowsy"],
        )
        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertEqual(result["confusion_matrix"]["normal"]["drowsy"], 1)
        self.assertAlmostEqual(result["per_class"]["drowsy"]["false_positive_rate"], 0.5)
        self.assertAlmostEqual(result["per_class"]["drowsy"]["false_negative_rate"], 1.0)
        self.assertAlmostEqual(result["macro_f1"], 0.25)

    def test_yolo_metrics_include_map_and_background_matrix(self):
        samples = [{
            "ground_truth": [{"class_id": 0, "box": [0, 0, 10, 10]}],
            "predictions": [
                {"class_id": 0, "box": [0, 0, 10, 10], "confidence": 0.9}
            ],
        }]
        result = detection_metrics(samples, {0: "Phone"})
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["map50"], 1.0)
        self.assertAlmostEqual(result["map50_95"], 1.0)
        self.assertEqual(result["confusion_matrix"]["Phone"]["Phone"], 1)

    def test_yolo_matching_is_one_to_one_and_requires_same_class(self):
        samples = [{
            "ground_truth": [
                {"class_id": 0, "box": [0, 0, 10, 10]},
                {"class_id": 0, "box": [1, 1, 11, 11]},
            ],
            "predictions": [
                {"class_id": 0, "box": [0, 0, 10, 10], "confidence": 0.9},
                {"class_id": 1, "box": [0, 0, 10, 10], "confidence": 0.8},
            ],
        }]
        result = detection_metrics(samples, {0: "Phone", 1: "Cup"})
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["false_negative"], 1)
        self.assertEqual(result["confusion_matrix"]["Phone"]["background"], 1)
        self.assertEqual(result["confusion_matrix"]["background"]["Cup"], 1)


if __name__ == "__main__":
    unittest.main()
